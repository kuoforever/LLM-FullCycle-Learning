"""Probe deterministic BF16 LoRA load/merge divergence on frozen eval-001."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import accelerate  # type: ignore[import-not-found]
import peft  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import transformers  # type: ignore[import-not-found]
from huggingface_hub import (  # type: ignore[import-not-found]
    __version__ as hub_version,
)
from peft import PeftModel  # type: ignore[import-not-found]
from safetensors import __version__ as safetensors_version  # type: ignore[import-not-found]
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_merge_stability import (  # noqa: E402
    MERGE_STABILITY_VERSION,
    analyze_token_runs,
    classify_logit_divergence,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats != 2:
        parser.error("--repeats must be exactly 2 for the locked v1 probe")

    config = _load_json(args.config)
    evidence = _load_json(args.training_evidence)
    _verify_environment(config)
    config_digest = canonical_config_sha256(config)
    if evidence["config_sha256"] != config_digest:
        raise RuntimeError("training evidence config mismatch")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if adapter_manifest != evidence["final_adapter"]["files"]:
        raise RuntimeError("adapter artifact mismatch")
    model_weight_path = args.model_dir / config["model"]["weight_file"]
    if model_weight_path.stat().st_size != config["model"]["weight_bytes"]:
        raise RuntimeError("model weight size mismatch")
    model_weight_sha256 = file_sha256(model_weight_path)
    if model_weight_sha256 != f"sha256:{config['model']['weight_sha256']}":
        raise RuntimeError("model weight digest mismatch")
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("evaluation digest mismatch")
    record = evaluation[0]
    if record["example_id"] != "eval-001":
        raise RuntimeError("locked probe requires eval-001 first")
    prompt_path = ROOT / config["prompt"]["path"]
    if file_sha256(prompt_path).removeprefix("sha256:") != config["prompt"]["sha256"]:
        raise RuntimeError("prompt digest mismatch")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": render_user_payload(record)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded_cpu = tokenizer(rendered, return_tensors="pt")
    input_token_ids = [int(value) for value in encoded_cpu["input_ids"][0]]
    independent_runs: list[list[int]] = []
    merged_runs: list[list[int]] = []
    score_traces: dict[str, list[Any]] = {}
    run_evidence: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for path_name, destination, merge in (
        ("independent", independent_runs, False),
        ("merged", merged_runs, True),
    ):
        for repeat in range(1, args.repeats + 1):
            model = _load_model(args.model_dir, args.adapter_dir, config, merge)
            token_ids, scores = _generate_tokens(
                model, encoded_cpu, config, tokenizer
            )
            destination.append(token_ids)
            if repeat == 1:
                score_traces[path_name] = scores
            run_evidence.append(
                {
                    "path": path_name,
                    "repeat": repeat,
                    "token_count": len(token_ids),
                    "token_ids_sha256": _token_digest(token_ids),
                    "output_sha256": "sha256:"
                    + hashlib.sha256(
                        tokenizer.decode(token_ids, skip_special_tokens=True)
                        .strip()
                        .encode("utf-8")
                    ).hexdigest(),
                }
            )
            _release_model(model)

    token_analysis = analyze_token_runs(independent_runs, merged_runs)
    logit_evidence: dict[str, Any] | None = None
    final_classification = token_analysis["classification"]
    divergence_index = token_analysis["first_divergent_token_index"]
    if divergence_index is not None:
        independent_logits = score_traces["independent"][divergence_index]
        merged_logits = score_traces["merged"][divergence_index]
        independent_top = torch.topk(independent_logits, k=2)
        merged_top = torch.topk(merged_logits, k=2)
        independent_top_ids = [int(value) for value in independent_top.indices]
        merged_top_ids = [int(value) for value in merged_top.indices]
        final_classification = classify_logit_divergence(
            token_analysis,
            independent_top_ids[0],
            merged_top_ids[0],
        )
        delta = (independent_logits - merged_logits).abs()
        logit_evidence = {
            "common_prefix_generated_tokens": divergence_index,
            "score_source": "exact_generate_step_with_use_cache_true",
            "independent_top_token_ids": independent_top_ids,
            "independent_top_scores": [float(value) for value in independent_top.values],
            "merged_top_token_ids": merged_top_ids,
            "merged_top_scores": [float(value) for value in merged_top.values],
            "independent_divergent_token_text": _decode_token(
                tokenizer, token_analysis["independent_token_id"]
            ),
            "merged_divergent_token_text": _decode_token(
                tokenizer, token_analysis["merged_token_id"]
            ),
            "independent_margin": float(
                independent_top.values[0] - independent_top.values[1]
            ),
            "merged_margin": float(merged_top.values[0] - merged_top.values[1]),
            "max_abs_logit_delta": float(delta.max()),
            "mean_abs_logit_delta": float(delta.mean()),
        }

    logit_argmax_matches_generated_token = (
        divergence_index is None
        or final_classification == "deterministic_bf16_merge_logit_boundary_flip"
    )

    source_adapter_unchanged = (
        directory_artifact_manifest(args.adapter_dir) == adapter_manifest
    )
    eval_digest_unchanged = (
        fixture_digest(load_fixture(ROOT / config["data"]["eval_path"]))
        == config["data"]["eval_digest"]
    )
    acceptance = {
        "independent_repeats_identical": token_analysis[
            "independent_repeats_identical"
        ],
        "merged_repeats_identical": token_analysis["merged_repeats_identical"],
        "divergence_classified": final_classification
        in {
            "output_identity_restored",
            "deterministic_bf16_merge_logit_boundary_flip",
            "deterministic_bf16_merge_sequence_drift",
        },
        "logit_argmax_matches_generated_token": (
            logit_argmax_matches_generated_token
        ),
        "source_adapter_unchanged": source_adapter_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
    }
    result = {
        "merge_stability_version": MERGE_STABILITY_VERSION,
        "experiment_id": "fc-mvp-001-bf16-merge-stability-v1",
        "source_experiment_id": config["experiment_id"],
        "config_sha256": config_digest,
        "adapter_files": adapter_manifest,
        "model_weight_sha256": model_weight_sha256,
        "prompt_sha256": "sha256:" + config["prompt"]["sha256"],
        "eval_digest": config["data"]["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": _token_digest(input_token_ids),
        "repeats_per_path": args.repeats,
        "environment": {
            **config["environment"],
            "torch_dtype": "bfloat16",
            "attn_implementation": config["generation"]["attn_implementation"],
            "do_sample": False,
            "tf32": False,
        },
        "runs": run_evidence,
        "token_analysis": token_analysis,
        "logit_evidence": logit_evidence,
        "classification": final_classification,
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": token_analysis["cross_path_identical"],
        "constraints": {
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "runtime_integration": False,
            "full_eval_run": False,
        },
        "locked_next_action": {
            "gate_id": "FC-MVP-001-bf16-merge-numerics-v1",
            "action": (
                "on the frozen eval-001 common prefix, locate the earliest "
                "independent-versus-merged BF16 module divergence and quantify "
                "adapter-update versus safe-merged-weight rounding"
            ),
            "acceptance": {
                "earliest_module_divergence_located": True,
                "merge_rounding_quantified": True,
                "source_adapter_unchanged": True,
                "eval_digest_unchanged": True,
            },
            "constraints": {
                "new_data": False,
                "training": False,
                "eval_answer_tuning": False,
                "runtime_integration": False,
                "merged_artifact_promotion": False,
            },
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": (
            "deterministic_bf16_merge_logit_boundary_flip"
        ),
        "offline": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"merge stability acceptance failed: {acceptance!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _load_model(
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    merge: bool,
) -> Any:
    seed = config["generation"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if not getattr(model_config, "use_sliding_window", False):
        model_config.sliding_window = None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=model_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["generation"]["attn_implementation"],
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
    ).to("cuda")
    if merge:
        model = model.merge_and_unload(safe_merge=True)
        if any(".lora_" in name for name, _ in model.named_parameters()):
            raise RuntimeError("merged model still contains LoRA parameters")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


def _generate_tokens(
    model: Any,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[int], list[Any]]:
    encoded = {key: value.to("cuda") for key, value in encoded_cpu.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config["generation"]["max_new_tokens"],
            use_cache=config["generation"]["use_cache"],
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    new_tokens = generated.sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.cpu()]
    scores = [score[0].float().cpu() for score in generated.scores]
    if len(scores) != len(token_ids):
        raise RuntimeError("generated score count does not match token count")
    return token_ids, scores


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _token_digest(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _decode_token(tokenizer: Any, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _verify_environment(config: dict[str, Any]) -> None:
    expected = config["environment"]
    actual = {
        "python": ".".join(str(item) for item in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": hub_version,
        "safetensors": safetensors_version,
        "tokenizers": tokenizers_version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 0
        ),
        "compute_capability": (
            ".".join(str(item) for item in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else ""
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"environment mismatch: actual={actual}, expected={expected}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
