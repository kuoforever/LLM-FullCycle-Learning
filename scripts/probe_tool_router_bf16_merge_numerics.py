"""Locate and quantify frozen Tool Router BF16 safe-merge numerics drift."""

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
from fullcycle_bridge.tool_router_merge_numerics import (  # noqa: E402
    MERGE_NUMERICS_VERSION,
    analyze_module_comparisons,
    classify_merge_numerics,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
CAPTURE_STAGES = (
    "model.embed_tokens",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.0",
    "model.norm",
    "lm_head",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_json(args.config)
    training = _load_json(args.training_evidence)
    stability = _load_json(args.stability_evidence)
    _verify_sources(config, training, stability, args)
    _verify_environment(config)
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    record = evaluation[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt_path = ROOT / config["prompt"]["path"]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": render_user_payload(record)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded_cpu = tokenizer(rendered, return_tensors="pt")
    target_index = stability["token_analysis"]["first_divergent_token_index"]

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    path_results: dict[str, dict[str, Any]] = {}
    for path_name, merge in (("independent", False), ("merged", True)):
        model = _load_model(args.model_dir, args.adapter_dir, config, merge)
        path_results[path_name] = _capture_generation_step(
            model,
            encoded_cpu,
            config,
            tokenizer,
            target_index,
        )
        _release_model(model)

    _verify_generation_reproduction(path_results, stability, target_index)
    comparisons = _compare_captures(
        path_results["independent"]["captures"],
        path_results["merged"]["captures"],
    )
    module_analysis = analyze_module_comparisons(comparisons)

    independent_model = _load_model(
        args.model_dir, args.adapter_dir, config, merge=False
    )
    merged_model = _load_model(args.model_dir, args.adapter_dir, config, merge=True)
    rounding = _quantify_merge_rounding(independent_model, merged_model)
    _release_model(merged_model)
    _release_model(independent_model)
    classification = classify_merge_numerics(module_analysis, rounding)

    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    source_adapter_unchanged = adapter_manifest == training["final_adapter"]["files"]
    eval_digest_unchanged = (
        fixture_digest(load_fixture(ROOT / config["data"]["eval_path"]))
        == config["data"]["eval_digest"]
    )
    expected_first = "model.layers.0.self_attn.q_proj"
    acceptance = {
        "stability_evidence_reproduced": True,
        "earliest_module_divergence_located": (
            module_analysis["first_divergent_module"] == expected_first
        ),
        "preceding_modules_identical": module_analysis[
            "preceding_modules_identical"
        ],
        "safe_merge_weights_match_peft_algorithm": (
            rounding["actual_merged_mismatched_weights"] == 0
        ),
        "merge_rounding_quantified": (
            rounding["ideal_nonzero_updates_rounded_to_base"] > 0
        ),
        "source_adapter_unchanged": source_adapter_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
    }
    torch.cuda.synchronize()
    result = {
        "merge_numerics_version": MERGE_NUMERICS_VERSION,
        "experiment_id": "fc-mvp-001-bf16-merge-numerics-v1",
        "source_experiment_id": config["experiment_id"],
        "stability_evidence_sha256": file_sha256(args.stability_evidence),
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "eval_digest": config["data"]["eval_digest"],
        "example_id": record["example_id"],
        "target_generated_token_index": target_index,
        "capture_source": "exact_generate_step_with_use_cache_true",
        "environment": {
            **config["environment"],
            "torch_dtype": "bfloat16",
            "attn_implementation": config["generation"]["attn_implementation"],
            "do_sample": False,
            "tf32": False,
        },
        "generation_reproduction": {
            name: {
                "token_count": value["token_count"],
                "token_ids_sha256": value["token_ids_sha256"],
                "target_top_token_ids": value["target_top_token_ids"],
                "target_top_scores": value["target_top_scores"],
            }
            for name, value in path_results.items()
        },
        "module_comparisons": comparisons,
        "module_analysis": module_analysis,
        "merge_rounding": rounding,
        "classification": classification,
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": {
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "runtime_integration": False,
            "full_eval_run": False,
        },
        "locked_next_action": {
            "gate_id": "FC-MVP-001-bf16-merge-remediation-v1",
            "action": (
                "load the pinned base and Adapter in FP32, safe-merge in FP32, "
                "retain FP32 for greedy SDPA inference on frozen eval-001, "
                "and require repeat identity against the independent BF16 "
                "Adapter before any full-eval run or artifact promotion"
            ),
            "acceptance": {
                "candidate_repeats_identical": True,
                "independent_output_identity": True,
                "source_adapter_unchanged": True,
                "eval_digest_unchanged": True,
            },
            "constraints": {
                "new_data": False,
                "training": False,
                "eval_answer_tuning": False,
                "runtime_integration": False,
                "full_eval_before_identity": False,
                "merged_artifact_promotion": False,
            },
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": classification,
        "offline": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"merge numerics acceptance failed: {acceptance!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _capture_generation_step(
    model: Any,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
    target_index: int,
) -> dict[str, Any]:
    causal = _causal_model(model)
    modules = dict(causal.named_modules())
    missing = [name for name in CAPTURE_STAGES if name not in modules]
    if missing:
        raise RuntimeError(f"capture modules missing: {missing!r}")
    state = {"call_index": -1}
    captures: dict[str, Any] = {}
    handles: list[Any] = []

    def capture(name: str, output: Any) -> None:
        if state["call_index"] != target_index:
            return
        tensor = _first_tensor(output)
        captures[name] = tensor.detach().float().cpu()

    def embedding_hook(_module: Any, _inputs: Any, output: Any) -> None:
        state["call_index"] += 1
        capture(CAPTURE_STAGES[0], output)

    handles.append(modules[CAPTURE_STAGES[0]].register_forward_hook(embedding_hook))
    for name in CAPTURE_STAGES[1:]:
        handles.append(
            modules[name].register_forward_hook(
                lambda _module, _inputs, output, stage=name: capture(stage, output)
            )
        )
    encoded = {key: value.to("cuda") for key, value in encoded_cpu.items()}
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["generation"]["max_new_tokens"],
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(CAPTURE_STAGES):
        raise RuntimeError(f"incomplete target-step captures: {sorted(captures)!r}")
    new_tokens = generated.sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.cpu()]
    target_scores = generated.scores[target_index][0].float().cpu()
    top = torch.topk(target_scores, k=2)
    return {
        "token_count": len(token_ids),
        "token_ids_sha256": _token_digest(token_ids),
        "target_top_token_ids": [int(value) for value in top.indices],
        "target_top_scores": [float(value) for value in top.values],
        "captures": captures,
    }


def _compare_captures(
    independent: dict[str, Any], merged: dict[str, Any]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for name in CAPTURE_STAGES:
        left = independent[name]
        right = merged[name]
        if left.shape != right.shape:
            raise RuntimeError(f"capture shape mismatch at {name}")
        delta = (left - right).abs()
        different = int(torch.count_nonzero(left != right))
        comparisons.append(
            {
                "name": name,
                "shape": list(left.shape),
                "equal": different == 0,
                "different_elements": different,
                "max_abs_delta": float(delta.max()),
                "mean_abs_delta": float(delta.mean()),
            }
        )
    return comparisons


def _quantify_merge_rounding(
    independent_model: Any, merged_model: Any
) -> dict[str, Any]:
    independent_modules = dict(_causal_model(independent_model).named_modules())
    merged_modules = dict(_causal_model(merged_model).named_modules())
    adapter_modules = {
        name: module
        for name, module in independent_modules.items()
        if name.rpartition(".")[2] in TARGET_MODULES
        and hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
    }
    expected_count = 28 * len(TARGET_MODULES)
    if len(adapter_modules) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} Adapter targets, got {len(adapter_modules)}"
        )
    totals: dict[str, Any] = {
        "target_modules": len(adapter_modules),
        "total_target_weights": 0,
        "ideal_nonzero_updates": 0,
        "effective_changed_weights": 0,
        "ideal_nonzero_updates_rounded_to_base": 0,
        "actual_merged_mismatched_weights": 0,
        "absolute_rounding_error_sum": 0.0,
        "absolute_ideal_update_sum": 0.0,
        "max_abs_rounding_error": 0.0,
    }
    first_target: dict[str, Any] | None = None
    with torch.inference_mode():
        for name in sorted(adapter_modules):
            module = adapter_modules[name]
            merged_module = merged_modules.get(name)
            if merged_module is None or not hasattr(merged_module, "weight"):
                raise RuntimeError(f"merged target missing: {name}")
            adapter_name = next(iter(module.lora_A))
            base = module.base_layer.weight.detach()
            a_weight = module.lora_A[adapter_name].weight.detach().float()
            b_weight = module.lora_B[adapter_name].weight.detach().float()
            ideal_delta = (b_weight @ a_weight) * module.scaling[adapter_name]
            if getattr(module, "fan_in_fan_out", False):
                ideal_delta = ideal_delta.transpose(0, 1)
            expected = base.clone()
            expected += ideal_delta
            actual = merged_module.weight.detach()
            effective_delta = expected.float() - base.float()
            rounding_error = effective_delta - ideal_delta
            ideal_nonzero = ideal_delta != 0
            changed = expected != base
            mismatched = actual != expected
            module_result = {
                "name": name,
                "shape": list(base.shape),
                "ideal_nonzero_updates": int(torch.count_nonzero(ideal_nonzero)),
                "effective_changed_weights": int(torch.count_nonzero(changed)),
                "ideal_nonzero_updates_rounded_to_base": int(
                    torch.count_nonzero(ideal_nonzero & ~changed)
                ),
                "actual_merged_mismatched_weights": int(
                    torch.count_nonzero(mismatched)
                ),
                "max_abs_rounding_error": float(rounding_error.abs().max()),
                "mean_abs_rounding_error": float(rounding_error.abs().mean()),
            }
            if name == "model.layers.0.self_attn.q_proj":
                first_target = module_result
            totals["total_target_weights"] += base.numel()
            totals["ideal_nonzero_updates"] += module_result[
                "ideal_nonzero_updates"
            ]
            totals["effective_changed_weights"] += module_result[
                "effective_changed_weights"
            ]
            totals["ideal_nonzero_updates_rounded_to_base"] += module_result[
                "ideal_nonzero_updates_rounded_to_base"
            ]
            totals["actual_merged_mismatched_weights"] += module_result[
                "actual_merged_mismatched_weights"
            ]
            totals["absolute_rounding_error_sum"] += float(
                rounding_error.abs().double().sum()
            )
            totals["absolute_ideal_update_sum"] += float(
                ideal_delta.abs().double().sum()
            )
            totals["max_abs_rounding_error"] = max(
                totals["max_abs_rounding_error"],
                module_result["max_abs_rounding_error"],
            )
    if first_target is None:
        raise RuntimeError("first divergent Adapter target was not measured")
    total_weights = totals["total_target_weights"]
    totals["mean_abs_rounding_error"] = (
        totals["absolute_rounding_error_sum"] / total_weights
    )
    totals["rounded_to_base_fraction_of_nonzero_updates"] = (
        totals["ideal_nonzero_updates_rounded_to_base"]
        / totals["ideal_nonzero_updates"]
    )
    totals["first_divergent_target"] = first_target
    return totals


def _verify_generation_reproduction(
    results: dict[str, dict[str, Any]],
    stability: dict[str, Any],
    target_index: int,
) -> None:
    expected_runs = {run["path"]: run for run in stability["runs"]}
    logits = stability["logit_evidence"]
    for name, result in results.items():
        expected = expected_runs[name]
        if (
            result["token_ids_sha256"] != expected["token_ids_sha256"]
            or result["token_count"] != expected["token_count"]
            or result["target_top_token_ids"]
            != logits[f"{name}_top_token_ids"]
        ):
            raise RuntimeError(f"stability evidence did not reproduce: {name}")
    if target_index != logits["common_prefix_generated_tokens"]:
        raise RuntimeError("stability target index mismatch")


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    config_digest = canonical_config_sha256(config)
    if training["config_sha256"] != config_digest:
        raise RuntimeError("training evidence config mismatch")
    if stability["config_sha256"] != config_digest:
        raise RuntimeError("stability evidence config mismatch")
    if stability["classification"] != "deterministic_bf16_merge_logit_boundary_flip":
        raise RuntimeError("required stability classification is missing")
    if not all(stability["acceptance"].values()):
        raise RuntimeError("stability evidence acceptance is incomplete")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if adapter_manifest != training["final_adapter"]["files"]:
        raise RuntimeError("adapter artifact mismatch")
    weight_path = args.model_dir / config["model"]["weight_file"]
    if weight_path.stat().st_size != config["model"]["weight_bytes"]:
        raise RuntimeError("model weight size mismatch")
    if file_sha256(weight_path) != f"sha256:{config['model']['weight_sha256']}":
        raise RuntimeError("model weight digest mismatch")
    prompt_path = ROOT / config["prompt"]["path"]
    if file_sha256(prompt_path) != f"sha256:{config['prompt']['sha256']}":
        raise RuntimeError("prompt digest mismatch")
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("eval digest mismatch")
    if evaluation[0]["example_id"] != "eval-001":
        raise RuntimeError("locked numerics probe requires eval-001 first")


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


def _causal_model(model: Any) -> Any:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def _first_tensor(output: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"module output does not begin with a tensor: {type(output)!r}")


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _token_digest(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
