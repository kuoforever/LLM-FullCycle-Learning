"""Test the pre-registered FP32 remediation for frozen BF16 merge drift."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import accelerate  # type: ignore[import-not-found]
import peft  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import transformers  # type: ignore[import-not-found]
from huggingface_hub import (  # type: ignore[import-not-found]
    __version__ as hub_version,
)
from peft import PeftModel  # type: ignore[import-not-found]
from peft.tuners.tuners_utils import (  # type: ignore[import-not-found]
    BaseTunerLayer,
)
from safetensors import safe_open  # type: ignore[import-not-found]
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
from fullcycle_bridge.tool_router_merge_remediation import (  # noqa: E402
    MERGE_REMEDIATION_VERSION,
    analyze_candidate_runs,
    token_ids_sha256,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)

EXPECTED_LORA_TARGETS = 112
EXPECTED_LORA_PARAMETER_TENSORS = EXPECTED_LORA_TARGETS * 2
MAX_RESIDUAL_CUDA_BYTES = 16 * 1024 * 1024
ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "fp32_safe_merge_output_identity_restored",
        "fp32_candidate_within_path_nondeterminism",
        "deterministic_fp32_merge_output_drift",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--numerics-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats != 2:
        parser.error("--repeats must be exactly 2 for the locked v1 probe")
    _verify_output_boundary(args.output, args.model_dir, args.adapter_dir)

    config = _load_json(args.config)
    training = _load_json(args.training_evidence)
    stability = _load_json(args.stability_evidence)
    numerics = _load_json(args.numerics_evidence)
    _verify_sources(config, training, stability, numerics, args)
    _verify_environment(config)

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    record = evaluation[0]
    prompt_path = ROOT / config["prompt"]["path"]
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
    input_digest = token_ids_sha256(input_token_ids)
    if (
        len(input_token_ids) != stability["input_token_count"]
        or input_digest != stability["input_token_ids_sha256"]
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")
    reference = _frozen_reference(
        stability,
        source_path="independent",
        evidence_path="independent_bf16_adapter",
    )
    bf16_merged_control = _frozen_reference(
        stability,
        source_path="merged",
        evidence_path="safe_merged_bf16",
    )

    started = time.perf_counter()
    candidate_tokens: list[list[int]] = []
    runs: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        token_ids, evidence = _run_candidate_once(
            args.model_dir,
            args.adapter_dir,
            config,
            encoded_cpu,
            tokenizer,
        )
        candidate_tokens.append(token_ids)
        runs.append({"path": "fp32_safe_merged", "repeat": repeat, **evidence})

    analysis = analyze_candidate_runs(
        candidate_tokens,
        reference_token_count=reference["token_count"],
        reference_token_ids_sha256=reference["token_ids_sha256"],
    )
    candidate_output_identity = all(
        run["output_sha256"] == reference["output_sha256"] for run in runs
    )
    bf16_merged_token_identity = all(
        run["token_count"] == bf16_merged_control["token_count"]
        and run["token_ids_sha256"] == bf16_merged_control["token_ids_sha256"]
        for run in runs
    )
    bf16_merged_output_identity = all(
        run["output_sha256"] == bf16_merged_control["output_sha256"]
        for run in runs
    )
    remediation_passed = (
        analysis["independent_bf16_reference_identity"]
        and candidate_output_identity
    )
    classification = analysis["classification"]
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError(f"unexpected remediation classification: {classification}")

    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    model_weight_path = args.model_dir / config["model"]["weight_file"]
    storage_audit = {
        "base_checkpoint": _safetensor_storage_audit(model_weight_path),
        "adapter": _safetensor_storage_audit(
            args.adapter_dir / "adapter_model.safetensors"
        ),
    }
    source_storage_dtypes_locked = storage_audit == {
        "base_checkpoint": {
            "tensors": 338,
            "elements": 1543714304,
            "dtype_tensors": {"bfloat16": 338},
            "dtype_elements": {"bfloat16": 1543714304},
        },
        "adapter": {
            "tensors": 224,
            "elements": 4358144,
            "dtype_tensors": {"float32": 224},
            "dtype_elements": {"float32": 4358144},
        },
    }
    source_adapter_unchanged = adapter_manifest == training["final_adapter"]["files"]
    source_model_unchanged = (
        file_sha256(model_weight_path)
        == f"sha256:{config['model']['weight_sha256']}"
    )
    eval_digest_unchanged = (
        fixture_digest(load_fixture(ROOT / config["data"]["eval_path"]))
        == config["data"]["eval_digest"]
    )
    prompt_digest_unchanged = (
        file_sha256(prompt_path) == f"sha256:{config['prompt']['sha256']}"
    )
    candidate_protocol_executed = all(
        _precision_protocol_passed(run["precision_audit"]) for run in runs
    )
    acceptance = {
        "upstream_evidence_locked": True,
        "frozen_input_reproduced": True,
        "candidate_runs_completed": len(runs) == 2,
        "candidate_result_classified": classification in ALLOWED_CLASSIFICATIONS,
        "candidate_protocol_executed": candidate_protocol_executed,
        "source_storage_dtypes_locked": source_storage_dtypes_locked,
        "frozen_bf16_merged_control_compared": True,
        "fresh_load_memory_isolated": all(
            run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
            and run["memory_allocated_after_release_bytes"]
            <= MAX_RESIDUAL_CUDA_BYTES
            for run in runs
        ),
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": source_model_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": prompt_digest_unchanged,
    }
    next_action = _locked_next_action(remediation_passed)
    result = {
        "merge_remediation_version": MERGE_REMEDIATION_VERSION,
        "experiment_id": "fc-mvp-001-bf16-merge-remediation-v1",
        "source_experiment_id": config["experiment_id"],
        "stability_evidence_sha256": file_sha256(args.stability_evidence),
        "numerics_evidence_sha256": file_sha256(args.numerics_evidence),
        "training_lock_sha256": file_sha256(
            ROOT / "requirements" / "training.lock"
        ),
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "model_weight_sha256": file_sha256(model_weight_path),
        "prompt_sha256": file_sha256(prompt_path),
        "eval_digest": config["data"]["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "reference": reference,
        "frozen_bf16_merged_control": bf16_merged_control,
        "candidate_protocol": {
            "checkpoint_storage_dtype": "bfloat16",
            "base_load_dtype": "float32",
            "adapter_storage_dtype": "float32",
            "adapter_load_dtype": "float32",
            "merge_dtype": "float32",
            "inference_dtype": "float32",
            "safe_merge": True,
            "adapter_names": ["default"],
            "attn_implementation": config["generation"]["attn_implementation"],
            "attention_class": "Qwen2Attention",
            "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
            "output_attentions": False,
            "do_sample": False,
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "use_cache": config["generation"]["use_cache"],
            "repetition_penalty": 1.1,
            "model_eos_token_ids": [151645, 151643],
            "model_pad_token_id": 151643,
            "call_pad_token_id": tokenizer.eos_token_id,
            "tf32": False,
            "autocast": False,
            "device": "cuda:0",
            "fresh_loads": args.repeats,
            "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        },
        "runs": runs,
        "analysis": analysis,
        "classification": classification,
        "remediation_gate": {
            "candidate_repeats_identical": analysis[
                "candidate_repeats_identical"
            ],
            "independent_bf16_reference_token_identity": analysis[
                "independent_bf16_reference_identity"
            ],
            "independent_bf16_reference_output_identity": (
                candidate_output_identity
            ),
            "frozen_bf16_merged_token_identity": bf16_merged_token_identity,
            "frozen_bf16_merged_output_identity": bf16_merged_output_identity,
            "passed": remediation_passed,
        },
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(run["peak_gpu_memory_bytes"] for run in runs),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": {
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "runtime_integration": False,
            "full_eval_run": False,
            "merged_artifact_promotion": False,
        },
        "locked_next_action": next_action,
        "runtime_eligible": False,
        "runtime_eligibility_reason": (
            "full_eval_and_artifact_promotion_not_evaluated"
            if remediation_passed
            else classification
        ),
        "offline": True,
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"merge remediation evidence invalid: {acceptance!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _run_candidate_once(
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    encoded_cpu: Any,
    tokenizer: Any,
) -> tuple[list[int], dict[str, Any]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            "fresh candidate load exceeded the residual CUDA ceiling: "
            f"{allocated_before} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    token_ids: list[int] | None = None
    evidence: dict[str, Any] | None = None
    try:
        model, precision = _load_fp32_merged_model(model_dir, adapter_dir, config)
        token_ids, score_dtypes = _generate_tokens(
            model, encoded_cpu, config, tokenizer
        )
        precision["generation"] = {
            "score_dtypes": score_dtypes,
            "all_scores_float32": score_dtypes == ["float32"],
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        if not _precision_protocol_passed(precision):
            raise RuntimeError(f"FP32 candidate protocol failed: {precision!r}")
        torch.cuda.synchronize()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        evidence = {
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": "sha256:"
            + hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "precision_audit": precision,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            "fresh candidate load retained model-scale CUDA memory: "
            f"{allocated_after} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    if token_ids is None or evidence is None:
        raise RuntimeError("candidate run did not produce evidence")
    evidence["memory_allocated_before_load_bytes"] = allocated_before
    evidence["memory_allocated_after_release_bytes"] = allocated_after
    return token_ids, evidence


def _load_fp32_merged_model(
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
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
        torch_dtype=torch.float32,
        attn_implementation=config["generation"]["attn_implementation"],
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
        autocast_adapter_dtype=False,
    ).to("cuda")
    pre_merge = _pre_merge_precision(model)
    if not _pre_merge_protocol_passed(pre_merge):
        raise RuntimeError(f"pre-merge FP32 protocol failed: {pre_merge!r}")
    model = model.merge_and_unload(
        safe_merge=True,
        adapter_names=["default"],
    )
    post_merge = _post_merge_precision(model)
    if not _post_merge_protocol_passed(post_merge):
        raise RuntimeError(f"post-merge FP32 protocol failed: {post_merge!r}")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    _verify_generation_semantics(model, config)
    return model, {"pre_merge": pre_merge, "post_merge": post_merge}


def _generate_tokens(
    model: Any,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[int], list[str]]:
    if torch.is_autocast_enabled():
        raise RuntimeError("candidate generation must not run under autocast")
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
    if len(generated.scores) != len(token_ids):
        raise RuntimeError("generated score count does not match token count")
    score_dtypes = sorted(
        {str(score.dtype).removeprefix("torch.") for score in generated.scores}
    )
    return token_ids, score_dtypes


def _pre_merge_precision(model: Any) -> dict[str, Any]:
    base_parameters: list[tuple[str, Any]] = []
    adapter_parameters: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        destination = adapter_parameters if ".lora_" in name else base_parameters
        destination.append((name, parameter))
    adapter_finite = all(
        bool(torch.isfinite(parameter).all())
        for _, parameter in adapter_parameters
        if parameter.is_floating_point()
    )
    causal = _causal_model(model)
    return {
        "base_parameters": _tensor_inventory(base_parameters),
        "adapter_parameters": _tensor_inventory(adapter_parameters),
        "floating_buffers": _tensor_inventory(model.named_buffers()),
        "lora_target_modules": sum(
            isinstance(module, BaseTunerLayer) for module in model.modules()
        ),
        "lora_parameter_tensors": len(adapter_parameters),
        "adapter_parameters_finite": adapter_finite,
        "active_adapters": list(model.active_adapters),
        "is_peft_model": isinstance(model, PeftModel),
        "input_output_embeddings_tied": _embeddings_are_tied(causal),
        "attn_implementation": causal.config._attn_implementation,
        "attention_class": causal.model.layers[0].self_attn.__class__.__name__,
        "output_attentions": causal.config.output_attentions,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }


def _post_merge_precision(model: Any) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    causal = _causal_model(model)
    return {
        "parameters": _tensor_inventory(parameters),
        "floating_buffers": _tensor_inventory(model.named_buffers()),
        "lora_target_modules": sum(
            isinstance(module, BaseTunerLayer) for module in model.modules()
        ),
        "lora_parameter_tensors": sum(".lora_" in name for name, _ in parameters),
        "is_peft_model": isinstance(model, PeftModel),
        "input_output_embeddings_tied": _embeddings_are_tied(causal),
        "attn_implementation": causal.config._attn_implementation,
        "attention_class": causal.model.layers[0].self_attn.__class__.__name__,
        "output_attentions": causal.config.output_attentions,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }


def _tensor_inventory(tensors: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "floating_tensors": 0,
        "floating_elements": 0,
        "dtypes": {},
        "devices": {},
    }
    for _name, tensor in tensors:
        if not tensor.is_floating_point():
            continue
        elements = tensor.numel()
        dtype = str(tensor.dtype).removeprefix("torch.")
        device = str(tensor.device)
        result["floating_tensors"] += 1
        result["floating_elements"] += elements
        result["dtypes"][dtype] = result["dtypes"].get(dtype, 0) + elements
        result["devices"][device] = result["devices"].get(device, 0) + elements
    result["dtypes"] = dict(sorted(result["dtypes"].items()))
    result["devices"] = dict(sorted(result["devices"].items()))
    return result


def _inventory_is_fp32_cuda(inventory: dict[str, Any]) -> bool:
    elements = inventory["floating_elements"]
    return (
        inventory["floating_tensors"] > 0
        and inventory["dtypes"] == {"float32": elements}
        and inventory["devices"] == {"cuda:0": elements}
    )


def _pre_merge_protocol_passed(value: dict[str, Any]) -> bool:
    return (
        _inventory_is_fp32_cuda(value["base_parameters"])
        and _inventory_is_fp32_cuda(value["adapter_parameters"])
        and _inventory_is_fp32_cuda(value["floating_buffers"])
        and value["lora_target_modules"] == EXPECTED_LORA_TARGETS
        and value["lora_parameter_tensors"] == EXPECTED_LORA_PARAMETER_TENSORS
        and value["adapter_parameters_finite"] is True
        and value["active_adapters"] == ["default"]
        and value["is_peft_model"] is True
        and value["input_output_embeddings_tied"] is True
        and value["attn_implementation"] == "sdpa"
        and value["attention_class"] == "Qwen2Attention"
        and value["output_attentions"] is False
        and value["hf_device_map"] is None
    )


def _post_merge_protocol_passed(value: dict[str, Any]) -> bool:
    return (
        _inventory_is_fp32_cuda(value["parameters"])
        and _inventory_is_fp32_cuda(value["floating_buffers"])
        and value["lora_target_modules"] == 0
        and value["lora_parameter_tensors"] == 0
        and value["is_peft_model"] is False
        and value["input_output_embeddings_tied"] is True
        and value["attn_implementation"] == "sdpa"
        and value["attention_class"] == "Qwen2Attention"
        and value["output_attentions"] is False
        and value["hf_device_map"] is None
    )


def _precision_protocol_passed(value: dict[str, Any]) -> bool:
    generation = value.get("generation")
    return (
        _pre_merge_protocol_passed(value["pre_merge"])
        and _post_merge_protocol_passed(value["post_merge"])
        and isinstance(generation, dict)
        and generation.get("score_dtypes") == ["float32"]
        and generation.get("all_scores_float32") is True
        and generation.get("autocast_enabled") is False
        and generation.get("training") is False
    )


def _verify_generation_semantics(model: Any, config: dict[str, Any]) -> None:
    generation = model.generation_config
    if (
        model.training
        or config["generation"]["attn_implementation"] != "sdpa"
        or config["generation"]["do_sample"] is not False
        or config["generation"]["use_cache"] is not True
        or config["generation"]["max_new_tokens"] != 256
        or generation.repetition_penalty != 1.1
        or generation.eos_token_id != [151645, 151643]
        or generation.pad_token_id != 151643
        or generation.temperature is not None
        or generation.top_p is not None
        or generation.top_k is not None
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("frozen greedy SDPA generation semantics drift")


def _frozen_reference(
    stability: dict[str, Any],
    *,
    source_path: str,
    evidence_path: str,
) -> dict[str, Any]:
    runs = [
        run for run in stability["runs"] if run.get("path") == source_path
    ]
    if len(runs) != 2 or [run.get("repeat") for run in runs] != [1, 2]:
        raise RuntimeError(f"frozen {source_path} BF16 reference runs are invalid")
    locked = {
        (
            run.get("token_count"),
            run.get("token_ids_sha256"),
            run.get("output_sha256"),
        )
        for run in runs
    }
    if len(locked) != 1:
        raise RuntimeError(f"frozen {source_path} BF16 reference is not repeat-stable")
    token_count, token_digest, output_digest = locked.pop()
    if not isinstance(token_count, int) or not isinstance(token_digest, str):
        raise RuntimeError(f"frozen {source_path} BF16 token evidence is invalid")
    if not isinstance(output_digest, str):
        raise RuntimeError(f"frozen {source_path} BF16 output evidence is invalid")
    return {
        "path": evidence_path,
        "source_experiment_id": stability["experiment_id"],
        "fresh_runs": 2,
        "token_count": token_count,
        "token_ids_sha256": token_digest,
        "output_sha256": output_digest,
    }


def _locked_next_action(remediation_passed: bool) -> dict[str, Any]:
    constraints = {
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "merged_artifact_promotion": False,
    }
    if remediation_passed:
        return {
            "gate_id": "FC-MVP-001-fp32-merge-full-eval-v1",
            "action": (
                "run the unchanged frozen 20-case eval through fresh FP32 "
                "safe-merged loading, compare raw tokens and compiled safety/quality "
                "metrics with the frozen independent BF16 Adapter, and keep artifact "
                "promotion prohibited until that separate gate passes"
            ),
            "acceptance": {
                "full_eval_source_inputs_unchanged": True,
                "candidate_full_eval_reproducible": True,
                "independent_adapter_token_parity_measured": True,
                "compiled_safety_and_quality_parity_measured": True,
            },
            "constraints": constraints,
        }
    return {
        "gate_id": "FC-MVP-001-fp32-merge-drift-analysis-v1",
        "action": (
            "reproduce the stable FP32 candidate and a fresh independent BF16 "
            "reference on eval-001, then locate the first token and logit divergence "
            "without changing the pre-registered candidate"
        ),
        "acceptance": {
            "candidate_failure_reproduced": True,
            "first_divergent_token_located": True,
            "logit_boundary_quantified": True,
            "source_inputs_unchanged": True,
        },
        "constraints": constraints,
    }


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    numerics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    config_digest = canonical_config_sha256(config)
    if any(
        evidence.get("config_sha256") != config_digest
        for evidence in (training, stability, numerics)
    ):
        raise RuntimeError("upstream config evidence mismatch")
    if (
        stability.get("classification")
        != "deterministic_bf16_merge_logit_boundary_flip"
        or not all(stability.get("acceptance", {}).values())
        or numerics.get("classification") != "bf16_safe_merge_weight_rounding"
        or not all(numerics.get("acceptance", {}).values())
        or numerics.get("stability_evidence_sha256")
        != file_sha256(args.stability_evidence)
        or numerics.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-bf16-merge-remediation-v1"
    ):
        raise RuntimeError("required merge-numerics source chain is invalid")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if (
        training["final_adapter"].get("files") != adapter_manifest
        or stability.get("adapter_files") != adapter_manifest
        or numerics.get("adapter_files") != adapter_manifest
    ):
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
    if (
        evaluation[0]["example_id"] != "eval-001"
        or stability.get("example_id") != "eval-001"
        or numerics.get("example_id") != "eval-001"
    ):
        raise RuntimeError("locked remediation probe requires eval-001")


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


def _verify_output_boundary(output: Path, model_dir: Path, adapter_dir: Path) -> None:
    resolved = output.resolve()
    for prohibited in (model_dir.resolve(), adapter_dir.resolve()):
        if resolved == prohibited or resolved.is_relative_to(prohibited):
            raise RuntimeError("probe output must not modify model or Adapter artifacts")


def _safetensor_storage_audit(path: Path) -> dict[str, Any]:
    dtype_names = {"BF16": "bfloat16", "F32": "float32"}
    dtype_tensors: dict[str, int] = {}
    dtype_elements: dict[str, int] = {}
    tensor_count = 0
    element_count = 0
    with safe_open(path, framework="pt", device="cpu") as source:
        for key in source.keys():
            metadata = source.get_slice(key)
            raw_dtype = metadata.get_dtype()
            dtype = dtype_names.get(raw_dtype, raw_dtype.lower())
            elements = 1
            for dimension in metadata.get_shape():
                elements *= dimension
            tensor_count += 1
            element_count += elements
            dtype_tensors[dtype] = dtype_tensors.get(dtype, 0) + 1
            dtype_elements[dtype] = dtype_elements.get(dtype, 0) + elements
    return {
        "tensors": tensor_count,
        "elements": element_count,
        "dtype_tensors": dict(sorted(dtype_tensors.items())),
        "dtype_elements": dict(sorted(dtype_elements.items())),
    }


def _causal_model(model: Any) -> Any:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def _embeddings_are_tied(model: Any) -> bool:
    return model.get_input_embeddings().weight is model.get_output_embeddings().weight


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
