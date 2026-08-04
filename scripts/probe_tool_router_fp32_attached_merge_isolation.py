"""Isolate FP32 attached-Adapter versus safe-merged execution drift."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from peft.tuners.tuners_utils import (  # type: ignore[import-not-found]
    BaseTunerLayer,
)
from safetensors import __version__ as safetensors_version  # type: ignore[import-not-found]
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_fp32_attached_merge_isolation import (  # noqa: E402
    FP32_ATTACHED_MERGE_ISOLATION_VERSION,
    analyze_attached_repeat_stability,
    analyze_same_dtype_tokens,
    classify_same_dtype_effect,
    select_comparison_step,
)
from fullcycle_bridge.tool_router_merge_remediation import (  # noqa: E402
    token_ids_sha256,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)
from scripts import (  # noqa: E402
    probe_tool_router_bf16_merge_remediation as remediation_probe,
)
from scripts import (  # noqa: E402
    probe_tool_router_fp32_merge_drift as drift_probe,
)

ATTACHED_PATH = "fp32_attached_adapter"
MERGED_PATH = "fp32_safe_merged"
RUN_PLAN = (
    (ATTACHED_PATH, 1, "fp32_attached-r1"),
    (ATTACHED_PATH, 2, "fp32_attached-r2"),
    (MERGED_PATH, 1, "fp32_safe_merged-r1"),
)
COMPARISON_PATH_ORDER = (ATTACHED_PATH, MERGED_PATH)
TOP_K = drift_probe.TOP_K
MAX_RESIDUAL_CUDA_BYTES = remediation_probe.MAX_RESIDUAL_CUDA_BYTES
FROZEN_BF16_BOUNDARY_INDEX = 45
EXPECTED_STABILITY_SHA256 = drift_probe.EXPECTED_STABILITY_SHA256
EXPECTED_NUMERICS_SHA256 = drift_probe.EXPECTED_NUMERICS_SHA256
EXPECTED_REMEDIATION_SHA256 = drift_probe.EXPECTED_REMEDIATION_SHA256
EXPECTED_DRIFT_SHA256 = (
    "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3"
)
ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "deterministic_fp32_attached_merged_full_trace_identity",
        (
            "deterministic_fp32_attached_vs_merged_"
            "numerical_drift_without_token_drift"
        ),
        (
            "deterministic_fp32_attached_vs_merged_"
            "raw_logit_boundary_flip"
        ),
        (
            "deterministic_fp32_attached_vs_merged_"
            "logits_processor_boundary_flip"
        ),
        "deterministic_fp32_attached_vs_merged_mixed_logit_score_drift",
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
    parser.add_argument("--remediation-evidence", type=Path, required=True)
    parser.add_argument("--drift-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    drift_probe._verify_output_boundary(  # noqa: SLF001
        args.output,
        args.model_dir,
        args.adapter_dir,
    )

    config = _load_json(args.config)
    training = _load_json(args.training_evidence)
    stability = _load_json(args.stability_evidence)
    numerics = _load_json(args.numerics_evidence)
    remediation = _load_json(args.remediation_evidence)
    drift = _load_json(args.drift_evidence)
    _verify_sources(
        config,
        training,
        stability,
        numerics,
        remediation,
        drift,
        args,
    )
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
        len(input_token_ids) != drift["input_token_count"]
        or input_digest != drift["input_token_ids_sha256"]
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")

    storage_audit = {
        "base_checkpoint": drift_probe._safetensor_storage_audit(  # noqa: SLF001
            args.model_dir / config["model"]["weight_file"]
        ),
        "adapter": drift_probe._safetensor_storage_audit(  # noqa: SLF001
            args.adapter_dir / "adapter_model.safetensors"
        ),
    }
    source_storage_dtypes_locked = (
        storage_audit == drift_probe._expected_storage_audit()  # noqa: SLF001
    )
    if not source_storage_dtypes_locked:
        raise RuntimeError(f"source storage dtype drift: {storage_audit!r}")

    frozen_fp32_merged = _frozen_fp32_merged_reference(drift)
    frozen_bf16_context = _frozen_bf16_context(stability)
    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    traces: dict[str, dict[str, list[Any]]] = {}
    for path_name, repeat, run_id in RUN_PLAN:
        run, trace = _run_path_once(
            path_name,
            repeat,
            run_id,
            args.model_dir,
            args.adapter_dir,
            config,
            encoded_cpu,
            tokenizer,
        )
        runs.append(run)
        traces[run_id] = trace

    attached_runs = [run for run in runs if run["path"] == ATTACHED_PATH]
    merged_runs = [run for run in runs if run["path"] == MERGED_PATH]
    if len(attached_runs) != 2 or len(merged_runs) != 1:
        raise RuntimeError("fixed FP32 isolation run plan was not executed")
    attached_repeat_stability = analyze_attached_repeat_stability(
        attached_runs[0]["generated_token_ids"],
        attached_runs[1]["generated_token_ids"],
        first_output_sha256=attached_runs[0]["output_sha256"],
        second_output_sha256=attached_runs[1]["output_sha256"],
        first_score_trace_sha256=attached_runs[0]["generation_trace"]["scores"][
            "trace_sha256"
        ],
        second_score_trace_sha256=attached_runs[1]["generation_trace"]["scores"][
            "trace_sha256"
        ],
        first_raw_logit_trace_sha256=attached_runs[0]["generation_trace"][
            "raw_logits"
        ]["trace_sha256"],
        second_raw_logit_trace_sha256=attached_runs[1]["generation_trace"][
            "raw_logits"
        ]["trace_sha256"],
        precision_audits_identical=(
            attached_runs[0]["precision_audit"]
            == attached_runs[1]["precision_audit"]
        ),
    )
    if not attached_repeat_stability["passed"]:
        raise RuntimeError(
            "attached FP32 repeat stability failed: "
            f"{attached_repeat_stability!r}"
        )

    merged_candidate_reproduction = _merged_candidate_reproduction(
        merged_runs[0],
        frozen_fp32_merged,
        traces[merged_runs[0]["run_id"]],
    )
    if not merged_candidate_reproduction["passed"]:
        raise RuntimeError(
            "locked FP32 merged candidate reproduction failed: "
            f"{merged_candidate_reproduction!r}"
        )

    attached = attached_runs[0]
    merged = merged_runs[0]
    token_analysis = analyze_same_dtype_tokens(
        attached["generated_token_ids"],
        merged["generated_token_ids"],
    )
    comparison_step = select_comparison_step(
        token_analysis,
        frozen_boundary_index=FROZEN_BF16_BOUNDARY_INDEX,
    )
    step_index = comparison_step["step_index"]
    attached_repeat_stability["comparison_score_vector_identity"] = (
        drift_probe._float32_vector_sha256(  # noqa: SLF001
            traces[attached_runs[0]["run_id"]]["generated.scores"][step_index]
        )
        == drift_probe._float32_vector_sha256(  # noqa: SLF001
            traces[attached_runs[1]["run_id"]]["generated.scores"][step_index]
        )
    )
    attached_repeat_stability["comparison_raw_logit_vector_identity"] = (
        drift_probe._float32_vector_sha256(  # noqa: SLF001
            traces[attached_runs[0]["run_id"]]["generated.logits"][step_index]
        )
        == drift_probe._float32_vector_sha256(  # noqa: SLF001
            traces[attached_runs[1]["run_id"]]["generated.logits"][step_index]
        )
    )
    attached_repeat_stability["passed"] = all(
        value
        for key, value in attached_repeat_stability.items()
        if key != "passed"
    )
    if not attached_repeat_stability["passed"]:
        raise RuntimeError(
            "attached FP32 comparison-step stability failed: "
            f"{attached_repeat_stability!r}"
        )
    attached_emitted_token_id = attached["generated_token_ids"][step_index]
    merged_emitted_token_id = merged["generated_token_ids"][step_index]
    compared_token_ids = _unique_token_ids(
        attached_emitted_token_id,
        merged_emitted_token_id,
        frozen_bf16_context["paths"]["bf16_attached_adapter"]["boundary_token_id"],
        frozen_bf16_context["paths"]["bf16_safe_merged"]["boundary_token_id"],
    )
    representative_traces = {
        ATTACHED_PATH: traces[attached["run_id"]],
        MERGED_PATH: traces[merged["run_id"]],
    }
    representative_runs = {ATTACHED_PATH: attached, MERGED_PATH: merged}
    selection_score_evidence = _step_evidence(
        source="generated.scores",
        semantics="processed_prediction_scores_after_logits_processors",
        value_key="score",
        step_index=step_index,
        comparison_basis=comparison_step["basis"],
        traces=representative_traces,
        runs=representative_runs,
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    raw_logit_evidence = _step_evidence(
        source="generated.logits",
        semantics="unprocessed_lm_head_prediction_scores",
        value_key="raw_logit",
        step_index=step_index,
        comparison_basis=comparison_step["basis"],
        traces=representative_traces,
        runs=representative_runs,
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    same_dtype_trace_identity = {
        "token_identity": token_analysis["cross_path_identical"],
        "score_trace_identity": (
            attached["generation_trace"]["scores"]["trace_sha256"]
            == merged["generation_trace"]["scores"]["trace_sha256"]
        ),
        "raw_logit_trace_identity": (
            attached["generation_trace"]["raw_logits"]["trace_sha256"]
            == merged["generation_trace"]["raw_logits"]["trace_sha256"]
        ),
        "comparison_score_vector_identity": (
            selection_score_evidence["paths"][ATTACHED_PATH][
                "comparison_vector_sha256"
            ]
            == selection_score_evidence["paths"][MERGED_PATH][
                "comparison_vector_sha256"
            ]
        ),
        "comparison_raw_logit_vector_identity": (
            raw_logit_evidence["paths"][ATTACHED_PATH][
                "comparison_vector_sha256"
            ]
            == raw_logit_evidence["paths"][MERGED_PATH][
                "comparison_vector_sha256"
            ]
        ),
    }
    classification = classify_same_dtype_effect(
        token_analysis,
        attached_repeat_stable=attached_repeat_stability["passed"],
        merged_candidate_reproduced=merged_candidate_reproduction["passed"],
        attached_emitted_token_id=attached_emitted_token_id,
        merged_emitted_token_id=merged_emitted_token_id,
        attached_score_top_token_id=selection_score_evidence["paths"][
            ATTACHED_PATH
        ]["top_token_ids"][0],
        merged_score_top_token_id=selection_score_evidence["paths"][MERGED_PATH][
            "top_token_ids"
        ][0],
        attached_raw_logit_top_token_id=raw_logit_evidence["paths"][
            ATTACHED_PATH
        ]["top_token_ids"][0],
        merged_raw_logit_top_token_id=raw_logit_evidence["paths"][MERGED_PATH][
            "top_token_ids"
        ][0],
        full_score_traces_identical=same_dtype_trace_identity[
            "score_trace_identity"
        ],
        full_raw_logit_traces_identical=same_dtype_trace_identity[
            "raw_logit_trace_identity"
        ],
        comparison_score_vectors_identical=same_dtype_trace_identity[
            "comparison_score_vector_identity"
        ],
        comparison_raw_logit_vectors_identical=same_dtype_trace_identity[
            "comparison_raw_logit_vector_identity"
        ],
    )
    if classification == "generation_score_alignment_failure":
        raise RuntimeError("processed generation score does not select emitted token")
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError(f"unexpected isolation classification: {classification}")

    _link_comparison_step_hashes(
        runs,
        traces,
        selection_score_evidence,
        raw_logit_evidence,
    )
    processed_scores_captured = _trace_evidence_linked(
        runs,
        selection_score_evidence,
        traces,
        trace_key="scores",
    )
    raw_logits_captured = _trace_evidence_linked(
        runs,
        raw_logit_evidence,
        traces,
        trace_key="raw_logits",
    )

    model_weight_path = args.model_dir / config["model"]["weight_file"]
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    path_protocols_executed = all(
        run["path_protocol_passed"] is True for run in runs
    )
    fresh_load_memory_isolated = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"]
        <= MAX_RESIDUAL_CUDA_BYTES
        for run in runs
    )
    score_argmax_aligned = all(
        selection_score_evidence["paths"][path]["top_token_ids"][0]
        == representative_runs[path]["generated_token_ids"][step_index]
        for path in COMPARISON_PATH_ORDER
    )
    exact_cached_step_captured = (
        processed_scores_captured
        and raw_logits_captured
        and all(
            run["generation_trace"]["cache_returned"] is True
            and step_index < run["generation_trace"]["step_count"]
            for run in runs
        )
    )
    source_adapter_unchanged = (
        adapter_manifest == training["final_adapter"]["files"]
    )
    source_model_unchanged = (
        file_sha256(model_weight_path)
        == f"sha256:{config['model']['weight_sha256']}"
    )
    eval_digest_unchanged = (
        fixture_digest(evaluation) == config["data"]["eval_digest"]
    )
    prompt_digest_unchanged = (
        file_sha256(prompt_path) == f"sha256:{config['prompt']['sha256']}"
    )
    isolation_gate = {
        "attached_fp32_repeat_stable": attached_repeat_stability["passed"],
        "fp32_merged_candidate_reproduced": merged_candidate_reproduction[
            "passed"
        ],
        "same_dtype_exact_cached_step_compared": exact_cached_step_captured,
        "processed_score_argmax_matches_generated_token": score_argmax_aligned,
        "raw_logits_captured": raw_logits_captured,
        "same_dtype_effect_classified": classification in ALLOWED_CLASSIFICATIONS,
    }
    isolation_gate["passed"] = all(isolation_gate.values())
    acceptance = {
        "upstream_evidence_locked": True,
        "frozen_input_reproduced": True,
        "attached_fp32_repeat_stable": attached_repeat_stability["passed"],
        "fp32_candidate_reproduced": merged_candidate_reproduction["passed"],
        "same_dtype_exact_step_compared": exact_cached_step_captured,
        "same_dtype_attached_vs_merged_effect_classified": (
            classification in ALLOWED_CLASSIFICATIONS
        ),
        "generation_score_alignment_verified": score_argmax_aligned,
        "path_protocols_executed": path_protocols_executed,
        "source_storage_dtypes_locked": source_storage_dtypes_locked,
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": source_model_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": prompt_digest_unchanged,
        "frozen_bf16_context_only": frozen_bf16_context["context_only"],
    }
    if not isolation_gate["passed"] or not all(acceptance.values()):
        raise RuntimeError(
            "FP32 attached/merge isolation evidence invalid: "
            f"gate={isolation_gate!r}, acceptance={acceptance!r}"
        )

    constraints = _constraints()
    result = {
        "fp32_attached_merge_isolation_version": (
            FP32_ATTACHED_MERGE_ISOLATION_VERSION
        ),
        "experiment_id": "fc-mvp-001-fp32-attached-merge-isolation-v1",
        "source_experiment_id": config["experiment_id"],
        "drift_evidence_sha256": file_sha256(args.drift_evidence),
        "remediation_evidence_sha256": file_sha256(args.remediation_evidence),
        "stability_evidence_sha256": file_sha256(args.stability_evidence),
        "numerics_evidence_sha256": file_sha256(args.numerics_evidence),
        "training_lock_sha256": file_sha256(ROOT / "requirements" / "training.lock"),
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "model_weight_sha256": file_sha256(model_weight_path),
        "prompt_sha256": file_sha256(prompt_path),
        "eval_digest": config["data"]["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "protocol": _protocol(config, tokenizer),
        "frozen_fp32_merged_reference": frozen_fp32_merged,
        "frozen_bf16_context": frozen_bf16_context,
        "runs": runs,
        "attached_repeat_stability": attached_repeat_stability,
        "merged_candidate_reproduction": merged_candidate_reproduction,
        "same_dtype_token_analysis": {
            **token_analysis,
            "attached_token_text": _optional_decode(
                tokenizer,
                token_analysis["attached_token_id"],
            ),
            "merged_token_text": _optional_decode(
                tokenizer,
                token_analysis["merged_token_id"],
            ),
        },
        "comparison_step": comparison_step,
        "selection_score_evidence": selection_score_evidence,
        "raw_logit_evidence": raw_logit_evidence,
        "same_dtype_trace_identity": same_dtype_trace_identity,
        "classification": classification,
        "causal_scope": _causal_scope(),
        "isolation_gate": isolation_gate,
        "remediation_gate": {
            "source_gate_passed": drift["remediation_gate"]["passed"],
            "new_remediation_tested": False,
            "passed": False,
        },
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(
            run["peak_gpu_memory_bytes"] for run in runs
        ),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _locked_next_action(classification),
        "runtime_eligible": False,
        "runtime_eligibility_reason": classification,
        "offline": True,
    }
    _validate_result_scalars(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _run_path_once(
    path_name: str,
    repeat: int,
    run_id: str,
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    encoded_cpu: Any,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {path_name}#{repeat} load exceeded residual CUDA ceiling: "
            f"{allocated_before} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    evidence: dict[str, Any] | None = None
    trace: dict[str, list[Any]] | None = None
    try:
        if path_name == ATTACHED_PATH:
            model, precision = _load_fp32_attached_model(
                model_dir,
                adapter_dir,
                config,
            )
        elif path_name == MERGED_PATH:
            model, precision = remediation_probe._load_fp32_merged_model(  # noqa: SLF001
                model_dir,
                adapter_dir,
                config,
            )
        else:
            raise RuntimeError(f"unknown path: {path_name}")
        dropout_audit = _lora_dropout_audit(model)
        precision["lora_dropout"] = dropout_audit
        remediation_probe._verify_generation_semantics(model, config)  # noqa: SLF001
        token_ids, trace, generation_trace = drift_probe._generate_trace(  # noqa: SLF001
            model,
            encoded_cpu,
            config,
            tokenizer,
        )
        precision["generation"] = {
            "score_dtypes": generation_trace["scores"]["native_dtypes"],
            "all_scores_float32": generation_trace["scores"]["native_dtypes"]
            == ["float32"],
            "raw_logit_dtypes": generation_trace["raw_logits"]["native_dtypes"],
            "all_raw_logits_float32": generation_trace["raw_logits"][
                "native_dtypes"
            ]
            == ["float32"],
            "dtype_semantics": "transformers_generate_return_tensor_dtype",
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        if path_name == ATTACHED_PATH:
            protocol_passed = _attached_protocol_passed(precision)
        else:
            protocol_passed = (
                remediation_probe._precision_protocol_passed(precision)  # noqa: SLF001
                and precision["generation"]["all_raw_logits_float32"] is True
                and precision["lora_dropout"]
                == {"modules": 0, "training_modules": 0}
            )
        if not protocol_passed:
            raise RuntimeError(
                f"{path_name}#{repeat} protocol failed: {precision!r}"
            )
        torch.cuda.synchronize()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        evidence = {
            "run_id": run_id,
            "path": path_name,
            "repeat": repeat,
            "fresh_load": True,
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": "sha256:"
            + hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "precision_audit": precision,
            "generation_trace": generation_trace,
            "path_protocol_passed": protocol_passed,
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
            f"fresh {path_name}#{repeat} load retained model-scale CUDA memory: "
            f"{allocated_after} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    if evidence is None or trace is None:
        raise RuntimeError(f"{path_name}#{repeat} did not produce evidence")
    evidence["memory_allocated_before_load_bytes"] = allocated_before
    evidence["memory_allocated_after_release_bytes"] = allocated_after
    return evidence, trace


def _load_fp32_attached_model(
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
    precision = remediation_probe._pre_merge_precision(model)  # noqa: SLF001
    if not remediation_probe._pre_merge_protocol_passed(precision):  # noqa: SLF001
        raise RuntimeError(f"attached FP32 protocol failed: {precision!r}")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    remediation_probe._verify_generation_semantics(model, config)  # noqa: SLF001
    return model, precision


def _lora_dropout_audit(model: Any) -> dict[str, int]:
    dropout_modules: list[Any] = []
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            dropout_modules.extend(module.lora_dropout.values())
    return {
        "modules": len(dropout_modules),
        "training_modules": sum(module.training for module in dropout_modules),
    }


def _attached_protocol_passed(value: dict[str, Any]) -> bool:
    generation = value.get("generation")
    return (
        remediation_probe._pre_merge_protocol_passed(value)  # noqa: SLF001
        and value.get("lora_dropout")
        == {
            "modules": remediation_probe.EXPECTED_LORA_TARGETS,
            "training_modules": 0,
        }
        and isinstance(generation, dict)
        and generation.get("score_dtypes") == ["float32"]
        and generation.get("all_scores_float32") is True
        and generation.get("raw_logit_dtypes") == ["float32"]
        and generation.get("all_raw_logits_float32") is True
        and generation.get("dtype_semantics")
        == "transformers_generate_return_tensor_dtype"
        and generation.get("autocast_enabled") is False
        and generation.get("training") is False
    )


def _step_evidence(
    *,
    source: str,
    semantics: str,
    value_key: str,
    step_index: int,
    comparison_basis: str,
    traces: dict[str, dict[str, list[Any]]],
    runs: dict[str, dict[str, Any]],
    compared_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    vectors = {path: traces[path][source][step_index] for path in COMPARISON_PATH_ORDER}
    paths: dict[str, Any] = {}
    for path_name in COMPARISON_PATH_ORDER:
        run = runs[path_name]
        vector = vectors[path_name]
        top = torch.topk(vector, k=TOP_K)
        top_ids = [int(value) for value in top.indices]
        top_values = [float(value) for value in top.values]
        if top_values[0] <= top_values[1]:
            raise RuntimeError(f"non-unique top {value_key} for {path_name}")
        emitted_token_id = run["generated_token_ids"][step_index]
        paths[path_name] = {
            "top_token_ids": top_ids,
            "top_token_texts": [
                _decode_token(tokenizer, token_id) for token_id in top_ids
            ],
            f"top_{value_key}s": top_values,
            "top_margin": top_values[0] - top_values[1],
            "emitted_token_id": emitted_token_id,
            "emitted_token_text": _decode_token(tokenizer, emitted_token_id),
            f"emitted_token_{value_key}": float(vector[emitted_token_id]),
            "compared_tokens": [
                {
                    "token_id": token_id,
                    "token_text": _decode_token(tokenizer, token_id),
                    value_key: float(vector[token_id]),
                    "rank": int((vector > vector[token_id]).sum()) + 1,
                }
                for token_id in compared_token_ids
            ],
            "comparison_vector_sha256": drift_probe._float32_vector_sha256(  # noqa: SLF001
                vector
            ),
        }
    delta = (vectors[ATTACHED_PATH] - vectors[MERGED_PATH]).abs()
    return {
        "step_index": step_index,
        "comparison_basis": comparison_basis,
        "shared_generated_prefix_tokens_before_step": step_index,
        "source": source,
        "semantics": semantics,
        "comparison_dtype": "float32",
        "top_k": TOP_K,
        "paths": paths,
        "delta": {
            "vocabulary_elements": delta.numel(),
            "nonzero_elements": int(torch.count_nonzero(delta)),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "root_mean_square_delta": float(torch.sqrt(torch.mean(delta.square()))),
        },
    }


def _link_comparison_step_hashes(
    runs: list[dict[str, Any]],
    traces: dict[str, dict[str, list[Any]]],
    selection_score_evidence: dict[str, Any],
    raw_logit_evidence: dict[str, Any],
) -> None:
    step_index = selection_score_evidence["step_index"]
    if raw_logit_evidence["step_index"] != step_index:
        raise RuntimeError("score and raw-logit evidence steps do not match")
    for run in runs:
        path_name = run["path"]
        trace = traces[run["run_id"]]
        for trace_key, source, evidence in (
            ("scores", "generated.scores", selection_score_evidence),
            ("raw_logits", "generated.logits", raw_logit_evidence),
        ):
            vector_hash = drift_probe._float32_vector_sha256(  # noqa: SLF001
                trace[source][step_index]
            )
            expected_hash = evidence["paths"][path_name][
                "comparison_vector_sha256"
            ]
            if vector_hash != expected_hash:
                raise RuntimeError(
                    f"repeat trace comparison hash drift for {path_name}"
                )
            summary = run["generation_trace"][trace_key]
            summary["comparison_step_index"] = step_index
            summary["comparison_step_vector_sha256"] = vector_hash


def _trace_evidence_linked(
    runs: list[dict[str, Any]],
    step_evidence: dict[str, Any],
    traces: dict[str, dict[str, list[Any]]],
    *,
    trace_key: str,
) -> bool:
    step_index = step_evidence.get("step_index")
    source = "generated.scores" if trace_key == "scores" else "generated.logits"
    return isinstance(step_index, int) and all(
        run["generation_trace"]["step_count"] == run["token_count"]
        and run["generation_trace"][trace_key]["all_finite"] is True
        and run["generation_trace"][trace_key]["shape_per_step"]
        == [1, run["generation_trace"]["vocabulary_size"]]
        and run["generation_trace"][trace_key]["comparison_step_index"]
        == step_index
        and run["generation_trace"][trace_key][
            "comparison_step_vector_sha256"
        ]
        == drift_probe._float32_vector_sha256(  # noqa: SLF001
            traces[run["run_id"]][source][step_index]
        )
        and run["generation_trace"][trace_key][
            "comparison_step_vector_sha256"
        ]
        == step_evidence["paths"][run["path"]]["comparison_vector_sha256"]
        for run in runs
    )


def _merged_candidate_reproduction(
    run: dict[str, Any],
    reference: dict[str, Any],
    trace: dict[str, list[Any]],
) -> dict[str, bool]:
    comparison_step = reference["comparison_step_index"]
    result = {
        "token_identity": (
            run["token_count"] == reference["token_count"]
            and run["token_ids_sha256"] == reference["token_ids_sha256"]
        ),
        "output_identity": run["output_sha256"] == reference["output_sha256"],
        "score_trace_identity": (
            run["generation_trace"]["scores"]["trace_sha256"]
            == reference["score_trace_sha256"]
        ),
        "raw_logit_trace_identity": (
            run["generation_trace"]["raw_logits"]["trace_sha256"]
            == reference["raw_logit_trace_sha256"]
        ),
        "comparison_score_vector_identity": (
            drift_probe._float32_vector_sha256(  # noqa: SLF001
                trace["generated.scores"][comparison_step]
            )
            == reference["comparison_score_vector_sha256"]
        ),
        "comparison_raw_logit_vector_identity": (
            drift_probe._float32_vector_sha256(  # noqa: SLF001
                trace["generated.logits"][comparison_step]
            )
            == reference["comparison_raw_logit_vector_sha256"]
        ),
    }
    result["passed"] = all(result.values())
    return result


def _frozen_fp32_merged_reference(drift: dict[str, Any]) -> dict[str, Any]:
    matches = [
        run for run in drift.get("runs", []) if run.get("path") == MERGED_PATH
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen FP32 merged drift run is invalid")
    run = matches[0]
    score_summary = run["generation_trace"]["scores"]
    raw_summary = run["generation_trace"]["raw_logits"]
    if (
        score_summary["divergent_step_index"] != FROZEN_BF16_BOUNDARY_INDEX
        or raw_summary["divergent_step_index"] != FROZEN_BF16_BOUNDARY_INDEX
    ):
        raise RuntimeError("frozen FP32 merged comparison step is invalid")
    return {
        "path": MERGED_PATH,
        "source_experiment_id": drift["experiment_id"],
        "token_count": run["token_count"],
        "token_ids_sha256": run["token_ids_sha256"],
        "output_sha256": run["output_sha256"],
        "score_trace_sha256": run["generation_trace"]["scores"]["trace_sha256"],
        "raw_logit_trace_sha256": run["generation_trace"]["raw_logits"][
            "trace_sha256"
        ],
        "comparison_step_index": FROZEN_BF16_BOUNDARY_INDEX,
        "comparison_score_vector_sha256": score_summary[
            "divergent_step_comparison_vector_sha256"
        ],
        "comparison_raw_logit_vector_sha256": raw_summary[
            "divergent_step_comparison_vector_sha256"
        ],
    }


def _frozen_bf16_context(stability: dict[str, Any]) -> dict[str, Any]:
    token_analysis = stability.get("token_analysis", {})
    if (
        token_analysis.get("first_divergent_token_index")
        != FROZEN_BF16_BOUNDARY_INDEX
        or token_analysis.get("independent_token_id") != 1866
        or token_analysis.get("merged_token_id") != 3849
    ):
        raise RuntimeError("frozen BF16 token boundary context is invalid")
    paths: dict[str, Any] = {}
    for source_path, target_path, boundary_key in (
        ("independent", "bf16_attached_adapter", "independent_token_id"),
        ("merged", "bf16_safe_merged", "merged_token_id"),
    ):
        matches = [
            run
            for run in stability.get("runs", [])
            if run.get("path") == source_path
        ]
        if len(matches) != 2:
            raise RuntimeError(f"frozen BF16 {source_path} repeats are invalid")
        identities = {
            (
                run.get("token_count"),
                run.get("token_ids_sha256"),
                run.get("output_sha256"),
            )
            for run in matches
        }
        if len(identities) != 1:
            raise RuntimeError(f"frozen BF16 {source_path} is not repeat-stable")
        token_count, token_digest, output_digest = identities.pop()
        paths[target_path] = {
            "token_count": token_count,
            "token_ids_sha256": token_digest,
            "output_sha256": output_digest,
            "boundary_token_id": token_analysis[boundary_key],
        }
    return {
        "context_only": True,
        "gpu_paths_rerun": False,
        "source_experiment_id": stability["experiment_id"],
        "first_divergent_token_index": FROZEN_BF16_BOUNDARY_INDEX,
        "paths": paths,
    }


def _protocol(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": [
            {"run_id": run_id, "path": path_name, "repeat": repeat}
            for path_name, repeat, run_id in RUN_PLAN
        ],
        "fresh_loads_per_path": {ATTACHED_PATH: 2, MERGED_PATH: 1},
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "paths": {
            ATTACHED_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": False,
                "merge": False,
                "inference_parameter_dtype": "float32",
            },
            MERGED_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": False,
                "merge_dtype": "float32",
                "inference_parameter_dtype": "float32",
                "merge": True,
                "safe_merge": True,
                "adapter_names": ["default"],
            },
        },
        "generation": {
            "attn_implementation": config["generation"]["attn_implementation"],
            "attention_class": "Qwen2Attention",
            "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
            "low_level_cuda_kernel_identity_claimed": False,
            "output_attentions": False,
            "do_sample": False,
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "use_cache": config["generation"]["use_cache"],
            "repetition_penalty": 1.1,
            "model_eos_token_ids": [151645, 151643],
            "model_pad_token_id": 151643,
            "call_pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_logits": True,
            "generate_return_dtype_semantics": (
                "return_tensor_dtype_not_internal_compute_dtype"
            ),
            "tf32": False,
            "autocast": False,
            "device": "cuda:0",
        },
        "sdp_kernel_flags": drift_probe._sdp_kernel_flags(),  # noqa: SLF001
    }


def _constraints() -> dict[str, bool]:
    return {
        "failed_candidate_change": False,
        "frozen_bf16_path_rerun": False,
        "locked_path_dtype_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_promotion": False,
    }


def _causal_scope() -> dict[str, Any]:
    return {
        "isolated_variable": "attached_adapter_vs_materialized_safe_merge_execution",
        "controlled": [
            "base_checkpoint_values",
            "base_and_adapter_runtime_dtype_float32",
            "adapter_weights",
            "eval_001_rendered_input",
            "greedy_decoding",
            "high_level_transformers_sdpa_dispatch",
            "fresh_model_load_lifecycle",
        ],
        "supports": (
            "classification of the observed same-dtype FP32 execution-form effect"
        ),
        "does_not_support": [
            "peft_merge_implementation_bug_claim",
            "low_level_cuda_kernel_identity_or_root_cause",
            "full_eval_generalization",
            "merged_artifact_promotion",
            "runtime_eligibility",
        ],
    }


def _locked_next_action(classification: str) -> dict[str, Any]:
    constraints = _constraints()
    if classification == "deterministic_fp32_attached_merged_full_trace_identity":
        return {
            "gate_id": "FC-MVP-001-attached-dtype-isolation-v1",
            "action": (
                "compare only the frozen BF16 attached Adapter path with the "
                "repeat-stable FP32 attached Adapter path on eval-001, preserving "
                "attached execution while isolating parameter compute dtype"
            ),
            "acceptance": {
                "bf16_attached_reference_reproduced": True,
                "fp32_attached_reference_reproduced": True,
                "same_execution_exact_step_compared": True,
                "attached_dtype_effect_classified": True,
                "source_inputs_unchanged": True,
            },
            "constraints": constraints,
        }
    if classification == (
        "deterministic_fp32_attached_vs_merged_logits_processor_boundary_flip"
    ):
        return {
            "gate_id": (
                "FC-MVP-001-fp32-attached-merge-logits-processor-analysis-v1"
            ),
            "action": (
                "reproduce the unchanged logits processor transform at the same "
                "FP32 attached-versus-merged cached step before any new candidate"
            ),
            "acceptance": {
                "fp32_paths_reproduced": True,
                "processor_transform_reproduced": True,
                "selected_token_scores_reconciled": True,
                "source_inputs_unchanged": True,
            },
            "constraints": constraints,
        }
    if classification == (
        "deterministic_fp32_attached_vs_merged_"
        "numerical_drift_without_token_drift"
    ):
        return {
            "gate_id": "FC-MVP-001-fp32-attached-merge-numerics-v1",
            "action": (
                "at frozen comparison step index 45, locate the first module "
                "numerical divergence between repeat-stable FP32 attached LoRA "
                "execution and the unchanged materialized safe-merged execution, "
                "without claiming a same-dtype token boundary"
            ),
            "acceptance": {
                "fp32_paths_reproduced": True,
                "comparison_step_reproduced": True,
                "first_divergent_module_located": True,
                "operation_order_boundary_quantified": True,
                "source_inputs_unchanged": True,
            },
            "constraints": constraints,
        }
    return {
        "gate_id": "FC-MVP-001-fp32-attached-merge-numerics-v1",
        "action": (
            "at the first same-dtype FP32 token boundary, locate the first "
            "module divergence between attached LoRA execution and the unchanged "
            "materialized safe-merged execution without proposing a new candidate"
        ),
        "acceptance": {
            "fp32_paths_reproduced": True,
            "first_divergent_module_located": True,
            "operation_order_boundary_quantified": True,
            "source_inputs_unchanged": True,
        },
        "constraints": constraints,
    }


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    numerics: dict[str, Any],
    remediation: dict[str, Any],
    drift: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    drift_probe._verify_sources(  # noqa: SLF001
        config,
        training,
        stability,
        numerics,
        remediation,
        args,
    )
    config_digest = canonical_config_sha256(config)
    if (
        file_sha256(args.stability_evidence) != EXPECTED_STABILITY_SHA256
        or file_sha256(args.numerics_evidence) != EXPECTED_NUMERICS_SHA256
        or file_sha256(args.remediation_evidence) != EXPECTED_REMEDIATION_SHA256
        or file_sha256(args.drift_evidence) != EXPECTED_DRIFT_SHA256
        or drift.get("config_sha256") != config_digest
        or drift.get("stability_evidence_sha256")
        != file_sha256(args.stability_evidence)
        or drift.get("numerics_evidence_sha256")
        != file_sha256(args.numerics_evidence)
        or drift.get("remediation_evidence_sha256")
        != file_sha256(args.remediation_evidence)
        or drift.get("classification")
        != "deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip"
        or drift.get("analysis_gate", {}).get("passed") is not True
        or drift.get("remediation_gate", {}).get("passed") is not False
        or drift.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-fp32-attached-merge-isolation-v1"
        or drift.get("merged_artifact_saved") is not False
        or drift.get("merged_artifact_allowed") is not False
        or drift.get("runtime_eligible") is not False
    ):
        raise RuntimeError("required FP32 drift-analysis source chain is invalid")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if drift.get("adapter_files") != adapter_manifest:
        raise RuntimeError("drift-analysis Adapter artifact mismatch")
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if (
        evaluation[0]["example_id"] != "eval-001"
        or drift.get("example_id") != "eval-001"
        or fixture_digest(evaluation) != config["data"]["eval_digest"]
    ):
        raise RuntimeError("locked isolation probe requires frozen eval-001")


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


def _unique_token_ids(*values: int) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _optional_decode(tokenizer: Any, token_id: object) -> str | None:
    return _decode_token(tokenizer, token_id) if isinstance(token_id, int) else None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise RuntimeError(f"non-finite JSON constant: {value}")


def _validate_result_scalars(result: dict[str, Any]) -> None:
    elapsed = result.get("elapsed_seconds")
    peak_memory = result.get("peak_gpu_memory_bytes")
    if not isinstance(elapsed, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"invalid elapsed_seconds: {elapsed!r}")
    if (
        not isinstance(peak_memory, int)
        or isinstance(peak_memory, bool)
        or peak_memory < 0
    ):
        raise RuntimeError(f"invalid peak_gpu_memory_bytes: {peak_memory!r}")
    _require_finite_json(result, "$")


def _require_finite_json(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite result at {path}: {value!r}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{path}[{index}]")


if __name__ == "__main__":
    raise SystemExit(main())
