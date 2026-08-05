"""Strict JSON-only validator for frozen attached dtype numerics evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError
from .tool_router_attached_dtype_isolation import analyze_path_repeat_stability
from .tool_router_attached_dtype_numerics import (
    ATTACHED_DTYPE_NUMERICS_VERSION,
    CLASSIFICATION,
    REGISTERED_OUTPUT_STAGES,
    classify_attached_dtype_numerics,
)
from .tool_router_merge_remediation import token_ids_sha256

BF16_PATH = "bf16_attached_adapter"
FP32_PATH = "fp32_attached_adapter"
PATH_ORDER = (BF16_PATH, FP32_PATH)
TARGET_STEP_INDEX = 45
TARGET_INPUT_TOKEN_ID = 788
TARGET_CACHE_POSITION = 383
TARGET_FORWARD_CALLS = 48
MAX_RESIDUAL_CUDA_BYTES = 16 * 1024 * 1024
EXPECTED_INPUT_TOKEN_COUNT = 339
EXPECTED_INPUT_TOKEN_SHA256 = (
    "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
)
EXPECTED_CAPTURE_PLAN_SHA256 = (
    "sha256:945dc2b468edf361b73189e7adf1f4ef61599da4fd942942591fdc13c073b38a"
)
EXPECTED_CAPTURE_EVENT_SEQUENCE_SHA256 = (
    "sha256:875edb689b4afef1472a746953ea581de42c4a55b05d810dcbf3d8a05c870ef9"
)
EXPECTED_CAPTURE_MANIFEST_SHA256 = {
    BF16_PATH: (
        "sha256:e40227233795e440fb9138c542a843d3e3915e0f985347355c64717af62ba630"
    ),
    FP32_PATH: (
        "sha256:f0893f13c404a433005c3a7a3ac9fc0e29bc1a47b95423b0d1b2667111b93163"
    ),
}
EXPECTED_MODULE_COMPARISON_MANIFEST_SHA256 = (
    "sha256:f136842f6754030a07a29d7d5172ee6c1192e82b123f391a9768eb2ceac9befe"
)
EXPECTED_FIRST_UNEQUAL_INDEX = 1
EXPECTED_FIRST_UNEQUAL_MODULE = "model.layers.0.input_layernorm"
RUN_PLAN = (
    (BF16_PATH, 1, "bf16-attached-numerics-r1"),
    (FP32_PATH, 1, "fp32-attached-numerics-r1"),
    (FP32_PATH, 2, "fp32-attached-numerics-r2"),
    (BF16_PATH, 2, "bf16-attached-numerics-r2"),
)
REPRESENTATIVE_RUN = {
    BF16_PATH: "bf16-attached-numerics-r1",
    FP32_PATH: "fp32-attached-numerics-r1",
}
REPEAT_RUN = {
    BF16_PATH: "bf16-attached-numerics-r2",
    FP32_PATH: "fp32-attached-numerics-r2",
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset(
    {
        "attached_dtype_numerics_version",
        "experiment_id",
        "source_experiment_id",
        "source_gate_experiment_id",
        "source_lineage",
        "training_lock_sha256",
        "config_sha256",
        "adapter_files",
        "model_weight_sha256",
        "prompt_sha256",
        "eval_digest",
        "example_id",
        "input_token_count",
        "input_token_ids_sha256",
        "storage_audit",
        "environment",
        "protocol",
        "capture_plan",
        "capture_plan_sha256",
        "frozen_path_references",
        "runs",
        "path_repeat_stability",
        "path_reproduction",
        "capture_repeat_stability",
        "paired_comparison_repeat",
        "module_comparisons",
        "module_comparison_manifest_sha256",
        "module_analysis",
        "lm_head_frozen_delta_link",
        "delta_statistics_scope",
        "classification",
        "causal_scope",
        "numerics_gate",
        "remediation_gate",
        "acceptance",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "module_tensor_payload_saved",
        "module_tensor_sidecar_allowed",
        "merged_artifact_saved",
        "merged_artifact_allowed",
        "constraints",
        "locked_next_action",
        "runtime_eligible",
        "runtime_eligibility_reason",
        "offline",
    }
)
_RUN_KEYS = frozenset(
    {
        "run_id",
        "path",
        "repeat",
        "order_index",
        "fresh_load",
        "materialization_form",
        "base_load_dtype",
        "generated_token_ids",
        "token_count",
        "token_ids_sha256",
        "output_sha256",
        "precision_audit",
        "generation_trace",
        "target_alignment",
        "target_alignment_passed",
        "capture_records",
        "capture_events",
        "capture_record_count",
        "capture_event_sequence_sha256",
        "capture_manifest_sha256",
        "capture_plan_sha256",
        "capture_plan_passed",
        "lm_head_raw_logit_linked",
        "path_protocol_passed",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
)
_CAPTURE_EVENT_KEYS = frozenset(
    {
        "event_index",
        "module_name",
        "module_type",
        "occurrence_index",
        "call_index",
        "generation_step_index",
        "io_kind",
        "tensor_path",
    }
)
_CAPTURE_RECORD_KEYS = frozenset(
    {
        *_CAPTURE_EVENT_KEYS,
        "native_dtype",
        "native_shape",
        "native_stride",
        "native_layout",
        "comparison_dtype",
        "elements",
        "finite_elements",
        "all_finite",
        "native_payload_sha256",
        "canonical_float32_sha256",
        "signed_zero_normalized",
        "capture_plan_sha256",
        "record_sha256",
    }
)


def validate_attached_dtype_numerics_evidence(
    data: object,
    *,
    source_isolation: Mapping[str, Any],
    expected_source_lineage: Mapping[str, str],
    expected_adapter_files: list[dict[str, Any]],
    expected_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the complete frozen gate without importing ML dependencies."""

    evidence = _mapping(data, "$", _TOP_LEVEL_KEYS, "INVALID_EVIDENCE_SCHEMA")
    _finite_tree(evidence, "$")
    _validate_source_contract(
        evidence,
        source_isolation=source_isolation,
        expected_source_lineage=expected_source_lineage,
        expected_adapter_files=expected_adapter_files,
        expected_environment=expected_environment,
    )
    expected_plan = _expected_capture_plan()
    _require_exact(evidence["capture_plan"], expected_plan, "INVALID_CAPTURE_PLAN", "$.capture_plan")
    _require_exact(
        evidence["capture_plan_sha256"],
        EXPECTED_CAPTURE_PLAN_SHA256,
        "INVALID_CAPTURE_PLAN_DIGEST",
        "$.capture_plan_sha256",
    )
    _require_exact(
        _sha256(_canonical_json(expected_plan)),
        EXPECTED_CAPTURE_PLAN_SHA256,
        "CAPTURE_PLAN_DIGEST_MISMATCH",
        "$.capture_plan",
    )
    expected_protocol = _expected_protocol(source_isolation)
    _require_exact(evidence["protocol"], expected_protocol, "INVALID_PROTOCOL", "$.protocol")

    runs = _validate_runs(evidence, source_isolation)
    by_path = {
        path: [run for run in runs if run["path"] == path] for path in PATH_ORDER
    }
    path_repeat = _recompute_path_repeat_stability(by_path)
    _require_exact(
        evidence["path_repeat_stability"],
        path_repeat,
        "PATH_REPEAT_STABILITY_MISMATCH",
        "$.path_repeat_stability",
    )
    path_reproduction = _recompute_path_reproduction(by_path, source_isolation)
    _require_exact(
        evidence["path_reproduction"],
        path_reproduction,
        "PATH_REPRODUCTION_MISMATCH",
        "$.path_reproduction",
    )
    capture_repeat = _recompute_capture_repeat_stability(by_path)
    _require_exact(
        evidence["capture_repeat_stability"],
        capture_repeat,
        "CAPTURE_REPEAT_STABILITY_MISMATCH",
        "$.capture_repeat_stability",
    )

    comparisons = _list(evidence["module_comparisons"], "$.module_comparisons")
    if len(comparisons) != len(REGISTERED_OUTPUT_STAGES):
        _fail(
            "INVALID_MODULE_COMPARISON_COUNT",
            "$.module_comparisons",
            repr(len(comparisons)),
        )
    comparison_manifest = _sha256(_canonical_json(comparisons))
    _require_exact(
        evidence["module_comparison_manifest_sha256"],
        EXPECTED_MODULE_COMPARISON_MANIFEST_SHA256,
        "INVALID_MODULE_COMPARISON_MANIFEST",
        "$.module_comparison_manifest_sha256",
    )
    _require_exact(
        comparison_manifest,
        EXPECTED_MODULE_COMPARISON_MANIFEST_SHA256,
        "MODULE_COMPARISON_MANIFEST_MISMATCH",
        "$.module_comparisons",
    )
    _validate_comparison_capture_links(comparisons, runs)
    module_analysis = classify_attached_dtype_numerics(
        comparisons,
        bf16_repeat_stable=path_repeat[BF16_PATH]["passed"],
        fp32_repeat_stable=path_repeat[FP32_PATH]["passed"],
        bf16_reference_reproduced=path_reproduction[BF16_PATH]["passed"],
        fp32_reference_reproduced=path_reproduction[FP32_PATH]["passed"],
        capture_plan_executed=all(run["capture_plan_passed"] for run in runs),
        target_forward_aligned=all(run["target_alignment_passed"] for run in runs),
        lm_head_raw_logit_linked=all(run["lm_head_raw_logit_linked"] for run in runs),
        attached_execution_form_fixed=True,
        source_inputs_unchanged=True,
        module_tensor_payload_absent=True,
    )
    if (
        module_analysis["first_unequal_module_index"] != EXPECTED_FIRST_UNEQUAL_INDEX
        or module_analysis["first_unequal_module"] != EXPECTED_FIRST_UNEQUAL_MODULE
        or module_analysis["classification"] != CLASSIFICATION
    ):
        _fail("FROZEN_MODULE_ANALYSIS_DRIFT", "$.module_analysis", repr(module_analysis))
    _require_exact(
        evidence["module_analysis"],
        module_analysis,
        "MODULE_ANALYSIS_MISMATCH",
        "$.module_analysis",
    )
    _validate_paired_comparison_repeat(evidence["paired_comparison_repeat"])
    lm_head_link = _expected_lm_head_link(comparisons[-1], source_isolation)
    _require_exact(
        evidence["lm_head_frozen_delta_link"],
        lm_head_link,
        "LM_HEAD_FROZEN_DELTA_LINK_MISMATCH",
        "$.lm_head_frozen_delta_link",
    )
    _validate_derived_policy(
        evidence,
        runs=runs,
        path_repeat=path_repeat,
        path_reproduction=path_reproduction,
        capture_repeat=capture_repeat,
        module_analysis=module_analysis,
        lm_head_link=lm_head_link,
    )
    return {
        "frozen_gate_valid": True,
        "runs_validated": len(runs),
        "capture_records_validated": sum(
            len(run["capture_records"]) for run in runs
        ),
        "capture_manifests_validated": len(runs),
        "module_comparisons_validated": len(comparisons),
        "first_unequal_module": module_analysis["first_unequal_module"],
        "classification": module_analysis["classification"],
        "delta_statistics_scope": evidence["delta_statistics_scope"],
    }


def _validate_source_contract(
    evidence: Mapping[str, Any],
    *,
    source_isolation: Mapping[str, Any],
    expected_source_lineage: Mapping[str, str],
    expected_adapter_files: list[dict[str, Any]],
    expected_environment: Mapping[str, str],
) -> None:
    exact = {
        "attached_dtype_numerics_version": ATTACHED_DTYPE_NUMERICS_VERSION,
        "experiment_id": "fc-mvp-001-attached-dtype-numerics-v1",
        "source_experiment_id": source_isolation["source_experiment_id"],
        "source_gate_experiment_id": source_isolation["experiment_id"],
        "source_lineage": dict(expected_source_lineage),
        "training_lock_sha256": source_isolation["training_lock_sha256"],
        "config_sha256": source_isolation["config_sha256"],
        "adapter_files": expected_adapter_files,
        "model_weight_sha256": source_isolation["model_weight_sha256"],
        "prompt_sha256": source_isolation["prompt_sha256"],
        "eval_digest": source_isolation["eval_digest"],
        "example_id": "eval-001",
        "input_token_count": EXPECTED_INPUT_TOKEN_COUNT,
        "input_token_ids_sha256": EXPECTED_INPUT_TOKEN_SHA256,
        "storage_audit": source_isolation["storage_audit"],
        "environment": dict(expected_environment),
        "frozen_path_references": source_isolation["frozen_path_references"],
    }
    for key, expected in exact.items():
        _require_exact(evidence[key], expected, "SOURCE_CONTRACT_MISMATCH", f"$.{key}")


def _validate_runs(
    evidence: Mapping[str, Any],
    source_isolation: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw_runs = _list(evidence["runs"], "$.runs")
    if len(raw_runs) != len(RUN_PLAN):
        _fail("INVALID_RUN_COUNT", "$.runs", repr(len(raw_runs)))
    source_by_path = {
        path: next(run for run in source_isolation["runs"] if run["path"] == path)
        for path in PATH_ORDER
    }
    references = source_isolation["frozen_path_references"]
    runs: list[Mapping[str, Any]] = []
    for order_index, (path_name, repeat, run_id) in enumerate(RUN_PLAN):
        run_path = f"$.runs[{order_index}]"
        run = _mapping(raw_runs[order_index], run_path, _RUN_KEYS, "INVALID_RUN_SCHEMA")
        exact = {
            "run_id": run_id,
            "path": path_name,
            "repeat": repeat,
            "order_index": order_index,
            "fresh_load": True,
            "materialization_form": "attached_factorized_lora",
            "base_load_dtype": "bfloat16" if path_name == BF16_PATH else "float32",
            "token_count": 48,
            "precision_audit": source_by_path[path_name]["precision_audit"],
            "target_alignment_passed": True,
            "capture_record_count": len(REGISTERED_OUTPUT_STAGES),
            "capture_event_sequence_sha256": EXPECTED_CAPTURE_EVENT_SEQUENCE_SHA256,
            "capture_manifest_sha256": EXPECTED_CAPTURE_MANIFEST_SHA256[path_name],
            "capture_plan_sha256": EXPECTED_CAPTURE_PLAN_SHA256,
            "capture_plan_passed": True,
            "lm_head_raw_logit_linked": True,
            "path_protocol_passed": True,
        }
        for key, expected in exact.items():
            _require_exact(run[key], expected, "RUN_CONTRACT_MISMATCH", f"{run_path}.{key}")
        tokens = _tokens(run["generated_token_ids"], f"{run_path}.generated_token_ids")
        reference = references[path_name]
        if (
            tokens != reference["generated_token_ids"]
            or token_ids_sha256(tokens) != run["token_ids_sha256"]
            or run["token_ids_sha256"] != reference["token_ids_sha256"]
            or run["output_sha256"] != reference["output_sha256"]
            or tokens[TARGET_STEP_INDEX] != reference["boundary_token_id"]
        ):
            _fail("FROZEN_RUN_REPRODUCTION_MISMATCH", run_path, repr(run_id))
        _validate_generation_trace(
            run["generation_trace"],
            source_by_path[path_name]["generation_trace"],
            reference,
            path_name,
            f"{run_path}.generation_trace",
        )
        _validate_target_alignment(
            run["target_alignment"],
            reference,
            path_name,
            f"{run_path}.target_alignment",
        )
        _validate_capture(run, path_name, run_path)
        for key in ("elapsed_seconds", "peak_gpu_memory_bytes"):
            if not _positive_finite(run[key]):
                _fail("INVALID_RUN_RESOURCE", f"{run_path}.{key}", repr(run[key]))
        for key in (
            "memory_allocated_before_load_bytes",
            "memory_allocated_after_release_bytes",
        ):
            value = run[key]
            if not _nonnegative_int(value) or value > MAX_RESIDUAL_CUDA_BYTES:
                _fail("INVALID_RUN_MEMORY_ISOLATION", f"{run_path}.{key}", repr(value))
        runs.append(run)
    return runs


def _validate_generation_trace(
    value: object,
    source_trace: Mapping[str, Any],
    reference: Mapping[str, Any],
    path_name: str,
    path: str,
) -> None:
    trace = _mapping(
        value,
        path,
        frozenset({"step_count", "vocabulary_size", "cache_returned", "scores", "raw_logits", "lm_head_output"}),
        "INVALID_GENERATION_TRACE",
    )
    _require_exact(trace["step_count"], 48, "INVALID_GENERATION_TRACE", f"{path}.step_count")
    _require_exact(trace["vocabulary_size"], 151936, "INVALID_GENERATION_TRACE", f"{path}.vocabulary_size")
    _require_exact(trace["cache_returned"], True, "INVALID_GENERATION_TRACE", f"{path}.cache_returned")
    _require_exact(trace["scores"], source_trace["scores"], "FROZEN_SCORE_TRACE_MISMATCH", f"{path}.scores")
    _require_exact(trace["raw_logits"], source_trace["raw_logits"], "FROZEN_RAW_TRACE_MISMATCH", f"{path}.raw_logits")
    expected_lm = {
        "native_dtype": "bfloat16" if path_name == BF16_PATH else "float32",
        "shape": [1, 1, 151936],
        "comparison_dtype": "float32",
        "all_finite": True,
        "comparison_step_index": TARGET_STEP_INDEX,
        "canonical_float32_sha256": reference["comparison_raw_logit_vector_sha256"],
        "frozen_raw_logit_vector_sha256": reference["comparison_raw_logit_vector_sha256"],
    }
    _require_exact(trace["lm_head_output"], expected_lm, "INVALID_LM_HEAD_TRACE", f"{path}.lm_head_output")


def _validate_target_alignment(
    value: object,
    reference: Mapping[str, Any],
    path_name: str,
    path: str,
) -> None:
    digest = reference["comparison_raw_logit_vector_sha256"]
    expected = {
        "call_index": TARGET_STEP_INDEX,
        "generation_step_index": TARGET_STEP_INDEX,
        "input_token_ids": [TARGET_INPUT_TOKEN_ID],
        "input_shape": [1, 1],
        "cache_position": [TARGET_CACHE_POSITION],
        "position_ids": [TARGET_CACHE_POSITION],
        "past_length": TARGET_CACHE_POSITION,
        "causal_forward_calls": TARGET_FORWARD_CALLS,
        "lm_head_output_shape": [1, 1, 151936],
        "lm_head_output_native_dtype": (
            "bfloat16" if path_name == BF16_PATH else "float32"
        ),
        "lm_head_output_comparison_vector_sha256": digest,
        "generated_raw_logit_comparison_vector_sha256": digest,
        "lm_head_output_canonical_float32_sha256": digest,
        "generated_raw_logit_canonical_float32_sha256": digest,
        "generated_raw_logit_frozen_vector_sha256": digest,
    }
    _require_exact(value, expected, "INVALID_TARGET_ALIGNMENT", path)


def _validate_capture(
    run: Mapping[str, Any],
    path_name: str,
    run_path: str,
) -> None:
    events = _list(run["capture_events"], f"{run_path}.capture_events")
    records = _list(run["capture_records"], f"{run_path}.capture_records")
    if len(events) != len(REGISTERED_OUTPUT_STAGES) or len(records) != len(events):
        _fail("INVALID_CAPTURE_COUNT", run_path, repr((len(events), len(records))))
    expected_events: list[dict[str, Any]] = []
    expected_dtype = "bfloat16" if path_name == BF16_PATH else "float32"
    for index, stage in enumerate(REGISTERED_OUTPUT_STAGES):
        contract = _expected_stage_contract(stage)
        expected_event = {
            "event_index": index,
            "module_name": stage,
            "module_type": contract["module_type"],
            "occurrence_index": 0,
            "call_index": TARGET_STEP_INDEX,
            "generation_step_index": TARGET_STEP_INDEX,
            "io_kind": "output",
            "tensor_path": contract["tensor_path"],
        }
        event_path = f"{run_path}.capture_events[{index}]"
        event = _mapping(events[index], event_path, _CAPTURE_EVENT_KEYS, "INVALID_CAPTURE_EVENT_SCHEMA")
        _require_exact(event, expected_event, "CAPTURE_EVENT_MISMATCH", event_path)
        expected_events.append(expected_event)

        record_path = f"{run_path}.capture_records[{index}]"
        record = _mapping(records[index], record_path, _CAPTURE_RECORD_KEYS, "INVALID_CAPTURE_RECORD_SCHEMA")
        for key, expected in expected_event.items():
            _require_exact(record[key], expected, "CAPTURE_RECORD_EVENT_MISMATCH", f"{record_path}.{key}")
        shape = contract["shape"]
        elements = math.prod(shape)
        record_exact = {
            "native_dtype": expected_dtype,
            "native_shape": shape,
            "native_stride": _contiguous_stride(shape),
            "native_layout": "strided",
            "comparison_dtype": "float32",
            "elements": elements,
            "finite_elements": elements,
            "all_finite": True,
            "signed_zero_normalized": True,
            "capture_plan_sha256": EXPECTED_CAPTURE_PLAN_SHA256,
        }
        for key, expected in record_exact.items():
            _require_exact(record[key], expected, "CAPTURE_RECORD_CONTRACT_MISMATCH", f"{record_path}.{key}")
        for key in ("native_payload_sha256", "canonical_float32_sha256", "record_sha256"):
            _digest(record[key], f"{record_path}.{key}")
        payload = dict(record)
        record_sha256 = payload.pop("record_sha256")
        _require_exact(
            record_sha256,
            _sha256(_canonical_json(payload)),
            "CAPTURE_RECORD_DIGEST_MISMATCH",
            f"{record_path}.record_sha256",
        )
    event_digest = _sha256(_canonical_json(expected_events))
    _require_exact(
        event_digest,
        EXPECTED_CAPTURE_EVENT_SEQUENCE_SHA256,
        "CAPTURE_EVENT_SEQUENCE_DIGEST_MISMATCH",
        f"{run_path}.capture_events",
    )
    manifest = _sha256(_canonical_json(records))
    _require_exact(
        manifest,
        EXPECTED_CAPTURE_MANIFEST_SHA256[path_name],
        "CAPTURE_MANIFEST_DIGEST_MISMATCH",
        f"{run_path}.capture_records",
    )


def _validate_comparison_capture_links(
    comparisons: list[Any],
    runs: list[Mapping[str, Any]],
) -> None:
    by_id = {run["run_id"]: run for run in runs}
    representative_records = {
        path: {
            record["module_name"]: record
            for record in by_id[REPRESENTATIVE_RUN[path]]["capture_records"]
        }
        for path in PATH_ORDER
    }
    for index, value in enumerate(comparisons):
        item = _mapping(value, f"$.module_comparisons[{index}]", None, "INVALID_MODULE_COMPARISON")
        name = REGISTERED_OUTPUT_STAGES[index]
        _require_exact(item.get("name"), name, "MODULE_COMPARISON_ORDER_MISMATCH", f"$.module_comparisons[{index}].name")
        _require_exact(
            item.get("bf16_float32_sha256"),
            representative_records[BF16_PATH][name]["canonical_float32_sha256"],
            "BF16_COMPARISON_CAPTURE_LINK_MISMATCH",
            f"$.module_comparisons[{index}].bf16_float32_sha256",
        )
        _require_exact(
            item.get("fp32_float32_sha256"),
            representative_records[FP32_PATH][name]["canonical_float32_sha256"],
            "FP32_COMPARISON_CAPTURE_LINK_MISMATCH",
            f"$.module_comparisons[{index}].fp32_float32_sha256",
        )
        _require_exact(
            item.get("shape"),
            representative_records[BF16_PATH][name]["native_shape"],
            "COMPARISON_CAPTURE_SHAPE_MISMATCH",
            f"$.module_comparisons[{index}].shape",
        )


def _validate_paired_comparison_repeat(value: object) -> None:
    expected = {
        "representative_run_ids": REPRESENTATIVE_RUN,
        "repeat_run_ids": REPEAT_RUN,
        "representative_manifest_sha256": EXPECTED_MODULE_COMPARISON_MANIFEST_SHA256,
        "repeat_manifest_sha256": EXPECTED_MODULE_COMPARISON_MANIFEST_SHA256,
        "exact_identity": True,
    }
    _require_exact(value, expected, "PAIRED_COMPARISON_REPEAT_MISMATCH", "$.paired_comparison_repeat")


def _recompute_path_repeat_stability(
    by_path: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        value = analyze_path_repeat_stability(
            first["generated_token_ids"],
            second["generated_token_ids"],
            first_output_sha256=first["output_sha256"],
            second_output_sha256=second["output_sha256"],
            first_score_trace_sha256=first["generation_trace"]["scores"]["trace_sha256"],
            second_score_trace_sha256=second["generation_trace"]["scores"]["trace_sha256"],
            first_raw_logit_trace_sha256=first["generation_trace"]["raw_logits"]["trace_sha256"],
            second_raw_logit_trace_sha256=second["generation_trace"]["raw_logits"]["trace_sha256"],
            first_score_vector_sha256=first["generation_trace"]["scores"]["comparison_step_vector_sha256"],
            second_score_vector_sha256=second["generation_trace"]["scores"]["comparison_step_vector_sha256"],
            first_raw_logit_vector_sha256=first["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"],
            second_raw_logit_vector_sha256=second["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"],
            precision_audits_identical=first["precision_audit"] == second["precision_audit"],
        )
        value["target_alignment_identity"] = first["target_alignment"] == second["target_alignment"]
        value["capture_manifest_identity"] = first["capture_manifest_sha256"] == second["capture_manifest_sha256"]
        value["passed"] = all(item for key, item in value.items() if key != "passed")
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _recompute_path_reproduction(
    by_path: Mapping[str, list[Mapping[str, Any]]],
    source_isolation: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        reference = source_isolation["frozen_path_references"][path]
        runs = by_path[path]
        value = {
            "token_identity": all(run["generated_token_ids"] == reference["generated_token_ids"] and run["token_count"] == reference["token_count"] and run["token_ids_sha256"] == reference["token_ids_sha256"] for run in runs),
            "output_identity": all(run["output_sha256"] == reference["output_sha256"] for run in runs),
            "score_trace_identity": all(run["generation_trace"]["scores"]["trace_sha256"] == reference["score_trace_sha256"] for run in runs),
            "raw_logit_trace_identity": all(run["generation_trace"]["raw_logits"]["trace_sha256"] == reference["raw_logit_trace_sha256"] for run in runs),
            "comparison_score_vector_identity": all(run["generation_trace"]["scores"]["comparison_step_vector_sha256"] == reference["comparison_score_vector_sha256"] for run in runs),
            "comparison_raw_logit_vector_identity": all(run["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"] == reference["comparison_raw_logit_vector_sha256"] for run in runs),
            "boundary_token_identity": all(run["generated_token_ids"][TARGET_STEP_INDEX] == reference["boundary_token_id"] for run in runs),
        }
        value["passed"] = all(value.values())
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _recompute_capture_repeat_stability(
    by_path: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        first_records = first["capture_records"]
        second_records = second["capture_records"]
        value = {
            "capture_record_count": len(first_records),
            "capture_manifest_identity": first["capture_manifest_sha256"] == second["capture_manifest_sha256"],
            "capture_record_identity": first_records == second_records,
            "native_payload_digest_identity": all(left["native_payload_sha256"] == right["native_payload_sha256"] for left, right in zip(first_records, second_records, strict=True)),
            "canonical_float32_digest_identity": all(left["canonical_float32_sha256"] == right["canonical_float32_sha256"] for left, right in zip(first_records, second_records, strict=True)),
            "capture_event_sequence_identity": first["capture_event_sequence_sha256"] == second["capture_event_sequence_sha256"],
        }
        value["passed"] = all(item for key, item in value.items() if key not in {"capture_record_count", "passed"})
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _expected_lm_head_link(
    comparison: Mapping[str, Any],
    source_isolation: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = source_isolation["raw_logit_evidence"]
    delta = frozen["delta"]
    digest_links = {
        BF16_PATH: comparison["bf16_float32_sha256"] == frozen["paths"][BF16_PATH]["comparison_vector_sha256"],
        FP32_PATH: comparison["fp32_float32_sha256"] == frozen["paths"][FP32_PATH]["comparison_vector_sha256"],
    }
    summary_links = {
        "elements": comparison["elements"] == delta["vocabulary_elements"],
        "different_elements": comparison["different_elements"] == delta["nonzero_elements"],
        "max_abs_delta": comparison["max_abs_delta"] == delta["max_abs_delta"],
        "mean_abs_delta_close": math.isclose(comparison["mean_abs_delta"], delta["mean_abs_delta"], rel_tol=1e-6, abs_tol=1e-9),
        "root_mean_square_delta_close": math.isclose(comparison["root_mean_square_delta"], delta["root_mean_square_delta"], rel_tol=1e-6, abs_tol=1e-9),
    }
    return {
        "frozen_raw_logit_vector_sha256": {path: frozen["paths"][path]["comparison_vector_sha256"] for path in PATH_ORDER},
        "capture_digest_links": digest_links,
        "frozen_delta_summary": delta,
        "summary_links": summary_links,
        "comparison_reduction": "stdlib_math_fsum_float64",
        "frozen_reduction": "upstream_probe_torch_reduction",
        "passed": all(digest_links.values()) and all(summary_links.values()),
    }


def _validate_derived_policy(
    evidence: Mapping[str, Any],
    *,
    runs: list[Mapping[str, Any]],
    path_repeat: Mapping[str, Any],
    path_reproduction: Mapping[str, Any],
    capture_repeat: Mapping[str, Any],
    module_analysis: Mapping[str, Any],
    lm_head_link: Mapping[str, Any],
) -> None:
    numerics_gate = {
        "bf16_attached_repeat_stable": path_repeat[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": path_repeat[FP32_PATH]["passed"],
        "bf16_capture_repeat_exact": capture_repeat[BF16_PATH]["passed"],
        "fp32_capture_repeat_exact": capture_repeat[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": path_reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": path_reproduction[FP32_PATH]["passed"],
        "target_forward_aligned": all(run["target_alignment_passed"] for run in runs),
        "capture_plan_executed": all(run["capture_plan_passed"] for run in runs),
        "cross_run_event_sequence_identical": len({run["capture_event_sequence_sha256"] for run in runs}) == 1,
        "paired_comparison_repeat_exact": True,
        "first_registered_module_difference_located": module_analysis["first_unequal_module"] is not None,
        "preceding_registered_outputs_identical": module_analysis["preceding_registered_outputs_identical"],
        "registered_lm_head_difference_quantified": module_analysis["registered_lm_head_difference_observed"],
        "lm_head_raw_logit_linked": all(run["lm_head_raw_logit_linked"] for run in runs),
        "frozen_lm_head_delta_linked": lm_head_link["passed"],
    }
    numerics_gate["passed"] = all(numerics_gate.values())
    _require_exact(evidence["numerics_gate"], numerics_gate, "NUMERICS_GATE_MISMATCH", "$.numerics_gate")
    acceptance = {
        "upstream_isolation_evidence_locked": True,
        "frozen_input_reproduced": True,
        "attached_execution_form_fixed": True,
        "base_dtype_only_treatment": True,
        "capture_plan_pre_registered": True,
        "capture_plan_executed": numerics_gate["capture_plan_executed"],
        "path_repeat_stability": path_repeat["passed"],
        "capture_repeat_stability": capture_repeat["passed"],
        "paired_comparison_repeat_identity": True,
        "target_forward_aligned": numerics_gate["target_forward_aligned"],
        "lm_head_raw_logit_linked": numerics_gate["lm_head_raw_logit_linked"],
        "source_inputs_unchanged": True,
        "fresh_load_memory_isolated": all(run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES and run["memory_allocated_after_release_bytes"] <= MAX_RESIDUAL_CUDA_BYTES for run in runs),
        "module_tensor_payload_absent": True,
    }
    _require_exact(evidence["acceptance"], acceptance, "ACCEPTANCE_MISMATCH", "$.acceptance")
    exact = {
        "delta_statistics_scope": "probe_derived_summary_algebra_and_frozen_manifest_only",
        "classification": CLASSIFICATION,
        "causal_scope": _expected_causal_scope(),
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "module_tensor_payload_saved": False,
        "module_tensor_sidecar_allowed": False,
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": _expected_constraints(),
        "locked_next_action": _expected_locked_next_action(),
        "runtime_eligible": False,
        "runtime_eligibility_reason": CLASSIFICATION,
        "offline": True,
    }
    for key, expected in exact.items():
        _require_exact(evidence[key], expected, "FROZEN_POLICY_MISMATCH", f"$.{key}")
    if not _positive_finite(evidence["elapsed_seconds"]):
        _fail("INVALID_RESOURCE_EVIDENCE", "$.elapsed_seconds", repr(evidence["elapsed_seconds"]))
    peaks = [run["peak_gpu_memory_bytes"] for run in runs]
    _require_exact(evidence["peak_gpu_memory_bytes"], max(peaks), "PEAK_MEMORY_MISMATCH", "$.peak_gpu_memory_bytes")


def _expected_protocol(source_isolation: Mapping[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(source_isolation["protocol"])
    protocol["run_plan"] = [
        {"run_id": run_id, "path": path, "repeat": repeat, "order_index": index}
        for index, (path, repeat, run_id) in enumerate(RUN_PLAN)
    ]
    protocol["capture_plan_sha256"] = EXPECTED_CAPTURE_PLAN_SHA256
    protocol["capture_output_count"] = len(REGISTERED_OUTPUT_STAGES)
    return protocol


def _expected_capture_plan() -> dict[str, Any]:
    return {
        "scope": "target_forward_pre_registered_attached_dtype_module_outputs",
        "target_generation_step_index": TARGET_STEP_INDEX,
        "selection_basis": "existing_layer0_causal_spine_plus_all_decoder_block_outputs",
        "registered_output_stages": list(REGISTERED_OUTPUT_STAGES),
        "registered_output_count": len(REGISTERED_OUTPUT_STAGES),
        "stage_contracts": [{"module_name": stage, **_expected_stage_contract(stage)} for stage in REGISTERED_OUTPUT_STAGES],
        "occurrence_index": 0,
        "tensor_selection": "first_tensor_leaf_per_registered_module_output",
        "hook_capture": "gpu_clone_then_post_generation_cpu_canonical_float32_summary",
        "comparison": {
            "dtype": "float32",
            "contiguous": True,
            "signed_zero_normalized": True,
            "finite_only": True,
            "exactness": "canonical_float32_exact_no_tolerance",
            "reduction": "fixed_flatten_order_stdlib_math_fsum_float64",
        },
        "serialized_module_tensor_payload": False,
        "does_not_cover": [
            "module_inputs",
            "all_tensor_leaves",
            "unregistered_functional_operations",
            "all_internal_modules_inside_decoder_layers_1_through_27",
            "earliest_difference_across_generation_history",
            "dtype_conditioned_kv_cache_history_isolation",
            "independent_causal_propagation_from_first_registered_difference",
            "low_level_cuda_kernel_identity",
        ],
    }


def _expected_stage_contract(stage: str) -> dict[str, Any]:
    hidden = [1, 1, 1536]
    if stage == "model.embed_tokens":
        return {"module_type": "torch.nn.modules.sparse.Embedding", "tensor_path": "output", "shape": hidden}
    if stage in {"model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm", "model.norm"}:
        return {"module_type": "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm", "tensor_path": "output", "shape": hidden}
    if stage in {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj"}:
        size = 1536 if stage.endswith(("q_proj", "o_proj")) else 256
        return {"module_type": "peft.tuners.lora.layer.Linear", "tensor_path": "output", "shape": [1, 1, size]}
    if stage in {"model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"}:
        return {"module_type": "torch.nn.modules.linear.Linear", "tensor_path": "output", "shape": [1, 1, 8960]}
    if stage == "model.layers.0.mlp.down_proj":
        return {"module_type": "torch.nn.modules.linear.Linear", "tensor_path": "output", "shape": hidden}
    if stage.startswith("model.layers."):
        return {"module_type": "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer", "tensor_path": "output[0]", "shape": hidden}
    if stage == "lm_head":
        return {"module_type": "torch.nn.modules.linear.Linear", "tensor_path": "output", "shape": [1, 1, 151936]}
    _fail("UNKNOWN_REGISTERED_STAGE", "$.capture_plan", stage)


def _expected_causal_scope() -> dict[str, Any]:
    return {
        "isolated_variable": "attached_path_base_and_inference_dtype_bfloat16_vs_float32",
        "controlled": [
            "same_bfloat16_checkpoint_source_values",
            "same_fp32_adapter_source_and_runtime_values",
            "same_attached_factorized_lora_execution_form",
            "same_eval_001_rendered_input_and_generation_prefix",
            "same_greedy_decoding_and_high_level_sdpa_dispatch",
            "same_fresh_model_load_lifecycle",
            "same_pre_registered_40_output_capture_plan",
        ],
        "supports": "the first exact canonical-FP32 inequality inside the pre-registered 40-output plan at the frozen target forward and a descriptive registered downstream total-dtype delta profile reaching the linked LM-head output",
        "does_not_support": [
            "first_unregistered_operation",
            "earliest_difference_across_generation_history",
            "independent_causal_propagation_from_the_first_registered_output",
            "separation_of_accumulated_dtype_conditioned_kv_cache_history",
            "unique_floating_point_or_cuda_root_cause",
            "peft_bug_claim",
            "pristine_fp32_checkpoint_comparison",
            "full_eval_generalization",
            "artifact_promotion",
            "runtime_eligibility",
        ],
        "json_only_limitation": "without module tensor payloads the offline validator checks exact digests, linkage, repeat identity, frozen manifests, and summary algebra but cannot independently recompute intermediate full-tensor different counts or moments",
    }


def _expected_constraints() -> dict[str, bool]:
    return {
        "attached_execution_form_change": False,
        "adapter_runtime_dtype_change": False,
        "source_checkpoint_values_change": False,
        "target_step_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_save": False,
        "merged_artifact_promotion": False,
        "module_tensor_sidecar": False,
        "module_tensor_payload": False,
        "first_registered_boundary_intervention": False,
    }


def _expected_locked_next_action() -> dict[str, Any]:
    constraints = _expected_constraints()
    return {
        "gate_id": "FC-MVP-001-attached-dtype-boundary-control-v1",
        "action": "pre-register one bounded control that separates accumulated dtype-conditioned target-forward state from current-forward computation at the observed first registered output without changing attached execution form or claiming a unique low-level root cause",
        "acceptance": {
            "numerics_evidence_frozen": True,
            "one_control_pre_registered": True,
            "target_forward_identity_preserved": True,
            "observed_boundary_tested_without_post_hoc_threshold": True,
            "causal_claim_bounded": True,
        },
        "constraints": constraints,
    }


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return stride


def _mapping(
    value: object,
    path: str,
    keys: frozenset[str] | None,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, path, repr(type(value)))
    if keys is not None and set(value) != set(keys):
        _fail(code, path, repr(sorted(value)))
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("EXPECTED_LIST", path, repr(type(value)))
    return value


def _tokens(value: object, path: str) -> list[int]:
    tokens = _list(value, path)
    if not tokens or any(
        not isinstance(token, int)
        or isinstance(token, bool)
        or token < 0
        or token >= 151936
        for token in tokens
    ):
        _fail("INVALID_TOKEN_IDS", path, repr(tokens))
    return tokens


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail("INVALID_SHA256", path, repr(value))
    return value


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _finite_tree(value: object, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE_VALUE", path, repr(value))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
        return
    _fail("INVALID_JSON_VALUE", path, repr(type(value)))


def _require_exact(
    actual: object,
    expected: object,
    code: str,
    path: str,
) -> None:
    if not _strict_equal(actual, expected):
        _fail(code, path, f"expected {expected!r}, got {actual!r}")


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)
