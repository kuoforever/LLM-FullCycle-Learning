"""Strict JSON-only validation for the attached-dtype boundary control."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError
from .tool_router_attached_dtype_boundary_control import (
    ATTACHED_DTYPE_BOUNDARY_CONTROL_VERSION,
    BF16_PATH,
    BOUNDARY_MODULE,
    BOUNDARY_MODULE_TYPE,
    CONTROL_ID,
    FP32_PATH,
    PATH_ORDER,
    classify_attached_dtype_boundary_control,
)
from .tool_router_attached_dtype_isolation import analyze_path_repeat_stability
from .tool_router_merge_remediation import token_ids_sha256

TARGET_STEP_INDEX = 45
TARGET_INPUT_TOKEN_ID = 788
TARGET_CACHE_POSITION = 383
TARGET_FORWARD_CALLS = 48
EXPECTED_INPUT_TOKEN_COUNT = 339
EXPECTED_INPUT_TOKEN_SHA256 = (
    "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
)
EXPECTED_FORWARD_SOURCE_SHA256 = (
    "sha256:7d352cd525210579aabf6191da9bfc1b1086878c303fb1ea8b8ea21d0e081342"
)
EXPECTED_NUMERICS_EVIDENCE_SHA256 = (
    "sha256:de5b048a5d254f61ab3bef1ff23f1484b07808c86a1679bc0de4ee58e8c8d7c5"
)
EXPECTED_CONTROL_PLAN_SHA256 = (
    "sha256:fa6f6e04f2720b61ef4600c27c17073d429a0d6701765129fde995bddc5edd60"
)
EXPECTED_COMMON_SOURCE_MANIFEST_SHA256 = (
    "sha256:e1712fd7f5449cee23ab3101971cf57b0b3b672f82ca8f0c99bea0821aeb9cd8"
)
EXPECTED_ACTUAL_COMPARISON_MANIFEST_SHA256 = (
    "sha256:029be62e5d8a342761c979f53378fc4253877846f6f6b5ef1214d5fde0abbc27"
)
EXPECTED_ACTUAL_EVENT_SEQUENCE_SHA256 = (
    "sha256:67a95f7795a93ca77ed4105fae491f5eb9fb2f8e10cfb4af2577c9ed5fefa43e"
)
EXPECTED_CONTROL_EVENT_SEQUENCE_SHA256 = (
    "sha256:1f9dc52b9903670b792d783a427eb78c00a3c88af039d345825f2782ee6c9694"
)
MAX_RESIDUAL_CUDA_BYTES = 16 * 1024 * 1024

ACTUAL_RUN_PLAN = (
    (BF16_PATH, 1, "bf16-attached-boundary-r1"),
    (FP32_PATH, 1, "fp32-attached-boundary-r1"),
    (FP32_PATH, 2, "fp32-attached-boundary-r2"),
    (BF16_PATH, 2, "bf16-attached-boundary-r2"),
)
CONTROL_RUN_PLAN = (
    (BF16_PATH, 1, "bf16-rmsnorm-control-r1"),
    (FP32_PATH, 1, "fp32-rmsnorm-control-r1"),
    (FP32_PATH, 2, "fp32-rmsnorm-control-r2"),
    (BF16_PATH, 2, "bf16-rmsnorm-control-r2"),
)

EMBEDDING_MODULE = "model.embed_tokens"
EMBEDDING_MODULE_TYPE = "torch.nn.modules.sparse.Embedding"
ACTUAL_COMPARISON_NAMES = (
    EMBEDDING_MODULE,
    f"{BOUNDARY_MODULE}.input",
    f"{BOUNDARY_MODULE}.weight",
    BOUNDARY_MODULE,
)
CONTROL_COMPARISON_NAMES = ACTUAL_COMPARISON_NAMES[1:]

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TOP_LEVEL_KEYS = frozenset(
    {
        "attached_dtype_boundary_control_version",
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
        "control_plan",
        "control_plan_sha256",
        "common_source",
        "frozen_path_references",
        "actual_runs",
        "actual_path_repeat_stability",
        "actual_path_reproduction",
        "actual_capture_repeat_stability",
        "actual_comparisons",
        "actual_comparison_manifest_sha256",
        "actual_paired_comparison_repeat",
        "frozen_boundary_comparison",
        "control_runs",
        "control_capture_repeat_stability",
        "control_comparisons",
        "control_comparison_manifest_sha256",
        "control_paired_comparison_repeat",
        "classification_runs",
        "boundary_analysis",
        "classification",
        "causal_scope",
        "boundary_control_gate",
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
_ACTUAL_RUN_KEYS = frozenset(
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
        "capture_record_count",
        "capture_event_sequence_sha256",
        "capture_manifest_sha256",
        "control_plan_sha256",
        "capture_plan_passed",
        "path_protocol_passed",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
)
_CONTROL_RUN_KEYS = frozenset(
    {
        "run_id",
        "path",
        "repeat",
        "order_index",
        "control_id",
        "fresh_standalone_module",
        "module_name",
        "module_type",
        "forward_source_sha256",
        "variance_epsilon",
        "training",
        "autocast_enabled",
        "tf32_enabled",
        "cache_arguments_present",
        "output_injected",
        "source_manifest_sha256",
        "capture_records",
        "capture_record_count",
        "capture_event_sequence_sha256",
        "capture_manifest_sha256",
        "control_plan_sha256",
        "control_plan_passed",
        "common_source_roundtrip_exact",
        "control_weight_unchanged",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
)
_RECORD_KEYS = frozenset(
    {
        "record_id",
        "event_index",
        "capture_scope",
        "semantic_role",
        "module_name",
        "module_type",
        "occurrence_index",
        "call_index",
        "generation_step_index",
        "io_kind",
        "tensor_path",
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
        "control_plan_sha256",
        "record_sha256",
    }
)
_COMMON_SOURCE_KEYS = frozenset(
    {
        "checkpoint_weight_file_sha256",
        "embedding_tensor_name",
        "embedding_row_index",
        "rmsnorm_weight_tensor_name",
        "storage_dtype",
        "records",
        "record_count",
        "manifest_sha256",
    }
)


def validate_attached_dtype_boundary_control_evidence(
    data: object,
    *,
    source_numerics: Mapping[str, Any],
    expected_source_numerics_sha256: str,
    expected_source_lineage: Mapping[str, str],
    expected_adapter_files: list[dict[str, Any]],
    expected_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the frozen boundary-control artifact without ML imports."""

    evidence = _mapping(data, "$", _TOP_LEVEL_KEYS, "INVALID_EVIDENCE_SCHEMA")
    _finite_tree(evidence, "$")
    _validate_source_contract(
        evidence,
        source_numerics=source_numerics,
        expected_source_numerics_sha256=expected_source_numerics_sha256,
        expected_source_lineage=expected_source_lineage,
        expected_adapter_files=expected_adapter_files,
        expected_environment=expected_environment,
    )
    return _validate_body(evidence, source_numerics)


def _validate_source_contract(
    evidence: Mapping[str, Any],
    *,
    source_numerics: Mapping[str, Any],
    expected_source_numerics_sha256: str,
    expected_source_lineage: Mapping[str, str],
    expected_adapter_files: list[dict[str, Any]],
    expected_environment: Mapping[str, str],
) -> None:
    _digest(expected_source_numerics_sha256, "$.source_lineage")
    _require_exact(
        expected_source_numerics_sha256,
        EXPECTED_NUMERICS_EVIDENCE_SHA256,
        "SOURCE_NUMERICS_DIGEST_MISMATCH",
        "$.source_lineage.attached_dtype_numerics_evidence_sha256",
    )
    exact = {
        "attached_dtype_boundary_control_version": (
            ATTACHED_DTYPE_BOUNDARY_CONTROL_VERSION
        ),
        "experiment_id": "fc-mvp-001-attached-dtype-boundary-control-v1",
        "source_experiment_id": source_numerics["source_experiment_id"],
        "source_gate_experiment_id": source_numerics["experiment_id"],
        "source_lineage": dict(expected_source_lineage),
        "training_lock_sha256": source_numerics["training_lock_sha256"],
        "config_sha256": source_numerics["config_sha256"],
        "adapter_files": expected_adapter_files,
        "model_weight_sha256": source_numerics["model_weight_sha256"],
        "prompt_sha256": source_numerics["prompt_sha256"],
        "eval_digest": source_numerics["eval_digest"],
        "example_id": "eval-001",
        "input_token_count": EXPECTED_INPUT_TOKEN_COUNT,
        "input_token_ids_sha256": EXPECTED_INPUT_TOKEN_SHA256,
        "storage_audit": source_numerics["storage_audit"],
        "environment": dict(expected_environment),
        "frozen_path_references": source_numerics["frozen_path_references"],
    }
    if expected_source_lineage.get(
        "attached_dtype_numerics_evidence_sha256"
    ) != expected_source_numerics_sha256:
        _fail(
            "SOURCE_NUMERICS_LINEAGE_MISMATCH",
            "$.source_lineage.attached_dtype_numerics_evidence_sha256",
            repr(expected_source_lineage),
        )
    for key, expected in exact.items():
        _require_exact(
            evidence[key], expected, "SOURCE_CONTRACT_MISMATCH", f"$.{key}"
        )


def _validate_body(
    evidence: Mapping[str, Any],
    source_numerics: Mapping[str, Any],
) -> dict[str, Any]:
    control_plan = _mapping(
        evidence["control_plan"],
        "$.control_plan",
        None,
        "INVALID_CONTROL_PLAN",
    )
    plan_sha256 = _sha256(_canonical_json(control_plan))
    _require_exact(
        plan_sha256,
        EXPECTED_CONTROL_PLAN_SHA256,
        "FROZEN_CONTROL_PLAN_DRIFT",
        "$.control_plan",
    )
    _require_exact(
        evidence["control_plan_sha256"],
        plan_sha256,
        "CONTROL_PLAN_DIGEST_MISMATCH",
        "$.control_plan_sha256",
    )

    common_source = _validate_common_source(
        evidence["common_source"],
        evidence=evidence,
        control_plan=control_plan,
        control_plan_sha256=plan_sha256,
    )
    actual_runs = _validate_actual_runs(
        evidence["actual_runs"],
        source_numerics=source_numerics,
        control_plan_sha256=plan_sha256,
    )
    control_runs = _validate_control_runs(
        evidence["control_runs"],
        common_source=common_source,
        control_plan_sha256=plan_sha256,
    )

    actual_by_path = _runs_by_path(actual_runs)
    control_by_path = _runs_by_path(control_runs)
    actual_repeat = _capture_repeat_stability(actual_by_path)
    control_repeat = _capture_repeat_stability(control_by_path)
    _require_exact(
        evidence["actual_capture_repeat_stability"],
        actual_repeat,
        "ACTUAL_CAPTURE_REPEAT_MISMATCH",
        "$.actual_capture_repeat_stability",
    )
    _require_exact(
        evidence["control_capture_repeat_stability"],
        control_repeat,
        "CONTROL_CAPTURE_REPEAT_MISMATCH",
        "$.control_capture_repeat_stability",
    )

    _validate_actual_path_summaries(
        evidence,
        actual_by_path=actual_by_path,
        source_numerics=source_numerics,
    )
    actual_comparisons = _validate_comparisons(
        evidence["actual_comparisons"],
        names=ACTUAL_COMPARISON_NAMES,
        representative_runs=actual_by_path,
        expected_manifest=evidence["actual_comparison_manifest_sha256"],
        path="$.actual_comparisons",
    )
    _require_exact(
        evidence["actual_comparison_manifest_sha256"],
        EXPECTED_ACTUAL_COMPARISON_MANIFEST_SHA256,
        "FROZEN_ACTUAL_COMPARISON_MANIFEST_DRIFT",
        "$.actual_comparison_manifest_sha256",
    )
    _require_exact(
        actual_comparisons[0],
        source_numerics["module_comparisons"][0],
        "ACTUAL_EMBEDDING_SOURCE_MISMATCH",
        "$.actual_comparisons[0]",
    )
    expected_actual_input_comparison = dict(actual_comparisons[0])
    expected_actual_input_comparison["name"] = ACTUAL_COMPARISON_NAMES[1]
    _require_exact(
        actual_comparisons[1],
        expected_actual_input_comparison,
        "ACTUAL_INPUT_COMPARISON_MISMATCH",
        "$.actual_comparisons[1]",
    )
    control_comparisons = _validate_comparisons(
        evidence["control_comparisons"],
        names=CONTROL_COMPARISON_NAMES,
        representative_runs=control_by_path,
        expected_manifest=evidence["control_comparison_manifest_sha256"],
        path="$.control_comparisons",
    )
    _require_exact(
        control_comparisons[:2],
        actual_comparisons[1:3],
        "CONTROL_SOURCE_COMPARISON_MISMATCH",
        "$.control_comparisons[:2]",
    )
    frozen_boundary = source_numerics["module_comparisons"][1]
    _require_exact(
        evidence["frozen_boundary_comparison"],
        frozen_boundary,
        "FROZEN_BOUNDARY_SOURCE_MISMATCH",
        "$.frozen_boundary_comparison",
    )
    _require_exact(
        actual_comparisons[-1],
        frozen_boundary,
        "ACTUAL_BOUNDARY_SOURCE_MISMATCH",
        "$.actual_comparisons[3]",
    )
    _validate_paired_repeat(
        evidence["actual_paired_comparison_repeat"],
        representative={
            BF16_PATH: ACTUAL_RUN_PLAN[0][2],
            FP32_PATH: ACTUAL_RUN_PLAN[1][2],
        },
        repeat={
            BF16_PATH: ACTUAL_RUN_PLAN[3][2],
            FP32_PATH: ACTUAL_RUN_PLAN[2][2],
        },
        manifest=evidence["actual_comparison_manifest_sha256"],
        path="$.actual_paired_comparison_repeat",
    )
    _validate_paired_repeat(
        evidence["control_paired_comparison_repeat"],
        representative={
            BF16_PATH: CONTROL_RUN_PLAN[0][2],
            FP32_PATH: CONTROL_RUN_PLAN[1][2],
        },
        repeat={
            BF16_PATH: CONTROL_RUN_PLAN[3][2],
            FP32_PATH: CONTROL_RUN_PLAN[2][2],
        },
        manifest=evidence["control_comparison_manifest_sha256"],
        path="$.control_paired_comparison_repeat",
    )

    _validate_source_value_links(
        control_plan=control_plan,
        common_source=common_source,
        actual_runs=actual_runs,
        control_runs=control_runs,
    )
    classification_runs = _classification_runs(
        actual_runs=actual_runs,
        control_runs=control_runs,
        actual_path_reproduction=evidence["actual_path_reproduction"],
        actual_boundary=actual_comparisons[-1],
        control_boundary=control_comparisons[-1],
        control_plan=control_plan,
    )
    _require_exact(
        evidence["classification_runs"],
        classification_runs,
        "CLASSIFICATION_RUNS_MISMATCH",
        "$.classification_runs",
    )
    analysis = classify_attached_dtype_boundary_control(
        control_plan=control_plan,
        runs=classification_runs,
        frozen_boundary_comparison=frozen_boundary,
        actual_boundary_comparison=actual_comparisons[-1],
        control_boundary_comparison=control_comparisons[-1],
        source_evidence_locked=True,
        target_forward_identity_preserved=True,
        attached_execution_form_fixed=True,
        checkpoint_sources_unchanged=True,
    )
    _require_exact(
        evidence["boundary_analysis"],
        analysis,
        "BOUNDARY_ANALYSIS_MISMATCH",
        "$.boundary_analysis",
    )
    _require_exact(
        evidence["classification"],
        analysis["classification"],
        "CLASSIFICATION_MISMATCH",
        "$.classification",
    )
    _validate_native_output_links(
        actual_runs=actual_runs,
        control_runs=control_runs,
        analysis=analysis,
    )
    _validate_policy(
        evidence,
        source_numerics=source_numerics,
        analysis=analysis,
        actual_runs=actual_runs,
        control_runs=control_runs,
        actual_repeat=actual_repeat,
        control_repeat=control_repeat,
    )
    return {
        "frozen_gate_valid": True,
        "actual_runs_validated": len(actual_runs),
        "control_runs_validated": len(control_runs),
        "capture_records_validated": sum(
            len(run["capture_records"])
            for run in [*actual_runs, *control_runs]
        ),
        "actual_comparisons_validated": len(actual_comparisons),
        "control_comparisons_validated": len(control_comparisons),
        "protocol_completed": analysis["protocol_completed"],
        "current_forward_boundary_sufficiency_observed": analysis[
            "current_forward_boundary_sufficiency_observed"
        ],
        "classification": analysis["classification"],
    }


def _validate_common_source(
    value: object,
    *,
    evidence: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    control_plan_sha256: str,
) -> Mapping[str, Any]:
    source = _mapping(
        value,
        "$.common_source",
        _COMMON_SOURCE_KEYS,
        "INVALID_COMMON_SOURCE_SCHEMA",
    )
    exact = {
        "checkpoint_weight_file_sha256": evidence["model_weight_sha256"],
        "embedding_tensor_name": "model.embed_tokens.weight",
        "embedding_row_index": TARGET_INPUT_TOKEN_ID,
        "rmsnorm_weight_tensor_name": (
            "model.layers.0.input_layernorm.weight"
        ),
        "storage_dtype": "bfloat16",
        "record_count": 2,
    }
    for key, expected in exact.items():
        _require_exact(
            source[key], expected, "COMMON_SOURCE_CONTRACT_MISMATCH", f"$.common_source.{key}"
        )
    contracts = (
        {
            "semantic_role": "rmsnorm_input",
            "module_name": EMBEDDING_MODULE,
            "module_type": EMBEDDING_MODULE_TYPE,
            "io_kind": "source",
            "tensor_path": "checkpoint.model.embed_tokens.weight[788]",
            "shape": [1, 1, 1536],
        },
        {
            "semantic_role": "rmsnorm_weight",
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "io_kind": "parameter_source",
            "tensor_path": (
                "checkpoint.model.layers.0.input_layernorm.weight"
            ),
            "shape": [1536],
        },
    )
    records = _validate_records(
        source["records"],
        run_path="$.common_source",
        capture_scope="checkpoint_source",
        native_dtype="bfloat16",
        contracts=contracts,
        call_index=None,
        generation_step_index=None,
        control_plan_sha256=control_plan_sha256,
    )
    _require_exact(
        source["manifest_sha256"],
        _sha256(_canonical_json(records)),
        "COMMON_SOURCE_MANIFEST_MISMATCH",
        "$.common_source.manifest_sha256",
    )
    _require_exact(
        source["manifest_sha256"],
        EXPECTED_COMMON_SOURCE_MANIFEST_SHA256,
        "FROZEN_COMMON_SOURCE_MANIFEST_DRIFT",
        "$.common_source.manifest_sha256",
    )
    _require_exact(
        records[0]["canonical_float32_sha256"],
        control_plan["common_input_float32_sha256"],
        "CONTROL_PLAN_COMMON_INPUT_LINK_MISMATCH",
        "$.control_plan.common_input_float32_sha256",
    )
    _require_exact(
        records[1]["canonical_float32_sha256"],
        control_plan["common_weight_float32_sha256"],
        "CONTROL_PLAN_COMMON_WEIGHT_LINK_MISMATCH",
        "$.control_plan.common_weight_float32_sha256",
    )
    return source


def _validate_actual_runs(
    value: object,
    *,
    source_numerics: Mapping[str, Any],
    control_plan_sha256: str,
) -> list[Mapping[str, Any]]:
    values = _list(value, "$.actual_runs")
    if len(values) != len(ACTUAL_RUN_PLAN):
        _fail("INVALID_ACTUAL_RUN_COUNT", "$.actual_runs", repr(len(values)))
    source_by_path = {
        path: next(
            run for run in source_numerics["runs"] if run["path"] == path
        )
        for path in PATH_ORDER
    }
    result: list[Mapping[str, Any]] = []
    for index, (path_name, repeat, run_id) in enumerate(ACTUAL_RUN_PLAN):
        path = f"$.actual_runs[{index}]"
        run = _mapping(
            values[index], path, _ACTUAL_RUN_KEYS, "INVALID_ACTUAL_RUN_SCHEMA"
        )
        native_dtype = "bfloat16" if path_name == BF16_PATH else "float32"
        exact = {
            "run_id": run_id,
            "path": path_name,
            "repeat": repeat,
            "order_index": index,
            "fresh_load": True,
            "materialization_form": "attached_factorized_lora",
            "base_load_dtype": native_dtype,
            "token_count": 48,
            "precision_audit": source_by_path[path_name]["precision_audit"],
            "generation_trace": {
                key: source_by_path[path_name]["generation_trace"][key]
                for key in (
                    "step_count",
                    "vocabulary_size",
                    "cache_returned",
                    "scores",
                    "raw_logits",
                )
            },
            "target_alignment": {
                key: source_by_path[path_name]["target_alignment"][key]
                for key in (
                    "call_index",
                    "generation_step_index",
                    "input_token_ids",
                    "input_shape",
                    "cache_position",
                    "position_ids",
                    "past_length",
                    "causal_forward_calls",
                )
            },
            "target_alignment_passed": True,
            "capture_record_count": 4,
            "control_plan_sha256": control_plan_sha256,
            "capture_plan_passed": True,
            "path_protocol_passed": True,
        }
        for key, expected in exact.items():
            _require_exact(
                run[key], expected, "ACTUAL_RUN_CONTRACT_MISMATCH", f"{path}.{key}"
            )
        tokens = _tokens(run["generated_token_ids"], f"{path}.generated_token_ids")
        reference = source_numerics["frozen_path_references"][path_name]
        if (
            tokens != reference["generated_token_ids"]
            or token_ids_sha256(tokens) != run["token_ids_sha256"]
            or run["token_ids_sha256"] != reference["token_ids_sha256"]
            or run["output_sha256"] != reference["output_sha256"]
            or tokens[TARGET_STEP_INDEX] != reference["boundary_token_id"]
        ):
            _fail("ACTUAL_FROZEN_PATH_MISMATCH", path, run_id)
        contracts = (
            {
                "semantic_role": "embedding_output",
                "module_name": EMBEDDING_MODULE,
                "module_type": EMBEDDING_MODULE_TYPE,
                "io_kind": "output",
                "tensor_path": "output",
                "shape": [1, 1, 1536],
            },
            {
                "semantic_role": "rmsnorm_input",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "input",
                "tensor_path": "args[0]",
                "shape": [1, 1, 1536],
            },
            {
                "semantic_role": "rmsnorm_weight",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "parameter",
                "tensor_path": "module.weight",
                "shape": [1536],
            },
            {
                "semantic_role": "rmsnorm_output",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "output",
                "tensor_path": "output",
                "shape": [1, 1, 1536],
            },
        )
        records = _validate_records(
            run["capture_records"],
            run_path=path,
            capture_scope="actual_target_forward",
            native_dtype=native_dtype,
            contracts=contracts,
            call_index=TARGET_STEP_INDEX,
            generation_step_index=TARGET_STEP_INDEX,
            control_plan_sha256=control_plan_sha256,
        )
        _validate_record_manifests(run, records, path)
        _require_exact(
            run["capture_event_sequence_sha256"],
            EXPECTED_ACTUAL_EVENT_SEQUENCE_SHA256,
            "FROZEN_ACTUAL_EVENT_SEQUENCE_DRIFT",
            f"{path}.capture_event_sequence_sha256",
        )
        _validate_resources(run, path)
        result.append(run)
    return result


def _validate_control_runs(
    value: object,
    *,
    common_source: Mapping[str, Any],
    control_plan_sha256: str,
) -> list[Mapping[str, Any]]:
    values = _list(value, "$.control_runs")
    if len(values) != len(CONTROL_RUN_PLAN):
        _fail("INVALID_CONTROL_RUN_COUNT", "$.control_runs", repr(len(values)))
    result: list[Mapping[str, Any]] = []
    for index, (path_name, repeat, run_id) in enumerate(CONTROL_RUN_PLAN):
        path = f"$.control_runs[{index}]"
        run = _mapping(
            values[index], path, _CONTROL_RUN_KEYS, "INVALID_CONTROL_RUN_SCHEMA"
        )
        native_dtype = "bfloat16" if path_name == BF16_PATH else "float32"
        exact = {
            "run_id": run_id,
            "path": path_name,
            "repeat": repeat,
            "order_index": index,
            "control_id": CONTROL_ID,
            "fresh_standalone_module": True,
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "forward_source_sha256": EXPECTED_FORWARD_SOURCE_SHA256,
            "variance_epsilon": 1e-6,
            "training": False,
            "autocast_enabled": False,
            "tf32_enabled": False,
            "cache_arguments_present": False,
            "output_injected": False,
            "source_manifest_sha256": common_source["manifest_sha256"],
            "capture_record_count": 3,
            "control_plan_sha256": control_plan_sha256,
            "control_plan_passed": True,
            "common_source_roundtrip_exact": True,
            "control_weight_unchanged": True,
        }
        for key, expected in exact.items():
            _require_exact(
                run[key], expected, "CONTROL_RUN_CONTRACT_MISMATCH", f"{path}.{key}"
            )
        contracts = (
            {
                "semantic_role": "rmsnorm_input",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "input",
                "tensor_path": "control.input",
                "shape": [1, 1, 1536],
            },
            {
                "semantic_role": "rmsnorm_weight",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "parameter",
                "tensor_path": "module.weight",
                "shape": [1536],
            },
            {
                "semantic_role": "rmsnorm_output",
                "module_name": BOUNDARY_MODULE,
                "module_type": BOUNDARY_MODULE_TYPE,
                "io_kind": "output",
                "tensor_path": "control.output",
                "shape": [1, 1, 1536],
            },
        )
        records = _validate_records(
            run["capture_records"],
            run_path=path,
            capture_scope="standalone_control",
            native_dtype=native_dtype,
            contracts=contracts,
            call_index=None,
            generation_step_index=None,
            control_plan_sha256=control_plan_sha256,
        )
        _validate_record_manifests(run, records, path)
        _require_exact(
            run["capture_event_sequence_sha256"],
            EXPECTED_CONTROL_EVENT_SEQUENCE_SHA256,
            "FROZEN_CONTROL_EVENT_SEQUENCE_DRIFT",
            f"{path}.capture_event_sequence_sha256",
        )
        _validate_resources(run, path)
        result.append(run)
    return result


def _validate_records(
    value: object,
    *,
    run_path: str,
    capture_scope: str,
    native_dtype: str,
    contracts: tuple[dict[str, Any], ...],
    call_index: int | None,
    generation_step_index: int | None,
    control_plan_sha256: str,
) -> list[Mapping[str, Any]]:
    values = _list(value, f"{run_path}.capture_records")
    if len(values) != len(contracts):
        _fail("INVALID_CAPTURE_RECORD_COUNT", run_path, repr(len(values)))
    result: list[Mapping[str, Any]] = []
    for index, contract in enumerate(contracts):
        path = f"{run_path}.capture_records[{index}]"
        record = _mapping(
            values[index], path, _RECORD_KEYS, "INVALID_CAPTURE_RECORD_SCHEMA"
        )
        shape = contract["shape"]
        elements = math.prod(shape)
        exact = {
            "record_id": f"{capture_scope}|{contract['semantic_role']}",
            "event_index": index,
            "capture_scope": capture_scope,
            "semantic_role": contract["semantic_role"],
            "module_name": contract["module_name"],
            "module_type": contract["module_type"],
            "occurrence_index": 0,
            "call_index": call_index,
            "generation_step_index": generation_step_index,
            "io_kind": contract["io_kind"],
            "tensor_path": contract["tensor_path"],
            "native_dtype": native_dtype,
            "native_shape": shape,
            "native_stride": _contiguous_stride(shape),
            "native_layout": "strided",
            "comparison_dtype": "float32",
            "elements": elements,
            "finite_elements": elements,
            "all_finite": True,
            "signed_zero_normalized": True,
            "control_plan_sha256": control_plan_sha256,
        }
        for key, expected in exact.items():
            _require_exact(
                record[key], expected, "CAPTURE_RECORD_CONTRACT_MISMATCH", f"{path}.{key}"
            )
        for key in (
            "native_payload_sha256",
            "canonical_float32_sha256",
            "record_sha256",
        ):
            _digest(record[key], f"{path}.{key}")
        payload = dict(record)
        record_sha256 = payload.pop("record_sha256")
        _require_exact(
            record_sha256,
            _sha256(_canonical_json(payload)),
            "CAPTURE_RECORD_DIGEST_MISMATCH",
            f"{path}.record_sha256",
        )
        result.append(record)
    return result


_EVENT_LINK_KEYS = (
    "event_index",
    "capture_scope",
    "semantic_role",
    "module_name",
    "module_type",
    "occurrence_index",
    "call_index",
    "generation_step_index",
    "io_kind",
    "tensor_path",
)


def _validate_record_manifests(
    run: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    path: str,
) -> None:
    events = [{key: record[key] for key in _EVENT_LINK_KEYS} for record in records]
    _require_exact(
        run["capture_event_sequence_sha256"],
        _sha256(_canonical_json(events)),
        "CAPTURE_EVENT_SEQUENCE_MISMATCH",
        f"{path}.capture_event_sequence_sha256",
    )
    _require_exact(
        run["capture_manifest_sha256"],
        _sha256(_canonical_json(records)),
        "CAPTURE_MANIFEST_MISMATCH",
        f"{path}.capture_manifest_sha256",
    )


def _validate_resources(run: Mapping[str, Any], path: str) -> None:
    for key in ("elapsed_seconds", "peak_gpu_memory_bytes"):
        if not _positive_finite(run[key]):
            _fail("INVALID_RUN_RESOURCE", f"{path}.{key}", repr(run[key]))
    for key in (
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    ):
        value = run[key]
        if not _nonnegative_int(value) or value > MAX_RESIDUAL_CUDA_BYTES:
            _fail("INVALID_RUN_MEMORY_ISOLATION", f"{path}.{key}", repr(value))


def _tokens(value: object, path: str) -> list[int]:
    values = _list(value, path)
    result: list[int] = []
    for index, token in enumerate(values):
        if not _nonnegative_int(token):
            _fail("INVALID_TOKEN_IDS", f"{path}[{index}]", repr(token))
        result.append(token)
    return result


def _runs_by_path(
    runs: list[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result = {path: [run for run in runs if run["path"] == path] for path in PATH_ORDER}
    if any(len(result[path]) != 2 for path in PATH_ORDER):
        _fail("MISSING_PATH_REPEAT", "$.runs", repr(result))
    return result


def _capture_repeat_stability(
    by_path: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        left = first["capture_records"]
        right = second["capture_records"]
        value = {
            "capture_record_count": len(left),
            "capture_manifest_identity": (
                first["capture_manifest_sha256"]
                == second["capture_manifest_sha256"]
            ),
            "capture_record_identity": left == right,
            "native_payload_digest_identity": all(
                a["native_payload_sha256"] == b["native_payload_sha256"]
                for a, b in zip(left, right, strict=True)
            ),
            "canonical_float32_digest_identity": all(
                a["canonical_float32_sha256"]
                == b["canonical_float32_sha256"]
                for a, b in zip(left, right, strict=True)
            ),
            "capture_event_sequence_identity": (
                first["capture_event_sequence_sha256"]
                == second["capture_event_sequence_sha256"]
            ),
        }
        value["passed"] = all(
            item
            for key, item in value.items()
            if key not in {"capture_record_count", "passed"}
        )
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _validate_actual_path_summaries(
    evidence: Mapping[str, Any],
    *,
    actual_by_path: Mapping[str, list[Mapping[str, Any]]],
    source_numerics: Mapping[str, Any],
) -> None:
    repeat: dict[str, Any] = {}
    reproduction: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = actual_by_path[path]
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
        value["target_alignment_identity"] = (
            first["target_alignment"] == second["target_alignment"]
        )
        value["capture_manifest_identity"] = (
            first["capture_manifest_sha256"] == second["capture_manifest_sha256"]
        )
        value["passed"] = all(item for key, item in value.items() if key != "passed")
        repeat[path] = value

        reference = source_numerics["frozen_path_references"][path]
        path_runs = actual_by_path[path]
        reproduced = {
            "token_identity": all(
                run["generated_token_ids"] == reference["generated_token_ids"]
                and run["token_count"] == reference["token_count"]
                and run["token_ids_sha256"] == reference["token_ids_sha256"]
                for run in path_runs
            ),
            "output_identity": all(
                run["output_sha256"] == reference["output_sha256"] for run in path_runs
            ),
            "score_trace_identity": all(
                run["generation_trace"]["scores"]["trace_sha256"]
                == reference["score_trace_sha256"]
                for run in path_runs
            ),
            "raw_logit_trace_identity": all(
                run["generation_trace"]["raw_logits"]["trace_sha256"]
                == reference["raw_logit_trace_sha256"]
                for run in path_runs
            ),
            "comparison_score_vector_identity": all(
                run["generation_trace"]["scores"]["comparison_step_vector_sha256"]
                == reference["comparison_score_vector_sha256"]
                for run in path_runs
            ),
            "comparison_raw_logit_vector_identity": all(
                run["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"]
                == reference["comparison_raw_logit_vector_sha256"]
                for run in path_runs
            ),
            "boundary_token_identity": all(
                run["generated_token_ids"][TARGET_STEP_INDEX]
                == reference["boundary_token_id"]
                for run in path_runs
            ),
        }
        reproduced["passed"] = all(reproduced.values())
        reproduction[path] = reproduced
    repeat["passed"] = all(repeat[path]["passed"] for path in PATH_ORDER)
    reproduction["passed"] = all(
        reproduction[path]["passed"] for path in PATH_ORDER
    )
    _require_exact(
        evidence["actual_path_repeat_stability"],
        repeat,
        "ACTUAL_PATH_REPEAT_MISMATCH",
        "$.actual_path_repeat_stability",
    )
    _require_exact(
        evidence["actual_path_reproduction"],
        reproduction,
        "ACTUAL_PATH_REPRODUCTION_MISMATCH",
        "$.actual_path_reproduction",
    )


def _records_by_role(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = run.get("capture_records", run.get("records"))
    if not isinstance(records, list):
        _fail("CAPTURE_RECORDS_MISSING", "$.records", repr(type(records)))
    return {record["semantic_role"]: record for record in records}


def _validate_comparisons(
    value: object,
    *,
    names: tuple[str, ...],
    representative_runs: Mapping[str, list[Mapping[str, Any]]],
    expected_manifest: object,
    path: str,
) -> list[Mapping[str, Any]]:
    comparisons = _list(value, path)
    if len(comparisons) != len(names):
        _fail("INVALID_COMPARISON_COUNT", path, repr(len(comparisons)))
    _digest(expected_manifest, f"{path}.manifest")
    _require_exact(
        expected_manifest,
        _sha256(_canonical_json(comparisons)),
        "COMPARISON_MANIFEST_MISMATCH",
        path,
    )
    endpoint_records = {
        dtype_path: _records_by_role(representative_runs[dtype_path][0])
        for dtype_path in PATH_ORDER
    }
    role_by_name = {
        EMBEDDING_MODULE: "embedding_output",
        f"{BOUNDARY_MODULE}.input": "rmsnorm_input",
        f"{BOUNDARY_MODULE}.weight": "rmsnorm_weight",
        BOUNDARY_MODULE: "rmsnorm_output",
    }
    result: list[Mapping[str, Any]] = []
    boundary_keys: set[str] | None = None
    for index, name in enumerate(names):
        item_path = f"{path}[{index}]"
        item = _mapping(comparisons[index], item_path, None, "INVALID_COMPARISON")
        if boundary_keys is None:
            boundary_keys = set(item)
        elif set(item) != boundary_keys:
            _fail("COMPARISON_FIELD_DRIFT", item_path, repr(sorted(item)))
        _require_exact(item.get("name"), name, "COMPARISON_ORDER_MISMATCH", f"{item_path}.name")
        role = role_by_name[name]
        bf16_record = endpoint_records[BF16_PATH][role]
        fp32_record = endpoint_records[FP32_PATH][role]
        expected_shape = bf16_record["native_shape"]
        exact = {
            "shape": expected_shape,
            "elements": math.prod(expected_shape),
            "bf16_native_dtype": "bfloat16",
            "fp32_native_dtype": "float32",
            "comparison_dtype": "float32",
            "bf16_float32_sha256": bf16_record["canonical_float32_sha256"],
            "fp32_float32_sha256": fp32_record["canonical_float32_sha256"],
        }
        for key, expected in exact.items():
            _require_exact(item.get(key), expected, "COMPARISON_CAPTURE_LINK_MISMATCH", f"{item_path}.{key}")
        if name != BOUNDARY_MODULE:
            _validate_equal_comparison(item, item_path)
        result.append(item)
    return result


def _validate_equal_comparison(item: Mapping[str, Any], path: str) -> None:
    required = {
        "canonical_values_equal": True,
        "different_elements": 0,
        "first_different_flat_index": None,
        "max_abs_delta_flat_index": None,
        "bf16_value_at_first_difference": None,
        "fp32_value_at_first_difference": None,
        "bf16_value_at_max_abs_delta": None,
        "fp32_value_at_max_abs_delta": None,
        "max_abs_delta": 0.0,
        "mean_abs_delta": 0.0,
        "root_mean_square_delta": 0.0,
        "sum_abs_delta": 0.0,
        "sum_squared_delta": 0.0,
        "different_fraction": 0.0,
        "normalized_root_mean_square_delta": 0.0,
        "root_mean_square_delta_ratio_to_first_registered_difference": 0.0,
    }
    for key, expected in required.items():
        _require_exact(item.get(key), expected, "NON_BOUNDARY_VALUES_DIFFER", f"{path}.{key}")
    _require_exact(
        item.get("bf16_float32_sha256"),
        item.get("fp32_float32_sha256"),
        "EQUAL_COMPARISON_DIGEST_MISMATCH",
        path,
    )
    _require_exact(
        item.get("bf16_root_mean_square"),
        item.get("fp32_root_mean_square"),
        "EQUAL_COMPARISON_RMS_MISMATCH",
        path,
    )
    for key in ("bf16_root_mean_square", "fp32_root_mean_square"):
        value = item.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            _fail("INVALID_EQUAL_COMPARISON_RMS", f"{path}.{key}", repr(value))


def _validate_paired_repeat(
    value: object,
    *,
    representative: Mapping[str, str],
    repeat: Mapping[str, str],
    manifest: object,
    path: str,
) -> None:
    expected = {
        "representative_run_ids": dict(representative),
        "repeat_run_ids": dict(repeat),
        "representative_manifest_sha256": manifest,
        "repeat_manifest_sha256": manifest,
        "exact_identity": True,
    }
    _require_exact(value, expected, "PAIRED_COMPARISON_REPEAT_MISMATCH", path)


def _validate_source_value_links(
    *,
    control_plan: Mapping[str, Any],
    common_source: Mapping[str, Any],
    actual_runs: list[Mapping[str, Any]],
    control_runs: list[Mapping[str, Any]],
) -> None:
    source_records = _records_by_role(common_source)
    for index, (actual, control) in enumerate(
        zip(actual_runs, control_runs, strict=True)
    ):
        if (
            actual["path"] != control["path"]
            or actual["repeat"] != control["repeat"]
            or actual["order_index"] != control["order_index"]
        ):
            _fail("ACTUAL_CONTROL_PAIRING_MISMATCH", "$.runs", repr(index))
        actual_records = _records_by_role(actual)
        control_records = _records_by_role(control)
        common_input = control_plan["common_input_float32_sha256"]
        common_weight = control_plan["common_weight_float32_sha256"]
        input_digests = (
            source_records["rmsnorm_input"]["canonical_float32_sha256"],
            actual_records["embedding_output"]["canonical_float32_sha256"],
            actual_records["rmsnorm_input"]["canonical_float32_sha256"],
            control_records["rmsnorm_input"]["canonical_float32_sha256"],
        )
        weight_digests = (
            source_records["rmsnorm_weight"]["canonical_float32_sha256"],
            actual_records["rmsnorm_weight"]["canonical_float32_sha256"],
            control_records["rmsnorm_weight"]["canonical_float32_sha256"],
        )
        if any(value != common_input for value in input_digests):
            _fail("COMMON_INPUT_SOURCE_LINK_MISMATCH", "$.runs", repr(index))
        if any(value != common_weight for value in weight_digests):
            _fail("COMMON_WEIGHT_SOURCE_LINK_MISMATCH", "$.runs", repr(index))
        if (
            actual_records["embedding_output"]["native_payload_sha256"]
            != actual_records["rmsnorm_input"]["native_payload_sha256"]
        ):
            _fail("EMBEDDING_INPUT_NATIVE_LINK_MISMATCH", "$.runs", repr(index))
        for role in ("rmsnorm_input", "rmsnorm_weight"):
            if (
                actual_records[role]["native_payload_sha256"]
                != control_records[role]["native_payload_sha256"]
            ):
                _fail("ACTUAL_CONTROL_NATIVE_SOURCE_MISMATCH", "$.runs", role)
        if actual["path"] == BF16_PATH:
            for role in ("rmsnorm_input", "rmsnorm_weight"):
                if (
                    source_records[role]["native_payload_sha256"]
                    != actual_records[role]["native_payload_sha256"]
                ):
                    _fail("BF16_CHECKPOINT_NATIVE_LINK_MISMATCH", "$.runs", role)


def _classification_runs(
    *,
    actual_runs: list[Mapping[str, Any]],
    control_runs: list[Mapping[str, Any]],
    actual_path_reproduction: Mapping[str, Any],
    actual_boundary: Mapping[str, Any],
    control_boundary: Mapping[str, Any],
    control_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for actual, control in zip(actual_runs, control_runs, strict=True):
        actual_records = _records_by_role(actual)
        control_records = _records_by_role(control)
        prefix = "bf16" if actual["path"] == BF16_PATH else "fp32"
        item = {
            "run_id": f"{actual['run_id']}|{control['run_id']}",
            "path": actual["path"],
            "repeat": actual["repeat"],
            "order_index": actual["order_index"],
            "fresh_load": (
                actual["fresh_load"] and control["fresh_standalone_module"]
            ),
            "frozen_path_reproduced": actual_path_reproduction[actual["path"]][
                "passed"
            ],
            "target_forward_aligned": actual["target_alignment_passed"],
            "actual_boundary_reproduced": (
                actual_records["rmsnorm_output"]["canonical_float32_sha256"]
                == actual_boundary[f"{prefix}_float32_sha256"]
            ),
            "control_id": CONTROL_ID,
            "control_executed": control["control_plan_passed"],
            "control_standalone": control["fresh_standalone_module"],
            "common_source_roundtrip_exact": (
                actual_records["rmsnorm_input"]["canonical_float32_sha256"]
                == control_plan["common_input_float32_sha256"]
                and actual_records["rmsnorm_weight"]["canonical_float32_sha256"]
                == control_plan["common_weight_float32_sha256"]
                and control["common_source_roundtrip_exact"]
            ),
            "control_weight_unchanged": control["control_weight_unchanged"],
            "autocast_enabled": control["autocast_enabled"],
            "actual_input_native_dtype": actual_records["rmsnorm_input"]["native_dtype"],
            "actual_weight_native_dtype": actual_records["rmsnorm_weight"]["native_dtype"],
            "actual_output_native_dtype": actual_records["rmsnorm_output"]["native_dtype"],
            "actual_input_float32_sha256": actual_records["rmsnorm_input"]["canonical_float32_sha256"],
            "actual_weight_float32_sha256": actual_records["rmsnorm_weight"]["canonical_float32_sha256"],
            "actual_output_float32_sha256": actual_records["rmsnorm_output"]["canonical_float32_sha256"],
            "control_input_native_dtype": control_records["rmsnorm_input"]["native_dtype"],
            "control_weight_native_dtype": control_records["rmsnorm_weight"]["native_dtype"],
            "control_output_native_dtype": control_records["rmsnorm_output"]["native_dtype"],
            "control_input_float32_sha256": control_records["rmsnorm_input"]["canonical_float32_sha256"],
            "control_weight_float32_sha256": control_records["rmsnorm_weight"]["canonical_float32_sha256"],
            "control_output_float32_sha256": control_records["rmsnorm_output"]["canonical_float32_sha256"],
            "control_cache_arguments_present": control["cache_arguments_present"],
            "control_output_injected": control["output_injected"],
            "module_tensor_payload_saved": False,
            "module_tensor_sidecar_saved": False,
        }
        _require_exact(
            item["control_output_float32_sha256"],
            control_boundary[f"{prefix}_float32_sha256"],
            "CONTROL_OUTPUT_ENDPOINT_LINK_MISMATCH",
            "$.classification_runs",
        )
        result.append(item)
    return result


def _validate_native_output_links(
    *,
    actual_runs: list[Mapping[str, Any]],
    control_runs: list[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> None:
    if analysis["actual_control_output_exact"] is not True:
        return
    for index, (actual, control) in enumerate(
        zip(actual_runs, control_runs, strict=True)
    ):
        actual_output = _records_by_role(actual)["rmsnorm_output"]
        control_output = _records_by_role(control)["rmsnorm_output"]
        if (
            actual_output["native_dtype"] != control_output["native_dtype"]
            or actual_output["native_shape"] != control_output["native_shape"]
            or actual_output["native_payload_sha256"]
            != control_output["native_payload_sha256"]
            or actual_output["canonical_float32_sha256"]
            != control_output["canonical_float32_sha256"]
        ):
            _fail("ACTUAL_CONTROL_NATIVE_OUTPUT_LINK_MISMATCH", "$.runs", repr(index))


def _expected_protocol(source_numerics: Mapping[str, Any]) -> dict[str, Any]:
    upstream = source_numerics["protocol"]
    return {
        "freshness_scope": (
            "four_fresh_actual_attached_model_loads_then_four_fresh_"
            "standalone_rmsnorm_modules_in_fixed_process"
        ),
        "actual_run_plan": [
            {"run_id": run_id, "path": path, "repeat": repeat, "order_index": index}
            for index, (path, repeat, run_id) in enumerate(ACTUAL_RUN_PLAN)
        ],
        "control_run_plan": [
            {"run_id": run_id, "path": path, "repeat": repeat, "order_index": index}
            for index, (path, repeat, run_id) in enumerate(CONTROL_RUN_PLAN)
        ],
        "run_order_design": "actual_ABBA_then_standalone_control_ABBA",
        "fresh_loads_per_path": {"actual_attached": 2, "standalone_control": 2},
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "target_forward": upstream["target_forward"],
        "actual_treatment": upstream["treatment"],
        "paths": upstream["paths"],
        "generation": upstream["generation"],
        "sdp_kernel_flags": upstream["sdp_kernel_flags"],
        "actual_boundary_capture": {
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "generation_step_index": TARGET_STEP_INDEX,
            "capture_roles": [
                "embedding_output",
                "rmsnorm_input",
                "rmsnorm_weight",
                "rmsnorm_output",
            ],
            "capture_count_per_run": 4,
            "capture_method": "target_scoped_forward_hooks_gpu_clone_then_cpu_summary",
            "actual_model_control_injection": False,
        },
        "standalone_control": {
            "control_id": CONTROL_ID,
            "intervention_count": 1,
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "forward_source_sha256": EXPECTED_FORWARD_SOURCE_SHA256,
            "variance_epsilon": 1e-6,
            "hidden_size": 1536,
            "common_input_source": "model.embed_tokens.weight[788]",
            "common_weight_source": "model.layers.0.input_layernorm.weight",
            "fresh_standalone_module_per_run": True,
            "attached_model_loaded_during_control": False,
            "autocast": False,
            "tf32": False,
            "cache_arguments": False,
            "actual_model_output_injection": False,
            "serialized_tensor_payload": False,
            "module_tensor_sidecar": False,
        },
        "comparison": {
            "dtype": "float32",
            "contiguous": True,
            "signed_zero_normalized": True,
            "finite_only": True,
            "exactness": "canonical_float32_exact_no_tolerance",
            "reduction": "fixed_flatten_order_stdlib_math_fsum_float64",
        },
    }


def _expected_causal_scope(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "isolated_variable": (
            "Qwen2RMSNorm input_weight_output dtype materialization under one "
            "same-checkpoint-values standalone control"
        ),
        "controlled": [
            "same_bfloat16_checkpoint_embedding_row_788_source_values",
            "same_bfloat16_checkpoint_layer0_input_rmsnorm_weight_source_values",
            "same_Qwen2RMSNorm_class_forward_source_and_epsilon",
            "same_actual_attached_factorized_lora_execution_form",
            "same_eval_001_target_forward_and_generation_protocol",
            "fresh_actual_and_standalone_control_ABBA_repeats",
            "disabled_autocast_and_tf32",
        ],
        "supports": (
            "whether one pre-registered same-values standalone Qwen2RMSNorm "
            "dtype replay exactly reproduces the observed actual layer-0 "
            "input_layernorm output boundary"
        ),
        "observed_current_forward_boundary_sufficiency": analysis[
            "current_forward_boundary_sufficiency_observed"
        ],
        "does_not_support": [
            "unique_pow_mean_rsqrt_cast_multiply_or_cuda_kernel_root_cause",
            "earliest_difference_across_generation_history",
            "absence_of_dtype_conditioned_kv_cache_history",
            "independent_causal_propagation_beyond_the_rmsnorm_output",
            "peft_bug_claim",
            "pristine_fp32_checkpoint_comparison",
            "full_eval_generalization",
            "artifact_promotion",
            "runtime_eligibility",
        ],
        "json_only_limitation": (
            "without module tensor payloads the offline validator checks exact "
            "digests, frozen links, repeat identity, and summary algebra but "
            "cannot independently recompute full captured tensors"
        ),
    }


def _expected_constraints(source_numerics: Mapping[str, Any]) -> dict[str, bool]:
    constraints = dict(source_numerics["constraints"])
    constraints.update(
        {
            "module_tensor_sidecar": False,
            "module_tensor_payload": False,
            "second_boundary_intervention": False,
            "actual_model_control_injection": False,
            "control_cache_usage": False,
            "control_autocast": False,
            "control_tf32": False,
            "full_eval_run": False,
            "fp32_adapter_artifact_promotion": False,
        }
    )
    return constraints


def _expected_locked_next_action(
    analysis: Mapping[str, Any],
    constraints: Mapping[str, bool],
) -> dict[str, Any]:
    matched = bool(analysis["current_forward_boundary_sufficiency_observed"])
    next_gate_constraints = dict(constraints)
    next_gate_constraints["full_eval_run"] = True
    return {
        "gate_id": "FC-MVP-001-fp32-attached-remediation-eval-v1",
        "eligible_to_start": matched,
        "action": (
            "pre-register one resource-bounded full frozen evaluation of the "
            "FP32 attached Adapter with decision compilation fixed, compare it "
            "against frozen BF16 attached v2 metrics, and keep merge, artifact "
            "promotion, and Runtime integration prohibited until safety, "
            "regression, and resource gates pass"
        ),
        "acceptance": {
            "matched_boundary_control_required": True,
            "frozen_boundary_evidence_locked": True,
            "one_fp32_attached_candidate": True,
            "unchanged_twenty_case_eval": True,
            "unchanged_decision_compiler": True,
            "frozen_bf16_attached_reference": True,
            "resource_and_safety_comparison_required": True,
            "merge_prohibited": True,
            "artifact_promotion_prohibited": True,
            "runtime_integration_prohibited": True,
        },
        "constraints": next_gate_constraints,
    }


def _validate_policy(
    evidence: Mapping[str, Any],
    *,
    source_numerics: Mapping[str, Any],
    analysis: Mapping[str, Any],
    actual_runs: list[Mapping[str, Any]],
    control_runs: list[Mapping[str, Any]],
    actual_repeat: Mapping[str, Any],
    control_repeat: Mapping[str, Any],
) -> None:
    _require_exact(
        evidence["protocol"],
        _expected_protocol(source_numerics),
        "PROTOCOL_MISMATCH",
        "$.protocol",
    )
    actual_reproduction = evidence["actual_path_reproduction"]
    actual_path_repeat = evidence["actual_path_repeat_stability"]
    fresh_memory = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        for run in [*actual_runs, *control_runs]
    )
    acceptance = {
        "upstream_numerics_evidence_locked": True,
        "frozen_input_reproduced": True,
        "actual_attached_paths_reproduced": actual_reproduction["passed"],
        "actual_boundary_capture_repeat_stable": actual_repeat["passed"],
        "frozen_boundary_reproduced": True,
        "one_control_pre_registered": True,
        "standalone_control_abba_executed": len(control_runs) == 4,
        "standalone_control_repeat_stable": control_repeat["passed"],
        "same_checkpoint_sources_used": True,
        "target_forward_identity_preserved": all(
            run["target_alignment_passed"] for run in actual_runs
        ),
        "attached_execution_form_fixed": all(
            run["materialization_form"] == "attached_factorized_lora"
            and run["path_protocol_passed"]
            for run in actual_runs
        ),
        "source_adapter_unchanged": True,
        "source_model_unchanged": True,
        "eval_digest_unchanged": True,
        "prompt_digest_unchanged": True,
        "fresh_load_memory_isolated": fresh_memory,
        "control_protocol_completed_outcome_neutrally": analysis[
            "protocol_completed"
        ],
        "module_tensor_payload_absent": True,
    }
    _require_exact(
        evidence["acceptance"], acceptance, "ACCEPTANCE_MISMATCH", "$.acceptance"
    )
    gate = {
        "actual_abba_reproduced": actual_reproduction["passed"],
        "actual_path_repeat_stable": actual_path_repeat["passed"],
        "actual_capture_repeat_stable": actual_repeat["passed"],
        "actual_boundary_reproduced": analysis["actual_boundary_reproduced"],
        "control_plan_executed": all(
            run["control_plan_passed"] for run in control_runs
        ),
        "control_abba_repeat_stable": control_repeat["passed"],
        "same_values_preconditions_passed": analysis[
            "same_values_preconditions_passed"
        ],
        "protocol_completed": analysis["protocol_completed"],
        "fresh_load_memory_isolated": fresh_memory,
        "module_tensor_payload_absent": True,
    }
    gate["passed"] = all(gate.values())
    _require_exact(
        evidence["boundary_control_gate"],
        gate,
        "BOUNDARY_CONTROL_GATE_MISMATCH",
        "$.boundary_control_gate",
    )
    constraints = _expected_constraints(source_numerics)
    exact = {
        "causal_scope": _expected_causal_scope(analysis),
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "module_tensor_payload_saved": False,
        "module_tensor_sidecar_allowed": False,
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _expected_locked_next_action(analysis, constraints),
        "runtime_eligible": False,
        "runtime_eligibility_reason": analysis["classification"],
        "offline": True,
    }
    for key, expected in exact.items():
        _require_exact(
            evidence[key], expected, "FROZEN_POLICY_MISMATCH", f"$.{key}"
        )
    if not _positive_finite(evidence["elapsed_seconds"]):
        _fail(
            "INVALID_RESOURCE_EVIDENCE",
            "$.elapsed_seconds",
            repr(evidence["elapsed_seconds"]),
        )
    peaks = [run["peak_gpu_memory_bytes"] for run in [*actual_runs, *control_runs]]
    _require_exact(
        evidence["peak_gpu_memory_bytes"],
        max(peaks),
        "PEAK_MEMORY_MISMATCH",
        "$.peak_gpu_memory_bytes",
    )


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


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        stride[index] = stride[index + 1] * shape[index + 1]
    return stride


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("INVALID_LIST", path, repr(type(value)))
    return value


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail("INVALID_SHA256", path, repr(value))
    return value


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE_VALUE", path, repr(value))
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def _require_exact(
    actual: object,
    expected: object,
    code: str,
    path: str,
) -> None:
    if not _strict_equal(actual, expected):
        _fail(code, path, repr(actual))


def _strict_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


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
