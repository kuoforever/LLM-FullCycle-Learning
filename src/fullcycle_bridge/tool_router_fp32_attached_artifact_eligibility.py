"""Offline artifact-eligibility review for the FP32 attached Tool Router."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from .consumer import canonical_json_bytes
from .tool_router import ToolRouterValidationError
from .tool_router_sft import canonical_config_sha256

REVIEW_VERSION = 1
EXPERIMENT_ID = "fc-mvp-001-fp32-attached-artifact-eligibility-review-v1"
GATE_ID = "FC-MVP-001-fp32-attached-artifact-eligibility-review-v1"
CANDIDATE_ID = "fp32-attached-factorized-lora"
NEXT_GATE_ID = "FC-MVP-001-fp32-attached-offline-package-manifest-v1"

FAVORABLE_CLASSIFICATION = (
    "fp32_attached_full_eval_improves_quality_without_safety_or_resource_regression"
)
UPSTREAM_CLASSIFICATIONS = {
    "favorable": FAVORABLE_CLASSIFICATION,
    "neutral": (
        "fp32_attached_full_eval_preserves_quality_and_safety_within_resource_budget"
    ),
}
INCOMPLETE_CLASSIFICATIONS = {
    "favorable": (
        "fp32_attached_fixed_compiler_favorable_eval_but_offline_artifact_"
        "package_incomplete"
    ),
    "neutral": (
        "fp32_attached_fixed_compiler_neutral_eval_but_offline_artifact_"
        "package_incomplete"
    ),
}
ELIGIBLE_CLASSIFICATIONS = {
    "favorable": (
        "fp32_attached_fixed_compiler_favorable_eval_offline_artifact_package_eligible"
    ),
    "neutral": (
        "fp32_attached_fixed_compiler_neutral_eval_offline_artifact_package_eligible"
    ),
}

SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
TENSOR_NAME_PATTERN = re.compile(
    r"base_model\.model\.model\.layers\.(?P<layer>[0-9]+)\.self_attn\."
    r"(?P<target>q_proj|k_proj|v_proj|o_proj)\.lora_(?P<matrix>A|B)\.weight"
)
SOURCE_HASH_KEYS = frozenset(
    {
        "adapter_config",
        "adapter_readme",
        "adapter_verifier_source",
        "adapter_weights",
        "attached_dtype_isolation",
        "canonical_json_source",
        "decision_compiler_source",
        "evaluation_fixture",
        "gitattributes",
        "inference_runner_source",
        "lifecycle_evidence",
        "load_merge_evidence",
        "model_downloader_source",
        "prompt",
        "remediation_gate",
        "remediation_predictions",
        "remediation_preregistration",
        "review_builder_source",
        "review_contract_source",
        "sft_config",
        "sft_helpers_source",
        "training_evidence",
        "training_lock",
        "training_runner_source",
        "validation_error_source",
    }
)
PACKAGE_REQUIREMENT_KEYS = frozenset(
    {
        "base_model_revision_bound",
        "composite_manifest_present",
        "portable_base_model_bound",
        "required_compiler_bound",
        "tokenizer_file_manifest_bound",
        "use_and_limitations_documented",
    }
)
PACKAGE_BLOCKERS = {
    "base_model_revision_bound": "base_model_revision_binding_missing",
    "composite_manifest_present": "composite_manifest_missing",
    "portable_base_model_bound": "portable_base_model_binding_missing",
    "required_compiler_bound": "required_compiler_binding_missing",
    "tokenizer_file_manifest_bound": "tokenizer_file_manifest_missing",
    "use_and_limitations_documented": (
        "package_use_and_limitations_documentation_incomplete"
    ),
}

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}


def inspect_adapter_safetensors(path: Path) -> dict[str, Any]:
    """Inspect the frozen Adapter header using only the safetensors format."""

    if not path.is_file() or path.is_symlink():
        _fail("UNSAFE_ADAPTER_WEIGHT_PATH", "$.adapter_weights", str(path))
    return inspect_adapter_safetensors_bytes(path.read_bytes())


def inspect_adapter_safetensors_bytes(payload: bytes) -> dict[str, Any]:
    """Inspect one immutable byte snapshot of the frozen Adapter weights."""

    if not isinstance(payload, bytes):
        _fail(
            "INVALID_SOURCE_PAYLOAD",
            "$.adapter_weights",
            type(payload).__name__,
        )
    file_bytes = len(payload)
    prefix = payload[:8]
    if len(prefix) != 8:
        _fail("INVALID_SAFETENSORS_HEADER", "$.adapter_weights", "short prefix")
    header_bytes = struct.unpack("<Q", prefix)[0]
    if (
        header_bytes <= 0
        or header_bytes > 16 * 1024 * 1024
        or header_bytes % 8 != 0
        or 8 + header_bytes >= file_bytes
    ):
        _fail(
            "INVALID_SAFETENSORS_HEADER",
            "$.adapter_weights.header_bytes",
            repr(header_bytes),
        )
    raw_header = payload[8 : 8 + header_bytes]
    if len(raw_header) != header_bytes:
        _fail("INVALID_SAFETENSORS_HEADER", "$.adapter_weights", "short header")
    try:
        header = json.loads(
            raw_header,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolRouterValidationError(
            "INVALID_SAFETENSORS_HEADER", "$.adapter_weights", str(exc)
        ) from exc
    if not isinstance(header, dict):
        _fail("INVALID_SAFETENSORS_HEADER", "$.adapter_weights", "not object")

    metadata = header.pop("__metadata__", None)
    if metadata != {"format": "pt"}:
        _fail(
            "INVALID_SAFETENSORS_METADATA",
            "$.adapter_weights.__metadata__",
            repr(metadata),
        )
    if not header:
        _fail("EMPTY_SAFETENSORS", "$.adapter_weights", "no tensors")

    ranges: list[tuple[int, int, str]] = []
    dtype_tensor_counts: Counter[str] = Counter()
    dtype_element_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    target_tensor_counts: Counter[str] = Counter()
    matrix_tensor_counts: Counter[str] = Counter()
    layers: set[int] = set()
    modules: set[str] = set()
    parameter_count = 0
    expected_tensor_names = {
        (
            f"base_model.model.model.layers.{layer}.self_attn.{target}."
            f"lora_{matrix}.weight"
        )
        for layer in range(28)
        for target in ("q_proj", "k_proj", "v_proj", "o_proj")
        for matrix in ("A", "B")
    }
    if set(header) != expected_tensor_names:
        missing = sorted(expected_tensor_names - set(header))
        extra = sorted(set(header) - expected_tensor_names)
        _fail(
            "ADAPTER_TENSOR_TOPOLOGY_MISMATCH",
            "$.adapter_weights",
            f"missing={missing[:3]!r},extra={extra[:3]!r}",
        )

    for name, raw_descriptor in header.items():
        descriptor = _mapping(raw_descriptor, f"$.adapter_weights.{name}")
        _expect_exact_keys(
            descriptor,
            {"data_offsets", "dtype", "shape"},
            f"$.adapter_weights.{name}",
        )
        match = TENSOR_NAME_PATTERN.fullmatch(name)
        if match is None:
            _fail("INVALID_ADAPTER_TENSOR_NAME", "$.adapter_weights", name)
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
            _fail(
                "INVALID_TENSOR_DTYPE", f"$.adapter_weights.{name}.dtype", repr(dtype)
            )
        if not isinstance(shape, list) or not shape:
            _fail(
                "INVALID_TENSOR_SHAPE", f"$.adapter_weights.{name}.shape", repr(shape)
            )
        dimensions = [
            _positive_int(value, f"$.adapter_weights.{name}.shape[{index}]")
            for index, value in enumerate(shape)
        ]
        expected_shape = (
            [16, 1536]
            if match.group("matrix") == "A"
            else (
                [1536, 16]
                if match.group("target") in {"q_proj", "o_proj"}
                else [256, 16]
            )
        )
        if dimensions != expected_shape:
            _fail(
                "ADAPTER_TENSOR_SHAPE_MISMATCH",
                f"$.adapter_weights.{name}.shape",
                f"expected={expected_shape!r},actual={dimensions!r}",
            )
        if not isinstance(offsets, list) or len(offsets) != 2:
            _fail(
                "INVALID_TENSOR_OFFSETS",
                f"$.adapter_weights.{name}.data_offsets",
                repr(offsets),
            )
        start = _nonnegative_int(
            offsets[0], f"$.adapter_weights.{name}.data_offsets[0]"
        )
        end = _nonnegative_int(offsets[1], f"$.adapter_weights.{name}.data_offsets[1]")
        elements = math.prod(dimensions)
        expected_bytes = elements * _SAFETENSORS_DTYPE_BYTES[dtype]
        if end <= start or end - start != expected_bytes:
            _fail(
                "INVALID_TENSOR_OFFSETS",
                f"$.adapter_weights.{name}.data_offsets",
                f"range={start}:{end},expected_bytes={expected_bytes}",
            )
        ranges.append((start, end, name))
        dtype_tensor_counts[dtype] += 1
        dtype_element_counts[dtype] += elements
        shape_counts["x".join(str(item) for item in dimensions)] += 1
        target = match.group("target")
        matrix = match.group("matrix")
        target_tensor_counts[target] += 1
        matrix_tensor_counts[matrix] += 1
        layers.add(int(match.group("layer")))
        modules.add(name.rsplit(".lora_", 1)[0])
        parameter_count += elements

    ordered_ranges = sorted(ranges)
    cursor = 0
    for start, end, name in ordered_ranges:
        if start != cursor:
            _fail(
                "NONCONTIGUOUS_TENSOR_OFFSETS",
                f"$.adapter_weights.{name}.data_offsets",
                f"expected={cursor},actual={start}",
            )
        cursor = end
    data_bytes = file_bytes - 8 - header_bytes
    if cursor != data_bytes:
        _fail(
            "SAFETENSORS_DATA_SIZE_MISMATCH",
            "$.adapter_weights",
            f"offset_end={cursor},data_bytes={data_bytes}",
        )

    return {
        "format": "safetensors",
        "metadata": {"format": "pt"},
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "data_bytes": data_bytes,
        "tensor_count": len(header),
        "module_count": len(modules),
        "layer_count": len(layers),
        "layers": sorted(layers),
        "parameter_count": parameter_count,
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "dtype_element_counts": dict(sorted(dtype_element_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "target_module_tensor_counts": dict(sorted(target_tensor_counts.items())),
        "lora_matrix_tensor_counts": dict(sorted(matrix_tensor_counts.items())),
        "data_offsets_contiguous": True,
        "topology_complete": True,
    }


def classify_package_requirements(
    *,
    upstream_outcome: str,
    upstream_gate_passed: bool,
    requirements: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the same categorical package rubric to favorable and neutral runs."""

    if upstream_outcome not in {"favorable", "neutral"}:
        _fail("INVALID_UPSTREAM_OUTCOME", "$.upstream_outcome", upstream_outcome)
    if upstream_gate_passed is not True:
        _fail(
            "UPSTREAM_GATE_NOT_PASSED",
            "$.upstream_gate_passed",
            repr(upstream_gate_passed),
        )
    if frozenset(requirements) != PACKAGE_REQUIREMENT_KEYS or any(
        type(value) is not bool for value in requirements.values()
    ):
        _fail("INVALID_PACKAGE_REQUIREMENTS", "$.requirements", repr(requirements))
    blockers = sorted(
        PACKAGE_BLOCKERS[name] for name, passed in requirements.items() if not passed
    )
    eligible = not blockers
    return {
        "requirements": dict(sorted(requirements.items())),
        "blocking_findings": blockers,
        "package_complete": eligible,
        "offline_artifact_eligible": eligible,
        "classification": (
            ELIGIBLE_CLASSIFICATIONS[upstream_outcome]
            if eligible
            else INCOMPLETE_CLASSIFICATIONS[upstream_outcome]
        ),
    }


def build_fp32_attached_artifact_eligibility_review(
    remediation_preregistration: Mapping[str, Any],
    remediation_predictions: Mapping[str, Any],
    remediation_gate: Mapping[str, Any],
    attached_dtype_isolation: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    lifecycle_evidence: Mapping[str, Any],
    load_merge_evidence: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    adapter_readme: str,
    adapter_files: Sequence[Mapping[str, Any]],
    adapter_tensor_audit: Mapping[str, Any],
    training_lock: str,
    gitattributes: str,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Build the deterministic negative-or-positive eligibility review."""

    _validate_source_hashes(source_hashes)
    actual_adapter_files = [dict(item) for item in adapter_files]
    _validate_source_payload_bindings(
        source_payloads,
        source_hashes,
        remediation_preregistration=remediation_preregistration,
        remediation_predictions=remediation_predictions,
        remediation_gate=remediation_gate,
        attached_dtype_isolation=attached_dtype_isolation,
        sft_config=sft_config,
        training_evidence=training_evidence,
        lifecycle_evidence=lifecycle_evidence,
        load_merge_evidence=load_merge_evidence,
        adapter_config=adapter_config,
        adapter_readme=adapter_readme,
        adapter_files=actual_adapter_files,
        adapter_tensor_audit=adapter_tensor_audit,
        training_lock=training_lock,
        gitattributes=gitattributes,
    )
    source_facts = _validate_source_chain(
        remediation_preregistration,
        remediation_predictions,
        remediation_gate,
        sft_config,
        training_evidence,
        lifecycle_evidence,
        load_merge_evidence,
        actual_adapter_files,
        source_hashes,
    )
    adapter_facts = _validate_adapter_contract(
        remediation_preregistration,
        remediation_predictions,
        attached_dtype_isolation,
        sft_config,
        training_evidence,
        load_merge_evidence,
        adapter_config,
        actual_adapter_files,
        adapter_tensor_audit,
    )
    dependency_pins = _validate_training_lock(training_lock, sft_config)
    lfs_pattern = (
        "baseline/adapters/**/*.safetensors filter=lfs diff=lfs merge=lfs -text"
    )
    if lfs_pattern not in gitattributes.splitlines():
        _fail("MISSING_ADAPTER_LFS_POLICY", "$.gitattributes", lfs_pattern)

    assessment = _mapping(
        remediation_gate.get("assessment"), "$.remediation.assessment"
    )
    comparison = _mapping(
        remediation_gate.get("comparison"), "$.remediation.comparison"
    )
    raw_metrics = _mapping(
        remediation_gate.get("raw_metrics"), "$.remediation.raw_metrics"
    )
    compiled_metrics = _mapping(
        remediation_gate.get("compiled_metrics"), "$.remediation.compiled_metrics"
    )
    compilation = _mapping(
        remediation_gate.get("compilation"), "$.remediation.compilation"
    )
    reference_metrics = _mapping(
        comparison.get("reference_metrics"),
        "$.remediation.comparison.reference_metrics",
    )
    resource_comparison = _mapping(
        comparison.get("resource_comparison"),
        "$.remediation.comparison.resource_comparison",
    )
    peak = _mapping(
        resource_comparison.get("peak_gpu_memory_bytes"),
        "$.remediation.comparison.resource_comparison.peak_gpu_memory_bytes",
    )
    elapsed = _mapping(
        resource_comparison.get("elapsed_seconds"),
        "$.remediation.comparison.resource_comparison.elapsed_seconds",
    )
    strict_improvements = _strict_improvements(comparison)
    if assessment.get("outcome") == "favorable" and not strict_improvements:
        _fail("MISSING_FAVORABLE_IMPROVEMENT", "$.remediation.comparison", "none")
    if comparison.get("regression_event_count") != 0:
        _fail(
            "UPSTREAM_REGRESSION_PRESENT",
            "$.remediation.comparison.regression_event_count",
            repr(comparison.get("regression_event_count")),
        )

    compiler_lineage = _mapping(
        _mapping(
            remediation_preregistration.get("source_lineage"),
            "$.preregistration.source_lineage",
        ).get("decision_compiler_source"),
        "$.preregistration.source_lineage.decision_compiler_source",
    )
    model = copy.deepcopy(
        _mapping(
            _mapping(
                remediation_preregistration.get("frozen_inputs"),
                "$.preregistration.frozen_inputs",
            ).get("model"),
            "$.preregistration.frozen_inputs.model",
        )
    )
    tokenizer = copy.deepcopy(
        _mapping(
            remediation_preregistration["frozen_inputs"].get("tokenizer"),
            "$.preregistration.frozen_inputs.tokenizer",
        )
    )
    prompt = copy.deepcopy(
        _mapping(
            remediation_preregistration["frozen_inputs"].get("prompt"),
            "$.preregistration.frozen_inputs.prompt",
        )
    )
    generation = copy.deepcopy(
        _mapping(
            _mapping(
                remediation_preregistration.get("protocol"),
                "$.preregistration.protocol",
            ).get("generation"),
            "$.preregistration.protocol.generation",
        )
    )
    environment = copy.deepcopy(
        _mapping(
            remediation_preregistration["frozen_inputs"].get("environment"),
            "$.preregistration.frozen_inputs.environment",
        )
    )

    placeholder_count = adapter_readme.count("[More Information Needed]")
    readme_lower = adapter_readme.casefold()
    adapter_metadata_names_pinned_base = (
        adapter_config.get("base_model_name_or_path") == model["repo_id"]
    )
    adapter_metadata_pins_base_revision = (
        adapter_config.get("revision") == model["revision"]
    )
    adapter_readme_mentions_compiler = all(
        token.casefold() in readme_lower for token in ("compile_decision", "attached")
    )
    adapter_readme_documentation_complete = placeholder_count == 0 and all(
        token in readme_lower
        for token in ("limitations", "runtime", "evaluation", "memory")
    )
    # This review accepts no sidecar input. A future positive decision must come
    # from NEXT_GATE_ID's strict external composite-manifest validator, never
    # from a filename or README substring inside the immutable Adapter directory.
    composite_manifest_present = False
    portable_base_model_bound = False
    base_model_revision_bound = False
    required_compiler_bound = False
    tokenizer_file_manifest_bound = False
    documentation_complete = False
    package_decision = classify_package_requirements(
        upstream_outcome=str(assessment["outcome"]),
        upstream_gate_passed=assessment.get("evaluation_gate_passed") is True,
        requirements={
            "base_model_revision_bound": base_model_revision_bound,
            "composite_manifest_present": composite_manifest_present,
            "portable_base_model_bound": portable_base_model_bound,
            "required_compiler_bound": required_compiler_bound,
            "tokenizer_file_manifest_bound": tokenizer_file_manifest_bound,
            "use_and_limitations_documented": documentation_complete,
        },
    )

    raw_reference_semantic = _number(
        _mapping(lifecycle_evidence.get("metrics"), "$.lifecycle.metrics").get(
            "decision_semantic_validity"
        ),
        "$.lifecycle.metrics.decision_semantic_validity",
    )
    raw_candidate_semantic = _number(
        raw_metrics.get("decision_semantic_validity"),
        "$.remediation.raw_metrics.decision_semantic_validity",
    )
    compiled_candidate_semantic = _number(
        compiled_metrics.get("decision_semantic_validity"),
        "$.remediation.compiled_metrics.decision_semantic_validity",
    )
    peak_reference = _nonnegative_int(
        peak.get("reference"), "$.resources.peak.reference"
    )
    peak_candidate = _nonnegative_int(
        peak.get("candidate"), "$.resources.peak.candidate"
    )
    peak_cap = _nonnegative_int(peak.get("cap"), "$.resources.peak.cap")
    elapsed_reference = _number(
        elapsed.get("reference"),
        "$.resources.elapsed.reference",
    )
    elapsed_candidate = _number(
        elapsed.get("candidate"),
        "$.resources.elapsed.candidate",
    )
    elapsed_cap = _number(elapsed.get("cap"), "$.resources.elapsed.cap")
    resources = _mapping(remediation_gate.get("resources"), "$.remediation.resources")
    performance = _mapping(
        resources.get("performance"), "$.remediation.resources.performance"
    )
    caps = _mapping(resources.get("caps"), "$.remediation.resources.caps")
    within_caps = _mapping(
        resources.get("within_caps"),
        "$.remediation.resources.within_caps",
    )
    if (
        peak_reference <= 0
        or elapsed_reference <= 0
        or _number(peak.get("ratio"), "$.resources.peak.ratio")
        != peak_candidate / peak_reference
        or _number(elapsed.get("ratio"), "$.resources.elapsed.ratio")
        != elapsed_candidate / elapsed_reference
        or peak.get("within_cap") is not True
        or elapsed.get("within_cap") is not True
        or peak_candidate > peak_cap
        or elapsed_candidate > elapsed_cap
        or comparison.get("resource_gate_passed") is not True
        or performance.get("peak_gpu_memory_bytes") != peak_candidate
        or performance.get("elapsed_seconds") != elapsed_candidate
        or caps.get("peak_gpu_memory_bytes_max") != peak_cap
        or caps.get("elapsed_seconds_max") != elapsed_cap
        or within_caps.get("peak_gpu_memory_bytes") is not True
        or within_caps.get("elapsed_seconds") is not True
        or resources.get("passed") is not True
    ):
        _fail(
            "RESOURCE_EVIDENCE_MISMATCH",
            "$.remediation.resources",
            "ratio, cap, or gate drift",
        )

    review: dict[str, Any] = {
        "review_version": REVIEW_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "gate_id": GATE_ID,
        "source_artifacts": dict(sorted(source_hashes.items())),
        "source_provenance_scope": {
            "hash_coverage": (
                "direct_review_inputs_and_direct_review_code_dependencies"
            ),
            "upstream_evaluation_lineage": (
                "transitively_bound_by_frozen_preregistration_and_remediation_gate"
            ),
            "upstream_training_lineage": (
                "transitively_bound_by_frozen_config_training_and_lifecycle_evidence"
            ),
            "raw_payloads_stored_in_review": False,
        },
        "candidate_identity": {
            "candidate_id": CANDIDATE_ID,
            "eligibility_unit": (
                "pinned_base_and_tokenizer_plus_fp32_attached_adapter_plus_"
                "fixed_compiler_and_generation_contract"
            ),
            "model": model,
            "tokenizer": tokenizer,
            "adapter": {
                "execution_form": "attached_factorized_lora",
                "merge": False,
                "files": copy.deepcopy(actual_adapter_files),
                "tensor_audit": copy.deepcopy(dict(adapter_tensor_audit)),
                "storage_dtype": "float32",
                "runtime_dtype": "float32",
            },
            "precision": copy.deepcopy(adapter_facts["precision"]),
            "decision_compiler": {
                "required": True,
                "version": compiler_lineage["version"],
                "path": compiler_lineage["path"],
                "sha256": compiler_lineage["sha256"],
                "symbol": compiler_lineage["symbol"],
                "symbol_source_sha256": compiler_lineage["symbol_source_sha256"],
            },
            "prompt": prompt,
            "generation": generation,
            "environment": environment,
        },
        "quality_review": {
            "comparison_basis": "fixed_compiler_compiled_outputs",
            "evaluation_records": compiled_metrics["records"],
            "registered_full_eval_runs": assessment["run_count"],
            "upstream_outcome": assessment["outcome"],
            "upstream_classification": assessment["classification"],
            "upstream_evaluation_gate_passed": assessment["evaluation_gate_passed"],
            "compiled_metrics": {
                "tool_accuracy": compiled_metrics["tool_accuracy"],
                "argument_exact_match": compiled_metrics["argument_exact_match"],
                "argument_field_f1": compiled_metrics["argument_field_f1"],
                "risk_macro_f1": compiled_metrics["risk_macro_f1"],
                "decision_semantic_validity": compiled_candidate_semantic,
            },
            "compiled_reference_metrics": {
                "tool_accuracy": reference_metrics["tool_accuracy"],
                "argument_exact_match": reference_metrics["argument_exact_match"],
                "argument_field_f1": reference_metrics["argument_field_f1"],
                "risk_macro_f1": reference_metrics["risk_macro_f1"],
                "decision_semantic_validity": reference_metrics[
                    "decision_semantic_validity"
                ],
            },
            "core_quality_deltas": {
                name: comparison["core_quality_comparison"][name]["delta"]
                for name in (
                    "tool_accuracy",
                    "argument_exact_match",
                    "argument_field_f1",
                    "risk_macro_f1",
                )
            },
            "strict_per_example_improvements": strict_improvements,
            "compiled_regression_event_count": comparison["regression_event_count"],
            "compiled_safety_checks": copy.deepcopy(comparison["safety_checks"]),
            "raw_semantic_validity": {
                "bf16_reference": raw_reference_semantic,
                "fp32_candidate": raw_candidate_semantic,
                "delta": raw_candidate_semantic - raw_reference_semantic,
            },
            "full_eval_repeatability_estimated": False,
            "scope": "single_hardware_single_registered_run_frozen_twenty_case_eval",
        },
        "compiler_dependency": {
            "required": True,
            "raw_candidate_semantic_validity": raw_candidate_semantic,
            "compiled_candidate_semantic_validity": compiled_candidate_semantic,
            "changed_example_ids": copy.deepcopy(compilation["changed_example_ids"]),
            "bound_by_frozen_eval_evidence": True,
            "bound_by_adapter_package": required_compiler_bound,
            "bare_adapter_must_not_inherit_compiled_metrics": True,
        },
        "resource_review": {
            "elapsed_seconds": {
                "bf16_reference": elapsed_reference,
                "fp32_candidate": elapsed_candidate,
                "ratio": elapsed["ratio"],
                "registered_cap": elapsed_cap,
            },
            "peak_gpu_memory_bytes": {
                "bf16_reference": peak_reference,
                "fp32_candidate": peak_candidate,
                "ratio": peak["ratio"],
                "registered_cap": peak_cap,
                "headroom": peak_cap - peak_candidate,
            },
            "resource_gate_passed": comparison["resource_gate_passed"],
            "stable_speedup_established": False,
            "serving_capacity_established": False,
            "new_post_hoc_resource_threshold_applied": False,
        },
        "packaging_review": {
            "repository_local_hash_pinned_load_evidenced": adapter_facts[
                "repository_local_hash_pinned_load_evidenced"
            ],
            "fp32_attached_canary_repeat_evidenced": adapter_facts[
                "fp32_attached_canary_repeat_evidenced"
            ],
            "fp32_attached_canary": adapter_facts["fp32_attached_canary"],
            "adapter_manifest_exact": True,
            "adapter_safetensors_structurally_valid": True,
            "adapter_lfs_policy_declared": True,
            "adapter_config_base_model_name_or_path": adapter_config.get(
                "base_model_name_or_path"
            ),
            "adapter_config_revision": adapter_config.get("revision"),
            "adapter_readme_placeholder_count": placeholder_count,
            "adapter_metadata_names_pinned_base": adapter_metadata_names_pinned_base,
            "adapter_metadata_pins_base_revision": (
                adapter_metadata_pins_base_revision
            ),
            "adapter_readme_mentions_compiler": adapter_readme_mentions_compiler,
            "adapter_readme_documentation_complete": (
                adapter_readme_documentation_complete
            ),
            "direct_dependency_versions_pinned": dependency_pins,
            "transitive_dependency_hashes_pinned": "--hash=sha256:" in training_lock,
            "tokenizer_file_manifest_bound": tokenizer_file_manifest_bound,
            "training_execution_source_bound_by_original_evidence": (
                "runner_source_sha256" in training_evidence
                and "training_lock_sha256" in training_evidence
            ),
            "portable_base_model_bound": portable_base_model_bound,
            "base_model_revision_bound": base_model_revision_bound,
            "required_compiler_bound": required_compiler_bound,
            "use_and_limitations_documented": documentation_complete,
            "composite_manifest_present": composite_manifest_present,
            "package_complete": package_decision["package_complete"],
            "blocking_findings": package_decision["blocking_findings"],
        },
        "eligibility_decision": {
            "compiled_quality_evidence_favorable": (
                assessment["outcome"] == "favorable"
            ),
            "repository_local_evidence_usable": source_facts[
                "repository_local_evidence_usable"
            ],
            "offline_artifact_eligible": package_decision["offline_artifact_eligible"],
            "portable_package_eligible": package_decision["offline_artifact_eligible"],
            "preferred_offline_candidate": False,
            "serving_readiness_established": False,
            "artifact_promotion_allowed": False,
            "merged_artifact_allowed": False,
            "runtime_eligible": False,
            "classification": package_decision["classification"],
        },
        "constraints": {
            "attached_execution_form_change": False,
            "decision_compiler_change": False,
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "full_eval_run": False,
            "adapter_file_mutation": False,
            "generation_change": False,
            "model_weight_copy_or_mutation": False,
            "merged_artifact_save": False,
            "prompt_change": False,
            "tokenizer_file_mutation": False,
            "artifact_promotion": False,
            "runtime_integration": False,
            "provider_integration": False,
            "mcp_integration": False,
            "desktop_integration": False,
            "serving_integration": False,
        },
        "locked_next_action": {
            "gate_id": NEXT_GATE_ID,
            "action": (
                "create and validate a metadata-only composite offline package "
                "manifest without changing Adapter or model weights"
            ),
            "acceptance": {
                "unchanged_adapter_hashes_bound": True,
                "base_repo_revision_and_weight_hash_bound": True,
                "tokenizer_revision_and_file_manifest_bound": True,
                "required_compiler_file_and_symbol_hash_bound": True,
                "prompt_generation_precision_and_environment_bound": True,
                "attached_only_execution_bound": True,
                "local_adapter_base_path_has_no_authority": True,
                "missing_component_fails_closed": True,
                "metadata_only": True,
            },
            "constraints": {
                "attached_execution_form_change": False,
                "decision_compiler_change": False,
                "full_eval_run": False,
                "new_data": False,
                "training": False,
                "eval_answer_tuning": False,
                "adapter_file_mutation": False,
                "generation_change": False,
                "model_weight_copy_or_mutation": False,
                "merge_or_save_weights": False,
                "prompt_change": False,
                "tokenizer_file_mutation": False,
                "artifact_promotion": False,
                "runtime_integration": False,
                "provider_integration": False,
                "mcp_integration": False,
                "desktop_integration": False,
                "serving_integration": False,
            },
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": (
            "fp32_attached_offline_package_incomplete_and_serving_or_runtime_"
            "readiness_not_established"
        ),
        "offline": True,
    }
    review["report_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(review)).hexdigest()
    )
    return review


def validate_fp32_attached_artifact_eligibility_review(
    review_data: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    remediation_predictions: Mapping[str, Any],
    remediation_gate: Mapping[str, Any],
    attached_dtype_isolation: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    lifecycle_evidence: Mapping[str, Any],
    load_merge_evidence: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    adapter_readme: str,
    adapter_files: Sequence[Mapping[str, Any]],
    adapter_tensor_audit: Mapping[str, Any],
    training_lock: str,
    gitattributes: str,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute and validate the complete frozen eligibility review."""

    _validate_finite_json(review_data, "$.review")
    _validate_source_hashes(expected_source_hashes)
    if source_hashes != expected_source_hashes:
        _fail(
            "SOURCE_HASH_ROOT_MISMATCH",
            "$.source_hashes",
            "observed hashes differ from external roots",
        )
    expected = build_fp32_attached_artifact_eligibility_review(
        remediation_preregistration,
        remediation_predictions,
        remediation_gate,
        attached_dtype_isolation,
        sft_config,
        training_evidence,
        lifecycle_evidence,
        load_merge_evidence,
        adapter_config,
        adapter_readme=adapter_readme,
        adapter_files=adapter_files,
        adapter_tensor_audit=adapter_tensor_audit,
        training_lock=training_lock,
        gitattributes=gitattributes,
        source_hashes=source_hashes,
        source_payloads=source_payloads,
    )
    if review_data != expected:
        _fail("REVIEW_RECOMPUTATION_MISMATCH", "$.review", "frozen review differs")
    decision = _mapping(expected["eligibility_decision"], "$.eligibility_decision")
    return {
        "frozen_review_valid": True,
        "upstream_evaluation_favorable": decision[
            "compiled_quality_evidence_favorable"
        ],
        "repository_local_evidence_usable": decision[
            "repository_local_evidence_usable"
        ],
        "offline_artifact_eligible": decision["offline_artifact_eligible"],
        "portable_package_eligible": decision["portable_package_eligible"],
        "classification": decision["classification"],
        "blocking_finding_count": len(
            expected["packaging_review"]["blocking_findings"]
        ),
        "next_gate": expected["locked_next_action"]["gate_id"],
        "runtime_eligible": expected["runtime_eligible"],
    }


def _validate_source_payload_bindings(
    source_payloads: Mapping[str, bytes],
    source_hashes: Mapping[str, str],
    *,
    remediation_preregistration: Mapping[str, Any],
    remediation_predictions: Mapping[str, Any],
    remediation_gate: Mapping[str, Any],
    attached_dtype_isolation: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    lifecycle_evidence: Mapping[str, Any],
    load_merge_evidence: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    adapter_readme: str,
    adapter_files: list[dict[str, Any]],
    adapter_tensor_audit: Mapping[str, Any],
    training_lock: str,
    gitattributes: str,
) -> None:
    if frozenset(source_payloads) != SOURCE_HASH_KEYS:
        _fail(
            "INVALID_SOURCE_PAYLOAD_KEYS",
            "$.source_payloads",
            repr(sorted(source_payloads)),
        )
    observed_hashes: dict[str, str] = {}
    for name, payload in source_payloads.items():
        if not isinstance(payload, bytes):
            _fail(
                "INVALID_SOURCE_PAYLOAD",
                f"$.source_payloads.{name}",
                type(payload).__name__,
            )
        observed_hashes[name] = "sha256:" + hashlib.sha256(payload).hexdigest()
    if observed_hashes != source_hashes:
        _fail(
            "SOURCE_PAYLOAD_HASH_MISMATCH",
            "$.source_payloads",
            "payload bytes do not match observed hashes",
        )

    json_bindings: dict[str, Mapping[str, Any]] = {
        "adapter_config": adapter_config,
        "attached_dtype_isolation": attached_dtype_isolation,
        "lifecycle_evidence": lifecycle_evidence,
        "load_merge_evidence": load_merge_evidence,
        "remediation_gate": remediation_gate,
        "remediation_predictions": remediation_predictions,
        "remediation_preregistration": remediation_preregistration,
        "sft_config": sft_config,
        "training_evidence": training_evidence,
    }
    for name, expected_object in json_bindings.items():
        actual_object = _json_payload(
            source_payloads[name], f"$.source_payloads.{name}"
        )
        if actual_object != expected_object:
            _fail(
                "SOURCE_PAYLOAD_CONTENT_MISMATCH",
                f"$.source_payloads.{name}",
                "parsed object differs from payload",
            )

    text_bindings = {
        "adapter_readme": adapter_readme,
        "gitattributes": gitattributes,
        "training_lock": training_lock,
    }
    for name, expected_text in text_bindings.items():
        try:
            actual_text = source_payloads[name].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolRouterValidationError(
                "INVALID_SOURCE_TEXT_ENCODING",
                f"$.source_payloads.{name}",
                str(exc),
            ) from exc
        if actual_text != expected_text:
            _fail(
                "SOURCE_PAYLOAD_CONTENT_MISMATCH",
                f"$.source_payloads.{name}",
                "decoded text differs from payload",
            )

    adapter_sources = {
        "adapter_config.json": "adapter_config",
        "adapter_model.safetensors": "adapter_weights",
        "README.md": "adapter_readme",
    }
    expected_adapter_files = [
        {
            "path": path,
            "bytes": len(source_payloads[source_name]),
            "sha256": source_hashes[source_name],
        }
        for path, source_name in sorted(
            adapter_sources.items(), key=lambda item: item[0].casefold()
        )
    ]
    if adapter_files != expected_adapter_files:
        _fail(
            "ADAPTER_PAYLOAD_MANIFEST_MISMATCH",
            "$.adapter_files",
            "manifest differs from single-read payloads",
        )
    if (
        inspect_adapter_safetensors_bytes(source_payloads["adapter_weights"])
        != adapter_tensor_audit
    ):
        _fail(
            "ADAPTER_TENSOR_AUDIT_MISMATCH",
            "$.adapter_tensor_audit",
            "audit differs from single-read weights payload",
        )

    evaluation = _json_payload(
        source_payloads["evaluation_fixture"],
        "$.source_payloads.evaluation_fixture",
    )
    if not isinstance(evaluation, list):
        _fail(
            "EVALUATION_FIXTURE_MISMATCH",
            "$.source_payloads.evaluation_fixture",
            "expected array",
        )
    evaluation_records = [
        _mapping(item, f"$.source_payloads.evaluation_fixture[{index}]")
        for index, item in enumerate(evaluation)
    ]
    frozen_inputs = _mapping(
        remediation_preregistration.get("frozen_inputs"),
        "$.remediation_preregistration.frozen_inputs",
    )
    frozen_evaluation = _mapping(
        frozen_inputs.get("evaluation"),
        "$.remediation_preregistration.frozen_inputs.evaluation",
    )
    evaluation_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                evaluation_records,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    if (
        len(evaluation_records) != frozen_evaluation.get("records")
        or [item.get("example_id") for item in evaluation_records]
        != frozen_evaluation.get("order")
        or evaluation_digest != frozen_evaluation.get("digest")
    ):
        _fail(
            "EVALUATION_FIXTURE_MISMATCH",
            "$.source_payloads.evaluation_fixture",
            "records, order, or canonical digest",
        )


def _json_payload(payload: bytes, path: str) -> object:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolRouterValidationError(
            "INVALID_SOURCE_JSON",
            path,
            str(exc),
        ) from exc
    _validate_finite_json(value, path)
    return value


def _validate_source_hashes(source_hashes: Mapping[str, str]) -> None:
    if frozenset(source_hashes) != SOURCE_HASH_KEYS:
        _fail(
            "INVALID_SOURCE_HASH_KEYS",
            "$.source_hashes",
            repr(sorted(source_hashes)),
        )
    for name, value in source_hashes.items():
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            _fail("INVALID_SOURCE_HASH", f"$.source_hashes.{name}", repr(value))


def _validate_source_chain(
    preregistration: Mapping[str, Any],
    predictions: Mapping[str, Any],
    remediation: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    training: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    load_merge: Mapping[str, Any],
    adapter_files: list[dict[str, Any]],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if preregistration.get("gate_id") != "FC-MVP-001-fp32-attached-remediation-eval-v1":
        _fail(
            "PREREGISTRATION_GATE_MISMATCH",
            "$.preregistration.gate_id",
            repr(preregistration.get("gate_id")),
        )
    if predictions.get("gate_id") != preregistration.get("gate_id"):
        _fail(
            "PREDICTION_GATE_MISMATCH",
            "$.predictions.gate_id",
            repr(predictions.get("gate_id")),
        )
    if remediation.get("gate_id") != preregistration.get("gate_id"):
        _fail(
            "REMEDIATION_GATE_MISMATCH",
            "$.remediation.gate_id",
            repr(remediation.get("gate_id")),
        )
    if (
        remediation.get("preregistration_sha256")
        != source_hashes["remediation_preregistration"]
    ):
        _fail("PREREGISTRATION_DIGEST_MISMATCH", "$.remediation", "source hash")
    prediction_artifact = _mapping(
        remediation.get("prediction_artifact"), "$.remediation.prediction_artifact"
    )
    if prediction_artifact.get("sha256") != source_hashes["remediation_predictions"]:
        _fail("PREDICTION_DIGEST_MISMATCH", "$.remediation", "source hash")

    lineage = _mapping(
        preregistration.get("source_lineage"), "$.preregistration.source_lineage"
    )
    lineage_checks = {
        "sft_config": ("base_config", "sha256"),
        "training_evidence": ("training_evidence", "sha256"),
        "lifecycle_evidence": ("lifecycle_evidence", "sha256"),
        "training_lock": ("training_lock", "sha256"),
        "decision_compiler_source": ("decision_compiler_source", "sha256"),
    }
    for source_name, (lineage_name, field) in lineage_checks.items():
        record = _mapping(
            lineage.get(lineage_name),
            f"$.preregistration.source_lineage.{lineage_name}",
        )
        if record.get(field) != source_hashes[source_name]:
            _fail(
                "SOURCE_LINEAGE_MISMATCH",
                f"$.source_hashes.{source_name}",
                lineage_name,
            )

    config_digest = canonical_config_sha256(sft_config)
    base_config_lineage = _mapping(
        lineage.get("base_config"), "$.preregistration.source_lineage.base_config"
    )
    if (
        base_config_lineage.get("canonical_sha256") != config_digest
        or training.get("config_sha256") != config_digest
        or lifecycle.get("canonical_config_sha256") != config_digest
        or load_merge.get("config_sha256") != config_digest
    ):
        _fail("SFT_CONFIG_CHAIN_MISMATCH", "$.sft_config", config_digest)

    frozen_inputs = _mapping(
        preregistration.get("frozen_inputs"), "$.preregistration.frozen_inputs"
    )
    for name in ("model", "tokenizer", "environment"):
        if frozen_inputs.get(name) != sft_config.get(name) or predictions.get(
            name
        ) != sft_config.get(name):
            _fail("CANDIDATE_IDENTITY_MISMATCH", f"$.{name}", name)

    protocol = _mapping(preregistration.get("protocol"), "$.preregistration.protocol")
    generation = _mapping(
        protocol.get("generation"), "$.preregistration.protocol.generation"
    )
    if predictions.get("generation") != generation:
        _fail(
            "GENERATION_CONTRACT_MISMATCH",
            "$.predictions.generation",
            "preregistration",
        )
    generation_requirements = {
        "attn_implementation": "sdpa",
        "autocast": False,
        "do_sample": False,
        "low_level_cuda_kernel_identity_claimed": False,
        "tf32": False,
        "torch_dtype": "float32",
    }
    if any(
        generation.get(name) != expected
        for name, expected in generation_requirements.items()
    ):
        _fail(
            "GENERATION_CONTRACT_MISMATCH",
            "$.preregistration.protocol.generation",
            repr(generation),
        )
    prompt = _mapping(
        frozen_inputs.get("prompt"), "$.preregistration.frozen_inputs.prompt"
    )
    evaluation = _mapping(
        frozen_inputs.get("evaluation"),
        "$.preregistration.frozen_inputs.evaluation",
    )
    if (
        predictions.get("prompt_sha256") != prompt.get("sha256")
        or prompt.get("sha256") != source_hashes["prompt"]
        or predictions.get("eval_digest") != evaluation.get("digest")
        or predictions.get("example_order") != evaluation.get("order")
    ):
        _fail(
            "EVALUATION_IDENTITY_MISMATCH",
            "$.predictions",
            "prompt, digest, or example order",
        )

    manifests = (
        frozen_inputs.get("adapter_files"),
        predictions.get("adapter_files"),
        _mapping(training.get("final_adapter"), "$.training.final_adapter").get(
            "files"
        ),
        load_merge.get("adapter_files"),
    )
    if any(value != adapter_files for value in manifests):
        _fail("ADAPTER_MANIFEST_MISMATCH", "$.adapter_files", "source chain")
    adapter_source_names = {
        "README.md": "adapter_readme",
        "adapter_config.json": "adapter_config",
        "adapter_model.safetensors": "adapter_weights",
    }
    if {item.get("path") for item in adapter_files} != set(adapter_source_names) or any(
        item.get("sha256") != source_hashes[adapter_source_names[str(item.get("path"))]]
        for item in adapter_files
    ):
        _fail(
            "ADAPTER_SOURCE_HASH_MISMATCH",
            "$.adapter_files",
            "manifest does not bind observed Adapter sources",
        )
    lifecycle_hashes = _mapping(
        lifecycle.get("artifact_hashes"), "$.lifecycle.artifact_hashes"
    )
    if (
        lifecycle_hashes.get("baseline/fc-mvp-001-lora-sft-v2-load-merge.json")
        != source_hashes["load_merge_evidence"]
    ):
        _fail(
            "LOAD_MERGE_LIFECYCLE_HASH_MISMATCH",
            "$.lifecycle.artifact_hashes",
            "load/merge evidence",
        )
    for item in adapter_files:
        relative = item.get("path")
        expected_path = f"baseline/adapters/fc-mvp-001-lora-sft-v2/{relative}"
        if lifecycle_hashes.get(expected_path) != item.get("sha256"):
            _fail(
                "ADAPTER_LIFECYCLE_HASH_MISMATCH",
                f"$.adapter_files.{relative}",
                expected_path,
            )

    assessment = _mapping(remediation.get("assessment"), "$.remediation.assessment")
    gates = _mapping(remediation.get("gates"), "$.remediation.gates")
    expected_gate_keys = {
        "core_quality",
        "full_eval",
        "per_example_regression",
        "resource",
        "safety",
    }
    outcome = assessment.get("outcome")
    if (
        assessment.get("candidate_id") != CANDIDATE_ID
        or not isinstance(outcome, str)
        or outcome not in UPSTREAM_CLASSIFICATIONS
        or assessment.get("classification") != UPSTREAM_CLASSIFICATIONS[outcome]
        or assessment.get("candidate_count") != 1
        or assessment.get("run_count") != 1
        or assessment.get("evaluation_gate_passed") is not True
        or assessment.get("runtime_eligible") is not False
        or assessment.get("gates") != gates
        or set(gates) != expected_gate_keys
        or any(value is not True for value in gates.values())
        or remediation.get("compiled_model_saved") is not False
        or remediation.get("tensor_payload_saved") is not False
        or remediation.get("runtime_eligible") is not False
        or remediation.get("offline") is not True
    ):
        _fail(
            "UPSTREAM_EVALUATION_NOT_ELIGIBLE_FOR_REVIEW",
            "$.remediation",
            repr(assessment),
        )
    constraints = _mapping(
        preregistration.get("constraints"),
        "$.preregistration.constraints",
    )
    forbidden_constraint_keys = {
        "adapter_artifact_promotion",
        "desktop_integration",
        "eval_answer_tuning",
        "mcp_integration",
        "merged_artifact_promotion",
        "merged_artifact_save",
        "new_data",
        "provider_integration",
        "runtime_integration",
        "training",
    }
    if (
        remediation.get("constraints") != constraints
        or constraints.get("candidate_count") != 1
        or constraints.get("run_count") != 1
        or any(constraints.get(name) is not False for name in forbidden_constraint_keys)
    ):
        _fail(
            "UPSTREAM_CONSTRAINT_MISMATCH",
            "$.remediation.constraints",
            repr(remediation.get("constraints")),
        )
    locked = _mapping(
        remediation.get("locked_next_action"), "$.remediation.locked_next_action"
    )
    if (
        locked.get("gate_id") != GATE_ID
        or locked.get("artifact_promotion_allowed") is not False
        or locked.get("runtime_integration_allowed") is not False
    ):
        _fail(
            "UPSTREAM_NEXT_ACTION_MISMATCH",
            "$.remediation.locked_next_action",
            repr(locked),
        )
    return {"repository_local_evidence_usable": True}


def _validate_adapter_contract(
    preregistration: Mapping[str, Any],
    predictions: Mapping[str, Any],
    isolation: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    training: Mapping[str, Any],
    load_merge: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    adapter_files: list[dict[str, Any]],
    tensor_audit: Mapping[str, Any],
) -> dict[str, Any]:
    lora = _mapping(sft_config.get("lora"), "$.sft_config.lora")
    if (
        adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("inference_mode") is not True
        or adapter_config.get("r") != lora.get("rank")
        or adapter_config.get("lora_alpha") != lora.get("alpha")
        or adapter_config.get("lora_dropout") != lora.get("dropout")
        or adapter_config.get("bias") != lora.get("bias")
        or set(adapter_config.get("target_modules", []))
        != set(lora.get("target_modules", []))
    ):
        _fail("ADAPTER_CONFIG_CONTRACT_MISMATCH", "$.adapter_config", "LoRA contract")

    parameters = _mapping(training.get("parameters"), "$.training.parameters")
    storage = _mapping(predictions.get("storage_audit"), "$.predictions.storage_audit")
    adapter_storage = _mapping(
        storage.get("adapter"), "$.predictions.storage_audit.adapter"
    )
    weight_items = [
        item
        for item in adapter_files
        if item.get("path") == "adapter_model.safetensors"
    ]
    if len(weight_items) != 1:
        _fail(
            "ADAPTER_WEIGHT_MANIFEST_MISMATCH",
            "$.adapter_files",
            repr(weight_items),
        )
    if (
        tensor_audit.get("file_bytes") != weight_items[0].get("bytes")
        or tensor_audit.get("tensor_count") != 224
        or tensor_audit.get("module_count") != 112
        or tensor_audit.get("layer_count") != 28
        or tensor_audit.get("layers") != list(range(28))
        or tensor_audit.get("parameter_count") != parameters.get("trainable")
        or tensor_audit.get("parameter_count") != adapter_storage.get("elements")
        or tensor_audit.get("dtype_tensor_counts") != {"F32": 224}
        or tensor_audit.get("dtype_element_counts") != {"F32": 4_358_144}
        or tensor_audit.get("shape_counts")
        != {"1536x16": 56, "16x1536": 112, "256x16": 56}
        or tensor_audit.get("target_module_tensor_counts")
        != {"k_proj": 56, "o_proj": 56, "q_proj": 56, "v_proj": 56}
        or tensor_audit.get("lora_matrix_tensor_counts") != {"A": 112, "B": 112}
        or tensor_audit.get("data_offsets_contiguous") is not True
        or tensor_audit.get("topology_complete") is not True
    ):
        _fail(
            "ADAPTER_TENSOR_AUDIT_MISMATCH",
            "$.adapter_tensor_audit",
            repr(tensor_audit),
        )

    if (
        load_merge.get("adapter_files") != adapter_files
        or load_merge.get("safe_merge") is not True
        or load_merge.get("outputs_identical") is not False
        or load_merge.get("merged_model_saved") is not False
        or not isinstance(load_merge.get("loaded_output"), str)
        or not load_merge["loaded_output"]
    ):
        _fail("LOAD_MERGE_EVIDENCE_MISMATCH", "$.load_merge", "attached load")

    protocol = _mapping(isolation.get("protocol"), "$.isolation.protocol")
    paths = _mapping(protocol.get("paths"), "$.isolation.protocol.paths")
    fp32_path = _mapping(
        paths.get("fp32_attached_adapter"), "$.isolation.protocol.paths.fp32"
    )
    stability = _mapping(
        isolation.get("path_repeat_stability"), "$.isolation.path_repeat_stability"
    )
    fp32_stability = _mapping(
        stability.get("fp32_attached_adapter"), "$.isolation.path_repeat_stability.fp32"
    )
    dtype_gate = _mapping(
        isolation.get("dtype_isolation_gate"), "$.isolation.dtype_isolation_gate"
    )
    runs = [
        _mapping(item, f"$.isolation.runs[{index}]")
        for index, item in enumerate(
            _sequence(isolation.get("runs"), "$.isolation.runs")
        )
    ]
    expected_run_order = [
        (0, "bf16-attached-dtype-r1", "bf16_attached_adapter"),
        (1, "fp32-attached-dtype-r1", "fp32_attached_adapter"),
        (2, "fp32-attached-dtype-r2", "fp32_attached_adapter"),
        (3, "bf16-attached-dtype-r2", "bf16_attached_adapter"),
    ]
    if len(runs) != len(expected_run_order) or any(
        (
            run.get("order_index"),
            run.get("run_id"),
            run.get("path"),
        )
        != expected
        for run, expected in zip(runs, expected_run_order, strict=True)
    ):
        _fail("DTYPE_ISOLATION_RUN_ORDER_MISMATCH", "$.isolation.runs", "ABBA")
    fp32_runs = runs[1:3]
    token_digest = fp32_runs[0].get("token_ids_sha256")
    output_digest = fp32_runs[0].get("output_sha256")
    isolation_generation = _mapping(
        protocol.get("generation"),
        "$.isolation.protocol.generation",
    )
    if (
        isolation.get("example_id") != "eval-001"
        or any(item.get("fresh_load") is not True for item in fp32_runs)
        or not isinstance(token_digest, str)
        or SHA256_PATTERN.fullmatch(token_digest) is None
        or not isinstance(output_digest, str)
        or SHA256_PATTERN.fullmatch(output_digest) is None
        or any(item.get("token_ids_sha256") != token_digest for item in fp32_runs)
        or any(item.get("output_sha256") != output_digest for item in fp32_runs)
        or fp32_stability.get("passed") is not True
        or dtype_gate.get("passed") is not True
        or fp32_path.get("merge") is not False
        or fp32_path.get("base_load_dtype") != "float32"
        or fp32_path.get("adapter_runtime_dtype") != "float32"
        or fp32_path.get("autocast_adapter_dtype") is not True
        or isolation_generation.get("tf32") is not False
        or isolation_generation.get("autocast") is not False
        or isolation_generation.get("do_sample") is not False
        or isolation_generation.get("attn_implementation") != "sdpa"
    ):
        _fail("FP32_ATTACHED_REPEAT_EVIDENCE_MISMATCH", "$.isolation", "canary")

    precision = _mapping(
        predictions.get("precision_audit"), "$.predictions.precision_audit"
    )
    if (
        precision.get("attached_execution_form") != "attached_factorized_lora"
        or precision.get("autocast_adapter_dtype") is not True
        or precision.get("training") is not False
        or precision.get("autocast_enabled") is not False
        or precision.get("lora_target_modules") != 112
        or precision.get("lora_parameter_tensors") != 224
    ):
        _fail(
            "FP32_PRECISION_CONTRACT_MISMATCH",
            "$.predictions.precision_audit",
            repr(precision),
        )
    return {
        "precision": {
            "checkpoint_source_storage_dtype": "bfloat16",
            "base_runtime_dtype": "float32",
            "adapter_storage_dtype": "float32",
            "adapter_runtime_dtype": "float32",
            "autocast_adapter_dtype": True,
            "autocast_enabled": False,
            "tf32": False,
            "attached_execution_form": "attached_factorized_lora",
        },
        "repository_local_hash_pinned_load_evidenced": True,
        "fp32_attached_canary_repeat_evidenced": True,
        "fp32_attached_canary": {
            "example_id": isolation["example_id"],
            "fresh_loads": 2,
            "token_ids_sha256": token_digest,
            "output_sha256": output_digest,
            "full_trace_repeat_stable": True,
        },
    }


def _validate_training_lock(
    training_lock: str, sft_config: Mapping[str, Any]
) -> dict[str, str]:
    pins: dict[str, str] = {}
    for index, raw_line in enumerate(training_lock.splitlines()):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9-]+)==([^\s]+)", line)
        if match is None:
            _fail("INVALID_TRAINING_LOCK", f"$.training_lock[{index}]", line)
        name, version = match.groups()
        normalized = name.casefold()
        if normalized in pins:
            _fail("DUPLICATE_TRAINING_LOCK_PIN", f"$.training_lock[{index}]", name)
        pins[normalized] = version
    environment = _mapping(sft_config.get("environment"), "$.sft_config.environment")
    environment_names = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "huggingface-hub": "huggingface_hub",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
    }
    for package, field in environment_names.items():
        if pins.get(package) != environment.get(field):
            _fail(
                "TRAINING_LOCK_ENVIRONMENT_MISMATCH",
                f"$.training_lock.{package}",
                field,
            )
    return dict(sorted(pins.items()))


def _strict_improvements(comparison: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, raw_item in enumerate(
        _sequence(comparison.get("per_example"), "$.comparison.per_example")
    ):
        item = _mapping(raw_item, f"$.comparison.per_example[{index}]")
        example_id = item.get("example_id")
        if not isinstance(example_id, str):
            _fail(
                "INVALID_EXAMPLE_ID",
                f"$.comparison.per_example[{index}]",
                repr(example_id),
            )
        for dimension in _sequence(
            item.get("improvement_dimensions"),
            f"$.comparison.per_example[{index}].improvement_dimensions",
        ):
            if not isinstance(dimension, str):
                _fail(
                    "INVALID_IMPROVEMENT_DIMENSION",
                    f"$.comparison.per_example[{index}]",
                    repr(dimension),
                )
            result.append({"example_id": example_id, "dimension": dimension})
    return result


def _validate_finite_json(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("NON_STRING_JSON_KEY", path, repr(key))
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("NONFINITE_NUMBER", path, repr(value))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        _fail("INVALID_OBJECT_KEYS", path, repr(sorted(value)))


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", path, type(value).__name__)
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail("EXPECTED_ARRAY", path, type(value).__name__)
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        _fail("EXPECTED_NONNEGATIVE_INTEGER", path, repr(value))
    return value


def _positive_int(value: object, path: str) -> int:
    result = _nonnegative_int(value, path)
    if result == 0:
        _fail("EXPECTED_POSITIVE_INTEGER", path, repr(value))
    return result


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("EXPECTED_NUMBER", path, repr(value))
    result = float(value)
    if not math.isfinite(result):
        _fail("NONFINITE_NUMBER", path, repr(value))
    return result


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "CANDIDATE_ID",
    "EXPERIMENT_ID",
    "GATE_ID",
    "NEXT_GATE_ID",
    "REVIEW_VERSION",
    "SOURCE_HASH_KEYS",
    "build_fp32_attached_artifact_eligibility_review",
    "classify_package_requirements",
    "inspect_adapter_safetensors",
    "validate_fp32_attached_artifact_eligibility_review",
]
