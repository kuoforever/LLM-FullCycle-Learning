"""Pure analysis for the Tool Router FP32 attached/merge isolation gate."""

from __future__ import annotations

import re
from typing import Any, Iterable, NoReturn

from .tool_router import ToolRouterValidationError

FP32_ATTACHED_MERGE_ISOLATION_VERSION = 1


def analyze_attached_repeat_stability(
    first_token_ids: Iterable[int],
    second_token_ids: Iterable[int],
    *,
    first_output_sha256: str,
    second_output_sha256: str,
    first_score_trace_sha256: str,
    second_score_trace_sha256: str,
    first_raw_logit_trace_sha256: str,
    second_raw_logit_trace_sha256: str,
    precision_audits_identical: bool,
) -> dict[str, bool]:
    """Require two fresh attached-FP32 runs to be numerically repeat-stable."""

    first = _token_run(first_token_ids, "$.first_token_ids")
    second = _token_run(second_token_ids, "$.second_token_ids")
    digests = (
        ("$.first_output_sha256", first_output_sha256),
        ("$.second_output_sha256", second_output_sha256),
        ("$.first_score_trace_sha256", first_score_trace_sha256),
        ("$.second_score_trace_sha256", second_score_trace_sha256),
        ("$.first_raw_logit_trace_sha256", first_raw_logit_trace_sha256),
        ("$.second_raw_logit_trace_sha256", second_raw_logit_trace_sha256),
    )
    for path, value in digests:
        _sha256(value, path)
    if not isinstance(precision_audits_identical, bool):
        _fail(
            "INVALID_PRECISION_IDENTITY",
            "$.precision_audits_identical",
            repr(precision_audits_identical),
        )

    result = {
        "token_identity": first == second,
        "output_identity": first_output_sha256 == second_output_sha256,
        "score_trace_identity": (
            first_score_trace_sha256 == second_score_trace_sha256
        ),
        "raw_logit_trace_identity": (
            first_raw_logit_trace_sha256 == second_raw_logit_trace_sha256
        ),
        "precision_audit_identity": precision_audits_identical,
    }
    result["passed"] = all(result.values())
    return result


def analyze_same_dtype_tokens(
    attached_token_ids: Iterable[int],
    merged_token_ids: Iterable[int],
) -> dict[str, Any]:
    """Locate the earliest attached-FP32 versus merged-FP32 token drift."""

    attached = _token_run(attached_token_ids, "$.attached_token_ids")
    merged = _token_run(merged_token_ids, "$.merged_token_ids")
    first = _first_divergence(attached, merged)
    if first is None:
        return {
            "attached_token_count": len(attached),
            "merged_token_count": len(merged),
            "cross_path_identical": True,
            "common_prefix_generated_tokens": len(attached),
            "first_divergent_token_index": None,
            "attached_token_id": None,
            "merged_token_id": None,
            "classification": "same_dtype_token_identity",
        }

    index, attached_id, merged_id = first
    classification = (
        "same_dtype_token_drift"
        if attached_id is not None and merged_id is not None
        else "same_dtype_termination_drift"
    )
    return {
        "attached_token_count": len(attached),
        "merged_token_count": len(merged),
        "cross_path_identical": False,
        "common_prefix_generated_tokens": index,
        "first_divergent_token_index": index,
        "attached_token_id": attached_id,
        "merged_token_id": merged_id,
        "classification": classification,
    }


def select_comparison_step(
    token_analysis: dict[str, Any],
    *,
    frozen_boundary_index: int,
) -> dict[str, Any]:
    """Select a causally comparable generation step before inspecting vectors."""

    _nonnegative_int(frozen_boundary_index, "$.frozen_boundary_index")
    classification, attached_count, merged_count = _validate_token_analysis(
        token_analysis
    )
    if classification == "same_dtype_token_drift":
        step = _nonnegative_int(
            token_analysis.get("first_divergent_token_index"),
            "$.token_analysis.first_divergent_token_index",
        )
        basis = "first_same_dtype_generated_token_divergence"
    elif classification == "same_dtype_token_identity":
        step = frozen_boundary_index
        basis = "frozen_bf16_token_boundary_context"
    else:
        _fail("TOKEN_PATHS_NOT_STEP_COMPARABLE", "$.token_analysis", repr(token_analysis))
    if step >= attached_count or step >= merged_count:
        _fail("COMPARISON_STEP_OUT_OF_RANGE", "$.comparison_step", repr(step))
    return {
        "step_index": step,
        "basis": basis,
        "common_prefix_generated_tokens": (
            step if classification == "same_dtype_token_drift" else attached_count
        ),
    }


def classify_same_dtype_effect(
    token_analysis: dict[str, Any],
    *,
    attached_repeat_stable: bool,
    merged_candidate_reproduced: bool,
    attached_emitted_token_id: int,
    merged_emitted_token_id: int,
    attached_score_top_token_id: int,
    merged_score_top_token_id: int,
    attached_raw_logit_top_token_id: int,
    merged_raw_logit_top_token_id: int,
    full_score_traces_identical: bool,
    full_raw_logit_traces_identical: bool,
    comparison_score_vectors_identical: bool,
    comparison_raw_logit_vectors_identical: bool,
) -> str:
    """Classify the controlled FP32 execution-form effect without overclaiming."""

    if attached_repeat_stable is not True:
        _fail("ATTACHED_FP32_REPEAT_STABILITY_NOT_ESTABLISHED", "$", repr(token_analysis))
    if merged_candidate_reproduced is not True:
        _fail("FP32_MERGED_CANDIDATE_NOT_REPRODUCED", "$", repr(token_analysis))
    ids = (
        ("$.attached_emitted_token_id", attached_emitted_token_id),
        ("$.merged_emitted_token_id", merged_emitted_token_id),
        ("$.attached_score_top_token_id", attached_score_top_token_id),
        ("$.merged_score_top_token_id", merged_score_top_token_id),
        ("$.attached_raw_logit_top_token_id", attached_raw_logit_top_token_id),
        ("$.merged_raw_logit_top_token_id", merged_raw_logit_top_token_id),
    )
    for path, value in ids:
        _token_id(value, path)
    flags = (
        ("$.full_score_traces_identical", full_score_traces_identical),
        ("$.full_raw_logit_traces_identical", full_raw_logit_traces_identical),
        ("$.comparison_score_vectors_identical", comparison_score_vectors_identical),
        (
            "$.comparison_raw_logit_vectors_identical",
            comparison_raw_logit_vectors_identical,
        ),
    )
    for path, value in flags:
        if not isinstance(value, bool):
            _fail("INVALID_IDENTITY_FLAG", path, repr(value))
    token_classification, _, _ = _validate_token_analysis(token_analysis)
    if (
        attached_score_top_token_id != attached_emitted_token_id
        or merged_score_top_token_id != merged_emitted_token_id
    ):
        return "generation_score_alignment_failure"
    if full_score_traces_identical and not comparison_score_vectors_identical:
        _fail("SCORE_TRACE_VECTOR_IDENTITY_CONFLICT", "$", repr(token_analysis))
    if full_raw_logit_traces_identical and not comparison_raw_logit_vectors_identical:
        _fail("RAW_TRACE_VECTOR_IDENTITY_CONFLICT", "$", repr(token_analysis))
    if (
        comparison_score_vectors_identical
        and attached_score_top_token_id != merged_score_top_token_id
    ):
        _fail("SCORE_VECTOR_TOP_TOKEN_CONFLICT", "$", repr(token_analysis))
    if (
        comparison_raw_logit_vectors_identical
        and attached_raw_logit_top_token_id != merged_raw_logit_top_token_id
    ):
        _fail("RAW_VECTOR_TOP_TOKEN_CONFLICT", "$", repr(token_analysis))
    if token_classification == "same_dtype_token_identity":
        if attached_emitted_token_id != merged_emitted_token_id:
            _fail("IDENTICAL_PATH_EMISSION_MISMATCH", "$.token_analysis", repr(token_analysis))
        if all(
            (
                full_score_traces_identical,
                full_raw_logit_traces_identical,
                comparison_score_vectors_identical,
                comparison_raw_logit_vectors_identical,
            )
        ):
            return "deterministic_fp32_attached_merged_full_trace_identity"
        return (
            "deterministic_fp32_attached_vs_merged_"
            "numerical_drift_without_token_drift"
        )
    if token_classification != "same_dtype_token_drift":
        _fail("TOKEN_DRIFT_NOT_ESTABLISHED", "$.token_analysis", repr(token_analysis))
    if full_score_traces_identical or comparison_score_vectors_identical:
        _fail("TOKEN_DRIFT_SCORE_IDENTITY_CONFLICT", "$", repr(token_analysis))
    if (
        token_analysis.get("attached_token_id") != attached_emitted_token_id
        or token_analysis.get("merged_token_id") != merged_emitted_token_id
    ):
        _fail("DIVERGENT_EMISSION_MISMATCH", "$.token_analysis", repr(token_analysis))
    if (
        attached_raw_logit_top_token_id == attached_emitted_token_id
        and merged_raw_logit_top_token_id == merged_emitted_token_id
    ):
        return (
            "deterministic_fp32_attached_vs_merged_"
            "raw_logit_boundary_flip"
        )
    if attached_raw_logit_top_token_id == merged_raw_logit_top_token_id:
        return (
            "deterministic_fp32_attached_vs_merged_"
            "logits_processor_boundary_flip"
        )
    return "deterministic_fp32_attached_vs_merged_mixed_logit_score_drift"


def _token_run(value: Iterable[int], path: str) -> list[int]:
    run = list(value)
    if not run or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in run
    ):
        _fail("INVALID_TOKEN_RUN", path, repr(run))
    return run


def _validate_token_analysis(
    value: dict[str, Any],
) -> tuple[str, int, int]:
    if not isinstance(value, dict):
        _fail("INVALID_TOKEN_ANALYSIS", "$.token_analysis", repr(value))
    classification = value.get("classification")
    attached_count = _nonnegative_int(
        value.get("attached_token_count"),
        "$.token_analysis.attached_token_count",
    )
    merged_count = _nonnegative_int(
        value.get("merged_token_count"),
        "$.token_analysis.merged_token_count",
    )
    common_prefix = _nonnegative_int(
        value.get("common_prefix_generated_tokens"),
        "$.token_analysis.common_prefix_generated_tokens",
    )
    if classification == "same_dtype_token_identity":
        if (
            value.get("cross_path_identical") is not True
            or attached_count != merged_count
            or common_prefix != attached_count
            or value.get("first_divergent_token_index") is not None
            or value.get("attached_token_id") is not None
            or value.get("merged_token_id") is not None
        ):
            _fail("TOKEN_IDENTITY_ANALYSIS_CONFLICT", "$.token_analysis", repr(value))
    elif classification == "same_dtype_token_drift":
        divergence = _nonnegative_int(
            value.get("first_divergent_token_index"),
            "$.token_analysis.first_divergent_token_index",
        )
        attached_id = _token_id(
            value.get("attached_token_id"),
            "$.token_analysis.attached_token_id",
        )
        merged_id = _token_id(
            value.get("merged_token_id"),
            "$.token_analysis.merged_token_id",
        )
        if (
            value.get("cross_path_identical") is not False
            or common_prefix != divergence
            or divergence >= attached_count
            or divergence >= merged_count
            or attached_id == merged_id
        ):
            _fail("TOKEN_DRIFT_ANALYSIS_CONFLICT", "$.token_analysis", repr(value))
    elif classification == "same_dtype_termination_drift":
        divergence = _nonnegative_int(
            value.get("first_divergent_token_index"),
            "$.token_analysis.first_divergent_token_index",
        )
        ids = (value.get("attached_token_id"), value.get("merged_token_id"))
        if (
            value.get("cross_path_identical") is not False
            or common_prefix != divergence
            or divergence != min(attached_count, merged_count)
            or sum(item is None for item in ids) != 1
        ):
            _fail(
                "TOKEN_TERMINATION_ANALYSIS_CONFLICT",
                "$.token_analysis",
                repr(value),
            )
        for index, token_id in enumerate(ids):
            if token_id is not None:
                _token_id(token_id, f"$.token_analysis.termination_token_ids[{index}]")
    else:
        _fail("INVALID_TOKEN_ANALYSIS_CLASSIFICATION", "$.token_analysis", repr(value))
    return classification, attached_count, merged_count


def _token_id(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("INVALID_TOKEN_ID", path, repr(value))
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("INVALID_NONNEGATIVE_INTEGER", path, repr(value))
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail("INVALID_SHA256", path, repr(value))
    return value


def _first_divergence(
    attached: list[int],
    merged: list[int],
) -> tuple[int, int | None, int | None] | None:
    for index, (left, right) in enumerate(zip(attached, merged)):
        if left != right:
            return index, left, right
    if len(attached) == len(merged):
        return None
    index = min(len(attached), len(merged))
    attached_id = attached[index] if index < len(attached) else None
    merged_id = merged[index] if index < len(merged) else None
    return index, attached_id, merged_id


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "FP32_ATTACHED_MERGE_ISOLATION_VERSION",
    "analyze_attached_repeat_stability",
    "analyze_same_dtype_tokens",
    "classify_same_dtype_effect",
    "select_comparison_step",
]
