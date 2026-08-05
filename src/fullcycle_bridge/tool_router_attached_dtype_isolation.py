"""Pure analysis for the attached-Adapter BF16 versus FP32 dtype gate."""

from __future__ import annotations

import re
from typing import Any, Iterable, NoReturn

from .tool_router import ToolRouterValidationError

ATTACHED_DTYPE_ISOLATION_VERSION = 1
FROZEN_BOUNDARY_INDEX = 45
FROZEN_BF16_TOKEN_ID = 1866
FROZEN_FP32_TOKEN_ID = 3849
FROZEN_TOKEN_COUNT = 48
VOCABULARY_SIZE = 151_936

_TOKEN_ANALYSIS_KEYS = {
    "bf16_token_count",
    "fp32_token_count",
    "cross_dtype_identical",
    "common_prefix_generated_tokens",
    "first_divergent_token_index",
    "bf16_token_id",
    "fp32_token_id",
    "classification",
}


def analyze_path_repeat_stability(
    first_token_ids: Iterable[int],
    second_token_ids: Iterable[int],
    *,
    first_output_sha256: str,
    second_output_sha256: str,
    first_score_trace_sha256: str,
    second_score_trace_sha256: str,
    first_raw_logit_trace_sha256: str,
    second_raw_logit_trace_sha256: str,
    first_score_vector_sha256: str,
    second_score_vector_sha256: str,
    first_raw_logit_vector_sha256: str,
    second_raw_logit_vector_sha256: str,
    precision_audits_identical: bool,
) -> dict[str, bool]:
    """Require two fresh runs of one attached dtype path to be identical."""

    first = _token_run(first_token_ids, "$.first_token_ids")
    second = _token_run(second_token_ids, "$.second_token_ids")
    digests = (
        ("$.first_output_sha256", first_output_sha256),
        ("$.second_output_sha256", second_output_sha256),
        ("$.first_score_trace_sha256", first_score_trace_sha256),
        ("$.second_score_trace_sha256", second_score_trace_sha256),
        ("$.first_raw_logit_trace_sha256", first_raw_logit_trace_sha256),
        ("$.second_raw_logit_trace_sha256", second_raw_logit_trace_sha256),
        ("$.first_score_vector_sha256", first_score_vector_sha256),
        ("$.second_score_vector_sha256", second_score_vector_sha256),
        ("$.first_raw_logit_vector_sha256", first_raw_logit_vector_sha256),
        ("$.second_raw_logit_vector_sha256", second_raw_logit_vector_sha256),
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
        "comparison_score_vector_identity": (
            first_score_vector_sha256 == second_score_vector_sha256
        ),
        "comparison_raw_logit_vector_identity": (
            first_raw_logit_vector_sha256 == second_raw_logit_vector_sha256
        ),
        "precision_audit_identity": precision_audits_identical,
    }
    result["passed"] = all(result.values())
    return result


def analyze_attached_dtype_tokens(
    bf16_token_ids: Iterable[int],
    fp32_token_ids: Iterable[int],
) -> dict[str, Any]:
    """Locate the first BF16-attached versus FP32-attached token boundary."""

    bf16 = _token_run(bf16_token_ids, "$.bf16_token_ids")
    fp32 = _token_run(fp32_token_ids, "$.fp32_token_ids")
    first = _first_divergence(bf16, fp32)
    if first is None:
        return {
            "bf16_token_count": len(bf16),
            "fp32_token_count": len(fp32),
            "cross_dtype_identical": True,
            "common_prefix_generated_tokens": len(bf16),
            "first_divergent_token_index": None,
            "bf16_token_id": None,
            "fp32_token_id": None,
            "classification": "cross_dtype_token_identity",
        }

    index, bf16_id, fp32_id = first
    classification = (
        "cross_dtype_token_drift"
        if bf16_id is not None and fp32_id is not None
        else "cross_dtype_termination_drift"
    )
    return {
        "bf16_token_count": len(bf16),
        "fp32_token_count": len(fp32),
        "cross_dtype_identical": False,
        "common_prefix_generated_tokens": index,
        "first_divergent_token_index": index,
        "bf16_token_id": bf16_id,
        "fp32_token_id": fp32_id,
        "classification": classification,
    }


def select_locked_comparison_step(
    token_analysis: dict[str, Any],
    *,
    frozen_boundary_index: int,
) -> dict[str, Any]:
    """Require the fresh paths to reproduce the frozen first token boundary."""

    boundary = _nonnegative_int(
        frozen_boundary_index,
        "$.frozen_boundary_index",
    )
    classification, bf16_count, fp32_count = _validate_token_analysis(
        token_analysis
    )
    if classification != "cross_dtype_token_drift":
        _fail(
            "FROZEN_TOKEN_BOUNDARY_NOT_REPRODUCED",
            "$.token_analysis",
            repr(token_analysis),
        )
    step = _nonnegative_int(
        token_analysis.get("first_divergent_token_index"),
        "$.token_analysis.first_divergent_token_index",
    )
    if step != boundary:
        _fail(
            "FROZEN_TOKEN_BOUNDARY_INDEX_DRIFT",
            "$.token_analysis.first_divergent_token_index",
            repr(step),
        )
    if step >= bf16_count or step >= fp32_count:
        _fail("COMPARISON_STEP_OUT_OF_RANGE", "$.comparison_step", repr(step))
    return {
        "step_index": step,
        "basis": "frozen_first_cross_dtype_generated_token_divergence",
        "shared_generated_prefix_tokens": step,
    }


def classify_attached_dtype_effect(
    token_analysis: dict[str, Any],
    *,
    bf16_repeat_stable: bool,
    fp32_repeat_stable: bool,
    bf16_reference_reproduced: bool,
    fp32_reference_reproduced: bool,
    bf16_emitted_token_id: int,
    fp32_emitted_token_id: int,
    bf16_score_top_token_id: int,
    fp32_score_top_token_id: int,
    bf16_raw_logit_top_token_id: int,
    fp32_raw_logit_top_token_id: int,
) -> str:
    """Classify the locked attached-path dtype boundary without overclaiming."""

    prerequisites = (
        ("BF16_REPEAT_STABILITY_NOT_ESTABLISHED", bf16_repeat_stable),
        ("FP32_REPEAT_STABILITY_NOT_ESTABLISHED", fp32_repeat_stable),
        ("BF16_REFERENCE_NOT_REPRODUCED", bf16_reference_reproduced),
        ("FP32_REFERENCE_NOT_REPRODUCED", fp32_reference_reproduced),
    )
    for code, prerequisite_value in prerequisites:
        if prerequisite_value is not True:
            _fail(code, "$", repr(token_analysis))

    classification, _, _ = _validate_token_analysis(token_analysis)
    if classification != "cross_dtype_token_drift":
        _fail("TOKEN_DRIFT_NOT_ESTABLISHED", "$.token_analysis", repr(token_analysis))
    select_locked_comparison_step(
        token_analysis,
        frozen_boundary_index=FROZEN_BOUNDARY_INDEX,
    )
    if (
        token_analysis.get("bf16_token_count") != FROZEN_TOKEN_COUNT
        or token_analysis.get("fp32_token_count") != FROZEN_TOKEN_COUNT
        or token_analysis.get("bf16_token_id") != FROZEN_BF16_TOKEN_ID
        or token_analysis.get("fp32_token_id") != FROZEN_FP32_TOKEN_ID
    ):
        _fail(
            "FROZEN_TOKEN_REFERENCE_NOT_REPRODUCED",
            "$.token_analysis",
            repr(token_analysis),
        )

    ids = (
        ("$.bf16_emitted_token_id", bf16_emitted_token_id),
        ("$.fp32_emitted_token_id", fp32_emitted_token_id),
        ("$.bf16_score_top_token_id", bf16_score_top_token_id),
        ("$.fp32_score_top_token_id", fp32_score_top_token_id),
        ("$.bf16_raw_logit_top_token_id", bf16_raw_logit_top_token_id),
        ("$.fp32_raw_logit_top_token_id", fp32_raw_logit_top_token_id),
    )
    for path, token_value in ids:
        _token_id(token_value, path)

    if (
        token_analysis.get("bf16_token_id") != bf16_emitted_token_id
        or token_analysis.get("fp32_token_id") != fp32_emitted_token_id
    ):
        _fail(
            "DIVERGENT_EMISSION_MISMATCH",
            "$.token_analysis",
            repr(token_analysis),
        )
    if (
        bf16_score_top_token_id != bf16_emitted_token_id
        or fp32_score_top_token_id != fp32_emitted_token_id
    ):
        _fail(
            "GENERATION_SCORE_ALIGNMENT_FAILURE",
            "$",
            repr(token_analysis),
        )
    if (
        bf16_raw_logit_top_token_id == bf16_emitted_token_id
        and fp32_raw_logit_top_token_id == fp32_emitted_token_id
    ):
        return "deterministic_bf16_attached_vs_fp32_attached_raw_logit_boundary_flip"
    if bf16_raw_logit_top_token_id == fp32_raw_logit_top_token_id:
        return (
            "deterministic_bf16_attached_vs_fp32_attached_"
            "logits_processor_boundary_flip"
        )
    return (
        "deterministic_bf16_attached_vs_fp32_attached_"
        "mixed_raw_logit_and_logits_processor_boundary_flip"
    )


def _validate_token_analysis(value: dict[str, Any]) -> tuple[str, int, int]:
    if not isinstance(value, dict):
        _fail("INVALID_TOKEN_ANALYSIS", "$.token_analysis", repr(value))
    if set(value) != _TOKEN_ANALYSIS_KEYS:
        _fail(
            "INVALID_TOKEN_ANALYSIS_KEYS",
            "$.token_analysis",
            repr(sorted(value)),
        )
    classification = value.get("classification")
    bf16_count = _nonnegative_int(
        value.get("bf16_token_count"),
        "$.token_analysis.bf16_token_count",
    )
    fp32_count = _nonnegative_int(
        value.get("fp32_token_count"),
        "$.token_analysis.fp32_token_count",
    )
    common_prefix = _nonnegative_int(
        value.get("common_prefix_generated_tokens"),
        "$.token_analysis.common_prefix_generated_tokens",
    )
    if classification == "cross_dtype_token_identity":
        if (
            value.get("cross_dtype_identical") is not True
            or bf16_count != fp32_count
            or common_prefix != bf16_count
            or value.get("first_divergent_token_index") is not None
            or value.get("bf16_token_id") is not None
            or value.get("fp32_token_id") is not None
        ):
            _fail("TOKEN_IDENTITY_ANALYSIS_CONFLICT", "$.token_analysis", repr(value))
    elif classification == "cross_dtype_token_drift":
        divergence = _nonnegative_int(
            value.get("first_divergent_token_index"),
            "$.token_analysis.first_divergent_token_index",
        )
        bf16_id = _token_id(
            value.get("bf16_token_id"),
            "$.token_analysis.bf16_token_id",
        )
        fp32_id = _token_id(
            value.get("fp32_token_id"),
            "$.token_analysis.fp32_token_id",
        )
        if (
            value.get("cross_dtype_identical") is not False
            or common_prefix != divergence
            or divergence >= bf16_count
            or divergence >= fp32_count
            or bf16_id == fp32_id
        ):
            _fail("TOKEN_DRIFT_ANALYSIS_CONFLICT", "$.token_analysis", repr(value))
    elif classification == "cross_dtype_termination_drift":
        divergence = _nonnegative_int(
            value.get("first_divergent_token_index"),
            "$.token_analysis.first_divergent_token_index",
        )
        ids = (value.get("bf16_token_id"), value.get("fp32_token_id"))
        if (
            value.get("cross_dtype_identical") is not False
            or common_prefix != divergence
            or divergence != min(bf16_count, fp32_count)
            or sum(token_id is None for token_id in ids) != 1
        ):
            _fail(
                "TOKEN_TERMINATION_ANALYSIS_CONFLICT",
                "$.token_analysis",
                repr(value),
            )
        for index, token_id in enumerate(ids):
            if token_id is not None:
                _token_id(
                    token_id,
                    f"$.token_analysis.termination_token_ids[{index}]",
                )
    else:
        _fail(
            "INVALID_TOKEN_ANALYSIS_CLASSIFICATION",
            "$.token_analysis",
            repr(value),
        )
    return str(classification), bf16_count, fp32_count


def _token_run(value: Iterable[int], path: str) -> list[int]:
    try:
        run = list(value)
    except TypeError:
        _fail("INVALID_TOKEN_RUN", path, repr(value))
    if not run or any(
        not isinstance(token, int)
        or isinstance(token, bool)
        or token < 0
        or token >= VOCABULARY_SIZE
        for token in run
    ):
        _fail("INVALID_TOKEN_RUN", path, repr(run))
    return run


def _token_id(value: object, path: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= VOCABULARY_SIZE
    ):
        _fail("INVALID_TOKEN_ID", path, repr(value))
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("INVALID_NONNEGATIVE_INTEGER", path, repr(value))
    return value


def _sha256(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        _fail("INVALID_SHA256", path, repr(value))
    return value


def _first_divergence(
    bf16: list[int],
    fp32: list[int],
) -> tuple[int, int | None, int | None] | None:
    for index, (left, right) in enumerate(zip(bf16, fp32)):
        if left != right:
            return index, left, right
    if len(bf16) == len(fp32):
        return None
    index = min(len(bf16), len(fp32))
    bf16_id = bf16[index] if index < len(bf16) else None
    fp32_id = fp32[index] if index < len(fp32) else None
    return index, bf16_id, fp32_id


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "ATTACHED_DTYPE_ISOLATION_VERSION",
    "analyze_attached_dtype_tokens",
    "analyze_path_repeat_stability",
    "classify_attached_dtype_effect",
    "select_locked_comparison_step",
]
