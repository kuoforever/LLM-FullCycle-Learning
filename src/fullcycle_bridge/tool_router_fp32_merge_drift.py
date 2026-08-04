"""Pure analysis for the Tool Router FP32 merge drift gate."""

from __future__ import annotations

from typing import Any, Iterable, NoReturn

from .tool_router import ToolRouterValidationError

FP32_MERGE_DRIFT_VERSION = 1


def analyze_path_tokens(
    independent_token_ids: Iterable[int],
    candidate_token_ids: Iterable[int],
) -> dict[str, Any]:
    """Locate the earliest independent-BF16 versus FP32-candidate drift."""

    independent = _token_run(independent_token_ids, "$.independent_token_ids")
    candidate = _token_run(candidate_token_ids, "$.candidate_token_ids")
    first = _first_divergence(independent, candidate)
    if first is None:
        return {
            "independent_token_count": len(independent),
            "candidate_token_count": len(candidate),
            "cross_path_identical": True,
            "common_prefix_generated_tokens": len(independent),
            "first_divergent_token_index": None,
            "independent_token_id": None,
            "candidate_token_id": None,
            "classification": "cross_path_output_identity",
        }

    index, independent_id, candidate_id = first
    classification = (
        "cross_path_token_drift"
        if independent_id is not None and candidate_id is not None
        else "cross_path_termination_drift"
    )
    return {
        "independent_token_count": len(independent),
        "candidate_token_count": len(candidate),
        "cross_path_identical": False,
        "common_prefix_generated_tokens": index,
        "first_divergent_token_index": index,
        "independent_token_id": independent_id,
        "candidate_token_id": candidate_id,
        "classification": classification,
    }


def classify_generation_boundary(
    token_analysis: dict[str, Any],
    *,
    frozen_paths_reproduced: bool,
    independent_score_top_token_id: int,
    candidate_score_top_token_id: int,
    independent_raw_logit_top_token_id: int,
    candidate_raw_logit_top_token_id: int,
) -> str:
    """Classify the exact generate-step boundary without implying root cause."""

    if frozen_paths_reproduced is not True:
        _fail("FROZEN_PATH_REPRODUCTION_NOT_ESTABLISHED", "$", repr(token_analysis))
    if token_analysis.get("classification") != "cross_path_token_drift":
        _fail("TOKEN_DRIFT_NOT_ESTABLISHED", "$", repr(token_analysis))

    independent_token_id = token_analysis.get("independent_token_id")
    candidate_token_id = token_analysis.get("candidate_token_id")
    for path, value in (
        ("$.independent_score_top_token_id", independent_score_top_token_id),
        ("$.candidate_score_top_token_id", candidate_score_top_token_id),
        (
            "$.independent_raw_logit_top_token_id",
            independent_raw_logit_top_token_id,
        ),
        ("$.candidate_raw_logit_top_token_id", candidate_raw_logit_top_token_id),
    ):
        _token_id(value, path)

    if (
        independent_score_top_token_id != independent_token_id
        or candidate_score_top_token_id != candidate_token_id
    ):
        return "generation_score_alignment_failure"
    if (
        independent_raw_logit_top_token_id == independent_token_id
        and candidate_raw_logit_top_token_id == candidate_token_id
    ):
        return (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "raw_logit_boundary_flip"
        )
    if independent_raw_logit_top_token_id == candidate_raw_logit_top_token_id:
        return (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "logits_processor_boundary_flip"
        )
    return (
        "deterministic_bf16_attached_vs_fp32_merged_mixed_logit_score_drift"
    )


def _token_run(value: Iterable[int], path: str) -> list[int]:
    run = list(value)
    if not run or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in run
    ):
        _fail("INVALID_TOKEN_RUN", path, repr(run))
    return run


def _token_id(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("INVALID_TOKEN_ID", path, repr(value))
    return value


def _first_divergence(
    independent: list[int], candidate: list[int]
) -> tuple[int, int | None, int | None] | None:
    for index, (left, right) in enumerate(zip(independent, candidate)):
        if left != right:
            return index, left, right
    if len(independent) == len(candidate):
        return None
    index = min(len(independent), len(candidate))
    independent_id = independent[index] if index < len(independent) else None
    candidate_id = candidate[index] if index < len(candidate) else None
    return index, independent_id, candidate_id


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "FP32_MERGE_DRIFT_VERSION",
    "analyze_path_tokens",
    "classify_generation_boundary",
]
