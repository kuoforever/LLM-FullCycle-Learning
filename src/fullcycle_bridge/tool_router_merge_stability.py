"""Pure analysis for repeated independent and merged Tool Router outputs."""

from __future__ import annotations

from typing import Any, Iterable, NoReturn

from .tool_router import ToolRouterValidationError

MERGE_STABILITY_VERSION = 1


def analyze_token_runs(
    independent_runs: Iterable[Iterable[int]],
    merged_runs: Iterable[Iterable[int]],
) -> dict[str, Any]:
    """Classify within-path repeat stability and the first cross-path drift."""

    independent = [_token_run(run, f"$.independent[{index}]") for index, run in enumerate(independent_runs)]
    merged = [_token_run(run, f"$.merged[{index}]") for index, run in enumerate(merged_runs)]
    if len(independent) < 2 or len(merged) < 2:
        _fail("INSUFFICIENT_REPEATS", "$", f"{len(independent)}+{len(merged)}")
    independent_identical = all(run == independent[0] for run in independent[1:])
    merged_identical = all(run == merged[0] for run in merged[1:])
    if not independent_identical or not merged_identical:
        return {
            "independent_repeats_identical": independent_identical,
            "merged_repeats_identical": merged_identical,
            "cross_path_identical": False,
            "first_divergent_token_index": None,
            "independent_token_id": None,
            "merged_token_id": None,
            "classification": "within_path_nondeterminism",
        }
    first = _first_divergence(independent[0], merged[0])
    if first is None:
        return {
            "independent_repeats_identical": True,
            "merged_repeats_identical": True,
            "cross_path_identical": True,
            "first_divergent_token_index": None,
            "independent_token_id": None,
            "merged_token_id": None,
            "classification": "output_identity_restored",
        }
    index, independent_id, merged_id = first
    return {
        "independent_repeats_identical": True,
        "merged_repeats_identical": True,
        "cross_path_identical": False,
        "first_divergent_token_index": index,
        "independent_token_id": independent_id,
        "merged_token_id": merged_id,
        "classification": "deterministic_cross_path_token_drift",
    }


def classify_logit_divergence(
    token_analysis: dict[str, Any],
    independent_top_token_id: int,
    merged_top_token_id: int,
) -> str:
    """Refine deterministic token drift using logits at the common prefix."""

    if token_analysis.get("classification") != "deterministic_cross_path_token_drift":
        _fail("TOKEN_DRIFT_NOT_ESTABLISHED", "$", repr(token_analysis))
    if (
        independent_top_token_id == token_analysis.get("independent_token_id")
        and merged_top_token_id == token_analysis.get("merged_token_id")
        and independent_top_token_id != merged_top_token_id
    ):
        return "deterministic_bf16_merge_logit_boundary_flip"
    return "deterministic_bf16_merge_sequence_drift"


def _token_run(value: Iterable[int], path: str) -> list[int]:
    run = list(value)
    if not run or any(not isinstance(token, int) or token < 0 for token in run):
        _fail("INVALID_TOKEN_RUN", path, repr(run))
    return run


def _first_divergence(
    independent: list[int], merged: list[int]
) -> tuple[int, int | None, int | None] | None:
    for index, (left, right) in enumerate(zip(independent, merged)):
        if left != right:
            return index, left, right
    if len(independent) == len(merged):
        return None
    index = min(len(independent), len(merged))
    left_token: int | None = (
        independent[index] if index < len(independent) else None
    )
    right_token: int | None = merged[index] if index < len(merged) else None
    return index, left_token, right_token


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "MERGE_STABILITY_VERSION",
    "analyze_token_runs",
    "classify_logit_divergence",
]
