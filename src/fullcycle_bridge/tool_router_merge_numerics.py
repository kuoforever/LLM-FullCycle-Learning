"""Pure classification for Tool Router BF16 merge-numerics evidence."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, NoReturn

from .tool_router import ToolRouterValidationError

MERGE_NUMERICS_VERSION = 1


def analyze_module_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Locate the first unequal module output in execution order."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(comparisons):
        path = f"$.module_comparisons[{index}]"
        name = item.get("name")
        equal = item.get("equal")
        different_elements = item.get("different_elements")
        max_abs_delta = item.get("max_abs_delta")
        mean_abs_delta = item.get("mean_abs_delta")
        if not isinstance(name, str) or not name or name in seen:
            _fail("INVALID_MODULE_NAME", f"{path}.name", repr(name))
        if not isinstance(equal, bool):
            _fail("INVALID_EQUAL_FLAG", f"{path}.equal", repr(equal))
        if not isinstance(different_elements, int) or different_elements < 0:
            _fail(
                "INVALID_DIFFERENT_ELEMENT_COUNT",
                f"{path}.different_elements",
                repr(different_elements),
            )
        if not _nonnegative_number(max_abs_delta):
            _fail("INVALID_MAX_DELTA", f"{path}.max_abs_delta", repr(max_abs_delta))
        if not _nonnegative_number(mean_abs_delta):
            _fail(
                "INVALID_MEAN_DELTA",
                f"{path}.mean_abs_delta",
                repr(mean_abs_delta),
            )
        if equal != (different_elements == 0):
            _fail("INCONSISTENT_COMPARISON", path, repr(dict(item)))
        seen.add(name)
        normalized.append(dict(item))
    if not normalized:
        _fail("EMPTY_MODULE_COMPARISONS", "$.module_comparisons", "[]")
    first_index = next(
        (index for index, item in enumerate(normalized) if not item["equal"]),
        None,
    )
    if first_index is None:
        return {
            "first_divergent_module_index": None,
            "first_divergent_module": None,
            "preceding_modules_identical": True,
            "classification": "module_output_identity",
        }
    return {
        "first_divergent_module_index": first_index,
        "first_divergent_module": normalized[first_index]["name"],
        "preceding_modules_identical": all(
            item["equal"] for item in normalized[:first_index]
        ),
        "classification": "deterministic_module_output_divergence",
    }


def classify_merge_numerics(
    module_analysis: Mapping[str, Any],
    rounding: Mapping[str, Any],
) -> str:
    """Bind the first module divergence to verified BF16 merge rounding."""

    if module_analysis.get("classification") != "deterministic_module_output_divergence":
        _fail("MODULE_DIVERGENCE_NOT_ESTABLISHED", "$.module_analysis", repr(module_analysis))
    if rounding.get("actual_merged_mismatched_weights") != 0:
        return "safe_merge_implementation_mismatch"
    rounded_away = rounding.get("ideal_nonzero_updates_rounded_to_base")
    if not isinstance(rounded_away, int) or rounded_away <= 0:
        return "module_divergence_without_quantified_rounding"
    return "bf16_safe_merge_weight_rounding"


def _nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "MERGE_NUMERICS_VERSION",
    "analyze_module_comparisons",
    "classify_merge_numerics",
]
