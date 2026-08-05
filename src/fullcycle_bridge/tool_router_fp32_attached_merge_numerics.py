"""Pure contracts for FP32 attached-versus-merged module numerics evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError

FP32_ATTACHED_MERGE_NUMERICS_VERSION = 1
FIRST_REGISTERED_BOUNDARY = "model.layers.0.self_attn.q_proj"
COMMON_OUTPUT_STAGES = (
    "model.embed_tokens",
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.0",
    "model.norm",
    "lm_head",
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMPARISON_KEYS = frozenset(
    {
        "name",
        "shape",
        "dtype",
        "elements",
        "numerically_equal",
        "bitwise_equal",
        "different_elements",
        "bitwise_different_elements",
        "first_different_flat_index",
        "max_abs_delta_flat_index",
        "left_value_at_first_difference",
        "right_value_at_first_difference",
        "left_value_at_max_abs_delta",
        "right_value_at_max_abs_delta",
        "max_abs_delta",
        "mean_abs_delta",
        "root_mean_square_delta",
        "left_tensor_id",
        "right_tensor_id",
        "left_raw_sha256",
        "right_raw_sha256",
        "left_canonical_sha256",
        "right_canonical_sha256",
    }
)
_OPERATION_PAIRS = {
    "q_proj_input_identity": (
        "fp32-attached-numerics-r1|common|model.layers.0.self_attn.q_proj|input",
        "fp32-safe-merged-numerics-r1|common|model.layers.0.self_attn.q_proj|input",
    ),
    "attached_output_reconstruction": (
        "fp32-attached-numerics-r1|common|model.layers.0.self_attn.q_proj|output",
        "fp32-attached-numerics-r1|diagnostic|q_proj|base_plus_factorized",
    ),
    "attached_dropout_identity": (
        "fp32-attached-numerics-r1|common|model.layers.0.self_attn.q_proj|input",
        "fp32-attached-numerics-r1|diagnostic|q_proj|dropout_output",
    ),
    "merged_output_reconstruction": (
        "fp32-safe-merged-numerics-r1|common|model.layers.0.self_attn.q_proj|output",
        "fp32-safe-merged-numerics-r1|diagnostic|q_proj|recomputed",
    ),
    "expected_materialized_vs_merged_actual": (
        "fp32-attached-numerics-r1|diagnostic|q_proj|expected_materialized_linear",
        "fp32-safe-merged-numerics-r1|common|model.layers.0.self_attn.q_proj|output",
    ),
    "factorized_lora_vs_delta_weight_linear": (
        "fp32-attached-numerics-r1|diagnostic|q_proj|factorized_scaled",
        "fp32-attached-numerics-r1|diagnostic|q_proj|delta_weight_linear",
    ),
    "attached_factorized_output_vs_split_delta_output": (
        "fp32-attached-numerics-r1|diagnostic|q_proj|base_plus_factorized",
        (
            "fp32-attached-numerics-r1|diagnostic|q_proj|"
            "base_plus_delta_weight_linear"
        ),
    ),
    "split_base_plus_delta_vs_materialized_weight_linear": (
        "fp32-attached-numerics-r1|diagnostic|q_proj|base_plus_delta_weight_linear",
        "fp32-attached-numerics-r1|diagnostic|q_proj|expected_materialized_linear",
    ),
    "attached_output_vs_merged_output": (
        "fp32-attached-numerics-r1|common|model.layers.0.self_attn.q_proj|output",
        "fp32-safe-merged-numerics-r1|common|model.layers.0.self_attn.q_proj|output",
    ),
}
_WEIGHT_AUDIT_KEYS = frozenset(
    {
        "name",
        "shape",
        "dtype",
        "elements",
        "base_weight_sha256",
        "delta_weight_sha256",
        "expected_merged_weight_sha256",
        "actual_merged_weight_sha256",
        "expected_actual_equal",
        "actual_merged_mismatched_weights",
        "ideal_nonzero_updates",
        "effective_changed_weights",
        "ideal_nonzero_updates_rounded_to_base",
        "max_abs_materialization_error",
        "mean_abs_materialization_error",
        "bias_present",
        "bias_elements",
        "bias_mismatched_elements",
        "tensor_ids",
    }
)


def analyze_module_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Locate the first unequal paired common-module output."""

    normalized = _normalize_comparisons(comparisons, "$.module_comparisons")
    first_index = next(
        (
            index
            for index, item in enumerate(normalized)
            if not item["numerically_equal"]
        ),
        None,
    )
    if first_index is None:
        return {
            "module_count": len(normalized),
            "first_divergent_module_index": None,
            "first_divergent_module": None,
            "preceding_modules_identical": True,
            "classification": "paired_common_module_output_identity",
        }
    return {
        "module_count": len(normalized),
        "first_divergent_module_index": first_index,
        "first_divergent_module": normalized[first_index]["name"],
        "preceding_modules_identical": all(
            item["numerically_equal"] for item in normalized[:first_index]
        ),
        "classification": "paired_common_module_output_divergence",
    }


def classify_operation_order(
    module_comparisons: Iterable[Mapping[str, Any]],
    operation_comparisons: Iterable[Mapping[str, Any]],
    weight_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the registered q-projection execution-form boundary."""

    modules = _normalize_comparisons(
        module_comparisons,
        "$.module_comparisons",
    )
    if tuple(item["name"] for item in modules) != COMMON_OUTPUT_STAGES:
        _fail(
            "INVALID_COMMON_MODULE_SEQUENCE",
            "$.module_comparisons",
            repr([item["name"] for item in modules]),
        )
    for stage, item in zip(COMMON_OUTPUT_STAGES, modules, strict=True):
        expected_pair = (
            f"fp32-attached-numerics-r1|common|{stage}|output",
            f"fp32-safe-merged-numerics-r1|common|{stage}|output",
        )
        if (item["left_tensor_id"], item["right_tensor_id"]) != expected_pair:
            _fail(
                "INVALID_COMMON_MODULE_TENSOR_PAIR",
                f"$.module_comparisons.{stage}",
                repr((item["left_tensor_id"], item["right_tensor_id"])),
            )
    analysis = analyze_module_comparisons(modules)
    operations = _normalize_comparisons(
        operation_comparisons,
        "$.operation_comparisons",
    )
    by_name = {item["name"]: item for item in operations}
    if set(by_name) != set(_OPERATION_PAIRS):
        _fail(
            "INVALID_OPERATION_COMPARISON_SET",
            "$.operation_comparisons",
            repr(sorted(by_name)),
        )
    for name, expected_pair in _OPERATION_PAIRS.items():
        item = by_name[name]
        actual_pair = (item["left_tensor_id"], item["right_tensor_id"])
        if actual_pair != expected_pair:
            _fail(
                "INVALID_OPERATION_TENSOR_PAIR",
                f"$.operation_comparisons.{name}",
                repr(actual_pair),
            )
    weight = _normalize_weight_audit(weight_audit)

    if analysis["classification"] != "paired_common_module_output_divergence":
        _fail(
            "MODULE_DIVERGENCE_NOT_ESTABLISHED",
            "$.module_analysis",
            repr(analysis),
        )
    if analysis["first_divergent_module"] != FIRST_REGISTERED_BOUNDARY:
        _fail(
            "FIRST_DIVERGENCE_OUTSIDE_REGISTERED_BOUNDARY",
            "$.module_analysis.first_divergent_module",
            repr(analysis["first_divergent_module"]),
        )
    if analysis["preceding_modules_identical"] is not True:
        _fail(
            "PRECEDING_MODULE_DIVERGENCE",
            "$.module_analysis.preceding_modules_identical",
            repr(analysis),
        )

    required_identity = (
        "q_proj_input_identity",
        "attached_dropout_identity",
        "attached_output_reconstruction",
        "merged_output_reconstruction",
        "expected_materialized_vs_merged_actual",
    )
    for name in required_identity:
        if (
            by_name[name]["numerically_equal"] is not True
            or by_name[name]["bitwise_equal"] is not True
        ):
            _fail(
                "REQUIRED_OPERATION_IDENTITY_MISSING",
                f"$.operation_comparisons.{name}",
                repr(by_name[name]),
            )
    actual = by_name["attached_output_vs_merged_output"]
    if (
        actual["numerically_equal"] is not False
        or actual["max_abs_delta"] <= 0
    ):
        _fail(
            "ACTUAL_Q_PROJ_DIVERGENCE_MISSING",
            "$.operation_comparisons.attached_output_vs_merged_output",
            repr(actual),
        )

    first_module = modules[analysis["first_divergent_module_index"]]
    if (
        first_module["left_canonical_sha256"]
        != actual["left_canonical_sha256"]
        or first_module["right_canonical_sha256"]
        != actual["right_canonical_sha256"]
        or first_module["shape"] != actual["shape"]
    ):
        _fail(
            "Q_PROJ_COMPARISON_LINKAGE_MISMATCH",
            "$.operation_comparisons.attached_output_vs_merged_output",
            repr(actual),
        )

    if (
        weight["expected_actual_equal"] is not True
        or weight["actual_merged_mismatched_weights"] != 0
        or weight["expected_merged_weight_sha256"]
        != weight["actual_merged_weight_sha256"]
    ):
        _fail(
            "SAFE_MERGE_WEIGHT_REPRODUCTION_FAILED",
            "$.weight_materialization",
            repr(weight),
        )

    factorized_term_drift = not by_name[
        "factorized_lora_vs_delta_weight_linear"
    ]["numerically_equal"]
    factorized_drift = not by_name[
        "attached_factorized_output_vs_split_delta_output"
    ]["numerically_equal"]
    if factorized_drift and not factorized_term_drift:
        _fail(
            "INCONSISTENT_FACTORIZED_EXECUTION_FORM_DRIFT",
            "$.operation_comparisons",
            repr(by_name),
        )
    materialized_linear_drift = not by_name[
        "split_base_plus_delta_vs_materialized_weight_linear"
    ]["numerically_equal"]
    if not factorized_drift and not materialized_linear_drift:
        _fail(
            "REGISTERED_OPERATIONS_DO_NOT_EXPLAIN_OUTPUT_DRIFT",
            "$.operation_comparisons",
            repr(by_name),
        )
    if factorized_drift and materialized_linear_drift:
        classification = (
            "deterministic_fp32_factorized_lora_and_materialized_linear_"
            "execution_form_drift"
        )
    elif factorized_drift:
        classification = (
            "deterministic_fp32_factorized_lora_execution_form_drift"
        )
    else:
        classification = (
            "deterministic_fp32_split_sum_vs_materialized_linear_"
            "execution_form_drift"
        )
    return {
        "classification": classification,
        "first_divergent_module": FIRST_REGISTERED_BOUNDARY,
        "q_proj_input_identity": True,
        "attached_output_reproduced": True,
        "merged_output_reproduced": True,
        "safe_merge_weight_reproduced": True,
        "factorized_lora_term_vs_delta_weight_linear_drift": (
            factorized_term_drift
        ),
        "factorized_output_vs_split_delta_output_drift": factorized_drift,
        "split_sum_vs_materialized_weight_linear_drift": (
            materialized_linear_drift
        ),
        "registered_execution_form_boundary_quantified": True,
    }


def _normalize_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
    path: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(comparisons):
        item_path = f"{path}[{index}]"
        if set(item) != set(_COMPARISON_KEYS):
            _fail(
                "INVALID_COMPARISON_FIELDS",
                item_path,
                repr(sorted(item)),
            )
        name = item["name"]
        if not isinstance(name, str) or not name or name in seen_names:
            _fail("INVALID_COMPARISON_NAME", f"{item_path}.name", repr(name))
        shape = item["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in shape
            )
        ):
            _fail("INVALID_COMPARISON_SHAPE", f"{item_path}.shape", repr(shape))
        elements = item["elements"]
        if (
            not isinstance(elements, int)
            or isinstance(elements, bool)
            or elements != math.prod(shape)
        ):
            _fail(
                "INVALID_COMPARISON_ELEMENT_COUNT",
                f"{item_path}.elements",
                repr(elements),
            )
        if item["dtype"] != "float32":
            _fail(
                "INVALID_COMPARISON_DTYPE",
                f"{item_path}.dtype",
                repr(item["dtype"]),
            )
        numerically_equal = item["numerically_equal"]
        bitwise_equal = item["bitwise_equal"]
        different = item["different_elements"]
        bitwise_different = item["bitwise_different_elements"]
        if not isinstance(numerically_equal, bool):
            _fail(
                "INVALID_NUMERICAL_EQUAL_FLAG",
                f"{item_path}.numerically_equal",
                repr(numerically_equal),
            )
        if not isinstance(bitwise_equal, bool):
            _fail(
                "INVALID_BITWISE_EQUAL_FLAG",
                f"{item_path}.bitwise_equal",
                repr(bitwise_equal),
            )
        if (
            not isinstance(different, int)
            or isinstance(different, bool)
            or different < 0
            or different > elements
        ):
            _fail(
                "INVALID_DIFFERENT_ELEMENT_COUNT",
                f"{item_path}.different_elements",
                repr(different),
            )
        if (
            not isinstance(bitwise_different, int)
            or isinstance(bitwise_different, bool)
            or bitwise_different < different
            or bitwise_different > elements
        ):
            _fail(
                "INVALID_BITWISE_DIFFERENT_ELEMENT_COUNT",
                f"{item_path}.bitwise_different_elements",
                repr(bitwise_different),
            )
        deltas = [
            item["max_abs_delta"],
            item["mean_abs_delta"],
            item["root_mean_square_delta"],
        ]
        if any(not _nonnegative_finite_number(value) for value in deltas):
            _fail("INVALID_COMPARISON_DELTA", item_path, repr(deltas))
        maximum, mean, rms = deltas
        tolerance = max(1.0, maximum) * 1e-15
        if mean > rms + tolerance or rms > maximum + tolerance:
            _fail("INCONSISTENT_COMPARISON_DELTAS", item_path, repr(deltas))
        first_index = item["first_different_flat_index"]
        max_index = item["max_abs_delta_flat_index"]
        witness_keys = (
            "left_value_at_first_difference",
            "right_value_at_first_difference",
            "left_value_at_max_abs_delta",
            "right_value_at_max_abs_delta",
        )
        for key in ("left_tensor_id", "right_tensor_id"):
            if not isinstance(item[key], str) or not item[key]:
                _fail("INVALID_TENSOR_ID", f"{item_path}.{key}", repr(item[key]))
        for key in (
            "left_raw_sha256",
            "right_raw_sha256",
            "left_canonical_sha256",
            "right_canonical_sha256",
        ):
            if not isinstance(item[key], str) or _DIGEST.fullmatch(item[key]) is None:
                _fail("INVALID_TENSOR_DIGEST", f"{item_path}.{key}", repr(item[key]))
        raw_digests_equal = item["left_raw_sha256"] == item["right_raw_sha256"]
        canonical_digests_equal = (
            item["left_canonical_sha256"] == item["right_canonical_sha256"]
        )
        if bitwise_equal != (bitwise_different == 0 and raw_digests_equal):
            _fail("INCONSISTENT_BITWISE_EQUALITY", item_path, repr(dict(item)))
        if bitwise_equal and not numerically_equal:
            _fail("BITWISE_IDENTITY_WITH_NUMERICAL_DRIFT", item_path, repr(dict(item)))
        if numerically_equal:
            if (
                different != 0
                or any(value != 0 for value in deltas)
                or not canonical_digests_equal
                or first_index is not None
                or max_index is not None
                or any(item[key] is not None for key in witness_keys)
            ):
                _fail(
                    "INCONSISTENT_EQUAL_COMPARISON",
                    item_path,
                    repr(dict(item)),
                )
        else:
            if (
                different == 0
                or canonical_digests_equal
                or any(value <= 0 for value in deltas)
                or not _valid_flat_index(first_index, elements)
                or not _valid_flat_index(max_index, elements)
                or any(
                    not _finite_number(item[key])
                    for key in witness_keys
                )
                or item["left_value_at_first_difference"]
                == item["right_value_at_first_difference"]
                or abs(
                    item["left_value_at_max_abs_delta"]
                    - item["right_value_at_max_abs_delta"]
                )
                != maximum
            ):
                _fail(
                    "INCONSISTENT_UNEQUAL_COMPARISON",
                    item_path,
                    repr(dict(item)),
                )
        seen_names.add(name)
        normalized.append(dict(item))
    if not normalized:
        _fail("EMPTY_COMPARISONS", path, "[]")
    return normalized


def _normalize_weight_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    path = "$.weight_materialization"
    if set(value) != set(_WEIGHT_AUDIT_KEYS):
        _fail("INVALID_WEIGHT_AUDIT_FIELDS", path, repr(sorted(value)))
    if value["name"] != FIRST_REGISTERED_BOUNDARY or value["dtype"] != "float32":
        _fail("INVALID_WEIGHT_AUDIT_TARGET", path, repr(dict(value)))
    shape = value["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in shape
        )
    ):
        _fail("INVALID_WEIGHT_SHAPE", f"{path}.shape", repr(shape))
    elements = value["elements"]
    if (
        not isinstance(elements, int)
        or isinstance(elements, bool)
        or elements != math.prod(shape)
    ):
        _fail("INVALID_WEIGHT_ELEMENTS", f"{path}.elements", repr(elements))
    for key in (
        "base_weight_sha256",
        "delta_weight_sha256",
        "expected_merged_weight_sha256",
        "actual_merged_weight_sha256",
    ):
        if not isinstance(value[key], str) or _DIGEST.fullmatch(value[key]) is None:
            _fail("INVALID_WEIGHT_DIGEST", f"{path}.{key}", repr(value[key]))
    if not isinstance(value["expected_actual_equal"], bool):
        _fail(
            "INVALID_WEIGHT_EQUAL_FLAG",
            f"{path}.expected_actual_equal",
            repr(value["expected_actual_equal"]),
        )
    count_keys = (
        "actual_merged_mismatched_weights",
        "ideal_nonzero_updates",
        "effective_changed_weights",
        "ideal_nonzero_updates_rounded_to_base",
    )
    for key in count_keys:
        count = value[key]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count > elements
        ):
            _fail("INVALID_WEIGHT_COUNT", f"{path}.{key}", repr(count))
    if (
        value["effective_changed_weights"]
        + value["ideal_nonzero_updates_rounded_to_base"]
        != value["ideal_nonzero_updates"]
    ):
        _fail("INCONSISTENT_WEIGHT_CHANGE_COUNTS", path, repr(dict(value)))
    for key in ("max_abs_materialization_error", "mean_abs_materialization_error"):
        if not _nonnegative_finite_number(value[key]):
            _fail("INVALID_MATERIALIZATION_ERROR", f"{path}.{key}", repr(value[key]))
    if value["mean_abs_materialization_error"] > value["max_abs_materialization_error"]:
        _fail("INCONSISTENT_MATERIALIZATION_ERROR", path, repr(dict(value)))
    expected_equal = (
        value["actual_merged_mismatched_weights"] == 0
        and value["expected_merged_weight_sha256"]
        == value["actual_merged_weight_sha256"]
    )
    if value["expected_actual_equal"] != expected_equal:
        _fail("INCONSISTENT_WEIGHT_EQUALITY", path, repr(dict(value)))
    bias_present = value["bias_present"]
    bias_elements = value["bias_elements"]
    bias_mismatched = value["bias_mismatched_elements"]
    if (
        not isinstance(bias_present, bool)
        or not isinstance(bias_elements, int)
        or isinstance(bias_elements, bool)
        or bias_elements < 0
        or not isinstance(bias_mismatched, int)
        or isinstance(bias_mismatched, bool)
        or bias_mismatched < 0
        or bias_mismatched > bias_elements
        or bias_present != (bias_elements > 0)
        or bias_mismatched != 0
    ):
        _fail("INVALID_WEIGHT_BIAS_AUDIT", path, repr(dict(value)))
    tensor_ids = value["tensor_ids"]
    expected_tensor_keys = {
        "base_weight",
        "delta_weight",
        "expected_merged_weight",
        "actual_merged_weight",
        "attached_bias",
        "merged_bias",
    }
    if (
        not isinstance(tensor_ids, dict)
        or set(tensor_ids) != expected_tensor_keys
        or any(
            (not isinstance(tensor_ids[key], str) or not tensor_ids[key])
            for key in expected_tensor_keys - {"attached_bias", "merged_bias"}
        )
        or (
            bias_present
            and any(
                not isinstance(tensor_ids[key], str) or not tensor_ids[key]
                for key in ("attached_bias", "merged_bias")
            )
        )
        or (
            not bias_present
            and any(
                tensor_ids[key] is not None
                for key in ("attached_bias", "merged_bias")
            )
        )
    ):
        _fail("INVALID_WEIGHT_TENSOR_IDS", path, repr(tensor_ids))
    return dict(value)


def _nonnegative_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_flat_index(value: object, elements: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < elements
    )


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "COMMON_OUTPUT_STAGES",
    "FIRST_REGISTERED_BOUNDARY",
    "FP32_ATTACHED_MERGE_NUMERICS_VERSION",
    "analyze_module_comparisons",
    "classify_operation_order",
]
