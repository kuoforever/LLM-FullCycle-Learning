"""Pure contracts for attached BF16-versus-FP32 module numerics evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError

ATTACHED_DTYPE_NUMERICS_VERSION = 1
HIDDEN_LAYER_COUNT = 28
CLASSIFICATION = (
    "deterministic_attached_bf16_vs_fp32_registered_module_output_drift_"
    "reaching_lm_head"
)

LAYER_ZERO_DETAILED_OUTPUT_STAGES = (
    "model.layers.0.input_layernorm",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
)
REGISTERED_OUTPUT_STAGES = (
    "model.embed_tokens",
    *LAYER_ZERO_DETAILED_OUTPUT_STAGES,
    *(f"model.layers.{index}" for index in range(HIDDEN_LAYER_COUNT)),
    "model.norm",
    "lm_head",
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMPARISON_KEYS = frozenset(
    {
        "name",
        "shape",
        "elements",
        "bf16_native_dtype",
        "fp32_native_dtype",
        "comparison_dtype",
        "bf16_float32_sha256",
        "fp32_float32_sha256",
        "canonical_values_equal",
        "different_elements",
        "first_different_flat_index",
        "max_abs_delta_flat_index",
        "bf16_value_at_first_difference",
        "fp32_value_at_first_difference",
        "bf16_value_at_max_abs_delta",
        "fp32_value_at_max_abs_delta",
        "max_abs_delta",
        "mean_abs_delta",
        "root_mean_square_delta",
        "sum_abs_delta",
        "sum_squared_delta",
        "different_fraction",
        "bf16_root_mean_square",
        "fp32_root_mean_square",
        "normalized_root_mean_square_delta",
        "root_mean_square_delta_ratio_to_first_registered_difference",
    }
)


def analyze_registered_module_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Locate and quantify the first unequal registered module output."""

    normalized = _normalize_comparisons(comparisons)
    names = tuple(item["name"] for item in normalized)
    if names != REGISTERED_OUTPUT_STAGES:
        _fail(
            "INVALID_REGISTERED_MODULE_SEQUENCE",
            "$.module_comparisons",
            repr(names),
        )

    first_index = next(
        (
            index
            for index, item in enumerate(normalized)
            if item["canonical_values_equal"] is False
        ),
        None,
    )
    if first_index is None:
        _fail(
            "REGISTERED_MODULE_DIFFERENCE_MISSING",
            "$.module_comparisons",
            "all registered outputs are numerically equal",
        )

    first = normalized[first_index]
    first_rms = first["root_mean_square_delta"]
    if first_rms <= 0:
        _fail(
            "INVALID_FIRST_REGISTERED_DELTA",
            f"$.module_comparisons[{first_index}]",
            repr(first),
        )
    for index, item in enumerate(normalized):
        expected_ratio = item["root_mean_square_delta"] / first_rms
        if not _close(
            item[
                "root_mean_square_delta_ratio_to_first_registered_difference"
            ],
            expected_ratio,
        ):
            _fail(
                "INVALID_PROPAGATION_RATIO",
                (
                    f"$.module_comparisons[{index}]."
                    "root_mean_square_delta_ratio_to_first_registered_difference"
                ),
                repr(item),
            )

    lm_head = normalized[-1]
    if (
        lm_head["name"] != "lm_head"
        or lm_head["canonical_values_equal"] is not False
    ):
        _fail(
            "LM_HEAD_PROPAGATION_MISSING",
            f"$.module_comparisons[{len(normalized) - 1}]",
            repr(lm_head),
        )
    downstream = normalized[first_index + 1 :]
    downstream_unequal = sum(
        item["canonical_values_equal"] is False for item in downstream
    )
    return {
        "registered_module_count": len(normalized),
        "first_unequal_module_index": first_index,
        "first_unequal_module": first["name"],
        "preceding_registered_outputs_identical": all(
            item["canonical_values_equal"] is True
            for item in normalized[:first_index]
        ),
        "unequal_registered_module_count": sum(
            item["canonical_values_equal"] is False for item in normalized
        ),
        "registered_downstream_module_count": len(downstream),
        "registered_downstream_unequal_module_count": downstream_unequal,
        "first_root_mean_square_delta": first_rms,
        "registered_lm_head_difference_observed": True,
        "lm_head_different_elements": lm_head["different_elements"],
        "lm_head_max_abs_delta": lm_head["max_abs_delta"],
        "lm_head_mean_abs_delta": lm_head["mean_abs_delta"],
        "lm_head_root_mean_square_delta": lm_head["root_mean_square_delta"],
        "lm_head_root_mean_square_delta_ratio_to_first": lm_head[
            "root_mean_square_delta_ratio_to_first_registered_difference"
        ],
        "classification": CLASSIFICATION,
    }


def classify_attached_dtype_numerics(
    comparisons: Iterable[Mapping[str, Any]],
    *,
    bf16_repeat_stable: bool,
    fp32_repeat_stable: bool,
    bf16_reference_reproduced: bool,
    fp32_reference_reproduced: bool,
    capture_plan_executed: bool,
    target_forward_aligned: bool,
    lm_head_raw_logit_linked: bool,
    attached_execution_form_fixed: bool,
    source_inputs_unchanged: bool,
    module_tensor_payload_absent: bool,
) -> dict[str, Any]:
    """Fail closed unless the registered attached-dtype protocol is complete."""

    requirements = {
        "bf16_repeat_stable": bf16_repeat_stable,
        "fp32_repeat_stable": fp32_repeat_stable,
        "bf16_reference_reproduced": bf16_reference_reproduced,
        "fp32_reference_reproduced": fp32_reference_reproduced,
        "capture_plan_executed": capture_plan_executed,
        "target_forward_aligned": target_forward_aligned,
        "lm_head_raw_logit_linked": lm_head_raw_logit_linked,
        "attached_execution_form_fixed": attached_execution_form_fixed,
        "source_inputs_unchanged": source_inputs_unchanged,
        "module_tensor_payload_absent": module_tensor_payload_absent,
    }
    for name, passed in requirements.items():
        if passed is not True:
            _fail(
                "ATTACHED_DTYPE_NUMERICS_PROTOCOL_FAILED",
                f"$.requirements.{name}",
                repr(passed),
            )
    analysis = analyze_registered_module_comparisons(comparisons)
    if analysis["preceding_registered_outputs_identical"] is not True:
        _fail(
            "PRECEDING_REGISTERED_OUTPUT_DIFFERENCE",
            "$.module_analysis.preceding_registered_outputs_identical",
            repr(analysis),
        )
    return analysis


def _normalize_comparisons(
    comparisons: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, source in enumerate(comparisons):
        path = f"$.module_comparisons[{index}]"
        if not isinstance(source, Mapping):
            _fail(
                "INVALID_MODULE_COMPARISON",
                path,
                repr(type(source)),
            )
        if set(source) != set(_COMPARISON_KEYS):
            _fail(
                "INVALID_MODULE_COMPARISON_FIELDS",
                path,
                repr(sorted(source)),
            )
        item = dict(source)
        name = item["name"]
        if not isinstance(name, str) or not name or name in seen_names:
            _fail("INVALID_MODULE_NAME", f"{path}.name", repr(name))
        seen_names.add(name)

        shape = item["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(not _positive_int(value) for value in shape)
        ):
            _fail("INVALID_MODULE_SHAPE", f"{path}.shape", repr(shape))
        elements = item["elements"]
        if not _positive_int(elements) or elements != math.prod(shape):
            _fail("INVALID_MODULE_ELEMENTS", f"{path}.elements", repr(elements))
        if item["bf16_native_dtype"] != "bfloat16":
            _fail(
                "INVALID_BF16_NATIVE_DTYPE",
                f"{path}.bf16_native_dtype",
                repr(item["bf16_native_dtype"]),
            )
        if item["fp32_native_dtype"] != "float32":
            _fail(
                "INVALID_FP32_NATIVE_DTYPE",
                f"{path}.fp32_native_dtype",
                repr(item["fp32_native_dtype"]),
            )
        if item["comparison_dtype"] != "float32":
            _fail(
                "INVALID_COMPARISON_DTYPE",
                f"{path}.comparison_dtype",
                repr(item["comparison_dtype"]),
            )
        for key in ("bf16_float32_sha256", "fp32_float32_sha256"):
            value = item[key]
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                _fail("INVALID_MODULE_DIGEST", f"{path}.{key}", repr(value))
        if not isinstance(item["canonical_values_equal"], bool):
            _fail(
                "INVALID_COMPARISON_FLAG",
                f"{path}.canonical_values_equal",
                repr(item["canonical_values_equal"]),
            )
        different_elements = item["different_elements"]
        if not _nonnegative_int(different_elements) or different_elements > elements:
            _fail(
                "INVALID_DIFFERENCE_COUNT",
                f"{path}.different_elements",
                repr(different_elements),
            )

        for key in (
            "max_abs_delta",
            "mean_abs_delta",
            "root_mean_square_delta",
            "sum_abs_delta",
            "sum_squared_delta",
            "different_fraction",
            "bf16_root_mean_square",
            "fp32_root_mean_square",
            "normalized_root_mean_square_delta",
            "root_mean_square_delta_ratio_to_first_registered_difference",
        ):
            if not _nonnegative_finite(item[key]):
                _fail("INVALID_COMPARISON_STATISTIC", f"{path}.{key}", repr(item[key]))

        different = item["different_elements"]
        canonical_values_equal = item["canonical_values_equal"]
        if canonical_values_equal is not (different == 0):
            _fail("CANONICAL_EQUALITY_MISMATCH", path, repr(item))
        if not _close(item["different_fraction"], different / elements):
            _fail("INVALID_DIFFERENT_FRACTION", path, repr(item))
        if not _close(item["mean_abs_delta"], item["sum_abs_delta"] / elements):
            _fail("INVALID_MEAN_ABS_DELTA", path, repr(item))
        expected_rms = math.sqrt(item["sum_squared_delta"] / elements)
        if not _close(item["root_mean_square_delta"], expected_rms):
            _fail("INVALID_ROOT_MEAN_SQUARE_DELTA", path, repr(item))
        denominator = max(
            item["bf16_root_mean_square"], item["fp32_root_mean_square"]
        )
        expected_normalized = (
            item["root_mean_square_delta"] / denominator
            if denominator > 0
            else 0.0
        )
        if not _close(item["normalized_root_mean_square_delta"], expected_normalized):
            _fail("INVALID_NORMALIZED_DELTA", path, repr(item))

        nullable_indices = (
            "first_different_flat_index",
            "max_abs_delta_flat_index",
        )
        nullable_values = (
            "bf16_value_at_first_difference",
            "fp32_value_at_first_difference",
            "bf16_value_at_max_abs_delta",
            "fp32_value_at_max_abs_delta",
        )
        if different == 0:
            if any(item[key] is not None for key in (*nullable_indices, *nullable_values)):
                _fail("IDENTICAL_COMPARISON_HAS_WITNESS", path, repr(item))
            if any(
                item[key] != 0.0
                for key in (
                    "max_abs_delta",
                    "mean_abs_delta",
                    "root_mean_square_delta",
                    "sum_abs_delta",
                    "sum_squared_delta",
                    "normalized_root_mean_square_delta",
                )
            ):
                _fail("IDENTICAL_COMPARISON_HAS_DELTA", path, repr(item))
            if item["bf16_float32_sha256"] != item["fp32_float32_sha256"]:
                _fail("IDENTICAL_COMPARISON_DIGEST_MISMATCH", path, repr(item))
            if not _close(
                item["bf16_root_mean_square"], item["fp32_root_mean_square"]
            ):
                _fail("IDENTICAL_COMPARISON_RMS_MISMATCH", path, repr(item))
        else:
            for key in nullable_indices:
                value = item[key]
                if not _nonnegative_int(value) or value >= elements:
                    _fail("INVALID_DIFFERENCE_INDEX", f"{path}.{key}", repr(value))
            first_index = item["first_different_flat_index"]
            max_index = item["max_abs_delta_flat_index"]
            if first_index > max_index:
                _fail("DIFFERENCE_INDEX_ORDER_MISMATCH", path, repr(item))
            if different > elements - first_index:
                _fail("DIFFERENCE_INDEX_CAPACITY_MISMATCH", path, repr(item))
            for key in nullable_values:
                if not _finite(item[key]):
                    _fail("INVALID_DIFFERENCE_WITNESS", f"{path}.{key}", repr(item[key]))
            if (
                item["max_abs_delta"] <= 0
                or item["mean_abs_delta"] <= 0
                or item["root_mean_square_delta"] <= 0
                or item["sum_abs_delta"] <= 0
                or item["sum_squared_delta"] <= 0
            ):
                _fail("NONZERO_COMPARISON_HAS_ZERO_DELTA", path, repr(item))
            if denominator <= 0:
                _fail("NONZERO_COMPARISON_HAS_ZERO_ENDPOINT_RMS", path, repr(item))
            if item["mean_abs_delta"] > item["root_mean_square_delta"] and not _close(
                item["mean_abs_delta"], item["root_mean_square_delta"]
            ):
                _fail("DELTA_MOMENT_ORDER_MISMATCH", path, repr(item))
            if item["root_mean_square_delta"] > item["max_abs_delta"] and not _close(
                item["root_mean_square_delta"], item["max_abs_delta"]
            ):
                _fail("DELTA_MAX_ORDER_MISMATCH", path, repr(item))
            witness_max = abs(
                item["bf16_value_at_max_abs_delta"]
                - item["fp32_value_at_max_abs_delta"]
            )
            if not _close(witness_max, item["max_abs_delta"]):
                _fail("MAX_DELTA_WITNESS_MISMATCH", path, repr(item))
            witness_first = abs(
                item["bf16_value_at_first_difference"]
                - item["fp32_value_at_first_difference"]
            )
            if witness_first <= 0 or witness_first > item["max_abs_delta"]:
                _fail("FIRST_DELTA_WITNESS_MISMATCH", path, repr(item))
            witnesses_share_index = (
                item["first_different_flat_index"]
                == item["max_abs_delta_flat_index"]
            )
            if witnesses_share_index and (
                item["bf16_value_at_first_difference"]
                != item["bf16_value_at_max_abs_delta"]
                or item["fp32_value_at_first_difference"]
                != item["fp32_value_at_max_abs_delta"]
            ):
                _fail("SAME_INDEX_WITNESS_MISMATCH", path, repr(item))
            sum_abs_floor = item["max_abs_delta"]
            sum_squared_floor = item["max_abs_delta"] ** 2
            if not witnesses_share_index:
                sum_abs_floor += witness_first
                sum_squared_floor += witness_first**2
            if item["sum_abs_delta"] < sum_abs_floor and not _close(
                item["sum_abs_delta"], sum_abs_floor
            ):
                _fail("SUM_ABS_DELTA_LOWER_BOUND_MISMATCH", path, repr(item))
            sum_abs_ceiling = different * item["max_abs_delta"]
            if item["sum_abs_delta"] > sum_abs_ceiling and not _close(
                item["sum_abs_delta"], sum_abs_ceiling
            ):
                _fail("SUM_ABS_DELTA_BOUND_MISMATCH", path, repr(item))
            if item["sum_squared_delta"] < sum_squared_floor and not _close(
                item["sum_squared_delta"], sum_squared_floor
            ):
                _fail("SUM_SQUARED_DELTA_LOWER_BOUND_MISMATCH", path, repr(item))
            sum_squared_ceiling = different * item["max_abs_delta"] ** 2
            if item["sum_squared_delta"] > sum_squared_ceiling and not _close(
                item["sum_squared_delta"], sum_squared_ceiling
            ):
                _fail("SUM_SQUARED_DELTA_BOUND_MISMATCH", path, repr(item))
            squared_from_abs_ceiling = (
                item["max_abs_delta"] * item["sum_abs_delta"]
            )
            if item["sum_squared_delta"] > squared_from_abs_ceiling and not _close(
                item["sum_squared_delta"], squared_from_abs_ceiling
            ):
                _fail("DELTA_L1_L2_MAX_BOUND_MISMATCH", path, repr(item))
            cauchy_left = item["sum_abs_delta"] ** 2
            cauchy_right = different * item["sum_squared_delta"]
            if cauchy_left > cauchy_right and not _close(cauchy_left, cauchy_right):
                _fail("DELTA_CAUCHY_BOUND_MISMATCH", path, repr(item))
            endpoint_witnesses = (
                (
                    "bf16_root_mean_square",
                    item["bf16_value_at_first_difference"],
                    item["bf16_value_at_max_abs_delta"],
                ),
                (
                    "fp32_root_mean_square",
                    item["fp32_value_at_first_difference"],
                    item["fp32_value_at_max_abs_delta"],
                ),
            )
            for rms_key, first_value, max_value in endpoint_witnesses:
                witness_rms_floor = max(abs(first_value), abs(max_value)) / math.sqrt(
                    elements
                )
                if item[rms_key] < witness_rms_floor and not _close(
                    item[rms_key], witness_rms_floor
                ):
                    _fail("ENDPOINT_RMS_WITNESS_BOUND_MISMATCH", path, repr(item))
            reverse_triangle_floor = abs(
                item["bf16_root_mean_square"] - item["fp32_root_mean_square"]
            )
            if (
                item["root_mean_square_delta"] < reverse_triangle_floor
                and not _close(
                    item["root_mean_square_delta"], reverse_triangle_floor
                )
            ):
                _fail("DELTA_REVERSE_TRIANGLE_BOUND_MISMATCH", path, repr(item))
            triangle_ceiling = (
                item["bf16_root_mean_square"]
                + item["fp32_root_mean_square"]
            )
            if item["root_mean_square_delta"] > triangle_ceiling and not _close(
                item["root_mean_square_delta"], triangle_ceiling
            ):
                _fail("DELTA_TRIANGLE_BOUND_MISMATCH", path, repr(item))
            if (
                item["bf16_float32_sha256"]
                == item["fp32_float32_sha256"]
            ):
                _fail("DIFFERENT_COMPARISON_DIGEST_IDENTITY", path, repr(item))
        normalized.append(item)
    return normalized


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)
