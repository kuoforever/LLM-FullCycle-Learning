"""Pure contracts for the attached-dtype RMSNorm boundary control."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError

ATTACHED_DTYPE_BOUNDARY_CONTROL_VERSION = 1
CONTROL_ID = "layer0_input_rmsnorm_same_checkpoint_values_dtype_replay"
BOUNDARY_MODULE = "model.layers.0.input_layernorm"
BOUNDARY_MODULE_TYPE = (
    "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm"
)
BF16_PATH = "bf16_attached_adapter"
FP32_PATH = "fp32_attached_adapter"
PATH_ORDER = (BF16_PATH, FP32_PATH)
EXPECTED_RUN_PLAN = (
    (BF16_PATH, 1),
    (FP32_PATH, 1),
    (FP32_PATH, 2),
    (BF16_PATH, 2),
)

MATCHED_CLASSIFICATION = (
    "deterministic_same_values_rmsnorm_dtype_replay_reproduces_actual_"
    "boundary_drift"
)
NON_EQUIVALENT_CLASSIFICATION = (
    "deterministic_same_values_rmsnorm_dtype_replay_drift_not_equivalent_"
    "to_actual_boundary"
)
NO_REPLAY_DRIFT_CLASSIFICATION = (
    "same_values_rmsnorm_dtype_replay_does_not_reproduce_actual_boundary_drift"
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CANONICALIZATION = (
    "contiguous_cpu_float32_signed_zero_normalized_finite_exact_no_tolerance"
)
_PLAN_KEYS = frozenset(
    {
        "control_id",
        "intervention_count",
        "module_name",
        "module_type",
        "variance_epsilon",
        "input_shape",
        "weight_shape",
        "common_input_float32_sha256",
        "common_weight_float32_sha256",
        "canonicalization",
        "dtype_arms",
        "serialized_tensor_payload",
        "module_tensor_sidecar_allowed",
    }
)
_ARM_KEYS = frozenset(
    {"path", "input_dtype", "weight_dtype", "output_dtype"}
)
_RUN_KEYS = frozenset(
    {
        "run_id",
        "path",
        "repeat",
        "order_index",
        "fresh_load",
        "frozen_path_reproduced",
        "target_forward_aligned",
        "actual_boundary_reproduced",
        "control_id",
        "control_executed",
        "control_standalone",
        "common_source_roundtrip_exact",
        "control_weight_unchanged",
        "autocast_enabled",
        "actual_input_native_dtype",
        "actual_weight_native_dtype",
        "actual_output_native_dtype",
        "actual_input_float32_sha256",
        "actual_weight_float32_sha256",
        "actual_output_float32_sha256",
        "control_input_native_dtype",
        "control_weight_native_dtype",
        "control_output_native_dtype",
        "control_input_float32_sha256",
        "control_weight_float32_sha256",
        "control_output_float32_sha256",
        "control_cache_arguments_present",
        "control_output_injected",
        "module_tensor_payload_saved",
        "module_tensor_sidecar_saved",
    }
)
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


def classify_attached_dtype_boundary_control(
    *,
    control_plan: Mapping[str, Any],
    runs: Iterable[Mapping[str, Any]],
    frozen_boundary_comparison: Mapping[str, Any],
    actual_boundary_comparison: Mapping[str, Any],
    control_boundary_comparison: Mapping[str, Any],
    source_evidence_locked: bool,
    target_forward_identity_preserved: bool,
    attached_execution_form_fixed: bool,
    checkpoint_sources_unchanged: bool,
) -> dict[str, Any]:
    """Validate one outcome-neutral same-values RMSNorm control.

    Protocol completion is deliberately separate from whether the replay
    reproduces the actual boundary. A valid negative result is classified,
    rather than rejected as a failed protocol.
    """

    requirements = {
        "source_evidence_locked": source_evidence_locked,
        "target_forward_identity_preserved": target_forward_identity_preserved,
        "attached_execution_form_fixed": attached_execution_form_fixed,
        "checkpoint_sources_unchanged": checkpoint_sources_unchanged,
    }
    for name, value in requirements.items():
        if value is not True:
            _fail(
                "BOUNDARY_CONTROL_PROTOCOL_FAILED",
                f"$.requirements.{name}",
                repr(value),
            )

    plan = _validate_plan(control_plan)
    frozen = _validate_comparison(
        frozen_boundary_comparison,
        "$.frozen_boundary_comparison",
    )
    actual = _validate_comparison(
        actual_boundary_comparison,
        "$.actual_boundary_comparison",
    )
    control = _validate_comparison(
        control_boundary_comparison,
        "$.control_boundary_comparison",
    )
    if frozen["canonical_values_equal"] is not False:
        _fail(
            "FROZEN_UNEQUAL_BOUNDARY_MISSING",
            "$.frozen_boundary_comparison",
            repr(frozen),
        )
    if not _strict_equal(actual, frozen):
        _fail(
            "FROZEN_BOUNDARY_NOT_REPRODUCED",
            "$.actual_boundary_comparison",
            repr(actual),
        )

    normalized_runs = _validate_runs(runs, plan, actual, control)
    actual_control_output_exact = all(
        run["actual_output_float32_sha256"]
        == run["control_output_float32_sha256"]
        for run in normalized_runs
    )
    comparison_exact = _strict_equal(actual, control)
    endpoints_exact = (
        actual["bf16_float32_sha256"] == control["bf16_float32_sha256"]
        and actual["fp32_float32_sha256"] == control["fp32_float32_sha256"]
    )
    if endpoints_exact and not comparison_exact:
        _fail(
            "SAME_ENDPOINT_COMPARISON_SUMMARY_MISMATCH",
            "$.control_boundary_comparison",
            repr(control),
        )

    control_equal = control["canonical_values_equal"]
    if control_equal:
        classification = NO_REPLAY_DRIFT_CLASSIFICATION
        current_forward_sufficiency = False
    elif actual_control_output_exact and comparison_exact:
        classification = MATCHED_CLASSIFICATION
        current_forward_sufficiency = True
    else:
        classification = NON_EQUIVALENT_CLASSIFICATION
        current_forward_sufficiency = False

    return {
        "control_id": plan["control_id"],
        "run_count": len(normalized_runs),
        "dtype_arm_count": len(PATH_ORDER),
        "fresh_repeats_per_dtype": 2,
        "run_order_design": "ABBA",
        "actual_boundary_reproduced": True,
        "actual_boundary_unequal": True,
        "same_values_preconditions_passed": True,
        "actual_boundary_repeat_stable": True,
        "control_repeat_stable": True,
        "control_cross_dtype_values_equal": control_equal,
        "actual_control_output_exact": actual_control_output_exact,
        "actual_control_comparison_exact": comparison_exact,
        "current_forward_boundary_sufficiency_observed": (
            current_forward_sufficiency
        ),
        "protocol_completed": True,
        "classification": classification,
    }


def _validate_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        _fail("INVALID_CONTROL_PLAN", "$.control_plan", repr(type(source)))
    if set(source) != set(_PLAN_KEYS):
        _fail(
            "INVALID_CONTROL_PLAN_FIELDS",
            "$.control_plan",
            repr(sorted(source)),
        )
    plan = dict(source)
    expected_scalars = {
        "control_id": CONTROL_ID,
        "intervention_count": 1,
        "module_name": BOUNDARY_MODULE,
        "module_type": BOUNDARY_MODULE_TYPE,
        "variance_epsilon": 1e-6,
        "input_shape": [1, 1, 1536],
        "weight_shape": [1536],
        "canonicalization": _CANONICALIZATION,
        "serialized_tensor_payload": False,
        "module_tensor_sidecar_allowed": False,
    }
    for key, expected in expected_scalars.items():
        if not _strict_equal(plan[key], expected):
            _fail(
                "INVALID_CONTROL_PLAN_VALUE",
                f"$.control_plan.{key}",
                repr(plan[key]),
            )
    for key in (
        "common_input_float32_sha256",
        "common_weight_float32_sha256",
    ):
        _digest(plan[key], f"$.control_plan.{key}")

    arms = plan["dtype_arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        _fail("INVALID_DTYPE_ARMS", "$.control_plan.dtype_arms", repr(arms))
    expected_arms = (
        {
            "path": BF16_PATH,
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "output_dtype": "bfloat16",
        },
        {
            "path": FP32_PATH,
            "input_dtype": "float32",
            "weight_dtype": "float32",
            "output_dtype": "float32",
        },
    )
    for index, (arm, expected) in enumerate(zip(arms, expected_arms, strict=True)):
        path = f"$.control_plan.dtype_arms[{index}]"
        if not isinstance(arm, Mapping) or set(arm) != set(_ARM_KEYS):
            _fail("INVALID_DTYPE_ARM_FIELDS", path, repr(arm))
        if not _strict_equal(dict(arm), expected):
            _fail("INVALID_DTYPE_ARM", path, repr(arm))
    return plan


def _validate_runs(
    sources: Iterable[Mapping[str, Any]],
    plan: Mapping[str, Any],
    actual: Mapping[str, Any],
    control: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(sources, (str, bytes, Mapping)):
        _fail("INVALID_CONTROL_RUNS", "$.runs", repr(type(sources)))
    try:
        raw_runs = list(sources)
    except TypeError:
        _fail("INVALID_CONTROL_RUNS", "$.runs", repr(type(sources)))
    if len(raw_runs) != len(EXPECTED_RUN_PLAN):
        _fail("INVALID_CONTROL_RUN_COUNT", "$.runs", repr(len(raw_runs)))

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, (source, expected_arm) in enumerate(
        zip(raw_runs, EXPECTED_RUN_PLAN, strict=True)
    ):
        path = f"$.runs[{index}]"
        if not isinstance(source, Mapping):
            _fail("INVALID_CONTROL_RUN", path, repr(type(source)))
        if set(source) != set(_RUN_KEYS):
            _fail("INVALID_CONTROL_RUN_FIELDS", path, repr(sorted(source)))
        run = dict(source)
        run_id = run["run_id"]
        if not isinstance(run_id, str) or not run_id or run_id in seen_ids:
            _fail("INVALID_CONTROL_RUN_ID", f"{path}.run_id", repr(run_id))
        seen_ids.add(run_id)
        expected_path, expected_repeat = expected_arm
        if not all(
            (
                _strict_equal(run["order_index"], index),
                _strict_equal(run["path"], expected_path),
                _strict_equal(run["repeat"], expected_repeat),
            )
        ):
            _fail("INVALID_ABBA_RUN_PLAN", path, repr(run))
        if run["control_id"] != plan["control_id"]:
            _fail("MULTIPLE_OR_UNKNOWN_CONTROL", f"{path}.control_id", repr(run))

        required_true = (
            "fresh_load",
            "frozen_path_reproduced",
            "target_forward_aligned",
            "actual_boundary_reproduced",
            "control_executed",
            "control_standalone",
            "common_source_roundtrip_exact",
            "control_weight_unchanged",
        )
        for key in required_true:
            if run[key] is not True:
                _fail("CONTROL_RUN_REQUIREMENT_FAILED", f"{path}.{key}", repr(run[key]))
        required_false = (
            "autocast_enabled",
            "control_cache_arguments_present",
            "control_output_injected",
            "module_tensor_payload_saved",
            "module_tensor_sidecar_saved",
        )
        for key in required_false:
            if run[key] is not False:
                _fail("CONTROL_SAFETY_BOUNDARY_VIOLATION", f"{path}.{key}", repr(run[key]))

        native_dtype = "bfloat16" if expected_path == BF16_PATH else "float32"
        for key in (
            "actual_input_native_dtype",
            "actual_weight_native_dtype",
            "actual_output_native_dtype",
            "control_input_native_dtype",
            "control_weight_native_dtype",
            "control_output_native_dtype",
        ):
            if run[key] != native_dtype:
                _fail("INVALID_RUN_NATIVE_DTYPE", f"{path}.{key}", repr(run[key]))
        for key in (
            "actual_input_float32_sha256",
            "actual_weight_float32_sha256",
            "actual_output_float32_sha256",
            "control_input_float32_sha256",
            "control_weight_float32_sha256",
            "control_output_float32_sha256",
        ):
            _digest(run[key], f"{path}.{key}")

        if (
            run["actual_input_float32_sha256"]
            != plan["common_input_float32_sha256"]
            or run["control_input_float32_sha256"]
            != plan["common_input_float32_sha256"]
        ):
            _fail("COMMON_INPUT_IDENTITY_FAILED", path, repr(run))
        if (
            run["actual_weight_float32_sha256"]
            != plan["common_weight_float32_sha256"]
            or run["control_weight_float32_sha256"]
            != plan["common_weight_float32_sha256"]
        ):
            _fail("COMMON_WEIGHT_IDENTITY_FAILED", path, repr(run))
        prefix = "bf16" if expected_path == BF16_PATH else "fp32"
        if run["actual_output_float32_sha256"] != actual[
            f"{prefix}_float32_sha256"
        ]:
            _fail("ACTUAL_OUTPUT_COMPARISON_LINK_FAILED", path, repr(run))
        if run["control_output_float32_sha256"] != control[
            f"{prefix}_float32_sha256"
        ]:
            _fail("CONTROL_OUTPUT_COMPARISON_LINK_FAILED", path, repr(run))
        result.append(run)

    for path_name in PATH_ORDER:
        path_runs = [run for run in result if run["path"] == path_name]
        if len(path_runs) != 2 or {run["repeat"] for run in path_runs} != {1, 2}:
            _fail("DTYPE_ARM_REPEAT_MISSING", "$.runs", repr(path_runs))
        repeat_keys = (
            "actual_input_float32_sha256",
            "actual_weight_float32_sha256",
            "actual_output_float32_sha256",
            "control_input_float32_sha256",
            "control_weight_float32_sha256",
            "control_output_float32_sha256",
        )
        if any(path_runs[0][key] != path_runs[1][key] for key in repeat_keys):
            _fail("DTYPE_ARM_REPEAT_DRIFT", "$.runs", repr(path_runs))
    return result


def _validate_comparison(source: Mapping[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        _fail("INVALID_BOUNDARY_COMPARISON", path, repr(type(source)))
    if set(source) != set(_COMPARISON_KEYS):
        _fail("INVALID_BOUNDARY_COMPARISON_FIELDS", path, repr(sorted(source)))
    item = dict(source)
    if item["name"] != BOUNDARY_MODULE:
        _fail("INVALID_BOUNDARY_MODULE", f"{path}.name", repr(item["name"]))
    if not _strict_equal(item["shape"], [1, 1, 1536]) or not _strict_equal(
        item["elements"], 1536
    ):
        _fail("INVALID_BOUNDARY_SHAPE", path, repr(item))
    if (
        item["bf16_native_dtype"] != "bfloat16"
        or item["fp32_native_dtype"] != "float32"
        or item["comparison_dtype"] != "float32"
    ):
        _fail("INVALID_BOUNDARY_DTYPE", path, repr(item))
    for key in ("bf16_float32_sha256", "fp32_float32_sha256"):
        _digest(item[key], f"{path}.{key}")
    if not isinstance(item["canonical_values_equal"], bool):
        _fail("INVALID_BOUNDARY_EQUALITY", path, repr(item))
    different = item["different_elements"]
    if not _nonnegative_int(different) or different > item["elements"]:
        _fail("INVALID_BOUNDARY_DIFFERENCE_COUNT", path, repr(different))
    if item["canonical_values_equal"] is not (different == 0):
        _fail("BOUNDARY_EQUALITY_COUNT_MISMATCH", path, repr(item))
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
            _fail("INVALID_BOUNDARY_STATISTIC", f"{path}.{key}", repr(item[key]))
    elements = item["elements"]
    if not _close(item["different_fraction"], different / elements):
        _fail("INVALID_BOUNDARY_DIFFERENT_FRACTION", path, repr(item))
    if not _close(item["mean_abs_delta"], item["sum_abs_delta"] / elements):
        _fail("INVALID_BOUNDARY_MEAN", path, repr(item))
    if not _close(
        item["root_mean_square_delta"],
        math.sqrt(item["sum_squared_delta"] / elements),
    ):
        _fail("INVALID_BOUNDARY_RMS", path, repr(item))
    endpoint_rms = max(
        item["bf16_root_mean_square"],
        item["fp32_root_mean_square"],
    )
    expected_normalized = (
        item["root_mean_square_delta"] / endpoint_rms
        if endpoint_rms > 0
        else 0.0
    )
    if not _close(
        item["normalized_root_mean_square_delta"],
        expected_normalized,
    ):
        _fail("INVALID_BOUNDARY_NORMALIZED_RMS", path, repr(item))

    index_keys = ("first_different_flat_index", "max_abs_delta_flat_index")
    witness_keys = (
        "bf16_value_at_first_difference",
        "fp32_value_at_first_difference",
        "bf16_value_at_max_abs_delta",
        "fp32_value_at_max_abs_delta",
    )
    if different == 0:
        if any(item[key] is not None for key in (*index_keys, *witness_keys)):
            _fail("EQUAL_BOUNDARY_HAS_WITNESS", path, repr(item))
        if any(
            item[key] != 0.0
            for key in (
                "max_abs_delta",
                "mean_abs_delta",
                "root_mean_square_delta",
                "sum_abs_delta",
                "sum_squared_delta",
                "different_fraction",
                "normalized_root_mean_square_delta",
                "root_mean_square_delta_ratio_to_first_registered_difference",
            )
        ):
            _fail("EQUAL_BOUNDARY_HAS_DELTA", path, repr(item))
        if item["bf16_float32_sha256"] != item["fp32_float32_sha256"]:
            _fail("EQUAL_BOUNDARY_DIGEST_MISMATCH", path, repr(item))
        if not _close(
            item["bf16_root_mean_square"],
            item["fp32_root_mean_square"],
        ):
            _fail("EQUAL_BOUNDARY_RMS_MISMATCH", path, repr(item))
        return item

    for key in index_keys:
        value = item[key]
        if not _nonnegative_int(value) or value >= elements:
            _fail("INVALID_BOUNDARY_DIFFERENCE_INDEX", f"{path}.{key}", repr(value))
    for key in witness_keys:
        if not _finite(item[key]):
            _fail("INVALID_BOUNDARY_WITNESS", f"{path}.{key}", repr(item[key]))
    if item["first_different_flat_index"] > item["max_abs_delta_flat_index"]:
        _fail("BOUNDARY_INDEX_ORDER_MISMATCH", path, repr(item))
    if different > elements - item["first_different_flat_index"]:
        _fail("BOUNDARY_DIFFERENCE_CAPACITY_MISMATCH", path, repr(item))
    first_delta = abs(
        item["bf16_value_at_first_difference"]
        - item["fp32_value_at_first_difference"]
    )
    max_delta = abs(
        item["bf16_value_at_max_abs_delta"]
        - item["fp32_value_at_max_abs_delta"]
    )
    if first_delta <= 0 or first_delta > item["max_abs_delta"]:
        _fail("BOUNDARY_FIRST_WITNESS_MISMATCH", path, repr(item))
    if not _close(max_delta, item["max_abs_delta"]):
        _fail("BOUNDARY_MAX_WITNESS_MISMATCH", path, repr(item))
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
        _fail("BOUNDARY_SAME_INDEX_WITNESS_MISMATCH", path, repr(item))
    if any(
        item[key] <= 0
        for key in (
            "max_abs_delta",
            "mean_abs_delta",
            "root_mean_square_delta",
            "sum_abs_delta",
            "sum_squared_delta",
            "different_fraction",
        )
    ):
        _fail("UNEQUAL_BOUNDARY_HAS_ZERO_DELTA", path, repr(item))
    if item["mean_abs_delta"] > item["root_mean_square_delta"] and not _close(
        item["mean_abs_delta"], item["root_mean_square_delta"]
    ):
        _fail("BOUNDARY_MOMENT_ORDER_MISMATCH", path, repr(item))
    if item["root_mean_square_delta"] > item["max_abs_delta"] and not _close(
        item["root_mean_square_delta"], item["max_abs_delta"]
    ):
        _fail("BOUNDARY_MAX_ORDER_MISMATCH", path, repr(item))
    if not _close(
        item["root_mean_square_delta_ratio_to_first_registered_difference"],
        1.0,
    ):
        _fail("INVALID_BOUNDARY_FIRST_DIFFERENCE_RATIO", path, repr(item))
    sum_abs_floor = item["max_abs_delta"]
    sum_squared_floor = item["max_abs_delta"] ** 2
    if not witnesses_share_index:
        sum_abs_floor += first_delta
        sum_squared_floor += first_delta**2
    if item["sum_abs_delta"] < sum_abs_floor and not _close(
        item["sum_abs_delta"], sum_abs_floor
    ):
        _fail("BOUNDARY_SUM_ABS_LOWER_BOUND", path, repr(item))
    if item["sum_abs_delta"] > different * item["max_abs_delta"] and not _close(
        item["sum_abs_delta"], different * item["max_abs_delta"]
    ):
        _fail("BOUNDARY_SUM_ABS_UPPER_BOUND", path, repr(item))
    if item["sum_squared_delta"] < sum_squared_floor and not _close(
        item["sum_squared_delta"], sum_squared_floor
    ):
        _fail("BOUNDARY_SUM_SQUARED_LOWER_BOUND", path, repr(item))
    if (
        item["sum_squared_delta"] > different * item["max_abs_delta"] ** 2
        and not _close(
            item["sum_squared_delta"], different * item["max_abs_delta"] ** 2
        )
    ):
        _fail("BOUNDARY_SUM_SQUARED_UPPER_BOUND", path, repr(item))
    squared_from_abs_ceiling = item["max_abs_delta"] * item["sum_abs_delta"]
    if item["sum_squared_delta"] > squared_from_abs_ceiling and not _close(
        item["sum_squared_delta"], squared_from_abs_ceiling
    ):
        _fail("BOUNDARY_L1_L2_MAX_BOUND_MISMATCH", path, repr(item))
    cauchy_left = item["sum_abs_delta"] ** 2
    cauchy_right = different * item["sum_squared_delta"]
    if cauchy_left > cauchy_right and not _close(cauchy_left, cauchy_right):
        _fail("BOUNDARY_CAUCHY_BOUND_MISMATCH", path, repr(item))
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
            _fail("BOUNDARY_ENDPOINT_RMS_WITNESS_BOUND", path, repr(item))
    reverse_triangle_floor = abs(
        item["bf16_root_mean_square"] - item["fp32_root_mean_square"]
    )
    if (
        item["root_mean_square_delta"] < reverse_triangle_floor
        and not _close(item["root_mean_square_delta"], reverse_triangle_floor)
    ):
        _fail("BOUNDARY_REVERSE_TRIANGLE_MISMATCH", path, repr(item))
    triangle_ceiling = (
        item["bf16_root_mean_square"] + item["fp32_root_mean_square"]
    )
    if item["root_mean_square_delta"] > triangle_ceiling and not _close(
        item["root_mean_square_delta"], triangle_ceiling
    ):
        _fail("BOUNDARY_TRIANGLE_MISMATCH", path, repr(item))
    if item["bf16_float32_sha256"] == item["fp32_float32_sha256"]:
        _fail("UNEQUAL_BOUNDARY_DIGEST_IDENTITY", path, repr(item))
    return item


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail("INVALID_SHA256", path, repr(value))
    return value


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
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)
