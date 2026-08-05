from __future__ import annotations

import copy
import math
import unittest

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_attached_dtype_numerics import (
    CLASSIFICATION,
    REGISTERED_OUTPUT_STAGES,
    analyze_registered_module_comparisons,
    classify_attached_dtype_numerics,
)

ZERO = "sha256:" + "0" * 64
ONE = "sha256:" + "1" * 64


def _comparison(name: str, equal: bool) -> dict[str, object]:
    rms = 0.0 if equal else math.sqrt(0.5)
    return {
        "name": name,
        "shape": [1, 1, 4],
        "elements": 4,
        "bf16_native_dtype": "bfloat16",
        "fp32_native_dtype": "float32",
        "comparison_dtype": "float32",
        "bf16_float32_sha256": ZERO,
        "fp32_float32_sha256": ZERO if equal else ONE,
        "canonical_values_equal": equal,
        "different_elements": 0 if equal else 2,
        "first_different_flat_index": None if equal else 0,
        "max_abs_delta_flat_index": None if equal else 1,
        "bf16_value_at_first_difference": None if equal else 1.0,
        "fp32_value_at_first_difference": None if equal else 2.0,
        "bf16_value_at_max_abs_delta": None if equal else 2.0,
        "fp32_value_at_max_abs_delta": None if equal else 3.0,
        "max_abs_delta": 0.0 if equal else 1.0,
        "mean_abs_delta": 0.0 if equal else 0.5,
        "root_mean_square_delta": rms,
        "sum_abs_delta": 0.0 if equal else 2.0,
        "sum_squared_delta": 0.0 if equal else 2.0,
        "different_fraction": 0.0 if equal else 0.5,
        "bf16_root_mean_square": 2.0,
        "fp32_root_mean_square": 2.0 if equal else 2.5,
        "normalized_root_mean_square_delta": 0.0 if equal else rms / 2.5,
        "root_mean_square_delta_ratio_to_first_registered_difference": (
            0.0 if equal else 1.0
        ),
    }


def _comparisons(first_difference: int = 1) -> list[dict[str, object]]:
    return [
        _comparison(name, index < first_difference)
        for index, name in enumerate(REGISTERED_OUTPUT_STAGES)
    ]


def _classify(comparisons: list[dict[str, object]]) -> dict[str, object]:
    return classify_attached_dtype_numerics(
        comparisons,
        bf16_repeat_stable=True,
        fp32_repeat_stable=True,
        bf16_reference_reproduced=True,
        fp32_reference_reproduced=True,
        capture_plan_executed=True,
        target_forward_aligned=True,
        lm_head_raw_logit_linked=True,
        attached_execution_form_fixed=True,
        source_inputs_unchanged=True,
        module_tensor_payload_absent=True,
    )


class AttachedDtypeNumericsContractTests(unittest.TestCase):
    def test_locates_first_registered_difference_and_lm_head_propagation(self) -> None:
        result = _classify(_comparisons())

        self.assertEqual(result["registered_module_count"], 40)
        self.assertEqual(result["first_unequal_module_index"], 1)
        self.assertEqual(
            result["first_unequal_module"],
            "model.layers.0.input_layernorm",
        )
        self.assertTrue(result["preceding_registered_outputs_identical"])
        self.assertTrue(result["registered_lm_head_difference_observed"])
        self.assertEqual(result["classification"], CLASSIFICATION)

    def test_rejects_registered_stage_reordering(self) -> None:
        comparisons = _comparisons()
        comparisons[1], comparisons[2] = comparisons[2], comparisons[1]
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_REGISTERED_MODULE_SEQUENCE",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_all_registered_outputs_equal(self) -> None:
        comparisons = [
            _comparison(name, True) for name in REGISTERED_OUTPUT_STAGES
        ]
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "REGISTERED_MODULE_DIFFERENCE_MISSING",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_missing_lm_head_propagation(self) -> None:
        comparisons = _comparisons()
        comparisons[-1] = _comparison("lm_head", True)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "LM_HEAD_PROPAGATION_MISSING",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_probe_summary_algebra_forgery(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["mean_abs_delta"] = 0.25
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_MEAN_ABS_DELTA",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_propagation_ratio_forgery(self) -> None:
        comparisons = _comparisons()
        comparisons[-1][
            "root_mean_square_delta_ratio_to_first_registered_difference"
        ] = 2.0
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_PROPAGATION_RATIO",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_difference_with_identical_digest(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["fp32_float32_sha256"] = ZERO
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DIFFERENT_COMPARISON_DIGEST_IDENTITY",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_false_protocol_flag(self) -> None:
        comparisons = _comparisons()
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ATTACHED_DTYPE_NUMERICS_PROTOCOL_FAILED",
        ):
            classify_attached_dtype_numerics(
                comparisons,
                bf16_repeat_stable=True,
                fp32_repeat_stable=True,
                bf16_reference_reproduced=True,
                fp32_reference_reproduced=True,
                capture_plan_executed=False,
                target_forward_aligned=True,
                lm_head_raw_logit_linked=True,
                attached_execution_form_fixed=True,
                source_inputs_unchanged=True,
                module_tensor_payload_absent=True,
            )

    def test_rejects_unknown_comparison_field(self) -> None:
        comparisons = copy.deepcopy(_comparisons())
        comparisons[0]["unexpected"] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_MODULE_COMPARISON_FIELDS",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_non_mapping_comparison(self) -> None:
        comparisons: list[object] = list(_comparisons())
        comparisons[0] = []
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_MODULE_COMPARISON",
        ):
            analyze_registered_module_comparisons(comparisons)  # type: ignore[arg-type]

    def test_rejects_sum_bound_forgery(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["sum_abs_delta"] = 2.4
        comparisons[1]["mean_abs_delta"] = 0.6
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SUM_ABS_DELTA_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_sum_abs_delta_below_observed_maximum(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["sum_abs_delta"] = 0.5
        comparisons[1]["mean_abs_delta"] = 0.125
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SUM_ABS_DELTA_LOWER_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_sum_squared_delta_below_observed_maximum(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["sum_squared_delta"] = 1.0
        comparisons[1]["root_mean_square_delta"] = 0.5
        comparisons[1]["normalized_root_mean_square_delta"] = 0.2
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SUM_SQUARED_DELTA_LOWER_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_two_witnesses_counted_as_one_delta(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["sum_abs_delta"] = 1.0
        comparisons[1]["mean_abs_delta"] = 0.25
        comparisons[1]["sum_squared_delta"] = 1.0
        comparisons[1]["root_mean_square_delta"] = 0.5
        comparisons[1]["normalized_root_mean_square_delta"] = 0.2
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SUM_ABS_DELTA_LOWER_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_inconsistent_witnesses_at_same_index(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["first_different_flat_index"] = 1
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SAME_INDEX_WITNESS_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_squared_sum_above_l1_max_bound(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["bf16_value_at_first_difference"] = 1.0
        comparisons[1]["fp32_value_at_first_difference"] = 1.2
        comparisons[1]["sum_abs_delta"] = 1.2
        comparisons[1]["mean_abs_delta"] = 0.3
        comparisons[1]["sum_squared_delta"] = 1.5
        comparisons[1]["root_mean_square_delta"] = math.sqrt(0.375)
        comparisons[1]["normalized_root_mean_square_delta"] = (
            math.sqrt(0.375) / 2.5
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DELTA_L1_L2_MAX_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_l1_l2_cauchy_violation(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["bf16_value_at_first_difference"] = 1.0
        comparisons[1]["fp32_value_at_first_difference"] = 1.2
        comparisons[1]["sum_abs_delta"] = 1.8
        comparisons[1]["mean_abs_delta"] = 0.45
        comparisons[1]["sum_squared_delta"] = 1.04
        comparisons[1]["root_mean_square_delta"] = math.sqrt(0.26)
        comparisons[1]["normalized_root_mean_square_delta"] = (
            math.sqrt(0.26) / 2.5
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DELTA_CAUCHY_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_first_difference_after_maximum_difference(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["first_different_flat_index"] = 2
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DIFFERENCE_INDEX_ORDER_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_difference_count_beyond_first_index_capacity(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["first_different_flat_index"] = 3
        comparisons[1]["max_abs_delta_flat_index"] = 3
        comparisons[1]["bf16_value_at_first_difference"] = 2.0
        comparisons[1]["fp32_value_at_first_difference"] = 3.0
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DIFFERENCE_INDEX_CAPACITY_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_endpoint_rms_below_witness_bound(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["fp32_root_mean_square"] = 0.4
        comparisons[1]["normalized_root_mean_square_delta"] = (
            math.sqrt(0.5) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ENDPOINT_RMS_WITNESS_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)

    def test_rejects_endpoint_rms_reverse_triangle_violation(self) -> None:
        comparisons = _comparisons()
        comparisons[1]["bf16_root_mean_square"] = 10.0
        comparisons[1]["fp32_root_mean_square"] = 2.0
        comparisons[1]["normalized_root_mean_square_delta"] = (
            math.sqrt(0.5) / 10.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "DELTA_REVERSE_TRIANGLE_BOUND_MISMATCH",
        ):
            analyze_registered_module_comparisons(comparisons)


if __name__ == "__main__":
    unittest.main()
