from __future__ import annotations

import copy
import math
import unittest

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_attached_dtype_boundary_control import (
    BF16_PATH,
    BOUNDARY_MODULE,
    BOUNDARY_MODULE_TYPE,
    CONTROL_ID,
    FP32_PATH,
    MATCHED_CLASSIFICATION,
    NON_EQUIVALENT_CLASSIFICATION,
    NO_REPLAY_DRIFT_CLASSIFICATION,
    classify_attached_dtype_boundary_control,
)


def _digest(value: int) -> str:
    return f"sha256:{value:064x}"


COMMON_INPUT = _digest(10)
COMMON_WEIGHT = _digest(11)
ACTUAL_BF16 = _digest(12)
ACTUAL_FP32 = _digest(13)
CONTROL_BF16 = _digest(14)
CONTROL_FP32 = _digest(15)


def _comparison(*, equal: bool, bf16: str, fp32: str) -> dict[str, object]:
    elements = 1536
    sum_abs = 0.0 if equal else 2.0
    sum_squared = 0.0 if equal else 2.0
    return {
        "name": BOUNDARY_MODULE,
        "shape": [1, 1, elements],
        "elements": elements,
        "bf16_native_dtype": "bfloat16",
        "fp32_native_dtype": "float32",
        "comparison_dtype": "float32",
        "bf16_float32_sha256": bf16,
        "fp32_float32_sha256": fp32,
        "canonical_values_equal": equal,
        "different_elements": 0 if equal else 2,
        "first_different_flat_index": None if equal else 0,
        "max_abs_delta_flat_index": None if equal else 1,
        "bf16_value_at_first_difference": None if equal else 1.0,
        "fp32_value_at_first_difference": None if equal else 2.0,
        "bf16_value_at_max_abs_delta": None if equal else 2.0,
        "fp32_value_at_max_abs_delta": None if equal else 3.0,
        "max_abs_delta": 0.0 if equal else 1.0,
        "mean_abs_delta": sum_abs / elements,
        "root_mean_square_delta": math.sqrt(sum_squared / elements),
        "sum_abs_delta": sum_abs,
        "sum_squared_delta": sum_squared,
        "different_fraction": (0.0 if equal else 2 / elements),
        "bf16_root_mean_square": 2.0,
        "fp32_root_mean_square": 2.0,
        "normalized_root_mean_square_delta": (
            0.0 if equal else math.sqrt(sum_squared / elements) / 2.0
        ),
        "root_mean_square_delta_ratio_to_first_registered_difference": (
            0.0 if equal else 1.0
        ),
    }


def _plan() -> dict[str, object]:
    return {
        "control_id": CONTROL_ID,
        "intervention_count": 1,
        "module_name": BOUNDARY_MODULE,
        "module_type": BOUNDARY_MODULE_TYPE,
        "variance_epsilon": 1e-6,
        "input_shape": [1, 1, 1536],
        "weight_shape": [1536],
        "common_input_float32_sha256": COMMON_INPUT,
        "common_weight_float32_sha256": COMMON_WEIGHT,
        "canonicalization": (
            "contiguous_cpu_float32_signed_zero_normalized_finite_exact_no_tolerance"
        ),
        "dtype_arms": [
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
        ],
        "serialized_tensor_payload": False,
        "module_tensor_sidecar_allowed": False,
    }


def _runs(control: dict[str, object]) -> list[dict[str, object]]:
    plan = (
        (BF16_PATH, 1, "bf16-boundary-control-r1"),
        (FP32_PATH, 1, "fp32-boundary-control-r1"),
        (FP32_PATH, 2, "fp32-boundary-control-r2"),
        (BF16_PATH, 2, "bf16-boundary-control-r2"),
    )
    result: list[dict[str, object]] = []
    for order_index, (path, repeat, run_id) in enumerate(plan):
        prefix = "bf16" if path == BF16_PATH else "fp32"
        native_dtype = "bfloat16" if path == BF16_PATH else "float32"
        result.append(
            {
                "run_id": run_id,
                "path": path,
                "repeat": repeat,
                "order_index": order_index,
                "fresh_load": True,
                "frozen_path_reproduced": True,
                "target_forward_aligned": True,
                "actual_boundary_reproduced": True,
                "control_id": CONTROL_ID,
                "control_executed": True,
                "control_standalone": True,
                "common_source_roundtrip_exact": True,
                "control_weight_unchanged": True,
                "autocast_enabled": False,
                "actual_input_native_dtype": native_dtype,
                "actual_weight_native_dtype": native_dtype,
                "actual_output_native_dtype": native_dtype,
                "actual_input_float32_sha256": COMMON_INPUT,
                "actual_weight_float32_sha256": COMMON_WEIGHT,
                "actual_output_float32_sha256": (
                    ACTUAL_BF16 if prefix == "bf16" else ACTUAL_FP32
                ),
                "control_input_native_dtype": native_dtype,
                "control_weight_native_dtype": native_dtype,
                "control_output_native_dtype": native_dtype,
                "control_input_float32_sha256": COMMON_INPUT,
                "control_weight_float32_sha256": COMMON_WEIGHT,
                "control_output_float32_sha256": control[
                    f"{prefix}_float32_sha256"
                ],
                "control_cache_arguments_present": False,
                "control_output_injected": False,
                "module_tensor_payload_saved": False,
                "module_tensor_sidecar_saved": False,
            }
        )
    return result


def _classify(control: dict[str, object]) -> dict[str, object]:
    frozen = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
    return classify_attached_dtype_boundary_control(
        control_plan=_plan(),
        runs=_runs(control),
        frozen_boundary_comparison=frozen,
        actual_boundary_comparison=copy.deepcopy(frozen),
        control_boundary_comparison=control,
        source_evidence_locked=True,
        target_forward_identity_preserved=True,
        attached_execution_form_fixed=True,
        checkpoint_sources_unchanged=True,
    )


class AttachedDtypeBoundaryControlContractTests(unittest.TestCase):
    def test_classifies_exact_actual_reproduction_without_outcome_bias(self) -> None:
        result = _classify(
            _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        )

        self.assertTrue(result["protocol_completed"])
        self.assertTrue(result["actual_control_output_exact"])
        self.assertTrue(result["actual_control_comparison_exact"])
        self.assertTrue(result["current_forward_boundary_sufficiency_observed"])
        self.assertEqual(result["classification"], MATCHED_CLASSIFICATION)

    def test_classifies_unequal_non_equivalent_replay_as_valid_result(self) -> None:
        result = _classify(
            _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        )

        self.assertTrue(result["protocol_completed"])
        self.assertFalse(result["actual_control_output_exact"])
        self.assertFalse(result["current_forward_boundary_sufficiency_observed"])
        self.assertEqual(result["classification"], NON_EQUIVALENT_CLASSIFICATION)

    def test_classifies_equal_replay_as_valid_falsification(self) -> None:
        result = _classify(
            _comparison(equal=True, bf16=CONTROL_BF16, fp32=CONTROL_BF16)
        )

        self.assertTrue(result["protocol_completed"])
        self.assertTrue(result["control_cross_dtype_values_equal"])
        self.assertFalse(result["current_forward_boundary_sufficiency_observed"])
        self.assertEqual(result["classification"], NO_REPLAY_DRIFT_CLASSIFICATION)

    def test_rejects_false_global_protocol_requirement(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        frozen = copy.deepcopy(control)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_CONTROL_PROTOCOL_FAILED",
        ):
            classify_attached_dtype_boundary_control(
                control_plan=_plan(),
                runs=_runs(control),
                frozen_boundary_comparison=frozen,
                actual_boundary_comparison=copy.deepcopy(frozen),
                control_boundary_comparison=control,
                source_evidence_locked=False,
                target_forward_identity_preserved=True,
                attached_execution_form_fixed=True,
                checkpoint_sources_unchanged=True,
            )

    def test_rejects_second_intervention(self) -> None:
        plan = _plan()
        plan["intervention_count"] = 2
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_CONTROL_PLAN_VALUE",
        ):
            self._classify_with(plan=plan, runs=_runs(control), control=control)

    def test_rejects_unknown_control_plan_field(self) -> None:
        plan = _plan()
        plan["unexpected"] = True
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_CONTROL_PLAN_FIELDS",
        ):
            self._classify_with(plan=plan, runs=_runs(control), control=control)

    def test_rejects_bool_as_int_control_plan_shapes(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        for key, shape in (
            ("input_shape", [True, True, 1536]),
            ("weight_shape", [True]),
        ):
            with self.subTest(key=key):
                plan = _plan()
                plan[key] = shape
                with self.assertRaisesRegex(
                    ToolRouterValidationError,
                    "INVALID_CONTROL_PLAN_VALUE",
                ):
                    self._classify_with(
                        plan=plan,
                        runs=_runs(control),
                        control=control,
                    )

    def test_rejects_duplicate_dtype_arm(self) -> None:
        plan = _plan()
        plan["dtype_arms"] = [
            copy.deepcopy(plan["dtype_arms"][0]),  # type: ignore[index]
            copy.deepcopy(plan["dtype_arms"][0]),  # type: ignore[index]
        ]
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_DTYPE_ARM"):
            self._classify_with(plan=plan, runs=_runs(control), control=control)

    def test_rejects_non_abba_run_order(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[0]["path"], runs[1]["path"] = runs[1]["path"], runs[0]["path"]
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_ABBA_RUN_PLAN"):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def test_rejects_bool_as_int_abba_fields(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        for run_index, key, value in (
            (0, "order_index", False),
            (1, "order_index", True),
            (0, "repeat", True),
        ):
            with self.subTest(run_index=run_index, key=key):
                runs = _runs(control)
                runs[run_index][key] = value
                with self.assertRaisesRegex(
                    ToolRouterValidationError,
                    "INVALID_ABBA_RUN_PLAN",
                ):
                    self._classify_with(
                        plan=_plan(),
                        runs=runs,
                        control=control,
                    )

    def test_rejects_bool_as_int_comparison_shape(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        control["shape"] = [True, True, 1536]
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_BOUNDARY_SHAPE",
        ):
            self._classify_with(
                plan=_plan(),
                runs=_runs(control),
                control=control,
            )

    def test_rejects_unknown_second_control_id(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[2]["control_id"] = "another_control"
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "MULTIPLE_OR_UNKNOWN_CONTROL",
        ):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def test_rejects_common_actual_input_identity_drift(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[1]["actual_input_float32_sha256"] = _digest(99)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "COMMON_INPUT_IDENTITY_FAILED",
        ):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def test_rejects_common_weight_identity_drift(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[0]["control_weight_float32_sha256"] = _digest(99)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "COMMON_WEIGHT_IDENTITY_FAILED",
        ):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def test_rejects_equal_frozen_boundary(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        frozen = _comparison(equal=True, bf16=ACTUAL_BF16, fp32=ACTUAL_BF16)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_UNEQUAL_BOUNDARY_MISSING",
        ):
            self._classify_with(
                plan=_plan(),
                runs=_runs(control),
                control=control,
                frozen=frozen,
                actual=copy.deepcopy(frozen),
            )

    def test_rejects_actual_frozen_boundary_drift(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        frozen = copy.deepcopy(control)
        actual = copy.deepcopy(frozen)
        actual["different_elements"] = 3
        actual["different_fraction"] = 3.0 / 1536
        actual["sum_abs_delta"] = 3.0
        actual["mean_abs_delta"] = 3.0 / 1536
        actual["sum_squared_delta"] = 3.0
        actual["root_mean_square_delta"] = math.sqrt(3.0 / 1536)
        actual["normalized_root_mean_square_delta"] = (
            math.sqrt(3.0 / 1536) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_BOUNDARY_NOT_REPRODUCED",
        ):
            self._classify_with(
                plan=_plan(),
                runs=_runs(control),
                control=control,
                frozen=frozen,
                actual=actual,
            )

    def test_rejects_actual_output_comparison_link_drift(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[0]["actual_output_float32_sha256"] = _digest(99)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ACTUAL_OUTPUT_COMPARISON_LINK_FAILED",
        ):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def test_rejects_cache_bearing_control(self) -> None:
        self._assert_run_safety_rejected("control_cache_arguments_present", True)

    def test_rejects_generation_output_injection(self) -> None:
        self._assert_run_safety_rejected("control_output_injected", True)

    def test_rejects_module_tensor_payload(self) -> None:
        self._assert_run_safety_rejected("module_tensor_payload_saved", True)

    def test_rejects_module_tensor_sidecar(self) -> None:
        self._assert_run_safety_rejected("module_tensor_sidecar_saved", True)

    def test_rejects_autocast(self) -> None:
        self._assert_run_safety_rejected("autocast_enabled", True)

    def test_rejects_comparison_algebra_forgery(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["mean_abs_delta"] = 1.0
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_BOUNDARY_MEAN"):
            _classify(control)

    def test_rejects_same_index_witness_inconsistency(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["first_different_flat_index"] = 1
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_SAME_INDEX_WITNESS_MISMATCH",
        ):
            _classify(control)

    def test_rejects_sum_abs_below_both_registered_witnesses(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["sum_abs_delta"] = 1.5
        control["mean_abs_delta"] = 1.5 / 1536
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_SUM_ABS_LOWER_BOUND",
        ):
            _classify(control)

    def test_rejects_sum_squared_below_both_registered_witnesses(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["sum_squared_delta"] = 1.5
        control["root_mean_square_delta"] = math.sqrt(1.5 / 1536)
        control["normalized_root_mean_square_delta"] = (
            math.sqrt(1.5 / 1536) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_SUM_SQUARED_LOWER_BOUND",
        ):
            _classify(control)

    def test_rejects_l1_l2_max_bound_violation(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["fp32_value_at_first_difference"] = 1.2
        control["sum_abs_delta"] = 1.2
        control["mean_abs_delta"] = 1.2 / 1536
        control["sum_squared_delta"] = 1.5
        control["root_mean_square_delta"] = math.sqrt(1.5 / 1536)
        control["normalized_root_mean_square_delta"] = (
            math.sqrt(1.5 / 1536) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_L1_L2_MAX_BOUND_MISMATCH",
        ):
            _classify(control)

    def test_rejects_cauchy_bound_violation(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["fp32_value_at_first_difference"] = 1.2
        control["sum_abs_delta"] = 1.8
        control["mean_abs_delta"] = 1.8 / 1536
        control["sum_squared_delta"] = 1.04
        control["root_mean_square_delta"] = math.sqrt(1.04 / 1536)
        control["normalized_root_mean_square_delta"] = (
            math.sqrt(1.04 / 1536) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_CAUCHY_BOUND_MISMATCH",
        ):
            _classify(control)

    def test_rejects_endpoint_rms_below_witness_bound(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["bf16_root_mean_square"] = 0.01
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_ENDPOINT_RMS_WITNESS_BOUND",
        ):
            _classify(control)

    def test_rejects_reverse_triangle_violation(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["bf16_root_mean_square"] = 10.0
        control["normalized_root_mean_square_delta"] = (
            control["root_mean_square_delta"] / 10.0  # type: ignore[operator]
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_REVERSE_TRIANGLE_MISMATCH",
        ):
            _classify(control)

    def test_rejects_triangle_violation(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control.update(
            {
                "different_elements": 1536,
                "different_fraction": 1.0,
                "bf16_value_at_first_difference": 0.0,
                "fp32_value_at_first_difference": 1.0,
                "bf16_value_at_max_abs_delta": 0.0,
                "fp32_value_at_max_abs_delta": 1.0,
                "sum_abs_delta": 1536.0,
                "mean_abs_delta": 1.0,
                "sum_squared_delta": 1536.0,
                "root_mean_square_delta": 1.0,
                "bf16_root_mean_square": 0.4,
                "fp32_root_mean_square": 0.4,
                "normalized_root_mean_square_delta": 2.5,
            }
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "BOUNDARY_TRIANGLE_MISMATCH",
        ):
            _classify(control)

    def test_rejects_normalized_rms_forgery(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control["normalized_root_mean_square_delta"] = 1.0
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_BOUNDARY_NORMALIZED_RMS",
        ):
            _classify(control)

    def test_rejects_first_difference_ratio_forgery(self) -> None:
        control = _comparison(equal=False, bf16=CONTROL_BF16, fp32=CONTROL_FP32)
        control[
            "root_mean_square_delta_ratio_to_first_registered_difference"
        ] = 2.0
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_BOUNDARY_FIRST_DIFFERENCE_RATIO",
        ):
            _classify(control)

    def test_rejects_equal_endpoint_rms_mismatch(self) -> None:
        control = _comparison(equal=True, bf16=CONTROL_BF16, fp32=CONTROL_BF16)
        control["fp32_root_mean_square"] = 3.0
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "EQUAL_BOUNDARY_RMS_MISMATCH",
        ):
            _classify(control)

    def test_rejects_same_endpoint_summary_forgery(self) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        control["different_elements"] = 3
        control["different_fraction"] = 3.0 / 1536
        control["sum_abs_delta"] = 3.0
        control["mean_abs_delta"] = 3.0 / 1536
        control["sum_squared_delta"] = 3.0
        control["root_mean_square_delta"] = math.sqrt(3.0 / 1536)
        control["normalized_root_mean_square_delta"] = (
            math.sqrt(3.0 / 1536) / 2.0
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SAME_ENDPOINT_COMPARISON_SUMMARY_MISMATCH",
        ):
            _classify(control)

    def test_rejects_invalid_digest(self) -> None:
        plan = _plan()
        plan["common_input_float32_sha256"] = "not-a-digest"
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_SHA256"):
            self._classify_with(plan=plan, runs=_runs(control), control=control)

    def _assert_run_safety_rejected(self, field: str, value: object) -> None:
        control = _comparison(equal=False, bf16=ACTUAL_BF16, fp32=ACTUAL_FP32)
        runs = _runs(control)
        runs[0][field] = value
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_SAFETY_BOUNDARY_VIOLATION",
        ):
            self._classify_with(plan=_plan(), runs=runs, control=control)

    def _classify_with(
        self,
        *,
        plan: dict[str, object],
        runs: list[dict[str, object]],
        control: dict[str, object],
        frozen: dict[str, object] | None = None,
        actual: dict[str, object] | None = None,
    ) -> dict[str, object]:
        frozen_value = frozen or _comparison(
            equal=False,
            bf16=ACTUAL_BF16,
            fp32=ACTUAL_FP32,
        )
        actual_value = actual or copy.deepcopy(frozen_value)
        return classify_attached_dtype_boundary_control(
            control_plan=plan,
            runs=runs,
            frozen_boundary_comparison=frozen_value,
            actual_boundary_comparison=actual_value,
            control_boundary_comparison=control,
            source_evidence_locked=True,
            target_forward_identity_preserved=True,
            attached_execution_form_fixed=True,
            checkpoint_sources_unchanged=True,
        )


if __name__ == "__main__":
    unittest.main()
