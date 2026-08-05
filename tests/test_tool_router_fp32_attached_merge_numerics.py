from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_fp32_attached_merge_numerics import (
    COMMON_OUTPUT_STAGES,
    FIRST_REGISTERED_BOUNDARY,
    analyze_module_comparisons,
    classify_operation_order,
)
from fullcycle_bridge.tool_router_fp32_attached_merge_numerics_archive import (
    validate_frozen_numerics_evidence,
    validate_raw_tensor_archive,
)

ZERO = "sha256:" + "0" * 64
ONE = "sha256:" + "1" * 64
ROOT = Path(__file__).resolve().parents[1]
FROZEN_EVIDENCE = (
    ROOT / "baseline" / "fc-mvp-001-fp32-attached-merge-numerics-v1.json"
)
FROZEN_TENSORS = FROZEN_EVIDENCE.with_name(
    "fc-mvp-001-fp32-attached-merge-numerics-v1-tensors.bin"
)

OPERATION_PAIRS = {
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


def _comparison(
    name: str,
    equal: bool,
    *,
    left_tensor_id: str | None = None,
    right_tensor_id: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "shape": [1, 1, 4],
        "dtype": "float32",
        "elements": 4,
        "numerically_equal": equal,
        "bitwise_equal": equal,
        "different_elements": 0 if equal else 3,
        "bitwise_different_elements": 0 if equal else 3,
        "first_different_flat_index": None if equal else 0,
        "max_abs_delta_flat_index": None if equal else 2,
        "left_value_at_first_difference": None if equal else 1.0,
        "right_value_at_first_difference": None if equal else 1.25,
        "left_value_at_max_abs_delta": None if equal else 2.0,
        "right_value_at_max_abs_delta": None if equal else 3.0,
        "max_abs_delta": 0.0 if equal else 1.0,
        "mean_abs_delta": 0.0 if equal else 0.3125,
        "root_mean_square_delta": 0.0 if equal else 0.5153882032022076,
        "left_tensor_id": left_tensor_id or f"left.{name}",
        "right_tensor_id": right_tensor_id or f"right.{name}",
        "left_raw_sha256": ZERO,
        "right_raw_sha256": ZERO if equal else ONE,
        "left_canonical_sha256": ZERO,
        "right_canonical_sha256": ZERO if equal else ONE,
    }


def _modules() -> list[dict[str, object]]:
    return [
        _comparison(
            stage,
            stage != FIRST_REGISTERED_BOUNDARY,
            left_tensor_id=f"fp32-attached-numerics-r1|common|{stage}|output",
            right_tensor_id=(
                f"fp32-safe-merged-numerics-r1|common|{stage}|output"
            ),
        )
        for stage in COMMON_OUTPUT_STAGES
    ]


def _operations(
    *,
    factorized_drift: bool = True,
    materialized_drift: bool = True,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, (left, right) in OPERATION_PAIRS.items():
        equal = name not in {
            "attached_output_vs_merged_output",
            *(
                [
                    "factorized_lora_vs_delta_weight_linear",
                    "attached_factorized_output_vs_split_delta_output",
                ]
                if factorized_drift
                else []
            ),
            *(
                ["split_base_plus_delta_vs_materialized_weight_linear"]
                if materialized_drift
                else []
            ),
        }
        results.append(
            _comparison(
                name,
                equal,
                left_tensor_id=left,
                right_tensor_id=right,
            )
        )
    return results


def _weight_audit() -> dict[str, object]:
    return {
        "name": FIRST_REGISTERED_BOUNDARY,
        "shape": [2, 2],
        "dtype": "float32",
        "elements": 4,
        "base_weight_sha256": ZERO,
        "delta_weight_sha256": ONE,
        "expected_merged_weight_sha256": ZERO,
        "actual_merged_weight_sha256": ZERO,
        "expected_actual_equal": True,
        "actual_merged_mismatched_weights": 0,
        "ideal_nonzero_updates": 4,
        "effective_changed_weights": 3,
        "ideal_nonzero_updates_rounded_to_base": 1,
        "max_abs_materialization_error": 0.125,
        "mean_abs_materialization_error": 0.03125,
        "bias_present": True,
        "bias_elements": 2,
        "bias_mismatched_elements": 0,
        "tensor_ids": {
            "base_weight": "base-weight",
            "delta_weight": "delta-weight",
            "expected_merged_weight": "expected-weight",
            "actual_merged_weight": "actual-weight",
            "attached_bias": "attached-bias",
            "merged_bias": "merged-bias",
        },
    }


class ToolRouterFp32AttachedMergeNumericsTests(unittest.TestCase):
    def test_first_paired_common_module_divergence_is_recomputed(self) -> None:
        analysis = analyze_module_comparisons(_modules())
        self.assertEqual(analysis["module_count"], len(COMMON_OUTPUT_STAGES))
        self.assertEqual(analysis["first_divergent_module_index"], 2)
        self.assertEqual(
            analysis["first_divergent_module"],
            FIRST_REGISTERED_BOUNDARY,
        )
        self.assertTrue(analysis["preceding_modules_identical"])

    def test_module_identity_remains_distinct_from_a_divergence_claim(self) -> None:
        analysis = analyze_module_comparisons(
            [_comparison("model.embed_tokens", True)]
        )
        self.assertEqual(
            analysis["classification"],
            "paired_common_module_output_identity",
        )
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_operation_order(
                [_comparison("model.embed_tokens", True)],
                _operations(),
                _weight_audit(),
            )
        self.assertEqual(raised.exception.code, "INVALID_COMMON_MODULE_SEQUENCE")

    def test_both_registered_operation_forms_are_classified(self) -> None:
        result = classify_operation_order(
            _modules(),
            _operations(),
            _weight_audit(),
        )
        self.assertEqual(
            result["classification"],
            (
                "deterministic_fp32_factorized_lora_and_materialized_linear_"
                "execution_form_drift"
            ),
        )
        self.assertTrue(
            result["registered_execution_form_boundary_quantified"]
        )
        self.assertTrue(
            result["factorized_lora_term_vs_delta_weight_linear_drift"]
        )
        self.assertTrue(
            result["factorized_output_vs_split_delta_output_drift"]
        )
        self.assertTrue(
            result["split_sum_vs_materialized_weight_linear_drift"]
        )

    def test_each_registered_operation_form_has_a_narrow_classification(self) -> None:
        factorized = classify_operation_order(
            _modules(),
            _operations(materialized_drift=False),
            _weight_audit(),
        )
        self.assertEqual(
            factorized["classification"],
            "deterministic_fp32_factorized_lora_execution_form_drift",
        )
        materialized = classify_operation_order(
            _modules(),
            _operations(factorized_drift=False),
            _weight_audit(),
        )
        self.assertEqual(
            materialized["classification"],
            (
                "deterministic_fp32_split_sum_vs_materialized_linear_"
                "execution_form_drift"
            ),
        )

    def test_comparison_schema_and_numeric_contradictions_fail_closed(self) -> None:
        missing = _comparison("x", True)
        del missing["shape"]
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([missing])
        self.assertEqual(raised.exception.code, "INVALID_COMPARISON_FIELDS")

        boolean_count = _comparison("x", True)
        boolean_count["elements"] = True
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([boolean_count])
        self.assertEqual(
            raised.exception.code,
            "INVALID_COMPARISON_ELEMENT_COUNT",
        )

        nonfinite = _comparison("x", False)
        nonfinite["max_abs_delta"] = math.inf
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([nonfinite])
        self.assertEqual(raised.exception.code, "INVALID_COMPARISON_DELTA")

        forged_equal = _comparison("x", True)
        forged_equal["different_elements"] = 1
        forged_equal["bitwise_equal"] = False
        forged_equal["bitwise_different_elements"] = 1
        forged_equal["right_raw_sha256"] = ONE
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([forged_equal])
        self.assertEqual(raised.exception.code, "INCONSISTENT_EQUAL_COMPARISON")

        zero_drift = _comparison("x", False)
        zero_drift["max_abs_delta"] = 0.0
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([zero_drift])
        self.assertEqual(
            raised.exception.code,
            "INCONSISTENT_COMPARISON_DELTAS",
        )

    def test_duplicate_names_and_invalid_witnesses_fail_closed(self) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons(
                [_comparison("x", True), _comparison("x", True)]
            )
        self.assertEqual(raised.exception.code, "INVALID_COMPARISON_NAME")

        witness = _comparison("x", False)
        witness["first_different_flat_index"] = 4
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([witness])
        self.assertEqual(
            raised.exception.code,
            "INCONSISTENT_UNEQUAL_COMPARISON",
        )

    def test_operation_linkage_and_reconstructions_fail_closed(self) -> None:
        swapped = _operations()
        swapped[0]["left_tensor_id"] = "forged"
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_operation_order(_modules(), swapped, _weight_audit())
        self.assertEqual(
            raised.exception.code,
            "INVALID_OPERATION_TENSOR_PAIR",
        )

        missing_identity = _operations()
        missing_identity[0] = _comparison(
            "q_proj_input_identity",
            False,
            left_tensor_id=OPERATION_PAIRS["q_proj_input_identity"][0],
            right_tensor_id=OPERATION_PAIRS["q_proj_input_identity"][1],
        )
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_operation_order(
                _modules(),
                missing_identity,
                _weight_audit(),
            )
        self.assertEqual(
            raised.exception.code,
            "REQUIRED_OPERATION_IDENTITY_MISSING",
        )

    def test_weight_reproduction_and_registered_explanation_are_required(self) -> None:
        bad_weight = copy.deepcopy(_weight_audit())
        bad_weight["expected_actual_equal"] = False
        bad_weight["actual_merged_mismatched_weights"] = 1
        bad_weight["actual_merged_weight_sha256"] = ONE
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_operation_order(_modules(), _operations(), bad_weight)
        self.assertEqual(
            raised.exception.code,
            "SAFE_MERGE_WEIGHT_REPRODUCTION_FAILED",
        )

        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_operation_order(
                _modules(),
                _operations(
                    factorized_drift=False,
                    materialized_drift=False,
                ),
                _weight_audit(),
            )
        self.assertEqual(
            raised.exception.code,
            "REGISTERED_OPERATIONS_DO_NOT_EXPLAIN_OUTPUT_DRIFT",
        )


def _load_frozen() -> dict[str, Any]:
    value = json.loads(FROZEN_EVIDENCE.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("frozen numerics evidence must be an object")
    return value


def _sha256(payload: bytes | bytearray) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_float32_sha256(raw: bytes) -> str:
    canonical = bytearray(raw)
    for offset in range(0, len(canonical), 4):
        if struct.unpack_from("<I", canonical, offset)[0] == 0x80000000:
            struct.pack_into("<I", canonical, offset, 0)
    return _sha256(canonical)


def _bound_sha256(header: bytes, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _rebind_record(
    gate: dict[str, Any],
    payload: bytearray,
    tensor_id: str,
) -> None:
    record = next(
        item
        for item in gate["tensor_archive"]["records"]
        if item["tensor_id"] == tensor_id
    )
    start = record["byte_offset"]
    end = start + record["byte_length"]
    raw = bytes(payload[start:end])
    record["raw_payload_sha256"] = _sha256(raw)
    record["canonical_value_sha256"] = _canonical_float32_sha256(raw)
    header = dict(record)
    header.pop("bound_sha256")
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record["bound_sha256"] = _bound_sha256(encoded, raw)
    gate["tensor_archive"]["sha256"] = _sha256(payload)


class ToolRouterFp32AttachedMergeNumericsArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = _load_frozen()
        cls.payload = FROZEN_TENSORS.read_bytes()

    def test_frozen_raw_archive_recomputes_every_registered_claim(self) -> None:
        self.assertEqual(
            _sha256(FROZEN_EVIDENCE.read_bytes()),
            "sha256:cb1c2b4255ebc5c38aa2ff66436804cca55dc088e"
            "39ca8fe8959654488e41a91",
        )
        self.assertEqual(
            _sha256(self.payload),
            "sha256:550175dfcfe14b0739aabf17573825a124180a6e"
            "21826e25d4b5ff733fb298a9",
        )
        result = validate_frozen_numerics_evidence(self.gate, self.payload)
        self.assertTrue(result["valid"])
        self.assertTrue(result["frozen_gate_valid"])
        self.assertEqual(result["record_count"], 138)
        self.assertEqual(result["comparisons_recomputed"], 35)
        self.assertEqual(result["weight_elements_recomputed"], 2_359_296)

    def test_next_gate_requires_bf16_rerun_without_reusing_current_facts(
        self,
    ) -> None:
        self.assertFalse(self.gate["constraints"]["frozen_bf16_path_rerun"])
        next_constraints = self.gate["locked_next_action"]["constraints"]
        self.assertNotIn("frozen_bf16_path_rerun", next_constraints)
        self.assertTrue(next_constraints["fresh_bf16_attached_rerun_required"])
        self.assertTrue(next_constraints["fresh_fp32_attached_rerun_required"])
        self.assertFalse(next_constraints["attached_execution_form_change"])

    def test_run_trace_precision_form_and_resources_fail_closed(self) -> None:
        cases: list[tuple[str, dict[str, Any]]] = []
        trace = copy.deepcopy(self.gate)
        trace["runs"][0]["generation_trace"]["scores"]["native_dtypes"] = [
            "bfloat16"
        ]
        cases.append(("INVALID_GENERATION_TRACE_VECTOR", trace))
        shape = copy.deepcopy(self.gate)
        shape["runs"][0]["generation_trace"]["scores"]["shape_per_step"] = [
            999
        ]
        cases.append(("INVALID_GENERATION_TRACE_VECTOR", shape))
        precision = copy.deepcopy(self.gate)
        precision["runs"][0]["precision_audit"]["base_parameters"][
            "dtypes"
        ] = {"bfloat16": 1_543_714_304}
        cases.append(("INVALID_ARCHIVE_RUN_PLAN", precision))
        form = copy.deepcopy(self.gate)
        form["runs"][0]["materialization_form"] = "forged"
        cases.append(("INVALID_ARCHIVE_RUN_PLAN", form))
        resource = copy.deepcopy(self.gate)
        resource["runs"][1]["elapsed_seconds"] = -1.0
        cases.append(("FROZEN_RESOURCE_CLAIM_MISMATCH", resource))
        peak = copy.deepcopy(self.gate)
        peak["runs"][1]["peak_gpu_memory_bytes"] = -1
        cases.append(("FROZEN_RESOURCE_CLAIM_MISMATCH", peak))
        for expected_code, gate in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(ToolRouterValidationError) as raised:
                    validate_frozen_numerics_evidence(gate, self.payload)
                self.assertEqual(raised.exception.code, expected_code)

    def test_descriptor_ranges_and_identity_fail_closed(self) -> None:
        cases: list[tuple[str, Any]] = []
        wrong_path = copy.deepcopy(self.gate)
        wrong_path["tensor_archive"]["path"] = "elsewhere.bin"
        cases.append(("INVALID_ARCHIVE_DESCRIPTOR", wrong_path))
        gap = copy.deepcopy(self.gate)
        gap["tensor_archive"]["records"][1]["byte_offset"] += 4
        cases.append(("INVALID_ARCHIVE_RECORD_RANGE", gap))
        duplicate = copy.deepcopy(self.gate)
        duplicate["tensor_archive"]["records"][1]["tensor_id"] = duplicate[
            "tensor_archive"
        ]["records"][0]["tensor_id"]
        cases.append(("INVALID_ARCHIVE_RECORD_IDENTITY", duplicate))
        for expected_code, gate in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(ToolRouterValidationError) as raised:
                    validate_raw_tensor_archive(gate, self.payload)
                self.assertEqual(raised.exception.code, expected_code)

    def test_rebound_provenance_metadata_fails_closed(self) -> None:
        cases = (
            (
                "fp32-attached-numerics-r1|generation|scores|step45",
                "module_name",
                "forged.generation.module",
            ),
            (
                "fp32-attached-numerics-r1|diagnostic|q_proj|factorized_scaled",
                "event_scope",
                "forged_replay_scope",
            ),
            (
                "fp32-attached-numerics-r1|weight|q_proj|base_weight",
                "semantic_key",
                "forged|weight|meaning",
            ),
            (
                "fp32-attached-numerics-r1|diagnostic|q_proj|base_layer",
                "tensor_path",
                "forged.operation.path",
            ),
            (
                "fp32-safe-merged-numerics-r1|common|"
                "model.layers.0.self_attn.q_proj|output",
                "module_type",
                "forged.common.module",
            ),
        )
        for tensor_id, key, value in cases:
            with self.subTest(tensor_id=tensor_id, key=key):
                gate = copy.deepcopy(self.gate)
                payload = bytearray(self.payload)
                record = next(
                    item
                    for item in gate["tensor_archive"]["records"]
                    if item["tensor_id"] == tensor_id
                )
                record[key] = value
                _rebind_record(gate, payload, tensor_id)
                with self.assertRaises(ToolRouterValidationError) as raised:
                    validate_raw_tensor_archive(gate, bytes(payload))
                self.assertEqual(
                    raised.exception.code,
                    "ARCHIVE_RECORD_METADATA_MISMATCH",
                )

    def test_payload_record_and_finite_integrity_fail_closed(self) -> None:
        tensor_id = (
            "fp32-attached-numerics-r2|common|model.norm|output"
        )
        record = next(
            item
            for item in self.gate["tensor_archive"]["records"]
            if item["tensor_id"] == tensor_id
        )
        raw_tamper = bytearray(self.payload)
        raw_tamper[record["byte_offset"]] ^= 1
        raw_gate = copy.deepcopy(self.gate)
        raw_gate["tensor_archive"]["sha256"] = _sha256(raw_tamper)
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(raw_gate, bytes(raw_tamper))
        self.assertEqual(raised.exception.code, "INVALID_ARCHIVE_RECORD_INTEGRITY")

        bound_gate = copy.deepcopy(self.gate)
        bound_gate["tensor_archive"]["records"][0]["bound_sha256"] = ONE
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(bound_gate, self.payload)
        self.assertEqual(raised.exception.code, "INVALID_ARCHIVE_RECORD_INTEGRITY")

        nonfinite_gate = copy.deepcopy(self.gate)
        nonfinite = bytearray(self.payload)
        struct.pack_into("<f", nonfinite, record["byte_offset"], math.nan)
        _rebind_record(nonfinite_gate, nonfinite, tensor_id)
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(nonfinite_gate, bytes(nonfinite))
        self.assertEqual(raised.exception.code, "INVALID_ARCHIVE_RECORD_INTEGRITY")

    def test_derived_comparison_and_event_forgery_fail_closed(self) -> None:
        forged = copy.deepcopy(self.gate)
        forged["operation_comparisons"][-1]["different_elements"] -= 1
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(forged, self.payload)
        self.assertEqual(
            raised.exception.code,
            "RAW_OPERATION_COMPARISON_MISMATCH",
        )

        event_forgery = copy.deepcopy(self.gate)
        events = event_forgery["runs"][0]["capture_events"]
        events[0], events[1] = events[1], events[0]
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(event_forgery, self.payload)
        self.assertEqual(raised.exception.code, "INVALID_COMMON_CAPTURE_EVENT")

    def test_rebound_r2_tensor_drift_breaks_abba_repeat(self) -> None:
        tensor_id = (
            "fp32-attached-numerics-r2|common|model.norm|output"
        )
        gate = copy.deepcopy(self.gate)
        payload = bytearray(self.payload)
        record = next(
            item
            for item in gate["tensor_archive"]["records"]
            if item["tensor_id"] == tensor_id
        )
        payload[record["byte_offset"]] ^= 1
        _rebind_record(gate, payload, tensor_id)
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(gate, bytes(payload))
        self.assertEqual(raised.exception.code, "RAW_REPEAT_STABILITY_MISMATCH")

    def test_generation_link_and_weight_summary_fail_closed(self) -> None:
        link = copy.deepcopy(self.gate)
        link["runs"][0]["lm_head_raw_logit_linked"] = False
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(link, self.payload)
        self.assertEqual(raised.exception.code, "GENERATION_TENSOR_LINK_MISMATCH")

        weight = copy.deepcopy(self.gate)
        weight["weight_materialization"][
            "ideal_nonzero_updates_rounded_to_base"
        ] += 1
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_raw_tensor_archive(weight, self.payload)
        self.assertEqual(raised.exception.code, "RAW_WEIGHT_AUDIT_MISMATCH")

    def test_final_gate_tokens_alignment_and_acceptance_fail_closed(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["numerics_gate"]["passed"] = False
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_frozen_numerics_evidence(gate, self.payload)
        self.assertEqual(raised.exception.code, "FROZEN_NUMERICS_GATE_MISMATCH")

        acceptance = copy.deepcopy(self.gate)
        acceptance["acceptance"]["safe_merge_weight_reproduced"] = False
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_frozen_numerics_evidence(acceptance, self.payload)
        self.assertEqual(raised.exception.code, "FROZEN_ACCEPTANCE_MISMATCH")

        alignment = copy.deepcopy(self.gate)
        for index in (0, 3):
            alignment["runs"][index]["target_alignment"]["input_token_ids"] = [999]
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_frozen_numerics_evidence(alignment, self.payload)
        self.assertEqual(raised.exception.code, "FROZEN_RUN_CLAIM_MISMATCH")

        tokens = copy.deepcopy(self.gate)
        for index in (0, 3):
            tokens["runs"][index]["generated_token_ids"] = [1]
            tokens["runs"][index]["token_count"] = 1
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_frozen_numerics_evidence(tokens, self.payload)
        self.assertEqual(raised.exception.code, "FROZEN_RUN_CLAIM_MISMATCH")


if __name__ == "__main__":
    unittest.main()
