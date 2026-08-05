from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge import (
    tool_router_attached_dtype_boundary_control_evidence as evidence_validator,
)
from fullcycle_bridge.tool_router_attached_dtype_boundary_control_evidence import (
    EXPECTED_NUMERICS_EVIDENCE_SHA256,
    validate_attached_dtype_boundary_control_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    ROOT
    / "baseline"
    / "fc-mvp-001-attached-dtype-boundary-control-v1.json"
)
NUMERICS_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-numerics-v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(path)
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _reseal_record(record: dict[str, Any]) -> None:
    payload = dict(record)
    payload.pop("record_sha256")
    record["record_sha256"] = _canonical_sha256(payload)


def _reseal_run(run: dict[str, Any]) -> None:
    run["capture_manifest_sha256"] = _canonical_sha256(run["capture_records"])


class AttachedDtypeBoundaryControlEvidenceTests(unittest.TestCase):
    evidence: dict[str, Any]
    numerics: dict[str, Any]
    lineage: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = _load(EVIDENCE_PATH)
        cls.numerics = _load(NUMERICS_PATH)
        cls.lineage = dict(cls.numerics["source_lineage"])
        cls.lineage["attached_dtype_numerics_evidence_sha256"] = (
            EXPECTED_NUMERICS_EVIDENCE_SHA256
        )

    def _validate(
        self,
        value: object,
        *,
        source_sha256: str = EXPECTED_NUMERICS_EVIDENCE_SHA256,
    ) -> dict[str, Any]:
        return validate_attached_dtype_boundary_control_evidence(
            value,
            source_numerics=self.numerics,
            expected_source_numerics_sha256=source_sha256,
            expected_source_lineage=self.lineage,
            expected_adapter_files=self.numerics["adapter_files"],
            expected_environment=self.numerics["environment"],
        )

    @staticmethod
    def _reseal_comparison_family(value: dict[str, Any], family: str) -> str:
        manifest = _canonical_sha256(value[f"{family}_comparisons"])
        value[f"{family}_comparison_manifest_sha256"] = manifest
        paired = value[f"{family}_paired_comparison_repeat"]
        paired["representative_manifest_sha256"] = manifest
        paired["repeat_manifest_sha256"] = manifest
        return manifest

    def test_frozen_evidence_passes(self) -> None:
        self.assertEqual(
            self._validate(self.evidence),
            {
                "frozen_gate_valid": True,
                "actual_runs_validated": 4,
                "control_runs_validated": 4,
                "capture_records_validated": 28,
                "actual_comparisons_validated": 4,
                "control_comparisons_validated": 3,
                "protocol_completed": True,
                "current_forward_boundary_sufficiency_observed": True,
                "classification": (
                    "deterministic_same_values_rmsnorm_dtype_replay_"
                    "reproduces_actual_boundary_drift"
                ),
            },
        )

    def test_rejects_unknown_top_level_field(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["tensor_archive"] = {"forbidden": True}
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_EVIDENCE_SCHEMA"):
            self._validate(value)

    def test_rejects_wrong_upstream_numerics_digest(self) -> None:
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SOURCE_NUMERICS_DIGEST_MISMATCH",
        ):
            self._validate(self.evidence, source_sha256="sha256:" + "0" * 64)

    def test_rejects_source_lineage_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["source_lineage"]["attached_dtype_numerics_evidence_sha256"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(ToolRouterValidationError, "SOURCE_CONTRACT_MISMATCH"):
            self._validate(value)

    def test_rejects_control_plan_digest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_plan"]["intervention_count"] = 2
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_CONTROL_PLAN_DRIFT|CONTROL_PLAN_DIGEST_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_checkpoint_embedding_selector_drift(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["common_source"]["embedding_row_index"] = 789
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "COMMON_SOURCE_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_common_source_record_digest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["common_source"]["records"][0]["canonical_float32_sha256"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CAPTURE_RECORD_DIGEST_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_actual_abba_run_reordering(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["actual_runs"][0]["order_index"] = 1
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ACTUAL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_control_abba_run_reordering(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][1]["path"] = "bf16_attached_adapter"
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_actual_capture_event_reordering(self) -> None:
        value = copy.deepcopy(self.evidence)
        records = value["actual_runs"][0]["capture_records"]
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CAPTURE_RECORD_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_control_capture_manifest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][0]["capture_manifest_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ToolRouterValidationError, "CAPTURE_MANIFEST_MISMATCH"):
            self._validate(value)

    def test_rejects_common_actual_input_link_drift_after_resealing(self) -> None:
        value = copy.deepcopy(self.evidence)
        for run in (value["actual_runs"][1], value["actual_runs"][2]):
            record = run["capture_records"][1]
            record["canonical_float32_sha256"] = "sha256:" + "9" * 64
            _reseal_record(record)
            _reseal_run(run)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ACTUAL_CAPTURE_REPEAT_MISMATCH|COMPARISON_CAPTURE_LINK_MISMATCH|COMMON_INPUT_SOURCE_LINK_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_actual_control_native_output_mismatch(self) -> None:
        value = copy.deepcopy(self.evidence)
        for run in (value["control_runs"][0], value["control_runs"][3]):
            output = run["capture_records"][2]
            output["native_payload_sha256"] = "sha256:" + "8" * 64
            _reseal_record(output)
            _reseal_run(run)
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ACTUAL_CONTROL_NATIVE_OUTPUT_LINK_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_forward_source_drift(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][0]["forward_source_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_epsilon_drift(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][0]["variance_epsilon"] = 1e-5
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_cache_bearing_control(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][0]["cache_arguments_present"] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_actual_output_injection(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["control_runs"][0]["output_injected"] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_actual_comparison_manifest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["actual_comparisons"][0]["bf16_root_mean_square"] = 0.0
        with self.assertRaisesRegex(ToolRouterValidationError, "COMPARISON_MANIFEST_MISMATCH"):
            self._validate(value)

    def test_rejects_resealed_upstream_embedding_summary_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        comparison = value["actual_comparisons"][0]
        comparison["bf16_root_mean_square"] = 0.125
        comparison["fp32_root_mean_square"] = 0.125
        manifest = self._reseal_comparison_family(value, "actual")
        with mock.patch.object(
            evidence_validator,
            "EXPECTED_ACTUAL_COMPARISON_MANIFEST_SHA256",
            manifest,
        ):
            with self.assertRaisesRegex(
                ToolRouterValidationError,
                "ACTUAL_EMBEDDING_SOURCE_MISMATCH",
            ):
                self._validate(value)

    def test_rejects_resealed_embedding_input_summary_drift(self) -> None:
        value = copy.deepcopy(self.evidence)
        comparison = value["actual_comparisons"][1]
        comparison["bf16_root_mean_square"] = 0.125
        comparison["fp32_root_mean_square"] = 0.125
        manifest = self._reseal_comparison_family(value, "actual")
        with mock.patch.object(
            evidence_validator,
            "EXPECTED_ACTUAL_COMPARISON_MANIFEST_SHA256",
            manifest,
        ):
            with self.assertRaisesRegex(
                ToolRouterValidationError,
                "ACTUAL_INPUT_COMPARISON_MISMATCH",
            ):
                self._validate(value)

    def test_rejects_resealed_control_source_summary_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        for comparison, forged_rms in zip(
            value["control_comparisons"][:2],
            (0.125, 0.25),
            strict=True,
        ):
            comparison["bf16_root_mean_square"] = forged_rms
            comparison["fp32_root_mean_square"] = forged_rms
        self._reseal_comparison_family(value, "control")
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CONTROL_SOURCE_COMPARISON_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_frozen_boundary_drift(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["frozen_boundary_comparison"]["different_elements"] = 1
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_BOUNDARY_SOURCE_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_classification_run_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["classification_runs"][0]["control_standalone"] = False
        with self.assertRaisesRegex(ToolRouterValidationError, "CLASSIFICATION_RUNS_MISMATCH"):
            self._validate(value)

    def test_rejects_boundary_analysis_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["boundary_analysis"]["current_forward_boundary_sufficiency_observed"] = False
        with self.assertRaisesRegex(ToolRouterValidationError, "BOUNDARY_ANALYSIS_MISMATCH"):
            self._validate(value)

    def test_rejects_causal_overclaim(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["causal_scope"]["supports"] = "unique CUDA kernel root cause"
        with self.assertRaisesRegex(ToolRouterValidationError, "FROZEN_POLICY_MISMATCH"):
            self._validate(value)

    def test_rejects_payload_policy_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["module_tensor_payload_saved"] = True
        with self.assertRaisesRegex(ToolRouterValidationError, "FROZEN_POLICY_MISMATCH"):
            self._validate(value)

    def test_rejects_locked_next_action_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["locked_next_action"]["eligible_to_start"] = False
        with self.assertRaisesRegex(ToolRouterValidationError, "FROZEN_POLICY_MISMATCH"):
            self._validate(value)

    def test_next_gate_requires_and_allows_the_full_eval(self) -> None:
        self.assertIs(self.evidence["constraints"]["full_eval_run"], False)
        next_action = self.evidence["locked_next_action"]
        self.assertIs(next_action["eligible_to_start"], True)
        self.assertIs(
            next_action["acceptance"]["matched_boundary_control_required"],
            True,
        )
        self.assertIs(next_action["constraints"]["full_eval_run"], True)

    def test_next_gate_remains_ineligible_without_a_matched_control(self) -> None:
        current_constraints = {"full_eval_run": False}
        next_action = evidence_validator._expected_locked_next_action(  # noqa: SLF001
            {"current_forward_boundary_sufficiency_observed": False},
            current_constraints,
        )
        self.assertIs(next_action["eligible_to_start"], False)
        self.assertIs(
            next_action["acceptance"]["matched_boundary_control_required"],
            True,
        )
        self.assertIs(next_action["constraints"]["full_eval_run"], True)
        self.assertIs(current_constraints["full_eval_run"], False)

    def test_rejects_nonfinite_resource(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ToolRouterValidationError, "NONFINITE_VALUE"):
            self._validate(value)

    def test_rejects_boolean_token_id(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["actual_runs"][0]["generated_token_ids"][0] = True
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_TOKEN_IDS"):
            self._validate(value)

    def test_contains_no_serialized_module_tensor_payload(self) -> None:
        self.assertFalse(self.evidence["module_tensor_payload_saved"])
        self.assertFalse(self.evidence["module_tensor_sidecar_allowed"])
        forbidden = {"tensor", "offset", "length", "archive"}
        for run in [*self.evidence["actual_runs"], *self.evidence["control_runs"]]:
            for record in run["capture_records"]:
                self.assertTrue(forbidden.isdisjoint(record))


if __name__ == "__main__":
    unittest.main()
