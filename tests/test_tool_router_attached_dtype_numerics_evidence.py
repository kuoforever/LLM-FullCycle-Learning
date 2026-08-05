from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_attached_dtype_numerics_evidence import (
    validate_attached_dtype_numerics_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-numerics-v1.json"
)
ISOLATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-isolation-v1.json"
)
ISOLATION_SHA256 = (
    "sha256:7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(path)
    return value


class AttachedDtypeNumericsEvidenceTests(unittest.TestCase):
    evidence: dict[str, Any]
    isolation: dict[str, Any]
    lineage: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = _load(EVIDENCE_PATH)
        cls.isolation = _load(ISOLATION_PATH)
        cls.lineage = dict(cls.isolation["source_lineage"])
        cls.lineage["attached_dtype_isolation_evidence_sha256"] = (
            ISOLATION_SHA256
        )

    def _validate(self, value: object) -> dict[str, Any]:
        return validate_attached_dtype_numerics_evidence(
            value,
            source_isolation=self.isolation,
            expected_source_lineage=self.lineage,
            expected_adapter_files=self.isolation["adapter_files"],
            expected_environment=self.isolation["environment"],
        )

    def test_frozen_evidence_passes(self) -> None:
        self.assertEqual(
            self._validate(self.evidence),
            {
                "frozen_gate_valid": True,
                "runs_validated": 4,
                "capture_records_validated": 160,
                "capture_manifests_validated": 4,
                "module_comparisons_validated": 40,
                "first_unequal_module": (
                    "model.layers.0.input_layernorm"
                ),
                "classification": (
                    "deterministic_attached_bf16_vs_fp32_registered_"
                    "module_output_drift_reaching_lm_head"
                ),
                "delta_statistics_scope": (
                    "probe_derived_summary_algebra_and_frozen_manifest_only"
                ),
            },
        )

    def test_rejects_unknown_top_level_field(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["tensor_archive"] = {"forbidden": True}
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_EVIDENCE_SCHEMA",
        ):
            self._validate(value)

    def test_rejects_abba_run_identity_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["runs"][0]["run_id"] = "forged"
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_capture_event_reordering(self) -> None:
        value = copy.deepcopy(self.evidence)
        events = value["runs"][0]["capture_events"]
        events[0], events[1] = events[1], events[0]
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CAPTURE_EVENT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_capture_record_digest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["runs"][0]["capture_records"][0][
            "canonical_float32_sha256"
        ] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "CAPTURE_RECORD_DIGEST_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_capture_manifest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["runs"][0]["capture_manifest_sha256"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_comparison_manifest_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["module_comparisons"][1]["different_elements"] = 1
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "MODULE_COMPARISON_MANIFEST_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_first_boundary_summary_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["module_analysis"]["first_unequal_module"] = (
            "model.layers.0.self_attn.q_proj"
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "MODULE_ANALYSIS_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_lm_head_link_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["runs"][0]["lm_head_raw_logit_linked"] = False
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "RUN_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_paired_repeat_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["paired_comparison_repeat"]["exact_identity"] = False
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "PAIRED_COMPARISON_REPEAT_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_module_payload_policy_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["module_tensor_payload_saved"] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_POLICY_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_causal_overclaim(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["causal_scope"]["supports"] = "unique CUDA root cause"
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FROZEN_POLICY_MISMATCH",
        ):
            self._validate(value)

    def test_rejects_nonfinite_value(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "NONFINITE_VALUE",
        ):
            self._validate(value)

    def test_rejects_boolean_token_id(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["runs"][0]["generated_token_ids"][0] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "INVALID_TOKEN_IDS",
        ):
            self._validate(value)

    def test_rejects_source_lineage_forgery(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["source_lineage"][
            "attached_dtype_isolation_evidence_sha256"
        ] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SOURCE_CONTRACT_MISMATCH",
        ):
            self._validate(value)

    def test_contains_no_serialized_module_tensor_payload(self) -> None:
        self.assertFalse(self.evidence["module_tensor_payload_saved"])
        self.assertFalse(self.evidence["module_tensor_sidecar_allowed"])
        forbidden = {"tensor", "payload", "offset", "length", "archive"}
        for run in self.evidence["runs"]:
            for record in run["capture_records"]:
                self.assertTrue(forbidden.isdisjoint(record))


if __name__ == "__main__":
    unittest.main()
