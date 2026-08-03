from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_failure_classification import classify_v2_failures

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
PREDICTIONS = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
REPORT = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
TRAINING = BASELINE / "fc-mvp-001-lora-sft-v2-training.json"
LOAD_MERGE = BASELINE / "fc-mvp-001-lora-sft-v2-load-merge.json"
CLASSIFICATION = BASELINE / "fc-mvp-001-lora-sft-v2-failure-classification.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class ToolRouterFailureClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = _load(PREDICTIONS)
        self.report = _load(REPORT)
        self.training = _load(TRAINING)
        self.load_merge = _load(LOAD_MERGE)
        self.hashes = {
            "predictions": _sha256(PREDICTIONS),
            "report": _sha256(REPORT),
            "training": _sha256(TRAINING),
            "load_merge": _sha256(LOAD_MERGE),
        }

    def classify(self) -> dict[str, object]:
        return classify_v2_failures(
            self.predictions,
            self.report,
            self.training,
            self.load_merge,
            self.hashes,
        )

    def test_frozen_evidence_reproduces_exact_classification(self) -> None:
        result = self.classify()
        self.assertEqual(result, _load(CLASSIFICATION))
        self.assertEqual(
            result["category_groups"],
            {
                "data_coverage": [],
                "decision_contract_consistency": [
                    "conflicting_decision_flags",
                    "false_refusals",
                ],
                "bf16_adapter_merge_stability": ["load_merge_output_drift"],
            },
        )
        self.assertFalse(result["runtime_eligible"])

    def test_prediction_digest_drift_fails_closed(self) -> None:
        hashes = dict(self.hashes)
        hashes["predictions"] = "sha256:" + "0" * 64
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_v2_failures(
                self.predictions,
                self.report,
                self.training,
                self.load_merge,
                hashes,
            )
        self.assertEqual(raised.exception.code, "PREDICTION_DIGEST_MISMATCH")

    def test_unattributed_false_refusal_fails_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["metrics"]["false_refusals"] = 4  # type: ignore[index]
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_v2_failures(
                self.predictions,
                report,
                self.training,
                self.load_merge,
                self.hashes,
            )
        self.assertEqual(raised.exception.code, "UNATTRIBUTABLE_FALSE_REFUSALS")

    def test_unexpected_merge_drift_fails_closed(self) -> None:
        load_merge = copy.deepcopy(self.load_merge)
        merged = json.loads(str(load_merge["merged_output"]))
        merged["risk_level"] = "low"
        load_merge["merged_output"] = json.dumps(merged, sort_keys=True)
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_v2_failures(
                self.predictions,
                self.report,
                self.training,
                load_merge,
                self.hashes,
            )
        self.assertEqual(raised.exception.code, "UNEXPECTED_MERGE_DRIFT")


if __name__ == "__main__":
    unittest.main()
