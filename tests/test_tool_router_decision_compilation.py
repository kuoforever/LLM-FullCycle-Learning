from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import ToolRouterValidationError, fixture_digest, load_fixture
from fullcycle_bridge.tool_router_decision_compilation import (
    compile_decision,
    compile_frozen_v2_outputs,
)
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
PREDICTIONS = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
SOURCE_REPORT = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
CLASSIFICATION = BASELINE / "fc-mvp-001-lora-sft-v2-failure-classification.json"
COMPILED = BASELINE / "tool-router-lora-sft-v2-compiled-predictions.json"
COMPILED_REPORT = BASELINE / "tool-router-lora-sft-v2-compiled-report.json"
GATE = BASELINE / "fc-mvp-001-decision-compilation-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class ToolRouterDecisionCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = _load(PREDICTIONS)
        self.report = _load(SOURCE_REPORT)
        self.classification = _load(CLASSIFICATION)
        self.hashes = {
            "predictions": _sha256(PREDICTIONS),
            "report": _sha256(SOURCE_REPORT),
            "classification": _sha256(CLASSIFICATION),
        }

    def test_terminal_dispositions_compile_to_one_consistent_mode(self) -> None:
        template = {
            "selected_tool": "file_read",
            "arguments": {"path": "C:\\work\\note.txt"},
            "risk_level": "low",
            "requires_approval": False,
            "should_reject": True,
            "should_fallback": True,
            "expected_result": "rejection",
        }
        cases = {
            "reject_request": ("rejection", True, False, False),
            "fallback_to_strong_model": ("fallback", False, True, False),
            "request_clarification": ("clarification", False, False, False),
            "file_read": ("tool_candidate", False, False, False),
        }
        for tool, expected in cases.items():
            with self.subTest(tool=tool):
                decision = compile_decision({**template, "selected_tool": tool})
                self.assertEqual(
                    (
                        decision["expected_result"],
                        decision["should_reject"],
                        decision["should_fallback"],
                        decision["requires_approval"],
                    ),
                    expected,
                )
        approved = compile_decision(
            {**template, "selected_tool": "file_write", "requires_approval": True}
        )
        self.assertEqual(approved["expected_result"], "approval_required")
        self.assertTrue(approved["requires_approval"])

    def test_frozen_compilation_and_scoring_match_exact_gate(self) -> None:
        compiled = compile_frozen_v2_outputs(
            self.predictions,
            self.report,
            self.classification,
            self.hashes,
        )
        self.assertEqual(compiled, _load(COMPILED))
        self.assertEqual(
            compiled["changed_example_ids"], ["eval-001", "eval-014", "eval-020"]
        )
        evaluation = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "eval.json")
        self.assertEqual(fixture_digest(evaluation), compiled["eval_digest"])
        outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in compiled["outputs"]
        ]
        metrics, parsed = score_raw_outputs(evaluation, outputs)
        frozen_report = _load(COMPILED_REPORT)
        gate = _load(GATE)
        self.assertEqual(metrics, frozen_report["metrics"])
        self.assertEqual(parsed, frozen_report["parsed_outputs"])
        self.assertEqual(metrics, gate["metrics"])
        self.assertEqual(metrics["semantic_failure_counts"], {})
        self.assertEqual(metrics["false_refusals"], 0)
        self.assertEqual(metrics["dangerous_action_candidates"], 0)
        self.assertFalse(gate["runtime_eligible"])
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],  # type: ignore[index]
            "FC-MVP-001-bf16-merge-stability-v1",
        )

    def test_raw_prediction_drift_from_frozen_report_fails_closed(self) -> None:
        predictions = copy.deepcopy(self.predictions)
        predictions["outputs"][0]["raw_output"] = "{}"  # type: ignore[index]
        with self.assertRaises(ToolRouterValidationError) as raised:
            compile_frozen_v2_outputs(
                predictions,
                self.report,
                self.classification,
                self.hashes,
            )
        self.assertEqual(raised.exception.code, "PARSED_OUTPUT_DRIFT")

    def test_classification_next_gate_drift_fails_closed(self) -> None:
        classification = copy.deepcopy(self.classification)
        classification["locked_next_action"]["gate_id"] = "unexpected"  # type: ignore[index]
        with self.assertRaises(ToolRouterValidationError) as raised:
            compile_frozen_v2_outputs(
                self.predictions,
                self.report,
                classification,
                self.hashes,
            )
        self.assertEqual(raised.exception.code, "CLASSIFICATION_DIGEST_MISMATCH")


if __name__ == "__main__":
    unittest.main()
