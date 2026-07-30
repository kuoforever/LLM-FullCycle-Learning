from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import ToolRouterValidationError, load_fixture
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "fixtures" / "tool_router_v1" / "eval.json"
CONFIG = ROOT / "configs" / "tool_router_base_eval_v1.json"
PREDICTIONS = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-instruct-predictions.json"
REPORT = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-instruct-report.json"
BASELINE = ROOT / "baseline" / "fc-mvp-001-base-model-v1.json"


class ToolRouterModelEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_fixture(EVAL)

    def assert_code(self, expected: str, function, *args) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, expected)

    def test_strict_json_output_is_scored_and_invalid_output_is_penalized(self) -> None:
        records = self.records[:2]
        outputs = [
            {
                "example_id": records[0]["example_id"],
                "raw_output": json.dumps(records[0]["decision"]),
            },
            {
                "example_id": records[1]["example_id"],
                "raw_output": "```json\n{}\n```",
            },
        ]

        report, parsed = score_raw_outputs(records, outputs)

        self.assertEqual(report["records"], 2)
        self.assertEqual(report["valid_outputs"], 1)
        self.assertEqual(report["invalid_outputs"], 1)
        self.assertEqual(report["json_validity"], 0.5)
        self.assertEqual(report["decision_semantic_validity"], 0.5)
        self.assertEqual(report["tool_accuracy"], 0.5)
        self.assertEqual(report["argument_exact_match"], 0.5)
        self.assertEqual(report["approval_accuracy"], 0.5)
        self.assertEqual(report["parse_failure_counts"], {"MALFORMED_MODEL_JSON": 1})
        self.assertTrue(parsed[0]["valid"])
        self.assertFalse(parsed[1]["valid"])

    def test_wrong_tool_and_extra_keys_are_invalid_json_contract_outputs(self) -> None:
        record = self.records[0]
        unavailable = dict(record["decision"])
        unavailable["selected_tool"] = "computer_use"
        extra = dict(record["decision"])
        extra["commentary"] = "not allowed"

        unavailable_report, _ = score_raw_outputs(
            [record],
            [
                {
                    "example_id": record["example_id"],
                    "raw_output": json.dumps(unavailable),
                }
            ],
        )
        extra_report, _ = score_raw_outputs(
            [record],
            [
                {
                    "example_id": record["example_id"],
                    "raw_output": json.dumps(extra),
                }
            ],
        )

        self.assertEqual(
            unavailable_report["parse_failure_counts"],
            {"TOOL_NOT_AVAILABLE": 1},
        )
        self.assertEqual(
            extra_report["parse_failure_counts"],
            {"MODEL_OUTPUT_INVALID_KEYS": 1},
        )

    def test_dangerous_invalid_output_is_not_treated_as_an_approval(self) -> None:
        record = next(
            item for item in self.records if item["category"] == "dangerous_request"
        )
        report, _ = score_raw_outputs(
            [record],
            [{"example_id": record["example_id"], "raw_output": "not-json"}],
        )

        self.assertEqual(report["dangerous_false_approvals"], 0)
        self.assertEqual(report["dangerous_action_candidates"], 0)
        self.assertEqual(report["dangerous_invalid_outputs"], 1)
        self.assertEqual(report["approval_accuracy"], 0.0)

    def test_output_count_and_identifier_mismatches_fail_closed(self) -> None:
        record = self.records[0]
        self.assert_code("MODEL_OUTPUT_COUNT_MISMATCH", score_raw_outputs, [record], [])
        self.assert_code(
            "MODEL_OUTPUT_ID_MISMATCH",
            score_raw_outputs,
            [record],
            [{"example_id": "eval-wrong", "raw_output": "{}"}],
        )

    def test_frozen_base_model_report_reproduces_without_model_imports(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        artifact = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
        expected = json.loads(REPORT.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        outputs = [
            {
                "example_id": item["example_id"],
                "raw_output": item["raw_output"],
            }
            for item in artifact["outputs"]
        ]

        metrics, parsed = score_raw_outputs(self.records, outputs)

        self.assertEqual(artifact["model"], config["model"])
        self.assertEqual(artifact["generation"], config["generation"])
        self.assertEqual(metrics, expected["metrics"])
        self.assertEqual(parsed, expected["parsed_outputs"])
        self.assertEqual(metrics, baseline["metrics"])
        self.assertEqual(
            "sha256:" + hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
            expected["prediction_artifact_sha256"],
        )
        self.assertFalse(baseline["runtime_eligible"])
        self.assertEqual(metrics["dangerous_action_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
