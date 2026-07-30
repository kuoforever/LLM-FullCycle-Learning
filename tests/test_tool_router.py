from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from fullcycle_bridge.tool_router import (
    CATEGORIES,
    MAX_FIXTURE_BYTES,
    ToolRouterValidationError,
    baseline_predict,
    evaluate,
    fixture_digest,
    load_fixture,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "tool_router_v1"
SEED = FIXTURES / "seed.json"
EVAL = FIXTURES / "eval.json"
SCHEMA = ROOT / "schemas" / "tool_router_decision_v1.schema.json"


class ToolRouterTests(unittest.TestCase):
    def assert_code(self, expected: str, function, *args) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, expected)

    def test_frozen_seed_and_eval_are_strict_balanced_fixtures(self) -> None:
        seed = load_fixture(SEED)
        evaluation = load_fixture(EVAL)

        self.assertEqual(len(seed), 20)
        self.assertEqual(len(evaluation), 20)
        self.assertEqual(
            Counter(item["category"] for item in seed), {key: 2 for key in CATEGORIES}
        )
        self.assertEqual(
            Counter(item["category"] for item in evaluation),
            {key: 2 for key in CATEGORIES},
        )
        self.assertTrue(all(item["split"] == "seed" for item in seed))
        self.assertTrue(all(item["split"] == "eval" for item in evaluation))
        self.assertEqual(
            fixture_digest(seed),
            "sha256:6cd8589b0cc4c0f96cafc3b29378383148fad60732990b6abca5f4ef4e46b95c",
        )
        self.assertEqual(
            fixture_digest(evaluation),
            "sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a",
        )

    def test_machine_schema_is_closed_and_matches_record_topology(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        record = load_fixture(SEED)[0]

        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(record))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        for section in ("state", "decision"):
            contract = schema["$defs"][section]
            self.assertFalse(contract["additionalProperties"])
            self.assertEqual(set(contract["required"]), set(record[section]))

    def test_semantic_modes_are_normalized_and_mutually_consistent(self) -> None:
        for record in load_fixture(SEED) + load_fixture(EVAL):
            decision = record["decision"]
            self.assertEqual(
                decision["should_reject"],
                decision["selected_tool"] == "reject_request",
            )
            self.assertEqual(
                decision["should_fallback"],
                decision["selected_tool"] == "fallback_to_strong_model",
            )
            self.assertEqual(
                decision["requires_approval"],
                decision["expected_result"] == "approval_required",
            )

    def test_unknown_version_extra_fields_and_unavailable_tools_fail_closed(
        self,
    ) -> None:
        record = load_fixture(SEED)[0]
        changed = copy.deepcopy(record)
        changed["schema_version"] = 2
        self.assert_code("UNSUPPORTED_SCHEMA_VERSION", validate_record, changed)

        changed = copy.deepcopy(record)
        changed["provider"] = "forbidden"
        self.assert_code("INVALID_KEYS", validate_record, changed)

        changed = copy.deepcopy(record)
        changed["available_tools"] = ["fallback_to_strong_model"]
        self.assert_code("TOOL_NOT_AVAILABLE", validate_record, changed)

    def test_rich_content_and_invalid_argument_values_fail_closed(self) -> None:
        record = load_fixture(SEED)[0]
        changed = copy.deepcopy(record)
        changed["decision"]["arguments"]["memory"] = "secret"
        self.assert_code("FORBIDDEN_RICH_FIELD", validate_record, changed)

        changed = copy.deepcopy(record)
        changed["decision"]["arguments"]["nested"] = {"unsafe": "object"}
        self.assert_code("INVALID_ARGUMENT_VALUE", validate_record, changed)

    def test_danger_duplicate_failure_loop_and_flag_invariants_fail_closed(
        self,
    ) -> None:
        records = {item["category"]: item for item in load_fixture(EVAL)}
        dangerous = copy.deepcopy(records["dangerous_request"])
        dangerous["decision"] = copy.deepcopy(records["normal_tool_use"]["decision"])
        dangerous["available_tools"].append(dangerous["decision"]["selected_tool"])
        self.assert_code("DANGEROUS_NOT_REJECTED", validate_record, dangerous)

        duplicate = copy.deepcopy(records["duplicate_delivery"])
        duplicate["state"]["duplicate_delivery"] = False
        self.assert_code("DUPLICATE_NOT_REJECTED", validate_record, duplicate)

        failure = copy.deepcopy(records["tool_failure"])
        failure["state"]["tool_failures"] = []
        self.assert_code("TOOL_FAILURE_NOT_HANDLED", validate_record, failure)

        loop = copy.deepcopy(records["loop_limit"])
        loop["state"]["loop_count"] = 0
        self.assert_code("LOOP_LIMIT_NOT_HANDLED", validate_record, loop)

        rejection = copy.deepcopy(records["rejection"])
        rejection["decision"]["should_reject"] = False
        self.assert_code("INCONSISTENT_REJECTION", validate_record, rejection)

    def test_malformed_oversized_and_duplicate_fixture_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.json"
            malformed.write_text("[", encoding="utf-8")
            self.assert_code("MALFORMED_JSON", load_fixture, malformed)

            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * (MAX_FIXTURE_BYTES + 1))
            self.assert_code("FIXTURE_TOO_LARGE", load_fixture, oversized)

            duplicate = root / "duplicate.json"
            record = load_fixture(SEED)[0]
            duplicate.write_text(json.dumps([record, record]), encoding="utf-8")
            self.assert_code("DUPLICATE_EXAMPLE_ID", load_fixture, duplicate)

    def test_rule_baseline_does_not_read_gold_category_or_decision(self) -> None:
        record = load_fixture(EVAL)[0]
        changed = copy.deepcopy(record)
        changed["category"] = "dangerous_request"
        changed["decision"]["selected_tool"] = "reject_request"

        self.assertEqual(baseline_predict(record), baseline_predict(changed))

    def test_baseline_metrics_are_deterministic_and_guard_dangerous_approval(
        self,
    ) -> None:
        evaluation = load_fixture(EVAL)
        predictions = [baseline_predict(record) for record in evaluation]
        first = evaluate(evaluation, predictions)
        second = evaluate(evaluation, predictions)

        self.assertEqual(first, second)
        self.assertEqual(first["records"], 20)
        self.assertEqual(first["json_validity"], 1.0)
        self.assertEqual(first["tool_accuracy"], 1.0)
        self.assertEqual(first["argument_exact_match"], 0.0)
        self.assertEqual(first["argument_field_f1"], 0.0)
        self.assertEqual(first["approval_accuracy"], 1.0)
        self.assertEqual(first["rejection_accuracy"], 1.0)
        self.assertEqual(first["fallback_accuracy"], 1.0)
        self.assertEqual(first["dangerous_false_approvals"], 0)
        self.assertAlmostEqual(first["risk_macro_f1"], 0.8641148325358852)

    def test_cli_emits_report_and_machine_readable_failure(self) -> None:
        command = [
            sys.executable,
            "-m",
            "fullcycle_bridge.tool_router_cli",
            "--seed",
            str(SEED),
            "--eval",
            str(EVAL),
        ]
        success = subprocess.run(command, cwd=ROOT, check=False, capture_output=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(success.stdout)["valid"])

        failure = subprocess.run(
            command[:-1] + [str(ROOT / "missing.json")],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(json.loads(failure.stdout)["code"], "UNSAFE_FIXTURE")


if __name__ == "__main__":
    unittest.main()
