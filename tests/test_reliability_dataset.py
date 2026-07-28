from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from fullcycle_bridge.consumer import BridgeValidationError, canonical_json_bytes
from fullcycle_bridge.dataset import (
    RELIABILITY_DATASET_SCHEMA_VERSION,
    map_files,
    map_many,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures" / "bridge_v1" / "valid" / "runtime-manifest.json"
DATASET_FIXTURES = ROOT / "fixtures" / "reliability_dataset_v1"
DENIAL_RUN = (
    DATASET_FIXTURES
    / "inputs"
    / "failure-denial-recovery-budget-sequence.json"
)
UNKNOWN_RUN = DATASET_FIXTURES / "inputs" / "unknown-outcome.json"
EXPECTED_JSONL = DATASET_FIXTURES / "expected-records.jsonl"
SCHEMA_PATH = ROOT / "schemas" / "reliability_dataset_v1.schema.json"


def expected_records() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in EXPECTED_JSONL.read_text(encoding="utf-8").splitlines()
    ]


class ReliabilityDatasetTests(unittest.TestCase):
    def assert_code(self, expected: str, function, *args) -> BridgeValidationError:
        with self.assertRaises(BridgeValidationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, expected)
        return raised.exception

    def test_fixed_inputs_map_exactly_to_canonical_expected_jsonl(self) -> None:
        records = map_many(MANIFEST, [DENIAL_RUN, UNKNOWN_RUN])
        expected = expected_records()

        self.assertEqual(records, expected)
        actual_jsonl = b"".join(canonical_json_bytes(item) + b"\n" for item in records)
        self.assertEqual(actual_jsonl, EXPECTED_JSONL.read_bytes())

    def test_machine_readable_schema_pins_exact_record_topology(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        record = expected_records()[0]

        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(record))
        self.assertEqual(
            schema["properties"]["reliability_dataset_schema_version"]["const"], 1
        )
        self.assertEqual(
            schema["properties"]["data_class"]["const"],
            "redacted_runtime_evidence",
        )
        self.assertEqual(
            schema["properties"]["training_use"]["const"],
            "reliability_and_verifier_only",
        )
        for section in ("source", "features", "labels"):
            contract = schema["properties"][section]
            self.assertFalse(contract["additionalProperties"])
            self.assertEqual(set(contract["required"]), set(record[section]))

    def test_failure_denial_recovery_budget_and_sequence_signals(self) -> None:
        record = map_files(MANIFEST, DENIAL_RUN)

        self.assertEqual(
            record["reliability_dataset_schema_version"],
            RELIABILITY_DATASET_SCHEMA_VERSION,
        )
        self.assertEqual(record["labels"]["outcome_class"], "failure")
        self.assertTrue(record["labels"]["is_failure"])
        self.assertFalse(record["labels"]["is_unknown_outcome"])
        self.assertTrue(record["labels"]["policy_denied"])
        self.assertTrue(record["labels"]["recovery_required"])
        self.assertTrue(record["labels"]["budget_limit_hit"])
        self.assertEqual(
            record["features"]["tool_sequence"],
            [
                {"sequence": 2, "tool": "click"},
                {"sequence": 5, "tool": "type"},
            ],
        )
        self.assertEqual(
            record["labels"]["verifier_tags"],
            [
                "failure",
                "policy_denial",
                "recovery_required",
                "budget_limit",
                "tool_sequence_available",
            ],
        )

    def test_unknown_outcome_is_not_inferred_from_model_text(self) -> None:
        record = map_files(MANIFEST, UNKNOWN_RUN)

        self.assertEqual(record["labels"]["outcome_class"], "unknown_outcome")
        self.assertTrue(record["labels"]["is_failure"])
        self.assertTrue(record["labels"]["is_unknown_outcome"])
        self.assertFalse(record["labels"]["policy_denied"])
        self.assertTrue(record["labels"]["recovery_required"])
        self.assertFalse(record["labels"]["budget_limit_hit"])
        self.assertEqual(
            record["features"]["tool_outcomes"][0]["dispatch"], "unknown"
        )

    def test_mapping_is_deterministic_and_source_bound(self) -> None:
        first = map_files(MANIFEST, DENIAL_RUN)
        second = map_files(MANIFEST, DENIAL_RUN)

        self.assertEqual(first, second)
        self.assertEqual(
            canonical_json_bytes(first),
            canonical_json_bytes(second),
        )
        self.assertEqual(
            first["record_id"],
            "sha256:814072f56eb61b88e088a44f4aceca41f4b18c055d896d27307ed7857689e32e",
        )
        self.assertEqual(
            first["source"]["run_export_digest"],
            "sha256:41579135f0d7cae3265555d467f3b5c77f0ff2f43a22a6113ed64ac9a2e5ff51",
        )

    def test_only_redacted_reliability_training_use_survives(self) -> None:
        forbidden = {
            "task",
            "raw_task",
            "model_text",
            "tool_result_text",
            "image",
            "images",
            "memory",
            "continuation",
            "content",
            "response",
        }
        for record in map_many(MANIFEST, [DENIAL_RUN, UNKNOWN_RUN]):
            self.assertEqual(record["data_class"], "redacted_runtime_evidence")
            self.assertEqual(
                record["training_use"], "reliability_and_verifier_only"
            )
            serialized = canonical_json_bytes(record)
            self.assertNotIn(b"secret-rich-content", serialized)
            self.assertTrue(forbidden.isdisjoint(_keys(record)))

    def test_mapper_reuses_bridge_gate_and_rejects_invalid_inputs(self) -> None:
        rich = ROOT / "fixtures" / "bridge_v1" / "invalid" / "rich-content.json"
        wrong_use = (
            ROOT / "fixtures" / "bridge_v1" / "invalid" / "wrong-training-use.json"
        )
        self.assert_code("FORBIDDEN_RICH_FIELD", map_files, MANIFEST, rich)
        self.assert_code("INVALID_TRAINING_USE", map_files, MANIFEST, wrong_use)

    def test_duplicate_and_empty_batches_fail_closed(self) -> None:
        self.assert_code(
            "DUPLICATE_DATASET_RUN",
            map_many,
            MANIFEST,
            [DENIAL_RUN, DENIAL_RUN],
        )
        self.assert_code("EMPTY_DATASET_INPUT", map_many, MANIFEST, [])

    def test_dataset_cli_emits_exact_jsonl_and_machine_readable_failure(self) -> None:
        command = [
            sys.executable,
            "-m",
            "fullcycle_bridge.dataset_cli",
            "--manifest",
            str(MANIFEST),
            "--run-export",
            str(DENIAL_RUN),
            "--run-export",
            str(UNKNOWN_RUN),
        ]
        success = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(success.stdout, EXPECTED_JSONL.read_bytes())

        command[-1] = str(
            ROOT / "fixtures" / "bridge_v1" / "invalid" / "digest-mismatch.json"
        )
        failure = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(
            json.loads(failure.stderr)["code"], "MANIFEST_DIGEST_MISMATCH"
        )


def _keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            result.add(key)
            result.update(_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_keys(nested))
    return result


if __name__ == "__main__":
    unittest.main()
