from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from fullcycle_bridge.consumer import (
    CONSUMER_SCHEMA_VERSION,
    MAX_CHECKPOINT_BYTES,
    MAX_EVENT_BYTES,
    MAX_INPUT_BYTES,
    RUNTIME_GIT_COMMIT,
    BridgeValidationError,
    canonical_json_bytes,
    manifest_digest,
    validate_files,
    validate_manifest,
    validate_run_export,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "bridge_v1"
MANIFEST_PATH = FIXTURES / "valid" / "runtime-manifest.json"
RUN_PATH = FIXTURES / "valid" / "minimal-run-export.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class BridgeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load(MANIFEST_PATH)
        cls.run_export = load(RUN_PATH)

    def assert_code(self, expected: str, function, *args) -> BridgeValidationError:
        with self.assertRaises(BridgeValidationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, expected)
        return raised.exception

    def test_valid_pinned_fixture(self) -> None:
        summary = validate_files(MANIFEST_PATH, RUN_PATH)
        metadata = load(FIXTURES / "fixture-metadata.json")

        self.assertEqual(summary.consumer_schema_version, CONSUMER_SCHEMA_VERSION)
        self.assertEqual(summary.runtime_git_commit, RUNTIME_GIT_COMMIT)
        self.assertEqual(summary.manifest_digest, metadata["manifest_digest"])
        self.assertEqual(summary.run_id, "run_fixture_minimal")
        self.assertEqual(summary.event_count, 1)
        self.assertEqual(summary.data_class, "redacted_runtime_evidence")
        self.assertEqual(summary.training_use, "reliability_and_verifier_only")
        self.assertEqual(RUNTIME_GIT_COMMIT, metadata["runtime_git_commit"])

    def test_manifest_digest_uses_producer_canonical_json(self) -> None:
        self.assertEqual(
            manifest_digest(self.manifest),
            "sha256:" + __import__("hashlib").sha256(
                canonical_json_bytes(self.manifest)
            ).hexdigest(),
        )
        self.assertEqual(
            manifest_digest(self.manifest),
            self.run_export["manifest_digest"],
        )

    def test_every_manifest_version_is_exact(self) -> None:
        fields = (
            "fullcycle_manifest_version",
            "agent_contract_version",
            "driver_contract_version",
            "trace_version",
            "checkpoint_version",
            "plan_contract_version",
        )
        for field in fields:
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.manifest)
                invalid[field] = 999 if isinstance(invalid[field], int) else "unknown"
                self.assert_code("UNSUPPORTED_VERSION", validate_manifest, invalid)

    def test_automatic_export_requires_exact_six_false_declarations(self) -> None:
        for field in self.manifest["automatic_export"]:
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.manifest)
                invalid["automatic_export"][field] = True
                self.assert_code("UNSAFE_AUTOMATIC_EXPORT", validate_manifest, invalid)
        invalid = copy.deepcopy(self.manifest)
        invalid["automatic_export"]["contains_provider_data"] = False
        self.assert_code("UNSAFE_AUTOMATIC_EXPORT", validate_manifest, invalid)

    def test_saved_invalid_fixtures_fail_with_stable_codes(self) -> None:
        expected = {
            "unknown-version.json": "UNSUPPORTED_VERSION",
            "digest-mismatch.json": "MANIFEST_DIGEST_MISMATCH",
            "wrong-data-class.json": "INVALID_DATA_CLASS",
            "wrong-training-use.json": "INVALID_TRAINING_USE",
            "unexpected-field.json": "UNKNOWN_FIELD",
            "malformed.json": "MALFORMED_JSON",
            "incomplete-event.json": "MISSING_FIELD",
            "rich-content.json": "FORBIDDEN_RICH_FIELD",
        }
        for filename, code in expected.items():
            with self.subTest(filename=filename):
                self.assert_code(
                    code,
                    validate_files,
                    MANIFEST_PATH,
                    FIXTURES / "invalid" / filename,
                )

    def test_run_top_level_structure_is_exact(self) -> None:
        invalid = copy.deepcopy(self.run_export)
        invalid["provider"] = "forbidden"
        self.assert_code(
            "UNKNOWN_FIELD", validate_run_export, invalid, self.manifest
        )
        invalid = copy.deepcopy(self.run_export)
        del invalid["training_use"]
        self.assert_code(
            "MISSING_FIELD", validate_run_export, invalid, self.manifest
        )

    def test_checkpoint_and_event_completeness_are_cross_checked(self) -> None:
        invalid = copy.deepcopy(self.run_export)
        invalid["checkpoint"]["event_count"] = 2
        self.assert_code(
            "INCOMPLETE_EVENTS", validate_run_export, invalid, self.manifest
        )
        invalid = copy.deepcopy(self.run_export)
        invalid["events"][0]["sequence"] = 2
        self.assert_code(
            "INCOMPLETE_EVENTS", validate_run_export, invalid, self.manifest
        )
        invalid = copy.deepcopy(self.run_export)
        invalid["events"][0]["run_id"] = "other"
        self.assert_code(
            "RUN_ID_MISMATCH", validate_run_export, invalid, self.manifest
        )

    def test_every_event_kind_requires_complete_shape(self) -> None:
        examples = {
            "user_task": {"task_length": 1},
            "model_turn": {
                "text_length": 0,
                "tool_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
            },
            "tool_call": {"tool": "click", "arguments": {}, "redacted_fields": []},
            "tool_result": {
                "tool": "click",
                "status": "success",
                "dispatch": "dispatched",
                "text_length": 0,
                "image_count": 0,
            },
            "observation": {"tool": "ui_snapshot", "observation_epoch": 1},
            "policy_decision": {"decision": "allow"},
            "recovery": {"status": "ready"},
        }
        for kind, fields in examples.items():
            for missing in fields:
                with self.subTest(kind=kind, missing=missing):
                    invalid = copy.deepcopy(self.run_export)
                    event = {
                        "trace_version": 1,
                        "sequence": 1,
                        "run_id": "run_fixture_minimal",
                        "kind": kind,
                        **fields,
                    }
                    del event[missing]
                    invalid["events"] = [event]
                    self.assert_code(
                        "MISSING_FIELD",
                        validate_run_export,
                        invalid,
                        self.manifest,
                    )

    def test_rich_fields_are_rejected_recursively(self) -> None:
        for field in ("task", "model_text", "tool_result_text", "image", "memory", "continuation"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.run_export)
                invalid["events"][0][field] = "secret"
                self.assert_code(
                    "UNKNOWN_FIELD", validate_run_export, invalid, self.manifest
                )
        invalid = copy.deepcopy(self.run_export)
        invalid["checkpoint"]["event_count"] = 2
        invalid["events"] = [
            {
                "trace_version": 1,
                "sequence": 1,
                "run_id": "run_fixture_minimal",
                "kind": "user_task",
                "task_length": 1,
            },
            {
                "trace_version": 1,
                "sequence": 2,
                "run_id": "run_fixture_minimal",
                "kind": "tool_call",
                "tool": "type",
                "arguments": {"nested": {"content": "secret"}},
                "redacted_fields": [],
            }
        ]
        self.assert_code(
            "FORBIDDEN_RICH_FIELD", validate_run_export, invalid, self.manifest
        )

    def test_file_checkpoint_and_event_size_bounds(self) -> None:
        self.assertEqual(MAX_INPUT_BYTES, 24 * 1024 * 1024)
        self.assertEqual(MAX_CHECKPOINT_BYTES, 64 * 1024)
        self.assertEqual(MAX_EVENT_BYTES, 1024 * 1024)

        work = ROOT / "work" / "test-fixtures"
        work.mkdir(parents=True, exist_ok=True)
        oversized = work / "oversized.json"
        try:
            with oversized.open("wb") as stream:
                stream.seek(MAX_INPUT_BYTES)
                stream.write(b"}")
            self.assert_code("INPUT_TOO_LARGE", validate_files, oversized, RUN_PATH)
        finally:
            oversized.unlink(missing_ok=True)

        invalid = copy.deepcopy(self.run_export)
        invalid["checkpoint"]["policy_version"] = "x" * MAX_CHECKPOINT_BYTES
        self.assert_code(
            "CHECKPOINT_TOO_LARGE", validate_run_export, invalid, self.manifest
        )

        invalid = copy.deepcopy(self.run_export)
        invalid["events"] = [
            {
                "trace_version": 1,
                "sequence": 1,
                "run_id": "run_fixture_minimal",
                "kind": "tool_call",
                "tool": "safe",
                "arguments": {"metadata": "x" * MAX_EVENT_BYTES},
                "redacted_fields": [],
            }
        ]
        self.assert_code(
            "EVENT_TOO_LARGE", validate_run_export, invalid, self.manifest
        )

    def test_duplicate_keys_and_nonfinite_numbers_are_malformed(self) -> None:
        work = ROOT / "work" / "test-fixtures"
        work.mkdir(parents=True, exist_ok=True)
        duplicate = work / "duplicate.json"
        nonfinite = work / "nonfinite.json"
        try:
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            nonfinite.write_text('{"value":NaN}', encoding="utf-8")
            self.assert_code(
                "DUPLICATE_JSON_KEY", validate_files, duplicate, RUN_PATH
            )
            self.assert_code("NONFINITE_NUMBER", validate_files, nonfinite, RUN_PATH)
        finally:
            duplicate.unlink(missing_ok=True)
            nonfinite.unlink(missing_ok=True)

    def test_cli_returns_machine_readable_success_and_failure(self) -> None:
        command = [
            sys.executable,
            "-m",
            "fullcycle_bridge",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-export",
            str(RUN_PATH),
        ]
        success = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(success.stdout)["valid"])

        command[-1] = str(FIXTURES / "invalid" / "digest-mismatch.json")
        failure = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(
            json.loads(failure.stderr)["code"], "MANIFEST_DIGEST_MISMATCH"
        )


if __name__ == "__main__":
    unittest.main()
