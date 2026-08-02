from __future__ import annotations

import json
import unittest
from pathlib import Path

from fullcycle_bridge.consumer import CONSUMER_SCHEMA_VERSION, manifest_digest
from fullcycle_bridge.dataset import RELIABILITY_DATASET_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "baseline" / "runtime-freeze-v1.json"
FIXTURE_ROOT = ROOT / "fixtures" / "bridge_v1"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class RuntimeFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freeze = load(FREEZE_PATH)
        cls.metadata = load(FIXTURE_ROOT / "fixture-metadata.json")
        cls.manifest = load(FIXTURE_ROOT / "valid" / "runtime-manifest.json")
        cls.run_export = load(FIXTURE_ROOT / "valid" / "minimal-run-export.json")

    def test_freeze_pin_is_exact_and_separate_from_fixture_provenance(self) -> None:
        self.assertEqual(
            self.freeze["runtime_git_commit"],
            "324ff2fb5911e332ddb5c5f90eb41296e8faf7a9",
        )
        self.assertEqual(self.freeze["runtime_branch"], "main")
        self.assertEqual(
            self.freeze["lane_a_fixture_producer_commit"],
            self.metadata["runtime_git_commit"],
        )
        self.assertNotEqual(
            self.freeze["runtime_git_commit"],
            self.freeze["lane_a_fixture_producer_commit"],
        )

    def test_freeze_contract_matches_the_immutable_lane_a_fixture(self) -> None:
        for field in (
            "agent_contract_version",
            "driver_contract_version",
            "fullcycle_manifest_version",
            "trace_version",
            "checkpoint_version",
            "plan_contract_version",
        ):
            with self.subTest(field=field):
                self.assertEqual(self.freeze[field], self.manifest[field])
        self.assertEqual(
            self.freeze["fullcycle_run_export_version"],
            self.run_export["fullcycle_run_export_version"],
        )
        self.assertEqual(
            self.freeze["consumer_schema_version"], CONSUMER_SCHEMA_VERSION
        )
        self.assertEqual(
            self.freeze["reliability_dataset_schema_version"],
            RELIABILITY_DATASET_SCHEMA_VERSION,
        )
        self.assertEqual(self.freeze["manifest_digest"], manifest_digest(self.manifest))
        self.assertEqual(self.freeze["manifest_digest"], self.metadata["manifest_digest"])

    def test_freeze_records_the_fail_closed_lane_b_disposition(self) -> None:
        self.assertEqual(
            self.freeze["lane_b_disposition"],
            "deferred_pending_explicit_consent_security_privacy_review",
        )
        preflight = self.freeze["runtime_release_preflight"]
        self.assertIsInstance(preflight, dict)
        assert isinstance(preflight, dict)
        self.assertEqual(preflight["report_version"], 5)
        self.assertTrue(preflight["passed"])
        self.assertEqual(preflight["passed_tests"], 1566)
        self.assertEqual(preflight["skipped_tests"], 8)


if __name__ == "__main__":
    unittest.main()
