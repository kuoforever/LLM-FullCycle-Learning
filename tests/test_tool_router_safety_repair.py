from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import (
    ToolRouterValidationError,
    fixture_digest,
    load_fixture,
)
from fullcycle_bridge.tool_router_dataset import load_family_manifest
from fullcycle_bridge.tool_router_safety_repair import (
    audit_safety_repair_dataset,
    load_badcase_taxonomy,
)
from scripts.build_tool_router_safety_v2 import build

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures" / "tool_router_v1"
V2 = ROOT / "fixtures" / "tool_router_v2"
TAXONOMY = ROOT / "baseline" / "fc-mvp-001-safety-repair-badcases-v1.json"
BASELINE = ROOT / "baseline" / "fc-mvp-001-safety-repair-data-v2.json"


class ToolRouterSafetyRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_train = load_fixture(V1 / "train.json")
        self.base_validation = load_fixture(V1 / "validation.json")
        self.train = load_fixture(V2 / "train.json")
        self.validation = load_fixture(V2 / "validation.json")
        self.evaluation = load_fixture(V1 / "eval.json")
        self.base_manifest = load_family_manifest(V1 / "family-manifest.json")
        self.manifest = load_family_manifest(V2 / "family-manifest.json")
        self.taxonomy = load_badcase_taxonomy(TAXONOMY)
        self.eval_digest = fixture_digest(self.evaluation)

    def audit(self, **changes: object) -> dict[str, object]:
        values: dict[str, Any] = {
            "base_train": self.base_train,
            "base_validation": self.base_validation,
            "repaired_train": self.train,
            "repaired_validation": self.validation,
            "evaluation": self.evaluation,
            "base_manifest": self.base_manifest,
            "repaired_manifest": self.manifest,
            "taxonomy": self.taxonomy,
            "eval_digest": self.eval_digest,
        }
        values.update(changes)
        return audit_safety_repair_dataset(**values)

    def assert_code(self, expected: str, **changes: object) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            self.audit(**changes)
        self.assertEqual(raised.exception.code, expected)

    def test_report_matches_frozen_baseline_exactly(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

        self.assertEqual(self.audit(), baseline["expected_report"])
        self.assertEqual(self.eval_digest, baseline["frozen_eval_digest"])

    def test_reviewed_builder_reproduces_saved_increment_exactly(self) -> None:
        train, validation, manifest = build()

        self.assertEqual(train, self.train)
        self.assertEqual(validation, self.validation)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(train[: len(self.base_train)], self.base_train)
        self.assertEqual(
            validation[: len(self.base_validation)], self.base_validation
        )

    def test_source_and_artifact_hashes_are_frozen(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        expected = {
            baseline["source_badcase_taxonomy"]["path"]: baseline[
                "source_badcase_taxonomy"
            ]["sha256"],
            **baseline["base_artifact_hashes"],
            **baseline["artifact_hashes"],
        }
        for relative, digest in expected.items():
            actual = "sha256:" + hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_base_prefix_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.train)
        changed[0]["instruction"] += " drift"

        self.assert_code("BASE_TRAIN_DRIFT", repaired_train=changed)

    def test_eval_answer_copy_fails_closed(self) -> None:
        changed = copy.deepcopy(self.train)
        changed[len(self.base_train)]["instruction"] = self.evaluation[0][
            "instruction"
        ]

        self.assert_code("EVAL_ANSWER_COPY", repaired_train=changed)

    def test_dangerous_action_candidate_fails_closed(self) -> None:
        changed = copy.deepcopy(self.train)
        record = next(
            item
            for item in changed[len(self.base_train) :]
            if item["category"] == "dangerous_request"
        )
        record["decision"]["selected_tool"] = "computer_use"

        self.assert_code("DANGEROUS_ACTION_CANDIDATE", repaired_train=changed)

    def test_repair_family_target_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["families"][len(self.base_manifest["families"])][
            "description"
        ] = "unknown_target | drift"

        self.assert_code("INVALID_REPAIR_FAMILY_TARGETS", repaired_manifest=changed)

    def test_unsafe_taxonomy_policy_fails_closed(self) -> None:
        changed = copy.deepcopy(self.taxonomy)
        changed["repair_policy"]["eval_answers_included"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ToolRouterValidationError) as raised:
                load_badcase_taxonomy(path)

        self.assertEqual(raised.exception.code, "UNSAFE_REPAIR_POLICY")


if __name__ == "__main__":
    unittest.main()
