from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import (
    ToolRouterValidationError,
    fixture_digest,
    load_fixture,
)
from fullcycle_bridge.tool_router_dataset import (
    audit_dataset,
    load_family_manifest,
)
from scripts.build_tool_router_dataset import build

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "tool_router_v1"
TRAIN = FIXTURES / "train.json"
VALIDATION = FIXTURES / "validation.json"
EVAL = FIXTURES / "eval.json"
MANIFEST = FIXTURES / "family-manifest.json"
BASELINE = ROOT / "baseline" / "fc-mvp-001-data-v1.json"


class ToolRouterDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = load_fixture(TRAIN)
        self.validation = load_fixture(VALIDATION)
        self.evaluation = load_fixture(EVAL)
        self.manifest = load_family_manifest(MANIFEST)
        self.eval_digest = fixture_digest(self.evaluation)

    def assert_code(self, expected: str, function, *args) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, expected)

    def test_expanded_records_and_families_are_balanced_and_disjoint(self) -> None:
        report = audit_dataset(
            self.train,
            self.validation,
            self.evaluation,
            self.manifest,
            self.eval_digest,
        )

        self.assertEqual(len(self.train), 160)
        self.assertEqual(len(self.validation), 40)
        self.assertEqual(report["train_validation_records"], 200)
        self.assertEqual(report["task_families"], 60)
        self.assertEqual(report["task_family_overlap"], 0)
        self.assertEqual(report["exact_instruction_duplicates"], 0)
        self.assertEqual(report["dangerous_false_approvals"], 0)
        self.assertEqual(
            report["split_category_counts"]["train"],
            {category: 16 for category in report["split_category_counts"]["train"]},
        )
        self.assertEqual(
            report["split_category_counts"]["validation"],
            {category: 4 for category in report["split_category_counts"]["validation"]},
        )

    def test_report_matches_the_frozen_baseline_exactly(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        report = audit_dataset(
            self.train,
            self.validation,
            self.evaluation,
            self.manifest,
            self.eval_digest,
        )

        self.assertEqual(report, baseline["expected_report"])
        self.assertEqual(
            report["report_digest"],
            "sha256:b58af24bdc3cfd34eb4309f91e977f2f4fc6f76a53a229eaa8d3f757d1ebf9a4",
        )

    def test_reviewed_builder_reproduces_all_saved_records_exactly(self) -> None:
        train, validation, manifest = build()

        self.assertEqual(train, self.train)
        self.assertEqual(validation, self.validation)
        self.assertEqual(manifest, self.manifest)

    def test_eval_answers_remain_content_frozen(self) -> None:
        self.assertEqual(
            self.eval_digest,
            "sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a",
        )
        self.assertEqual(
            self.manifest["frozen_eval_digest"],
            self.eval_digest,
        )

    def test_manifest_is_strict_and_covers_every_train_validation_record(self) -> None:
        mapped = {
            example_id
            for family in self.manifest["families"]
            for example_id in family["example_ids"]
        }
        expected = {record["example_id"] for record in self.train + self.validation}

        self.assertEqual(mapped, expected)
        self.assertEqual(len(mapped), 200)
        with tempfile.TemporaryDirectory() as directory:
            changed = copy.deepcopy(self.manifest)
            changed["network"] = True
            path = Path(directory) / "extra-field.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            self.assert_code("INVALID_MANIFEST_KEYS", load_family_manifest, path)

    def test_family_overlap_and_missing_mapping_fail_closed(self) -> None:
        overlap = copy.deepcopy(self.manifest)
        train_family = next(
            family for family in overlap["families"] if family["split"] == "train"
        )
        validation_family = next(
            family for family in overlap["families"] if family["split"] == "validation"
        )
        validation_family["family_id"] = train_family["family_id"]
        self.assert_code(
            "TASK_FAMILY_LEAKAGE",
            audit_dataset,
            self.train,
            self.validation,
            self.evaluation,
            overlap,
            self.eval_digest,
        )

        missing = copy.deepcopy(self.manifest)
        missing["families"][0]["example_ids"].pop()
        self.assert_code(
            "UNMAPPED_DATASET_RECORD",
            audit_dataset,
            self.train,
            self.validation,
            self.evaluation,
            missing,
            self.eval_digest,
        )

    def test_exact_and_near_duplicate_instruction_leakage_fail_closed(self) -> None:
        exact = copy.deepcopy(self.validation)
        exact[0]["instruction"] = self.train[0]["instruction"]
        self.assert_code(
            "EXACT_INSTRUCTION_DUPLICATE",
            audit_dataset,
            self.train,
            exact,
            self.evaluation,
            self.manifest,
            self.eval_digest,
        )

        near = copy.deepcopy(self.validation)
        near[0]["instruction"] = self.train[0]["instruction"] + " Confirmed."
        self.assert_code(
            "NEAR_DUPLICATE_LEAKAGE",
            audit_dataset,
            self.train,
            near,
            self.evaluation,
            self.manifest,
            self.eval_digest,
        )

    def test_eval_digest_and_dangerous_approval_drift_fail_closed(self) -> None:
        self.assert_code(
            "FROZEN_EVAL_DIGEST_MISMATCH",
            audit_dataset,
            self.train,
            self.validation,
            self.evaluation,
            self.manifest,
            "sha256:" + "0" * 64,
        )

        dangerous = copy.deepcopy(self.train)
        record = next(
            item for item in dangerous if item["category"] == "dangerous_request"
        )
        record["decision"]["requires_approval"] = True
        self.assert_code(
            "DANGEROUS_FALSE_APPROVAL",
            audit_dataset,
            dangerous,
            self.validation,
            self.evaluation,
            self.manifest,
            self.eval_digest,
        )

    def test_dataset_cli_emits_the_frozen_report(self) -> None:
        command = [
            sys.executable,
            "-m",
            "fullcycle_bridge.tool_router_dataset_cli",
            "--train",
            str(TRAIN),
            "--validation",
            str(VALIDATION),
            "--eval",
            str(EVAL),
            "--family-manifest",
            str(MANIFEST),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertTrue(output["valid"])
        self.assertEqual(
            output["report"]["report_digest"],
            "sha256:b58af24bdc3cfd34eb4309f91e977f2f4fc6f76a53a229eaa8d3f757d1ebf9a4",
        )


if __name__ == "__main__":
    unittest.main()
