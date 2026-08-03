from __future__ import annotations

import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import fixture_digest, load_fixture
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tool_router_lora_sft_v2.json"
TRAINING = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
PREDICTIONS = (
    ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
)
REPORT = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
BASELINE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2.json"
LOAD_MERGE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-load-merge.json"
ADAPTER = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"


class ToolRouterSftV2Tests(unittest.TestCase):
    def test_locked_config_uses_only_passed_v2_data_and_frozen_eval(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        safety_baseline = json.loads(
            (ROOT / "baseline" / "fc-mvp-001-safety-repair-data-v2.json").read_text(
                encoding="utf-8"
            )
        )
        for split in ("train", "validation", "eval"):
            records = load_fixture(ROOT / config["data"][f"{split}_path"])
            self.assertEqual(len(records), config["data"][f"{split}_records"])
            self.assertEqual(
                fixture_digest(records), config["data"][f"{split}_digest"]
            )

        self.assertEqual(config["training"]["epochs"], 3)
        self.assertEqual(config["data"]["train_records"], 176)
        self.assertEqual(config["data"]["validation_records"], 48)
        self.assertEqual(
            config["data"]["eval_digest"], safety_baseline["frozen_eval_digest"]
        )

    def test_frozen_v2_training_and_eval_evidence_reproduce_offline(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        training = json.loads(TRAINING.read_text(encoding="utf-8"))
        predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        load_merge = json.loads(LOAD_MERGE.read_text(encoding="utf-8"))
        evaluation = load_fixture(ROOT / config["data"]["eval_path"])
        raw_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in predictions["outputs"]
        ]

        metrics, parsed = score_raw_outputs(evaluation, raw_outputs)
        config_digest = canonical_config_sha256(config)
        self.assertEqual(config_digest, baseline["canonical_config_sha256"])
        self.assertEqual(training["config_sha256"], config_digest)
        self.assertEqual(predictions["config_sha256"], config_digest)
        self.assertEqual(metrics, report["metrics"])
        self.assertEqual(parsed, report["parsed_outputs"])
        self.assertEqual(metrics, baseline["metrics"])
        self.assertEqual(
            directory_artifact_manifest(ADAPTER), training["final_adapter"]["files"]
        )
        for relative, digest in baseline["artifact_hashes"].items():
            self.assertEqual(file_sha256(ROOT / relative), digest, relative)

        self.assertTrue(report["safety_gate_passed"])
        self.assertEqual(metrics["dangerous_action_candidates"], 0)
        self.assertEqual(metrics["dangerous_false_approvals"], 0)
        self.assertFalse(report["runtime_eligible"])
        self.assertFalse(baseline["runtime_eligible"])

        self.assertTrue(load_merge["safe_merge"])
        self.assertFalse(load_merge["outputs_identical"])
        self.assertEqual(load_merge["remaining_adapter_parameter_tensors"], 0)
        self.assertEqual(
            baseline["runtime_eligibility_reason"],
            "decision_inconsistency_and_load_merge_output_drift",
        )


if __name__ == "__main__":
    unittest.main()
