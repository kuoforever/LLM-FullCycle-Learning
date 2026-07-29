from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import load_fixture
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_target,
    render_user_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tool_router_lora_sft_v1.json"
TRAINING_EVIDENCE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v1-training.json"
PREDICTIONS = (
    ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v1-predictions.json"
)
REPORT = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v1-report.json"
BASELINE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v1.json"
LOAD_MERGE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v1-load-merge.json"
ADAPTER = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v1"


class ToolRouterSftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = load_fixture(ROOT / "fixtures/tool_router_v1/train.json")[0]

    def test_user_payload_excludes_gold_and_metadata(self) -> None:
        payload = json.loads(render_user_payload(self.record))
        self.assertEqual(
            set(payload),
            {"instruction", "available_tools", "state"},
        )
        self.assertNotIn("decision", payload)
        self.assertNotIn("category", payload)
        self.assertNotIn("example_id", payload)

    def test_target_is_exact_compact_decision_json(self) -> None:
        target = render_target(self.record)
        self.assertEqual(json.loads(target), self.record["decision"])
        self.assertNotIn(" ", target)
        self.assertNotIn("\n", target)

    def test_config_digest_is_whitespace_independent(self) -> None:
        first = {"b": [2, 3], "a": 1}
        second = json.loads('{\n  "a": 1,\n  "b": [2, 3]\n}')
        self.assertEqual(
            canonical_config_sha256(first),
            canonical_config_sha256(second),
        )

    def test_adapter_manifest_is_sorted_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            manifest = directory_artifact_manifest(root)
        self.assertEqual([item["path"] for item in manifest], ["a.txt", "z.txt"])
        self.assertEqual([item["bytes"] for item in manifest], [1, 1])
        self.assertTrue(all(item["sha256"].startswith("sha256:") for item in manifest))

    def test_frozen_lora_report_and_adapter_reproduce_offline(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        evidence = json.loads(TRAINING_EVIDENCE.read_text(encoding="utf-8"))
        predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        load_merge = json.loads(LOAD_MERGE.read_text(encoding="utf-8"))
        eval_records = load_fixture(ROOT / config["data"]["eval_path"])
        raw_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in predictions["outputs"]
        ]

        metrics, parsed = score_raw_outputs(eval_records, raw_outputs)
        adapter_manifest = directory_artifact_manifest(ADAPTER)

        self.assertEqual(canonical_config_sha256(config), baseline["canonical_config_sha256"])
        self.assertEqual(evidence["config_sha256"], baseline["canonical_config_sha256"])
        self.assertEqual(predictions["config_sha256"], baseline["canonical_config_sha256"])
        self.assertEqual(metrics, report["metrics"])
        self.assertEqual(parsed, report["parsed_outputs"])
        self.assertEqual(metrics, baseline["metrics"])
        self.assertEqual(adapter_manifest, evidence["final_adapter"]["files"])
        self.assertEqual(
            file_sha256(ADAPTER / "adapter_model.safetensors"),
            baseline["adapter"]["adapter_weight_sha256"],
        )
        for relative_path, expected_digest in baseline["artifact_hashes"].items():
            self.assertEqual(file_sha256(ROOT / relative_path), expected_digest)
        self.assertFalse(report["safety_gate_passed"])
        self.assertFalse(report["runtime_eligible"])
        self.assertEqual(metrics["dangerous_action_candidates"], 1)
        self.assertEqual(metrics["dangerous_false_approvals"], 0)
        self.assertTrue(load_merge["safe_merge"])
        self.assertTrue(load_merge["outputs_identical"])
        self.assertEqual(load_merge["remaining_adapter_parameter_tensors"], 0)


if __name__ == "__main__":
    unittest.main()
