from __future__ import annotations

import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import (
    ToolRouterValidationError,
    fixture_digest,
    load_fixture,
)
from fullcycle_bridge.tool_router_merge_stability import (
    analyze_token_runs,
    classify_logit_divergence,
)
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "baseline" / "fc-mvp-001-bf16-merge-stability-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


class ToolRouterMergeStabilityTests(unittest.TestCase):
    def test_frozen_probe_matches_all_locked_inputs_and_acceptance(self) -> None:
        gate = _load(GATE)
        config = _load(ROOT / "configs" / "tool_router_lora_sft_v2.json")
        training = _load(
            ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
        )
        adapter = (
            ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
        )
        evaluation = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "eval.json")

        self.assertEqual(gate["config_sha256"], canonical_config_sha256(config))
        self.assertEqual(gate["adapter_files"], directory_artifact_manifest(adapter))
        self.assertEqual(gate["adapter_files"], training["final_adapter"]["files"])  # type: ignore[index]
        self.assertEqual(gate["eval_digest"], fixture_digest(evaluation))
        self.assertEqual(
            gate["prompt_sha256"],
            file_sha256(ROOT / "prompts" / "tool_router_v1.txt"),
        )
        self.assertTrue(all(gate["acceptance"].values()))  # type: ignore[union-attr]
        runs = gate["runs"]  # type: ignore[assignment]
        self.assertEqual(runs[0]["token_ids_sha256"], runs[1]["token_ids_sha256"])  # type: ignore[index]
        self.assertEqual(runs[2]["token_ids_sha256"], runs[3]["token_ids_sha256"])  # type: ignore[index]
        self.assertNotEqual(runs[0]["token_ids_sha256"], runs[2]["token_ids_sha256"])  # type: ignore[index]
        token_analysis = gate["token_analysis"]  # type: ignore[assignment]
        logits = gate["logit_evidence"]  # type: ignore[assignment]
        self.assertEqual(token_analysis["first_divergent_token_index"], 45)  # type: ignore[index]
        self.assertEqual(logits["independent_top_token_ids"][0], 1866)  # type: ignore[index]
        self.assertEqual(logits["merged_top_token_ids"][0], 3849)  # type: ignore[index]
        self.assertEqual(
            gate["classification"],
            "deterministic_bf16_merge_logit_boundary_flip",
        )
        self.assertFalse(gate["merged_artifact_allowed"])
        self.assertFalse(gate["runtime_eligible"])
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],  # type: ignore[index]
            "FC-MVP-001-bf16-merge-numerics-v1",
        )

    def test_repeat_stable_cross_path_drift_is_located_exactly(self) -> None:
        analysis = analyze_token_runs(
            [[10, 20, 30], [10, 20, 30]],
            [[10, 21, 30], [10, 21, 30]],
        )
        self.assertTrue(analysis["independent_repeats_identical"])
        self.assertTrue(analysis["merged_repeats_identical"])
        self.assertFalse(analysis["cross_path_identical"])
        self.assertEqual(analysis["first_divergent_token_index"], 1)
        self.assertEqual(
            classify_logit_divergence(analysis, 20, 21),
            "deterministic_bf16_merge_logit_boundary_flip",
        )

    def test_identical_paths_are_classified_without_divergence(self) -> None:
        analysis = analyze_token_runs([[1, 2], [1, 2]], [[1, 2], [1, 2]])
        self.assertTrue(analysis["cross_path_identical"])
        self.assertEqual(analysis["classification"], "output_identity_restored")

    def test_within_path_nondeterminism_is_not_mislabeled_as_merge_drift(self) -> None:
        analysis = analyze_token_runs([[1, 2], [1, 3]], [[1, 4], [1, 4]])
        self.assertEqual(analysis["classification"], "within_path_nondeterminism")
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_logit_divergence(analysis, 2, 4)
        self.assertEqual(raised.exception.code, "TOKEN_DRIFT_NOT_ESTABLISHED")

    def test_logit_argmax_mismatch_remains_sequence_drift(self) -> None:
        analysis = analyze_token_runs(
            [[10, 20], [10, 20]],
            [[10, 21], [10, 21]],
        )
        self.assertEqual(
            classify_logit_divergence(analysis, 21, 21),
            "deterministic_bf16_merge_sequence_drift",
        )

    def test_short_or_invalid_runs_fail_closed(self) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_token_runs([[1, 2]], [[1, 2], [1, 2]])
        self.assertEqual(raised.exception.code, "INSUFFICIENT_REPEATS")
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_token_runs([[1, -1], [1, -1]], [[1, 2], [1, 2]])
        self.assertEqual(raised.exception.code, "INVALID_TOKEN_RUN")


if __name__ == "__main__":
    unittest.main()
