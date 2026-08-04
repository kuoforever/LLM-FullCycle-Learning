from __future__ import annotations

import json
import unittest
from pathlib import Path

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_merge_numerics import (
    analyze_module_comparisons,
    classify_merge_numerics,
)
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "baseline" / "fc-mvp-001-bf16-merge-numerics-v1.json"
STABILITY = ROOT / "baseline" / "fc-mvp-001-bf16-merge-stability-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _comparison(name: str, equal: bool) -> dict[str, object]:
    return {
        "name": name,
        "equal": equal,
        "different_elements": 0 if equal else 3,
        "max_abs_delta": 0.0 if equal else 0.25,
        "mean_abs_delta": 0.0 if equal else 0.1,
    }


class ToolRouterMergeNumericsTests(unittest.TestCase):
    def test_frozen_numerics_gate_matches_locked_sources(self) -> None:
        gate = _load(GATE)
        config = _load(ROOT / "configs" / "tool_router_lora_sft_v2.json")
        adapter = (
            ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
        )
        self.assertEqual(gate["config_sha256"], canonical_config_sha256(config))
        self.assertEqual(gate["adapter_files"], directory_artifact_manifest(adapter))
        self.assertEqual(gate["stability_evidence_sha256"], file_sha256(STABILITY))
        self.assertTrue(all(gate["acceptance"].values()))  # type: ignore[union-attr]
        analysis = gate["module_analysis"]  # type: ignore[assignment]
        self.assertEqual(analysis["first_divergent_module_index"], 2)  # type: ignore[index]
        self.assertEqual(
            analysis["first_divergent_module"],  # type: ignore[index]
            "model.layers.0.self_attn.q_proj",
        )
        rounding = gate["merge_rounding"]  # type: ignore[assignment]
        self.assertEqual(rounding["target_modules"], 112)  # type: ignore[index]
        self.assertEqual(rounding["actual_merged_mismatched_weights"], 0)  # type: ignore[index]
        self.assertGreater(
            rounding["ideal_nonzero_updates_rounded_to_base"],  # type: ignore[index]
            0,
        )
        self.assertEqual(gate["classification"], "bf16_safe_merge_weight_rounding")
        self.assertFalse(gate["merged_artifact_allowed"])
        self.assertFalse(gate["runtime_eligible"])
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],  # type: ignore[index]
            "FC-MVP-001-bf16-merge-remediation-v1",
        )

    def test_first_module_divergence_is_located_in_execution_order(self) -> None:
        analysis = analyze_module_comparisons(
            [
                _comparison("embed_tokens", True),
                _comparison("layer0.input_layernorm", True),
                _comparison("layer0.self_attn.q_proj", False),
                _comparison("layer0.self_attn.k_proj", False),
            ]
        )
        self.assertEqual(analysis["first_divergent_module_index"], 2)
        self.assertEqual(
            analysis["first_divergent_module"],
            "layer0.self_attn.q_proj",
        )
        self.assertTrue(analysis["preceding_modules_identical"])
        self.assertEqual(
            classify_merge_numerics(
                analysis,
                {
                    "actual_merged_mismatched_weights": 0,
                    "ideal_nonzero_updates_rounded_to_base": 10,
                },
            ),
            "bf16_safe_merge_weight_rounding",
        )

    def test_identity_and_unverified_rounding_do_not_overclaim(self) -> None:
        identity = analyze_module_comparisons([_comparison("embed_tokens", True)])
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_merge_numerics(identity, {})
        self.assertEqual(raised.exception.code, "MODULE_DIVERGENCE_NOT_ESTABLISHED")
        divergent = analyze_module_comparisons([_comparison("q_proj", False)])
        self.assertEqual(
            classify_merge_numerics(
                divergent,
                {
                    "actual_merged_mismatched_weights": 0,
                    "ideal_nonzero_updates_rounded_to_base": 0,
                },
            ),
            "module_divergence_without_quantified_rounding",
        )

    def test_malformed_or_duplicate_comparisons_fail_closed(self) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons([])
        self.assertEqual(raised.exception.code, "EMPTY_MODULE_COMPARISONS")
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_module_comparisons(
                [_comparison("q_proj", True), _comparison("q_proj", False)]
            )
        self.assertEqual(raised.exception.code, "INVALID_MODULE_NAME")


if __name__ == "__main__":
    unittest.main()
