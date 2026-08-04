from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_fp32_attached_merge_isolation import (
    analyze_attached_repeat_stability,
    analyze_same_dtype_tokens,
    classify_same_dtype_effect,
    select_comparison_step,
)
from fullcycle_bridge.tool_router_merge_remediation import token_ids_sha256
from fullcycle_bridge.tool_router_sft import file_sha256

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "baseline" / "fc-mvp-001-fp32-attached-merge-isolation-v1.json"
DRIFT = ROOT / "baseline" / "fc-mvp-001-fp32-merge-drift-analysis-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected object: {path}")
    return value


def _stable(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "first_output_sha256": SHA_A,
        "second_output_sha256": SHA_A,
        "first_score_trace_sha256": SHA_A,
        "second_score_trace_sha256": SHA_A,
        "first_raw_logit_trace_sha256": SHA_B,
        "second_raw_logit_trace_sha256": SHA_B,
        "precision_audits_identical": True,
    }
    values.update(overrides)
    return values


def _classify(token_analysis: dict[str, object], **overrides: object) -> str:
    values: dict[str, object] = {
        "attached_repeat_stable": True,
        "merged_candidate_reproduced": True,
        "attached_emitted_token_id": 11,
        "merged_emitted_token_id": 12,
        "attached_score_top_token_id": 11,
        "merged_score_top_token_id": 12,
        "attached_raw_logit_top_token_id": 11,
        "merged_raw_logit_top_token_id": 12,
        "full_score_traces_identical": False,
        "full_raw_logit_traces_identical": False,
        "comparison_score_vectors_identical": False,
        "comparison_raw_logit_vectors_identical": False,
    }
    values.update(overrides)
    return classify_same_dtype_effect(token_analysis, **values)  # type: ignore[arg-type]


class ToolRouterFp32AttachedMergeIsolationTests(unittest.TestCase):
    def test_frozen_isolation_artifact_is_closed_and_recomputable(self) -> None:
        gate = _load(GATE)
        self.assertEqual(
            file_sha256(GATE),
            "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26",
        )
        self.assertEqual(
            gate["drift_evidence_sha256"],
            file_sha256(DRIFT),
        )
        self.assertEqual(
            set(gate),
            {
                "fp32_attached_merge_isolation_version",
                "experiment_id",
                "source_experiment_id",
                "drift_evidence_sha256",
                "remediation_evidence_sha256",
                "stability_evidence_sha256",
                "numerics_evidence_sha256",
                "training_lock_sha256",
                "config_sha256",
                "adapter_files",
                "model_weight_sha256",
                "prompt_sha256",
                "eval_digest",
                "example_id",
                "input_token_count",
                "input_token_ids_sha256",
                "storage_audit",
                "protocol",
                "frozen_fp32_merged_reference",
                "frozen_bf16_context",
                "runs",
                "attached_repeat_stability",
                "merged_candidate_reproduction",
                "same_dtype_token_analysis",
                "comparison_step",
                "selection_score_evidence",
                "raw_logit_evidence",
                "same_dtype_trace_identity",
                "classification",
                "causal_scope",
                "isolation_gate",
                "remediation_gate",
                "acceptance",
                "elapsed_seconds",
                "peak_gpu_memory_bytes",
                "merged_artifact_saved",
                "merged_artifact_allowed",
                "constraints",
                "locked_next_action",
                "runtime_eligible",
                "runtime_eligibility_reason",
                "offline",
            },
        )
        runs = gate["runs"]
        self.assertEqual(
            [run["run_id"] for run in runs],
            ["fp32_attached-r1", "fp32_attached-r2", "fp32_safe_merged-r1"],
        )
        for run in runs:
            self.assertTrue(run["fresh_load"])
            self.assertEqual(run["token_count"], len(run["generated_token_ids"]))
            self.assertEqual(
                run["token_ids_sha256"],
                token_ids_sha256(run["generated_token_ids"]),
            )
            self.assertLessEqual(
                run["memory_allocated_before_load_bytes"],
                gate["protocol"]["max_residual_cuda_bytes"],
            )
            self.assertLessEqual(
                run["memory_allocated_after_release_bytes"],
                gate["protocol"]["max_residual_cuda_bytes"],
            )
        attached = runs[:2]
        merged = runs[2]
        repeat = analyze_attached_repeat_stability(
            attached[0]["generated_token_ids"],
            attached[1]["generated_token_ids"],
            first_output_sha256=attached[0]["output_sha256"],
            second_output_sha256=attached[1]["output_sha256"],
            first_score_trace_sha256=attached[0]["generation_trace"]["scores"][
                "trace_sha256"
            ],
            second_score_trace_sha256=attached[1]["generation_trace"]["scores"][
                "trace_sha256"
            ],
            first_raw_logit_trace_sha256=attached[0]["generation_trace"][
                "raw_logits"
            ]["trace_sha256"],
            second_raw_logit_trace_sha256=attached[1]["generation_trace"][
                "raw_logits"
            ]["trace_sha256"],
            precision_audits_identical=(
                attached[0]["precision_audit"] == attached[1]["precision_audit"]
            ),
        )
        repeat["comparison_score_vector_identity"] = (
            attached[0]["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ]
            == attached[1]["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ]
        )
        repeat["comparison_raw_logit_vector_identity"] = (
            attached[0]["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ]
            == attached[1]["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ]
        )
        repeat["passed"] = all(
            value for key, value in repeat.items() if key != "passed"
        )
        self.assertEqual(gate["attached_repeat_stability"], repeat)
        reference = gate["frozen_fp32_merged_reference"]
        reproduction = {
            "token_identity": (
                merged["token_count"] == reference["token_count"]
                and merged["token_ids_sha256"] == reference["token_ids_sha256"]
            ),
            "output_identity": merged["output_sha256"]
            == reference["output_sha256"],
            "score_trace_identity": merged["generation_trace"]["scores"][
                "trace_sha256"
            ]
            == reference["score_trace_sha256"],
            "raw_logit_trace_identity": merged["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
            == reference["raw_logit_trace_sha256"],
            "comparison_score_vector_identity": merged["generation_trace"][
                "scores"
            ]["comparison_step_vector_sha256"]
            == reference["comparison_score_vector_sha256"],
            "comparison_raw_logit_vector_identity": merged["generation_trace"][
                "raw_logits"
            ]["comparison_step_vector_sha256"]
            == reference["comparison_raw_logit_vector_sha256"],
        }
        reproduction["passed"] = all(reproduction.values())
        self.assertEqual(gate["merged_candidate_reproduction"], reproduction)
        token_analysis = analyze_same_dtype_tokens(
            attached[0]["generated_token_ids"],
            merged["generated_token_ids"],
        )
        self.assertEqual(
            gate["same_dtype_token_analysis"],
            {**token_analysis, "attached_token_text": None, "merged_token_text": None},
        )
        self.assertEqual(
            gate["comparison_step"],
            select_comparison_step(token_analysis, frozen_boundary_index=45),
        )
        score = gate["selection_score_evidence"]
        raw = gate["raw_logit_evidence"]
        trace_identity = {
            "token_identity": token_analysis["cross_path_identical"],
            "score_trace_identity": attached[0]["generation_trace"]["scores"][
                "trace_sha256"
            ]
            == merged["generation_trace"]["scores"]["trace_sha256"],
            "raw_logit_trace_identity": attached[0]["generation_trace"][
                "raw_logits"
            ]["trace_sha256"]
            == merged["generation_trace"]["raw_logits"]["trace_sha256"],
            "comparison_score_vector_identity": score["paths"][
                "fp32_attached_adapter"
            ]["comparison_vector_sha256"]
            == score["paths"]["fp32_safe_merged"]["comparison_vector_sha256"],
            "comparison_raw_logit_vector_identity": raw["paths"][
                "fp32_attached_adapter"
            ]["comparison_vector_sha256"]
            == raw["paths"]["fp32_safe_merged"]["comparison_vector_sha256"],
        }
        self.assertEqual(gate["same_dtype_trace_identity"], trace_identity)
        classification = classify_same_dtype_effect(
            token_analysis,
            attached_repeat_stable=repeat["passed"],
            merged_candidate_reproduced=reproduction["passed"],
            attached_emitted_token_id=score["paths"]["fp32_attached_adapter"][
                "emitted_token_id"
            ],
            merged_emitted_token_id=score["paths"]["fp32_safe_merged"][
                "emitted_token_id"
            ],
            attached_score_top_token_id=score["paths"]["fp32_attached_adapter"][
                "top_token_ids"
            ][0],
            merged_score_top_token_id=score["paths"]["fp32_safe_merged"][
                "top_token_ids"
            ][0],
            attached_raw_logit_top_token_id=raw["paths"][
                "fp32_attached_adapter"
            ]["top_token_ids"][0],
            merged_raw_logit_top_token_id=raw["paths"]["fp32_safe_merged"][
                "top_token_ids"
            ][0],
            full_score_traces_identical=trace_identity["score_trace_identity"],
            full_raw_logit_traces_identical=trace_identity[
                "raw_logit_trace_identity"
            ],
            comparison_score_vectors_identical=trace_identity[
                "comparison_score_vector_identity"
            ],
            comparison_raw_logit_vectors_identical=trace_identity[
                "comparison_raw_logit_vector_identity"
            ],
        )
        self.assertEqual(gate["classification"], classification)
        self.assertTrue(gate["isolation_gate"]["passed"])
        self.assertFalse(gate["remediation_gate"]["passed"])
        self.assertFalse(gate["runtime_eligible"])
        self.assertFalse(gate["merged_artifact_allowed"])
        self.assertTrue(math.isfinite(gate["elapsed_seconds"]))
        self.assertGreater(gate["elapsed_seconds"], 0)
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],
            "FC-MVP-001-fp32-attached-merge-numerics-v1",
        )

    def test_attached_repeat_stability_requires_exact_numeric_traces(self) -> None:
        passed = analyze_attached_repeat_stability([1, 2], [1, 2], **_stable())
        self.assertEqual(
            passed,
            {
                "token_identity": True,
                "output_identity": True,
                "score_trace_identity": True,
                "raw_logit_trace_identity": True,
                "precision_audit_identity": True,
                "passed": True,
            },
        )
        failed = analyze_attached_repeat_stability(
            [1, 2],
            [1, 2],
            **_stable(second_raw_logit_trace_sha256=SHA_A),
        )
        self.assertFalse(failed["raw_logit_trace_identity"])
        self.assertFalse(failed["passed"])

    def test_attached_repeat_stability_rejects_invalid_evidence(self) -> None:
        with self.assertRaises(ToolRouterValidationError):
            analyze_attached_repeat_stability(
                [1, True],
                [1, 2],
                **_stable(),
            )
        with self.assertRaises(ToolRouterValidationError):
            analyze_attached_repeat_stability(
                [1, 2],
                [1, 2],
                **_stable(first_output_sha256="sha256:invalid"),
            )

    def test_same_dtype_token_identity_and_drift_are_distinguished(self) -> None:
        identity = analyze_same_dtype_tokens([1, 2], [1, 2])
        self.assertEqual(identity["classification"], "same_dtype_token_identity")
        drift = analyze_same_dtype_tokens([1, 2, 3], [1, 4, 3])
        self.assertEqual(drift["classification"], "same_dtype_token_drift")
        self.assertEqual(drift["first_divergent_token_index"], 1)
        self.assertEqual(drift["attached_token_id"], 2)
        self.assertEqual(drift["merged_token_id"], 4)

    def test_comparison_step_uses_first_drift_or_frozen_context(self) -> None:
        drift = analyze_same_dtype_tokens([1, 2, 3], [1, 4, 3])
        self.assertEqual(
            select_comparison_step(drift, frozen_boundary_index=2),
            {
                "step_index": 1,
                "basis": "first_same_dtype_generated_token_divergence",
                "common_prefix_generated_tokens": 1,
            },
        )
        identity = analyze_same_dtype_tokens([1, 2, 3], [1, 2, 3])
        self.assertEqual(
            select_comparison_step(identity, frozen_boundary_index=2)["step_index"],
            2,
        )
        with self.assertRaises(ToolRouterValidationError):
            select_comparison_step(identity, frozen_boundary_index=3)

    def test_raw_logit_boundary_flip_is_classified(self) -> None:
        analysis = analyze_same_dtype_tokens([1, 11], [1, 12])
        self.assertEqual(
            _classify(analysis),
            "deterministic_fp32_attached_vs_merged_raw_logit_boundary_flip",
        )

    def test_processor_and_mixed_boundary_drift_are_distinguished(self) -> None:
        analysis = analyze_same_dtype_tokens([1, 11], [1, 12])
        self.assertEqual(
            _classify(
                analysis,
                attached_raw_logit_top_token_id=20,
                merged_raw_logit_top_token_id=20,
            ),
            "deterministic_fp32_attached_vs_merged_logits_processor_boundary_flip",
        )
        self.assertEqual(
            _classify(
                analysis,
                attached_raw_logit_top_token_id=20,
                merged_raw_logit_top_token_id=21,
            ),
            "deterministic_fp32_attached_vs_merged_mixed_logit_score_drift",
        )

    def test_identity_classifies_exact_or_numeric_trace_effect(self) -> None:
        analysis = analyze_same_dtype_tokens([1, 11], [1, 11])
        exact = _classify(
            analysis,
            merged_emitted_token_id=11,
            merged_score_top_token_id=11,
            merged_raw_logit_top_token_id=11,
            full_score_traces_identical=True,
            full_raw_logit_traces_identical=True,
            comparison_score_vectors_identical=True,
            comparison_raw_logit_vectors_identical=True,
        )
        self.assertEqual(
            exact,
            "deterministic_fp32_attached_merged_full_trace_identity",
        )
        numeric = _classify(
            analysis,
            merged_emitted_token_id=11,
            merged_score_top_token_id=11,
            merged_raw_logit_top_token_id=11,
        )
        self.assertEqual(
            numeric,
            "deterministic_fp32_attached_vs_merged_numerical_drift_without_token_drift",
        )

    def test_classification_fails_closed_before_cross_path_claim(self) -> None:
        analysis = analyze_same_dtype_tokens([1, 11], [1, 12])
        with self.assertRaises(ToolRouterValidationError):
            _classify(analysis, attached_repeat_stable=False)
        with self.assertRaises(ToolRouterValidationError):
            _classify(analysis, merged_candidate_reproduced=False)
        self.assertEqual(
            _classify(analysis, attached_score_top_token_id=99),
            "generation_score_alignment_failure",
        )

    def test_contradictory_identity_flags_fail_closed(self) -> None:
        identity = analyze_same_dtype_tokens([1, 11], [1, 11])
        with self.assertRaises(ToolRouterValidationError):
            _classify(
                identity,
                merged_emitted_token_id=11,
                merged_score_top_token_id=11,
                attached_raw_logit_top_token_id=20,
                merged_raw_logit_top_token_id=21,
                full_score_traces_identical=True,
                full_raw_logit_traces_identical=True,
                comparison_score_vectors_identical=True,
                comparison_raw_logit_vectors_identical=True,
            )
        drift = analyze_same_dtype_tokens([1, 11], [1, 12])
        with self.assertRaises(ToolRouterValidationError):
            _classify(
                drift,
                full_score_traces_identical=True,
                comparison_score_vectors_identical=True,
            )

    def test_forged_token_analysis_and_termination_fail_closed(self) -> None:
        forged = analyze_same_dtype_tokens([1, 11], [1, 12])
        forged["cross_path_identical"] = True
        with self.assertRaises(ToolRouterValidationError):
            _classify(forged)
        termination = analyze_same_dtype_tokens([1], [1, 2])
        self.assertEqual(
            termination["classification"],
            "same_dtype_termination_drift",
        )
        with self.assertRaises(ToolRouterValidationError):
            select_comparison_step(termination, frozen_boundary_index=0)


if __name__ == "__main__":
    unittest.main()
