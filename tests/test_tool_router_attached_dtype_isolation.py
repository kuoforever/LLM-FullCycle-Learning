from __future__ import annotations

import unittest

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_attached_dtype_isolation import (
    analyze_attached_dtype_tokens,
    analyze_path_repeat_stability,
    classify_attached_dtype_effect,
    select_locked_comparison_step,
)

ZERO = "sha256:" + "0" * 64
ONE = "sha256:" + "1" * 64
TWO = "sha256:" + "2" * 64


class AttachedDtypeIsolationAnalysisTests(unittest.TestCase):
    def _stability(self, **overrides: object) -> dict[str, bool]:
        values: dict[str, object] = {
            "first_output_sha256": ZERO,
            "second_output_sha256": ZERO,
            "first_score_trace_sha256": ONE,
            "second_score_trace_sha256": ONE,
            "first_raw_logit_trace_sha256": TWO,
            "second_raw_logit_trace_sha256": TWO,
            "first_score_vector_sha256": ZERO,
            "second_score_vector_sha256": ZERO,
            "first_raw_logit_vector_sha256": ONE,
            "second_raw_logit_vector_sha256": ONE,
            "precision_audits_identical": True,
        }
        values.update(overrides)
        return analyze_path_repeat_stability([10, 20], [10, 20], **values)  # type: ignore[arg-type]

    def _token_drift(self) -> dict[str, object]:
        shared_prefix = list(range(45))
        return analyze_attached_dtype_tokens(
            [*shared_prefix, 1866, 92, 151645],
            [*shared_prefix, 3849, 92, 151645],
        )

    def _classify(self, **overrides: object) -> str:
        values: dict[str, object] = {
            "bf16_repeat_stable": True,
            "fp32_repeat_stable": True,
            "bf16_reference_reproduced": True,
            "fp32_reference_reproduced": True,
            "bf16_emitted_token_id": 1866,
            "fp32_emitted_token_id": 3849,
            "bf16_score_top_token_id": 1866,
            "fp32_score_top_token_id": 3849,
            "bf16_raw_logit_top_token_id": 1866,
            "fp32_raw_logit_top_token_id": 3849,
        }
        values.update(overrides)
        return classify_attached_dtype_effect(
            self._token_drift(),
            **values,  # type: ignore[arg-type]
        )

    def test_repeat_stability_requires_every_identity(self) -> None:
        stable = self._stability()
        self.assertTrue(stable["passed"])
        mutations = {
            "second_output_sha256": ONE,
            "second_score_trace_sha256": TWO,
            "second_raw_logit_trace_sha256": ZERO,
            "second_score_vector_sha256": TWO,
            "second_raw_logit_vector_sha256": ZERO,
            "precision_audits_identical": False,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                self.assertFalse(self._stability(**{key: value})["passed"])

        token_drift = analyze_path_repeat_stability(
            [10, 20],
            [10, 21],
            first_output_sha256=ZERO,
            second_output_sha256=ZERO,
            first_score_trace_sha256=ONE,
            second_score_trace_sha256=ONE,
            first_raw_logit_trace_sha256=TWO,
            second_raw_logit_trace_sha256=TWO,
            first_score_vector_sha256=ZERO,
            second_score_vector_sha256=ZERO,
            first_raw_logit_vector_sha256=ONE,
            second_raw_logit_vector_sha256=ONE,
            precision_audits_identical=True,
        )
        self.assertFalse(token_drift["token_identity"])
        self.assertFalse(token_drift["passed"])

    def test_repeat_stability_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ToolRouterValidationError):
            self._stability(first_output_sha256="not-a-digest")
        with self.assertRaises(ToolRouterValidationError):
            analyze_path_repeat_stability(
                [True],
                [True],
                first_output_sha256=ZERO,
                second_output_sha256=ZERO,
                first_score_trace_sha256=ZERO,
                second_score_trace_sha256=ZERO,
                first_raw_logit_trace_sha256=ZERO,
                second_raw_logit_trace_sha256=ZERO,
                first_score_vector_sha256=ZERO,
                second_score_vector_sha256=ZERO,
                first_raw_logit_vector_sha256=ZERO,
                second_raw_logit_vector_sha256=ZERO,
                precision_audits_identical=True,
            )

    def test_token_analysis_covers_identity_drift_and_termination(self) -> None:
        identity = analyze_attached_dtype_tokens([1, 2], [1, 2])
        self.assertEqual(identity["classification"], "cross_dtype_token_identity")
        self.assertEqual(identity["common_prefix_generated_tokens"], 2)

        drift = analyze_attached_dtype_tokens([10, 20, 1866], [10, 20, 3849])
        self.assertEqual(drift["classification"], "cross_dtype_token_drift")
        self.assertEqual(drift["first_divergent_token_index"], 2)
        self.assertEqual(drift["bf16_token_id"], 1866)
        self.assertEqual(drift["fp32_token_id"], 3849)

        termination = analyze_attached_dtype_tokens([1], [1, 2])
        self.assertEqual(
            termination["classification"],
            "cross_dtype_termination_drift",
        )
        self.assertIsNone(termination["bf16_token_id"])
        self.assertEqual(termination["fp32_token_id"], 2)

    def test_locked_step_must_reproduce_frozen_boundary(self) -> None:
        selected = select_locked_comparison_step(
            self._token_drift(),
            frozen_boundary_index=45,
        )
        self.assertEqual(
            selected,
            {
                "step_index": 45,
                "basis": "frozen_first_cross_dtype_generated_token_divergence",
                "shared_generated_prefix_tokens": 45,
            },
        )
        with self.assertRaises(ToolRouterValidationError):
            select_locked_comparison_step(
                self._token_drift(),
                frozen_boundary_index=44,
            )
        with self.assertRaises(ToolRouterValidationError):
            select_locked_comparison_step(
                analyze_attached_dtype_tokens([1, 2], [1, 2]),
                frozen_boundary_index=1,
            )

    def test_classification_distinguishes_raw_processor_and_mixed_effects(self) -> None:
        self.assertEqual(
            self._classify(),
            "deterministic_bf16_attached_vs_fp32_attached_raw_logit_boundary_flip",
        )
        self.assertEqual(
            self._classify(
                bf16_raw_logit_top_token_id=1866,
                fp32_raw_logit_top_token_id=1866,
            ),
            (
                "deterministic_bf16_attached_vs_fp32_attached_"
                "logits_processor_boundary_flip"
            ),
        )
        self.assertEqual(
            self._classify(
                bf16_raw_logit_top_token_id=100,
                fp32_raw_logit_top_token_id=200,
            ),
            (
                "deterministic_bf16_attached_vs_fp32_attached_"
                "mixed_raw_logit_and_logits_processor_boundary_flip"
            ),
        )
        with self.assertRaises(ToolRouterValidationError):
            self._classify(fp32_score_top_token_id=1866)

    def test_classification_fails_closed_on_prerequisites_and_forgery(self) -> None:
        for key in (
            "bf16_repeat_stable",
            "fp32_repeat_stable",
            "bf16_reference_reproduced",
            "fp32_reference_reproduced",
        ):
            with self.subTest(key=key), self.assertRaises(
                ToolRouterValidationError
            ):
                self._classify(**{key: False})

        forged = self._token_drift()
        forged["common_prefix_generated_tokens"] = 1
        with self.assertRaises(ToolRouterValidationError):
            classify_attached_dtype_effect(
                forged,
                bf16_repeat_stable=True,
                fp32_repeat_stable=True,
                bf16_reference_reproduced=True,
                fp32_reference_reproduced=True,
                bf16_emitted_token_id=1866,
                fp32_emitted_token_id=3849,
                bf16_score_top_token_id=1866,
                fp32_score_top_token_id=3849,
                bf16_raw_logit_top_token_id=1866,
                fp32_raw_logit_top_token_id=3849,
            )

        with self.assertRaises(ToolRouterValidationError) as raised:
            self._classify(
                bf16_raw_logit_top_token_id=151_936,
                fp32_raw_logit_top_token_id=151_936,
            )
        self.assertEqual(raised.exception.code, "INVALID_TOKEN_ID")
        self.assertEqual(raised.exception.path, "$.bf16_raw_logit_top_token_id")


if __name__ == "__main__":
    unittest.main()
