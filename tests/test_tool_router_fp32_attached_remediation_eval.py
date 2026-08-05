from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError, load_fixture
from fullcycle_bridge.tool_router_fp32_attached_remediation_eval import (
    CANDIDATE_ID,
    EXPERIMENT_ID,
    FROZEN_BF16_COMPILED_METRICS,
    MAX_ELAPSED_SECONDS,
    MAX_PEAK_GPU_MEMORY_BYTES,
    RUN_ID,
    classify_candidate,
    compare_candidate,
    compile_candidate_outputs,
    score_compiled_candidate,
)
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "fixtures" / "tool_router_v1" / "eval.json"
RAW = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
REFERENCE = ROOT / "baseline" / "tool-router-lora-sft-v2-compiled-predictions.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


class ToolRouterFP32AttachedRemediationEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_fixture(EVAL)
        raw = _load(RAW)
        self.raw_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in raw["outputs"]
        ]
        _, self.raw_parsed = score_raw_outputs(self.records, self.raw_outputs)
        reference = _load(REFERENCE)
        self.reference_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in reference["outputs"]
        ]

    def assert_code(self, expected: str, function, *args, **kwargs) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, expected)

    def test_compilation_reproduces_frozen_reference_and_neutral_pass(self) -> None:
        compilation = compile_candidate_outputs(self.raw_outputs, self.raw_parsed)
        self.assertEqual(compilation["outputs"], self.reference_outputs)
        self.assertEqual(
            compilation["changed_example_ids"], ["eval-001", "eval-014", "eval-020"]
        )
        metrics, parsed = score_compiled_candidate(self.records, compilation)
        self.assertEqual(metrics, FROZEN_BF16_COMPILED_METRICS)
        self.assertEqual(len(parsed), 20)

        comparison = compare_candidate(
            self.records,
            compilation["outputs"],
            self.reference_outputs,
            elapsed_seconds=76.0,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=8_519_680,
        )
        assessment = classify_candidate(comparison)
        self.assertEqual(assessment["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(assessment["candidate_id"], CANDIDATE_ID)
        self.assertEqual(assessment["run_id"], RUN_ID)
        self.assertEqual(assessment["outcome"], "neutral")
        self.assertTrue(assessment["evaluation_gate_passed"])
        self.assertEqual(comparison["regression_event_count"], 0)

    def test_invalid_raw_output_is_preserved_as_negative_evidence(self) -> None:
        raw_outputs = copy.deepcopy(self.raw_outputs)
        raw_outputs[0]["raw_output"] = "not-json"
        _, parsed = score_raw_outputs(self.records, raw_outputs)
        compilation = compile_candidate_outputs(raw_outputs, parsed)
        self.assertEqual(compilation["outputs"][0]["raw_output"], "not-json")
        self.assertFalse(compilation["provenance"][0]["compilation_applied"])
        self.assertTrue(
            compilation["provenance"][0]["invalid_transparently_preserved"]
        )
        comparison = compare_candidate(
            self.records,
            compilation["outputs"],
            self.reference_outputs,
            elapsed_seconds=76.0,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=0,
        )
        assessment = classify_candidate(comparison)
        self.assertEqual(assessment["outcome"], "adverse")
        self.assertFalse(comparison["full_eval_gate_passed"])
        self.assertGreater(comparison["regression_event_count"], 0)

    def test_reference_and_parsed_tampering_fail_closed(self) -> None:
        parsed = copy.deepcopy(self.raw_parsed)
        parsed[0]["prediction"]["risk_level"] = "high"
        self.assert_code(
            "PARSED_OUTPUT_DRIFT",
            compile_candidate_outputs,
            self.raw_outputs,
            parsed,
        )

        reference = copy.deepcopy(self.reference_outputs)
        reference[0]["raw_output"] = "not-json"
        self.assert_code(
            "FROZEN_BF16_REFERENCE_DRIFT",
            compare_candidate,
            self.records,
            self.reference_outputs,
            reference,
            elapsed_seconds=76.0,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=0,
        )

    def test_resource_caps_and_nonfinite_values_fail_or_classify_adverse(self) -> None:
        comparison = compare_candidate(
            self.records,
            self.reference_outputs,
            self.reference_outputs,
            elapsed_seconds=MAX_ELAPSED_SECONDS + 0.01,
            peak_gpu_memory_bytes=MAX_PEAK_GPU_MEMORY_BYTES + 1,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=16_777_217,
        )
        assessment = classify_candidate(comparison)
        self.assertFalse(comparison["resource_gate_passed"])
        self.assertEqual(assessment["outcome"], "adverse")
        self.assertEqual(assessment["failure_reasons"], ["resource_gate_failed"])

        before_load_over_cap = compare_candidate(
            self.records,
            self.reference_outputs,
            self.reference_outputs,
            elapsed_seconds=76.0,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=16_777_217,
            released_gpu_memory_bytes=0,
        )
        before_load_assessment = classify_candidate(before_load_over_cap)
        self.assertFalse(
            before_load_over_cap["resource_comparison"]
            ["memory_allocated_before_load_bytes"]["within_cap"]
        )
        self.assertFalse(before_load_over_cap["resource_gate_passed"])
        self.assertEqual(before_load_assessment["outcome"], "adverse")
        self.assertEqual(
            before_load_assessment["failure_reasons"], ["resource_gate_failed"]
        )
        self.assert_code(
            "INVALID_POSITIVE_FINITE",
            compare_candidate,
            self.records,
            self.reference_outputs,
            self.reference_outputs,
            elapsed_seconds=math.nan,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=0,
        )

    def test_classification_rejects_regression_count_tampering(self) -> None:
        comparison = compare_candidate(
            self.records,
            self.reference_outputs,
            self.reference_outputs,
            elapsed_seconds=76.0,
            peak_gpu_memory_bytes=3_200_000_000,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=0,
        )
        comparison["regression_event_count"] = 1
        self.assert_code("REGRESSION_EVENT_COUNT_DRIFT", classify_candidate, comparison)


if __name__ == "__main__":
    unittest.main()
