from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError, load_fixture
from fullcycle_bridge import (
    tool_router_fp32_attached_remediation_eval_evidence as validator,
)
from fullcycle_bridge.tool_router_fp32_attached_remediation_eval import (
    CANDIDATE_ID,
    RUN_ID,
    classify_candidate,
    compare_candidate,
    compile_candidate_outputs,
    score_compiled_candidate,
)
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tool_router_fp32_attached_remediation_eval_v1.json"
EVAL = ROOT / "fixtures" / "tool_router_v1" / "eval.json"
RAW = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
REFERENCE_PREDICTIONS = (
    ROOT / "baseline" / "tool-router-lora-sft-v2-compiled-predictions.json"
)
REFERENCE_REPORT = ROOT / "baseline" / "tool-router-lora-sft-v2-compiled-report.json"
BOUNDARY = ROOT / "baseline" / "fc-mvp-001-attached-dtype-boundary-control-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class FP32AttachedRemediationEvalEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preregistration = _load(CONFIG)
        self.records = load_fixture(EVAL)
        self.raw_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in _load(RAW)["outputs"]
        ]
        self.reference_predictions = _load(REFERENCE_PREDICTIONS)
        self.reference_outputs = [
            {"example_id": item["example_id"], "raw_output": item["raw_output"]}
            for item in self.reference_predictions["outputs"]
        ]
        self.reference_report = _load(REFERENCE_REPORT)
        self.boundary = _load(BOUNDARY)
        self.preregistration_sha256 = _file_sha256(CONFIG)

    def _bundle(
        self,
        *,
        outputs: list[dict[str, Any]] | None = None,
        elapsed_seconds: float = 76.0,
        peak_gpu_memory_bytes: int = 3_200_000_000,
        released_gpu_memory_bytes: int = 8_519_680,
    ) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, str]]:
        raw_outputs = copy.deepcopy(outputs if outputs is not None else self.raw_outputs)
        artifact_lineage = validator._artifact_source_lineage(
            self.preregistration, self.preregistration_sha256
        )
        performance = {
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "memory_allocated_before_load_bytes": 0,
            "memory_allocated_after_release_bytes": released_gpu_memory_bytes,
        }
        predictions = {
            "artifact_version": 1,
            "experiment_id": validator.EXPERIMENT_ID,
            "gate_id": validator.GATE_ID,
            "preregistration_sha256": self.preregistration_sha256,
            "source_lineage": artifact_lineage,
            "model": self.preregistration["frozen_inputs"]["model"],
            "tokenizer": self.preregistration["frozen_inputs"]["tokenizer"],
            "environment": self.preregistration["frozen_inputs"]["environment"],
            "generation": self.preregistration["protocol"]["generation"],
            "prompt_sha256": self.preregistration["frozen_inputs"]["prompt"][
                "sha256"
            ],
            "eval_digest": self.preregistration["frozen_inputs"]["evaluation"][
                "digest"
            ],
            "example_order": self.preregistration["frozen_inputs"]["evaluation"][
                "order"
            ],
            "adapter_files": self.preregistration["frozen_inputs"]["adapter_files"],
            "storage_audit": self.preregistration["frozen_inputs"]["storage_audit"],
            "run": {
                "run_id": RUN_ID,
                "candidate_id": CANDIDATE_ID,
                "order_index": 0,
                "fresh_model_loads": 1,
                "full_eval_runs": 1,
                "generate_calls": 20,
                "retries": 0,
                "completed": True,
            },
            "precision_audit": validator._fp32_boundary_precision_audit(self.boundary),
            "performance": performance,
            "outputs": raw_outputs,
        }
        raw_metrics, raw_parsed = score_raw_outputs(self.records, raw_outputs)
        compilation = compile_candidate_outputs(raw_outputs, raw_parsed)
        compiled_metrics, compiled_parsed = score_compiled_candidate(
            self.records, compilation
        )
        comparison = compare_candidate(
            self.records,
            compilation["outputs"],
            self.reference_outputs,
            elapsed_seconds=elapsed_seconds,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            memory_allocated_before_load_bytes=0,
            released_gpu_memory_bytes=released_gpu_memory_bytes,
        )
        assessment = classify_candidate(comparison)
        resources = validator._expected_resources(
            performance, self.preregistration["resource_caps"]
        )
        prediction_bytes = _payload(predictions)
        prediction_sha256 = "sha256:" + hashlib.sha256(prediction_bytes).hexdigest()
        evidence = {
            "gate_version": 1,
            "experiment_id": validator.EXPERIMENT_ID,
            "gate_id": validator.GATE_ID,
            "preregistration_sha256": self.preregistration_sha256,
            "source_lineage": artifact_lineage,
            "prediction_artifact": {
                "path": "work/test-fixtures/candidate-predictions.json",
                "bytes": len(prediction_bytes),
                "sha256": prediction_sha256,
            },
            "raw_metrics": raw_metrics,
            "raw_parsed_outputs": raw_parsed,
            "compilation": compilation,
            "compiled_metrics": compiled_metrics,
            "compiled_parsed_outputs": compiled_parsed,
            "comparison": comparison,
            "assessment": assessment,
            "gates": assessment["gates"],
            "resources": resources,
            "constraints": self.preregistration["constraints"],
            "claims": self.preregistration["claims"],
            "locked_next_action": validator._locked_next_action(
                assessment, self.preregistration
            ),
            "compiled_model_saved": False,
            "tensor_payload_saved": False,
            "runtime_eligible": False,
            "runtime_eligibility_reason": assessment["classification"],
            "offline": True,
        }
        return predictions, evidence, prediction_sha256, artifact_lineage

    def _validate(
        self,
        predictions: dict[str, Any],
        evidence: dict[str, Any],
        prediction_sha256: str,
        lineage: dict[str, str],
        *,
        preregistration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return validator.validate_fp32_attached_remediation_eval_evidence(
            self.preregistration if preregistration is None else preregistration,
            predictions,
            evidence,
            evaluation=self.records,
            reference_compiled_report=self.reference_report,
            source_boundary_control=self.boundary,
            expected_source_lineage=lineage,
            expected_model=self.preregistration["frozen_inputs"]["model"],
            expected_tokenizer=self.preregistration["frozen_inputs"]["tokenizer"],
            expected_environment=self.preregistration["frozen_inputs"]["environment"],
            expected_adapter_files=self.preregistration["frozen_inputs"][
                "adapter_files"
            ],
            expected_preregistration_sha256=self.preregistration_sha256,
            expected_prediction_artifact_sha256=prediction_sha256,
        )

    def test_neutral_candidate_passes_full_recomputation(self) -> None:
        predictions, evidence, digest, lineage = self._bundle()
        result = self._validate(predictions, evidence, digest, lineage)
        self.assertTrue(result["frozen_gate_valid"])
        self.assertTrue(result["remediation_passed"])
        self.assertEqual(evidence["assessment"]["outcome"], "neutral")

    def test_favorable_candidate_passes_without_regression(self) -> None:
        outputs = copy.deepcopy(self.raw_outputs)
        outputs[0]["raw_output"] = json.dumps(
            self.records[0]["decision"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        predictions, evidence, digest, lineage = self._bundle(outputs=outputs)
        result = self._validate(predictions, evidence, digest, lineage)
        self.assertTrue(result["remediation_passed"])
        self.assertEqual(evidence["assessment"]["outcome"], "favorable")

    def test_adverse_resource_outcome_is_valid_negative_evidence(self) -> None:
        predictions, evidence, digest, lineage = self._bundle(
            elapsed_seconds=validator.MAX_ELAPSED_SECONDS + 0.01
        )
        result = self._validate(predictions, evidence, digest, lineage)
        self.assertTrue(result["frozen_gate_valid"])
        self.assertFalse(result["remediation_passed"])
        self.assertEqual(evidence["assessment"]["outcome"], "adverse")

    def test_empty_raw_output_is_frozen_as_valid_adverse_evidence(self) -> None:
        outputs = copy.deepcopy(self.raw_outputs)
        outputs[0]["raw_output"] = ""
        predictions, evidence, digest, lineage = self._bundle(outputs=outputs)

        result = self._validate(predictions, evidence, digest, lineage)

        self.assertTrue(result["frozen_gate_valid"])
        self.assertFalse(result["remediation_passed"])
        self.assertEqual(evidence["raw_metrics"]["invalid_outputs"], 1)
        self.assertEqual(evidence["compilation"]["invalid_source_outputs"], 1)
        self.assertEqual(
            evidence["compilation"]["invalid_preserved_example_ids"],
            ["eval-001"],
        )
        self.assertEqual(evidence["assessment"]["outcome"], "adverse")

    def test_resealed_summary_forgery_is_rejected(self) -> None:
        predictions, evidence, digest, lineage = self._bundle(
            elapsed_seconds=validator.MAX_ELAPSED_SECONDS + 0.01
        )
        forged = copy.deepcopy(evidence)
        forged["comparison"]["resource_gate_passed"] = True
        forged["assessment"]["outcome"] = "neutral"
        forged["assessment"]["evaluation_gate_passed"] = True
        forged["gates"]["resource"] = True
        forged["locked_next_action"] = validator._locked_next_action(
            forged["assessment"], self.preregistration
        )
        with self.assertRaises(ToolRouterValidationError):
            self._validate(predictions, forged, digest, lineage)

    def test_schema_precision_policy_and_number_tampering_fail_closed(self) -> None:
        predictions, evidence, digest, lineage = self._bundle()
        extra = copy.deepcopy(evidence)
        extra["promotion"] = True
        with self.assertRaisesRegex(ToolRouterValidationError, "INVALID_EVIDENCE_SCHEMA"):
            self._validate(predictions, extra, digest, lineage)

        precision = copy.deepcopy(predictions)
        precision["precision_audit"]["base_parameters"]["dtypes"] = {
            "bfloat16": 1_543_714_304
        }
        with self.assertRaisesRegex(ToolRouterValidationError, "PREDICTION_LOCK_MISMATCH"):
            self._validate(precision, evidence, digest, lineage)

        policy = copy.deepcopy(self.preregistration)
        policy["constraints"]["training"] = True
        with self.assertRaisesRegex(ToolRouterValidationError, "PREREGISTRATION_LOCK_MISMATCH"):
            self._validate(
                predictions,
                evidence,
                digest,
                lineage,
                preregistration=policy,
            )

        runner_source = copy.deepcopy(self.preregistration)
        runner_source["source_lineage"]["runner_source"]["sha256"] = (
            "sha256:" + "0" * 64
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError, "RUNNER_SOURCE_LOCK_MISMATCH"
        ):
            self._validate(
                predictions,
                evidence,
                digest,
                lineage,
                preregistration=runner_source,
            )

        number = copy.deepcopy(predictions)
        number["performance"]["elapsed_seconds"] = math.nan
        with self.assertRaisesRegex(ToolRouterValidationError, "NONFINITE_NUMBER"):
            self._validate(number, evidence, digest, lineage)


if __name__ == "__main__":
    unittest.main()
