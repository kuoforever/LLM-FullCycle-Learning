from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.consumer import canonical_json_bytes
from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_fp32_attached_artifact_eligibility import (
    INCOMPLETE_CLASSIFICATIONS,
    NEXT_GATE_ID,
    classify_package_requirements,
    inspect_adapter_safetensors,
    validate_fp32_attached_artifact_eligibility_review,
)
from scripts.review_tool_router_fp32_attached_artifact_eligibility import (
    build,
    load_review_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_WEIGHTS = (
    ROOT
    / "baseline"
    / "adapters"
    / "fc-mvp-001-lora-sft-v2"
    / "adapter_model.safetensors"
)


class FP32AttachedArtifactEligibilityTests(unittest.TestCase):
    inputs: dict[str, Any]
    review: dict[str, Any]
    source_roots: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_review_inputs()
        cls.review = build()
        cls.source_roots = copy.deepcopy(cls.inputs["source_hashes"])

    def _validate(
        self,
        *,
        review: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        values = copy.deepcopy(self.inputs if inputs is None else inputs)
        return validate_fp32_attached_artifact_eligibility_review(
            copy.deepcopy(self.review if review is None else review),
            **values,
            expected_source_hashes=copy.deepcopy(
                self.source_roots
                if expected_source_hashes is None
                else expected_source_hashes
            ),
        )

    def test_frozen_review_is_valid_negative_evidence(self) -> None:
        validation = self._validate()

        self.assertEqual(
            validation,
            {
                "frozen_review_valid": True,
                "upstream_evaluation_favorable": True,
                "repository_local_evidence_usable": True,
                "offline_artifact_eligible": False,
                "portable_package_eligible": False,
                "classification": INCOMPLETE_CLASSIFICATIONS["favorable"],
                "blocking_finding_count": 6,
                "next_gate": NEXT_GATE_ID,
                "runtime_eligible": False,
            },
        )
        self.assertEqual(
            self.review["quality_review"]["strict_per_example_improvements"],
            [{"example_id": "eval-016", "dimension": "arguments"}],
        )
        self.assertEqual(
            self.review["packaging_review"]["blocking_findings"],
            [
                "base_model_revision_binding_missing",
                "composite_manifest_missing",
                "package_use_and_limitations_documentation_incomplete",
                "portable_base_model_binding_missing",
                "required_compiler_binding_missing",
                "tokenizer_file_manifest_missing",
            ],
        )

    def test_package_rubric_is_outcome_neutral_and_requires_tokenizer_manifest(
        self,
    ) -> None:
        complete = {
            "base_model_revision_bound": True,
            "composite_manifest_present": True,
            "portable_base_model_bound": True,
            "required_compiler_bound": True,
            "tokenizer_file_manifest_bound": True,
            "use_and_limitations_documented": True,
        }
        favorable = classify_package_requirements(
            upstream_outcome="favorable",
            upstream_gate_passed=True,
            requirements=complete,
        )
        neutral = classify_package_requirements(
            upstream_outcome="neutral",
            upstream_gate_passed=True,
            requirements=complete,
        )
        self.assertTrue(favorable["offline_artifact_eligible"])
        self.assertTrue(neutral["offline_artifact_eligible"])
        self.assertEqual(favorable["requirements"], neutral["requirements"])

        missing_tokenizer = {**complete, "tokenizer_file_manifest_bound": False}
        decision = classify_package_requirements(
            upstream_outcome="favorable",
            upstream_gate_passed=True,
            requirements=missing_tokenizer,
        )
        self.assertFalse(decision["offline_artifact_eligible"])
        self.assertEqual(
            decision["blocking_findings"],
            ["tokenizer_file_manifest_missing"],
        )

    def test_adapter_safetensors_structure_is_exact(self) -> None:
        audit = inspect_adapter_safetensors(ADAPTER_WEIGHTS)

        self.assertEqual(audit["file_bytes"], 17_462_432)
        self.assertEqual(audit["data_bytes"], 17_432_576)
        self.assertEqual(audit["tensor_count"], 224)
        self.assertEqual(audit["module_count"], 112)
        self.assertEqual(audit["parameter_count"], 4_358_144)
        self.assertEqual(audit["dtype_tensor_counts"], {"F32": 224})
        self.assertEqual(
            audit["shape_counts"],
            {"1536x16": 56, "16x1536": 112, "256x16": 56},
        )

    def test_external_source_root_drift_fails_closed(self) -> None:
        changed = copy.deepcopy(self.inputs)
        changed["source_hashes"]["adapter_readme"] = "sha256:" + "0" * 64

        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SOURCE_HASH_ROOT_MISMATCH",
        ):
            self._validate(inputs=changed)

    def test_generation_and_upstream_gate_drift_fail_closed(self) -> None:
        generation = copy.deepcopy(self.inputs)
        generation["remediation_predictions"]["generation"]["tf32"] = True
        _replace_json_source(generation, "remediation_predictions")
        generation["remediation_gate"]["prediction_artifact"]["sha256"] = generation[
            "source_hashes"
        ]["remediation_predictions"]
        _replace_json_source(generation, "remediation_gate")
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "GENERATION_CONTRACT_MISMATCH",
        ):
            build_from_inputs(generation)

        gates = copy.deepcopy(self.inputs)
        gates["remediation_gate"]["gates"]["post_hoc"] = True
        gates["remediation_gate"]["assessment"]["gates"]["post_hoc"] = True
        _replace_json_source(gates, "remediation_gate")
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "UPSTREAM_EVALUATION_NOT_ELIGIBLE_FOR_REVIEW",
        ):
            build_from_inputs(gates)

    def test_missing_canary_digests_and_tensor_shape_summary_fail_closed(self) -> None:
        canary = copy.deepcopy(self.inputs)
        for run in canary["attached_dtype_isolation"]["runs"][1:3]:
            run["token_ids_sha256"] = None
            run["output_sha256"] = None
        _replace_json_source(canary, "attached_dtype_isolation")
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "FP32_ATTACHED_REPEAT_EVIDENCE_MISMATCH",
        ):
            build_from_inputs(canary)

        tensor = copy.deepcopy(self.inputs)
        tensor["adapter_tensor_audit"]["shape_counts"] = {"16x1536": 224}
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "ADAPTER_TENSOR_AUDIT_MISMATCH",
        ):
            self._validate(inputs=tensor)

    def test_resealed_decision_forgery_and_nonfinite_value_fail_closed(self) -> None:
        forged = copy.deepcopy(self.review)
        forged["eligibility_decision"]["offline_artifact_eligible"] = True
        forged.pop("report_digest")
        forged["report_digest"] = (
            "sha256:" + hashlib.sha256(canonical_json_bytes(forged)).hexdigest()
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "REVIEW_RECOMPUTATION_MISMATCH",
        ):
            self._validate(review=forged)

        nonfinite = copy.deepcopy(self.review)
        nonfinite["resource_review"]["elapsed_seconds"]["ratio"] = math.nan
        with self.assertRaisesRegex(ToolRouterValidationError, "NONFINITE_NUMBER"):
            self._validate(review=nonfinite)

    def test_parsed_objects_and_ignored_fields_must_match_source_payloads(self) -> None:
        changed = copy.deepcopy(self.inputs)
        model = changed["remediation_preregistration"]["frozen_inputs"]["model"]
        changed["adapter_config"]["base_model_name_or_path"] = model["repo_id"]
        changed["adapter_config"]["revision"] = model["revision"]
        changed["adapter_readme"] = (
            "compile_decision attached limitations runtime evaluation memory"
        )
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SOURCE_PAYLOAD_CONTENT_MISMATCH",
        ):
            self._validate(inputs=changed)

        extra = copy.deepcopy(self.inputs)
        extra["remediation_gate"]["artifact_promotion_override"] = True
        with self.assertRaisesRegex(
            ToolRouterValidationError,
            "SOURCE_PAYLOAD_CONTENT_MISMATCH",
        ):
            self._validate(inputs=extra)


def build_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    from fullcycle_bridge.tool_router_fp32_attached_artifact_eligibility import (
        build_fp32_attached_artifact_eligibility_review,
    )

    return build_fp32_attached_artifact_eligibility_review(**inputs)


def _replace_json_source(inputs: dict[str, Any], source_name: str) -> None:
    payload = (
        json.dumps(
            inputs[source_name],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    inputs["source_payloads"][source_name] = payload
    inputs["source_hashes"][source_name] = (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


if __name__ == "__main__":
    unittest.main()
