from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router_decision_compilation import (  # noqa: E402
    compile_decision,
)
from fullcycle_bridge.tool_router_fp32_attached_offline_package_manifest import (  # noqa: E402
    REPOSITORY_SOURCE_PATHS,
)
from fullcycle_bridge.tool_router_fp32_attached_offline_package_reproducibility import (  # noqa: E402
    ADAPTER_ROOT_RELATIVE_TO_REPOSITORY,
    BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION,
    BEHAVIORAL_DRIFT_CLASSIFICATION,
    EVALUATION_FILE_SHA256,
    EVALUATION_PATH,
    FAILURE_NEXT_GATE_ID,
    MANIFEST_PATH,
    PASS_CLASSIFICATION,
    PASS_NEXT_GATE_ID,
    PREREGISTRATION_PATH,
    PROTOCOL_SOURCE_PATHS,
    REFERENCE_EVIDENCE_PATH,
    REFERENCE_PREDICTIONS_PATH,
    RESOURCE_EXCEEDED_CLASSIFICATION,
    ReproducibilityContractError,
    artifact_json_bytes,
    authenticate_manifest_and_references,
    build_replay_artifact,
    build_reproducibility_evidence,
    canonical_json_bytes,
    classify_reproducibility_gates,
    compare_behavioral_replay,
    expected_preregistration,
    load_and_validate_preregistration,
    load_manifest_source_bundle,
    parse_strict_json_bytes,
    sha256_bytes,
    validate_materialization_receipt,
    validate_preregistration,
    validate_reproducibility_evidence,
)

FREEZE_COMMIT = "a" * 40
PROTOCOL_HASHES = {
    "contract_source": "sha256:" + "1" * 64,
    "materializer_source": "sha256:" + "2" * 64,
    "runner_source": "sha256:" + "3" * 64,
}


def _frozen_preregistration() -> dict[str, Any]:
    return expected_preregistration(
        freeze_status="frozen",
        protocol_source_hashes=PROTOCOL_HASHES,
    )


def _digest_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _precision_audit() -> dict[str, Any]:
    def inventory(tensors: int, elements: int) -> dict[str, Any]:
        return {
            "floating_tensors": tensors,
            "floating_elements": elements,
            "dtypes": {"float32": elements},
            "devices": {"cuda:0": elements},
        }

    return {
        "base_parameters": inventory(338, 1_543_714_304),
        "adapter_parameters": inventory(224, 4_358_144),
        "floating_buffers": inventory(1, 64),
        "lora_target_modules": 112,
        "lora_parameter_tensors": 224,
        "adapter_parameters_finite": True,
        "active_adapters": ["default"],
        "is_peft_model": True,
        "input_output_embeddings_tied": True,
        "attn_implementation": "sdpa",
        "attention_class": "Qwen2Attention",
        "output_attentions": False,
        "hf_device_map": None,
        "training": False,
        "autocast_enabled": False,
        "lora_dropout": {"modules": 112, "training_modules": 0},
        "autocast_adapter_dtype": True,
        "attached_execution_form": "attached_factorized_lora",
    }


def _clean_resolution(repository_bytes: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolution_version": 1,
        "package_id": "fc-mvp-001-fp32-attached-factorized-lora-package-v1",
        "manifest_file_sha256": (
            "sha256:4125f2eef2a4b8f07015169ac7fb77b830514e053a4624aa703e5f5a64943eb0"
        ),
        "caller_supplied_roots": True,
        "manifest_machine_paths_used": False,
        "adapter_local_base_path_used": False,
        "resolved": True,
        "eligible_for_clean_location_reproducibility_test": True,
        "offline_artifact_eligible": False,
        "runtime_eligible": False,
        "groups": [
            {
                "root_role": "base_model_and_tokenizer",
                "resolved": True,
                "expected_files": 9,
                "matched_files": 9,
                "matched_bytes": 3_098_971_928,
                "issues": [],
            },
            {
                "root_role": "adapter",
                "resolved": True,
                "expected_files": 3,
                "matched_files": 3,
                "matched_bytes": 17_468_332,
                "issues": [],
            },
            {
                "root_role": "repository",
                "resolved": True,
                "expected_files": 15,
                "matched_files": 15,
                "matched_bytes": repository_bytes,
                "issues": [],
            },
        ],
        "failure_mode": None,
    }
    result["resolution_digest"] = _digest_json(result)
    return result


def _receipt(
    preregistration: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "receipt_version": 1,
        "gate_id": ("FC-MVP-001-fp32-attached-offline-package-reproducibility-v1"),
        "experiment_id": (
            "fc-mvp-001-fp32-attached-offline-package-reproducibility-v1"
        ),
        "package_id": "fc-mvp-001-fp32-attached-factorized-lora-package-v1",
        "preregistration_sha256": sha256_bytes(artifact_json_bytes(preregistration)),
        "protocol_freeze_commit": FREEZE_COMMIT,
        "manifest_file_sha256": (
            "sha256:4125f2eef2a4b8f07015169ac7fb77b830514e053a4624aa703e5f5a64943eb0"
        ),
        "phase_order": preregistration["materialization_protocol"]["phase_order"],
        "destination": {
            "destination_id": "b" * 32,
            "caller_supplied_root": True,
            "root_was_absent": True,
            "root_created_exclusive": True,
            "children": {
                "repository": "repository",
                "base_model_and_tokenizer": "base_model_and_tokenizer",
            },
            "adapter_root_relative_to_repository": ADAPTER_ROOT_RELATIVE_TO_REPOSITORY,
            "absolute_paths_recorded": False,
            "symlinks_used": False,
            "reparse_points_used": False,
            "hardlinks_used": False,
            "overwrite_used": False,
        },
        "transport": {
            "repository_remote_url": (
                "https://github.com/kuoforever/reliable-agent-model-lifecycle.git"
            ),
            "fresh_git_checkout": True,
            "git_fetch_used": True,
            "git_lfs_checkout_used": True,
            "model_downloader_path": "scripts/download_pinned_tool_router_model.py",
            "model_downloader_sha256": (
                "sha256:1d0d3321a55b185128de020f4b5a2a9c3ecc22f5abb0535c4712c4fd545d3a28"
            ),
            "model_downloader_invoked": True,
            "destination_scoped_hf_home": True,
            "destination_scoped_cache": True,
            "network_used_during_materialization": True,
            "network_used_during_execution": False,
            "alternate_remote_used": False,
            "alternate_revision_fallback_used": False,
            "historical_adapter_base_path_used": False,
        },
        "clean_resolution_digest": resolution["resolution_digest"],
        "clean_groups": copy.deepcopy(resolution["groups"]),
        "protocol_sources": {
            name: {
                "path": preregistration["source_lineage"]["protocol_sources"][name][
                    "path"
                ],
                "sha256": digest,
                "bytes": 100 + index,
            }
            for index, (name, digest) in enumerate(sorted(PROTOCOL_HASHES.items()))
        },
        "materialization_passed": True,
        "issues": [],
    }


def _replay_output(example_id: str, raw_output: str, index: int) -> dict[str, Any]:
    source = json.loads(raw_output)
    compiled = compile_decision(source)
    changed_fields = [
        f"$.{key}" for key in sorted(source) if source.get(key) != compiled.get(key)
    ]
    return {
        "example_id": example_id,
        "rendered_prompt_sha256": sha256_bytes(f"prompt-{index}".encode()),
        "input_token_ids_sha256": _digest_json([index, 100 + index]),
        "input_token_count": 2,
        "output_token_ids_sha256": _digest_json([index, 200 + index]),
        "output_token_count": 2,
        "raw_output": raw_output,
        "raw_output_utf8_sha256": sha256_bytes(raw_output.encode("utf-8")),
        "compiler_valid": True,
        "compiler_input_canonical_sha256": _digest_json(source),
        "compiled_output": compiled,
        "compiled_output_canonical_sha256": _digest_json(compiled),
        "compiler_changed_fields": changed_fields,
        "compilation_error": None,
    }


class OfflinePackageReproducibilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preregistration = _frozen_preregistration()
        cls.preregistration_payload = artifact_json_bytes(cls.preregistration)
        cls.preregistration_sha256 = sha256_bytes(cls.preregistration_payload)
        cls.bundle = load_manifest_source_bundle(
            repository_root=ROOT,
            adapter_root=ROOT / ADAPTER_ROOT_RELATIVE_TO_REPOSITORY,
        )
        cls.manifest_payload = (ROOT / MANIFEST_PATH).read_bytes()
        cls.reference_predictions_payload = (
            ROOT / REFERENCE_PREDICTIONS_PATH
        ).read_bytes()
        cls.reference_evidence_payload = (ROOT / REFERENCE_EVIDENCE_PATH).read_bytes()
        cls.evaluation_payload = (ROOT / EVALUATION_PATH).read_bytes()
        cls.authenticated = authenticate_manifest_and_references(
            cls.preregistration,
            manifest_payload=cls.manifest_payload,
            reference_predictions_payload=cls.reference_predictions_payload,
            reference_evidence_payload=cls.reference_evidence_payload,
            evaluation_payload=cls.evaluation_payload,
            manifest_sources=cls.bundle,
        )
        cls.repository_bytes = sum(
            len(cls.bundle.source_payloads[name]) for name in REPOSITORY_SOURCE_PATHS
        )
        cls.resolution = _clean_resolution(cls.repository_bytes)
        cls.receipt = _receipt(cls.preregistration, cls.resolution)
        cls.outputs = [
            _replay_output(item["example_id"], item["raw_output"], index)
            for index, item in enumerate(cls.authenticated.reference_outputs)
        ]
        cls.environment = cls.authenticated.manifest["components"]["environment"][
            "recorded_environment"
        ]
        cls.performance = {
            "elapsed_seconds": 70.0,
            "peak_gpu_memory_bytes": 6_200_000_000,
            "memory_allocated_before_load_bytes": 0,
            "memory_allocated_after_release_bytes": 8_519_680,
        }

    def _artifacts(
        self,
        *,
        performance: dict[str, Any] | None = None,
        outputs: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        replay = build_replay_artifact(
            self.preregistration,
            self.authenticated,
            preregistration_sha256=self.preregistration_sha256,
            protocol_freeze_commit=FREEZE_COMMIT,
            materialization_receipt=self.receipt,
            clean_resolution=self.resolution,
            observed_environment=self.environment,
            precision_audit=_precision_audit(),
            performance=self.performance if performance is None else performance,
            outputs=self.outputs if outputs is None else outputs,
        )
        evidence = build_reproducibility_evidence(
            self.preregistration,
            self.authenticated,
            preregistration_sha256=self.preregistration_sha256,
            protocol_freeze_commit=FREEZE_COMMIT,
            materialization_receipt=self.receipt,
            clean_resolution=self.resolution,
            replay_artifact=replay,
            replay_artifact_path=self.preregistration["execution_protocol"][
                "output_policy"
            ]["replay_file"],
        )
        return replay, evidence

    def assert_code(self, code: str, callback: Any) -> None:
        with self.assertRaises(ReproducibilityContractError) as raised:
            callback()
        self.assertEqual(raised.exception.code, code)

    def test_frozen_config_exactly_binds_protocol_sources(self) -> None:
        loaded = load_and_validate_preregistration(
            ROOT / PREREGISTRATION_PATH,
        )
        observed_hashes = {
            name: sha256_bytes((ROOT / path).read_bytes())
            for name, path in PROTOCOL_SOURCE_PATHS.items()
        }
        self.assertEqual(loaded.data["freeze_status"], "frozen")
        self.assertEqual(
            loaded.data,
            expected_preregistration(
                freeze_status="frozen",
                protocol_source_hashes=observed_hashes,
            ),
        )
        draft = expected_preregistration(
            freeze_status="draft",
            protocol_source_hashes={
                name: "sha256:" + "0" * 64 for name in observed_hashes
            },
        )
        self.assert_code(
            "PREREGISTRATION_NOT_FROZEN",
            lambda: validate_preregistration(draft),
        )
        validate_preregistration(self.preregistration)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        self.assert_code(
            "DUPLICATE_JSON_KEY",
            lambda: parse_strict_json_bytes(b'{"x":1,"x":2}', path="$"),
        )
        self.assert_code(
            "NONFINITE_JSON_NUMBER",
            lambda: parse_strict_json_bytes(b'{"x":NaN}', path="$"),
        )

    def test_external_reference_hashes_are_checked_before_parse(self) -> None:
        self.assert_code(
            "RAW_PAYLOAD_HASH_MISMATCH",
            lambda: authenticate_manifest_and_references(
                self.preregistration,
                manifest_payload=self.manifest_payload + b" ",
                reference_predictions_payload=self.reference_predictions_payload,
                reference_evidence_payload=self.reference_evidence_payload,
                evaluation_payload=self.evaluation_payload,
                manifest_sources=self.bundle,
            ),
        )
        self.assertEqual(sha256_bytes(self.evaluation_payload), EVALUATION_FILE_SHA256)

    def test_exact_replay_comparison_passes_and_raw_drift_is_adverse(self) -> None:
        passed = compare_behavioral_replay(self.authenticated, self.outputs)
        self.assertTrue(passed["behavioral_reproducibility_established"])
        self.assertEqual(passed["raw_outputs_exact"], 20)
        self.assertEqual(passed["compiled_outputs_exact"], 20)

        drifted = copy.deepcopy(self.outputs)
        raw = drifted[0]["raw_output"].replace("capability_unavailable", "changed")
        drifted[0] = _replay_output("eval-001", raw, 0)
        result = compare_behavioral_replay(self.authenticated, drifted)
        self.assertFalse(result["behavioral_reproducibility_established"])
        self.assertEqual(result["raw_mismatch_example_ids"], ["eval-001"])

    def test_replay_compiler_receipt_cannot_self_authorize(self) -> None:
        drifted = copy.deepcopy(self.outputs)
        drifted[0]["compiled_output_canonical_sha256"] = "sha256:" + "f" * 64
        self.assert_code(
            "REPLAY_COMPILER_RECEIPT_MISMATCH",
            lambda: compare_behavioral_replay(self.authenticated, drifted),
        )

    def test_materialization_receipt_is_closed_and_path_redacted(self) -> None:
        validated = validate_materialization_receipt(
            self.preregistration,
            self.receipt,
            preregistration_sha256=self.preregistration_sha256,
            expected_freeze_commit=FREEZE_COMMIT,
            clean_resolution=self.resolution,
        )
        self.assertFalse(validated["destination"]["absolute_paths_recorded"])
        forged = copy.deepcopy(self.receipt)
        forged["transport"]["historical_adapter_base_path_used"] = True
        self.assert_code(
            "MATERIALIZATION_RECEIPT_MISMATCH",
            lambda: validate_materialization_receipt(
                self.preregistration,
                forged,
                preregistration_sha256=self.preregistration_sha256,
                expected_freeze_commit=FREEZE_COMMIT,
                clean_resolution=self.resolution,
            ),
        )

    def test_pass_evidence_keeps_all_artifact_and_runtime_claims_false(self) -> None:
        _, evidence = self._artifacts()
        self.assertEqual(evidence["classification"], PASS_CLASSIFICATION)
        self.assertTrue(evidence["formal_gate_passed"])
        self.assertEqual(
            evidence["remaining_blocking_findings"],
            ["remote_revision_origin_unverified"],
        )
        self.assertEqual(evidence["locked_next_action"]["gate_id"], PASS_NEXT_GATE_ID)
        for key in (
            "remote_revision_origin_attested",
            "offline_artifact_eligible",
            "portable_package_eligible",
            "preferred_offline_candidate",
            "serving_readiness_established",
            "artifact_promotion_allowed",
            "merged_artifact_allowed",
            "runtime_eligible",
        ):
            self.assertFalse(evidence["derived_claims"][key])

    def test_failure_classification_algebra_is_outcome_neutral(self) -> None:
        gates = {
            "metadata_validation": True,
            "materialization": True,
            "clean_location_resolution": True,
            "environment": True,
            "execution_contract": True,
            "behavioral_replay": False,
            "resources": True,
        }
        self.assertEqual(
            classify_reproducibility_gates(gates), BEHAVIORAL_DRIFT_CLASSIFICATION
        )
        gates["behavioral_replay"] = True
        gates["resources"] = False
        self.assertEqual(
            classify_reproducibility_gates(gates), RESOURCE_EXCEEDED_CLASSIFICATION
        )
        gates["behavioral_replay"] = False
        self.assertEqual(
            classify_reproducibility_gates(gates),
            BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION,
        )

    def test_resource_only_failure_has_stable_blockers_and_adverse_action(self) -> None:
        performance = copy.deepcopy(self.performance)
        performance["elapsed_seconds"] = (
            self.preregistration["resource_caps"]["elapsed_seconds_max"] + 1.0
        )
        _, evidence = self._artifacts(performance=performance)

        self.assertEqual(evidence["classification"], RESOURCE_EXCEEDED_CLASSIFICATION)
        self.assertFalse(evidence["formal_gate_passed"])
        self.assertTrue(
            evidence["derived_claims"]["behavioral_reproducibility_established"]
        )
        self.assertEqual(
            evidence["remaining_blocking_findings"],
            ["resource_budget_exceeded", "remote_revision_origin_unverified"],
        )
        self.assertEqual(
            evidence["locked_next_action"]["gate_id"], FAILURE_NEXT_GATE_ID
        )
        self.assertEqual(
            evidence["locked_next_action"]["classification"],
            RESOURCE_EXCEEDED_CLASSIFICATION,
        )
        self.assertEqual(
            evidence["locked_next_action"]["remaining_blocking_findings"],
            evidence["remaining_blocking_findings"],
        )

    def test_behavior_and_resource_failure_has_stable_blocker_order(self) -> None:
        performance = copy.deepcopy(self.performance)
        performance["elapsed_seconds"] = (
            self.preregistration["resource_caps"]["elapsed_seconds_max"] + 1.0
        )
        outputs = copy.deepcopy(self.outputs)
        outputs[0]["raw_output"] += " "
        outputs[0]["raw_output_utf8_sha256"] = sha256_bytes(
            outputs[0]["raw_output"].encode("utf-8")
        )
        _, evidence = self._artifacts(performance=performance, outputs=outputs)

        self.assertEqual(
            evidence["classification"],
            BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION,
        )
        self.assertFalse(evidence["formal_gate_passed"])
        self.assertFalse(
            evidence["derived_claims"]["behavioral_reproducibility_established"]
        )
        self.assertEqual(evidence["comparison"]["raw_outputs_exact"], 19)
        self.assertEqual(evidence["comparison"]["compiled_outputs_exact"], 20)
        self.assertEqual(
            evidence["remaining_blocking_findings"],
            [
                "behavioral_reproducibility_unverified",
                "resource_budget_exceeded",
                "remote_revision_origin_unverified",
            ],
        )
        self.assertEqual(
            evidence["locked_next_action"]["gate_id"], FAILURE_NEXT_GATE_ID
        )
        self.assertEqual(
            evidence["locked_next_action"]["classification"],
            BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION,
        )
        self.assertEqual(
            evidence["locked_next_action"]["remaining_blocking_findings"],
            evidence["remaining_blocking_findings"],
        )

    def test_post_run_validator_recomputes_the_complete_pass(self) -> None:
        replay, evidence = self._artifacts()
        replay_payload = artifact_json_bytes(replay)
        evidence_payload = artifact_json_bytes(evidence)
        result = validate_reproducibility_evidence(
            self.preregistration_payload,
            replay_payload,
            evidence_payload,
            expected_preregistration_sha256=self.preregistration_sha256,
            expected_replay_artifact_sha256=sha256_bytes(replay_payload),
            expected_evidence_sha256=sha256_bytes(evidence_payload),
            expected_protocol_freeze_commit=FREEZE_COMMIT,
            replay_artifact_path=evidence["replay_artifact"]["path"],
            manifest_payload=self.manifest_payload,
            reference_predictions_payload=self.reference_predictions_payload,
            reference_evidence_payload=self.reference_evidence_payload,
            evaluation_payload=self.evaluation_payload,
            manifest_sources=self.bundle,
        )
        self.assertTrue(result["frozen_gate_valid"])
        self.assertEqual(result["classification"], PASS_CLASSIFICATION)

    def test_post_run_validator_rejects_prediction_tamper(self) -> None:
        replay, evidence = self._artifacts()
        replay["outputs"][0]["raw_output"] += " "
        replay_payload = artifact_json_bytes(replay)
        evidence_payload = artifact_json_bytes(evidence)
        self.assert_code(
            "RAW_OUTPUT_DIGEST_MISMATCH",
            lambda: validate_reproducibility_evidence(
                self.preregistration_payload,
                replay_payload,
                evidence_payload,
                expected_preregistration_sha256=self.preregistration_sha256,
                expected_replay_artifact_sha256=sha256_bytes(replay_payload),
                expected_evidence_sha256=sha256_bytes(evidence_payload),
                expected_protocol_freeze_commit=FREEZE_COMMIT,
                replay_artifact_path=evidence["replay_artifact"]["path"],
                manifest_payload=self.manifest_payload,
                reference_predictions_payload=self.reference_predictions_payload,
                reference_evidence_payload=self.reference_evidence_payload,
                evaluation_payload=self.evaluation_payload,
                manifest_sources=self.bundle,
            ),
        )

    def test_post_run_validator_rejects_self_authorized_decision_drift(self) -> None:
        replay, evidence = self._artifacts()
        replay_payload = artifact_json_bytes(replay)
        for mutate in (
            lambda item: item["derived_claims"].__setitem__("runtime_eligible", True),
            lambda item: item.__setitem__("classification", "self_authorized"),
            lambda item: item["locked_next_action"].__setitem__("gate_id", "unsafe"),
            lambda item: item.__setitem__("remaining_blocking_findings", []),
        ):
            forged = copy.deepcopy(evidence)
            mutate(forged)
            forged_payload = artifact_json_bytes(forged)
            with self.subTest(forged=forged.get("classification")):
                self.assert_code(
                    "EVIDENCE_RECOMPUTATION_MISMATCH",
                    lambda: validate_reproducibility_evidence(
                        self.preregistration_payload,
                        replay_payload,
                        forged_payload,
                        expected_preregistration_sha256=self.preregistration_sha256,
                        expected_replay_artifact_sha256=sha256_bytes(replay_payload),
                        expected_evidence_sha256=sha256_bytes(forged_payload),
                        expected_protocol_freeze_commit=FREEZE_COMMIT,
                        replay_artifact_path=evidence["replay_artifact"]["path"],
                        manifest_payload=self.manifest_payload,
                        reference_predictions_payload=self.reference_predictions_payload,
                        reference_evidence_payload=self.reference_evidence_payload,
                        evaluation_payload=self.evaluation_payload,
                        manifest_sources=self.bundle,
                    ),
                )

    def test_post_run_validator_rejects_duplicate_and_nonfinite_evidence(self) -> None:
        replay, evidence = self._artifacts()
        replay_payload = artifact_json_bytes(replay)
        evidence_payload = artifact_json_bytes(evidence)
        duplicate = evidence_payload.replace(
            b"{\n", b'{\n  "evidence_version": 1,\n', 1
        )
        nonfinite = evidence_payload.replace(
            b'"evidence_version": 1', b'"evidence_version": NaN', 1
        )
        for payload, code in (
            (duplicate, "DUPLICATE_JSON_KEY"),
            (nonfinite, "NONFINITE_JSON_NUMBER"),
        ):
            with self.subTest(code=code):
                self.assert_code(
                    code,
                    lambda: validate_reproducibility_evidence(
                        self.preregistration_payload,
                        replay_payload,
                        payload,
                        expected_preregistration_sha256=self.preregistration_sha256,
                        expected_replay_artifact_sha256=sha256_bytes(replay_payload),
                        expected_evidence_sha256=sha256_bytes(payload),
                        expected_protocol_freeze_commit=FREEZE_COMMIT,
                        replay_artifact_path=evidence["replay_artifact"]["path"],
                        manifest_payload=self.manifest_payload,
                        reference_predictions_payload=self.reference_predictions_payload,
                        reference_evidence_payload=self.reference_evidence_payload,
                        evaluation_payload=self.evaluation_payload,
                        manifest_sources=self.bundle,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
