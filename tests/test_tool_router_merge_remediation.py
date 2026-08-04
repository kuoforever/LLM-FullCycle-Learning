from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from fullcycle_bridge.tool_router import (
    ToolRouterValidationError,
    fixture_digest,
    load_fixture,
)
from fullcycle_bridge.tool_router_merge_remediation import (
    analyze_candidate_runs,
    token_ids_sha256,
)
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "baseline" / "fc-mvp-001-bf16-merge-remediation-v1.json"
STABILITY = ROOT / "baseline" / "fc-mvp-001-bf16-merge-stability-v1.json"
NUMERICS = ROOT / "baseline" / "fc-mvp-001-bf16-merge-numerics-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


class ToolRouterMergeRemediationTests(unittest.TestCase):
    def test_frozen_fp32_candidate_records_reproducible_negative_gate(self) -> None:
        gate = _load(GATE)
        stability = _load(STABILITY)
        config = _load(ROOT / "configs" / "tool_router_lora_sft_v2.json")
        training = _load(
            ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
        )
        adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
        evaluation = load_fixture(ROOT / config["data"]["eval_path"])

        self.assertEqual(
            set(gate),
            {
                "merge_remediation_version",
                "experiment_id",
                "source_experiment_id",
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
                "reference",
                "frozen_bf16_merged_control",
                "candidate_protocol",
                "runs",
                "analysis",
                "classification",
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
        self.assertEqual(gate["merge_remediation_version"], 1)
        self.assertEqual(
            gate["experiment_id"], "fc-mvp-001-bf16-merge-remediation-v1"
        )
        self.assertEqual(gate["source_experiment_id"], config["experiment_id"])
        self.assertEqual(gate["config_sha256"], canonical_config_sha256(config))
        self.assertEqual(gate["adapter_files"], directory_artifact_manifest(adapter))
        self.assertEqual(gate["adapter_files"], training["final_adapter"]["files"])
        self.assertEqual(gate["stability_evidence_sha256"], file_sha256(STABILITY))
        self.assertEqual(gate["numerics_evidence_sha256"], file_sha256(NUMERICS))
        self.assertEqual(
            gate["training_lock_sha256"],
            file_sha256(ROOT / "requirements" / "training.lock"),
        )
        self.assertEqual(gate["eval_digest"], fixture_digest(evaluation))
        self.assertEqual(
            gate["model_weight_sha256"],
            f"sha256:{config['model']['weight_sha256']}",
        )
        self.assertEqual(
            gate["prompt_sha256"],
            file_sha256(ROOT / config["prompt"]["path"]),
        )
        self.assertEqual(gate["example_id"], "eval-001")
        self.assertEqual(gate["input_token_count"], stability["input_token_count"])
        self.assertEqual(
            gate["input_token_ids_sha256"],
            stability["input_token_ids_sha256"],
        )
        self.assertEqual(
            gate["storage_audit"],
            {
                "base_checkpoint": {
                    "tensors": 338,
                    "elements": 1543714304,
                    "dtype_tensors": {"bfloat16": 338},
                    "dtype_elements": {"bfloat16": 1543714304},
                },
                "adapter": {
                    "tensors": 224,
                    "elements": 4358144,
                    "dtype_tensors": {"float32": 224},
                    "dtype_elements": {"float32": 4358144},
                },
            },
        )

        independent = [
            run for run in stability["runs"] if run["path"] == "independent"
        ]
        merged = [run for run in stability["runs"] if run["path"] == "merged"]
        reference = gate["reference"]
        control = gate["frozen_bf16_merged_control"]
        self.assertEqual(reference["token_count"], independent[0]["token_count"])
        self.assertEqual(
            reference["token_ids_sha256"], independent[0]["token_ids_sha256"]
        )
        self.assertEqual(reference["output_sha256"], independent[0]["output_sha256"])
        self.assertEqual(control["token_count"], merged[0]["token_count"])
        self.assertEqual(control["token_ids_sha256"], merged[0]["token_ids_sha256"])
        self.assertEqual(control["output_sha256"], merged[0]["output_sha256"])

        runs = gate["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual([run["repeat"] for run in runs], [1, 2])
        self.assertTrue(all(run["path"] == "fp32_safe_merged" for run in runs))
        self.assertEqual(runs[0]["token_ids_sha256"], runs[1]["token_ids_sha256"])
        self.assertEqual(runs[0]["output_sha256"], runs[1]["output_sha256"])
        self.assertEqual(runs[0]["token_count"], reference["token_count"])
        self.assertEqual(runs[0]["token_count"], control["token_count"])
        self.assertEqual(runs[0]["token_ids_sha256"], control["token_ids_sha256"])
        self.assertNotEqual(
            runs[0]["token_ids_sha256"], reference["token_ids_sha256"]
        )
        residual_ceiling = gate["candidate_protocol"]["max_residual_cuda_bytes"]
        for run in runs:
            precision = run["precision_audit"]
            pre = precision["pre_merge"]
            post = precision["post_merge"]
            generation = precision["generation"]
            self.assertEqual(pre["lora_target_modules"], 112)
            self.assertEqual(pre["lora_parameter_tensors"], 224)
            self.assertEqual(
                pre["base_parameters"]["dtypes"], {"float32": 1543714304}
            )
            self.assertEqual(
                pre["adapter_parameters"]["dtypes"], {"float32": 4358144}
            )
            self.assertEqual(post["lora_target_modules"], 0)
            self.assertEqual(post["lora_parameter_tensors"], 0)
            self.assertEqual(
                post["parameters"]["dtypes"], {"float32": 1543714304}
            )
            self.assertEqual(generation["score_dtypes"], ["float32"])
            self.assertFalse(generation["autocast_enabled"])
            self.assertLessEqual(
                run["memory_allocated_before_load_bytes"], residual_ceiling
            )
            self.assertLessEqual(
                run["memory_allocated_after_release_bytes"], residual_ceiling
            )

        self.assertEqual(
            set(gate["acceptance"]),
            {
                "upstream_evidence_locked",
                "frozen_input_reproduced",
                "candidate_runs_completed",
                "candidate_result_classified",
                "candidate_protocol_executed",
                "source_storage_dtypes_locked",
                "frozen_bf16_merged_control_compared",
                "fresh_load_memory_isolated",
                "source_adapter_unchanged",
                "source_model_unchanged",
                "eval_digest_unchanged",
                "prompt_digest_unchanged",
            },
        )
        self.assertTrue(all(gate["acceptance"].values()))
        self.assertEqual(gate["classification"], "deterministic_fp32_merge_output_drift")
        self.assertTrue(gate["remediation_gate"]["candidate_repeats_identical"])
        self.assertFalse(
            gate["remediation_gate"]["independent_bf16_reference_token_identity"]
        )
        self.assertTrue(
            gate["remediation_gate"]["frozen_bf16_merged_token_identity"]
        )
        self.assertEqual(
            set(gate["remediation_gate"]),
            {
                "candidate_repeats_identical",
                "independent_bf16_reference_token_identity",
                "independent_bf16_reference_output_identity",
                "frozen_bf16_merged_token_identity",
                "frozen_bf16_merged_output_identity",
                "passed",
            },
        )
        self.assertFalse(gate["remediation_gate"]["passed"])
        self.assertFalse(gate["merged_artifact_saved"])
        self.assertFalse(gate["merged_artifact_allowed"])
        self.assertEqual(
            gate["constraints"],
            {
                "new_data": False,
                "training": False,
                "eval_answer_tuning": False,
                "runtime_integration": False,
                "full_eval_run": False,
                "merged_artifact_promotion": False,
            },
        )
        self.assertFalse(gate["runtime_eligible"])
        self.assertEqual(
            gate["runtime_eligibility_reason"],
            "deterministic_fp32_merge_output_drift",
        )
        self.assertTrue(gate["offline"])
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],
            "FC-MVP-001-fp32-merge-drift-analysis-v1",
        )

    def test_two_identical_candidates_restore_frozen_reference_identity(self) -> None:
        reference = [10, 20, 30]
        analysis = analyze_candidate_runs(
            [reference, reference],
            reference_token_count=len(reference),
            reference_token_ids_sha256=token_ids_sha256(reference),
        )
        self.assertTrue(analysis["candidate_repeats_identical"])
        self.assertTrue(analysis["reference_token_count_match"])
        self.assertTrue(analysis["reference_token_digest_match"])
        self.assertTrue(analysis["independent_bf16_reference_identity"])
        self.assertEqual(
            analysis["classification"],
            "fp32_safe_merge_output_identity_restored",
        )

    def test_candidate_nondeterminism_precedes_reference_comparison(self) -> None:
        reference = [10, 20, 30]
        analysis = analyze_candidate_runs(
            [reference, [10, 20, 31]],
            reference_token_count=len(reference),
            reference_token_ids_sha256=token_ids_sha256(reference),
        )
        self.assertFalse(analysis["candidate_repeats_identical"])
        self.assertFalse(analysis["independent_bf16_reference_identity"])
        self.assertEqual(
            analysis["classification"],
            "fp32_candidate_within_path_nondeterminism",
        )

    def test_repeat_stable_reference_drift_is_not_mislabeled_as_repaired(self) -> None:
        reference = [10, 20, 30]
        candidate = [10, 20, 31]
        analysis = analyze_candidate_runs(
            [candidate, candidate],
            reference_token_count=len(reference),
            reference_token_ids_sha256=token_ids_sha256(reference),
        )
        self.assertTrue(analysis["candidate_repeats_identical"])
        self.assertFalse(analysis["reference_token_digest_match"])
        self.assertEqual(
            analysis["classification"],
            "deterministic_fp32_merge_output_drift",
        )

    def test_wrong_repeat_count_and_invalid_reference_fail_closed(self) -> None:
        digest = token_ids_sha256([1, 2])
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_candidate_runs(
                [[1, 2]],
                reference_token_count=2,
                reference_token_ids_sha256=digest,
            )
        self.assertEqual(raised.exception.code, "INVALID_CANDIDATE_REPEAT_COUNT")
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_candidate_runs(
                [[1, 2], [1, 2]],
                reference_token_count=0,
                reference_token_ids_sha256=digest,
            )
        self.assertEqual(raised.exception.code, "INVALID_REFERENCE_TOKEN_COUNT")
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_candidate_runs(
                [[1, 2], [1, 2]],
                reference_token_count=2,
                reference_token_ids_sha256="sha256:not-a-digest",
            )
        self.assertEqual(raised.exception.code, "INVALID_REFERENCE_TOKEN_DIGEST")

    def test_invalid_token_values_fail_closed(self) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            token_ids_sha256([1, True])
        self.assertEqual(raised.exception.code, "INVALID_TOKEN_RUN")


if __name__ == "__main__":
    unittest.main()
