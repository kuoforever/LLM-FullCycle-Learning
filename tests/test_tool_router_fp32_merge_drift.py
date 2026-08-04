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
from fullcycle_bridge.tool_router_fp32_merge_drift import (
    analyze_path_tokens,
    classify_generation_boundary,
)
from fullcycle_bridge.tool_router_merge_remediation import token_ids_sha256
from fullcycle_bridge.tool_router_sft import (
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "baseline" / "fc-mvp-001-fp32-merge-drift-analysis-v1.json"
STABILITY = ROOT / "baseline" / "fc-mvp-001-bf16-merge-stability-v1.json"
NUMERICS = ROOT / "baseline" / "fc-mvp-001-bf16-merge-numerics-v1.json"
REMEDIATION = ROOT / "baseline" / "fc-mvp-001-bf16-merge-remediation-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _inventory(dtype: str, elements: int, tensors: int) -> dict[str, Any]:
    return {
        "floating_tensors": tensors,
        "floating_elements": elements,
        "dtypes": {dtype: elements},
        "devices": {"cuda:0": elements},
    }


class ToolRouterFp32MergeDriftTests(unittest.TestCase):
    def test_frozen_exact_step_drift_evidence_is_closed_and_recomputable(
        self,
    ) -> None:
        gate = _load(GATE)
        config = _load(ROOT / "configs" / "tool_router_lora_sft_v2.json")
        training = _load(
            ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
        )
        stability = _load(STABILITY)
        remediation = _load(REMEDIATION)
        adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
        evaluation = load_fixture(ROOT / config["data"]["eval_path"])

        self.assertEqual(
            file_sha256(GATE),
            "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3",
        )
        self.assertEqual(
            set(gate),
            {
                "fp32_merge_drift_analysis_version",
                "experiment_id",
                "source_experiment_id",
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
                "frozen_references",
                "runs",
                "reproduction",
                "token_analysis",
                "selection_score_evidence",
                "raw_logit_evidence",
                "classification",
                "analysis_gate",
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
        self.assertEqual(gate["fp32_merge_drift_analysis_version"], 1)
        self.assertEqual(
            gate["experiment_id"],
            "fc-mvp-001-fp32-merge-drift-analysis-v1",
        )
        self.assertEqual(gate["source_experiment_id"], config["experiment_id"])
        self.assertEqual(gate["config_sha256"], canonical_config_sha256(config))
        self.assertEqual(gate["adapter_files"], directory_artifact_manifest(adapter))
        self.assertEqual(gate["adapter_files"], training["final_adapter"]["files"])
        self.assertEqual(gate["stability_evidence_sha256"], file_sha256(STABILITY))
        self.assertEqual(gate["numerics_evidence_sha256"], file_sha256(NUMERICS))
        self.assertEqual(
            gate["remediation_evidence_sha256"], file_sha256(REMEDIATION)
        )
        self.assertEqual(
            gate["training_lock_sha256"],
            file_sha256(ROOT / "requirements" / "training.lock"),
        )
        self.assertEqual(
            gate["model_weight_sha256"],
            f"sha256:{config['model']['weight_sha256']}",
        )
        self.assertEqual(
            gate["prompt_sha256"],
            file_sha256(ROOT / config["prompt"]["path"]),
        )
        self.assertEqual(gate["eval_digest"], fixture_digest(evaluation))
        self.assertEqual(gate["example_id"], "eval-001")
        self.assertEqual(gate["input_token_count"], 339)
        self.assertEqual(gate["input_token_count"], stability["input_token_count"])
        self.assertEqual(
            gate["input_token_ids_sha256"],
            "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f",
        )
        self.assertEqual(
            gate["input_token_ids_sha256"], stability["input_token_ids_sha256"]
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
        self.assertEqual(
            gate["protocol"],
            {
                "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
                "path_order": ["independent_bf16_adapter", "fp32_safe_merged"],
                "fresh_loads_per_path": 1,
                "max_residual_cuda_bytes": 16777216,
                "paths": {
                    "independent_bf16_adapter": {
                        "checkpoint_storage_dtype": "bfloat16",
                        "base_load_dtype": "bfloat16",
                        "adapter_storage_dtype": "float32",
                        "adapter_runtime_dtype": "float32",
                        "merge": False,
                        "inference_parameter_dtypes": ["bfloat16", "float32"],
                    },
                    "fp32_safe_merged": {
                        "checkpoint_storage_dtype": "bfloat16",
                        "base_load_dtype": "float32",
                        "adapter_storage_dtype": "float32",
                        "adapter_runtime_dtype": "float32",
                        "merge_dtype": "float32",
                        "inference_dtype": "float32",
                        "merge": True,
                        "safe_merge": True,
                        "adapter_names": ["default"],
                    },
                },
                "generation": {
                    "attn_implementation": "sdpa",
                    "attention_class": "Qwen2Attention",
                    "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
                    "output_attentions": False,
                    "do_sample": False,
                    "max_new_tokens": 256,
                    "use_cache": True,
                    "repetition_penalty": 1.1,
                    "model_eos_token_ids": [151645, 151643],
                    "model_pad_token_id": 151643,
                    "call_pad_token_id": 151645,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "output_logits": True,
                    "tf32": False,
                    "autocast": False,
                    "device": "cuda:0",
                },
                "sdp_kernel_flags": {
                    "flash_sdp_enabled": True,
                    "math_sdp_enabled": True,
                    "mem_efficient_sdp_enabled": True,
                    "cudnn_sdp_enabled": True,
                    "fp16_bf16_reduction_math_sdp_allowed": False,
                },
            },
        )

        references = gate["frozen_references"]
        self.assertEqual(
            references["independent_bf16_adapter"], remediation["reference"]
        )
        self.assertEqual(
            references["fp32_safe_merged"]["token_ids_sha256"],
            remediation["runs"][0]["token_ids_sha256"],
        )
        self.assertEqual(
            references["fp32_safe_merged"]["output_sha256"],
            remediation["runs"][0]["output_sha256"],
        )
        runs = gate["runs"]
        self.assertEqual(
            [run["path"] for run in runs],
            ["independent_bf16_adapter", "fp32_safe_merged"],
        )
        self.assertEqual([run["fresh_load"] for run in runs], [1, 1])
        for run in runs:
            reference = references[run["path"]]
            self.assertEqual(run["token_count"], 48)
            self.assertEqual(len(run["generated_token_ids"]), 48)
            self.assertEqual(
                run["token_ids_sha256"],
                token_ids_sha256(run["generated_token_ids"]),
            )
            self.assertEqual(run["token_ids_sha256"], reference["token_ids_sha256"])
            self.assertEqual(run["output_sha256"], reference["output_sha256"])
            self.assertTrue(run["path_protocol_passed"])
            self.assertLessEqual(
                run["memory_allocated_before_load_bytes"],
                gate["protocol"]["max_residual_cuda_bytes"],
            )
            self.assertLessEqual(
                run["memory_allocated_after_release_bytes"],
                gate["protocol"]["max_residual_cuda_bytes"],
            )
            trace = run["generation_trace"]
            self.assertEqual(trace["step_count"], run["token_count"])
            self.assertEqual(trace["vocabulary_size"], 151936)
            self.assertTrue(trace["cache_returned"])
            for trace_key, evidence_key in (
                ("scores", "selection_score_evidence"),
                ("raw_logits", "raw_logit_evidence"),
            ):
                summary = trace[trace_key]
                self.assertEqual(summary["native_dtypes"], ["float32"])
                self.assertEqual(summary["shape_per_step"], [1, 151936])
                self.assertEqual(summary["comparison_dtype"], "float32")
                self.assertTrue(summary["all_finite"])
                self.assertEqual(summary["divergent_step_index"], 45)
                self.assertEqual(
                    summary["divergent_step_comparison_vector_sha256"],
                    gate[evidence_key]["paths"][run["path"]][
                        "comparison_vector_sha256"
                    ],
                )

        independent_precision = runs[0]["precision_audit"]
        self.assertEqual(
            independent_precision["base_parameters"],
            _inventory("bfloat16", 1543714304, 338),
        )
        self.assertEqual(
            independent_precision["adapter_parameters"],
            _inventory("float32", 4358144, 224),
        )
        self.assertEqual(
            independent_precision["floating_buffers"],
            _inventory("float32", 64, 1),
        )
        self.assertEqual(independent_precision["lora_target_modules"], 112)
        self.assertEqual(independent_precision["lora_parameter_tensors"], 224)
        candidate_precision = runs[1]["precision_audit"]
        self.assertEqual(
            candidate_precision["pre_merge"]["base_parameters"],
            _inventory("float32", 1543714304, 338),
        )
        self.assertEqual(
            candidate_precision["pre_merge"]["adapter_parameters"],
            _inventory("float32", 4358144, 224),
        )
        self.assertEqual(
            candidate_precision["post_merge"]["parameters"],
            _inventory("float32", 1543714304, 338),
        )
        self.assertEqual(candidate_precision["post_merge"]["lora_target_modules"], 0)
        for precision in (independent_precision, candidate_precision):
            self.assertEqual(
                precision["generation"],
                {
                    "score_dtypes": ["float32"],
                    "all_scores_float32": True,
                    "raw_logit_dtypes": ["float32"],
                    "all_raw_logits_float32": True,
                    "autocast_enabled": False,
                    "training": False,
                },
            )

        recomputed = analyze_path_tokens(
            runs[0]["generated_token_ids"],
            runs[1]["generated_token_ids"],
        )
        for key, value in recomputed.items():
            self.assertEqual(gate["token_analysis"][key], value)
        self.assertEqual(gate["token_analysis"]["first_divergent_token_index"], 45)
        self.assertEqual(gate["token_analysis"]["independent_token_id"], 1866)
        self.assertEqual(gate["token_analysis"]["candidate_token_id"], 3849)

        scores = gate["selection_score_evidence"]
        raw_logits = gate["raw_logit_evidence"]
        self.assertEqual(scores["source"], "generated.scores")
        self.assertEqual(raw_logits["source"], "generated.logits")
        for evidence in (scores, raw_logits):
            self.assertEqual(evidence["step_index"], 45)
            self.assertEqual(evidence["common_prefix_generated_tokens"], 45)
            self.assertEqual(evidence["comparison_dtype"], "float32")
            self.assertEqual(evidence["top_k"], 5)
            for run in runs:
                path = evidence["paths"][run["path"]]
                values = path[
                    "top_scores" if evidence is scores else "top_raw_logits"
                ]
                self.assertEqual(values, sorted(values, reverse=True))
                self.assertAlmostEqual(path["top_margin"], values[0] - values[1])
                self.assertEqual(
                    path["top_token_ids"][0],
                    run["generated_token_ids"][45],
                )
        self.assertEqual(
            scores["paths"]["independent_bf16_adapter"]["top_token_ids"][:2],
            [1866, 3849],
        )
        self.assertEqual(
            scores["paths"]["fp32_safe_merged"]["top_token_ids"][:2],
            [3849, 1866],
        )
        self.assertEqual(
            raw_logits["paths"]["independent_bf16_adapter"]["top_raw_logits"][
                :2
            ],
            [38.0, 37.5],
        )
        self.assertEqual(
            raw_logits["paths"]["fp32_safe_merged"]["top_raw_logits"][:2],
            [39.17210388183594, 36.48637390136719],
        )
        self.assertEqual(
            raw_logits["delta"],
            {
                "vocabulary_elements": 151936,
                "nonzero_elements": 151936,
                "max_abs_delta": 1.9437971115112305,
                "mean_abs_delta": 0.22757971286773682,
                "root_mean_square_delta": 0.2932598292827606,
            },
        )
        self.assertEqual(
            gate["classification"],
            "deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip",
        )
        self.assertTrue(all(gate["reproduction"].values()))
        self.assertTrue(gate["analysis_gate"]["passed"])
        self.assertTrue(
            all(
                value is True
                for key, value in gate["analysis_gate"].items()
                if key != "passed"
            )
        )
        self.assertEqual(
            gate["remediation_gate"],
            {
                "candidate_failure_reproduced": True,
                "independent_bf16_reference_identity": False,
                "passed": False,
            },
        )
        self.assertTrue(all(gate["acceptance"].values()))
        self.assertEqual(
            gate["peak_gpu_memory_bytes"],
            max(run["peak_gpu_memory_bytes"] for run in runs),
        )
        self.assertGreater(gate["elapsed_seconds"], 0)
        self.assertFalse(gate["merged_artifact_saved"])
        self.assertFalse(gate["merged_artifact_allowed"])
        self.assertFalse(gate["runtime_eligible"])
        self.assertEqual(
            gate["runtime_eligibility_reason"], gate["classification"]
        )
        self.assertTrue(gate["offline"])
        self.assertEqual(
            gate["locked_next_action"]["gate_id"],
            "FC-MVP-001-fp32-attached-merge-isolation-v1",
        )
        self.assertEqual(
            gate["locked_next_action"]["acceptance"],
            {
                "attached_fp32_repeat_stable": True,
                "fp32_candidate_reproduced": True,
                "same_dtype_exact_step_compared": True,
                "same_dtype_attached_vs_merged_effect_classified": True,
                "source_inputs_unchanged": True,
            },
        )
        self.assertEqual(
            gate["constraints"], gate["locked_next_action"]["constraints"]
        )

    def test_first_cross_path_token_drift_is_located(self) -> None:
        analysis = analyze_path_tokens([10, 20, 30], [10, 21, 30])
        self.assertFalse(analysis["cross_path_identical"])
        self.assertEqual(analysis["common_prefix_generated_tokens"], 1)
        self.assertEqual(analysis["first_divergent_token_index"], 1)
        self.assertEqual(analysis["independent_token_id"], 20)
        self.assertEqual(analysis["candidate_token_id"], 21)
        self.assertEqual(analysis["classification"], "cross_path_token_drift")

    def test_raw_logit_argmax_flip_is_classified(self) -> None:
        analysis = analyze_path_tokens([10, 20], [10, 21])
        self.assertEqual(
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=True,
                independent_score_top_token_id=20,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=20,
                candidate_raw_logit_top_token_id=21,
            ),
            "deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip",
        )

    def test_shared_raw_argmax_is_a_logits_processor_boundary_flip(self) -> None:
        analysis = analyze_path_tokens([10, 20], [10, 21])
        self.assertEqual(
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=True,
                independent_score_top_token_id=20,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=30,
                candidate_raw_logit_top_token_id=30,
            ),
            (
                "deterministic_bf16_attached_vs_fp32_merged_"
                "logits_processor_boundary_flip"
            ),
        )

    def test_distinct_nonselected_raw_argmax_is_mixed_drift(self) -> None:
        analysis = analyze_path_tokens([10, 20], [10, 21])
        self.assertEqual(
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=True,
                independent_score_top_token_id=20,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=30,
                candidate_raw_logit_top_token_id=31,
            ),
            (
                "deterministic_bf16_attached_vs_fp32_merged_"
                "mixed_logit_score_drift"
            ),
        )

    def test_processed_score_alignment_failure_is_not_a_boundary_claim(self) -> None:
        analysis = analyze_path_tokens([10, 20], [10, 21])
        self.assertEqual(
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=True,
                independent_score_top_token_id=21,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=20,
                candidate_raw_logit_top_token_id=21,
            ),
            "generation_score_alignment_failure",
        )

    def test_identity_and_termination_drift_are_distinguished(self) -> None:
        identity = analyze_path_tokens([1, 2], [1, 2])
        self.assertEqual(identity["classification"], "cross_path_output_identity")
        termination = analyze_path_tokens([1, 2], [1, 2, 3])
        self.assertEqual(
            termination["classification"], "cross_path_termination_drift"
        )
        self.assertEqual(termination["first_divergent_token_index"], 2)
        self.assertIsNone(termination["independent_token_id"])

    def test_reproduction_and_input_validation_fail_closed(self) -> None:
        analysis = analyze_path_tokens([10, 20], [10, 21])
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=False,
                independent_score_top_token_id=20,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=20,
                candidate_raw_logit_top_token_id=21,
            )
        self.assertEqual(
            raised.exception.code, "FROZEN_PATH_REPRODUCTION_NOT_ESTABLISHED"
        )
        with self.assertRaises(ToolRouterValidationError) as raised:
            analyze_path_tokens([1, True], [1, 2])
        self.assertEqual(raised.exception.code, "INVALID_TOKEN_RUN")
        with self.assertRaises(ToolRouterValidationError) as raised:
            classify_generation_boundary(
                analysis,
                frozen_paths_reproduced=True,
                independent_score_top_token_id=-1,
                candidate_score_top_token_id=21,
                independent_raw_logit_top_token_id=20,
                candidate_raw_logit_top_token_id=21,
            )
        self.assertEqual(raised.exception.code, "INVALID_TOKEN_ID")


if __name__ == "__main__":
    unittest.main()
