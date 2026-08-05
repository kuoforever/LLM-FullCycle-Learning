from __future__ import annotations

import copy
import hashlib
import json
import unittest
from typing import Any

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge.tool_router_attached_dtype_isolation_evidence import (
    validate_attached_dtype_isolation_evidence,
)

BF16_PATH = "bf16_attached_adapter"
FP32_PATH = "fp32_attached_adapter"
PATHS = (BF16_PATH, FP32_PATH)
TARGET_STEP = 45
VOCABULARY_SIZE = 151_936
BF16_TOKEN_ID = 1866
FP32_TOKEN_ID = 3849
CLASSIFICATION = (
    "deterministic_bf16_attached_vs_fp32_attached_"
    "raw_logit_boundary_flip"
)

BF16_TOKENS = [
    4913,
    16370,
    22317,
    19895,
    4136,
    3252,
    81251,
    4907,
    10334,
    58640,
    7325,
    5287,
    3252,
    265,
    7606,
    2198,
    41375,
    95613,
    788,
    3849,
    1335,
    80943,
    8274,
    3252,
    26086,
    2198,
    4525,
    22785,
    3252,
    73311,
    2346,
    2895,
    644,
    5047,
    2198,
    5445,
    761,
    3420,
    788,
    1866,
    1335,
    5445,
    1288,
    583,
    788,
    1866,
    92,
    151645,
]
FP32_TOKENS = [*BF16_TOKENS[:TARGET_STEP], FP32_TOKEN_ID, 92, 151645]

SOURCE_LINEAGE = {
    "stability_evidence_sha256": (
        "sha256:82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc"
    ),
    "drift_evidence_sha256": (
        "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3"
    ),
    "isolation_evidence_sha256": (
        "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26"
    ),
    "fp32_numerics_evidence_sha256": (
        "sha256:cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91"
    ),
    "training_evidence_sha256": (
        "sha256:641b1a7ef3dc0de0d9f2124b9122cb2c4be46b42de9265d558ab6f5b25b41a30"
    ),
}

ADAPTER_FILES = [
    {
        "path": "adapter_config.json",
        "bytes": 793,
        "sha256": (
            "sha256:8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16"
        ),
    },
    {
        "path": "adapter_model.safetensors",
        "bytes": 17_462_432,
        "sha256": (
            "sha256:efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342"
        ),
    },
    {
        "path": "README.md",
        "bytes": 5_107,
        "sha256": (
            "sha256:353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522"
        ),
    },
]

ENVIRONMENT = {
    "python": "3.12.12",
    "torch": "2.6.0+cu124",
    "transformers": "4.49.0",
    "peft": "0.14.0",
    "accelerate": "1.3.0",
    "huggingface_hub": "0.29.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
    "gpu_vram_bytes": 17_170_956_288,
    "compute_capability": "8.9",
}

REFERENCE_DIGESTS = {
    BF16_PATH: {
        "token_ids_sha256": (
            "sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173"
        ),
        "output_sha256": (
            "sha256:b3bef0f22aad858ad94275466af0cc082dbb7fe42a7814320aa4a8995dff0bc5"
        ),
        "score_trace_sha256": (
            "sha256:116cc75b8420f5e95188818583e490a689de9392d34438c1d569e66822d7ee49"
        ),
        "raw_logit_trace_sha256": (
            "sha256:a7aab2daf284030ff1eab20b01b77f59064b1fc341ba59765e3d0b957c8174cf"
        ),
        "comparison_score_vector_sha256": (
            "sha256:5b78c36066365bb9c52a4894b6f642006fe891552ebc0d6a294f82aa9a8a80db"
        ),
        "comparison_raw_logit_vector_sha256": (
            "sha256:aa7ae2fab3c2be5b0ddeecb7e4a10d01dcfd8636a6a404d7e48e9ef19eb9bf9e"
        ),
    },
    FP32_PATH: {
        "token_ids_sha256": (
            "sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca"
        ),
        "output_sha256": (
            "sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca"
        ),
        "score_trace_sha256": (
            "sha256:e878f06653e43ebf6946a00396fbed7797eecc02dcf25501f0738169a932fdde"
        ),
        "raw_logit_trace_sha256": (
            "sha256:61a891ab427bce3002c3367e2faefd854a11ecb62929d5b187b974a9c3b7f357"
        ),
        "comparison_score_vector_sha256": (
            "sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451"
        ),
        "comparison_raw_logit_vector_sha256": (
            "sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a"
        ),
    },
}


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def _token_digest(tokens: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in tokens).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tokens(path: str) -> list[int]:
    return list(BF16_TOKENS if path == BF16_PATH else FP32_TOKENS)


def _reference(path: str) -> dict[str, Any]:
    digests = REFERENCE_DIGESTS[path]
    return {
        "path": path,
        "source_experiment_ids": (
            [
                "fc-mvp-001-bf16-merge-stability-v1",
                "fc-mvp-001-fp32-merge-drift-analysis-v1",
            ]
            if path == BF16_PATH
            else [
                "fc-mvp-001-fp32-attached-merge-isolation-v1",
                "fc-mvp-001-fp32-attached-merge-numerics-v1",
            ]
        ),
        "generated_token_ids": _tokens(path),
        "token_count": 48,
        "token_ids_sha256": digests["token_ids_sha256"],
        "output_sha256": digests["output_sha256"],
        "score_trace_sha256": digests["score_trace_sha256"],
        "raw_logit_trace_sha256": digests["raw_logit_trace_sha256"],
        "comparison_step_index": TARGET_STEP,
        "comparison_score_vector_sha256": digests[
            "comparison_score_vector_sha256"
        ],
        "comparison_raw_logit_vector_sha256": digests[
            "comparison_raw_logit_vector_sha256"
        ],
        "boundary_token_id": (
            BF16_TOKEN_ID if path == BF16_PATH else FP32_TOKEN_ID
        ),
    }


def _storage_audit() -> dict[str, Any]:
    return {
        "base_checkpoint": {
            "tensors": 338,
            "elements": 1_543_714_304,
            "dtype_tensors": {"bfloat16": 338},
            "dtype_elements": {"bfloat16": 1_543_714_304},
        },
        "adapter": {
            "tensors": 224,
            "elements": 4_358_144,
            "dtype_tensors": {"float32": 224},
            "dtype_elements": {"float32": 4_358_144},
        },
    }


def _protocol() -> dict[str, Any]:
    run_plan = [
        {
            "run_id": "bf16-attached-dtype-r1",
            "path": BF16_PATH,
            "repeat": 1,
            "order_index": 0,
        },
        {
            "run_id": "fp32-attached-dtype-r1",
            "path": FP32_PATH,
            "repeat": 1,
            "order_index": 1,
        },
        {
            "run_id": "fp32-attached-dtype-r2",
            "path": FP32_PATH,
            "repeat": 2,
            "order_index": 2,
        },
        {
            "run_id": "bf16-attached-dtype-r2",
            "path": BF16_PATH,
            "repeat": 2,
            "order_index": 3,
        },
    ]
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": run_plan,
        "run_order_design": "ABBA",
        "fresh_loads_per_path": {BF16_PATH: 2, FP32_PATH: 2},
        "max_residual_cuda_bytes": 16 * 1024 * 1024,
        "target_forward": {
            "generation_step_index": 45,
            "input_generated_token_index": 44,
            "input_token_id": 788,
            "predicted_token_ids": {
                BF16_PATH: BF16_TOKEN_ID,
                FP32_PATH: FP32_TOKEN_ID,
            },
            "past_length": 383,
            "cache_position": [383],
            "causal_forward_calls": 48,
        },
        "treatment": {
            "isolated_variable": "attached_path_base_and_inference_dtype",
            "bf16_condition": "bfloat16",
            "fp32_condition": "float32",
            "controlled_adapter_runtime_dtype": "float32",
            "attached_execution_form_fixed": True,
        },
        "paths": {
            BF16_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "bfloat16",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": True,
                "merge": False,
                "inference_parameter_dtypes": ["bfloat16", "float32"],
            },
            FP32_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": True,
                "merge": False,
                "inference_parameter_dtypes": ["float32"],
            },
        },
        "generation": {
            "attn_implementation": "sdpa",
            "attention_class": "Qwen2Attention",
            "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
            "low_level_cuda_kernel_identity_claimed": False,
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
            "generate_return_dtype_semantics": (
                "return_tensor_dtype_not_internal_compute_dtype"
            ),
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
    }


def _inventory(dtype: str, tensors: int, elements: int) -> dict[str, Any]:
    return {
        "floating_tensors": tensors,
        "floating_elements": elements,
        "dtypes": {dtype: elements},
        "devices": {"cuda:0": elements},
    }


def _precision(path: str) -> dict[str, Any]:
    base_dtype = "bfloat16" if path == BF16_PATH else "float32"
    return {
        "base_parameters": _inventory(base_dtype, 338, 1_543_714_304),
        "adapter_parameters": _inventory("float32", 224, 4_358_144),
        "floating_buffers": _inventory("float32", 1, 64),
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
        "lora_dropout": {"modules": 112, "training_modules": 0},
        "generation": {
            "score_dtypes": ["float32"],
            "all_scores_float32": True,
            "raw_logit_dtypes": ["float32"],
            "all_raw_logits_float32": True,
            "dtype_semantics": "transformers_generate_return_tensor_dtype",
            "autocast_enabled": False,
            "training": False,
        },
    }


def _trace_vector(path: str, kind: str) -> dict[str, Any]:
    digests = REFERENCE_DIGESTS[path]
    target_key = (
        "comparison_score_vector_sha256"
        if kind == "scores"
        else "comparison_raw_logit_vector_sha256"
    )
    trace_key = "score_trace_sha256" if kind == "scores" else "raw_logit_trace_sha256"
    manifest = [_digest(f"{path}:{kind}:{index}") for index in range(48)]
    manifest[TARGET_STEP] = digests[target_key]
    return {
        "native_dtypes": ["float32"],
        "shape_per_step": [1, VOCABULARY_SIZE],
        "comparison_dtype": "float32",
        "all_finite": True,
        "trace_sha256": digests[trace_key],
        "comparison_vector_sha256_per_step": manifest,
        "comparison_step_index": TARGET_STEP,
        "comparison_step_vector_sha256": digests[target_key],
    }


def _generation_trace(path: str) -> dict[str, Any]:
    raw_digest = REFERENCE_DIGESTS[path]["comparison_raw_logit_vector_sha256"]
    native_dtype = "bfloat16" if path == BF16_PATH else "float32"
    return {
        "step_count": 48,
        "vocabulary_size": VOCABULARY_SIZE,
        "cache_returned": True,
        "scores": _trace_vector(path, "scores"),
        "raw_logits": _trace_vector(path, "raw_logits"),
        "lm_head_output": {
            "native_dtype": native_dtype,
            "shape": [1, 1, VOCABULARY_SIZE],
            "comparison_dtype": "float32",
            "all_finite": True,
            "comparison_step_index": TARGET_STEP,
            "comparison_vector_sha256": raw_digest,
        },
    }


def _target_alignment(path: str) -> dict[str, Any]:
    native_dtype = "bfloat16" if path == BF16_PATH else "float32"
    raw_digest = REFERENCE_DIGESTS[path]["comparison_raw_logit_vector_sha256"]
    return {
        "call_index": TARGET_STEP,
        "generation_step_index": TARGET_STEP,
        "input_token_ids": [788],
        "input_shape": [1, 1],
        "cache_position": [383],
        "position_ids": [383],
        "past_length": 383,
        "causal_forward_calls": 48,
        "lm_head_output_shape": [1, 1, VOCABULARY_SIZE],
        "lm_head_output_native_dtype": native_dtype,
        "lm_head_output_comparison_vector_sha256": raw_digest,
        "generated_raw_logit_comparison_vector_sha256": raw_digest,
    }


def _run(path: str, repeat: int, order_index: int) -> dict[str, Any]:
    prefix = "bf16" if path == BF16_PATH else "fp32"
    tokens = _tokens(path)
    return {
        "run_id": f"{prefix}-attached-dtype-r{repeat}",
        "path": path,
        "repeat": repeat,
        "order_index": order_index,
        "fresh_load": True,
        "base_load_dtype": "bfloat16" if path == BF16_PATH else "float32",
        "generated_token_ids": tokens,
        "token_count": len(tokens),
        "token_ids_sha256": _token_digest(tokens),
        "output_sha256": REFERENCE_DIGESTS[path]["output_sha256"],
        "precision_audit": _precision(path),
        "generation_trace": _generation_trace(path),
        "target_alignment": _target_alignment(path),
        "target_alignment_passed": True,
        "lm_head_raw_logit_linked": True,
        "path_protocol_passed": True,
        "elapsed_seconds": float(order_index + 1),
        "peak_gpu_memory_bytes": 6_000_000_000 + order_index,
        "memory_allocated_before_load_bytes": 0,
        "memory_allocated_after_release_bytes": 0,
    }


def _all_true(keys: tuple[str, ...]) -> dict[str, bool]:
    return {key: True for key in keys}


def _repeat_stability() -> dict[str, Any]:
    keys = (
        "token_identity",
        "output_identity",
        "score_trace_identity",
        "raw_logit_trace_identity",
        "comparison_score_vector_identity",
        "comparison_raw_logit_vector_identity",
        "precision_audit_identity",
        "passed",
        "target_alignment_identity",
        "lm_head_output_identity",
    )
    result: dict[str, Any] = {path: _all_true(keys) for path in PATHS}
    result["passed"] = True
    return result


def _path_reproduction() -> dict[str, Any]:
    keys = (
        "token_identity",
        "output_identity",
        "score_trace_identity",
        "raw_logit_trace_identity",
        "comparison_score_vector_identity",
        "comparison_raw_logit_vector_identity",
        "boundary_token_identity",
        "passed",
    )
    result: dict[str, Any] = {path: _all_true(keys) for path in PATHS}
    result["passed"] = True
    return result


def _step_path(path: str, *, raw: bool) -> dict[str, Any]:
    value_key = "raw_logit" if raw else "score"
    top_values_key = "top_raw_logits" if raw else "top_scores"
    emitted_value_key = (
        "emitted_token_raw_logit" if raw else "emitted_token_score"
    )
    if path == BF16_PATH:
        ids = [BF16_TOKEN_ID, FP32_TOKEN_ID, 100, 101, 102]
        values = [20.0, 17.0, 16.0, 15.0, 14.0] if raw else [10.0, 8.0, 7.0, 6.0, 5.0]
        emitted_id = BF16_TOKEN_ID
        emitted_text = "true"
    else:
        ids = [FP32_TOKEN_ID, BF16_TOKEN_ID, 100, 101, 102]
        values = [22.0, 18.0, 17.0, 16.0, 15.0] if raw else [11.0, 9.0, 8.0, 7.0, 6.0]
        emitted_id = FP32_TOKEN_ID
        emitted_text = "false"
    values_by_id = dict(zip(ids, values))
    digest_key = (
        "comparison_raw_logit_vector_sha256"
        if raw
        else "comparison_score_vector_sha256"
    )
    return {
        "top_token_ids": ids,
        "top_token_texts": [
            "true" if token_id == BF16_TOKEN_ID else "false"
            if token_id == FP32_TOKEN_ID
            else f"token-{token_id}"
            for token_id in ids
        ],
        top_values_key: values,
        "top_margin": values[0] - values[1],
        "emitted_token_id": emitted_id,
        "emitted_token_text": emitted_text,
        emitted_value_key: values[0],
        "compared_tokens": [
            {
                "token_id": BF16_TOKEN_ID,
                "token_text": "true",
                value_key: values_by_id[BF16_TOKEN_ID],
                "rank": ids.index(BF16_TOKEN_ID) + 1,
            },
            {
                "token_id": FP32_TOKEN_ID,
                "token_text": "false",
                value_key: values_by_id[FP32_TOKEN_ID],
                "rank": ids.index(FP32_TOKEN_ID) + 1,
            },
        ],
        "decision_contrast_true_minus_false": (
            values_by_id[BF16_TOKEN_ID] - values_by_id[FP32_TOKEN_ID]
        ),
        "comparison_vector_sha256": REFERENCE_DIGESTS[path][digest_key],
    }


def _step_evidence(*, raw: bool) -> dict[str, Any]:
    return {
        "step_index": TARGET_STEP,
        "comparison_basis": (
            "frozen_first_cross_dtype_generated_token_divergence"
        ),
        "shared_generated_prefix_tokens_before_step": TARGET_STEP,
        "source": "generated.logits" if raw else "generated.scores",
        "semantics": (
            "unprocessed_lm_head_prediction_scores"
            if raw
            else "processed_prediction_scores_after_logits_processors"
        ),
        "comparison_dtype": "float32",
        "top_k": 5,
        "paths": {path: _step_path(path, raw=raw) for path in PATHS},
        "delta": {
            "vocabulary_elements": VOCABULARY_SIZE,
            "nonzero_elements": VOCABULARY_SIZE,
            "max_abs_delta": 10.0 if raw else 5.0,
            "mean_abs_delta": 1.0,
            "root_mean_square_delta": 2.0,
        },
    }


def _causal_scope() -> dict[str, Any]:
    return {
        "isolated_variable": (
            "attached_path_base_and_inference_dtype_bfloat16_vs_float32"
        ),
        "controlled": [
            "same_bfloat16_checkpoint_source_values",
            "same_fp32_adapter_source_and_runtime_values",
            "same_attached_factorized_lora_execution_form",
            "same_eval_001_rendered_input_and_generation_prefix",
            "same_greedy_decoding_and_high_level_sdpa_dispatch",
            "same_fresh_model_load_lifecycle",
        ],
        "supports": (
            "classification of the repeat-stable total dtype effect on the "
            "frozen attached generation path at token boundary 45"
        ),
        "does_not_support": [
            "all_bf16_versus_all_fp32_path_claim",
            "pristine_fp32_checkpoint_comparison",
            "earliest_temporal_or_module_root_cause",
            "low_level_cuda_kernel_identity_or_unique_root_cause",
            "peft_bug_claim",
            "full_eval_generalization",
            "merged_artifact_promotion",
            "runtime_eligibility",
        ],
    }


def _constraints() -> dict[str, bool]:
    return {
        "attached_execution_form_change": False,
        "adapter_runtime_dtype_change": False,
        "source_checkpoint_values_change": False,
        "target_step_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_save": False,
        "merged_artifact_promotion": False,
        "module_tensor_sidecar": False,
    }


def valid_evidence() -> dict[str, Any]:
    runs = [
        _run(BF16_PATH, 1, 0),
        _run(FP32_PATH, 1, 1),
        _run(FP32_PATH, 2, 2),
        _run(BF16_PATH, 2, 3),
    ]
    stability = _repeat_stability()
    reproduction = _path_reproduction()
    gate = _all_true(
        (
            "bf16_attached_repeat_stable",
            "fp32_attached_repeat_stable",
            "bf16_frozen_reference_reproduced",
            "fp32_frozen_reference_reproduced",
            "first_token_boundary_reproduced",
            "target_forward_aligned",
            "lm_head_raw_logit_linked",
            "processed_score_argmax_matches_emitted_token",
            "raw_logits_captured",
            "attached_dtype_effect_classified",
            "passed",
        )
    )
    acceptance = _all_true(
        (
            "upstream_evidence_locked",
            "frozen_input_reproduced",
            "attached_execution_form_fixed",
            "base_dtype_only_treatment",
            "bf16_attached_repeat_stable",
            "fp32_attached_repeat_stable",
            "bf16_frozen_reference_reproduced",
            "fp32_frozen_reference_reproduced",
            "first_token_boundary_reproduced",
            "target_forward_aligned",
            "full_generation_traces_captured",
            "lm_head_raw_logit_linked",
            "generation_score_alignment_verified",
            "path_protocols_executed",
            "source_storage_dtypes_locked",
            "fresh_load_memory_isolated",
            "source_adapter_unchanged",
            "source_model_unchanged",
            "eval_digest_unchanged",
            "prompt_digest_unchanged",
        )
    )
    constraints = _constraints()
    return {
        "attached_dtype_isolation_version": 1,
        "experiment_id": "fc-mvp-001-attached-dtype-isolation-v1",
        "source_experiment_id": "fc-mvp-001-lora-sft-v2",
        "source_lineage": copy.deepcopy(SOURCE_LINEAGE),
        "training_lock_sha256": (
            "sha256:e6e23f51834b1815578368ce54c78034e72a7158395892e77fdf75594548931f"
        ),
        "config_sha256": (
            "sha256:5a038ea786526f188c796a6e5eea4c4d3aa47fc66977dc4f6ff16f52999236d8"
        ),
        "adapter_files": copy.deepcopy(ADAPTER_FILES),
        "model_weight_sha256": (
            "sha256:dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        ),
        "prompt_sha256": (
            "sha256:4a7d15063b0b074ef999c2848d0fc073a6cc00ed4999ea81f770e2e42cfa6d97"
        ),
        "eval_digest": (
            "sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a"
        ),
        "example_id": "eval-001",
        "input_token_count": 339,
        "input_token_ids_sha256": (
            "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
        ),
        "storage_audit": _storage_audit(),
        "environment": copy.deepcopy(ENVIRONMENT),
        "protocol": _protocol(),
        "frozen_path_references": {path: _reference(path) for path in PATHS},
        "runs": runs,
        "path_repeat_stability": stability,
        "path_reproduction": reproduction,
        "cross_dtype_token_analysis": {
            "bf16_token_count": 48,
            "fp32_token_count": 48,
            "cross_dtype_identical": False,
            "common_prefix_generated_tokens": TARGET_STEP,
            "first_divergent_token_index": TARGET_STEP,
            "bf16_token_id": BF16_TOKEN_ID,
            "fp32_token_id": FP32_TOKEN_ID,
            "classification": "cross_dtype_token_drift",
            "bf16_token_text": "true",
            "fp32_token_text": "false",
        },
        "comparison_step": {
            "step_index": TARGET_STEP,
            "basis": "frozen_first_cross_dtype_generated_token_divergence",
            "shared_generated_prefix_tokens": TARGET_STEP,
        },
        "selection_score_evidence": _step_evidence(raw=False),
        "raw_logit_evidence": _step_evidence(raw=True),
        "cross_dtype_trace_identity": {
            "token_identity": False,
            "output_identity": False,
            "score_trace_identity": False,
            "raw_logit_trace_identity": False,
            "comparison_score_vector_identity": False,
            "comparison_raw_logit_vector_identity": False,
        },
        "classification": CLASSIFICATION,
        "causal_scope": _causal_scope(),
        "dtype_isolation_gate": gate,
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "acceptance": acceptance,
        "elapsed_seconds": 10.5,
        "peak_gpu_memory_bytes": 6_000_000_003,
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": {
            "gate_id": "FC-MVP-001-attached-dtype-numerics-v1",
            "action": (
                "on the same frozen target forward, locate the first registered "
                "BF16-versus-FP32 attached module-output difference and quantify "
                "its propagation without changing execution form or claiming a "
                "unique low-level root cause"
            ),
            "acceptance": {
                "fresh_attached_paths_reproduced": True,
                "target_forward_reproduced": True,
                "first_registered_module_difference_located": True,
                "dtype_effect_quantified": True,
                "source_inputs_unchanged": True,
            },
            "constraints": constraints,
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": CLASSIFICATION,
        "offline": True,
    }


def _step_manifest_locks(
    evidence: dict[str, Any],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in PATHS:
        run = next(item for item in evidence["runs"] if item["path"] == path)
        result[path] = {}
        for kind in ("scores", "raw_logits"):
            manifest = run["generation_trace"][kind][
                "comparison_vector_sha256_per_step"
            ]
            payload = json.dumps(
                manifest,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            result[path][kind] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return result


class AttachedDtypeIsolationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = valid_evidence()
        self.manifest_locks = _step_manifest_locks(self.evidence)

    def assert_rejected(
        self,
        evidence: dict[str, Any],
        *,
        code: str,
        path: str,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "expected_step_manifest_sha256",
            self.manifest_locks,
        )
        with self.assertRaises(ToolRouterValidationError) as raised:
            validate_attached_dtype_isolation_evidence(evidence, **kwargs)
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(raised.exception.path, path)

    def test_valid_cpu_only_synthetic_evidence(self) -> None:
        json_round_trip = json.loads(json.dumps(self.evidence, allow_nan=False))
        self.assertEqual(
            validate_attached_dtype_isolation_evidence(
                json_round_trip,
                expected_step_manifest_sha256=self.manifest_locks,
            ),
            {
                "frozen_gate_valid": True,
                "runs_validated": 4,
                "token_digests_recomputed": 6,
                "comparison_step_manifests_validated": 8,
                "classification": CLASSIFICATION,
                "delta_statistics_scope": "probe_derived_summary_algebra_only",
            },
        )

    def test_rejects_top_level_schema_and_source_mutations(self) -> None:
        extra = copy.deepcopy(self.evidence)
        extra["unreviewed_claim"] = True
        self.assert_rejected(extra, code="INVALID_EVIDENCE_SCHEMA", path="$")

        lineage = copy.deepcopy(self.evidence)
        lineage["source_lineage"]["drift_evidence_sha256"] = _digest("forged")
        self.assert_rejected(
            lineage,
            code="INVALID_SOURCE_LINEAGE",
            path="$.source_lineage",
        )

        expected = copy.deepcopy(SOURCE_LINEAGE)
        expected["training_evidence_sha256"] = _digest("wrong-expected-source")
        self.assert_rejected(
            self.evidence,
            code="INVALID_SOURCE_LINEAGE",
            path="$.source_lineage",
            expected_source_lineage=expected,
        )

    def test_rejects_adapter_and_environment_pin_mutations(self) -> None:
        adapter = copy.deepcopy(self.evidence)
        adapter["adapter_files"][1]["bytes"] += 1
        self.assert_rejected(
            adapter,
            code="INVALID_ADAPTER_MANIFEST",
            path="$.adapter_files",
        )

        environment = copy.deepcopy(self.evidence)
        environment["environment"]["torch"] = "2.7.0"
        self.assert_rejected(
            environment,
            code="INVALID_ENVIRONMENT",
            path="$.environment",
        )

    def test_rejects_abba_reorder_duplicate_path_and_protocol_drift(self) -> None:
        reordered = copy.deepcopy(self.evidence)
        reordered["runs"][0], reordered["runs"][1] = (
            reordered["runs"][1],
            reordered["runs"][0],
        )
        self.assert_rejected(
            reordered,
            code="INVALID_RUN_PLAN",
            path="$.runs[0].run_id",
        )

        duplicate_path = copy.deepcopy(self.evidence)
        duplicate_path["runs"][1]["path"] = BF16_PATH
        self.assert_rejected(
            duplicate_path,
            code="INVALID_RUN_PLAN",
            path="$.runs[1].path",
        )

        protocol = copy.deepcopy(self.evidence)
        protocol["protocol"]["run_plan"][0]["order_index"] = 1
        self.assert_rejected(protocol, code="INVALID_PROTOCOL", path="$.protocol")

    def test_rejects_bool_as_protocol_or_resource_integer(self) -> None:
        protocol = copy.deepcopy(self.evidence)
        protocol["protocol"]["fresh_loads_per_path"][BF16_PATH] = True
        self.assert_rejected(protocol, code="INVALID_PROTOCOL", path="$.protocol")

        resource = copy.deepcopy(self.evidence)
        resource["runs"][0]["peak_gpu_memory_bytes"] = True
        self.assert_rejected(
            resource,
            code="INVALID_RUN_RESOURCE",
            path="$.runs[0].peak_gpu_memory_bytes",
        )

    def test_recomputes_run_token_digest_and_boundary_summary(self) -> None:
        digest = copy.deepcopy(self.evidence)
        digest["runs"][0]["generated_token_ids"][0] += 1
        self.assert_rejected(
            digest,
            code="INVALID_TOKEN_DIGEST",
            path="$.runs[0].token_ids_sha256",
        )

        boundary = copy.deepcopy(self.evidence)
        boundary["cross_dtype_token_analysis"]["first_divergent_token_index"] = 44
        self.assert_rejected(
            boundary,
            code="TOKEN_ANALYSIS_MISMATCH",
            path="$.cross_dtype_token_analysis",
        )

    def test_rejects_step_manifest_and_target_vector_link_mutations(self) -> None:
        short_manifest = copy.deepcopy(self.evidence)
        short_manifest["runs"][0]["generation_trace"]["scores"][
            "comparison_vector_sha256_per_step"
        ].pop()
        self.assert_rejected(
            short_manifest,
            code="INVALID_STEP_MANIFEST",
            path=(
                "$.runs[0].generation_trace.scores."
                "comparison_vector_sha256_per_step"
            ),
        )

        unlinked = copy.deepcopy(self.evidence)
        unlinked["runs"][0]["generation_trace"]["scores"][
            "comparison_vector_sha256_per_step"
        ][TARGET_STEP] = _digest("unlinked-target")
        self.assert_rejected(
            unlinked,
            code="STEP_VECTOR_LINK_MISMATCH",
            path=(
                "$.runs[0].generation_trace.scores."
                "comparison_step_vector_sha256"
            ),
        )

        paired_forgery = copy.deepcopy(self.evidence)
        for run in paired_forgery["runs"]:
            if run["path"] == BF16_PATH:
                run["generation_trace"]["scores"][
                    "comparison_vector_sha256_per_step"
                ][0] = _digest("paired-manifest-forgery")
        self.assert_rejected(
            paired_forgery,
            code="STEP_MANIFEST_LOCK_MISMATCH",
            path=(
                "$.runs[0].generation_trace.scores."
                "comparison_vector_sha256_per_step"
            ),
        )

    def test_rejects_lm_head_raw_logit_and_target_alignment_mutations(self) -> None:
        lm_head = copy.deepcopy(self.evidence)
        lm_head["runs"][0]["generation_trace"]["lm_head_output"][
            "comparison_vector_sha256"
        ] = _digest("forged-lm-head")
        self.assert_rejected(
            lm_head,
            code="LM_HEAD_RAW_LOGIT_LINK_MISMATCH",
            path="$.runs[0].generation_trace.lm_head_output",
        )

        target = copy.deepcopy(self.evidence)
        target["runs"][1]["target_alignment"]["past_length"] = 382
        self.assert_rejected(
            target,
            code="INVALID_TARGET_ALIGNMENT",
            path="$.runs[1].target_alignment",
        )

    def test_rejects_precision_inventory_and_generation_semantics_mutations(
        self,
    ) -> None:
        precision = copy.deepcopy(self.evidence)
        precision["runs"][0]["precision_audit"]["base_parameters"]["dtypes"] = {
            "float32": 1_543_714_304
        }
        self.assert_rejected(
            precision,
            code="INVALID_PRECISION_AUDIT",
            path="$.runs[0].precision_audit",
        )

        generation = copy.deepcopy(self.evidence)
        generation["runs"][0]["precision_audit"]["generation"][
            "autocast_enabled"
        ] = True
        self.assert_rejected(
            generation,
            code="INVALID_PRECISION_AUDIT",
            path="$.runs[0].precision_audit",
        )

    def test_recomputes_top_k_margin_and_decision_contrast(self) -> None:
        margin = copy.deepcopy(self.evidence)
        margin["selection_score_evidence"]["paths"][BF16_PATH][
            "top_margin"
        ] += 0.5
        self.assert_rejected(
            margin,
            code="INVALID_TOP_K_EVIDENCE",
            path=(
                "$.selection_score_evidence.paths."
                f"{BF16_PATH}.top_margin"
            ),
        )

        contrast = copy.deepcopy(self.evidence)
        contrast["raw_logit_evidence"]["paths"][FP32_PATH][
            "decision_contrast_true_minus_false"
        ] = 1.0
        self.assert_rejected(
            contrast,
            code="INVALID_TOP_K_EVIDENCE",
            path=(
                "$.raw_logit_evidence.paths."
                f"{FP32_PATH}.decision_contrast_true_minus_false"
            ),
        )

    def test_rejects_impossible_delta_algebra(self) -> None:
        delta = copy.deepcopy(self.evidence)
        delta["raw_logit_evidence"]["delta"]["root_mean_square_delta"] = 11.0
        self.assert_rejected(
            delta,
            code="INVALID_DELTA_SUMMARY",
            path="$.raw_logit_evidence.delta",
        )

        exposed = copy.deepcopy(self.evidence)
        exposed["selection_score_evidence"]["delta"]["max_abs_delta"] = 2.0
        self.assert_rejected(
            exposed,
            code="INVALID_DELTA_SUMMARY",
            path="$.selection_score_evidence.delta",
        )

    def test_rejects_forged_repeat_and_classification_summaries(self) -> None:
        repeat = copy.deepcopy(self.evidence)
        repeat["path_repeat_stability"][BF16_PATH]["passed"] = False
        self.assert_rejected(
            repeat,
            code="REPEAT_STABILITY_MISMATCH",
            path="$.path_repeat_stability",
        )

        classification = copy.deepcopy(self.evidence)
        classification["classification"] = (
            "deterministic_bf16_attached_vs_fp32_attached_"
            "logits_processor_boundary_flip"
        )
        self.assert_rejected(
            classification,
            code="CLASSIFICATION_MISMATCH",
            path="$.classification",
        )

    def test_rejects_forged_gate_and_acceptance_claims(self) -> None:
        gate = copy.deepcopy(self.evidence)
        gate["dtype_isolation_gate"]["target_forward_aligned"] = False
        self.assert_rejected(
            gate,
            code="GATE_CLAIM_MISMATCH",
            path="$.dtype_isolation_gate",
        )

        acceptance = copy.deepcopy(self.evidence)
        acceptance["acceptance"]["source_model_unchanged"] = False
        self.assert_rejected(
            acceptance,
            code="ACCEPTANCE_CLAIM_MISMATCH",
            path="$.acceptance",
        )

    def test_rejects_resource_and_policy_promotion_mutations(self) -> None:
        resource = copy.deepcopy(self.evidence)
        resource["peak_gpu_memory_bytes"] -= 1
        self.assert_rejected(
            resource,
            code="FROZEN_RESOURCE_CLAIM_MISMATCH",
            path="$",
        )

        for key in (
            "merged_artifact_saved",
            "merged_artifact_allowed",
            "runtime_eligible",
        ):
            with self.subTest(key=key):
                policy = copy.deepcopy(self.evidence)
                policy[key] = True
                self.assert_rejected(
                    policy,
                    code="INVALID_FROZEN_POLICY",
                    path=f"$.{key}",
                )


if __name__ == "__main__":
    unittest.main()
