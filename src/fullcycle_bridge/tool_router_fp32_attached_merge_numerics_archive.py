"""Stdlib-only validation of raw FP32 attached-versus-merged evidence."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from collections.abc import Mapping
from typing import Any, NoReturn, cast

from .tool_router import ToolRouterValidationError
from .tool_router_fp32_attached_merge_numerics import (
    COMMON_OUTPUT_STAGES,
    FIRST_REGISTERED_BOUNDARY,
    analyze_module_comparisons,
    classify_operation_order,
)
from .tool_router_merge_remediation import token_ids_sha256

ARCHIVE_BASENAME = (
    "fc-mvp-001-fp32-attached-merge-numerics-v1-tensors.bin"
)
ATTACHED_PATH = "fp32_attached_adapter"
MERGED_PATH = "fp32_safe_merged"
ATTACHED_R1 = "fp32-attached-numerics-r1"
MERGED_R1 = "fp32-safe-merged-numerics-r1"
RUN_PLAN = (
    (ATTACHED_R1, ATTACHED_PATH, 1, 0),
    (MERGED_R1, MERGED_PATH, 1, 1),
    ("fp32-safe-merged-numerics-r2", MERGED_PATH, 2, 2),
    ("fp32-attached-numerics-r2", ATTACHED_PATH, 2, 3),
)
RUN_PATHS = {run_id: path for run_id, path, _repeat, _order in RUN_PLAN}
TARGET_STEP_INDEX = 45
EXPECTED_ATTACHED_OPERATION_SEQUENCE = (
    "base_layer",
    "dropout_output",
    "lora_a_output",
    "lora_b_output",
)
EXPECTED_COMMON_EVENT_KEYS = (
    ("model.embed_tokens", "input"),
    ("model.embed_tokens", "output"),
    ("model.layers.0", "input"),
    ("model.layers.0.input_layernorm", "input"),
    ("model.layers.0.input_layernorm", "output"),
    ("model.layers.0.self_attn.q_proj", "input"),
    ("model.layers.0.self_attn.q_proj", "output"),
    ("model.layers.0.self_attn.k_proj", "input"),
    ("model.layers.0.self_attn.k_proj", "output"),
    ("model.layers.0.self_attn.v_proj", "input"),
    ("model.layers.0.self_attn.v_proj", "output"),
    ("model.layers.0.self_attn.o_proj", "input"),
    ("model.layers.0.self_attn.o_proj", "output"),
    ("model.layers.0.post_attention_layernorm", "input"),
    ("model.layers.0.post_attention_layernorm", "output"),
    ("model.layers.0.mlp.gate_proj", "input"),
    ("model.layers.0.mlp.gate_proj", "output"),
    ("model.layers.0.mlp.up_proj", "input"),
    ("model.layers.0.mlp.up_proj", "output"),
    ("model.layers.0.mlp.down_proj", "input"),
    ("model.layers.0.mlp.down_proj", "output"),
    ("model.layers.0", "output"),
    ("model.norm", "input"),
    ("model.norm", "output"),
    ("lm_head", "input"),
    ("lm_head", "output"),
)
ATTACHED_REPLAY_LABELS = (
    "factorized_scaled",
    "base_plus_factorized",
    "delta_weight_linear",
    "base_plus_delta_weight_linear",
    "expected_materialized_linear",
)
OPERATION_PAIRS = (
    (
        "q_proj_input_identity",
        f"{ATTACHED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|input",
        f"{MERGED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|input",
    ),
    (
        "attached_dropout_identity",
        f"{ATTACHED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|input",
        f"{ATTACHED_R1}|diagnostic|q_proj|dropout_output",
    ),
    (
        "attached_output_reconstruction",
        f"{ATTACHED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|output",
        f"{ATTACHED_R1}|diagnostic|q_proj|base_plus_factorized",
    ),
    (
        "merged_output_reconstruction",
        f"{MERGED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|output",
        f"{MERGED_R1}|diagnostic|q_proj|recomputed",
    ),
    (
        "expected_materialized_vs_merged_actual",
        f"{ATTACHED_R1}|diagnostic|q_proj|expected_materialized_linear",
        f"{MERGED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|output",
    ),
    (
        "factorized_lora_vs_delta_weight_linear",
        f"{ATTACHED_R1}|diagnostic|q_proj|factorized_scaled",
        f"{ATTACHED_R1}|diagnostic|q_proj|delta_weight_linear",
    ),
    (
        "attached_factorized_output_vs_split_delta_output",
        f"{ATTACHED_R1}|diagnostic|q_proj|base_plus_factorized",
        f"{ATTACHED_R1}|diagnostic|q_proj|base_plus_delta_weight_linear",
    ),
    (
        "split_base_plus_delta_vs_materialized_weight_linear",
        f"{ATTACHED_R1}|diagnostic|q_proj|base_plus_delta_weight_linear",
        f"{ATTACHED_R1}|diagnostic|q_proj|expected_materialized_linear",
    ),
    (
        "attached_output_vs_merged_output",
        f"{ATTACHED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|output",
        f"{MERGED_R1}|common|{FIRST_REGISTERED_BOUNDARY}|output",
    ),
)
WEIGHT_TENSOR_IDS = {
    "base_weight": f"{ATTACHED_R1}|weight|q_proj|base_weight",
    "delta_weight": f"{ATTACHED_R1}|weight|q_proj|delta_weight",
    "expected_merged_weight": (
        f"{ATTACHED_R1}|weight|q_proj|expected_merged_weight"
    ),
    "actual_merged_weight": f"{MERGED_R1}|weight|q_proj|actual_merged_weight",
    "attached_bias": f"{ATTACHED_R1}|weight|q_proj|attached_bias",
    "merged_bias": f"{MERGED_R1}|weight|q_proj|merged_bias",
}
_ARCHIVE_KEYS = frozenset(
    {"version", "path", "encoding", "byte_count", "sha256", "record_count", "records"}
)
_RECORD_KEYS = frozenset(
    {
        "tensor_id",
        "run_id",
        "path",
        "semantic_key",
        "event_scope",
        "event_index",
        "module_name",
        "module_type",
        "occurrence_index",
        "io_kind",
        "tensor_path",
        "native_dtype",
        "native_shape",
        "native_stride",
        "native_layout",
        "comparison_dtype",
        "shape",
        "elements",
        "all_finite",
        "byte_offset",
        "byte_length",
        "raw_payload_sha256",
        "canonical_value_sha256",
        "bound_sha256",
    }
)
_CAPTURE_EVENT_KEYS = frozenset(
    {
        "event_index",
        "module_name",
        "module_type",
        "occurrence_index",
        "call_index",
        "generation_step_index",
        "io_kind",
        "tensor_path",
        "tensor_id",
    }
)
_OPERATION_EVENT_KEYS = frozenset(
    {
        "event_index",
        "operation",
        "module_type",
        "call_index",
        "generation_step_index",
        "tensor_id",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "fp32_attached_merge_numerics_version",
        "experiment_id",
        "source_experiment_id",
        "source_lineage",
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
        "environment",
        "protocol",
        "frozen_path_references",
        "frozen_bf16_context",
        "runs",
        "generation_reproduction",
        "instrumented_repeat_stability",
        "capture_plan",
        "operation_graph_audit",
        "tensor_archive",
        "module_input_comparisons",
        "module_output_comparisons",
        "module_analysis",
        "operation_comparisons",
        "registered_execution_form_boundary",
        "weight_materialization",
        "materialization_axis",
        "classification",
        "causal_scope",
        "numerics_gate",
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
    }
)
_RUN_KEYS = frozenset(
    {
        "run_id",
        "path",
        "repeat",
        "order_index",
        "fresh_load",
        "materialization_form",
        "generated_token_ids",
        "token_count",
        "token_ids_sha256",
        "output_sha256",
        "generation_trace",
        "precision_audit",
        "target_alignment",
        "target_alignment_passed",
        "capture_events",
        "operation_graph_events",
        "capture_event_count",
        "capture_event_sequence_sha256",
        "common_capture_plan_passed",
        "operation_graph_plan_passed",
        "capture_plan_passed",
        "capture_tensor_ids",
        "lm_head_raw_logit_linked",
        "path_protocol_passed",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "frozen_path_reproduced",
        "memory_allocated_after_release_bytes",
        "elapsed_seconds",
    }
)
_TRACE_KEYS = frozenset(
    {"step_count", "vocabulary_size", "cache_returned", "scores", "raw_logits"}
)
_TRACE_VECTOR_KEYS = frozenset(
    {
        "native_dtypes",
        "shape_per_step",
        "comparison_dtype",
        "all_finite",
        "trace_sha256",
        "comparison_step_index",
        "comparison_step_vector_sha256",
    }
)
_ATTENTION_PROJECTION_STAGES = frozenset(
    {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    }
)
_COMMON_MODULE_TYPES = {
    "model.embed_tokens": "torch.nn.modules.sparse.Embedding",
    "model.layers.0": (
        "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer"
    ),
    "model.layers.0.input_layernorm": (
        "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm"
    ),
    "model.layers.0.post_attention_layernorm": (
        "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm"
    ),
    "model.layers.0.mlp.gate_proj": "torch.nn.modules.linear.Linear",
    "model.layers.0.mlp.up_proj": "torch.nn.modules.linear.Linear",
    "model.layers.0.mlp.down_proj": "torch.nn.modules.linear.Linear",
    "model.norm": "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
    "lm_head": "torch.nn.modules.linear.Linear",
}
_OPERATION_MODULE_TYPES = {
    "base_layer": "torch.nn.modules.linear.Linear",
    "dropout_output": "torch.nn.modules.dropout.Dropout",
    "lora_a_output": "torch.nn.modules.linear.Linear",
    "lora_b_output": "torch.nn.modules.linear.Linear",
}


def _record_contract(
    *,
    tensor_id: str,
    run_id: str,
    semantic_key: str,
    event_scope: str,
    event_index: int | None,
    module_name: str,
    module_type: str,
    occurrence_index: int | None,
    io_kind: str,
    tensor_path: str,
    native_dtype: str,
    shape: list[int],
    native_stride: list[int],
) -> dict[str, Any]:
    return {
        "tensor_id": tensor_id,
        "run_id": run_id,
        "path": RUN_PATHS[run_id],
        "semantic_key": semantic_key,
        "event_scope": event_scope,
        "event_index": event_index,
        "module_name": module_name,
        "module_type": module_type,
        "occurrence_index": occurrence_index,
        "io_kind": io_kind,
        "tensor_path": tensor_path,
        "native_dtype": native_dtype,
        "native_shape": shape,
        "native_stride": native_stride,
        "native_layout": "strided",
        "comparison_dtype": "float32",
        "shape": shape,
        "elements": math.prod(shape),
        "all_finite": True,
    }


def _common_module_type(stage: str, path: str) -> str:
    if stage in _ATTENTION_PROJECTION_STAGES:
        return (
            "peft.tuners.lora.layer.Linear"
            if path == ATTACHED_PATH
            else "torch.nn.modules.linear.Linear"
        )
    return _COMMON_MODULE_TYPES[stage]


def _common_tensor_contract(
    stage: str,
    io_kind: str,
) -> tuple[str, list[int], list[int]]:
    if stage == "model.embed_tokens" and io_kind == "input":
        return "int64", [1, 1], [1, 1]
    if io_kind == "input":
        width = 8960 if stage == "model.layers.0.mlp.down_proj" else 1536
    elif stage in {
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }:
        width = 256
    elif stage in {
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    }:
        width = 8960
    elif stage == "lm_head":
        width = 151936
    else:
        width = 1536
    return "float32", [1, 1, width], [width, width, 1]


def _expected_record_metadata() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for run_id, path, _repeat, _order in RUN_PLAN:
        for event_index, (stage, io_kind) in enumerate(
            EXPECTED_COMMON_EVENT_KEYS
        ):
            tensor_id = f"{run_id}|common|{stage}|{io_kind}"
            native_dtype, shape, native_stride = _common_tensor_contract(
                stage,
                io_kind,
            )
            expected[tensor_id] = _record_contract(
                tensor_id=tensor_id,
                run_id=run_id,
                semantic_key=f"common|{stage}|{io_kind}",
                event_scope="common",
                event_index=event_index,
                module_name=stage,
                module_type=_common_module_type(stage, path),
                occurrence_index=0,
                io_kind=io_kind,
                tensor_path=(
                    "args[0]"
                    if io_kind == "input"
                    else "output[0]"
                    if stage == "model.layers.0"
                    else "output"
                ),
                native_dtype=native_dtype,
                shape=shape,
                native_stride=native_stride,
            )
        for kind in ("scores", "raw_logits"):
            tensor_id = f"{run_id}|generation|{kind}|step{TARGET_STEP_INDEX}"
            expected[tensor_id] = _record_contract(
                tensor_id=tensor_id,
                run_id=run_id,
                semantic_key=f"generation|{kind}|step{TARGET_STEP_INDEX}",
                event_scope="generation",
                event_index=TARGET_STEP_INDEX,
                module_name="generate",
                module_type="transformers.generation",
                occurrence_index=0,
                io_kind=kind,
                tensor_path=f"generated.{kind}[{TARGET_STEP_INDEX}]",
                native_dtype="float32",
                shape=[151936],
                native_stride=[1],
            )
        replay_labels: tuple[str, ...]
        if path == ATTACHED_PATH:
            for event_index, operation in enumerate(
                EXPECTED_ATTACHED_OPERATION_SEQUENCE
            ):
                tensor_id = f"{run_id}|diagnostic|q_proj|{operation}"
                width = 16 if operation == "lora_a_output" else 1536
                expected[tensor_id] = _record_contract(
                    tensor_id=tensor_id,
                    run_id=run_id,
                    semantic_key=f"diagnostic|q_proj|{operation}",
                    event_scope="operation_graph",
                    event_index=event_index,
                    module_name=f"{FIRST_REGISTERED_BOUNDARY}.{operation}",
                    module_type=_OPERATION_MODULE_TYPES[operation],
                    occurrence_index=0,
                    io_kind="output",
                    tensor_path="output",
                    native_dtype="float32",
                    shape=[1, 1, width],
                    native_stride=[width, width, 1],
                )
            replay_labels = ATTACHED_REPLAY_LABELS
        else:
            replay_labels = ("recomputed",)
        for label in replay_labels:
            tensor_id = f"{run_id}|diagnostic|q_proj|{label}"
            expected[tensor_id] = _record_contract(
                tensor_id=tensor_id,
                run_id=run_id,
                semantic_key=f"diagnostic|q_proj|{label}",
                event_scope="replay",
                event_index=None,
                module_name=FIRST_REGISTERED_BOUNDARY,
                module_type="torch.nn.functional",
                occurrence_index=None,
                io_kind="replay_output",
                tensor_path=label,
                native_dtype="float32",
                shape=[1, 1, 1536],
                native_stride=[1536, 1536, 1],
            )
    for key, tensor_id in WEIGHT_TENSOR_IDS.items():
        run_id = tensor_id.split("|", maxsplit=1)[0]
        is_bias = key.endswith("bias")
        shape = [1536] if is_bias else [1536, 1536]
        expected[tensor_id] = _record_contract(
            tensor_id=tensor_id,
            run_id=run_id,
            semantic_key=f"weight|q_proj|{key}",
            event_scope="weight_materialization",
            event_index=None,
            module_name=FIRST_REGISTERED_BOUNDARY,
            module_type="torch.Tensor",
            occurrence_index=None,
            io_kind="bias" if is_bias else "weight",
            tensor_path=key,
            native_dtype="float32",
            shape=shape,
            native_stride=[1] if is_bias else [1536, 1],
        )
    return expected


def _validate_record_metadata_closure(
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = _expected_record_metadata()
    if set(records) != set(expected):
        _fail(
            "ARCHIVE_RECORD_METADATA_CLOSURE_MISMATCH",
            "$.tensor_archive.records",
            repr(sorted(set(records) ^ set(expected))),
        )
    for tensor_id, contract in expected.items():
        record = records[tensor_id]
        actual = {key: record.get(key) for key in contract}
        if actual != contract:
            _fail(
                "ARCHIVE_RECORD_METADATA_MISMATCH",
                f"$.tensor_archive.records.{tensor_id}",
                repr(actual),
            )


def _floating_inventory(tensors: int, elements: int) -> dict[str, Any]:
    return {
        "floating_tensors": tensors,
        "floating_elements": elements,
        "dtypes": {"float32": elements},
        "devices": {"cuda:0": elements},
    }


def _generation_precision_contract() -> dict[str, Any]:
    return {
        "score_dtypes": ["float32"],
        "all_scores_float32": True,
        "raw_logit_dtypes": ["float32"],
        "all_raw_logits_float32": True,
        "dtype_semantics": "transformers_generate_return_tensor_dtype",
        "autocast_enabled": False,
        "training": False,
    }


def _expected_precision_audit(path: str) -> dict[str, Any]:
    base = _floating_inventory(338, 1_543_714_304)
    adapter = _floating_inventory(224, 4_358_144)
    buffers = _floating_inventory(1, 64)
    shared = {
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
    }
    if path == ATTACHED_PATH:
        return {
            "base_parameters": base,
            "adapter_parameters": adapter,
            "floating_buffers": buffers,
            **shared,
            "lora_dropout": {"modules": 112, "training_modules": 0},
            "generation": _generation_precision_contract(),
        }
    return {
        "pre_merge": {
            "base_parameters": base,
            "adapter_parameters": adapter,
            "floating_buffers": buffers,
            **shared,
        },
        "post_merge": {
            "parameters": base,
            "floating_buffers": buffers,
            "lora_target_modules": 0,
            "lora_parameter_tensors": 0,
            "is_peft_model": False,
            "input_output_embeddings_tied": True,
            "attn_implementation": "sdpa",
            "attention_class": "Qwen2Attention",
            "output_attentions": False,
            "hf_device_map": None,
        },
        "lora_dropout": {"modules": 0, "training_modules": 0},
        "generation": _generation_precision_contract(),
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validate_generation_trace(trace: object, path: str) -> None:
    value = _require_mapping(trace, path)
    if set(value) != set(_TRACE_KEYS):
        _fail("INVALID_GENERATION_TRACE_FIELDS", path, repr(value))
    if (
        value.get("step_count") != 48
        or value.get("vocabulary_size") != 151936
        or value.get("cache_returned") is not True
    ):
        _fail("INVALID_GENERATION_TRACE", path, repr(value))
    for kind in ("scores", "raw_logits"):
        vector = _require_mapping(value.get(kind), f"{path}.{kind}")
        if set(vector) != set(_TRACE_VECTOR_KEYS) or vector != {
            "native_dtypes": ["float32"],
            "shape_per_step": [1, 151936],
            "comparison_dtype": "float32",
            "all_finite": True,
            "trace_sha256": vector.get("trace_sha256"),
            "comparison_step_index": TARGET_STEP_INDEX,
            "comparison_step_vector_sha256": vector.get(
                "comparison_step_vector_sha256"
            ),
        }:
            _fail("INVALID_GENERATION_TRACE_VECTOR", f"{path}.{kind}", repr(vector))
        if not _valid_sha256(vector["trace_sha256"]) or not _valid_sha256(
            vector["comparison_step_vector_sha256"]
        ):
            _fail("INVALID_GENERATION_TRACE_DIGEST", f"{path}.{kind}", repr(vector))


def validate_frozen_numerics_evidence(
    evidence: Mapping[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    """Validate both the raw sidecar and the final frozen gate claims."""

    if not isinstance(evidence, Mapping) or set(evidence) != set(_TOP_LEVEL_KEYS):
        _fail("INVALID_FROZEN_EVIDENCE_FIELDS", "$", repr(sorted(evidence)))
    if (
        evidence.get("fp32_attached_merge_numerics_version") != 1
        or evidence.get("experiment_id")
        != "fc-mvp-001-fp32-attached-merge-numerics-v1"
        or evidence.get("source_experiment_id")
        != "fc-mvp-001-fp32-attached-merge-isolation-v1"
        or evidence.get("example_id") != "eval-001"
        or evidence.get("input_token_count") != 339
        or evidence.get("input_token_ids_sha256")
        != "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
    ):
        _fail("INVALID_FROZEN_EVIDENCE_IDENTITY", "$", repr(evidence))
    protocol = _require_mapping(evidence.get("protocol"), "$.protocol")
    expected_run_plan = [
        {"path": path, "repeat": repeat, "run_id": run_id}
        for run_id, path, repeat, _order in RUN_PLAN
    ]
    expected_protocol = {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": expected_run_plan,
        "run_order_design": "ABBA",
        "fresh_loads_per_path": {ATTACHED_PATH: 2, MERGED_PATH: 2},
        "max_residual_cuda_bytes": 16_777_216,
        "target_forward": {
            "generation_step_index": 45,
            "input_generated_token_index": 44,
            "input_token_id": 788,
            "predicted_token_id": 3849,
            "past_length": 383,
            "cache_position": [383],
        },
        "paths": {
            ATTACHED_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": False,
                "merge": False,
                "inference_parameter_dtype": "float32",
            },
            MERGED_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": False,
                "merge_dtype": "float32",
                "inference_parameter_dtype": "float32",
                "merge": True,
                "safe_merge": True,
                "adapter_names": ["default"],
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
        "operation_graphs": {
            "attached": (
                "base_linear(x) + lora_B(lora_A(dropout(x))) * scale"
            ),
            "merged": (
                "linear(x, base_weight + (lora_B_weight @ lora_A_weight) * scale)"
            ),
        },
    }
    if protocol != expected_protocol:
        _fail("INVALID_FROZEN_PROTOCOL", "$.protocol", repr(protocol))
    capture_plan = _require_mapping(
        evidence.get("capture_plan"),
        "$.capture_plan",
    )
    expected_capture_plan = {
        "scope": "target_forward_pre_registered_paired_common_semantic_modules",
        "target_generation_step_index": 45,
        "selection_basis": "frozen_bf16_token_boundary_context",
        "common_stages": list(COMMON_OUTPUT_STAGES),
        "expected_common_event_keys": [
            {"module_name": module, "io_kind": io_kind}
            for module, io_kind in EXPECTED_COMMON_EVENT_KEYS
        ],
        "expected_attached_operation_sequence": list(
            EXPECTED_ATTACHED_OPERATION_SEQUENCE
        ),
        "expected_adapter_hyperparameters": {
            "rank": 16,
            "alpha": 32.0,
            "scaling": 2.0,
            "dropout_probability": 0.05,
        },
        "tensor_selection": "first_tensor_leaf_per_registered_module_input_output",
        "hook_capture": "gpu_clone_then_post_generation_cpu_float32_archive",
        "exactness": "numerical_and_bitwise_no_tolerance",
        "does_not_cover": [
            "all_tensor_leaves",
            "unregistered_functional_operations",
            "earliest_temporal_divergence_across_generation_history",
            "low_level_cuda_kernel_identity",
        ],
    }
    if capture_plan != expected_capture_plan:
        _fail("INVALID_FROZEN_CAPTURE_PLAN", "$.capture_plan", repr(capture_plan))
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != len(RUN_PLAN):
        _fail("INVALID_FROZEN_RUNS", "$.runs", repr(raw_runs))
    runs: dict[str, Mapping[str, Any]] = {}
    for index, raw_run in enumerate(raw_runs):
        run = _require_mapping(raw_run, f"$.runs[{index}]")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or run_id in runs:
            _fail("INVALID_FROZEN_RUN_ID", f"$.runs[{index}]", repr(run_id))
        runs[run_id] = run
    if set(runs) != set(RUN_PATHS):
        _fail("INVALID_FROZEN_RUN_SET", "$.runs", repr(sorted(runs)))
    expected_alignment = {
        "call_index": 45,
        "generation_step_index": 45,
        "input_token_ids": [788],
        "input_shape": [1, 1],
        "cache_position": [383],
        "position_ids": [383],
        "past_length": 383,
        "causal_forward_calls": 48,
    }
    references = _require_mapping(
        evidence.get("frozen_path_references"),
        "$.frozen_path_references",
    )
    expected_references = {
        ATTACHED_PATH: {
            "token_count": 48,
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
            "comparison_step_index": 45,
            "comparison_score_vector_sha256": (
                "sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451"
            ),
            "comparison_raw_logit_vector_sha256": (
                "sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a"
            ),
        },
        MERGED_PATH: {
            "token_count": 48,
            "token_ids_sha256": (
                "sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca"
            ),
            "output_sha256": (
                "sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca"
            ),
            "score_trace_sha256": (
                "sha256:1b7b93ba7cba872cfb8dd4d50e452df8fb76b76e4ddda5c513ec9697e67e1fe9"
            ),
            "raw_logit_trace_sha256": (
                "sha256:a0a03e4bf73123db942a0e44122b3d312d3885aa195a49782d0dd92a9d4a65ee"
            ),
            "comparison_step_index": 45,
            "comparison_score_vector_sha256": (
                "sha256:c645fd357a4d34fc94dab70978e90143886051e898edfc0049ad67a370a14d8b"
            ),
            "comparison_raw_logit_vector_sha256": (
                "sha256:87d3bee7986814bb3a9bb22b249247235cc52e8444ba96b22b82b409ed1e0c93"
            ),
        },
    }
    if references != expected_references:
        _fail("FROZEN_REFERENCE_MISMATCH", "$.frozen_path_references", repr(references))
    for run_id, path, _repeat, _order in RUN_PLAN:
        run = runs[run_id]
        reference = _require_mapping(references.get(path), f"$.references.{path}")
        token_ids = run.get("generated_token_ids")
        if (
            not isinstance(token_ids, list)
            or run.get("token_count") != len(token_ids)
            or run.get("token_ids_sha256") != token_ids_sha256(token_ids)
            or run.get("token_count") != reference.get("token_count")
            or run.get("token_ids_sha256") != reference.get("token_ids_sha256")
            or run.get("output_sha256") != reference.get("output_sha256")
            or run.get("generation_trace", {}).get("scores", {}).get(
                "trace_sha256"
            )
            != reference.get("score_trace_sha256")
            or run.get("generation_trace", {}).get("raw_logits", {}).get(
                "trace_sha256"
            )
            != reference.get("raw_logit_trace_sha256")
            or run.get("generation_trace", {}).get("scores", {}).get(
                "comparison_step_vector_sha256"
            )
            != reference.get("comparison_score_vector_sha256")
            or run.get("generation_trace", {}).get("raw_logits", {}).get(
                "comparison_step_vector_sha256"
            )
            != reference.get("comparison_raw_logit_vector_sha256")
            or reference.get("comparison_step_index") != TARGET_STEP_INDEX
            or token_ids[TARGET_STEP_INDEX] != 3849
            or run.get("target_alignment") != expected_alignment
            or run.get("target_alignment_passed") is not True
            or run.get("frozen_path_reproduced") is not True
            or run.get("path_protocol_passed") is not True
        ):
            _fail("FROZEN_RUN_CLAIM_MISMATCH", f"$.runs.{run_id}", repr(run))
    raw_run_peaks = [run.get("peak_gpu_memory_bytes") for run in runs.values()]
    raw_run_elapsed = [run.get("elapsed_seconds") for run in runs.values()]
    top_peak = evidence.get("peak_gpu_memory_bytes")
    top_elapsed = evidence.get("elapsed_seconds")
    if (
        any(not _strict_int(value) for value in raw_run_peaks)
        or not _strict_int(top_peak)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in raw_run_elapsed
        )
        or not isinstance(top_elapsed, (int, float))
        or isinstance(top_elapsed, bool)
        or not math.isfinite(top_elapsed)
    ):
        _fail("FROZEN_RESOURCE_CLAIM_MISMATCH", "$.runs", repr(raw_runs))
    run_peaks = [cast(int, value) for value in raw_run_peaks]
    run_elapsed = [
        float(cast(int | float, value)) for value in raw_run_elapsed
    ]
    top_peak_value = cast(int, top_peak)
    top_elapsed_value = float(cast(int | float, top_elapsed))
    if (
        any(value <= 0 for value in run_peaks)
        or top_peak_value != max(run_peaks)
        or any(value <= 0 for value in run_elapsed)
        or top_elapsed_value < math.fsum(run_elapsed)
    ):
        _fail("FROZEN_RESOURCE_CLAIM_MISMATCH", "$.runs", repr(raw_runs))
    representative_attached = runs[ATTACHED_R1]
    representative_merged = runs[MERGED_R1]
    generation_reproduction = {
        "all_runs_reproduce_frozen_paths": all(
            run["frozen_path_reproduced"] for run in runs.values()
        ),
        "cross_path_token_identity": (
            representative_attached["generated_token_ids"]
            == representative_merged["generated_token_ids"]
        ),
        "cross_path_output_identity": (
            representative_attached["output_sha256"]
            == representative_merged["output_sha256"]
        ),
        "score_trace_identity": (
            representative_attached["generation_trace"]["scores"]["trace_sha256"]
            == representative_merged["generation_trace"]["scores"]["trace_sha256"]
        ),
        "raw_logit_trace_identity": (
            representative_attached["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
            == representative_merged["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
        ),
        "same_dtype_token_boundary": False,
    }
    _require_exact(
        evidence.get("generation_reproduction"),
        generation_reproduction,
        "GENERATION_REPRODUCTION_MISMATCH",
        "$.generation_reproduction",
    )
    if generation_reproduction != {
        "all_runs_reproduce_frozen_paths": True,
        "cross_path_token_identity": True,
        "cross_path_output_identity": True,
        "score_trace_identity": False,
        "raw_logit_trace_identity": False,
        "same_dtype_token_boundary": False,
    }:
        _fail(
            "FROZEN_GENERATION_OUTCOME_MISMATCH",
            "$.generation_reproduction",
            repr(generation_reproduction),
        )
    raw_result = validate_raw_tensor_archive(evidence, payload)
    weight = evidence["weight_materialization"]
    expected_materialization_axis = (
        "fp32_materialization_rounding_present"
        if weight["ideal_nonzero_updates_rounded_to_base"] > 0
        else "fp32_materialization_rounding_absent"
    )
    if evidence.get("materialization_axis") != expected_materialization_axis:
        _fail(
            "MATERIALIZATION_AXIS_MISMATCH",
            "$.materialization_axis",
            repr(evidence.get("materialization_axis")),
        )
    acceptance = _require_mapping(evidence.get("acceptance"), "$.acceptance")
    expected_acceptance = {
        "upstream_isolation_locked": True,
        "source_adapter_unchanged": True,
        "source_model_unchanged": True,
        "eval_digest_unchanged": True,
        "prompt_digest_unchanged": True,
        "frozen_input_reproduced": True,
        "all_instrumented_runs_reproduce_frozen_paths": generation_reproduction[
            "all_runs_reproduce_frozen_paths"
        ],
        "instrumented_repeat_stable": evidence[
            "instrumented_repeat_stability"
        ]["passed"],
        "target_forward_aligned": all(
            run["target_alignment_passed"] for run in runs.values()
        ),
        "capture_plan_executed": all(
            run["capture_plan_passed"] for run in runs.values()
        ),
        "paired_event_sequence_identical": len(
            {run["capture_event_sequence_sha256"] for run in runs.values()}
        )
        == 1,
        "lm_head_raw_logit_linked": all(
            run["lm_head_raw_logit_linked"] for run in runs.values()
        ),
        "first_captured_output_divergence_located": (
            evidence["module_analysis"]["first_divergent_module"]
            == FIRST_REGISTERED_BOUNDARY
        ),
        "preceding_captured_outputs_identical": evidence["module_analysis"][
            "preceding_modules_identical"
        ],
        "first_divergent_module_input_identical": next(
            item
            for item in evidence["module_input_comparisons"]
            if item["name"] == f"{FIRST_REGISTERED_BOUNDARY}.input"
        )["bitwise_equal"],
        "operation_graph_executed": evidence["operation_graph_audit"]["passed"],
        "safe_merge_weight_reproduced": evidence[
            "registered_execution_form_boundary"
        ]["safe_merge_weight_reproduced"],
        "registered_execution_form_boundary_quantified": evidence[
            "registered_execution_form_boundary"
        ]["registered_execution_form_boundary_quantified"],
        "cross_path_token_identity_preserved": generation_reproduction[
            "cross_path_token_identity"
        ],
        "bf16_context_only": evidence["frozen_bf16_context"]["context_only"],
    }
    if acceptance != expected_acceptance or not all(
        value is True for value in expected_acceptance.values()
    ):
        _fail("FROZEN_ACCEPTANCE_MISMATCH", "$.acceptance", repr(acceptance))
    expected_numerics_gate = {
        "path_reproduction": generation_reproduction[
            "all_runs_reproduce_frozen_paths"
        ],
        "instrumented_repeat_stability": evidence[
            "instrumented_repeat_stability"
        ]["passed"],
        "target_forward_alignment": all(
            run["target_alignment_passed"] for run in runs.values()
        ),
        "first_paired_common_boundary_located": (
            evidence["module_analysis"]["first_divergent_module"]
            == FIRST_REGISTERED_BOUNDARY
        ),
        "registered_execution_form_boundary_quantified": evidence[
            "registered_execution_form_boundary"
        ]["registered_execution_form_boundary_quantified"],
        "passed": all(value is True for value in acceptance.values()),
    }
    _require_exact(
        evidence.get("numerics_gate"),
        expected_numerics_gate,
        "FROZEN_NUMERICS_GATE_MISMATCH",
        "$.numerics_gate",
    )
    expected_constraints = {
        "failed_candidate_change": False,
        "frozen_bf16_path_rerun": False,
        "locked_path_dtype_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_promotion": False,
    }
    expected_causal_scope = {
        "supports": (
            "the first numerical divergence within the pre-registered paired "
            "common-module capture plan at frozen generation step 45 and its "
            "registered FP32 q-projection execution-form replays; both "
            "registered counterfactual differences remain nonzero at the "
            "q-projection output boundary"
        ),
        "does_not_support": [
            "earliest_temporal_divergence_across_the_full_generation_history",
            "all_model_operations_or_unregistered_functional_boundaries",
            "same_dtype_token_boundary",
            "specific_cuda_kernel_identity_or_root_cause",
            "pure_floating_point_nonassociativity_as_the_only_root_cause",
            "independent_counterfactual_propagation_beyond_q_proj",
            "peft_implementation_bug_claim",
            "full_eval_generalization",
            "merged_artifact_promotion",
            "runtime_eligibility",
        ],
    }
    expected_bf16_context = {
        "context_only": True,
        "gpu_paths_rerun": False,
        "source_experiment_id": "fc-mvp-001-bf16-merge-stability-v1",
        "first_divergent_token_index": 45,
        "paths": {
            "bf16_attached_adapter": {
                "token_count": 48,
                "token_ids_sha256": (
                    "sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173"
                ),
                "output_sha256": (
                    "sha256:b3bef0f22aad858ad94275466af0cc082dbb7fe42a7814320aa4a8995dff0bc5"
                ),
                "boundary_token_id": 1866,
            },
            "bf16_safe_merged": {
                "token_count": 48,
                "token_ids_sha256": (
                    "sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca"
                ),
                "output_sha256": (
                    "sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca"
                ),
                "boundary_token_id": 3849,
            },
        },
    }
    expected_next_action = {
        "gate_id": "FC-MVP-001-attached-dtype-isolation-v1",
        "action": (
            "hold the Adapter attached and compare fresh repeat-stable BF16 "
            "and FP32 execution on frozen eval-001 at the same cached step 45, "
            "to isolate the remaining token-level precision effect without "
            "changing merge form"
        ),
        "acceptance": {
            "attached_bf16_path_reproduced": True,
            "attached_fp32_path_reproduced": True,
            "same_execution_form_dtype_effect_classified": True,
            "first_token_boundary_reproduced": True,
            "source_inputs_unchanged": True,
        },
        "constraints": {
            "attached_execution_form_change": False,
            "frozen_bf16_reference_change": False,
            "fresh_bf16_attached_rerun_required": True,
            "frozen_fp32_reference_change": False,
            "fresh_fp32_attached_rerun_required": True,
            "target_step_change": False,
            "locked_path_backend_change": False,
            "locked_path_decoding_change": False,
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "runtime_integration": False,
            "full_eval_run": False,
            "merged_artifact_promotion": False,
        },
    }
    if (
        evidence.get("constraints") != expected_constraints
        or evidence.get("causal_scope") != expected_causal_scope
        or evidence.get("frozen_bf16_context") != expected_bf16_context
        or evidence.get("locked_next_action") != expected_next_action
        or evidence.get("remediation_gate")
        != {
            "source_gate_passed": False,
            "new_remediation_tested": False,
            "passed": False,
        }
        or evidence.get("merged_artifact_saved") is not False
        or evidence.get("merged_artifact_allowed") is not False
        or evidence.get("runtime_eligible") is not False
        or evidence.get("runtime_eligibility_reason")
        != evidence.get("classification")
        or evidence.get("offline") is not True
        or evidence.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-isolation-v1"
    ):
        _fail("FROZEN_CLOSURE_MISMATCH", "$", repr(evidence))
    return {**raw_result, "frozen_gate_valid": True}


def validate_raw_tensor_archive(
    evidence: Mapping[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    """Recompute the frozen numerics claims from the raw float32 sidecar."""

    if not isinstance(evidence, Mapping):
        _fail("INVALID_EVIDENCE", "$", repr(type(evidence)))
    if not isinstance(payload, bytes):
        _fail("INVALID_ARCHIVE_PAYLOAD", "$.tensor_archive", repr(type(payload)))
    descriptor = _require_mapping(
        evidence.get("tensor_archive"),
        "$.tensor_archive",
    )
    if set(descriptor) != set(_ARCHIVE_KEYS):
        _fail(
            "INVALID_ARCHIVE_FIELDS",
            "$.tensor_archive",
            repr(sorted(descriptor)),
        )
    if (
        descriptor.get("version") != 1
        or descriptor.get("path") != ARCHIVE_BASENAME
        or descriptor.get("encoding")
        != "contiguous_ieee754_float32_little_endian"
        or descriptor.get("byte_count") != len(payload)
        or descriptor.get("sha256") != _sha256(payload)
    ):
        _fail("INVALID_ARCHIVE_DESCRIPTOR", "$.tensor_archive", repr(descriptor))
    records, record_slices = _validate_records(descriptor, payload)
    _validate_record_metadata_closure(records)
    runs = _validate_capture_closure(evidence, records)
    module_inputs = [
        _recompute_comparison(
            f"{stage}.input",
            f"{ATTACHED_R1}|common|{stage}|input",
            f"{MERGED_R1}|common|{stage}|input",
            records,
            record_slices,
        )
        for stage in COMMON_OUTPUT_STAGES
    ]
    module_outputs = [
        _recompute_comparison(
            stage,
            f"{ATTACHED_R1}|common|{stage}|output",
            f"{MERGED_R1}|common|{stage}|output",
            records,
            record_slices,
        )
        for stage in COMMON_OUTPUT_STAGES
    ]
    operations = [
        _recompute_comparison(name, left, right, records, record_slices)
        for name, left, right in OPERATION_PAIRS
    ]
    _require_exact(
        evidence.get("module_input_comparisons"),
        module_inputs,
        "RAW_MODULE_INPUT_COMPARISON_MISMATCH",
        "$.module_input_comparisons",
    )
    _require_exact(
        evidence.get("module_output_comparisons"),
        module_outputs,
        "RAW_MODULE_OUTPUT_COMPARISON_MISMATCH",
        "$.module_output_comparisons",
    )
    _require_exact(
        evidence.get("operation_comparisons"),
        operations,
        "RAW_OPERATION_COMPARISON_MISMATCH",
        "$.operation_comparisons",
    )
    module_analysis = analyze_module_comparisons(module_outputs)
    _require_exact(
        evidence.get("module_analysis"),
        module_analysis,
        "RAW_MODULE_ANALYSIS_MISMATCH",
        "$.module_analysis",
    )
    weight_audit = _recompute_weight_audit(
        evidence,
        records,
        record_slices,
    )
    boundary = classify_operation_order(module_outputs, operations, weight_audit)
    _require_exact(
        evidence.get("registered_execution_form_boundary"),
        boundary,
        "RAW_EXECUTION_FORM_BOUNDARY_MISMATCH",
        "$.registered_execution_form_boundary",
    )
    if evidence.get("classification") != boundary["classification"]:
        _fail(
            "RAW_CLASSIFICATION_MISMATCH",
            "$.classification",
            repr(evidence.get("classification")),
        )
    repeat_stability = _recompute_repeat_stability(runs, records)
    _require_exact(
        evidence.get("instrumented_repeat_stability"),
        repeat_stability,
        "RAW_REPEAT_STABILITY_MISMATCH",
        "$.instrumented_repeat_stability",
    )
    _validate_generation_links(runs, records, record_slices)
    _validate_operation_graph(evidence, runs, records, record_slices)
    return {
        "valid": True,
        "archive_sha256": descriptor["sha256"],
        "byte_count": len(payload),
        "record_count": len(records),
        "comparisons_recomputed": (
            len(module_inputs) + len(module_outputs) + len(operations)
        ),
        "weight_elements_recomputed": weight_audit["elements"],
        "bias_elements_recomputed": weight_audit["bias_elements"],
        "repeat_stability_recomputed": repeat_stability["passed"],
        "classification": boundary["classification"],
    }


def _validate_records(
    descriptor: Mapping[str, Any],
    payload: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, memoryview]]:
    raw_records = descriptor.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        _fail("INVALID_ARCHIVE_RECORDS", "$.tensor_archive.records", repr(raw_records))
    if descriptor.get("record_count") != len(raw_records):
        _fail(
            "INVALID_ARCHIVE_RECORD_COUNT",
            "$.tensor_archive.record_count",
            repr(descriptor.get("record_count")),
        )
    records: dict[str, dict[str, Any]] = {}
    slices: dict[str, memoryview] = {}
    semantic_keys: set[tuple[str, str]] = set()
    cursor = 0
    payload_view = memoryview(payload)
    for index, raw_record in enumerate(raw_records):
        path = f"$.tensor_archive.records[{index}]"
        record = _require_mapping(raw_record, path)
        if set(record) != set(_RECORD_KEYS):
            _fail("INVALID_ARCHIVE_RECORD_FIELDS", path, repr(sorted(record)))
        tensor_id = record.get("tensor_id")
        run_id = record.get("run_id")
        semantic_key = record.get("semantic_key")
        if (
            not isinstance(tensor_id, str)
            or not tensor_id
            or tensor_id in records
            or run_id not in RUN_PATHS
            or record.get("path") != RUN_PATHS.get(run_id)
            or not isinstance(semantic_key, str)
            or not semantic_key
            or (run_id, semantic_key) in semantic_keys
        ):
            _fail("INVALID_ARCHIVE_RECORD_IDENTITY", path, repr(record))
        shape = _validate_shape(record.get("shape"), f"{path}.shape")
        native_shape = _validate_shape(
            record.get("native_shape"),
            f"{path}.native_shape",
        )
        elements = record.get("elements")
        offset = record.get("byte_offset")
        length = record.get("byte_length")
        if (
            shape != native_shape
            or not _strict_int(elements)
            or elements != math.prod(shape)
            or not _strict_int(offset)
            or not _strict_int(length)
            or offset != cursor
            or offset % 4 != 0
            or length != elements * 4
            or offset + length > len(payload)
        ):
            _fail("INVALID_ARCHIVE_RECORD_RANGE", path, repr(record))
        native_stride = record.get("native_stride")
        if (
            not isinstance(native_stride, list)
            or len(native_stride) != len(shape)
            or any(not _strict_int(value) or value < 0 for value in native_stride)
            or record.get("native_dtype") not in {"float32", "int64"}
            or record.get("native_layout") != "strided"
            or record.get("comparison_dtype") != "float32"
            or record.get("all_finite") is not True
        ):
            _fail("INVALID_ARCHIVE_RECORD_TENSOR", path, repr(record))
        for key in (
            "event_scope",
            "module_name",
            "module_type",
            "io_kind",
            "tensor_path",
        ):
            if not isinstance(record.get(key), str) or not record[key]:
                _fail("INVALID_ARCHIVE_RECORD_METADATA", f"{path}.{key}", repr(record))
        for key in ("event_index", "occurrence_index"):
            if record.get(key) is not None and not _strict_int(record[key]):
                _fail("INVALID_ARCHIVE_RECORD_METADATA", f"{path}.{key}", repr(record))
        raw = payload_view[offset : offset + length]
        raw_bytes = raw.tobytes()
        header = dict(record)
        bound = header.pop("bound_sha256")
        if (
            record.get("raw_payload_sha256") != _sha256(raw)
            or record.get("canonical_value_sha256")
            != _canonical_float32_sha256(raw_bytes)
            or bound != _bound_sha256(_canonical_json(header), raw)
            or not _all_finite_float32(raw_bytes)
        ):
            _fail("INVALID_ARCHIVE_RECORD_INTEGRITY", path, tensor_id)
        records[tensor_id] = dict(record)
        slices[tensor_id] = raw
        semantic_keys.add((run_id, semantic_key))
        cursor += length
    if cursor != len(payload):
        _fail("ARCHIVE_PAYLOAD_NOT_EXHAUSTIVE", "$.tensor_archive", repr(cursor))
    return records, slices


def _validate_capture_closure(
    evidence: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    raw_runs = evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != len(RUN_PLAN):
        _fail("INVALID_ARCHIVE_RUNS", "$.runs", repr(raw_runs))
    runs: dict[str, dict[str, Any]] = {}
    capture_ids: set[str] = set()
    previous_after: int | None = None
    for index, (expected_run, expected_path, repeat, order) in enumerate(RUN_PLAN):
        run_path = f"$.runs[{index}]"
        run = _require_mapping(raw_runs[index], run_path)
        if set(run) != set(_RUN_KEYS):
            _fail("INVALID_ARCHIVE_RUN_FIELDS", run_path, repr(sorted(run)))
        expected_form = (
            "attached_factorized_lora"
            if expected_path == ATTACHED_PATH
            else "materialized_safe_merge"
        )
        peak = run.get("peak_gpu_memory_bytes")
        before = run.get("memory_allocated_before_load_bytes")
        after = run.get("memory_allocated_after_release_bytes")
        elapsed = run.get("elapsed_seconds")
        if (
            not _strict_int(peak)
            or not _strict_int(before)
            or not _strict_int(after)
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
        ):
            _fail("INVALID_ARCHIVE_RUN_PLAN", run_path, repr(run))
        peak_value = cast(int, peak)
        before_value = cast(int, before)
        after_value = cast(int, after)
        elapsed_value = float(cast(int | float, elapsed))
        if (
            run.get("run_id") != expected_run
            or run.get("path") != expected_path
            or run.get("repeat") != repeat
            or run.get("order_index") != order
            or run.get("fresh_load") is not True
            or run.get("materialization_form") != expected_form
            or run.get("precision_audit")
            != _expected_precision_audit(expected_path)
            or peak_value <= 0
            or before_value < 0
            or before_value > 16_777_216
            or after_value < 0
            or after_value > 16_777_216
            or (index == 0 and before_value != 0)
            or (
                previous_after is not None
                and before_value != previous_after
            )
            or elapsed_value <= 0
        ):
            _fail("INVALID_ARCHIVE_RUN_PLAN", run_path, repr(run))
        _validate_generation_trace(
            run.get("generation_trace"),
            f"{run_path}.generation_trace",
        )
        previous_after = after_value
        run_ids = run.get("capture_tensor_ids")
        expected_ids = _expected_capture_ids(expected_run, expected_path)
        if (
            not isinstance(run_ids, list)
            or len(run_ids) != len(set(run_ids))
            or set(run_ids) != expected_ids
            or any(tensor_id not in records for tensor_id in run_ids)
        ):
            _fail("INVALID_RUN_CAPTURE_TENSOR_IDS", run_path, repr(run_ids))
        archived_order = [
            tensor_id
            for tensor_id in records
            if tensor_id in expected_ids
        ]
        if run_ids != archived_order:
            _fail("RUN_CAPTURE_ORDER_MISMATCH", run_path, repr(run_ids))
        events = run.get("capture_events")
        if not isinstance(events, list) or len(events) != len(
            EXPECTED_COMMON_EVENT_KEYS
        ):
            _fail("INVALID_COMMON_CAPTURE_EVENTS", run_path, repr(events))
        actual_keys: list[tuple[str, str]] = []
        for event_index, raw_event in enumerate(events):
            event_path = f"{run_path}.capture_events[{event_index}]"
            event = _require_mapping(raw_event, event_path)
            if set(event) != set(_CAPTURE_EVENT_KEYS):
                _fail("INVALID_COMMON_CAPTURE_EVENT_FIELDS", event_path, repr(event))
            module_name = event.get("module_name")
            io_kind = event.get("io_kind")
            tensor_id = event.get("tensor_id")
            if not all(
                isinstance(value, str)
                for value in (module_name, io_kind, tensor_id)
            ):
                _fail("INVALID_COMMON_CAPTURE_EVENT", event_path, repr(event))
            module_name_value = cast(str, module_name)
            io_kind_value = cast(str, io_kind)
            tensor_id_value = cast(str, tensor_id)
            actual_keys.append((module_name_value, io_kind_value))
            expected_id = (
                f"{expected_run}|common|{module_name_value}|{io_kind_value}"
            )
            if (
                event.get("event_index") != event_index
                or event.get("occurrence_index") != 0
                or event.get("call_index") != TARGET_STEP_INDEX
                or event.get("generation_step_index") != TARGET_STEP_INDEX
                or tensor_id_value != expected_id
            ):
                _fail("INVALID_COMMON_CAPTURE_EVENT", event_path, repr(event))
            record = records[tensor_id_value]
            if (
                record["event_scope"] != "common"
                or record["event_index"] != event_index
                or record["module_name"] != module_name_value
                or record["module_type"] != event.get("module_type")
                or record["occurrence_index"] != 0
                or record["io_kind"] != io_kind_value
                or record["tensor_path"] != event.get("tensor_path")
                or record["semantic_key"]
                != f"common|{module_name_value}|{io_kind_value}"
            ):
                _fail(
                    "COMMON_CAPTURE_RECORD_LINK_MISMATCH",
                    event_path,
                    tensor_id_value,
                )
        if tuple(actual_keys) != EXPECTED_COMMON_EVENT_KEYS:
            _fail("COMMON_CAPTURE_SEQUENCE_MISMATCH", run_path, repr(actual_keys))
        semantic_sequence = [
            {
                "event_index": event["event_index"],
                "module_name": event["module_name"],
                "occurrence_index": event["occurrence_index"],
                "call_index": event["call_index"],
                "generation_step_index": event["generation_step_index"],
                "io_kind": event["io_kind"],
                "tensor_path": event["tensor_path"],
            }
            for event in events
        ]
        if (
            run.get("capture_event_count") != len(events)
            or run.get("capture_event_sequence_sha256")
            != _sha256(_canonical_json(semantic_sequence))
            or run.get("common_capture_plan_passed") is not True
            or run.get("capture_plan_passed") is not True
        ):
            _fail("COMMON_CAPTURE_RUN_AUDIT_MISMATCH", run_path, repr(run))
        _validate_run_operation_events(run, records, run_path)
        capture_ids.update(run_ids)
        runs[expected_run] = dict(run)
    weight_ids = _expected_weight_ids(evidence)
    if set(records) != capture_ids | weight_ids:
        _fail(
            "ARCHIVE_RECORD_CLOSURE_MISMATCH",
            "$.tensor_archive.records",
            repr(sorted(set(records) ^ (capture_ids | weight_ids))),
        )
    return runs


def _validate_run_operation_events(
    run: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    path: str,
) -> None:
    events = run.get("operation_graph_events")
    if not isinstance(events, list):
        _fail("INVALID_OPERATION_EVENTS", path, repr(events))
    expected = (
        EXPECTED_ATTACHED_OPERATION_SEQUENCE
        if run["path"] == ATTACHED_PATH
        else ()
    )
    if tuple(event.get("operation") for event in events) != expected:
        _fail("OPERATION_EVENT_SEQUENCE_MISMATCH", path, repr(events))
    for event_index, raw_event in enumerate(events):
        event_path = f"{path}.operation_graph_events[{event_index}]"
        event = _require_mapping(raw_event, event_path)
        if set(event) != set(_OPERATION_EVENT_KEYS):
            _fail("INVALID_OPERATION_EVENT_FIELDS", event_path, repr(event))
        operation = event["operation"]
        tensor_id = f"{run['run_id']}|diagnostic|q_proj|{operation}"
        if (
            event["event_index"] != event_index
            or event["call_index"] != TARGET_STEP_INDEX
            or event["generation_step_index"] != TARGET_STEP_INDEX
            or event["tensor_id"] != tensor_id
        ):
            _fail("INVALID_OPERATION_EVENT", event_path, repr(event))
        record = records[tensor_id]
        if (
            record["event_scope"] != "operation_graph"
            or record["event_index"] != event_index
            or record["module_type"] != event["module_type"]
            or record["semantic_key"] != f"diagnostic|q_proj|{operation}"
        ):
            _fail("OPERATION_EVENT_RECORD_LINK_MISMATCH", event_path, tensor_id)
    if run.get("operation_graph_plan_passed") is not True:
        _fail("OPERATION_GRAPH_RUN_NOT_PASSED", path, repr(run))


def _expected_capture_ids(run_id: str, path: str) -> set[str]:
    result = {
        f"{run_id}|common|{module}|{io_kind}"
        for module, io_kind in EXPECTED_COMMON_EVENT_KEYS
    }
    result.update(
        {
            f"{run_id}|generation|scores|step{TARGET_STEP_INDEX}",
            f"{run_id}|generation|raw_logits|step{TARGET_STEP_INDEX}",
        }
    )
    if path == ATTACHED_PATH:
        result.update(
            f"{run_id}|diagnostic|q_proj|{label}"
            for label in (
                *EXPECTED_ATTACHED_OPERATION_SEQUENCE,
                *ATTACHED_REPLAY_LABELS,
            )
        )
    else:
        result.add(f"{run_id}|diagnostic|q_proj|recomputed")
    return result


def _expected_weight_ids(evidence: Mapping[str, Any]) -> set[str]:
    audit = _require_mapping(
        evidence.get("weight_materialization"),
        "$.weight_materialization",
    )
    tensor_ids = _require_mapping(
        audit.get("tensor_ids"),
        "$.weight_materialization.tensor_ids",
    )
    if tensor_ids != WEIGHT_TENSOR_IDS:
        _fail(
            "WEIGHT_TENSOR_ID_MAP_MISMATCH",
            "$.weight_materialization.tensor_ids",
            repr(tensor_ids),
        )
    return set(WEIGHT_TENSOR_IDS.values())


def _recompute_comparison(
    name: str,
    left_id: str,
    right_id: str,
    records: Mapping[str, Mapping[str, Any]],
    slices: Mapping[str, memoryview],
) -> dict[str, Any]:
    try:
        left_record = records[left_id]
        right_record = records[right_id]
        left_raw = slices[left_id]
        right_raw = slices[right_id]
    except KeyError as exc:
        _fail("COMPARISON_TENSOR_MISSING", f"$.comparisons.{name}", str(exc))
    if left_record["shape"] != right_record["shape"]:
        _fail(
            "COMPARISON_SHAPE_MISMATCH",
            f"$.comparisons.{name}",
            repr((left_record["shape"], right_record["shape"])),
        )
    left_values = [value[0] for value in struct.iter_unpack("<f", left_raw)]
    right_values = [value[0] for value in struct.iter_unpack("<f", right_raw)]
    deltas = [
        abs(left - right)
        for left, right in zip(left_values, right_values, strict=True)
    ]
    different_indices = [
        index
        for index, (left, right) in enumerate(
            zip(left_values, right_values, strict=True)
        )
        if left != right
    ]
    bitwise_different = sum(
        left[0] != right[0]
        for left, right in zip(
            struct.iter_unpack("<I", left_raw),
            struct.iter_unpack("<I", right_raw),
            strict=True,
        )
    )
    maximum = max(deltas, default=0.0)
    first_index: int | None
    max_index: int | None
    if different_indices:
        first_value_index = different_indices[0]
        max_value_index = deltas.index(maximum)
        first_index = first_value_index
        max_index = max_value_index
        left_first: float | None = left_values[first_value_index]
        right_first: float | None = right_values[first_value_index]
        left_max: float | None = left_values[max_value_index]
        right_max: float | None = right_values[max_value_index]
    else:
        first_index = None
        max_index = None
        left_first = None
        right_first = None
        left_max = None
        right_max = None
    mean = math.fsum(deltas) / len(deltas)
    rms = math.sqrt(
        math.fsum(delta * delta for delta in deltas) / len(deltas)
    )
    return {
        "name": name,
        "shape": left_record["shape"],
        "dtype": "float32",
        "elements": len(deltas),
        "numerically_equal": not different_indices,
        "bitwise_equal": bitwise_different == 0,
        "different_elements": len(different_indices),
        "bitwise_different_elements": bitwise_different,
        "first_different_flat_index": first_index,
        "max_abs_delta_flat_index": max_index,
        "left_value_at_first_difference": left_first,
        "right_value_at_first_difference": right_first,
        "left_value_at_max_abs_delta": left_max,
        "right_value_at_max_abs_delta": right_max,
        "max_abs_delta": maximum,
        "mean_abs_delta": mean,
        "root_mean_square_delta": rms,
        "left_tensor_id": left_id,
        "right_tensor_id": right_id,
        "left_raw_sha256": left_record["raw_payload_sha256"],
        "right_raw_sha256": right_record["raw_payload_sha256"],
        "left_canonical_sha256": left_record["canonical_value_sha256"],
        "right_canonical_sha256": right_record["canonical_value_sha256"],
    }


def _recompute_weight_audit(
    evidence: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    slices: Mapping[str, memoryview],
) -> dict[str, Any]:
    stored = _require_mapping(
        evidence.get("weight_materialization"),
        "$.weight_materialization",
    )
    tensor_ids = stored["tensor_ids"]
    if tensor_ids != WEIGHT_TENSOR_IDS:
        _fail(
            "WEIGHT_TENSOR_ID_MAP_MISMATCH",
            "$.weight_materialization.tensor_ids",
            repr(tensor_ids),
        )
    base_record = records[WEIGHT_TENSOR_IDS["base_weight"]]
    delta_record = records[WEIGHT_TENSOR_IDS["delta_weight"]]
    expected_record = records[WEIGHT_TENSOR_IDS["expected_merged_weight"]]
    actual_record = records[WEIGHT_TENSOR_IDS["actual_merged_weight"]]
    shape = [1536, 1536]
    if any(
        record["shape"] != shape
        or record["event_scope"] != "weight_materialization"
        or record["io_kind"] != "weight"
        or record["module_name"] != FIRST_REGISTERED_BOUNDARY
        for record in (base_record, delta_record, expected_record, actual_record)
    ):
        _fail(
            "INVALID_WEIGHT_ARCHIVE_RECORD",
            "$.weight_materialization",
            repr(tensor_ids),
        )
    base_raw = slices[WEIGHT_TENSOR_IDS["base_weight"]]
    delta_raw = slices[WEIGHT_TENSOR_IDS["delta_weight"]]
    expected_raw = slices[WEIGHT_TENSOR_IDS["expected_merged_weight"]]
    actual_raw = slices[WEIGHT_TENSOR_IDS["actual_merged_weight"]]
    ideal_nonzero = 0
    effective_changed = 0
    rounded_to_base = 0
    reconstruction_mismatch = 0
    max_error = 0.0
    for base_item, delta_item, expected_item in zip(
        struct.iter_unpack("<f", base_raw),
        struct.iter_unpack("<f", delta_raw),
        struct.iter_unpack("<f", expected_raw),
        strict=True,
    ):
        base_value = base_item[0]
        delta_value = delta_item[0]
        expected_value = expected_item[0]
        reconstructed = _round_float32(base_value + delta_value)
        reconstruction_mismatch += reconstructed != expected_value
        nonzero_update = delta_value != 0.0
        changed = expected_value != base_value
        ideal_nonzero += nonzero_update
        effective_changed += changed
        rounded_to_base += nonzero_update and not changed
        max_error = max(
            max_error,
            abs((expected_value - base_value) - delta_value),
        )
    if reconstruction_mismatch:
        _fail(
            "WEIGHT_MATERIALIZATION_RECONSTRUCTION_MISMATCH",
            "$.weight_materialization",
            repr(reconstruction_mismatch),
        )
    elements = math.prod(shape)
    mean_error = math.fsum(
        abs((expected_item[0] - base_item[0]) - delta_item[0])
        for base_item, delta_item, expected_item in zip(
            struct.iter_unpack("<f", base_raw),
            struct.iter_unpack("<f", delta_raw),
            struct.iter_unpack("<f", expected_raw),
            strict=True,
        )
    ) / elements
    actual_mismatched = _numeric_mismatch_count(expected_raw, actual_raw)
    bias_ids = (
        WEIGHT_TENSOR_IDS["attached_bias"],
        WEIGHT_TENSOR_IDS["merged_bias"],
    )
    attached_bias_record = records[bias_ids[0]]
    merged_bias_record = records[bias_ids[1]]
    if any(
        record["shape"] != [1536]
        or record["event_scope"] != "weight_materialization"
        or record["io_kind"] != "bias"
        or record["module_name"] != FIRST_REGISTERED_BOUNDARY
        for record in (attached_bias_record, merged_bias_record)
    ):
        _fail(
            "INVALID_BIAS_ARCHIVE_RECORD",
            "$.weight_materialization",
            repr(bias_ids),
        )
    bias_mismatched = _numeric_mismatch_count(
        slices[bias_ids[0]],
        slices[bias_ids[1]],
    )
    expected_digest = expected_record["raw_payload_sha256"]
    actual_digest = actual_record["raw_payload_sha256"]
    recomputed = {
        "name": FIRST_REGISTERED_BOUNDARY,
        "shape": shape,
        "dtype": "float32",
        "elements": elements,
        "base_weight_sha256": base_record["raw_payload_sha256"],
        "delta_weight_sha256": delta_record["raw_payload_sha256"],
        "expected_merged_weight_sha256": expected_digest,
        "actual_merged_weight_sha256": actual_digest,
        "expected_actual_equal": (
            actual_mismatched == 0 and expected_digest == actual_digest
        ),
        "actual_merged_mismatched_weights": actual_mismatched,
        "ideal_nonzero_updates": ideal_nonzero,
        "effective_changed_weights": effective_changed,
        "ideal_nonzero_updates_rounded_to_base": rounded_to_base,
        "max_abs_materialization_error": max_error,
        "mean_abs_materialization_error": mean_error,
        "bias_present": True,
        "bias_elements": 1536,
        "bias_mismatched_elements": bias_mismatched,
        "tensor_ids": dict(WEIGHT_TENSOR_IDS),
    }
    _require_exact(
        stored,
        recomputed,
        "RAW_WEIGHT_AUDIT_MISMATCH",
        "$.weight_materialization",
    )
    return recomputed


def _recompute_repeat_stability(
    runs: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in (ATTACHED_PATH, MERGED_PATH):
        path_runs = sorted(
            (run for run in runs.values() if run["path"] == path),
            key=lambda item: item["repeat"],
        )
        if len(path_runs) != 2:
            _fail("INVALID_REPEAT_RUN_SET", "$.runs", path)
        first, second = path_runs
        first_records = {
            records[tensor_id]["semantic_key"]: records[tensor_id]
            for tensor_id in first["capture_tensor_ids"]
        }
        second_records = {
            records[tensor_id]["semantic_key"]: records[tensor_id]
            for tensor_id in second["capture_tensor_ids"]
        }
        semantic_set_identity = set(first_records) == set(second_records)
        tensor_identity = semantic_set_identity and all(
            first_records[key]["raw_payload_sha256"]
            == second_records[key]["raw_payload_sha256"]
            and first_records[key]["native_dtype"]
            == second_records[key]["native_dtype"]
            and first_records[key]["native_shape"]
            == second_records[key]["native_shape"]
            and first_records[key]["native_stride"]
            == second_records[key]["native_stride"]
            for key in first_records
        )
        path_result = {
            "run_ids": [first["run_id"], second["run_id"]],
            "token_identity": (
                first["generated_token_ids"] == second["generated_token_ids"]
            ),
            "output_identity": first["output_sha256"] == second["output_sha256"],
            "score_trace_identity": (
                first["generation_trace"]["scores"]["trace_sha256"]
                == second["generation_trace"]["scores"]["trace_sha256"]
            ),
            "raw_logit_trace_identity": (
                first["generation_trace"]["raw_logits"]["trace_sha256"]
                == second["generation_trace"]["raw_logits"]["trace_sha256"]
            ),
            "target_alignment_identity": (
                first["target_alignment"] == second["target_alignment"]
            ),
            "event_sequence_identity": (
                first["capture_event_sequence_sha256"]
                == second["capture_event_sequence_sha256"]
            ),
            "semantic_tensor_set_identity": semantic_set_identity,
            "all_captured_tensors_bitwise_identical": tensor_identity,
        }
        path_result["passed"] = all(
            value
            for key, value in path_result.items()
            if key not in {"run_ids", "passed"}
        )
        result[path] = path_result
    result["passed"] = all(result[path]["passed"] for path in result)
    return result


def _validate_generation_links(
    runs: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    slices: Mapping[str, memoryview],
) -> None:
    for run_id, run in runs.items():
        lm_id = f"{run_id}|common|lm_head|output"
        raw_id = f"{run_id}|generation|raw_logits|step{TARGET_STEP_INDEX}"
        score_id = f"{run_id}|generation|scores|step{TARGET_STEP_INDEX}"
        raw_trace = run["generation_trace"]["raw_logits"]
        score_trace = run["generation_trace"]["scores"]
        if (
            slices[lm_id].tobytes() != slices[raw_id].tobytes()
            or records[raw_id]["raw_payload_sha256"]
            != raw_trace["comparison_step_vector_sha256"]
            or records[score_id]["raw_payload_sha256"]
            != score_trace["comparison_step_vector_sha256"]
            or raw_trace["comparison_step_index"] != TARGET_STEP_INDEX
            or score_trace["comparison_step_index"] != TARGET_STEP_INDEX
            or run.get("lm_head_raw_logit_linked") is not True
        ):
            _fail("GENERATION_TENSOR_LINK_MISMATCH", f"$.runs.{run_id}", run_id)


def _validate_operation_graph(
    evidence: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    slices: Mapping[str, memoryview],
) -> None:
    expected_hyperparameters = {
        "rank": 16,
        "alpha": 32.0,
        "scaling": 2.0,
        "dropout_probability": 0.05,
    }
    run_sequences = {
        run_id: [event["operation"] for event in run["operation_graph_events"]]
        for run_id, run in runs.items()
    }
    recomputed_audit = {
        "expected_attached_sequence": list(
            EXPECTED_ATTACHED_OPERATION_SEQUENCE
        ),
        "run_sequences": run_sequences,
        "attached_sequence_identity": all(
            sequence == list(EXPECTED_ATTACHED_OPERATION_SEQUENCE)
            for run_id, sequence in run_sequences.items()
            if RUN_PATHS[run_id] == ATTACHED_PATH
        ),
        "merged_sequences_empty": all(
            sequence == []
            for run_id, sequence in run_sequences.items()
            if RUN_PATHS[run_id] == MERGED_PATH
        ),
        "adapter_hyperparameters": expected_hyperparameters,
        "attached_hyperparameters_identical": True,
        "run_audits_passed": all(
            run["operation_graph_plan_passed"] for run in runs.values()
        ),
    }
    recomputed_audit["passed"] = all(
        recomputed_audit[key]
        for key in (
            "attached_sequence_identity",
            "merged_sequences_empty",
            "attached_hyperparameters_identical",
            "run_audits_passed",
        )
    )
    _require_exact(
        evidence.get("operation_graph_audit"),
        recomputed_audit,
        "OPERATION_GRAPH_AUDIT_MISMATCH",
        "$.operation_graph_audit",
    )
    attached = ATTACHED_R1
    _require_float32_binary_operation(
        slices[f"{attached}|diagnostic|q_proj|lora_b_output"],
        None,
        slices[f"{attached}|diagnostic|q_proj|factorized_scaled"],
        "multiply_scalar",
        expected_hyperparameters["scaling"],
    )
    _require_float32_binary_operation(
        slices[f"{attached}|diagnostic|q_proj|base_layer"],
        slices[f"{attached}|diagnostic|q_proj|factorized_scaled"],
        slices[f"{attached}|diagnostic|q_proj|base_plus_factorized"],
        "add",
        None,
    )
    _require_float32_binary_operation(
        slices[f"{attached}|diagnostic|q_proj|base_layer"],
        slices[f"{attached}|diagnostic|q_proj|delta_weight_linear"],
        slices[f"{attached}|diagnostic|q_proj|base_plus_delta_weight_linear"],
        "add",
        None,
    )
    if records[f"{attached}|diagnostic|q_proj|dropout_output"][
        "raw_payload_sha256"
    ] != records[f"{attached}|common|{FIRST_REGISTERED_BOUNDARY}|input"][
        "raw_payload_sha256"
    ]:
        _fail("DROPOUT_IDENTITY_MISMATCH", "$.operation_graph_audit", attached)


def _require_float32_binary_operation(
    left: memoryview,
    right: memoryview | None,
    expected: memoryview,
    operation: str,
    scalar: float | None,
) -> None:
    right_values = (
        None if right is None else struct.iter_unpack("<f", right)
    )
    expected_values = struct.iter_unpack("<f", expected)
    if operation == "multiply_scalar":
        if scalar is None:
            _fail("INVALID_REPLAY_OPERATION", "$.operation_graph_audit", operation)
        mismatch = sum(
            _round_float32(left_item[0] * scalar) != expected_item[0]
            for left_item, expected_item in zip(
                struct.iter_unpack("<f", left),
                expected_values,
                strict=True,
            )
        )
    elif operation == "add" and right_values is not None:
        mismatch = sum(
            _round_float32(left_item[0] + right_item[0]) != expected_item[0]
            for left_item, right_item, expected_item in zip(
                struct.iter_unpack("<f", left),
                right_values,
                expected_values,
                strict=True,
            )
        )
    else:
        _fail("INVALID_REPLAY_OPERATION", "$.operation_graph_audit", operation)
    if mismatch:
        _fail(
            "RAW_REPLAY_RECONSTRUCTION_MISMATCH",
            "$.operation_graph_audit",
            repr((operation, mismatch)),
        )


def _canonical_float32_sha256(raw: bytes) -> str:
    words = array("I")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    for index, word in enumerate(words):
        if word == 0x80000000:
            words[index] = 0
    if sys.byteorder != "little":
        words.byteswap()
    return _sha256(words.tobytes())


def _all_finite_float32(raw: bytes) -> bool:
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return all(math.isfinite(value) for value in values)


def _numeric_mismatch_count(left: memoryview, right: memoryview) -> int:
    if len(left) != len(right):
        _fail("FLOAT32_LENGTH_MISMATCH", "$", repr((len(left), len(right))))
    return sum(
        left_item[0] != right_item[0]
        for left_item, right_item in zip(
            struct.iter_unpack("<f", left),
            struct.iter_unpack("<f", right),
            strict=True,
        )
    )


def _round_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _validate_shape(value: object, path: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not _strict_int(item) or item <= 0 for item in value)
    ):
        _fail("INVALID_ARCHIVE_TENSOR_SHAPE", path, repr(value))
    return value


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", path, repr(type(value)))
    return value


def _require_exact(
    actual: object,
    expected: object,
    code: str,
    path: str,
) -> None:
    if actual != expected:
        _fail(code, path, repr(actual))


def _sha256(payload: bytes | memoryview) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bound_sha256(header: bytes, payload: memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "ARCHIVE_BASENAME",
    "validate_frozen_numerics_evidence",
    "validate_raw_tensor_archive",
]
