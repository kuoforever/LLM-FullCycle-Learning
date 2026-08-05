"""Locate the FP32 attached-versus-merged projection execution boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from peft import PeftModel  # type: ignore[import-not-found]
from transformers import AutoTokenizer  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_fp32_attached_merge_numerics import (  # noqa: E402
    COMMON_OUTPUT_STAGES,
    FIRST_REGISTERED_BOUNDARY,
    FP32_ATTACHED_MERGE_NUMERICS_VERSION,
    analyze_module_comparisons,
    classify_operation_order,
)
from fullcycle_bridge.tool_router_merge_remediation import (  # noqa: E402
    token_ids_sha256,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)
from scripts import (  # noqa: E402
    probe_tool_router_bf16_merge_remediation as remediation_probe,
)
from scripts import (  # noqa: E402
    probe_tool_router_fp32_attached_merge_isolation as isolation_probe,
)
from scripts import probe_tool_router_fp32_merge_drift as drift_probe  # noqa: E402

ATTACHED_PATH = isolation_probe.ATTACHED_PATH
MERGED_PATH = isolation_probe.MERGED_PATH
RUN_PLAN = (
    (ATTACHED_PATH, 1, "fp32-attached-numerics-r1"),
    (MERGED_PATH, 1, "fp32-safe-merged-numerics-r1"),
    (MERGED_PATH, 2, "fp32-safe-merged-numerics-r2"),
    (ATTACHED_PATH, 2, "fp32-attached-numerics-r2"),
)
REPRESENTATIVE_RUN = {
    ATTACHED_PATH: "fp32-attached-numerics-r1",
    MERGED_PATH: "fp32-safe-merged-numerics-r1",
}
TARGET_STEP_INDEX = 45
TARGET_INPUT_TOKEN_ID = 788
TARGET_EMITTED_TOKEN_ID = 3849
TARGET_CACHE_POSITION = 383
MAX_RESIDUAL_CUDA_BYTES = isolation_probe.MAX_RESIDUAL_CUDA_BYTES
EXPECTED_ISOLATION_SHA256 = (
    "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26"
)
Q_PROJ = FIRST_REGISTERED_BOUNDARY
COMMON_STAGES = COMMON_OUTPUT_STAGES
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


class _TensorArchive:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.records: list[dict[str, Any]] = []
        self.tensors: dict[str, Any] = {}
        self._by_id: dict[str, dict[str, Any]] = {}

    def add(self, tensor_id: str, tensor: Any, metadata: dict[str, Any]) -> None:
        if tensor_id in self._by_id:
            raise RuntimeError(f"duplicate tensor id: {tensor_id}")
        native = tensor.detach().contiguous()
        native_dtype = str(native.dtype).removeprefix("torch.")
        native_shape = list(native.shape)
        native_stride = list(native.stride())
        comparison = native.to(dtype=torch.float32).contiguous().cpu()
        if not bool(torch.isfinite(comparison).all()):
            raise RuntimeError(f"non-finite tensor capture: {tensor_id}")
        raw = comparison.view(torch.uint8).numpy().tobytes()
        canonical = comparison.clone()
        canonical[canonical == 0] = 0.0
        canonical_raw = canonical.view(torch.uint8).numpy().tobytes()
        offset = len(self.payload)
        self.payload.extend(raw)
        record = {
            "tensor_id": tensor_id,
            "run_id": metadata["run_id"],
            "path": metadata["path"],
            "semantic_key": metadata["semantic_key"],
            "event_scope": metadata["event_scope"],
            "event_index": metadata.get("event_index"),
            "module_name": metadata["module_name"],
            "module_type": metadata["module_type"],
            "occurrence_index": metadata.get("occurrence_index"),
            "io_kind": metadata["io_kind"],
            "tensor_path": metadata["tensor_path"],
            "native_dtype": native_dtype,
            "native_shape": native_shape,
            "native_stride": native_stride,
            "native_layout": str(native.layout).removeprefix("torch."),
            "comparison_dtype": "float32",
            "shape": list(comparison.shape),
            "elements": comparison.numel(),
            "all_finite": True,
            "byte_offset": offset,
            "byte_length": len(raw),
            "raw_payload_sha256": _sha256(raw),
            "canonical_value_sha256": _sha256(canonical_raw),
        }
        header = _canonical_json(record)
        record["bound_sha256"] = _bound_sha256(header, raw)
        self.records.append(record)
        self._by_id[tensor_id] = record
        self.tensors[tensor_id] = comparison

    def record(self, tensor_id: str) -> dict[str, Any]:
        try:
            return self._by_id[tensor_id]
        except KeyError as exc:
            raise RuntimeError(f"tensor record missing: {tensor_id}") from exc

    def descriptor(self, path: Path) -> dict[str, Any]:
        payload = bytes(self.payload)
        return {
            "version": 1,
            "path": path.name,
            "encoding": "contiguous_ieee754_float32_little_endian",
            "byte_count": len(payload),
            "sha256": _sha256(payload),
            "record_count": len(self.records),
            "records": self.records,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--numerics-evidence", type=Path, required=True)
    parser.add_argument("--remediation-evidence", type=Path, required=True)
    parser.add_argument("--drift-evidence", type=Path, required=True)
    parser.add_argument("--isolation-evidence", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("tensor archive requires a little-endian host")
    for output in (args.output, args.tensor_output):
        drift_probe._verify_output_boundary(  # noqa: SLF001
            output,
            args.model_dir,
            args.adapter_dir,
        )
    if args.output.resolve() == args.tensor_output.resolve():
        raise RuntimeError("JSON and tensor outputs must be distinct")
    if args.output.parent.resolve() != args.tensor_output.parent.resolve():
        raise RuntimeError("JSON and tensor outputs must be sibling artifacts")

    config = isolation_probe._load_json(args.config)  # noqa: SLF001
    training = isolation_probe._load_json(args.training_evidence)  # noqa: SLF001
    stability = isolation_probe._load_json(args.stability_evidence)  # noqa: SLF001
    numerics = isolation_probe._load_json(args.numerics_evidence)  # noqa: SLF001
    remediation = isolation_probe._load_json(  # noqa: SLF001
        args.remediation_evidence
    )
    drift = isolation_probe._load_json(args.drift_evidence)  # noqa: SLF001
    isolation = isolation_probe._load_json(args.isolation_evidence)  # noqa: SLF001
    _verify_sources(
        config,
        training,
        stability,
        numerics,
        remediation,
        drift,
        isolation,
        args,
    )
    isolation_probe._verify_environment(config)  # noqa: SLF001

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    record = evaluation[0]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    prompt_path = ROOT / config["prompt"]["path"]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": render_user_payload(record)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded_cpu = tokenizer(rendered, return_tensors="pt")
    input_token_ids = [int(value) for value in encoded_cpu["input_ids"][0]]
    input_digest = token_ids_sha256(input_token_ids)
    if (
        len(input_token_ids) != isolation["input_token_count"]
        or input_digest != isolation["input_token_ids_sha256"]
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")

    storage_audit = {
        "base_checkpoint": drift_probe._safetensor_storage_audit(  # noqa: SLF001
            args.model_dir / config["model"]["weight_file"]
        ),
        "adapter": drift_probe._safetensor_storage_audit(  # noqa: SLF001
            args.adapter_dir / "adapter_model.safetensors"
        ),
    }
    if storage_audit != isolation["storage_audit"]:
        raise RuntimeError("source storage audit drift")

    archive = _TensorArchive()
    runs: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for order_index, (path, repeat, run_id) in enumerate(RUN_PLAN):
        run_started = time.perf_counter()
        run, snapshot = _run_instrumented_path(
            path=path,
            repeat=repeat,
            run_id=run_id,
            order_index=order_index,
            model_dir=args.model_dir,
            adapter_dir=args.adapter_dir,
            config=config,
            encoded_cpu=encoded_cpu,
            tokenizer=tokenizer,
            isolation=isolation,
            archive=archive,
        )
        run["elapsed_seconds"] = time.perf_counter() - run_started
        runs.append(run)
        snapshots[run_id] = snapshot

    _verify_cross_run_event_sequences(runs)
    operation_graph_audit = _operation_graph_audit(runs, snapshots)
    if not operation_graph_audit["passed"]:
        raise RuntimeError(
            f"operation graph audit failed: {operation_graph_audit!r}"
        )
    repeat_stability = _instrumented_repeat_stability(runs, archive)
    if not repeat_stability["passed"]:
        raise RuntimeError(
            f"instrumented repeat stability failed: {repeat_stability!r}"
        )

    attached_run_id = REPRESENTATIVE_RUN[ATTACHED_PATH]
    merged_run_id = REPRESENTATIVE_RUN[MERGED_PATH]
    module_input_comparisons = [
        _compare_archive_tensors(
            f"{stage}.input",
            _common_id(attached_run_id, stage, "input"),
            _common_id(merged_run_id, stage, "input"),
            archive,
        )
        for stage in COMMON_STAGES
    ]
    module_output_comparisons = [
        _compare_archive_tensors(
            stage,
            _common_id(attached_run_id, stage, "output"),
            _common_id(merged_run_id, stage, "output"),
            archive,
        )
        for stage in COMMON_STAGES
    ]
    module_analysis = analyze_module_comparisons(module_output_comparisons)
    weight_materialization = _weight_materialization_audit(
        snapshots[attached_run_id],
        snapshots[merged_run_id],
        archive,
    )
    operation_comparisons = _operation_comparisons(archive)
    execution_form_boundary = classify_operation_order(
        module_output_comparisons,
        operation_comparisons,
        weight_materialization,
    )
    q_input = next(
        item
        for item in module_input_comparisons
        if item["name"] == f"{Q_PROJ}.input"
    )
    if not q_input["numerically_equal"] or not q_input["bitwise_equal"]:
        raise RuntimeError("first divergent q_proj input is not bitwise identical")

    run_by_id = {run["run_id"]: run for run in runs}
    generation_reproduction = {
        "all_runs_reproduce_frozen_paths": all(
            run["frozen_path_reproduced"] for run in runs
        ),
        "cross_path_token_identity": (
            run_by_id[attached_run_id]["generated_token_ids"]
            == run_by_id[merged_run_id]["generated_token_ids"]
        ),
        "cross_path_output_identity": (
            run_by_id[attached_run_id]["output_sha256"]
            == run_by_id[merged_run_id]["output_sha256"]
        ),
        "score_trace_identity": (
            run_by_id[attached_run_id]["generation_trace"]["scores"][
                "trace_sha256"
            ]
            == run_by_id[merged_run_id]["generation_trace"]["scores"][
                "trace_sha256"
            ]
        ),
        "raw_logit_trace_identity": (
            run_by_id[attached_run_id]["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
            == run_by_id[merged_run_id]["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
        ),
        "same_dtype_token_boundary": False,
    }
    if generation_reproduction != {
        "all_runs_reproduce_frozen_paths": True,
        "cross_path_token_identity": True,
        "cross_path_output_identity": True,
        "score_trace_identity": False,
        "raw_logit_trace_identity": False,
        "same_dtype_token_boundary": False,
    }:
        raise RuntimeError(
            f"generation reproduction contradiction: {generation_reproduction!r}"
        )

    materialization_axis = (
        "fp32_materialization_rounding_present"
        if weight_materialization["ideal_nonzero_updates_rounded_to_base"] > 0
        else "fp32_materialization_rounding_absent"
    )
    classification = execution_form_boundary["classification"]
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    source_adapter_unchanged = adapter_manifest == isolation["adapter_files"]
    eval_digest_unchanged = fixture_digest(evaluation) == isolation["eval_digest"]
    event_sequence_identity = len(
        {run["capture_event_sequence_sha256"] for run in runs}
    ) == 1
    all_target_alignment = all(run["target_alignment_passed"] for run in runs)
    all_lm_head_linked = all(run["lm_head_raw_logit_linked"] for run in runs)
    capture_plan_passed = all(run["capture_plan_passed"] for run in runs)
    acceptance = {
        "upstream_isolation_locked": True,
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": True,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": True,
        "frozen_input_reproduced": True,
        "all_instrumented_runs_reproduce_frozen_paths": generation_reproduction[
            "all_runs_reproduce_frozen_paths"
        ],
        "instrumented_repeat_stable": repeat_stability["passed"],
        "target_forward_aligned": all_target_alignment,
        "capture_plan_executed": capture_plan_passed,
        "paired_event_sequence_identical": event_sequence_identity,
        "lm_head_raw_logit_linked": all_lm_head_linked,
        "first_captured_output_divergence_located": (
            module_analysis["first_divergent_module"] == Q_PROJ
        ),
        "preceding_captured_outputs_identical": module_analysis[
            "preceding_modules_identical"
        ],
        "first_divergent_module_input_identical": q_input[
            "bitwise_equal"
        ],
        "operation_graph_executed": operation_graph_audit["passed"],
        "safe_merge_weight_reproduced": execution_form_boundary[
            "safe_merge_weight_reproduced"
        ],
        "registered_execution_form_boundary_quantified": (
            execution_form_boundary[
                "registered_execution_form_boundary_quantified"
            ]
        ),
        "cross_path_token_identity_preserved": generation_reproduction[
            "cross_path_token_identity"
        ],
        "bf16_context_only": True,
    }
    numerics_gate = {
        "path_reproduction": generation_reproduction[
            "all_runs_reproduce_frozen_paths"
        ],
        "instrumented_repeat_stability": repeat_stability["passed"],
        "target_forward_alignment": all_target_alignment,
        "first_paired_common_boundary_located": (
            module_analysis["first_divergent_module"] == Q_PROJ
        ),
        "registered_execution_form_boundary_quantified": (
            execution_form_boundary[
                "registered_execution_form_boundary_quantified"
            ]
        ),
        "passed": all(acceptance.values()),
    }
    if not numerics_gate["passed"]:
        raise RuntimeError(f"FP32 numerics acceptance failed: {acceptance!r}")

    tensor_archive = archive.descriptor(args.tensor_output)
    result = {
        "fp32_attached_merge_numerics_version": (
            FP32_ATTACHED_MERGE_NUMERICS_VERSION
        ),
        "experiment_id": "fc-mvp-001-fp32-attached-merge-numerics-v1",
        "source_experiment_id": isolation["experiment_id"],
        "source_lineage": {
            "isolation_evidence_sha256": file_sha256(args.isolation_evidence),
            "drift_evidence_sha256": file_sha256(args.drift_evidence),
            "remediation_evidence_sha256": file_sha256(
                args.remediation_evidence
            ),
            "stability_evidence_sha256": file_sha256(args.stability_evidence),
            "bf16_numerics_context_sha256": file_sha256(
                args.numerics_evidence
            ),
        },
        "training_lock_sha256": isolation["training_lock_sha256"],
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "model_weight_sha256": isolation["model_weight_sha256"],
        "prompt_sha256": isolation["prompt_sha256"],
        "eval_digest": isolation["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "environment": {
            **config["environment"],
            "base_and_adapter_runtime_dtype": "float32",
            "autocast": False,
            "tf32": False,
        },
        "protocol": _protocol(config, tokenizer),
        "frozen_path_references": _frozen_path_references(isolation),
        "frozen_bf16_context": isolation["frozen_bf16_context"],
        "runs": runs,
        "generation_reproduction": generation_reproduction,
        "instrumented_repeat_stability": repeat_stability,
        "capture_plan": _capture_plan(),
        "operation_graph_audit": operation_graph_audit,
        "tensor_archive": tensor_archive,
        "module_input_comparisons": module_input_comparisons,
        "module_output_comparisons": module_output_comparisons,
        "module_analysis": module_analysis,
        "operation_comparisons": operation_comparisons,
        "registered_execution_form_boundary": execution_form_boundary,
        "weight_materialization": weight_materialization,
        "materialization_axis": materialization_axis,
        "classification": classification,
        "causal_scope": _causal_scope(),
        "numerics_gate": numerics_gate,
        "remediation_gate": {
            "source_gate_passed": False,
            "new_remediation_tested": False,
            "passed": False,
        },
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(
            run["peak_gpu_memory_bytes"] for run in runs
        ),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": _constraints(),
        "locked_next_action": _locked_next_action(),
        "runtime_eligible": False,
        "runtime_eligibility_reason": classification,
        "offline": True,
    }
    isolation_probe._validate_result_scalars(result)  # noqa: SLF001
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_output.write_bytes(bytes(archive.payload))
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _run_instrumented_path(
    *,
    path: str,
    repeat: int,
    run_id: str,
    order_index: int,
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    encoded_cpu: Any,
    tokenizer: Any,
    isolation: dict[str, Any],
    archive: _TensorArchive,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} exceeded residual CUDA ceiling before load: "
            f"{allocated_before}"
        )
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    run: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    try:
        if path == ATTACHED_PATH:
            model, precision = isolation_probe._load_fp32_attached_model(  # noqa: SLF001
                model_dir,
                adapter_dir,
                config,
            )
        elif path == MERGED_PATH:
            model, precision = remediation_probe._load_fp32_merged_model(  # noqa: SLF001
                model_dir,
                adapter_dir,
                config,
            )
        else:
            raise RuntimeError(f"unknown numerics path: {path}")
        dropout_audit = isolation_probe._lora_dropout_audit(model)  # noqa: SLF001
        precision["lora_dropout"] = dropout_audit
        (
            token_ids,
            generation_trace,
            capture_events,
            operation_events,
            target_alignment,
            snapshot,
            tensor_ids,
            lm_head_linked,
        ) = _instrumented_generate(
            model=model,
            path=path,
            run_id=run_id,
            encoded_cpu=encoded_cpu,
            config=config,
            tokenizer=tokenizer,
            archive=archive,
        )
        precision["generation"] = {
            "score_dtypes": generation_trace["scores"]["native_dtypes"],
            "all_scores_float32": (
                generation_trace["scores"]["native_dtypes"] == ["float32"]
            ),
            "raw_logit_dtypes": generation_trace["raw_logits"][
                "native_dtypes"
            ],
            "all_raw_logits_float32": (
                generation_trace["raw_logits"]["native_dtypes"]
                == ["float32"]
            ),
            "dtype_semantics": "transformers_generate_return_tensor_dtype",
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        if path == ATTACHED_PATH:
            path_protocol = isolation_probe._attached_protocol_passed(  # noqa: SLF001
                precision
            )
        else:
            path_protocol = (
                remediation_probe._precision_protocol_passed(precision)  # noqa: SLF001
                and precision["generation"]["all_raw_logits_float32"] is True
                and precision["lora_dropout"]
                == {"modules": 0, "training_modules": 0}
            )
        if not path_protocol:
            raise RuntimeError(f"instrumented path protocol failed: {precision!r}")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        event_sequence = _semantic_event_sequence(capture_events)
        event_keys = tuple(
            (item["module_name"], item["io_kind"]) for item in capture_events
        )
        target_alignment_passed = _target_alignment_passed(target_alignment)
        common_capture_plan_passed = event_keys == EXPECTED_COMMON_EVENT_KEYS
        operation_graph_plan_passed = _run_operation_graph_passed(
            path,
            run_id,
            operation_events,
            snapshot,
        )
        capture_plan_passed = (
            common_capture_plan_passed and operation_graph_plan_passed
        )
        run = {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order_index,
            "fresh_load": True,
            "materialization_form": (
                "attached_factorized_lora"
                if path == ATTACHED_PATH
                else "materialized_safe_merge"
            ),
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": _sha256(decoded.encode("utf-8")),
            "generation_trace": generation_trace,
            "precision_audit": precision,
            "target_alignment": target_alignment,
            "target_alignment_passed": target_alignment_passed,
            "capture_events": capture_events,
            "operation_graph_events": operation_events,
            "capture_event_count": len(capture_events),
            "capture_event_sequence_sha256": _sha256(
                _canonical_json(event_sequence)
            ),
            "common_capture_plan_passed": common_capture_plan_passed,
            "operation_graph_plan_passed": operation_graph_plan_passed,
            "capture_plan_passed": capture_plan_passed,
            "capture_tensor_ids": tensor_ids,
            "lm_head_raw_logit_linked": lm_head_linked,
            "path_protocol_passed": path_protocol,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "memory_allocated_before_load_bytes": allocated_before,
        }
        _verify_frozen_path_reproduction(run, isolation)
        run["frozen_path_reproduced"] = True
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} retained model-scale CUDA memory: {allocated_after}"
        )
    if run is None or snapshot is None:
        raise RuntimeError(f"instrumented run did not complete: {run_id}")
    run["memory_allocated_after_release_bytes"] = allocated_after
    return run, snapshot


def _instrumented_generate(
    *,
    model: Any,
    path: str,
    run_id: str,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
    archive: _TensorArchive,
) -> tuple[
    list[int],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[str],
    bool,
]:
    causal = _causal_model(model)
    modules = dict(causal.named_modules())
    missing = [name for name in COMMON_STAGES if name not in modules]
    if missing:
        raise RuntimeError(f"capture modules missing: {missing!r}")
    q_module = modules[Q_PROJ]
    state: dict[str, Any] = {
        "call_index": -1,
        "active": False,
        "target_count": 0,
    }
    captured: dict[str, dict[str, Any]] = {}
    capture_events: list[dict[str, Any]] = []
    operation_events: list[dict[str, Any]] = []
    occurrences: dict[tuple[str, str], int] = {}
    alignment: dict[str, Any] = {}
    handles: list[Any] = []

    def stash(
        tensor_id: str,
        tensor: Any,
        *,
        semantic_key: str,
        event_scope: str,
        event_index: int | None,
        module_name: str,
        module_type: str,
        occurrence_index: int | None,
        io_kind: str,
        tensor_path: str,
    ) -> None:
        if tensor_id in captured:
            raise RuntimeError(f"duplicate target-step tensor: {tensor_id}")
        captured[tensor_id] = {
            "tensor": tensor.detach().clone(),
            "metadata": {
                "run_id": run_id,
                "path": path,
                "semantic_key": semantic_key,
                "event_scope": event_scope,
                "event_index": event_index,
                "module_name": module_name,
                "module_type": module_type,
                "occurrence_index": occurrence_index,
                "io_kind": io_kind,
                "tensor_path": tensor_path,
            },
        }

    def causal_pre_hook(_module: Any, args: Any, kwargs: Any) -> None:
        state["call_index"] += 1
        state["active"] = state["call_index"] == TARGET_STEP_INDEX
        if not state["active"]:
            return
        state["target_count"] += 1
        alignment.update(_target_alignment(args, kwargs, state["call_index"]))

    def causal_post_hook(_module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
        if state["active"]:
            state["active"] = False

    handles.append(causal.register_forward_pre_hook(causal_pre_hook, with_kwargs=True))
    handles.append(causal.register_forward_hook(causal_post_hook, with_kwargs=True))

    def make_pre_hook(stage: str) -> Any:
        def hook(module: Any, args: Any, kwargs: Any) -> None:
            if not state["active"]:
                return
            tensor, tensor_path = _module_input_tensor(args, kwargs)
            key = (stage, "input")
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            event_index = len(capture_events)
            tensor_id = _common_id(run_id, stage, "input")
            stash(
                tensor_id,
                tensor,
                semantic_key=f"common|{stage}|input",
                event_scope="common",
                event_index=event_index,
                module_name=stage,
                module_type=_module_type(module),
                occurrence_index=occurrence,
                io_kind="input",
                tensor_path=tensor_path,
            )
            capture_events.append(
                {
                    "event_index": event_index,
                    "module_name": stage,
                    "module_type": _module_type(module),
                    "occurrence_index": occurrence,
                    "call_index": state["call_index"],
                    "generation_step_index": TARGET_STEP_INDEX,
                    "io_kind": "input",
                    "tensor_path": tensor_path,
                    "tensor_id": tensor_id,
                }
            )

        return hook

    def make_post_hook(stage: str) -> Any:
        def hook(module: Any, _args: Any, _kwargs: Any, output: Any) -> None:
            if not state["active"]:
                return
            tensor, tensor_path = _first_tensor_leaf(output, "output")
            key = (stage, "output")
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            event_index = len(capture_events)
            tensor_id = _common_id(run_id, stage, "output")
            stash(
                tensor_id,
                tensor,
                semantic_key=f"common|{stage}|output",
                event_scope="common",
                event_index=event_index,
                module_name=stage,
                module_type=_module_type(module),
                occurrence_index=occurrence,
                io_kind="output",
                tensor_path=tensor_path,
            )
            capture_events.append(
                {
                    "event_index": event_index,
                    "module_name": stage,
                    "module_type": _module_type(module),
                    "occurrence_index": occurrence,
                    "call_index": state["call_index"],
                    "generation_step_index": TARGET_STEP_INDEX,
                    "io_kind": "output",
                    "tensor_path": tensor_path,
                    "tensor_id": tensor_id,
                }
            )

        return hook

    for stage in COMMON_STAGES:
        handles.append(
            modules[stage].register_forward_pre_hook(
                make_pre_hook(stage),
                with_kwargs=True,
            )
        )
        handles.append(
            modules[stage].register_forward_hook(
                make_post_hook(stage),
                with_kwargs=True,
            )
        )

    if path == ATTACHED_PATH:
        adapter_name = _active_adapter_name(q_module)
        diagnostic_modules = (
            ("base_layer", q_module.base_layer),
            ("dropout_output", q_module.lora_dropout[adapter_name]),
            ("lora_a_output", q_module.lora_A[adapter_name]),
            ("lora_b_output", q_module.lora_B[adapter_name]),
        )

        def make_diagnostic_hook(label: str) -> Any:
            def hook(module: Any, _args: Any, output: Any) -> None:
                if not state["active"]:
                    return
                tensor, tensor_path = _first_tensor_leaf(output, "output")
                event_index = len(operation_events)
                tensor_id = _diag_id(run_id, label)
                stash(
                    tensor_id,
                    tensor,
                    semantic_key=f"diagnostic|q_proj|{label}",
                    event_scope="operation_graph",
                    event_index=event_index,
                    module_name=f"{Q_PROJ}.{label}",
                    module_type=_module_type(module),
                    occurrence_index=0,
                    io_kind="output",
                    tensor_path=tensor_path,
                )
                operation_events.append(
                    {
                        "event_index": event_index,
                        "operation": label,
                        "module_type": _module_type(module),
                        "call_index": state["call_index"],
                        "generation_step_index": TARGET_STEP_INDEX,
                        "tensor_id": tensor_id,
                    }
                )

            return hook

        for label, module in diagnostic_modules:
            handles.append(module.register_forward_hook(make_diagnostic_hook(label)))

    try:
        token_ids, trace, generation_trace = drift_probe._generate_trace(  # noqa: SLF001
            model,
            encoded_cpu,
            config,
            tokenizer,
        )
    finally:
        for handle in handles:
            handle.remove()
    if state["target_count"] != 1:
        raise RuntimeError(f"target forward count mismatch: {state!r}")
    alignment["causal_forward_calls"] = state["call_index"] + 1

    score_vector = trace["generated.scores"][TARGET_STEP_INDEX]
    raw_vector = trace["generated.logits"][TARGET_STEP_INDEX]
    score_hash = drift_probe._float32_vector_sha256(score_vector)  # noqa: SLF001
    raw_hash = drift_probe._float32_vector_sha256(raw_vector)  # noqa: SLF001
    generation_trace["scores"]["comparison_step_index"] = TARGET_STEP_INDEX
    generation_trace["scores"]["comparison_step_vector_sha256"] = score_hash
    generation_trace["raw_logits"]["comparison_step_index"] = TARGET_STEP_INDEX
    generation_trace["raw_logits"]["comparison_step_vector_sha256"] = raw_hash
    _stash_generated_vector(captured, run_id, path, "scores", score_vector)
    _stash_generated_vector(captured, run_id, path, "raw_logits", raw_vector)

    if path == ATTACHED_PATH:
        snapshot = _attached_operation_replay(q_module, run_id, captured, stash)
    else:
        snapshot = _merged_operation_replay(q_module, run_id, captured, stash)

    tensor_ids: list[str] = []
    for tensor_id, value in captured.items():
        archive.add(tensor_id, value["tensor"], value["metadata"])
        tensor_ids.append(tensor_id)
    lm_record = archive.record(_common_id(run_id, "lm_head", "output"))
    raw_record = archive.record(_generation_id(run_id, "raw_logits"))
    lm_head_linked = (
        lm_record["elements"] == raw_record["elements"]
        and lm_record["raw_payload_sha256"] == raw_record["raw_payload_sha256"]
        and raw_record["raw_payload_sha256"] == raw_hash
    )
    if not lm_head_linked:
        raise RuntimeError(f"lm_head/raw-logit linkage failed: {run_id}")
    return (
        token_ids,
        generation_trace,
        capture_events,
        operation_events,
        alignment,
        snapshot,
        tensor_ids,
        lm_head_linked,
    )


def _attached_operation_replay(
    q_module: Any,
    run_id: str,
    captured: dict[str, dict[str, Any]],
    stash: Any,
) -> dict[str, Any]:
    adapter_name = _active_adapter_name(q_module)
    q_input = captured[_common_id(run_id, Q_PROJ, "input")]["tensor"]
    base_output = captured[_diag_id(run_id, "base_layer")]["tensor"]
    lora_b_output = captured[_diag_id(run_id, "lora_b_output")]["tensor"]
    scaling = float(q_module.scaling[adapter_name])
    with torch.inference_mode():
        factorized_scaled = lora_b_output * scaling
        base_plus_factorized = base_output + factorized_scaled
        delta_weight = q_module.get_delta_weight(adapter_name).detach().clone()
        delta_weight_linear = F.linear(q_input, delta_weight, None)
        base_plus_delta = base_output + delta_weight_linear
        base_weight = q_module.base_layer.weight.detach().clone()
        expected_weight = base_weight.clone()
        expected_weight += delta_weight
        bias = q_module.base_layer.bias
        expected_materialized = F.linear(q_input, expected_weight, bias)
    diagnostics = {
        "factorized_scaled": factorized_scaled,
        "base_plus_factorized": base_plus_factorized,
        "delta_weight_linear": delta_weight_linear,
        "base_plus_delta_weight_linear": base_plus_delta,
        "expected_materialized_linear": expected_materialized,
    }
    for label, tensor in diagnostics.items():
        stash(
            _diag_id(run_id, label),
            tensor,
            semantic_key=f"diagnostic|q_proj|{label}",
            event_scope="replay",
            event_index=None,
            module_name=Q_PROJ,
            module_type="torch.nn.functional",
            occurrence_index=None,
            io_kind="replay_output",
            tensor_path=label,
        )
    base_cpu = base_weight.float().cpu()
    delta_cpu = delta_weight.float().cpu()
    expected_cpu = expected_weight.float().cpu()
    return {
        "path": ATTACHED_PATH,
        "base_weight": base_cpu,
        "delta_weight": delta_cpu,
        "expected_weight": expected_cpu,
        "bias": None if bias is None else bias.detach().float().cpu(),
        "rank": int(q_module.r[adapter_name]),
        "alpha": float(q_module.lora_alpha[adapter_name]),
        "scaling": scaling,
        "dropout_probability": float(q_module.lora_dropout[adapter_name].p),
    }


def _merged_operation_replay(
    q_module: Any,
    run_id: str,
    captured: dict[str, dict[str, Any]],
    stash: Any,
) -> dict[str, Any]:
    q_input = captured[_common_id(run_id, Q_PROJ, "input")]["tensor"]
    with torch.inference_mode():
        recomputed = F.linear(q_input, q_module.weight, q_module.bias)
    stash(
        _diag_id(run_id, "recomputed"),
        recomputed,
        semantic_key="diagnostic|q_proj|recomputed",
        event_scope="replay",
        event_index=None,
        module_name=Q_PROJ,
        module_type="torch.nn.functional",
        occurrence_index=None,
        io_kind="replay_output",
        tensor_path="recomputed",
    )
    return {
        "path": MERGED_PATH,
        "actual_weight": q_module.weight.detach().float().cpu(),
        "bias": (
            None
            if q_module.bias is None
            else q_module.bias.detach().float().cpu()
        ),
    }


def _weight_materialization_audit(
    attached: dict[str, Any],
    merged: dict[str, Any],
    archive: _TensorArchive,
) -> dict[str, Any]:
    expected = attached["expected_weight"]
    actual = merged["actual_weight"]
    if list(expected.shape) != list(actual.shape):
        raise RuntimeError("q_proj merged weight shape mismatch")
    materialization = _weight_materialization_statistics(
        attached["base_weight"],
        attached["delta_weight"],
        expected,
    )
    mismatched = _float32_mismatched_elements(expected, actual)
    expected_digest = _float32_tensor_sha256(expected)
    actual_digest = _float32_tensor_sha256(actual)
    if (attached["bias"] is None) != (merged["bias"] is None):
        raise RuntimeError("q_proj bias presence changed across merge")
    attached_bias = attached["bias"]
    merged_bias = merged["bias"]
    bias_mismatched = (
        0
        if attached_bias is None
        else _float32_mismatched_elements(attached_bias, merged_bias)
    )
    if bias_mismatched:
        raise RuntimeError("q_proj bias changed across merge")
    attached_run = REPRESENTATIVE_RUN[ATTACHED_PATH]
    merged_run = REPRESENTATIVE_RUN[MERGED_PATH]
    tensor_ids: dict[str, str | None] = {
        "base_weight": _weight_id(attached_run, "base_weight"),
        "delta_weight": _weight_id(attached_run, "delta_weight"),
        "expected_merged_weight": _weight_id(
            attached_run,
            "expected_merged_weight",
        ),
        "actual_merged_weight": _weight_id(merged_run, "actual_merged_weight"),
        "attached_bias": (
            None
            if attached_bias is None
            else _weight_id(attached_run, "attached_bias")
        ),
        "merged_bias": (
            None
            if merged_bias is None
            else _weight_id(merged_run, "merged_bias")
        ),
    }
    for key, tensor, path, run_id in (
        ("base_weight", attached["base_weight"], ATTACHED_PATH, attached_run),
        ("delta_weight", attached["delta_weight"], ATTACHED_PATH, attached_run),
        ("expected_merged_weight", expected, ATTACHED_PATH, attached_run),
        ("actual_merged_weight", actual, MERGED_PATH, merged_run),
        ("attached_bias", attached_bias, ATTACHED_PATH, attached_run),
        ("merged_bias", merged_bias, MERGED_PATH, merged_run),
    ):
        tensor_id = tensor_ids[key]
        if tensor is None or tensor_id is None:
            continue
        archive.add(
            tensor_id,
            tensor,
            {
                "run_id": run_id,
                "path": path,
                "semantic_key": f"weight|q_proj|{key}",
                "event_scope": "weight_materialization",
                "event_index": None,
                "module_name": Q_PROJ,
                "module_type": "torch.Tensor",
                "occurrence_index": None,
                "io_kind": "bias" if key.endswith("bias") else "weight",
                "tensor_path": key,
            },
        )
    bias_present = attached_bias is not None
    return {
        "name": Q_PROJ,
        "shape": list(expected.shape),
        "dtype": "float32",
        "elements": expected.numel(),
        "base_weight_sha256": _float32_tensor_sha256(attached["base_weight"]),
        "delta_weight_sha256": _float32_tensor_sha256(attached["delta_weight"]),
        "expected_merged_weight_sha256": expected_digest,
        "actual_merged_weight_sha256": actual_digest,
        "expected_actual_equal": mismatched == 0 and expected_digest == actual_digest,
        "actual_merged_mismatched_weights": mismatched,
        **materialization,
        "bias_present": bias_present,
        "bias_elements": 0 if attached_bias is None else attached_bias.numel(),
        "bias_mismatched_elements": bias_mismatched,
        "tensor_ids": tensor_ids,
    }


def _operation_comparisons(archive: _TensorArchive) -> list[dict[str, Any]]:
    attached = REPRESENTATIVE_RUN[ATTACHED_PATH]
    merged = REPRESENTATIVE_RUN[MERGED_PATH]
    pairs = (
        (
            "q_proj_input_identity",
            _common_id(attached, Q_PROJ, "input"),
            _common_id(merged, Q_PROJ, "input"),
        ),
        (
            "attached_dropout_identity",
            _common_id(attached, Q_PROJ, "input"),
            _diag_id(attached, "dropout_output"),
        ),
        (
            "attached_output_reconstruction",
            _common_id(attached, Q_PROJ, "output"),
            _diag_id(attached, "base_plus_factorized"),
        ),
        (
            "merged_output_reconstruction",
            _common_id(merged, Q_PROJ, "output"),
            _diag_id(merged, "recomputed"),
        ),
        (
            "expected_materialized_vs_merged_actual",
            _diag_id(attached, "expected_materialized_linear"),
            _common_id(merged, Q_PROJ, "output"),
        ),
        (
            "factorized_lora_vs_delta_weight_linear",
            _diag_id(attached, "factorized_scaled"),
            _diag_id(attached, "delta_weight_linear"),
        ),
        (
            "attached_factorized_output_vs_split_delta_output",
            _diag_id(attached, "base_plus_factorized"),
            _diag_id(attached, "base_plus_delta_weight_linear"),
        ),
        (
            "split_base_plus_delta_vs_materialized_weight_linear",
            _diag_id(attached, "base_plus_delta_weight_linear"),
            _diag_id(attached, "expected_materialized_linear"),
        ),
        (
            "attached_output_vs_merged_output",
            _common_id(attached, Q_PROJ, "output"),
            _common_id(merged, Q_PROJ, "output"),
        ),
    )
    return [
        _compare_archive_tensors(name, left, right, archive)
        for name, left, right in pairs
    ]


def _compare_archive_tensors(
    name: str,
    left_id: str,
    right_id: str,
    archive: _TensorArchive,
) -> dict[str, Any]:
    left_tensor = archive.tensors[left_id].contiguous()
    right_tensor = archive.tensors[right_id].contiguous()
    if list(left_tensor.shape) != list(right_tensor.shape):
        raise RuntimeError(f"comparison shape mismatch: {name}")
    left = left_tensor.view(-1)
    right = right_tensor.view(-1)
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    deltas = [abs(a - b) for a, b in zip(left_values, right_values, strict=True)]
    different_indices = [
        index
        for index, (a, b) in enumerate(
            zip(left_values, right_values, strict=True)
        )
        if a != b
    ]
    left_raw = left.view(torch.uint8).numpy().tobytes()
    right_raw = right.view(torch.uint8).numpy().tobytes()
    left_bits = struct.iter_unpack("<I", left_raw)
    right_bits = struct.iter_unpack("<I", right_raw)
    bitwise_different = sum(
        a[0] != b[0] for a, b in zip(left_bits, right_bits, strict=True)
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
    rms = math.sqrt(math.fsum(value * value for value in deltas) / len(deltas))
    left_record = archive.record(left_id)
    right_record = archive.record(right_id)
    return {
        "name": name,
        "shape": list(archive.tensors[left_id].shape),
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


def _instrumented_repeat_stability(
    runs: list[dict[str, Any]],
    archive: _TensorArchive,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in (ATTACHED_PATH, MERGED_PATH):
        path_runs = sorted(
            (run for run in runs if run["path"] == path),
            key=lambda item: item["repeat"],
        )
        if len(path_runs) != 2:
            raise RuntimeError(f"expected two instrumented runs for {path}")
        first, second = path_runs
        first_records = {
            record["semantic_key"]: record
            for record in archive.records
            if record["run_id"] == first["run_id"]
        }
        second_records = {
            record["semantic_key"]: record
            for record in archive.records
            if record["run_id"] == second["run_id"]
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


def _run_operation_graph_passed(
    path: str,
    run_id: str,
    events: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> bool:
    if path == MERGED_PATH:
        return events == []
    if path != ATTACHED_PATH:
        return False
    return (
        [event["operation"] for event in events]
        == list(EXPECTED_ATTACHED_OPERATION_SEQUENCE)
        and [event["event_index"] for event in events]
        == list(range(len(EXPECTED_ATTACHED_OPERATION_SEQUENCE)))
        and all(
            event["call_index"] == TARGET_STEP_INDEX
            and event["generation_step_index"] == TARGET_STEP_INDEX
            and event["tensor_id"] == _diag_id(run_id, event["operation"])
            for event in events
        )
        and snapshot["rank"] == 16
        and snapshot["alpha"] == 32.0
        and snapshot["scaling"] == 2.0
        and snapshot["dropout_probability"] == 0.05
    )


def _operation_graph_audit(
    runs: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_sequences = {
        run["run_id"]: [
            event["operation"] for event in run["operation_graph_events"]
        ]
        for run in runs
    }
    attached_snapshots = [
        snapshots[run["run_id"]]
        for run in runs
        if run["path"] == ATTACHED_PATH
    ]
    hyperparameters = [
        {
            "rank": snapshot["rank"],
            "alpha": snapshot["alpha"],
            "scaling": snapshot["scaling"],
            "dropout_probability": snapshot["dropout_probability"],
        }
        for snapshot in attached_snapshots
    ]
    attached_sequence_identity = all(
        sequence == list(EXPECTED_ATTACHED_OPERATION_SEQUENCE)
        for run_id, sequence in run_sequences.items()
        if run_id.startswith("fp32-attached-")
    )
    merged_sequences_empty = all(
        sequence == []
        for run_id, sequence in run_sequences.items()
        if run_id.startswith("fp32-safe-merged-")
    )
    hyperparameters_identical = (
        len(hyperparameters) == 2
        and hyperparameters[0] == hyperparameters[1]
        and hyperparameters[0]
        == {
            "rank": 16,
            "alpha": 32.0,
            "scaling": 2.0,
            "dropout_probability": 0.05,
        }
    )
    run_audits_passed = all(
        run["operation_graph_plan_passed"] for run in runs
    )
    result = {
        "expected_attached_sequence": list(
            EXPECTED_ATTACHED_OPERATION_SEQUENCE
        ),
        "run_sequences": run_sequences,
        "attached_sequence_identity": attached_sequence_identity,
        "merged_sequences_empty": merged_sequences_empty,
        "adapter_hyperparameters": hyperparameters[0],
        "attached_hyperparameters_identical": hyperparameters_identical,
        "run_audits_passed": run_audits_passed,
    }
    result["passed"] = all(
        value
        for key, value in result.items()
        if key
        in {
            "attached_sequence_identity",
            "merged_sequences_empty",
            "attached_hyperparameters_identical",
            "run_audits_passed",
        }
    )
    return result


def _verify_cross_run_event_sequences(runs: list[dict[str, Any]]) -> None:
    sequences = [_semantic_event_sequence(run["capture_events"]) for run in runs]
    if any(sequence != sequences[0] for sequence in sequences[1:]):
        raise RuntimeError("paired common capture event sequences differ")
    if tuple(
        (item["module_name"], item["io_kind"]) for item in runs[0]["capture_events"]
    ) != EXPECTED_COMMON_EVENT_KEYS:
        raise RuntimeError("actual common capture sequence differs from plan")


def _verify_frozen_path_reproduction(
    run: dict[str, Any], isolation: dict[str, Any]
) -> None:
    references = [item for item in isolation["runs"] if item["path"] == run["path"]]
    if not references:
        raise RuntimeError(f"frozen path reference missing: {run['path']}")
    reference = references[0]
    score = run["generation_trace"]["scores"]
    raw = run["generation_trace"]["raw_logits"]
    expected_score = reference["generation_trace"]["scores"]
    expected_raw = reference["generation_trace"]["raw_logits"]
    if (
        run["generated_token_ids"] != reference["generated_token_ids"]
        or run["token_count"] != reference["token_count"]
        or run["token_ids_sha256"] != reference["token_ids_sha256"]
        or run["output_sha256"] != reference["output_sha256"]
        or score["trace_sha256"] != expected_score["trace_sha256"]
        or raw["trace_sha256"] != expected_raw["trace_sha256"]
        or score["comparison_step_vector_sha256"]
        != expected_score["comparison_step_vector_sha256"]
        or raw["comparison_step_vector_sha256"]
        != expected_raw["comparison_step_vector_sha256"]
        or score["comparison_step_index"] != TARGET_STEP_INDEX
        or raw["comparison_step_index"] != TARGET_STEP_INDEX
        or run["generated_token_ids"][TARGET_STEP_INDEX]
        != TARGET_EMITTED_TOKEN_ID
    ):
        raise RuntimeError(f"instrumented frozen path did not reproduce: {run!r}")


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    numerics: dict[str, Any],
    remediation: dict[str, Any],
    drift: dict[str, Any],
    isolation: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    isolation_probe._verify_sources(  # noqa: SLF001
        config,
        training,
        stability,
        numerics,
        remediation,
        drift,
        args,
    )
    if (
        file_sha256(args.isolation_evidence) != EXPECTED_ISOLATION_SHA256
        or isolation.get("experiment_id")
        != "fc-mvp-001-fp32-attached-merge-isolation-v1"
        or isolation.get("classification")
        != (
            "deterministic_fp32_attached_vs_merged_"
            "numerical_drift_without_token_drift"
        )
        or isolation.get("isolation_gate", {}).get("passed") is not True
        or isolation.get("remediation_gate", {}).get("passed") is not False
        or isolation.get("same_dtype_token_analysis", {}).get(
            "cross_path_identical"
        )
        is not True
        or isolation.get("comparison_step", {}).get("step_index")
        != TARGET_STEP_INDEX
        or isolation.get("frozen_bf16_context", {}).get("context_only")
        is not True
        or isolation.get("frozen_bf16_context", {}).get("gpu_paths_rerun")
        is not False
        or isolation.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-fp32-attached-merge-numerics-v1"
        or isolation.get("merged_artifact_saved") is not False
        or isolation.get("merged_artifact_allowed") is not False
        or isolation.get("runtime_eligible") is not False
    ):
        raise RuntimeError("required FP32 attached/merge isolation chain is invalid")
    if (
        isolation.get("drift_evidence_sha256")
        != file_sha256(args.drift_evidence)
        or isolation.get("remediation_evidence_sha256")
        != file_sha256(args.remediation_evidence)
        or isolation.get("stability_evidence_sha256")
        != file_sha256(args.stability_evidence)
        or isolation.get("numerics_evidence_sha256")
        != file_sha256(args.numerics_evidence)
        or isolation.get("config_sha256") != canonical_config_sha256(config)
        or isolation.get("adapter_files")
        != directory_artifact_manifest(args.adapter_dir)
    ):
        raise RuntimeError("FP32 isolation source pins do not reproduce")


def _target_alignment(args: Any, kwargs: Any, call_index: int) -> dict[str, Any]:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args and torch.is_tensor(args[0]):
        input_ids = args[0]
    if input_ids is None:
        raise RuntimeError("target forward input_ids missing")
    cache_position = kwargs.get("cache_position")
    position_ids = kwargs.get("position_ids")
    past = kwargs.get("past_key_values")
    if cache_position is None or not torch.is_tensor(cache_position):
        raise RuntimeError("target forward cache_position missing")
    if past is None or not hasattr(past, "get_seq_length"):
        raise RuntimeError("target forward cache object missing")
    return {
        "call_index": call_index,
        "generation_step_index": TARGET_STEP_INDEX,
        "input_token_ids": [int(value) for value in input_ids.detach().cpu().view(-1)],
        "input_shape": list(input_ids.shape),
        "cache_position": [
            int(value) for value in cache_position.detach().cpu().view(-1)
        ],
        "position_ids": (
            None
            if position_ids is None
            else [int(value) for value in position_ids.detach().cpu().view(-1)]
        ),
        "past_length": int(past.get_seq_length()),
    }


def _target_alignment_passed(value: dict[str, Any]) -> bool:
    position_ids = value.get("position_ids")
    return (
        value.get("call_index") == TARGET_STEP_INDEX
        and value.get("generation_step_index") == TARGET_STEP_INDEX
        and value.get("input_token_ids") == [TARGET_INPUT_TOKEN_ID]
        and value.get("input_shape") == [1, 1]
        and value.get("cache_position") == [TARGET_CACHE_POSITION]
        and position_ids in (None, [TARGET_CACHE_POSITION])
        and value.get("past_length") == TARGET_CACHE_POSITION
        and value.get("causal_forward_calls") == 48
    )


def _module_input_tensor(args: Any, kwargs: Any) -> tuple[Any, str]:
    if args:
        try:
            return _first_tensor_leaf(args, "args")
        except RuntimeError:
            pass
    for key in ("hidden_states", "x", "input_ids", "inputs_embeds"):
        if key in kwargs:
            try:
                return _first_tensor_leaf(kwargs[key], f"kwargs.{key}")
            except RuntimeError:
                pass
    return _first_tensor_leaf(kwargs, "kwargs")


def _first_tensor_leaf(value: Any, path: str) -> tuple[Any, str]:
    if torch.is_tensor(value):
        return value, path
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            try:
                return _first_tensor_leaf(item, f"{path}[{index}]")
            except RuntimeError:
                continue
    if isinstance(value, dict):
        for key in sorted(value):
            try:
                return _first_tensor_leaf(value[key], f"{path}.{key}")
            except RuntimeError:
                continue
    raise RuntimeError(f"tensor leaf missing at {path}: {type(value)!r}")


def _stash_generated_vector(
    captured: dict[str, dict[str, Any]],
    run_id: str,
    path: str,
    kind: str,
    tensor: Any,
) -> None:
    tensor_id = _generation_id(run_id, kind)
    captured[tensor_id] = {
        "tensor": tensor.detach().clone(),
        "metadata": {
            "run_id": run_id,
            "path": path,
            "semantic_key": f"generation|{kind}|step{TARGET_STEP_INDEX}",
            "event_scope": "generation",
            "event_index": TARGET_STEP_INDEX,
            "module_name": "generate",
            "module_type": "transformers.generation",
            "occurrence_index": 0,
            "io_kind": kind,
            "tensor_path": f"generated.{kind}[{TARGET_STEP_INDEX}]",
        },
    }


def _active_adapter_name(module: Any) -> str:
    names = list(module.active_adapters)
    if names != ["default"] or set(module.lora_A) != {"default"}:
        raise RuntimeError(f"unexpected active Adapter names: {names!r}")
    return names[0]


def _causal_model(model: Any) -> Any:
    return model.get_base_model() if isinstance(model, PeftModel) else model


def _module_type(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _common_id(run_id: str, stage: str, io_kind: str) -> str:
    return f"{run_id}|common|{stage}|{io_kind}"


def _diag_id(run_id: str, label: str) -> str:
    return f"{run_id}|diagnostic|q_proj|{label}"


def _generation_id(run_id: str, kind: str) -> str:
    return f"{run_id}|generation|{kind}|step{TARGET_STEP_INDEX}"


def _weight_id(run_id: str, label: str) -> str:
    return f"{run_id}|weight|q_proj|{label}"


def _semantic_event_sequence(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
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


def _float32_tensor_sha256(tensor: Any) -> str:
    return _sha256(_float32_tensor_bytes(tensor))


def _float32_tensor_bytes(tensor: Any) -> bytes:
    value = tensor.detach().to(dtype=torch.float32).contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _float32_mismatched_elements(left: Any, right: Any) -> int:
    left_raw = _float32_tensor_bytes(left)
    right_raw = _float32_tensor_bytes(right)
    if len(left_raw) != len(right_raw):
        raise RuntimeError("float32 tensor byte-length mismatch")
    return sum(
        left_value[0] != right_value[0]
        for left_value, right_value in zip(
            struct.iter_unpack("<f", left_raw),
            struct.iter_unpack("<f", right_raw),
            strict=True,
        )
    )


def _weight_materialization_statistics(
    base: Any,
    delta: Any,
    expected: Any,
) -> dict[str, Any]:
    base_raw = _float32_tensor_bytes(base)
    delta_raw = _float32_tensor_bytes(delta)
    expected_raw = _float32_tensor_bytes(expected)
    if len({len(base_raw), len(delta_raw), len(expected_raw)}) != 1:
        raise RuntimeError("weight materialization tensor byte-length mismatch")
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
        reconstructed = struct.unpack(
            "<f",
            struct.pack("<f", base_value + delta_value),
        )[0]
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
        raise RuntimeError(
            "archived expected weight does not reproduce FP32 base + delta"
        )
    elements = len(base_raw) // 4
    mean_error = math.fsum(
        abs((expected_item[0] - base_item[0]) - delta_item[0])
        for base_item, delta_item, expected_item in zip(
            struct.iter_unpack("<f", base_raw),
            struct.iter_unpack("<f", delta_raw),
            struct.iter_unpack("<f", expected_raw),
            strict=True,
        )
    ) / elements
    return {
        "ideal_nonzero_updates": ideal_nonzero,
        "effective_changed_weights": effective_changed,
        "ideal_nonzero_updates_rounded_to_base": rounded_to_base,
        "max_abs_materialization_error": max_error,
        "mean_abs_materialization_error": mean_error,
    }


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bound_sha256(header: bytes, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _frozen_path_references(isolation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in (ATTACHED_PATH, MERGED_PATH):
        run = next(item for item in isolation["runs"] if item["path"] == path)
        result[path] = {
            "token_count": run["token_count"],
            "token_ids_sha256": run["token_ids_sha256"],
            "output_sha256": run["output_sha256"],
            "score_trace_sha256": run["generation_trace"]["scores"][
                "trace_sha256"
            ],
            "raw_logit_trace_sha256": run["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            "comparison_step_index": TARGET_STEP_INDEX,
            "comparison_score_vector_sha256": run["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ],
            "comparison_raw_logit_vector_sha256": run["generation_trace"][
                "raw_logits"
            ]["comparison_step_vector_sha256"],
        }
    return result


def _capture_plan() -> dict[str, Any]:
    return {
        "scope": "target_forward_pre_registered_paired_common_semantic_modules",
        "target_generation_step_index": TARGET_STEP_INDEX,
        "selection_basis": "frozen_bf16_token_boundary_context",
        "common_stages": list(COMMON_STAGES),
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


def _protocol(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    isolation_protocol = isolation_probe._protocol(config, tokenizer)  # noqa: SLF001
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": [
            {"path": path, "repeat": repeat, "run_id": run_id}
            for path, repeat, run_id in RUN_PLAN
        ],
        "run_order_design": "ABBA",
        "fresh_loads_per_path": {ATTACHED_PATH: 2, MERGED_PATH: 2},
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "target_forward": {
            "generation_step_index": TARGET_STEP_INDEX,
            "input_generated_token_index": TARGET_STEP_INDEX - 1,
            "input_token_id": TARGET_INPUT_TOKEN_ID,
            "predicted_token_id": TARGET_EMITTED_TOKEN_ID,
            "past_length": TARGET_CACHE_POSITION,
            "cache_position": [TARGET_CACHE_POSITION],
        },
        "paths": isolation_protocol["paths"],
        "generation": isolation_protocol["generation"],
        "sdp_kernel_flags": isolation_protocol["sdp_kernel_flags"],
        "operation_graphs": {
            "attached": "base_linear(x) + lora_B(lora_A(dropout(x))) * scale",
            "merged": "linear(x, base_weight + (lora_B_weight @ lora_A_weight) * scale)",
        },
    }


def _causal_scope() -> dict[str, Any]:
    return {
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


def _constraints() -> dict[str, bool]:
    return {
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


def _next_action_constraints() -> dict[str, bool]:
    return {
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
    }


def _locked_next_action() -> dict[str, Any]:
    return {
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
        "constraints": _next_action_constraints(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
