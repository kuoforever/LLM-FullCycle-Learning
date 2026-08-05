"""Trace registered BF16-versus-FP32 attached module-output numerics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
from transformers import AutoTokenizer  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_attached_dtype_numerics import (  # noqa: E402
    ATTACHED_DTYPE_NUMERICS_VERSION,
    CLASSIFICATION,
    REGISTERED_OUTPUT_STAGES,
    classify_attached_dtype_numerics,
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
from scripts import probe_tool_router_attached_dtype_isolation as isolation_probe  # noqa: E402

BF16_PATH = isolation_probe.BF16_PATH
FP32_PATH = isolation_probe.FP32_PATH
PATH_ORDER = (BF16_PATH, FP32_PATH)
RUN_PLAN = (
    (BF16_PATH, 1, "bf16-attached-numerics-r1"),
    (FP32_PATH, 1, "fp32-attached-numerics-r1"),
    (FP32_PATH, 2, "fp32-attached-numerics-r2"),
    (BF16_PATH, 2, "bf16-attached-numerics-r2"),
)
REPRESENTATIVE_RUN = {
    BF16_PATH: "bf16-attached-numerics-r1",
    FP32_PATH: "fp32-attached-numerics-r1",
}
REPEAT_RUN = {
    BF16_PATH: "bf16-attached-numerics-r2",
    FP32_PATH: "fp32-attached-numerics-r2",
}
TARGET_STEP_INDEX = isolation_probe.TARGET_STEP_INDEX
TARGET_INPUT_TOKEN_ID = isolation_probe.TARGET_INPUT_TOKEN_ID
TARGET_CACHE_POSITION = isolation_probe.TARGET_CACHE_POSITION
TARGET_FORWARD_CALLS = isolation_probe.TARGET_FORWARD_CALLS
MAX_RESIDUAL_CUDA_BYTES = isolation_probe.MAX_RESIDUAL_CUDA_BYTES
EXPECTED_INPUT_TOKEN_COUNT = isolation_probe.EXPECTED_INPUT_TOKEN_COUNT
EXPECTED_INPUT_TOKEN_SHA256 = isolation_probe.EXPECTED_INPUT_TOKEN_SHA256
EXPECTED_ISOLATION_EVIDENCE_SHA256 = (
    "sha256:7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--isolation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isolation_probe._verify_output_boundary(args.output)  # noqa: SLF001

    config = isolation_probe._load_json(args.config)  # noqa: SLF001
    isolation = isolation_probe._load_json(args.isolation_evidence)  # noqa: SLF001
    source_lineage = _verify_sources(config, isolation, args)
    environment = isolation_probe._verify_environment(config)  # noqa: SLF001
    if environment != isolation["environment"]:
        raise RuntimeError("attached dtype environment drift")

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("evaluation digest mismatch")
    record = evaluation[0]
    if record["example_id"] != "eval-001":
        raise RuntimeError("locked numerics probe requires eval-001 first")
    prompt_path = ROOT / config["prompt"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
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
        len(input_token_ids) != EXPECTED_INPUT_TOKEN_COUNT
        or input_digest != EXPECTED_INPUT_TOKEN_SHA256
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")

    storage_audit = {
        "base_checkpoint": isolation_probe._safetensor_storage_audit(  # noqa: SLF001
            args.model_dir / config["model"]["weight_file"]
        ),
        "adapter": isolation_probe._safetensor_storage_audit(  # noqa: SLF001
            args.adapter_dir / "adapter_model.safetensors"
        ),
    }
    if storage_audit != isolation["storage_audit"]:
        raise RuntimeError("source storage audit drift")
    capture_plan = _capture_plan()
    capture_plan_sha256 = _sha256(_canonical_json(capture_plan))

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    for order_index, (path, repeat, run_id) in enumerate(RUN_PLAN):
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
            capture_plan_sha256=capture_plan_sha256,
        )
        runs.append(run)
        snapshots[run_id] = snapshot

    by_path = {
        path: [run for run in runs if run["path"] == path]
        for path in PATH_ORDER
    }
    if any(len(by_path[path]) != 2 for path in PATH_ORDER):
        raise RuntimeError("fixed ABBA attached dtype run plan was not executed")

    frozen_references = isolation["frozen_path_references"]
    path_reproduction: dict[str, Any] = {
        path: isolation_probe._path_reproduction(  # noqa: SLF001
            by_path[path], frozen_references[path]
        )
        for path in PATH_ORDER
    }
    path_reproduction["passed"] = all(
        path_reproduction[path]["passed"] for path in PATH_ORDER
    )
    if path_reproduction["passed"] is not True:
        raise RuntimeError(f"frozen path reproduction failed: {path_reproduction!r}")

    path_repeat_stability = _path_repeat_stability(by_path)
    capture_repeat_stability = _capture_repeat_stability(by_path)
    if (
        path_repeat_stability["passed"] is not True
        or capture_repeat_stability["passed"] is not True
    ):
        raise RuntimeError(
            "attached dtype repeat stability failed: "
            f"paths={path_repeat_stability!r}, captures={capture_repeat_stability!r}"
        )
    event_sequence_identity = len(
        {run["capture_event_sequence_sha256"] for run in runs}
    ) == 1
    if not event_sequence_identity:
        raise RuntimeError("cross-run capture event sequence drift")

    representative_comparisons = _compare_snapshots(
        snapshots[REPRESENTATIVE_RUN[BF16_PATH]],
        snapshots[REPRESENTATIVE_RUN[FP32_PATH]],
    )
    repeat_comparisons = _compare_snapshots(
        snapshots[REPEAT_RUN[BF16_PATH]],
        snapshots[REPEAT_RUN[FP32_PATH]],
    )
    representative_manifest_sha256 = _sha256(
        _canonical_json(representative_comparisons)
    )
    repeat_manifest_sha256 = _sha256(_canonical_json(repeat_comparisons))
    paired_comparison_repeat_identity = (
        representative_comparisons == repeat_comparisons
        and representative_manifest_sha256 == repeat_manifest_sha256
    )
    if not paired_comparison_repeat_identity:
        raise RuntimeError("paired cross-dtype comparison summary drift")

    capture_plan_executed = all(run["capture_plan_passed"] for run in runs)
    target_forward_aligned = all(run["target_alignment_passed"] for run in runs)
    lm_head_raw_logit_linked = all(
        run["lm_head_raw_logit_linked"] for run in runs
    )
    source_adapter_unchanged = (
        directory_artifact_manifest(args.adapter_dir) == isolation["adapter_files"]
    )
    source_model_unchanged = (
        file_sha256(args.model_dir / config["model"]["weight_file"])
        == isolation["model_weight_sha256"]
    )
    eval_digest_unchanged = fixture_digest(evaluation) == isolation["eval_digest"]
    prompt_digest_unchanged = file_sha256(prompt_path) == isolation["prompt_sha256"]
    source_inputs_unchanged = all(
        (
            source_adapter_unchanged,
            source_model_unchanged,
            eval_digest_unchanged,
            prompt_digest_unchanged,
            input_digest == isolation["input_token_ids_sha256"],
        )
    )
    module_analysis = classify_attached_dtype_numerics(
        representative_comparisons,
        bf16_repeat_stable=path_repeat_stability[BF16_PATH]["passed"],
        fp32_repeat_stable=path_repeat_stability[FP32_PATH]["passed"],
        bf16_reference_reproduced=path_reproduction[BF16_PATH]["passed"],
        fp32_reference_reproduced=path_reproduction[FP32_PATH]["passed"],
        capture_plan_executed=capture_plan_executed,
        target_forward_aligned=target_forward_aligned,
        lm_head_raw_logit_linked=lm_head_raw_logit_linked,
        attached_execution_form_fixed=True,
        source_inputs_unchanged=source_inputs_unchanged,
        module_tensor_payload_absent=True,
    )
    if module_analysis["classification"] != CLASSIFICATION:
        raise RuntimeError(f"unexpected numerics classification: {module_analysis!r}")

    lm_head_link = _lm_head_frozen_delta_link(
        representative_comparisons[-1], isolation
    )
    fresh_load_memory_isolated = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"]
        <= MAX_RESIDUAL_CUDA_BYTES
        for run in runs
    )
    numerics_gate = {
        "bf16_attached_repeat_stable": path_repeat_stability[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": path_repeat_stability[FP32_PATH]["passed"],
        "bf16_capture_repeat_exact": capture_repeat_stability[BF16_PATH]["passed"],
        "fp32_capture_repeat_exact": capture_repeat_stability[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": path_reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": path_reproduction[FP32_PATH]["passed"],
        "target_forward_aligned": target_forward_aligned,
        "capture_plan_executed": capture_plan_executed,
        "cross_run_event_sequence_identical": event_sequence_identity,
        "paired_comparison_repeat_exact": paired_comparison_repeat_identity,
        "first_registered_module_difference_located": (
            module_analysis["first_unequal_module"] is not None
        ),
        "preceding_registered_outputs_identical": module_analysis[
            "preceding_registered_outputs_identical"
        ],
        "registered_lm_head_difference_quantified": module_analysis[
            "registered_lm_head_difference_observed"
        ],
        "lm_head_raw_logit_linked": lm_head_raw_logit_linked,
        "frozen_lm_head_delta_linked": lm_head_link["passed"],
    }
    numerics_gate["passed"] = all(numerics_gate.values())
    acceptance = {
        "upstream_isolation_evidence_locked": True,
        "frozen_input_reproduced": True,
        "attached_execution_form_fixed": True,
        "base_dtype_only_treatment": True,
        "capture_plan_pre_registered": True,
        "capture_plan_executed": capture_plan_executed,
        "path_repeat_stability": path_repeat_stability["passed"],
        "capture_repeat_stability": capture_repeat_stability["passed"],
        "paired_comparison_repeat_identity": paired_comparison_repeat_identity,
        "target_forward_aligned": target_forward_aligned,
        "lm_head_raw_logit_linked": lm_head_raw_logit_linked,
        "source_inputs_unchanged": source_inputs_unchanged,
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "module_tensor_payload_absent": True,
    }
    if numerics_gate["passed"] is not True or not all(acceptance.values()):
        raise RuntimeError(
            f"attached dtype numerics gate failed: {numerics_gate!r}, "
            f"acceptance={acceptance!r}"
        )

    constraints = _constraints()
    result = {
        "attached_dtype_numerics_version": ATTACHED_DTYPE_NUMERICS_VERSION,
        "experiment_id": "fc-mvp-001-attached-dtype-numerics-v1",
        "source_experiment_id": isolation["source_experiment_id"],
        "source_gate_experiment_id": isolation["experiment_id"],
        "source_lineage": source_lineage,
        "training_lock_sha256": isolation["training_lock_sha256"],
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": isolation["adapter_files"],
        "model_weight_sha256": isolation["model_weight_sha256"],
        "prompt_sha256": isolation["prompt_sha256"],
        "eval_digest": isolation["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "environment": environment,
        "protocol": _protocol(config, tokenizer, capture_plan_sha256),
        "capture_plan": capture_plan,
        "capture_plan_sha256": capture_plan_sha256,
        "frozen_path_references": frozen_references,
        "runs": runs,
        "path_repeat_stability": path_repeat_stability,
        "path_reproduction": path_reproduction,
        "capture_repeat_stability": capture_repeat_stability,
        "paired_comparison_repeat": {
            "representative_run_ids": REPRESENTATIVE_RUN,
            "repeat_run_ids": REPEAT_RUN,
            "representative_manifest_sha256": representative_manifest_sha256,
            "repeat_manifest_sha256": repeat_manifest_sha256,
            "exact_identity": paired_comparison_repeat_identity,
        },
        "module_comparisons": representative_comparisons,
        "module_comparison_manifest_sha256": representative_manifest_sha256,
        "module_analysis": module_analysis,
        "lm_head_frozen_delta_link": lm_head_link,
        "delta_statistics_scope": (
            "probe_derived_summary_algebra_and_frozen_manifest_only"
        ),
        "classification": module_analysis["classification"],
        "causal_scope": _causal_scope(),
        "numerics_gate": numerics_gate,
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(run["peak_gpu_memory_bytes"] for run in runs),
        "module_tensor_payload_saved": False,
        "module_tensor_sidecar_allowed": False,
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _locked_next_action(constraints),
        "runtime_eligible": False,
        "runtime_eligibility_reason": module_analysis["classification"],
        "offline": True,
    }
    isolation_probe._require_finite_json(result, "$")  # noqa: SLF001
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
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
    capture_plan_sha256: str,
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
    started = time.perf_counter()
    model: Any | None = None
    run: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    try:
        model, precision = isolation_probe._load_attached_model(  # noqa: SLF001
            model_dir,
            adapter_dir,
            config,
            path=path,
        )
        (
            token_ids,
            generation_trace,
            target_vectors,
            target_alignment,
            capture_records,
            canonical_tensors,
            capture_events,
            lm_head_linked,
        ) = _instrumented_generate(
            model=model,
            path=path,
            encoded_cpu=encoded_cpu,
            config=config,
            tokenizer=tokenizer,
        )
        precision["generation"] = {
            "score_dtypes": generation_trace["scores"]["native_dtypes"],
            "all_scores_float32": (
                generation_trace["scores"]["native_dtypes"] == ["float32"]
            ),
            "raw_logit_dtypes": generation_trace["raw_logits"]["native_dtypes"],
            "all_raw_logits_float32": (
                generation_trace["raw_logits"]["native_dtypes"] == ["float32"]
            ),
            "dtype_semantics": "transformers_generate_return_tensor_dtype",
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        path_protocol_passed = isolation_probe._path_protocol_passed(  # noqa: SLF001
            precision, path=path
        )
        target_alignment_passed = isolation_probe._target_alignment_passed(  # noqa: SLF001
            target_alignment, path=path
        )
        event_sequence = _semantic_event_sequence(capture_events)
        capture_event_sequence_sha256 = _sha256(_canonical_json(event_sequence))
        capture_manifest_sha256 = _sha256(_canonical_json(capture_records))
        capture_plan_passed = (
            tuple(event["module_name"] for event in capture_events)
            == REGISTERED_OUTPUT_STAGES
            and len(capture_records) == len(REGISTERED_OUTPUT_STAGES)
            and all(
                value["capture_plan_sha256"] == capture_plan_sha256
                for value in capture_records
            )
        )
        if not all(
            (
                path_protocol_passed,
                target_alignment_passed,
                capture_plan_passed,
                lm_head_linked,
            )
        ):
            raise RuntimeError(
                f"instrumented protocol failed for {run_id}: "
                f"path={path_protocol_passed}, target={target_alignment_passed}, "
                f"capture={capture_plan_passed}, lm_head={lm_head_linked}"
            )
        torch.cuda.synchronize()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        run = {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order_index,
            "fresh_load": True,
            "materialization_form": "attached_factorized_lora",
            "base_load_dtype": isolation_probe._dtype_name_for_path(path),  # noqa: SLF001
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": _sha256(decoded.encode("utf-8")),
            "precision_audit": precision,
            "generation_trace": generation_trace,
            "target_alignment": target_alignment,
            "target_alignment_passed": target_alignment_passed,
            "capture_records": capture_records,
            "capture_events": capture_events,
            "capture_record_count": len(capture_records),
            "capture_event_sequence_sha256": capture_event_sequence_sha256,
            "capture_manifest_sha256": capture_manifest_sha256,
            "capture_plan_sha256": capture_plan_sha256,
            "capture_plan_passed": capture_plan_passed,
            "lm_head_raw_logit_linked": lm_head_linked,
            "path_protocol_passed": path_protocol_passed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "memory_allocated_before_load_bytes": allocated_before,
        }
        snapshot = {
            "path": path,
            "records": {value["module_name"]: value for value in capture_records},
            "tensors": canonical_tensors,
            "target_score": target_vectors["score"],
            "target_raw_logit": target_vectors["raw_logit"],
        }
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
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[
    list[int],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    bool,
]:
    if torch.is_autocast_enabled():
        raise RuntimeError("generation must not run under autocast")
    causal = isolation_probe._causal_model(model)  # noqa: SLF001
    modules = dict(causal.named_modules())
    missing = [stage for stage in REGISTERED_OUTPUT_STAGES if stage not in modules]
    if missing:
        raise RuntimeError(f"registered capture modules missing: {missing!r}")
    state: dict[str, Any] = {
        "call_index": -1,
        "active": False,
        "target_count": 0,
        "alignment": {},
    }
    gpu_captures: dict[str, dict[str, Any]] = {}
    capture_events: list[dict[str, Any]] = []
    handles: list[Any] = []

    def causal_pre_hook(_module: Any, hook_args: Any, kwargs: Any) -> None:
        state["call_index"] += 1
        state["active"] = state["call_index"] == TARGET_STEP_INDEX
        if state["active"]:
            state["target_count"] += 1
            state["alignment"] = isolation_probe._target_alignment(  # noqa: SLF001
                hook_args,
                kwargs,
                state["call_index"],
            )

    def causal_post_hook(
        _module: Any,
        _args: Any,
        _kwargs: Any,
        _output: Any,
    ) -> None:
        if state["active"]:
            state["active"] = False

    handles.append(causal.register_forward_pre_hook(causal_pre_hook, with_kwargs=True))
    handles.append(causal.register_forward_hook(causal_post_hook, with_kwargs=True))

    def make_output_hook(stage: str) -> Any:
        def hook(module: Any, _args: Any, output: Any) -> None:
            if not state["active"]:
                return
            if stage in gpu_captures:
                raise RuntimeError(f"duplicate registered output capture: {stage}")
            tensor, tensor_path = _first_tensor_leaf(output, "output")
            expected = _expected_stage_contract(stage)
            module_type = _module_type(module)
            if module_type != expected["module_type"]:
                raise RuntimeError(
                    f"registered module type drift for {stage}: {module_type}"
                )
            if tensor_path != expected["tensor_path"]:
                raise RuntimeError(
                    f"registered tensor path drift for {stage}: {tensor_path}"
                )
            event_index = len(capture_events)
            gpu_captures[stage] = {
                "tensor": tensor.detach().clone(),
                "event_index": event_index,
                "module_name": stage,
                "module_type": module_type,
                "occurrence_index": 0,
                "call_index": state["call_index"],
                "generation_step_index": TARGET_STEP_INDEX,
                "io_kind": "output",
                "tensor_path": tensor_path,
            }
            capture_events.append(
                {
                    key: value
                    for key, value in gpu_captures[stage].items()
                    if key != "tensor"
                }
            )

        return hook

    for stage in REGISTERED_OUTPUT_STAGES:
        handles.append(modules[stage].register_forward_hook(make_output_hook(stage)))

    encoded = {key: value.to("cuda") for key, value in encoded_cpu.items()}
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["generation"]["max_new_tokens"],
                use_cache=config["generation"]["use_cache"],
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    new_tokens = generated.sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.cpu()]
    if generated.scores is None or generated.logits is None:
        raise RuntimeError("generate did not return scores and raw logits")
    if len(generated.scores) != len(token_ids) or len(generated.logits) != len(
        token_ids
    ):
        raise RuntimeError("generated trace count does not match token count")
    expected_shape = [1, int(model.config.vocab_size)]
    score_summary, target_score = isolation_probe._summarize_trace(  # noqa: SLF001
        generated.scores,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    raw_summary, target_raw_logit = isolation_probe._summarize_trace(  # noqa: SLF001
        generated.logits,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    if generated.past_key_values is None:
        raise RuntimeError("cached generation did not return past_key_values")
    if state["target_count"] != 1:
        raise RuntimeError(f"target forward count drift: {state['target_count']!r}")
    state["alignment"]["causal_forward_calls"] = state["call_index"] + 1
    if tuple(event["module_name"] for event in capture_events) != REGISTERED_OUTPUT_STAGES:
        raise RuntimeError(f"registered capture sequence drift: {capture_events!r}")
    capture_records: list[dict[str, Any]] = []
    canonical_tensors: dict[str, Any] = {}
    capture_plan_sha256 = _sha256(_canonical_json(_capture_plan()))
    for stage in REGISTERED_OUTPUT_STAGES:
        captured = gpu_captures[stage]
        native = captured["tensor"].detach().contiguous().cpu()
        canonical = _canonical_float32_tensor(native)
        stage_record = _capture_record(
            captured,
            native,
            canonical,
            path=path,
            capture_plan_sha256=capture_plan_sha256,
        )
        capture_records.append(stage_record)
        canonical_tensors[stage] = canonical
    gpu_captures.clear()
    torch.cuda.empty_cache()

    lm_head = canonical_tensors["lm_head"].reshape(-1, expected_shape[1])[-1]
    raw_canonical = _canonical_float32_tensor(target_raw_logit)
    lm_head_linked = torch.equal(lm_head, raw_canonical)
    lm_head_sha256 = _float32_sha256(lm_head)
    raw_canonical_sha256 = _float32_sha256(raw_canonical)
    raw_native_sha256 = isolation_probe._float32_vector_sha256(  # noqa: SLF001
        target_raw_logit
    )
    if not lm_head_linked or lm_head_sha256 != raw_canonical_sha256:
        raise RuntimeError("registered LM-head output does not match raw logits")
    state["alignment"].update(
        {
            "lm_head_output_shape": list(canonical_tensors["lm_head"].shape),
            "lm_head_output_native_dtype": capture_records[-1]["native_dtype"],
            "lm_head_output_comparison_vector_sha256": lm_head_sha256,
            "generated_raw_logit_comparison_vector_sha256": raw_native_sha256,
            "lm_head_output_canonical_float32_sha256": lm_head_sha256,
            "generated_raw_logit_canonical_float32_sha256": raw_canonical_sha256,
            "generated_raw_logit_frozen_vector_sha256": raw_native_sha256,
        }
    )
    score_summary["comparison_step_index"] = TARGET_STEP_INDEX
    score_summary["comparison_step_vector_sha256"] = (
        isolation_probe._float32_vector_sha256(target_score)  # noqa: SLF001
    )
    raw_summary["comparison_step_index"] = TARGET_STEP_INDEX
    raw_summary["comparison_step_vector_sha256"] = raw_native_sha256
    generation_trace = {
        "step_count": len(token_ids),
        "vocabulary_size": expected_shape[1],
        "cache_returned": True,
        "scores": score_summary,
        "raw_logits": raw_summary,
        "lm_head_output": {
            "native_dtype": capture_records[-1]["native_dtype"],
            "shape": list(canonical_tensors["lm_head"].shape),
            "comparison_dtype": "float32",
            "all_finite": True,
            "comparison_step_index": TARGET_STEP_INDEX,
            "canonical_float32_sha256": lm_head_sha256,
            "frozen_raw_logit_vector_sha256": raw_native_sha256,
        },
    }
    return (
        token_ids,
        generation_trace,
        {"score": target_score, "raw_logit": target_raw_logit},
        dict(state["alignment"]),
        capture_records,
        canonical_tensors,
        capture_events,
        lm_head_linked,
    )


def _capture_record(
    event: Mapping[str, Any],
    native: Any,
    canonical: Any,
    *,
    path: str,
    capture_plan_sha256: str,
) -> dict[str, Any]:
    expected_dtype = "bfloat16" if path == BF16_PATH else "float32"
    native_dtype = str(native.dtype).removeprefix("torch.")
    if native_dtype != expected_dtype:
        raise RuntimeError(
            f"registered output dtype drift for {event['module_name']}: "
            f"{native_dtype} != {expected_dtype}"
        )
    expected_shape = _expected_stage_contract(str(event["module_name"]))["shape"]
    if list(native.shape) != expected_shape:
        raise RuntimeError(
            f"registered output shape drift for {event['module_name']}: "
            f"{list(native.shape)!r}"
        )
    finite_elements = int(torch.isfinite(canonical).sum())
    elements = canonical.numel()
    if finite_elements != elements:
        raise RuntimeError(f"non-finite registered output: {event['module_name']}")
    record = {
        **{key: value for key, value in event.items() if key != "tensor"},
        "native_dtype": native_dtype,
        "native_shape": list(native.shape),
        "native_stride": list(native.stride()),
        "native_layout": str(native.layout).removeprefix("torch."),
        "comparison_dtype": "float32",
        "elements": elements,
        "finite_elements": finite_elements,
        "all_finite": True,
        "native_payload_sha256": _tensor_bytes_sha256(native),
        "canonical_float32_sha256": _float32_sha256(canonical),
        "signed_zero_normalized": True,
        "capture_plan_sha256": capture_plan_sha256,
    }
    record["record_sha256"] = _sha256(_canonical_json(record))
    return record


def _compare_snapshots(
    bf16_snapshot: Mapping[str, Any],
    fp32_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        bf16_snapshot["path"] != BF16_PATH
        or fp32_snapshot["path"] != FP32_PATH
    ):
        raise RuntimeError("cross-dtype snapshot roles are invalid")
    comparisons = [
        _compare_stage(
            stage,
            bf16_snapshot["tensors"][stage],
            fp32_snapshot["tensors"][stage],
            bf16_snapshot["records"][stage],
            fp32_snapshot["records"][stage],
        )
        for stage in REGISTERED_OUTPUT_STAGES
    ]
    first_rms = next(
        value["root_mean_square_delta"]
        for value in comparisons
        if value["canonical_values_equal"] is False
    )
    if first_rms <= 0:
        raise RuntimeError("first registered difference has zero RMS")
    for value in comparisons:
        value[
            "root_mean_square_delta_ratio_to_first_registered_difference"
        ] = value["root_mean_square_delta"] / first_rms
    return comparisons


def _compare_stage(
    stage: str,
    bf16_tensor: Any,
    fp32_tensor: Any,
    bf16_record: Mapping[str, Any],
    fp32_record: Mapping[str, Any],
) -> dict[str, Any]:
    if list(bf16_tensor.shape) != list(fp32_tensor.shape):
        raise RuntimeError(f"cross-dtype registered shape drift: {stage}")
    bf16_values = [float(value) for value in bf16_tensor.reshape(-1).tolist()]
    fp32_values = [float(value) for value in fp32_tensor.reshape(-1).tolist()]
    absolute_deltas = [
        abs(left - right)
        for left, right in zip(bf16_values, fp32_values, strict=True)
    ]
    different_indices = [
        index for index, delta in enumerate(absolute_deltas) if delta != 0.0
    ]
    elements = len(absolute_deltas)
    sum_abs_delta = math.fsum(absolute_deltas)
    sum_squared_delta = math.fsum(delta * delta for delta in absolute_deltas)
    bf16_sum_squared = math.fsum(value * value for value in bf16_values)
    fp32_sum_squared = math.fsum(value * value for value in fp32_values)
    different_elements = len(different_indices)
    if different_indices:
        selected_first_index = different_indices[0]
        selected_max_index = max(
            different_indices, key=absolute_deltas.__getitem__
        )
        first_index: int | None = selected_first_index
        max_index: int | None = selected_max_index
        max_abs_delta = absolute_deltas[selected_max_index]
        bf16_first: float | None = bf16_values[selected_first_index]
        fp32_first: float | None = fp32_values[selected_first_index]
        bf16_max: float | None = bf16_values[selected_max_index]
        fp32_max: float | None = fp32_values[selected_max_index]
    else:
        first_index = None
        max_index = None
        max_abs_delta = 0.0
        bf16_first = None
        fp32_first = None
        bf16_max = None
        fp32_max = None
    mean_abs_delta = sum_abs_delta / elements
    rms_delta = math.sqrt(sum_squared_delta / elements)
    bf16_rms = math.sqrt(bf16_sum_squared / elements)
    fp32_rms = math.sqrt(fp32_sum_squared / elements)
    endpoint_rms = max(bf16_rms, fp32_rms)
    return {
        "name": stage,
        "shape": list(bf16_tensor.shape),
        "elements": elements,
        "bf16_native_dtype": bf16_record["native_dtype"],
        "fp32_native_dtype": fp32_record["native_dtype"],
        "comparison_dtype": "float32",
        "bf16_float32_sha256": bf16_record["canonical_float32_sha256"],
        "fp32_float32_sha256": fp32_record["canonical_float32_sha256"],
        "canonical_values_equal": different_elements == 0,
        "different_elements": different_elements,
        "first_different_flat_index": first_index,
        "max_abs_delta_flat_index": max_index,
        "bf16_value_at_first_difference": bf16_first,
        "fp32_value_at_first_difference": fp32_first,
        "bf16_value_at_max_abs_delta": bf16_max,
        "fp32_value_at_max_abs_delta": fp32_max,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "root_mean_square_delta": rms_delta,
        "sum_abs_delta": sum_abs_delta,
        "sum_squared_delta": sum_squared_delta,
        "different_fraction": different_elements / elements,
        "bf16_root_mean_square": bf16_rms,
        "fp32_root_mean_square": fp32_rms,
        "normalized_root_mean_square_delta": (
            rms_delta / endpoint_rms if endpoint_rms > 0 else 0.0
        ),
        "root_mean_square_delta_ratio_to_first_registered_difference": 0.0,
    }


def _path_repeat_stability(
    by_path: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        value = isolation_probe.analyze_path_repeat_stability(
            first["generated_token_ids"],
            second["generated_token_ids"],
            first_output_sha256=first["output_sha256"],
            second_output_sha256=second["output_sha256"],
            first_score_trace_sha256=first["generation_trace"]["scores"][
                "trace_sha256"
            ],
            second_score_trace_sha256=second["generation_trace"]["scores"][
                "trace_sha256"
            ],
            first_raw_logit_trace_sha256=first["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            second_raw_logit_trace_sha256=second["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            first_score_vector_sha256=first["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ],
            second_score_vector_sha256=second["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ],
            first_raw_logit_vector_sha256=first["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ],
            second_raw_logit_vector_sha256=second["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ],
            precision_audits_identical=(
                first["precision_audit"] == second["precision_audit"]
            ),
        )
        value["target_alignment_identity"] = (
            first["target_alignment"] == second["target_alignment"]
        )
        value["capture_manifest_identity"] = (
            first["capture_manifest_sha256"]
            == second["capture_manifest_sha256"]
        )
        value["passed"] = all(
            item for key, item in value.items() if key != "passed"
        )
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _capture_repeat_stability(
    by_path: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        first_records = first["capture_records"]
        second_records = second["capture_records"]
        record_identity = first_records == second_records
        native_digest_identity = all(
            left["native_payload_sha256"] == right["native_payload_sha256"]
            for left, right in zip(first_records, second_records, strict=True)
        )
        canonical_digest_identity = all(
            left["canonical_float32_sha256"]
            == right["canonical_float32_sha256"]
            for left, right in zip(first_records, second_records, strict=True)
        )
        value = {
            "capture_record_count": len(first_records),
            "capture_manifest_identity": (
                first["capture_manifest_sha256"]
                == second["capture_manifest_sha256"]
            ),
            "capture_record_identity": record_identity,
            "native_payload_digest_identity": native_digest_identity,
            "canonical_float32_digest_identity": canonical_digest_identity,
            "capture_event_sequence_identity": (
                first["capture_event_sequence_sha256"]
                == second["capture_event_sequence_sha256"]
            ),
        }
        value["passed"] = all(
            item
            for key, item in value.items()
            if key not in {"capture_record_count", "passed"}
        )
        result[path] = value
    result["passed"] = all(result[path]["passed"] for path in PATH_ORDER)
    return result


def _lm_head_frozen_delta_link(
    comparison: Mapping[str, Any],
    isolation: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = isolation["raw_logit_evidence"]
    delta = frozen["delta"]
    digest_links = {
        path: comparison[f"{'bf16' if path == BF16_PATH else 'fp32'}_float32_sha256"]
        == frozen["paths"][path]["comparison_vector_sha256"]
        for path in PATH_ORDER
    }
    summary_links = {
        "elements": comparison["elements"] == delta["vocabulary_elements"],
        "different_elements": (
            comparison["different_elements"] == delta["nonzero_elements"]
        ),
        "max_abs_delta": comparison["max_abs_delta"] == delta["max_abs_delta"],
        "mean_abs_delta_close": math.isclose(
            comparison["mean_abs_delta"],
            delta["mean_abs_delta"],
            rel_tol=1e-6,
            abs_tol=1e-9,
        ),
        "root_mean_square_delta_close": math.isclose(
            comparison["root_mean_square_delta"],
            delta["root_mean_square_delta"],
            rel_tol=1e-6,
            abs_tol=1e-9,
        ),
    }
    return {
        "frozen_raw_logit_vector_sha256": {
            path: frozen["paths"][path]["comparison_vector_sha256"]
            for path in PATH_ORDER
        },
        "capture_digest_links": digest_links,
        "frozen_delta_summary": delta,
        "summary_links": summary_links,
        "comparison_reduction": "stdlib_math_fsum_float64",
        "frozen_reduction": "upstream_probe_torch_reduction",
        "passed": all(digest_links.values()) and all(summary_links.values()),
    }


def _verify_sources(
    config: Mapping[str, Any],
    isolation: Mapping[str, Any],
    args: Any,
) -> dict[str, str]:
    isolation_sha256 = file_sha256(args.isolation_evidence)
    if isolation_sha256 != EXPECTED_ISOLATION_EVIDENCE_SHA256:
        raise RuntimeError("attached dtype isolation evidence hash drift")
    if (
        isolation.get("attached_dtype_isolation_version") != 1
        or isolation.get("experiment_id")
        != "fc-mvp-001-attached-dtype-isolation-v1"
        or isolation.get("dtype_isolation_gate", {}).get("passed") is not True
        or isolation.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-numerics-v1"
        or isolation.get("config_sha256") != canonical_config_sha256(config)
    ):
        raise RuntimeError("attached dtype isolation source contract drift")
    source_lineage = dict(isolation["source_lineage"])
    source_lineage["attached_dtype_isolation_evidence_sha256"] = isolation_sha256
    return source_lineage


def _capture_plan() -> dict[str, Any]:
    return {
        "scope": "target_forward_pre_registered_attached_dtype_module_outputs",
        "target_generation_step_index": TARGET_STEP_INDEX,
        "selection_basis": (
            "existing_layer0_causal_spine_plus_all_decoder_block_outputs"
        ),
        "registered_output_stages": list(REGISTERED_OUTPUT_STAGES),
        "registered_output_count": len(REGISTERED_OUTPUT_STAGES),
        "stage_contracts": [
            {"module_name": stage, **_expected_stage_contract(stage)}
            for stage in REGISTERED_OUTPUT_STAGES
        ],
        "occurrence_index": 0,
        "tensor_selection": "first_tensor_leaf_per_registered_module_output",
        "hook_capture": (
            "gpu_clone_then_post_generation_cpu_canonical_float32_summary"
        ),
        "comparison": {
            "dtype": "float32",
            "contiguous": True,
            "signed_zero_normalized": True,
            "finite_only": True,
            "exactness": "canonical_float32_exact_no_tolerance",
            "reduction": "fixed_flatten_order_stdlib_math_fsum_float64",
        },
        "serialized_module_tensor_payload": False,
        "does_not_cover": [
            "module_inputs",
            "all_tensor_leaves",
            "unregistered_functional_operations",
            "all_internal_modules_inside_decoder_layers_1_through_27",
            "earliest_difference_across_generation_history",
            "dtype_conditioned_kv_cache_history_isolation",
            "independent_causal_propagation_from_first_registered_difference",
            "low_level_cuda_kernel_identity",
        ],
    }


def _expected_stage_contract(stage: str) -> dict[str, Any]:
    hidden_shape = [1, 1, 1536]
    if stage == "model.embed_tokens":
        return {
            "module_type": "torch.nn.modules.sparse.Embedding",
            "tensor_path": "output",
            "shape": hidden_shape,
        }
    if stage in {
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
        "model.norm",
    }:
        return {
            "module_type": (
                "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm"
            ),
            "tensor_path": "output",
            "shape": hidden_shape,
        }
    if stage in {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    }:
        output_size = 1536 if stage.endswith(("q_proj", "o_proj")) else 256
        return {
            "module_type": "peft.tuners.lora.layer.Linear",
            "tensor_path": "output",
            "shape": [1, 1, output_size],
        }
    if stage in {
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    }:
        return {
            "module_type": "torch.nn.modules.linear.Linear",
            "tensor_path": "output",
            "shape": [1, 1, 8960],
        }
    if stage == "model.layers.0.mlp.down_proj":
        return {
            "module_type": "torch.nn.modules.linear.Linear",
            "tensor_path": "output",
            "shape": hidden_shape,
        }
    if stage.startswith("model.layers."):
        return {
            "module_type": (
                "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer"
            ),
            "tensor_path": "output[0]",
            "shape": hidden_shape,
        }
    if stage == "lm_head":
        return {
            "module_type": "torch.nn.modules.linear.Linear",
            "tensor_path": "output",
            "shape": [1, 1, 151936],
        }
    raise RuntimeError(f"unknown registered output stage: {stage}")


def _protocol(
    config: Mapping[str, Any],
    tokenizer: Any,
    capture_plan_sha256: str,
) -> dict[str, Any]:
    base = isolation_probe._protocol(dict(config), tokenizer)  # noqa: SLF001
    base["run_plan"] = [
        {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order_index,
        }
        for order_index, (path, repeat, run_id) in enumerate(RUN_PLAN)
    ]
    base["capture_plan_sha256"] = capture_plan_sha256
    base["capture_output_count"] = len(REGISTERED_OUTPUT_STAGES)
    return base


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
            "same_pre_registered_40_output_capture_plan",
        ],
        "supports": (
            "the first exact canonical-FP32 inequality inside the pre-registered "
            "40-output plan at the frozen target forward and a descriptive "
            "registered downstream total-dtype delta profile reaching the linked "
            "LM-head output"
        ),
        "does_not_support": [
            "first_unregistered_operation",
            "earliest_difference_across_generation_history",
            "independent_causal_propagation_from_the_first_registered_output",
            "separation_of_accumulated_dtype_conditioned_kv_cache_history",
            "unique_floating_point_or_cuda_root_cause",
            "peft_bug_claim",
            "pristine_fp32_checkpoint_comparison",
            "full_eval_generalization",
            "artifact_promotion",
            "runtime_eligibility",
        ],
        "json_only_limitation": (
            "without module tensor payloads the offline validator checks exact "
            "digests, linkage, repeat identity, frozen manifests, and summary "
            "algebra but cannot independently recompute intermediate full-tensor "
            "different counts or moments"
        ),
    }


def _constraints() -> dict[str, bool]:
    constraints = isolation_probe._constraints()  # noqa: SLF001
    constraints["module_tensor_sidecar"] = False
    constraints["module_tensor_payload"] = False
    constraints["first_registered_boundary_intervention"] = False
    return constraints


def _locked_next_action(constraints: dict[str, bool]) -> dict[str, Any]:
    return {
        "gate_id": "FC-MVP-001-attached-dtype-boundary-control-v1",
        "action": (
            "pre-register one bounded control that separates accumulated "
            "dtype-conditioned target-forward state from current-forward "
            "computation at the observed first registered output without "
            "changing attached execution form or claiming a unique low-level "
            "root cause"
        ),
        "acceptance": {
            "numerics_evidence_frozen": True,
            "one_control_pre_registered": True,
            "target_forward_identity_preserved": True,
            "observed_boundary_tested_without_post_hoc_threshold": True,
            "causal_claim_bounded": True,
        },
        "constraints": constraints,
    }


def _semantic_event_sequence(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "event_index",
        "module_name",
        "module_type",
        "occurrence_index",
        "call_index",
        "generation_step_index",
        "io_kind",
        "tensor_path",
    )
    return [{key: event[key] for key in keys} for event in events]


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


def _module_type(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonical_float32_tensor(tensor: Any) -> Any:
    canonical = tensor.detach().contiguous().cpu().float().contiguous().clone()
    if not bool(torch.isfinite(canonical).all()):
        raise RuntimeError("canonical comparison tensor is not finite")
    canonical[canonical == 0] = 0.0
    return canonical


def _tensor_bytes_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return _sha256(payload)


def _float32_sha256(tensor: Any) -> str:
    if str(tensor.dtype) != "torch.float32":
        raise RuntimeError("canonical comparison tensor must be float32")
    return _tensor_bytes_sha256(tensor)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


if __name__ == "__main__":
    raise SystemExit(main())
