"""Run the bounded same-values RMSNorm attached-dtype control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
from safetensors import safe_open  # type: ignore[import-not-found]
from transformers import AutoTokenizer  # type: ignore[import-not-found]
from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore[import-not-found]
    Qwen2RMSNorm,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_attached_dtype_boundary_control import (  # noqa: E402
    ATTACHED_DTYPE_BOUNDARY_CONTROL_VERSION,
    BF16_PATH,
    BOUNDARY_MODULE,
    BOUNDARY_MODULE_TYPE,
    CONTROL_ID,
    EXPECTED_RUN_PLAN,
    FP32_PATH,
    classify_attached_dtype_boundary_control,
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
from scripts import probe_tool_router_attached_dtype_numerics as numerics_probe  # noqa: E402

PATH_ORDER = (BF16_PATH, FP32_PATH)
ACTUAL_RUN_PLAN = (
    (BF16_PATH, 1, "bf16-attached-boundary-r1"),
    (FP32_PATH, 1, "fp32-attached-boundary-r1"),
    (FP32_PATH, 2, "fp32-attached-boundary-r2"),
    (BF16_PATH, 2, "bf16-attached-boundary-r2"),
)
CONTROL_RUN_PLAN = (
    (BF16_PATH, 1, "bf16-rmsnorm-control-r1"),
    (FP32_PATH, 1, "fp32-rmsnorm-control-r1"),
    (FP32_PATH, 2, "fp32-rmsnorm-control-r2"),
    (BF16_PATH, 2, "bf16-rmsnorm-control-r2"),
)
REPRESENTATIVE_ACTUAL_RUN = {
    BF16_PATH: "bf16-attached-boundary-r1",
    FP32_PATH: "fp32-attached-boundary-r1",
}
REPEAT_ACTUAL_RUN = {
    BF16_PATH: "bf16-attached-boundary-r2",
    FP32_PATH: "fp32-attached-boundary-r2",
}
REPRESENTATIVE_CONTROL_RUN = {
    BF16_PATH: "bf16-rmsnorm-control-r1",
    FP32_PATH: "fp32-rmsnorm-control-r1",
}
REPEAT_CONTROL_RUN = {
    BF16_PATH: "bf16-rmsnorm-control-r2",
    FP32_PATH: "fp32-rmsnorm-control-r2",
}

TARGET_STEP_INDEX = isolation_probe.TARGET_STEP_INDEX
TARGET_INPUT_TOKEN_ID = isolation_probe.TARGET_INPUT_TOKEN_ID
TARGET_CACHE_POSITION = isolation_probe.TARGET_CACHE_POSITION
TARGET_FORWARD_CALLS = isolation_probe.TARGET_FORWARD_CALLS
MAX_RESIDUAL_CUDA_BYTES = isolation_probe.MAX_RESIDUAL_CUDA_BYTES
EXPECTED_INPUT_TOKEN_COUNT = isolation_probe.EXPECTED_INPUT_TOKEN_COUNT
EXPECTED_INPUT_TOKEN_SHA256 = isolation_probe.EXPECTED_INPUT_TOKEN_SHA256
EXPECTED_NUMERICS_EVIDENCE_SHA256 = (
    "sha256:de5b048a5d254f61ab3bef1ff23f1484b07808c86a1679bc0de4ee58e8c8d7c5"
)
EXPECTED_RMSNORM_FORWARD_SOURCE_SHA256 = (
    "sha256:7d352cd525210579aabf6191da9bfc1b1086878c303fb1ea8b8ea21d0e081342"
)
EMBEDDING_TENSOR_NAME = "model.embed_tokens.weight"
RMSNORM_WEIGHT_TENSOR_NAME = "model.layers.0.input_layernorm.weight"
EMBEDDING_MODULE = "model.embed_tokens"
EMBEDDING_MODULE_TYPE = "torch.nn.modules.sparse.Embedding"
HIDDEN_SIZE = 1536
RMSNORM_EPSILON = 1e-6
CANONICALIZATION = (
    "contiguous_cpu_float32_signed_zero_normalized_finite_exact_no_tolerance"
)

ACTUAL_CAPTURE_ROLES = (
    "embedding_output",
    "rmsnorm_input",
    "rmsnorm_weight",
    "rmsnorm_output",
)
CONTROL_CAPTURE_ROLES = (
    "rmsnorm_input",
    "rmsnorm_weight",
    "rmsnorm_output",
)
ACTUAL_COMPARISON_NAMES = (
    EMBEDDING_MODULE,
    f"{BOUNDARY_MODULE}.input",
    f"{BOUNDARY_MODULE}.weight",
    BOUNDARY_MODULE,
)
CONTROL_COMPARISON_NAMES = ACTUAL_COMPARISON_NAMES[1:]
BOUNDARY_COMPARISON_KEYS = (
    "name",
    "shape",
    "elements",
    "bf16_native_dtype",
    "fp32_native_dtype",
    "comparison_dtype",
    "bf16_float32_sha256",
    "fp32_float32_sha256",
    "canonical_values_equal",
    "different_elements",
    "first_different_flat_index",
    "max_abs_delta_flat_index",
    "bf16_value_at_first_difference",
    "fp32_value_at_first_difference",
    "bf16_value_at_max_abs_delta",
    "fp32_value_at_max_abs_delta",
    "max_abs_delta",
    "mean_abs_delta",
    "root_mean_square_delta",
    "sum_abs_delta",
    "sum_squared_delta",
    "different_fraction",
    "bf16_root_mean_square",
    "fp32_root_mean_square",
    "normalized_root_mean_square_delta",
    "root_mean_square_delta_ratio_to_first_registered_difference",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--numerics-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isolation_probe._verify_output_boundary(args.output)  # noqa: SLF001

    config = isolation_probe._load_json(args.config)  # noqa: SLF001
    numerics = isolation_probe._load_json(args.numerics_evidence)  # noqa: SLF001
    source_lineage = _verify_sources(config, numerics, args)
    environment = isolation_probe._verify_environment(config)  # noqa: SLF001
    if environment != numerics["environment"]:
        raise RuntimeError("attached dtype boundary-control environment drift")
    forward_source_sha256 = _rmsnorm_forward_source_sha256()

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("evaluation digest mismatch")
    record = evaluation[0]
    if record["example_id"] != "eval-001":
        raise RuntimeError("locked boundary control requires eval-001 first")
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
    if storage_audit != numerics["storage_audit"]:
        raise RuntimeError("source storage audit drift")

    source_tensors = _load_checkpoint_sources(
        args.model_dir / config["model"]["weight_file"]
    )
    control_plan = _control_plan(source_tensors)
    control_plan_sha256 = _sha256(_canonical_json(control_plan))
    common_source = _common_source_record(
        source_tensors,
        checkpoint_weight_file_sha256=file_sha256(
            args.model_dir / config["model"]["weight_file"]
        ),
        control_plan_sha256=control_plan_sha256,
    )

    started = time.perf_counter()
    actual_runs: list[dict[str, Any]] = []
    actual_snapshots: dict[str, dict[str, Any]] = {}
    for order_index, (path, repeat, run_id) in enumerate(ACTUAL_RUN_PLAN):
        run, snapshot = _run_actual_path(
            path=path,
            repeat=repeat,
            run_id=run_id,
            order_index=order_index,
            model_dir=args.model_dir,
            adapter_dir=args.adapter_dir,
            config=config,
            encoded_cpu=encoded_cpu,
            tokenizer=tokenizer,
            control_plan_sha256=control_plan_sha256,
        )
        actual_runs.append(run)
        actual_snapshots[run_id] = snapshot

    actual_by_path = {
        path: [run for run in actual_runs if run["path"] == path]
        for path in PATH_ORDER
    }
    if any(len(actual_by_path[path]) != 2 for path in PATH_ORDER):
        raise RuntimeError("fixed actual ABBA run plan was not executed")
    frozen_references = numerics["frozen_path_references"]
    actual_path_reproduction: dict[str, Any] = {
        path: isolation_probe._path_reproduction(  # noqa: SLF001
            actual_by_path[path], frozen_references[path]
        )
        for path in PATH_ORDER
    }
    actual_path_reproduction["passed"] = all(
        actual_path_reproduction[path]["passed"] for path in PATH_ORDER
    )
    actual_path_repeat_stability = numerics_probe._path_repeat_stability(  # noqa: SLF001
        actual_by_path
    )
    actual_capture_repeat_stability = _capture_repeat_stability(actual_by_path)
    if not all(
        (
            actual_path_reproduction["passed"],
            actual_path_repeat_stability["passed"],
            actual_capture_repeat_stability["passed"],
        )
    ):
        raise RuntimeError("actual attached boundary reproduction failed")

    actual_comparisons, actual_repeat_comparisons = _paired_comparisons(
        actual_snapshots,
        representative_runs=REPRESENTATIVE_ACTUAL_RUN,
        repeat_runs=REPEAT_ACTUAL_RUN,
        roles=ACTUAL_CAPTURE_ROLES,
        names=ACTUAL_COMPARISON_NAMES,
    )
    actual_comparison_manifest_sha256 = _sha256(
        _canonical_json(actual_comparisons)
    )
    actual_repeat_manifest_sha256 = _sha256(
        _canonical_json(actual_repeat_comparisons)
    )
    actual_paired_comparison_repeat = {
        "representative_run_ids": REPRESENTATIVE_ACTUAL_RUN,
        "repeat_run_ids": REPEAT_ACTUAL_RUN,
        "representative_manifest_sha256": actual_comparison_manifest_sha256,
        "repeat_manifest_sha256": actual_repeat_manifest_sha256,
        "exact_identity": actual_comparisons == actual_repeat_comparisons,
    }
    if actual_paired_comparison_repeat["exact_identity"] is not True:
        raise RuntimeError("actual boundary comparison repeat drift")

    frozen_boundary_comparison = _bounded_comparison(
        numerics["module_comparisons"][1]
    )
    actual_boundary_comparison = actual_comparisons[-1]
    if actual_boundary_comparison != frozen_boundary_comparison:
        raise RuntimeError("frozen layer-0 RMSNorm boundary was not reproduced")

    if (
        torch.cuda.memory_allocated() > MAX_RESIDUAL_CUDA_BYTES
        or any(
            run["memory_allocated_after_release_bytes"] > MAX_RESIDUAL_CUDA_BYTES
            for run in actual_runs
        )
    ):
        raise RuntimeError("actual models were not fully released before control")

    control_runs: list[dict[str, Any]] = []
    control_snapshots: dict[str, dict[str, Any]] = {}
    for order_index, (path, repeat, run_id) in enumerate(CONTROL_RUN_PLAN):
        run, snapshot = _run_standalone_control(
            path=path,
            repeat=repeat,
            run_id=run_id,
            order_index=order_index,
            source_tensors=source_tensors,
            source_manifest_sha256=common_source["manifest_sha256"],
            control_plan_sha256=control_plan_sha256,
            forward_source_sha256=forward_source_sha256,
        )
        control_runs.append(run)
        control_snapshots[run_id] = snapshot

    control_by_path = {
        path: [run for run in control_runs if run["path"] == path]
        for path in PATH_ORDER
    }
    if any(len(control_by_path[path]) != 2 for path in PATH_ORDER):
        raise RuntimeError("fixed standalone-control ABBA run plan was not executed")
    control_capture_repeat_stability = _capture_repeat_stability(control_by_path)
    if control_capture_repeat_stability["passed"] is not True:
        raise RuntimeError("standalone RMSNorm control repeat drift")
    control_comparisons, control_repeat_comparisons = _paired_comparisons(
        control_snapshots,
        representative_runs=REPRESENTATIVE_CONTROL_RUN,
        repeat_runs=REPEAT_CONTROL_RUN,
        roles=CONTROL_CAPTURE_ROLES,
        names=CONTROL_COMPARISON_NAMES,
    )
    control_comparison_manifest_sha256 = _sha256(
        _canonical_json(control_comparisons)
    )
    control_repeat_manifest_sha256 = _sha256(
        _canonical_json(control_repeat_comparisons)
    )
    control_paired_comparison_repeat = {
        "representative_run_ids": REPRESENTATIVE_CONTROL_RUN,
        "repeat_run_ids": REPEAT_CONTROL_RUN,
        "representative_manifest_sha256": control_comparison_manifest_sha256,
        "repeat_manifest_sha256": control_repeat_manifest_sha256,
        "exact_identity": control_comparisons == control_repeat_comparisons,
    }
    if control_paired_comparison_repeat["exact_identity"] is not True:
        raise RuntimeError("standalone control comparison repeat drift")

    control_boundary_comparison = control_comparisons[-1]
    classification_runs = _classification_runs(
        actual_runs=actual_runs,
        control_runs=control_runs,
        actual_path_reproduction=actual_path_reproduction,
        actual_boundary_comparison=actual_boundary_comparison,
        control_boundary_comparison=control_boundary_comparison,
        control_plan=control_plan,
    )
    source_adapter_unchanged = (
        directory_artifact_manifest(args.adapter_dir) == numerics["adapter_files"]
    )
    source_model_unchanged = (
        file_sha256(args.model_dir / config["model"]["weight_file"])
        == numerics["model_weight_sha256"]
    )
    eval_digest_unchanged = fixture_digest(evaluation) == numerics["eval_digest"]
    prompt_digest_unchanged = file_sha256(prompt_path) == numerics["prompt_sha256"]
    checkpoint_sources_unchanged = all(
        (
            source_model_unchanged,
            common_source["checkpoint_weight_file_sha256"]
            == numerics["model_weight_sha256"],
            _source_records_match_plan(common_source, control_plan),
        )
    )
    target_forward_identity_preserved = all(
        run["target_alignment_passed"] for run in actual_runs
    )
    attached_execution_form_fixed = all(
        run["materialization_form"] == "attached_factorized_lora"
        and run["path_protocol_passed"]
        for run in actual_runs
    )
    source_evidence_locked = file_sha256(args.numerics_evidence) == (
        EXPECTED_NUMERICS_EVIDENCE_SHA256
    )
    boundary_analysis = classify_attached_dtype_boundary_control(
        control_plan=control_plan,
        runs=classification_runs,
        frozen_boundary_comparison=frozen_boundary_comparison,
        actual_boundary_comparison=actual_boundary_comparison,
        control_boundary_comparison=control_boundary_comparison,
        source_evidence_locked=source_evidence_locked,
        target_forward_identity_preserved=target_forward_identity_preserved,
        attached_execution_form_fixed=attached_execution_form_fixed,
        checkpoint_sources_unchanged=checkpoint_sources_unchanged,
    )

    fresh_load_memory_isolated = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"]
        <= MAX_RESIDUAL_CUDA_BYTES
        for run in [*actual_runs, *control_runs]
    )
    acceptance = {
        "upstream_numerics_evidence_locked": source_evidence_locked,
        "frozen_input_reproduced": True,
        "actual_attached_paths_reproduced": actual_path_reproduction["passed"],
        "actual_boundary_capture_repeat_stable": (
            actual_capture_repeat_stability["passed"]
        ),
        "frozen_boundary_reproduced": actual_boundary_comparison
        == frozen_boundary_comparison,
        "one_control_pre_registered": control_plan["intervention_count"] == 1,
        "standalone_control_abba_executed": len(control_runs) == 4,
        "standalone_control_repeat_stable": (
            control_capture_repeat_stability["passed"]
        ),
        "same_checkpoint_sources_used": checkpoint_sources_unchanged,
        "target_forward_identity_preserved": target_forward_identity_preserved,
        "attached_execution_form_fixed": attached_execution_form_fixed,
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": source_model_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": prompt_digest_unchanged,
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "control_protocol_completed_outcome_neutrally": boundary_analysis[
            "protocol_completed"
        ],
        "module_tensor_payload_absent": True,
    }
    boundary_control_gate = {
        "actual_abba_reproduced": actual_path_reproduction["passed"],
        "actual_path_repeat_stable": actual_path_repeat_stability["passed"],
        "actual_capture_repeat_stable": actual_capture_repeat_stability["passed"],
        "actual_boundary_reproduced": boundary_analysis[
            "actual_boundary_reproduced"
        ],
        "control_plan_executed": all(
            run["control_plan_passed"] for run in control_runs
        ),
        "control_abba_repeat_stable": control_capture_repeat_stability["passed"],
        "same_values_preconditions_passed": boundary_analysis[
            "same_values_preconditions_passed"
        ],
        "protocol_completed": boundary_analysis["protocol_completed"],
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "module_tensor_payload_absent": True,
    }
    boundary_control_gate["passed"] = all(boundary_control_gate.values())
    if boundary_control_gate["passed"] is not True or not all(acceptance.values()):
        raise RuntimeError(
            "attached dtype boundary-control protocol failed: "
            f"gate={boundary_control_gate!r}, acceptance={acceptance!r}"
        )

    constraints = _constraints()
    result = {
        "attached_dtype_boundary_control_version": (
            ATTACHED_DTYPE_BOUNDARY_CONTROL_VERSION
        ),
        "experiment_id": "fc-mvp-001-attached-dtype-boundary-control-v1",
        "source_experiment_id": numerics["source_experiment_id"],
        "source_gate_experiment_id": numerics["experiment_id"],
        "source_lineage": source_lineage,
        "training_lock_sha256": numerics["training_lock_sha256"],
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": numerics["adapter_files"],
        "model_weight_sha256": numerics["model_weight_sha256"],
        "prompt_sha256": numerics["prompt_sha256"],
        "eval_digest": numerics["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "environment": environment,
        "protocol": _protocol(config, tokenizer, forward_source_sha256),
        "control_plan": control_plan,
        "control_plan_sha256": control_plan_sha256,
        "common_source": common_source,
        "frozen_path_references": frozen_references,
        "actual_runs": actual_runs,
        "actual_path_repeat_stability": actual_path_repeat_stability,
        "actual_path_reproduction": actual_path_reproduction,
        "actual_capture_repeat_stability": actual_capture_repeat_stability,
        "actual_comparisons": actual_comparisons,
        "actual_comparison_manifest_sha256": (
            actual_comparison_manifest_sha256
        ),
        "actual_paired_comparison_repeat": actual_paired_comparison_repeat,
        "frozen_boundary_comparison": frozen_boundary_comparison,
        "control_runs": control_runs,
        "control_capture_repeat_stability": control_capture_repeat_stability,
        "control_comparisons": control_comparisons,
        "control_comparison_manifest_sha256": (
            control_comparison_manifest_sha256
        ),
        "control_paired_comparison_repeat": control_paired_comparison_repeat,
        "classification_runs": classification_runs,
        "boundary_analysis": boundary_analysis,
        "classification": boundary_analysis["classification"],
        "causal_scope": _causal_scope(boundary_analysis),
        "boundary_control_gate": boundary_control_gate,
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(
            run["peak_gpu_memory_bytes"] for run in [*actual_runs, *control_runs]
        ),
        "module_tensor_payload_saved": False,
        "module_tensor_sidecar_allowed": False,
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _locked_next_action(
            boundary_analysis,
            constraints,
        ),
        "runtime_eligible": False,
        "runtime_eligibility_reason": boundary_analysis["classification"],
        "offline": True,
    }
    isolation_probe._require_finite_json(result, "$")  # noqa: SLF001
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _verify_sources(
    config: Mapping[str, Any],
    numerics: Mapping[str, Any],
    args: Any,
) -> dict[str, str]:
    numerics_sha256 = file_sha256(args.numerics_evidence)
    if numerics_sha256 != EXPECTED_NUMERICS_EVIDENCE_SHA256:
        raise RuntimeError("attached dtype numerics evidence hash drift")
    if (
        numerics.get("attached_dtype_numerics_version") != 1
        or numerics.get("experiment_id")
        != "fc-mvp-001-attached-dtype-numerics-v1"
        or numerics.get("numerics_gate", {}).get("passed") is not True
        or numerics.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-boundary-control-v1"
        or numerics.get("config_sha256") != canonical_config_sha256(config)
        or numerics.get("adapter_files")
        != directory_artifact_manifest(args.adapter_dir)
    ):
        raise RuntimeError("attached dtype numerics source contract drift")
    expected_helper_plan = tuple((path, repeat) for path, repeat, _ in ACTUAL_RUN_PLAN)
    expected_control_plan = tuple(
        (path, repeat) for path, repeat, _ in CONTROL_RUN_PLAN
    )
    if (
        expected_helper_plan != EXPECTED_RUN_PLAN
        or expected_control_plan != EXPECTED_RUN_PLAN
    ):
        raise RuntimeError("helper/probe ABBA protocol drift")
    source_lineage = dict(numerics["source_lineage"])
    source_lineage["attached_dtype_numerics_evidence_sha256"] = numerics_sha256
    return source_lineage


def _rmsnorm_forward_source_sha256() -> str:
    module_type = _module_type(Qwen2RMSNorm(HIDDEN_SIZE, eps=RMSNORM_EPSILON))
    if module_type != BOUNDARY_MODULE_TYPE:
        raise RuntimeError(f"Qwen2 RMSNorm class drift: {module_type}")
    source = inspect.getsource(Qwen2RMSNorm.forward).encode("utf-8")
    digest = _sha256(source)
    if digest != EXPECTED_RMSNORM_FORWARD_SOURCE_SHA256:
        raise RuntimeError(f"Qwen2 RMSNorm forward source drift: {digest}")
    return digest


def _load_checkpoint_sources(weight_path: Path) -> dict[str, Any]:
    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        required = {EMBEDDING_TENSOR_NAME, RMSNORM_WEIGHT_TENSOR_NAME}
        if not required.issubset(keys):
            raise RuntimeError(f"checkpoint boundary sources missing: {required - keys}")
        embedding_slice = checkpoint.get_slice(EMBEDDING_TENSOR_NAME)
        if list(embedding_slice.get_shape()) != [151936, HIDDEN_SIZE]:
            raise RuntimeError("checkpoint embedding shape drift")
        embedding_row = (
            embedding_slice[TARGET_INPUT_TOKEN_ID : TARGET_INPUT_TOKEN_ID + 1]
            .reshape(1, 1, HIDDEN_SIZE)
            .contiguous()
            .clone()
        )
        rmsnorm_weight = (
            checkpoint.get_tensor(RMSNORM_WEIGHT_TENSOR_NAME).contiguous().clone()
        )
    if (
        str(embedding_row.dtype) != "torch.bfloat16"
        or list(embedding_row.shape) != [1, 1, HIDDEN_SIZE]
        or str(rmsnorm_weight.dtype) != "torch.bfloat16"
        or list(rmsnorm_weight.shape) != [HIDDEN_SIZE]
        or not bool(torch.isfinite(embedding_row).all())
        or not bool(torch.isfinite(rmsnorm_weight).all())
    ):
        raise RuntimeError("checkpoint boundary source contract drift")
    return {
        "rmsnorm_input": embedding_row,
        "rmsnorm_weight": rmsnorm_weight,
    }


def _control_plan(source_tensors: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "control_id": CONTROL_ID,
        "intervention_count": 1,
        "module_name": BOUNDARY_MODULE,
        "module_type": BOUNDARY_MODULE_TYPE,
        "variance_epsilon": RMSNORM_EPSILON,
        "input_shape": [1, 1, HIDDEN_SIZE],
        "weight_shape": [HIDDEN_SIZE],
        "common_input_float32_sha256": _float32_sha256(
            _canonical_float32_tensor(source_tensors["rmsnorm_input"])
        ),
        "common_weight_float32_sha256": _float32_sha256(
            _canonical_float32_tensor(source_tensors["rmsnorm_weight"])
        ),
        "canonicalization": CANONICALIZATION,
        "dtype_arms": [
            {
                "path": BF16_PATH,
                "input_dtype": "bfloat16",
                "weight_dtype": "bfloat16",
                "output_dtype": "bfloat16",
            },
            {
                "path": FP32_PATH,
                "input_dtype": "float32",
                "weight_dtype": "float32",
                "output_dtype": "float32",
            },
        ],
        "serialized_tensor_payload": False,
        "module_tensor_sidecar_allowed": False,
    }


def _common_source_record(
    source_tensors: Mapping[str, Any],
    *,
    checkpoint_weight_file_sha256: str,
    control_plan_sha256: str,
) -> dict[str, Any]:
    records = [
        _tensor_record(
            source_tensors["rmsnorm_input"],
            record_id="checkpoint_source|rmsnorm_input",
            event_index=0,
            capture_scope="checkpoint_source",
            semantic_role="rmsnorm_input",
            module_name=EMBEDDING_MODULE,
            module_type=EMBEDDING_MODULE_TYPE,
            occurrence_index=0,
            call_index=None,
            generation_step_index=None,
            io_kind="source",
            tensor_path=f"checkpoint.{EMBEDDING_TENSOR_NAME}[{TARGET_INPUT_TOKEN_ID}]",
            control_plan_sha256=control_plan_sha256,
        ),
        _tensor_record(
            source_tensors["rmsnorm_weight"],
            record_id="checkpoint_source|rmsnorm_weight",
            event_index=1,
            capture_scope="checkpoint_source",
            semantic_role="rmsnorm_weight",
            module_name=BOUNDARY_MODULE,
            module_type=BOUNDARY_MODULE_TYPE,
            occurrence_index=0,
            call_index=None,
            generation_step_index=None,
            io_kind="parameter_source",
            tensor_path=f"checkpoint.{RMSNORM_WEIGHT_TENSOR_NAME}",
            control_plan_sha256=control_plan_sha256,
        ),
    ]
    return {
        "checkpoint_weight_file_sha256": checkpoint_weight_file_sha256,
        "embedding_tensor_name": EMBEDDING_TENSOR_NAME,
        "embedding_row_index": TARGET_INPUT_TOKEN_ID,
        "rmsnorm_weight_tensor_name": RMSNORM_WEIGHT_TENSOR_NAME,
        "storage_dtype": "bfloat16",
        "records": records,
        "record_count": len(records),
        "manifest_sha256": _sha256(_canonical_json(records)),
    }


def _run_actual_path(
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
    control_plan_sha256: str,
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
            target_alignment,
            capture_records,
            snapshot,
        ) = _instrumented_actual_generate(
            model=model,
            path=path,
            encoded_cpu=encoded_cpu,
            config=config,
            tokenizer=tokenizer,
            control_plan_sha256=control_plan_sha256,
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
        target_alignment_passed = _target_alignment_passed(
            target_alignment,
            path=path,
        )
        capture_plan_passed = _actual_capture_plan_passed(
            capture_records,
            control_plan_sha256=control_plan_sha256,
        )
        if not all(
            (path_protocol_passed, target_alignment_passed, capture_plan_passed)
        ):
            raise RuntimeError(
                f"actual boundary protocol failed for {run_id}: "
                f"path={path_protocol_passed}, target={target_alignment_passed}, "
                f"capture={capture_plan_passed}, records={capture_records!r}"
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
            "capture_record_count": len(capture_records),
            "capture_event_sequence_sha256": _capture_event_sequence_sha256(
                capture_records
            ),
            "capture_manifest_sha256": _sha256(_canonical_json(capture_records)),
            "control_plan_sha256": control_plan_sha256,
            "capture_plan_passed": capture_plan_passed,
            "path_protocol_passed": path_protocol_passed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "memory_allocated_before_load_bytes": allocated_before,
        }
    finally:
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} retained model-scale CUDA memory: {allocated_after}"
        )
    if run is None or snapshot is None:
        raise RuntimeError(f"actual boundary run did not complete: {run_id}")
    run["memory_allocated_after_release_bytes"] = allocated_after
    return run, snapshot


def _instrumented_actual_generate(
    *,
    model: Any,
    path: str,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
    control_plan_sha256: str,
) -> tuple[
    list[int],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if torch.is_autocast_enabled():
        raise RuntimeError("actual generation must not run under autocast")
    causal = isolation_probe._causal_model(model)  # noqa: SLF001
    modules = dict(causal.named_modules())
    if EMBEDDING_MODULE not in modules or BOUNDARY_MODULE not in modules:
        raise RuntimeError("registered boundary modules missing")
    embedding = modules[EMBEDDING_MODULE]
    boundary = modules[BOUNDARY_MODULE]
    if (
        _module_type(embedding) != EMBEDDING_MODULE_TYPE
        or _module_type(boundary) != BOUNDARY_MODULE_TYPE
        or float(boundary.variance_epsilon) != RMSNORM_EPSILON
    ):
        raise RuntimeError("registered boundary module contract drift")

    state: dict[str, Any] = {
        "call_index": -1,
        "active": False,
        "target_count": 0,
        "alignment": {},
    }
    captures: dict[str, dict[str, Any]] = {}
    capture_order: list[str] = []
    handles: list[Any] = []

    def stash(
        role: str,
        tensor: Any,
        *,
        module_name: str,
        module_type: str,
        io_kind: str,
        tensor_path: str,
    ) -> None:
        if role in captures:
            raise RuntimeError(f"duplicate actual boundary capture: {role}")
        captures[role] = {
            "tensor": tensor.detach().clone(),
            "event_index": len(capture_order),
            "module_name": module_name,
            "module_type": module_type,
            "io_kind": io_kind,
            "tensor_path": tensor_path,
        }
        capture_order.append(role)

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

    def embedding_hook(module: Any, _args: Any, output: Any) -> None:
        if not state["active"]:
            return
        tensor, tensor_path = numerics_probe._first_tensor_leaf(  # noqa: SLF001
            output, "output"
        )
        stash(
            "embedding_output",
            tensor,
            module_name=EMBEDDING_MODULE,
            module_type=_module_type(module),
            io_kind="output",
            tensor_path=tensor_path,
        )

    def boundary_pre_hook(module: Any, hook_args: Any, kwargs: Any) -> None:
        if not state["active"]:
            return
        tensor, tensor_path = _module_input_tensor(hook_args, kwargs)
        stash(
            "rmsnorm_input",
            tensor,
            module_name=BOUNDARY_MODULE,
            module_type=_module_type(module),
            io_kind="input",
            tensor_path=tensor_path,
        )
        stash(
            "rmsnorm_weight",
            module.weight,
            module_name=BOUNDARY_MODULE,
            module_type=_module_type(module),
            io_kind="parameter",
            tensor_path="module.weight",
        )

    def boundary_post_hook(
        module: Any,
        _args: Any,
        _kwargs: Any,
        output: Any,
    ) -> None:
        if not state["active"]:
            return
        tensor, tensor_path = numerics_probe._first_tensor_leaf(  # noqa: SLF001
            output, "output"
        )
        stash(
            "rmsnorm_output",
            tensor,
            module_name=BOUNDARY_MODULE,
            module_type=_module_type(module),
            io_kind="output",
            tensor_path=tensor_path,
        )

    handles.extend(
        (
            causal.register_forward_pre_hook(causal_pre_hook, with_kwargs=True),
            causal.register_forward_hook(causal_post_hook, with_kwargs=True),
            embedding.register_forward_hook(embedding_hook),
            boundary.register_forward_pre_hook(boundary_pre_hook, with_kwargs=True),
            boundary.register_forward_hook(boundary_post_hook, with_kwargs=True),
        )
    )
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
    score_summary, _target_score = isolation_probe._summarize_trace(  # noqa: SLF001
        generated.scores,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    raw_summary, _target_raw_logit = isolation_probe._summarize_trace(  # noqa: SLF001
        generated.logits,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    if generated.past_key_values is None:
        raise RuntimeError("cached generation did not return past_key_values")
    if state["target_count"] != 1:
        raise RuntimeError(f"target forward count drift: {state['target_count']!r}")
    state["alignment"]["causal_forward_calls"] = state["call_index"] + 1
    if tuple(capture_order) != ACTUAL_CAPTURE_ROLES:
        raise RuntimeError(f"actual boundary capture order drift: {capture_order!r}")

    capture_records: list[dict[str, Any]] = []
    canonical_tensors: dict[str, Any] = {}
    for role in ACTUAL_CAPTURE_ROLES:
        captured = captures[role]
        native = captured["tensor"].detach().contiguous().cpu()
        record = _tensor_record(
            native,
            record_id=f"actual_target_forward|{role}",
            event_index=captured["event_index"],
            capture_scope="actual_target_forward",
            semantic_role=role,
            module_name=captured["module_name"],
            module_type=captured["module_type"],
            occurrence_index=0,
            call_index=TARGET_STEP_INDEX,
            generation_step_index=TARGET_STEP_INDEX,
            io_kind=captured["io_kind"],
            tensor_path=captured["tensor_path"],
            control_plan_sha256=control_plan_sha256,
        )
        capture_records.append(record)
        canonical_tensors[role] = _canonical_float32_tensor(native)
    captures.clear()
    torch.cuda.empty_cache()
    generation_trace = {
        "step_count": len(token_ids),
        "vocabulary_size": expected_shape[1],
        "cache_returned": True,
        "scores": score_summary,
        "raw_logits": raw_summary,
    }
    return (
        token_ids,
        generation_trace,
        dict(state["alignment"]),
        capture_records,
        {
            "path": path,
            "records": {
                value["semantic_role"]: value for value in capture_records
            },
            "tensors": canonical_tensors,
        },
    )


def _run_standalone_control(
    *,
    path: str,
    repeat: int,
    run_id: str,
    order_index: int,
    source_tensors: Mapping[str, Any],
    source_manifest_sha256: str,
    control_plan_sha256: str,
    forward_source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} exceeded residual CUDA ceiling: {allocated_before}"
        )
    if torch.is_autocast_enabled():
        raise RuntimeError("standalone control must not run under autocast")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("standalone control must not run with TF32")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    module: Any | None = None
    run: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    try:
        dtype = isolation_probe._torch_dtype_for_path(path)  # noqa: SLF001
        module = Qwen2RMSNorm(HIDDEN_SIZE, eps=RMSNORM_EPSILON).to(
            device="cuda",
            dtype=dtype,
        )
        module.eval()
        if (
            _module_type(module) != BOUNDARY_MODULE_TYPE
            or float(module.variance_epsilon) != RMSNORM_EPSILON
            or forward_source_sha256 != EXPECTED_RMSNORM_FORWARD_SOURCE_SHA256
        ):
            raise RuntimeError("fresh standalone RMSNorm module contract drift")
        control_input = source_tensors["rmsnorm_input"].to(
            device="cuda", dtype=dtype
        )
        source_weight = source_tensors["rmsnorm_weight"].to(
            device="cuda", dtype=dtype
        )
        with torch.inference_mode():
            module.weight.copy_(source_weight)
            weight_before = module.weight.detach().clone()
            control_output = module(control_input)
            weight_after = module.weight.detach().clone()
        if not torch.equal(weight_before, weight_after):
            raise RuntimeError("standalone control changed its source weight")
        native_tensors = {
            "rmsnorm_input": control_input.detach().contiguous().cpu(),
            "rmsnorm_weight": weight_before.detach().contiguous().cpu(),
            "rmsnorm_output": control_output.detach().contiguous().cpu(),
        }
        paths = {
            "rmsnorm_input": ("input", "control.input"),
            "rmsnorm_weight": ("parameter", "module.weight"),
            "rmsnorm_output": ("output", "control.output"),
        }
        capture_records = [
            _tensor_record(
                native_tensors[role],
                record_id=f"standalone_control|{role}",
                event_index=index,
                capture_scope="standalone_control",
                semantic_role=role,
                module_name=BOUNDARY_MODULE,
                module_type=BOUNDARY_MODULE_TYPE,
                occurrence_index=0,
                call_index=None,
                generation_step_index=None,
                io_kind=paths[role][0],
                tensor_path=paths[role][1],
                control_plan_sha256=control_plan_sha256,
            )
            for index, role in enumerate(CONTROL_CAPTURE_ROLES)
        ]
        source_input_sha256 = _float32_sha256(
            _canonical_float32_tensor(source_tensors["rmsnorm_input"])
        )
        source_weight_sha256 = _float32_sha256(
            _canonical_float32_tensor(source_tensors["rmsnorm_weight"])
        )
        record_by_role = {
            value["semantic_role"]: value for value in capture_records
        }
        common_source_roundtrip_exact = (
            record_by_role["rmsnorm_input"]["canonical_float32_sha256"]
            == source_input_sha256
            and record_by_role["rmsnorm_weight"]["canonical_float32_sha256"]
            == source_weight_sha256
        )
        control_weight_unchanged = (
            _tensor_bytes_sha256(weight_before)
            == _tensor_bytes_sha256(weight_after)
            and _float32_sha256(_canonical_float32_tensor(weight_before))
            == _float32_sha256(_canonical_float32_tensor(weight_after))
        )
        control_plan_passed = (
            tuple(record_by_role) == CONTROL_CAPTURE_ROLES
            and all(
                value["control_plan_sha256"] == control_plan_sha256
                for value in capture_records
            )
            and common_source_roundtrip_exact
            and control_weight_unchanged
        )
        if not control_plan_passed:
            raise RuntimeError(f"standalone control plan failed: {run_id}")
        torch.cuda.synchronize()
        run = {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order_index,
            "control_id": CONTROL_ID,
            "fresh_standalone_module": True,
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "forward_source_sha256": forward_source_sha256,
            "variance_epsilon": RMSNORM_EPSILON,
            "training": module.training,
            "autocast_enabled": torch.is_autocast_enabled(),
            "tf32_enabled": (
                torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32
            ),
            "cache_arguments_present": False,
            "output_injected": False,
            "source_manifest_sha256": source_manifest_sha256,
            "capture_records": capture_records,
            "capture_record_count": len(capture_records),
            "capture_event_sequence_sha256": _capture_event_sequence_sha256(
                capture_records
            ),
            "capture_manifest_sha256": _sha256(_canonical_json(capture_records)),
            "control_plan_sha256": control_plan_sha256,
            "control_plan_passed": control_plan_passed,
            "common_source_roundtrip_exact": common_source_roundtrip_exact,
            "control_weight_unchanged": control_weight_unchanged,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "memory_allocated_before_load_bytes": allocated_before,
        }
        snapshot = {
            "path": path,
            "records": record_by_role,
            "tensors": {
                role: _canonical_float32_tensor(native_tensors[role])
                for role in CONTROL_CAPTURE_ROLES
            },
        }
    finally:
        module = None
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} retained CUDA memory: {allocated_after}"
        )
    if run is None or snapshot is None:
        raise RuntimeError(f"standalone control did not complete: {run_id}")
    run["memory_allocated_after_release_bytes"] = allocated_after
    return run, snapshot


def _tensor_record(
    tensor: Any,
    *,
    record_id: str,
    event_index: int,
    capture_scope: str,
    semantic_role: str,
    module_name: str,
    module_type: str,
    occurrence_index: int,
    call_index: int | None,
    generation_step_index: int | None,
    io_kind: str,
    tensor_path: str,
    control_plan_sha256: str,
) -> dict[str, Any]:
    native = tensor.detach().contiguous().cpu()
    canonical = _canonical_float32_tensor(native)
    elements = int(canonical.numel())
    finite_elements = int(torch.isfinite(canonical).sum())
    if finite_elements != elements:
        raise RuntimeError(f"non-finite boundary capture: {record_id}")
    result = {
        "record_id": record_id,
        "event_index": event_index,
        "capture_scope": capture_scope,
        "semantic_role": semantic_role,
        "module_name": module_name,
        "module_type": module_type,
        "occurrence_index": occurrence_index,
        "call_index": call_index,
        "generation_step_index": generation_step_index,
        "io_kind": io_kind,
        "tensor_path": tensor_path,
        "native_dtype": str(native.dtype).removeprefix("torch."),
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
        "control_plan_sha256": control_plan_sha256,
    }
    result["record_sha256"] = _sha256(_canonical_json(result))
    return result


def _target_alignment_passed(value: Mapping[str, Any], *, path: str) -> bool:
    return (
        value.get("call_index") == TARGET_STEP_INDEX
        and value.get("generation_step_index") == TARGET_STEP_INDEX
        and value.get("input_token_ids") == [TARGET_INPUT_TOKEN_ID]
        and value.get("input_shape") == [1, 1]
        and value.get("cache_position") == [TARGET_CACHE_POSITION]
        and value.get("position_ids") == [TARGET_CACHE_POSITION]
        and value.get("past_length") == TARGET_CACHE_POSITION
        and value.get("causal_forward_calls") == TARGET_FORWARD_CALLS
        and isolation_probe._dtype_name_for_path(path)  # noqa: SLF001
        in {"bfloat16", "float32"}
    )


def _actual_capture_plan_passed(
    records: list[dict[str, Any]],
    *,
    control_plan_sha256: str,
) -> bool:
    by_role = {value["semantic_role"]: value for value in records}
    expected_dtype = by_role.get("rmsnorm_output", {}).get("native_dtype")
    expected = {
        "embedding_output": (
            EMBEDDING_MODULE,
            EMBEDDING_MODULE_TYPE,
            "output",
            "output",
            [1, 1, HIDDEN_SIZE],
        ),
        "rmsnorm_input": (
            BOUNDARY_MODULE,
            BOUNDARY_MODULE_TYPE,
            "input",
            "args[0]",
            [1, 1, HIDDEN_SIZE],
        ),
        "rmsnorm_weight": (
            BOUNDARY_MODULE,
            BOUNDARY_MODULE_TYPE,
            "parameter",
            "module.weight",
            [HIDDEN_SIZE],
        ),
        "rmsnorm_output": (
            BOUNDARY_MODULE,
            BOUNDARY_MODULE_TYPE,
            "output",
            "output",
            [1, 1, HIDDEN_SIZE],
        ),
    }
    return (
        tuple(by_role) == ACTUAL_CAPTURE_ROLES
        and len(records) == len(ACTUAL_CAPTURE_ROLES)
        and expected_dtype in {"bfloat16", "float32"}
        and all(
            (
                value["module_name"],
                value["module_type"],
                value["io_kind"],
                value["tensor_path"],
                value["native_shape"],
            )
            == expected[value["semantic_role"]]
            and value["native_dtype"] == expected_dtype
            and value["call_index"] == TARGET_STEP_INDEX
            and value["generation_step_index"] == TARGET_STEP_INDEX
            and value["control_plan_sha256"] == control_plan_sha256
            for value in records
        )
    )


def _capture_repeat_stability(
    by_path: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        first_records = first["capture_records"]
        second_records = second["capture_records"]
        value = {
            "capture_record_count": len(first_records),
            "capture_manifest_identity": first["capture_manifest_sha256"]
            == second["capture_manifest_sha256"],
            "capture_record_identity": first_records == second_records,
            "native_payload_digest_identity": all(
                left["native_payload_sha256"] == right["native_payload_sha256"]
                for left, right in zip(first_records, second_records, strict=True)
            ),
            "canonical_float32_digest_identity": all(
                left["canonical_float32_sha256"]
                == right["canonical_float32_sha256"]
                for left, right in zip(first_records, second_records, strict=True)
            ),
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


def _paired_comparisons(
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    representative_runs: Mapping[str, str],
    repeat_runs: Mapping[str, str],
    roles: tuple[str, ...],
    names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def compare(run_ids: Mapping[str, str]) -> list[dict[str, Any]]:
        bf16 = snapshots[run_ids[BF16_PATH]]
        fp32 = snapshots[run_ids[FP32_PATH]]
        return [
            _compare_role(
                name,
                bf16["tensors"][role],
                fp32["tensors"][role],
                bf16["records"][role],
                fp32["records"][role],
            )
            for role, name in zip(roles, names, strict=True)
        ]

    return compare(representative_runs), compare(repeat_runs)


def _compare_role(
    name: str,
    bf16_tensor: Any,
    fp32_tensor: Any,
    bf16_record: Mapping[str, Any],
    fp32_record: Mapping[str, Any],
) -> dict[str, Any]:
    result = numerics_probe._compare_stage(  # noqa: SLF001
        name,
        bf16_tensor,
        fp32_tensor,
        bf16_record,
        fp32_record,
    )
    result["root_mean_square_delta_ratio_to_first_registered_difference"] = (
        0.0 if result["canonical_values_equal"] else 1.0
    )
    return _bounded_comparison(result)


def _bounded_comparison(source: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(BOUNDARY_COMPARISON_KEYS) - set(source)
    if missing:
        raise RuntimeError(f"boundary comparison fields missing: {missing!r}")
    return {key: source[key] for key in BOUNDARY_COMPARISON_KEYS}


def _classification_runs(
    *,
    actual_runs: list[dict[str, Any]],
    control_runs: list[dict[str, Any]],
    actual_path_reproduction: Mapping[str, Any],
    actual_boundary_comparison: Mapping[str, Any],
    control_boundary_comparison: Mapping[str, Any],
    control_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for actual, control in zip(actual_runs, control_runs, strict=True):
        if (
            actual["path"] != control["path"]
            or actual["repeat"] != control["repeat"]
            or actual["order_index"] != control["order_index"]
        ):
            raise RuntimeError("actual/control ABBA pairing drift")
        actual_records = {
            item["semantic_role"]: item for item in actual["capture_records"]
        }
        control_records = {
            item["semantic_role"]: item for item in control["capture_records"]
        }
        prefix = "bf16" if actual["path"] == BF16_PATH else "fp32"
        result.append(
            {
                "run_id": f"{actual['run_id']}|{control['run_id']}",
                "path": actual["path"],
                "repeat": actual["repeat"],
                "order_index": actual["order_index"],
                "fresh_load": actual["fresh_load"]
                and control["fresh_standalone_module"],
                "frozen_path_reproduced": actual_path_reproduction[
                    actual["path"]
                ]["passed"],
                "target_forward_aligned": actual["target_alignment_passed"],
                "actual_boundary_reproduced": actual_records["rmsnorm_output"][
                    "canonical_float32_sha256"
                ]
                == actual_boundary_comparison[f"{prefix}_float32_sha256"],
                "control_id": CONTROL_ID,
                "control_executed": control["control_plan_passed"],
                "control_standalone": control["fresh_standalone_module"],
                "common_source_roundtrip_exact": (
                    actual_records["rmsnorm_input"]["canonical_float32_sha256"]
                    == control_plan["common_input_float32_sha256"]
                    and actual_records["rmsnorm_weight"][
                        "canonical_float32_sha256"
                    ]
                    == control_plan["common_weight_float32_sha256"]
                    and control["common_source_roundtrip_exact"]
                ),
                "control_weight_unchanged": control["control_weight_unchanged"],
                "autocast_enabled": control["autocast_enabled"],
                "actual_input_native_dtype": actual_records["rmsnorm_input"][
                    "native_dtype"
                ],
                "actual_weight_native_dtype": actual_records["rmsnorm_weight"][
                    "native_dtype"
                ],
                "actual_output_native_dtype": actual_records["rmsnorm_output"][
                    "native_dtype"
                ],
                "actual_input_float32_sha256": actual_records["rmsnorm_input"][
                    "canonical_float32_sha256"
                ],
                "actual_weight_float32_sha256": actual_records["rmsnorm_weight"][
                    "canonical_float32_sha256"
                ],
                "actual_output_float32_sha256": actual_records["rmsnorm_output"][
                    "canonical_float32_sha256"
                ],
                "control_input_native_dtype": control_records["rmsnorm_input"][
                    "native_dtype"
                ],
                "control_weight_native_dtype": control_records["rmsnorm_weight"][
                    "native_dtype"
                ],
                "control_output_native_dtype": control_records["rmsnorm_output"][
                    "native_dtype"
                ],
                "control_input_float32_sha256": control_records["rmsnorm_input"][
                    "canonical_float32_sha256"
                ],
                "control_weight_float32_sha256": control_records[
                    "rmsnorm_weight"
                ]["canonical_float32_sha256"],
                "control_output_float32_sha256": control_records[
                    "rmsnorm_output"
                ]["canonical_float32_sha256"],
                "control_cache_arguments_present": control[
                    "cache_arguments_present"
                ],
                "control_output_injected": control["output_injected"],
                "module_tensor_payload_saved": False,
                "module_tensor_sidecar_saved": False,
            }
        )
        if result[-1]["control_output_float32_sha256"] != (
            control_boundary_comparison[f"{prefix}_float32_sha256"]
        ):
            raise RuntimeError("control output comparison endpoint link failed")
    return result


def _source_records_match_plan(
    common_source: Mapping[str, Any],
    control_plan: Mapping[str, Any],
) -> bool:
    by_role = {
        item["semantic_role"]: item for item in common_source["records"]
    }
    return (
        tuple(by_role) == ("rmsnorm_input", "rmsnorm_weight")
        and by_role["rmsnorm_input"]["canonical_float32_sha256"]
        == control_plan["common_input_float32_sha256"]
        and by_role["rmsnorm_weight"]["canonical_float32_sha256"]
        == control_plan["common_weight_float32_sha256"]
    )


def _capture_event_sequence_sha256(records: list[dict[str, Any]]) -> str:
    keys = (
        "event_index",
        "capture_scope",
        "semantic_role",
        "module_name",
        "module_type",
        "occurrence_index",
        "call_index",
        "generation_step_index",
        "io_kind",
        "tensor_path",
    )
    sequence = [{key: record[key] for key in keys} for record in records]
    return _sha256(_canonical_json(sequence))


def _module_input_tensor(args: Any, kwargs: Any) -> tuple[Any, str]:
    if args:
        try:
            return numerics_probe._first_tensor_leaf(args, "args")  # noqa: SLF001
        except RuntimeError:
            pass
    for key in ("hidden_states", "x", "inputs_embeds"):
        if key in kwargs:
            try:
                return numerics_probe._first_tensor_leaf(  # noqa: SLF001
                    kwargs[key], f"kwargs.{key}"
                )
            except RuntimeError:
                pass
    return numerics_probe._first_tensor_leaf(kwargs, "kwargs")  # noqa: SLF001


def _protocol(
    config: Mapping[str, Any],
    tokenizer: Any,
    forward_source_sha256: str,
) -> dict[str, Any]:
    upstream = isolation_probe._protocol(dict(config), tokenizer)  # noqa: SLF001
    return {
        "freshness_scope": (
            "four_fresh_actual_attached_model_loads_then_four_fresh_"
            "standalone_rmsnorm_modules_in_fixed_process"
        ),
        "actual_run_plan": [
            {
                "run_id": run_id,
                "path": path,
                "repeat": repeat,
                "order_index": index,
            }
            for index, (path, repeat, run_id) in enumerate(ACTUAL_RUN_PLAN)
        ],
        "control_run_plan": [
            {
                "run_id": run_id,
                "path": path,
                "repeat": repeat,
                "order_index": index,
            }
            for index, (path, repeat, run_id) in enumerate(CONTROL_RUN_PLAN)
        ],
        "run_order_design": "actual_ABBA_then_standalone_control_ABBA",
        "fresh_loads_per_path": {
            "actual_attached": 2,
            "standalone_control": 2,
        },
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "target_forward": upstream["target_forward"],
        "actual_treatment": upstream["treatment"],
        "paths": upstream["paths"],
        "generation": upstream["generation"],
        "sdp_kernel_flags": upstream["sdp_kernel_flags"],
        "actual_boundary_capture": {
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "generation_step_index": TARGET_STEP_INDEX,
            "capture_roles": list(ACTUAL_CAPTURE_ROLES),
            "capture_count_per_run": len(ACTUAL_CAPTURE_ROLES),
            "capture_method": (
                "target_scoped_forward_hooks_gpu_clone_then_cpu_summary"
            ),
            "actual_model_control_injection": False,
        },
        "standalone_control": {
            "control_id": CONTROL_ID,
            "intervention_count": 1,
            "module_name": BOUNDARY_MODULE,
            "module_type": BOUNDARY_MODULE_TYPE,
            "forward_source_sha256": forward_source_sha256,
            "variance_epsilon": RMSNORM_EPSILON,
            "hidden_size": HIDDEN_SIZE,
            "common_input_source": (
                f"{EMBEDDING_TENSOR_NAME}[{TARGET_INPUT_TOKEN_ID}]"
            ),
            "common_weight_source": RMSNORM_WEIGHT_TENSOR_NAME,
            "fresh_standalone_module_per_run": True,
            "attached_model_loaded_during_control": False,
            "autocast": False,
            "tf32": False,
            "cache_arguments": False,
            "actual_model_output_injection": False,
            "serialized_tensor_payload": False,
            "module_tensor_sidecar": False,
        },
        "comparison": {
            "dtype": "float32",
            "contiguous": True,
            "signed_zero_normalized": True,
            "finite_only": True,
            "exactness": "canonical_float32_exact_no_tolerance",
            "reduction": "fixed_flatten_order_stdlib_math_fsum_float64",
        },
    }


def _causal_scope(boundary_analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "isolated_variable": (
            "Qwen2RMSNorm input_weight_output dtype materialization under one "
            "same-checkpoint-values standalone control"
        ),
        "controlled": [
            "same_bfloat16_checkpoint_embedding_row_788_source_values",
            "same_bfloat16_checkpoint_layer0_input_rmsnorm_weight_source_values",
            "same_Qwen2RMSNorm_class_forward_source_and_epsilon",
            "same_actual_attached_factorized_lora_execution_form",
            "same_eval_001_target_forward_and_generation_protocol",
            "fresh_actual_and_standalone_control_ABBA_repeats",
            "disabled_autocast_and_tf32",
        ],
        "supports": (
            "whether one pre-registered same-values standalone Qwen2RMSNorm "
            "dtype replay exactly reproduces the observed actual layer-0 "
            "input_layernorm output boundary"
        ),
        "observed_current_forward_boundary_sufficiency": boundary_analysis[
            "current_forward_boundary_sufficiency_observed"
        ],
        "does_not_support": [
            "unique_pow_mean_rsqrt_cast_multiply_or_cuda_kernel_root_cause",
            "earliest_difference_across_generation_history",
            "absence_of_dtype_conditioned_kv_cache_history",
            "independent_causal_propagation_beyond_the_rmsnorm_output",
            "peft_bug_claim",
            "pristine_fp32_checkpoint_comparison",
            "full_eval_generalization",
            "artifact_promotion",
            "runtime_eligibility",
        ],
        "json_only_limitation": (
            "without module tensor payloads the offline validator checks exact "
            "digests, frozen links, repeat identity, and summary algebra but "
            "cannot independently recompute full captured tensors"
        ),
    }


def _constraints() -> dict[str, bool]:
    constraints = numerics_probe._constraints()  # noqa: SLF001
    constraints.update(
        {
            "module_tensor_sidecar": False,
            "module_tensor_payload": False,
            "second_boundary_intervention": False,
            "actual_model_control_injection": False,
            "control_cache_usage": False,
            "control_autocast": False,
            "control_tf32": False,
            "full_eval_run": False,
            "fp32_adapter_artifact_promotion": False,
        }
    )
    return constraints


def _locked_next_action(
    boundary_analysis: Mapping[str, Any],
    constraints: Mapping[str, bool],
) -> dict[str, Any]:
    matched = bool(
        boundary_analysis["current_forward_boundary_sufficiency_observed"]
    )
    next_gate_constraints = dict(constraints)
    next_gate_constraints["full_eval_run"] = True
    return {
        "gate_id": "FC-MVP-001-fp32-attached-remediation-eval-v1",
        "eligible_to_start": matched,
        "action": (
            "pre-register one resource-bounded full frozen evaluation of the "
            "FP32 attached Adapter with decision compilation fixed, compare it "
            "against frozen BF16 attached v2 metrics, and keep merge, artifact "
            "promotion, and Runtime integration prohibited until safety, "
            "regression, and resource gates pass"
        ),
        "acceptance": {
            "matched_boundary_control_required": True,
            "frozen_boundary_evidence_locked": True,
            "one_fp32_attached_candidate": True,
            "unchanged_twenty_case_eval": True,
            "unchanged_decision_compiler": True,
            "frozen_bf16_attached_reference": True,
            "resource_and_safety_comparison_required": True,
            "merge_prohibited": True,
            "artifact_promotion_prohibited": True,
            "runtime_integration_prohibited": True,
        },
        "constraints": next_gate_constraints,
    }


def _canonical_float32_tensor(tensor: Any) -> Any:
    return numerics_probe._canonical_float32_tensor(tensor)  # noqa: SLF001


def _tensor_bytes_sha256(tensor: Any) -> str:
    return numerics_probe._tensor_bytes_sha256(tensor)  # noqa: SLF001


def _float32_sha256(tensor: Any) -> str:
    return numerics_probe._float32_sha256(tensor)  # noqa: SLF001


def _module_type(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


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
