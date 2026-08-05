"""Self-contained FC-MVP-000 validation gate for Python 3.11-3.13."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import tomllib
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
BASELINE_PATH = ROOT / "baseline" / "fc-mvp-000.json"
TOOL_ROUTER_BASELINE_PATH = ROOT / "baseline" / "fc-mvp-001-schema-eval.json"
TOOL_ROUTER_DATA_BASELINE_PATH = ROOT / "baseline" / "fc-mvp-001-data-v1.json"
TOOL_ROUTER_SAFETY_REPAIR_BASELINE_PATH = (
    ROOT / "baseline" / "fc-mvp-001-safety-repair-data-v2.json"
)
TOOL_ROUTER_MODEL_BASELINE_PATH = (
    ROOT / "baseline" / "fc-mvp-001-base-model-v1.json"
)
TOOL_ROUTER_SFT_BASELINE_PATH = (
    ROOT / "baseline" / "fc-mvp-001-lora-sft-v1.json"
)
TOOL_ROUTER_SFT_V2_BASELINE_PATH = (
    ROOT / "baseline" / "fc-mvp-001-lora-sft-v2.json"
)
TOOL_ROUTER_FAILURE_CLASSIFICATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-failure-classification.json"
)
TOOL_ROUTER_DECISION_COMPILATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-decision-compilation-v1.json"
)
TOOL_ROUTER_COMPILED_PREDICTIONS_PATH = (
    ROOT / "baseline" / "tool-router-lora-sft-v2-compiled-predictions.json"
)
TOOL_ROUTER_COMPILED_REPORT_PATH = (
    ROOT / "baseline" / "tool-router-lora-sft-v2-compiled-report.json"
)
TOOL_ROUTER_MERGE_STABILITY_PATH = (
    ROOT / "baseline" / "fc-mvp-001-bf16-merge-stability-v1.json"
)
TOOL_ROUTER_MERGE_NUMERICS_PATH = (
    ROOT / "baseline" / "fc-mvp-001-bf16-merge-numerics-v1.json"
)
TOOL_ROUTER_MERGE_REMEDIATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-bf16-merge-remediation-v1.json"
)
TOOL_ROUTER_FP32_MERGE_DRIFT_PATH = (
    ROOT / "baseline" / "fc-mvp-001-fp32-merge-drift-analysis-v1.json"
)
TOOL_ROUTER_FP32_ATTACHED_MERGE_ISOLATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-fp32-attached-merge-isolation-v1.json"
)
TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_PATH = (
    ROOT / "baseline" / "fc-mvp-001-fp32-attached-merge-numerics-v1.json"
)
TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_TENSORS_PATH = (
    ROOT
    / "baseline"
    / "fc-mvp-001-fp32-attached-merge-numerics-v1-tensors.bin"
)
TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-isolation-v1.json"
)
TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-numerics-v1.json"
)
TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH = (
    ROOT / "baseline" / "fc-mvp-001-attached-dtype-boundary-control-v1.json"
)
TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_PREREGISTRATION_PATH = (
    ROOT / "configs" / "tool_router_fp32_attached_remediation_eval_v1.json"
)
TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_PREDICTIONS_PATH = (
    ROOT / "baseline" / "tool-router-fp32-attached-remediation-v1-predictions.json"
)
TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_EVAL_PATH = (
    ROOT / "baseline" / "fc-mvp-001-fp32-attached-remediation-eval-v1.json"
)
SUPPORTED_MINORS = {(3, 11), (3, 12), (3, 13)}
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "computer_use_agent",
        "computer_use_mcp",
        "httpx",
        "mcp",
        "requests",
        "socket",
        "urllib",
        "webbrowser",
    }
)


class GateError(RuntimeError):
    pass


def main() -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(SRC)
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT))

    version = (sys.version_info.major, sys.version_info.minor)
    if version not in SUPPORTED_MINORS:
        raise GateError(f"unsupported Python minor: {version[0]}.{version[1]}")

    baseline = _load_json(BASELINE_PATH)
    _validate_project_metadata(baseline)
    _validate_artifact_hashes(baseline)
    audited_files = _audit_import_boundary()
    (
        bridge_summary,
        record_count,
        router_summary,
        data_report,
        safety_repair_report,
        model_metrics,
        sft_metrics,
        sft_v2_metrics,
        failure_classification,
        decision_compilation,
    ) = _validate_fixed_outputs()
    merge_stability = _validate_merge_stability()
    merge_numerics = _validate_merge_numerics(merge_stability)
    merge_remediation = _validate_merge_remediation(
        merge_stability,
        merge_numerics,
    )
    fp32_merge_drift = _validate_fp32_merge_drift(merge_remediation)
    fp32_attached_merge_isolation = _validate_fp32_attached_merge_isolation(
        fp32_merge_drift
    )
    fp32_attached_merge_numerics = _validate_fp32_attached_merge_numerics(
        fp32_attached_merge_isolation
    )
    attached_dtype_isolation = _validate_attached_dtype_isolation(
        fp32_attached_merge_numerics
    )
    attached_dtype_numerics = _validate_attached_dtype_numerics(
        attached_dtype_isolation
    )
    attached_dtype_boundary_control = _validate_attached_dtype_boundary_control(
        attached_dtype_numerics
    )
    fp32_attached_remediation_eval = _validate_fp32_attached_remediation_eval(
        attached_dtype_boundary_control
    )
    tests_run = _run_tests()

    result = {
        "valid": True,
        "baseline_version": baseline["baseline_version"],
        "source_code_commit": baseline["source_code_commit"],
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "runtime_dependencies": 0,
        "artifact_hashes_verified": len(baseline["artifacts"]),
        "source_files_audited": audited_files,
        "tests_run": tests_run,
        "manifest_digest": bridge_summary.manifest_digest,
        "dataset_records": record_count,
        "tool_router_seed_records": router_summary["seed_records"],
        "tool_router_eval_records": router_summary["eval_records"],
        "tool_router_eval_digest": router_summary["eval_digest"],
        "tool_router_dangerous_false_approvals": router_summary["baseline"][
            "dangerous_false_approvals"
        ],
        "tool_router_train_records": data_report["train_records"],
        "tool_router_validation_records": data_report["validation_records"],
        "tool_router_task_families": data_report["task_families"],
        "tool_router_data_report_digest": data_report["report_digest"],
        "tool_router_safety_repair_train_records": safety_repair_report[
            "train_records"
        ],
        "tool_router_safety_repair_validation_records": safety_repair_report[
            "validation_records"
        ],
        "tool_router_safety_repair_task_families": safety_repair_report[
            "task_families"
        ],
        "tool_router_safety_repair_dangerous_action_candidates": (
            safety_repair_report["dangerous_action_candidates"]
        ),
        "tool_router_safety_repair_report_digest": safety_repair_report[
            "report_digest"
        ],
        "tool_router_base_model_json_validity": model_metrics["json_validity"],
        "tool_router_base_model_tool_accuracy": model_metrics["tool_accuracy"],
        "tool_router_base_model_dangerous_action_candidates": model_metrics[
            "dangerous_action_candidates"
        ],
        "tool_router_lora_sft_tool_accuracy": sft_metrics["tool_accuracy"],
        "tool_router_lora_sft_dangerous_action_candidates": sft_metrics[
            "dangerous_action_candidates"
        ],
        "tool_router_lora_sft_v2_tool_accuracy": sft_v2_metrics["tool_accuracy"],
        "tool_router_lora_sft_v2_dangerous_action_candidates": sft_v2_metrics[
            "dangerous_action_candidates"
        ],
        "tool_router_lora_sft_v2_decision_semantic_validity": sft_v2_metrics[
            "decision_semantic_validity"
        ],
        "tool_router_lora_sft_v2_failure_report_digest": failure_classification[
            "report_digest"
        ],
        "tool_router_failure_classification_next_gate": failure_classification[
            "locked_next_action"
        ]["gate_id"],
        "tool_router_compiled_decision_semantic_validity": decision_compilation[
            "metrics"
        ]["decision_semantic_validity"],
        "tool_router_compiled_false_refusals": decision_compilation["metrics"][
            "false_refusals"
        ],
        "tool_router_decision_compilation_next_gate": decision_compilation[
            "locked_next_action"
        ]["gate_id"],
        "tool_router_merge_classification": merge_stability["classification"],
        "tool_router_merge_first_divergent_token_index": merge_stability[
            "token_analysis"
        ]["first_divergent_token_index"],
        "tool_router_merge_stability_next_gate": merge_stability[
            "locked_next_action"
        ]["gate_id"],
        "tool_router_merge_numerics_classification": merge_numerics[
            "classification"
        ],
        "tool_router_merge_numerics_first_module": merge_numerics[
            "module_analysis"
        ]["first_divergent_module"],
        "tool_router_merge_numerics_next_gate": merge_numerics[
            "locked_next_action"
        ]["gate_id"],
        "tool_router_merge_remediation_classification": merge_remediation[
            "classification"
        ],
        "tool_router_merge_remediation_passed": merge_remediation[
            "remediation_gate"
        ]["passed"],
        "tool_router_merge_remediation_matches_bf16_merged": merge_remediation[
            "remediation_gate"
        ]["frozen_bf16_merged_token_identity"],
        "tool_router_merge_remediation_next_gate": merge_remediation[
            "locked_next_action"
        ]["gate_id"],
        "tool_router_fp32_merge_drift_classification": fp32_merge_drift[
            "classification"
        ],
        "tool_router_fp32_merge_drift_first_token_index": fp32_merge_drift[
            "token_analysis"
        ]["first_divergent_token_index"],
        "tool_router_fp32_merge_drift_analysis_passed": fp32_merge_drift[
            "analysis_gate"
        ]["passed"],
        "tool_router_fp32_merge_drift_remediation_passed": fp32_merge_drift[
            "remediation_gate"
        ]["passed"],
        "tool_router_fp32_attached_merge_isolation_classification": (
            fp32_attached_merge_isolation["classification"]
        ),
        "tool_router_fp32_attached_merge_token_identity": (
            fp32_attached_merge_isolation["same_dtype_token_analysis"][
                "cross_path_identical"
            ]
        ),
        "tool_router_fp32_attached_merge_comparison_step": (
            fp32_attached_merge_isolation["comparison_step"]["step_index"]
        ),
        "tool_router_fp32_attached_merge_isolation_passed": (
            fp32_attached_merge_isolation["isolation_gate"]["passed"]
        ),
        "tool_router_fp32_attached_merge_remediation_passed": (
            fp32_attached_merge_isolation["remediation_gate"]["passed"]
        ),
        "tool_router_fp32_attached_merge_numerics_classification": (
            fp32_attached_merge_numerics["classification"]
        ),
        "tool_router_fp32_attached_merge_numerics_first_module": (
            fp32_attached_merge_numerics["module_analysis"][
                "first_divergent_module"
            ]
        ),
        "tool_router_fp32_attached_merge_numerics_passed": (
            fp32_attached_merge_numerics["numerics_gate"]["passed"]
        ),
        "tool_router_fp32_attached_merge_numerics_records": (
            fp32_attached_merge_numerics["tensor_archive"]["record_count"]
        ),
        "tool_router_attached_dtype_isolation_classification": (
            attached_dtype_isolation["classification"]
        ),
        "tool_router_attached_dtype_isolation_comparison_step": (
            attached_dtype_isolation["comparison_step"]["step_index"]
        ),
        "tool_router_attached_dtype_isolation_bf16_token_id": (
            attached_dtype_isolation["cross_dtype_token_analysis"][
                "bf16_token_id"
            ]
        ),
        "tool_router_attached_dtype_isolation_fp32_token_id": (
            attached_dtype_isolation["cross_dtype_token_analysis"][
                "fp32_token_id"
            ]
        ),
        "tool_router_attached_dtype_isolation_passed": (
            attached_dtype_isolation["dtype_isolation_gate"]["passed"]
        ),
        "tool_router_attached_dtype_numerics_classification": (
            attached_dtype_numerics["classification"]
        ),
        "tool_router_attached_dtype_numerics_first_module": (
            attached_dtype_numerics["module_analysis"][
                "first_unequal_module"
            ]
        ),
        "tool_router_attached_dtype_numerics_passed": (
            attached_dtype_numerics["numerics_gate"]["passed"]
        ),
        "tool_router_attached_dtype_numerics_capture_records": sum(
            run["capture_record_count"] for run in attached_dtype_numerics["runs"]
        ),
        "tool_router_attached_dtype_boundary_control_classification": (
            attached_dtype_boundary_control["classification"]
        ),
        "tool_router_attached_dtype_boundary_control_passed": (
            attached_dtype_boundary_control["boundary_control_gate"]["passed"]
        ),
        "tool_router_attached_dtype_boundary_control_actual_runs": len(
            attached_dtype_boundary_control["actual_runs"]
        ),
        "tool_router_attached_dtype_boundary_control_control_runs": len(
            attached_dtype_boundary_control["control_runs"]
        ),
        "tool_router_attached_dtype_boundary_control_capture_records": sum(
            run["capture_record_count"]
            for run in [
                *attached_dtype_boundary_control["actual_runs"],
                *attached_dtype_boundary_control["control_runs"],
            ]
        ),
        "tool_router_attached_dtype_boundary_control_current_forward_sufficient": (
            attached_dtype_boundary_control["boundary_analysis"][
                "current_forward_boundary_sufficiency_observed"
            ]
        ),
        "tool_router_fp32_attached_remediation_eval_classification": (
            fp32_attached_remediation_eval["assessment"]["classification"]
        ),
        "tool_router_fp32_attached_remediation_eval_passed": (
            fp32_attached_remediation_eval["assessment"][
                "evaluation_gate_passed"
            ]
        ),
        "tool_router_fp32_attached_remediation_raw_semantic_validity": (
            fp32_attached_remediation_eval["raw_metrics"][
                "decision_semantic_validity"
            ]
        ),
        "tool_router_fp32_attached_remediation_argument_exact_match": (
            fp32_attached_remediation_eval["compiled_metrics"][
                "argument_exact_match"
            ]
        ),
        "tool_router_fp32_attached_remediation_argument_field_f1": (
            fp32_attached_remediation_eval["compiled_metrics"][
                "argument_field_f1"
            ]
        ),
        "tool_router_fp32_attached_remediation_elapsed_seconds": (
            fp32_attached_remediation_eval["resources"]["performance"][
                "elapsed_seconds"
            ]
        ),
        "tool_router_fp32_attached_remediation_peak_gpu_memory_bytes": (
            fp32_attached_remediation_eval["resources"]["performance"][
                "peak_gpu_memory_bytes"
            ]
        ),
        "tool_router_next_gate": fp32_attached_remediation_eval[
            "locked_next_action"
        ]["gate_id"],
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_project_metadata(baseline: dict[str, Any]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    expected_minors = [f"{major}.{minor}" for major, minor in sorted(SUPPORTED_MINORS)]
    if project["requires-python"] != baseline["python_requires"]:
        raise GateError("python_requires does not match baseline")
    if project["version"] != baseline["package_version"]:
        raise GateError("package version does not match baseline")
    if project["dependencies"] != baseline["runtime_dependencies"]:
        raise GateError("runtime dependencies do not match baseline")
    if baseline["required_python_minors"] != expected_minors:
        raise GateError("Python matrix does not match gate")
    lock_lines = [
        line.strip()
        for line in (ROOT / "requirements" / "runtime.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lock_lines:
        raise GateError("runtime.lock must remain empty for the stdlib baseline")


def _validate_artifact_hashes(baseline: dict[str, Any]) -> None:
    artifacts = baseline.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise GateError("baseline artifacts are missing")
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GateError(f"unsafe or missing baseline artifact: {relative}")
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise GateError(f"baseline artifact digest mismatch: {relative}")


def _audit_import_boundary() -> int:
    count = 0
    for path in sorted((SRC / "fullcycle_bridge").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += 1
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.partition(".")[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        raise GateError(f"forbidden import {root} in {path.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = node.module.partition(".")[0]
            if imported in FORBIDDEN_IMPORT_ROOTS:
                raise GateError(f"forbidden import {imported} in {path.name}")
    return count


def _validate_fixed_outputs() -> tuple[
    Any,
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    from fullcycle_bridge import validate_files
    from fullcycle_bridge.consumer import canonical_json_bytes
    from fullcycle_bridge.dataset import map_many
    from fullcycle_bridge.tool_router import (
        baseline_predict,
        evaluate,
        fixture_digest,
        load_fixture,
    )
    from fullcycle_bridge.tool_router_dataset import (
        audit_dataset,
        load_family_manifest,
    )
    from fullcycle_bridge.tool_router_model_eval import score_raw_outputs
    from fullcycle_bridge.tool_router_decision_compilation import (
        compile_frozen_v2_outputs,
    )
    from fullcycle_bridge.tool_router_failure_classification import (
        classify_v2_failures,
    )
    from fullcycle_bridge.tool_router_safety_repair import (
        audit_safety_repair_dataset,
        load_badcase_taxonomy,
    )
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    manifest = ROOT / "fixtures" / "bridge_v1" / "valid" / "runtime-manifest.json"
    minimal = ROOT / "fixtures" / "bridge_v1" / "valid" / "minimal-run-export.json"
    input_root = ROOT / "fixtures" / "reliability_dataset_v1" / "inputs"
    inputs = [
        input_root / "failure-denial-recovery-budget-sequence.json",
        input_root / "unknown-outcome.json",
    ]
    summary = validate_files(manifest, minimal)
    records = map_many(manifest, inputs)
    actual = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    expected = (
        ROOT / "fixtures" / "reliability_dataset_v1" / "expected-records.jsonl"
    ).read_bytes()
    if actual != expected:
        raise GateError("dataset JSONL differs from the frozen fixture")
    router_baseline = _load_json(TOOL_ROUTER_BASELINE_PATH)
    _validate_named_hashes(router_baseline["artifact_hashes"])
    seed = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "seed.json")
    evaluation = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "eval.json")
    router_summary: dict[str, Any] = {
        "seed_records": len(seed),
        "eval_records": len(evaluation),
        "seed_digest": fixture_digest(seed),
        "eval_digest": fixture_digest(evaluation),
        "baseline": evaluate(
            evaluation, [baseline_predict(record) for record in evaluation]
        ),
    }
    for key in ("seed_records", "eval_records", "seed_digest", "eval_digest"):
        if router_summary[key] != router_baseline[key]:
            raise GateError(f"Tool Router baseline mismatch: {key}")
    if router_summary["baseline"] != router_baseline["deterministic_rule_baseline"]:
        raise GateError("Tool Router metrics differ from the frozen baseline")
    data_baseline = _load_json(TOOL_ROUTER_DATA_BASELINE_PATH)
    _validate_named_hashes(data_baseline["artifact_hashes"])
    train = load_fixture(ROOT / "fixtures" / "tool_router_v1" / "train.json")
    validation = load_fixture(
        ROOT / "fixtures" / "tool_router_v1" / "validation.json"
    )
    family_manifest = load_family_manifest(
        ROOT / "fixtures" / "tool_router_v1" / "family-manifest.json"
    )
    data_report = audit_dataset(
        train,
        validation,
        evaluation,
        family_manifest,
        router_summary["eval_digest"],
    )
    if data_report != data_baseline["expected_report"]:
        raise GateError("Tool Router data audit differs from the frozen baseline")
    safety_repair_baseline = _load_json(TOOL_ROUTER_SAFETY_REPAIR_BASELINE_PATH)
    _validate_named_hashes(safety_repair_baseline["base_artifact_hashes"])
    _validate_named_hashes(safety_repair_baseline["artifact_hashes"])
    taxonomy_source = safety_repair_baseline["source_badcase_taxonomy"]
    _validate_named_hashes({taxonomy_source["path"]: taxonomy_source["sha256"]})
    safety_repair_train = load_fixture(
        ROOT / "fixtures" / "tool_router_v2" / "train.json"
    )
    safety_repair_validation = load_fixture(
        ROOT / "fixtures" / "tool_router_v2" / "validation.json"
    )
    safety_repair_manifest = load_family_manifest(
        ROOT / "fixtures" / "tool_router_v2" / "family-manifest.json"
    )
    safety_repair_taxonomy = load_badcase_taxonomy(ROOT / taxonomy_source["path"])
    if (
        safety_repair_taxonomy["source_prediction_sha256"]
        != taxonomy_source["source_prediction_sha256"]
        or safety_repair_taxonomy["source_report_sha256"]
        != taxonomy_source["source_report_sha256"]
    ):
        raise GateError("Tool Router safety-repair source provenance mismatch")
    safety_repair_report = audit_safety_repair_dataset(
        train,
        validation,
        safety_repair_train,
        safety_repair_validation,
        evaluation,
        family_manifest,
        safety_repair_manifest,
        safety_repair_taxonomy,
        router_summary["eval_digest"],
    )
    if safety_repair_report != safety_repair_baseline["expected_report"]:
        raise GateError("Tool Router safety-repair audit differs from frozen baseline")
    model_baseline = _load_json(TOOL_ROUTER_MODEL_BASELINE_PATH)
    _validate_named_hashes(model_baseline["artifact_hashes"])
    prediction_artifact = _load_json(
        ROOT
        / "baseline"
        / "tool-router-qwen2.5-1.5b-instruct-predictions.json"
    )
    frozen_report = _load_json(
        ROOT / "baseline" / "tool-router-qwen2.5-1.5b-instruct-report.json"
    )
    raw_outputs = [
        {
            "example_id": item["example_id"],
            "raw_output": item["raw_output"],
        }
        for item in prediction_artifact["outputs"]
    ]
    model_metrics, parsed_outputs = score_raw_outputs(evaluation, raw_outputs)
    if model_metrics != model_baseline["metrics"]:
        raise GateError("Tool Router model metrics differ from the frozen baseline")
    if model_metrics != frozen_report["metrics"]:
        raise GateError("Tool Router frozen model report metrics mismatch")
    if parsed_outputs != frozen_report["parsed_outputs"]:
        raise GateError("Tool Router frozen parsed outputs mismatch")
    sft_baseline = _load_json(TOOL_ROUTER_SFT_BASELINE_PATH)
    _validate_named_hashes(sft_baseline["artifact_hashes"])
    sft_config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v1.json")
    sft_evidence = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v1-training.json"
    )
    sft_predictions = _load_json(
        ROOT
        / "baseline"
        / "tool-router-qwen2.5-1.5b-lora-sft-v1-predictions.json"
    )
    sft_report = _load_json(
        ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v1-report.json"
    )
    config_digest = canonical_config_sha256(sft_config)
    if config_digest != sft_baseline["canonical_config_sha256"]:
        raise GateError("Tool Router SFT config digest mismatch")
    if sft_evidence["config_sha256"] != config_digest:
        raise GateError("Tool Router SFT evidence config mismatch")
    if sft_predictions["config_sha256"] != config_digest:
        raise GateError("Tool Router SFT prediction config mismatch")
    adapter_dir = (
        ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v1"
    )
    if directory_artifact_manifest(adapter_dir) != sft_evidence["final_adapter"]["files"]:
        raise GateError("Tool Router SFT adapter manifest mismatch")
    adapter_weight = adapter_dir / "adapter_model.safetensors"
    if file_sha256(adapter_weight) != sft_baseline["adapter"]["adapter_weight_sha256"]:
        raise GateError("Tool Router SFT adapter weight digest mismatch")
    sft_raw_outputs = [
        {
            "example_id": item["example_id"],
            "raw_output": item["raw_output"],
        }
        for item in sft_predictions["outputs"]
    ]
    sft_metrics, sft_parsed = score_raw_outputs(evaluation, sft_raw_outputs)
    if sft_metrics != sft_baseline["metrics"]:
        raise GateError("Tool Router SFT metrics differ from the frozen baseline")
    if sft_metrics != sft_report["metrics"]:
        raise GateError("Tool Router SFT report metrics mismatch")
    if sft_parsed != sft_report["parsed_outputs"]:
        raise GateError("Tool Router SFT parsed outputs mismatch")
    if sft_report["runtime_eligible"] is not False:
        raise GateError("Tool Router SFT must remain Runtime ineligible")
    sft_v2_baseline = _load_json(TOOL_ROUTER_SFT_V2_BASELINE_PATH)
    _validate_named_hashes(sft_v2_baseline["artifact_hashes"])
    sft_v2_config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    sft_v2_evidence = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    sft_v2_predictions = _load_json(
        ROOT
        / "baseline"
        / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
    )
    sft_v2_report = _load_json(
        ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
    )
    sft_v2_load_merge = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-load-merge.json"
    )
    sft_v2_config_digest = canonical_config_sha256(sft_v2_config)
    if sft_v2_config_digest != sft_v2_baseline["canonical_config_sha256"]:
        raise GateError("Tool Router SFT v2 config digest mismatch")
    if sft_v2_evidence["config_sha256"] != sft_v2_config_digest:
        raise GateError("Tool Router SFT v2 evidence config mismatch")
    sft_v2_adapter_dir = (
        ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    )
    if (
        directory_artifact_manifest(sft_v2_adapter_dir)
        != sft_v2_evidence["final_adapter"]["files"]
    ):
        raise GateError("Tool Router SFT v2 adapter manifest mismatch")
    sft_v2_raw_outputs = [
        {
            "example_id": item["example_id"],
            "raw_output": item["raw_output"],
        }
        for item in sft_v2_predictions["outputs"]
    ]
    sft_v2_metrics, sft_v2_parsed = score_raw_outputs(
        evaluation, sft_v2_raw_outputs
    )
    if sft_v2_metrics != sft_v2_baseline["metrics"]:
        raise GateError("Tool Router SFT v2 metrics differ from frozen baseline")
    if sft_v2_metrics != sft_v2_report["metrics"]:
        raise GateError("Tool Router SFT v2 report metrics mismatch")
    if sft_v2_parsed != sft_v2_report["parsed_outputs"]:
        raise GateError("Tool Router SFT v2 parsed outputs mismatch")
    if not sft_v2_report["safety_gate_passed"]:
        raise GateError("Tool Router SFT v2 safety gate must remain passed")
    if sft_v2_report["runtime_eligible"] is not False:
        raise GateError("Tool Router SFT v2 must remain Runtime ineligible")
    if (
        sft_v2_load_merge["outputs_identical"] is not False
        or sft_v2_load_merge["safe_merge"] is not True
        or sft_v2_load_merge["remaining_adapter_parameter_tensors"] != 0
    ):
        raise GateError("Tool Router SFT v2 load/merge evidence mismatch")
    failure_baseline = _load_json(TOOL_ROUTER_FAILURE_CLASSIFICATION_PATH)
    failure_source_paths = {
        "predictions": (
            ROOT
            / "baseline"
            / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
        ),
        "report": (
            ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
        ),
        "training": ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json",
        "load_merge": ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-load-merge.json",
    }
    failure_source_hashes = {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in failure_source_paths.items()
    }
    failure_classification = classify_v2_failures(
        sft_v2_predictions,
        sft_v2_report,
        sft_v2_evidence,
        sft_v2_load_merge,
        failure_source_hashes,
    )
    if failure_classification != failure_baseline:
        raise GateError("Tool Router SFT v2 failure classification drift")
    decision_compilation = _load_json(TOOL_ROUTER_DECISION_COMPILATION_PATH)
    compiled_predictions = _load_json(TOOL_ROUTER_COMPILED_PREDICTIONS_PATH)
    compiled_report = _load_json(TOOL_ROUTER_COMPILED_REPORT_PATH)
    classification_path = TOOL_ROUTER_FAILURE_CLASSIFICATION_PATH
    compilation_source_hashes = {
        "predictions": failure_source_hashes["predictions"],
        "report": failure_source_hashes["report"],
        "classification": (
            "sha256:" + hashlib.sha256(classification_path.read_bytes()).hexdigest()
        ),
    }
    if decision_compilation["source_hashes"] != compilation_source_hashes:
        raise GateError("Tool Router decision compilation source drift")
    _validate_named_hashes(decision_compilation["artifact_hashes"])
    reproduced_compilation = compile_frozen_v2_outputs(
        sft_v2_predictions,
        sft_v2_report,
        failure_classification,
        compilation_source_hashes,
    )
    if reproduced_compilation != compiled_predictions:
        raise GateError("Tool Router compiled predictions drift")
    compiled_raw_outputs = [
        {"example_id": item["example_id"], "raw_output": item["raw_output"]}
        for item in compiled_predictions["outputs"]
    ]
    compiled_metrics, compiled_parsed = score_raw_outputs(
        evaluation, compiled_raw_outputs
    )
    if (
        compiled_metrics != compiled_report["metrics"]
        or compiled_metrics != decision_compilation["metrics"]
        or compiled_parsed != compiled_report["parsed_outputs"]
        or compiled_report["acceptance"] != decision_compilation["acceptance"]
    ):
        raise GateError("Tool Router decision compilation report drift")
    if (
        decision_compilation["runtime_eligible"] is not False
        or compiled_report["runtime_eligible"] is not False
    ):
        raise GateError("Tool Router compiled output must remain Runtime ineligible")
    return (
        summary,
        len(records),
        router_summary,
        data_report,
        safety_repair_report,
        model_metrics,
        sft_metrics,
        sft_v2_metrics,
        failure_classification,
        decision_compilation,
    )


def _validate_named_hashes(artifacts: object) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise GateError("named artifact hashes are missing")
    for relative, expected in artifacts.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise GateError(f"unsafe or missing artifact: {relative}")
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise GateError(f"artifact digest mismatch: {relative}")


def _validate_merge_stability() -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_MERGE_STABILITY_PATH)
    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    training = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    runs = gate.get("runs")
    if (
        gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("adapter_files") != training["final_adapter"]["files"]
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
    ):
        raise GateError("Tool Router merge-stability source drift")
    if not isinstance(runs, list) or len(runs) != 4:
        raise GateError("Tool Router merge-stability runs are invalid")
    token_digests = [run.get("token_ids_sha256") for run in runs]
    if not (
        token_digests[0] == token_digests[1]
        and token_digests[2] == token_digests[3]
        and token_digests[0] != token_digests[2]
    ):
        raise GateError("Tool Router merge-stability repeat evidence drift")
    acceptance = gate.get("acceptance")
    token_analysis = gate.get("token_analysis")
    locked_next_action = gate.get("locked_next_action")
    if (
        not isinstance(acceptance, dict)
        or not acceptance
        or not all(value is True for value in acceptance.values())
        or not isinstance(token_analysis, dict)
        or token_analysis.get("first_divergent_token_index") != 45
        or gate.get("classification")
        != "deterministic_bf16_merge_logit_boundary_flip"
        or gate.get("merged_artifact_allowed") is not False
        or gate.get("merged_artifact_saved") is not False
        or gate.get("runtime_eligible") is not False
        or not isinstance(locked_next_action, dict)
        or locked_next_action.get("gate_id")
        != "FC-MVP-001-bf16-merge-numerics-v1"
    ):
        raise GateError("Tool Router merge-stability acceptance drift")
    return gate


def _validate_merge_numerics(stability: dict[str, Any]) -> dict[str, Any]:
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_MERGE_NUMERICS_PATH)
    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    if (
        gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("stability_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_STABILITY_PATH)
        or gate.get("eval_digest") != stability.get("eval_digest")
    ):
        raise GateError("Tool Router merge-numerics source drift")
    acceptance = gate.get("acceptance")
    analysis = gate.get("module_analysis")
    rounding = gate.get("merge_rounding")
    next_action = gate.get("locked_next_action")
    if (
        not isinstance(acceptance, dict)
        or not acceptance
        or not all(value is True for value in acceptance.values())
        or not isinstance(analysis, dict)
        or analysis.get("first_divergent_module")
        != "model.layers.0.self_attn.q_proj"
        or analysis.get("first_divergent_module_index") != 2
        or analysis.get("preceding_modules_identical") is not True
        or not isinstance(rounding, dict)
        or rounding.get("target_modules") != 112
        or rounding.get("actual_merged_mismatched_weights") != 0
        or not isinstance(
            rounding.get("ideal_nonzero_updates_rounded_to_base"), int
        )
        or rounding["ideal_nonzero_updates_rounded_to_base"] <= 0
        or gate.get("classification") != "bf16_safe_merge_weight_rounding"
        or gate.get("merged_artifact_allowed") is not False
        or gate.get("merged_artifact_saved") is not False
        or gate.get("runtime_eligible") is not False
        or not isinstance(next_action, dict)
        or next_action.get("gate_id")
        != "FC-MVP-001-bf16-merge-remediation-v1"
    ):
        raise GateError("Tool Router merge-numerics acceptance drift")
    return gate


def _validate_merge_remediation(
    stability: dict[str, Any],
    numerics: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_MERGE_REMEDIATION_PATH)
    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    expected_top_level = {
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
    }
    if (
        set(gate) != expected_top_level
        or gate.get("merge_remediation_version") != 1
        or gate.get("experiment_id")
        != "fc-mvp-001-bf16-merge-remediation-v1"
        or gate.get("source_experiment_id") != config.get("experiment_id")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("stability_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_STABILITY_PATH)
        or gate.get("numerics_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_NUMERICS_PATH)
        or gate.get("training_lock_sha256")
        != file_sha256(ROOT / "requirements" / "training.lock")
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("eval_digest") != numerics.get("eval_digest")
        or gate.get("input_token_count") != stability.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != stability.get("input_token_ids_sha256")
        or gate.get("example_id") != "eval-001"
    ):
        raise GateError("Tool Router merge-remediation source drift")
    if gate.get("storage_audit") != {
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
    }:
        raise GateError("Tool Router merge-remediation storage dtype drift")

    independent = [
        run for run in stability["runs"] if run.get("path") == "independent"
    ]
    merged = [run for run in stability["runs"] if run.get("path") == "merged"]
    reference = gate.get("reference")
    control = gate.get("frozen_bf16_merged_control")
    if (
        len(independent) != 2
        or len(merged) != 2
        or not isinstance(reference, dict)
        or reference.get("path") != "independent_bf16_adapter"
        or reference.get("token_count") != independent[0].get("token_count")
        or reference.get("token_ids_sha256")
        != independent[0].get("token_ids_sha256")
        or reference.get("output_sha256") != independent[0].get("output_sha256")
        or not isinstance(control, dict)
        or control.get("path") != "safe_merged_bf16"
        or control.get("token_count") != merged[0].get("token_count")
        or control.get("token_ids_sha256") != merged[0].get("token_ids_sha256")
        or control.get("output_sha256") != merged[0].get("output_sha256")
    ):
        raise GateError("Tool Router merge-remediation frozen references drift")

    protocol = gate.get("candidate_protocol")
    expected_protocol = {
        "checkpoint_storage_dtype": "bfloat16",
        "base_load_dtype": "float32",
        "adapter_storage_dtype": "float32",
        "adapter_load_dtype": "float32",
        "merge_dtype": "float32",
        "inference_dtype": "float32",
        "safe_merge": True,
        "adapter_names": ["default"],
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
        "tf32": False,
        "autocast": False,
        "device": "cuda:0",
        "fresh_loads": 2,
        "max_residual_cuda_bytes": 16777216,
    }
    if protocol != expected_protocol:
        raise GateError("Tool Router merge-remediation protocol drift")

    runs = gate.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or [run.get("path") for run in runs]
        != ["fp32_safe_merged", "fp32_safe_merged"]
        or [run.get("repeat") for run in runs] != [1, 2]
        or runs[0].get("token_count") != runs[1].get("token_count")
        or runs[0].get("token_count") != reference.get("token_count")
        or runs[0].get("token_count") != control.get("token_count")
        or runs[0].get("token_ids_sha256")
        != runs[1].get("token_ids_sha256")
        or runs[0].get("output_sha256") != runs[1].get("output_sha256")
        or runs[0].get("token_ids_sha256") != control.get("token_ids_sha256")
        or runs[0].get("output_sha256") != control.get("output_sha256")
        or runs[0].get("token_ids_sha256") == reference.get("token_ids_sha256")
    ):
        raise GateError("Tool Router merge-remediation run evidence drift")
    for run in runs:
        precision = run.get("precision_audit")
        if not isinstance(precision, dict):
            raise GateError("Tool Router merge-remediation precision evidence missing")
        pre = precision.get("pre_merge")
        post = precision.get("post_merge")
        generation = precision.get("generation")
        if (
            not isinstance(pre, dict)
            or not isinstance(post, dict)
            or not isinstance(generation, dict)
            or pre.get("lora_target_modules") != 112
            or pre.get("lora_parameter_tensors") != 224
            or pre.get("adapter_parameters_finite") is not True
            or pre.get("active_adapters") != ["default"]
            or pre.get("is_peft_model") is not True
            or pre.get("input_output_embeddings_tied") is not True
            or pre.get("attn_implementation") != "sdpa"
            or pre.get("attention_class") != "Qwen2Attention"
            or pre.get("output_attentions") is not False
            or pre.get("hf_device_map") is not None
            or not _fp32_cuda_inventory(
                pre.get("base_parameters"),
                expected_elements=1543714304,
            )
            or not _fp32_cuda_inventory(
                pre.get("adapter_parameters"),
                expected_elements=4358144,
            )
            or not _fp32_cuda_inventory(
                pre.get("floating_buffers"),
                expected_elements=64,
            )
            or post.get("lora_target_modules") != 0
            or post.get("lora_parameter_tensors") != 0
            or post.get("is_peft_model") is not False
            or post.get("input_output_embeddings_tied") is not True
            or post.get("attn_implementation") != "sdpa"
            or post.get("attention_class") != "Qwen2Attention"
            or post.get("output_attentions") is not False
            or post.get("hf_device_map") is not None
            or not _fp32_cuda_inventory(
                post.get("parameters"),
                expected_elements=1543714304,
            )
            or not _fp32_cuda_inventory(
                post.get("floating_buffers"),
                expected_elements=64,
            )
            or generation
            != {
                "score_dtypes": ["float32"],
                "all_scores_float32": True,
                "autocast_enabled": False,
                "training": False,
            }
            or not isinstance(run.get("memory_allocated_before_load_bytes"), int)
            or run["memory_allocated_before_load_bytes"]
            > protocol["max_residual_cuda_bytes"]
            or not isinstance(run.get("memory_allocated_after_release_bytes"), int)
            or run["memory_allocated_after_release_bytes"]
            > protocol["max_residual_cuda_bytes"]
        ):
            raise GateError("Tool Router merge-remediation FP32 audit drift")

    acceptance = gate.get("acceptance")
    expected_acceptance = {
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
    }
    remediation = gate.get("remediation_gate")
    next_action = gate.get("locked_next_action")
    constraints = {
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_promotion": False,
    }
    if (
        not isinstance(acceptance, dict)
        or set(acceptance) != expected_acceptance
        or not all(value is True for value in acceptance.values())
        or gate.get("classification") != "deterministic_fp32_merge_output_drift"
        or not isinstance(remediation, dict)
        or set(remediation)
        != {
            "candidate_repeats_identical",
            "independent_bf16_reference_token_identity",
            "independent_bf16_reference_output_identity",
            "frozen_bf16_merged_token_identity",
            "frozen_bf16_merged_output_identity",
            "passed",
        }
        or remediation.get("candidate_repeats_identical") is not True
        or remediation.get("independent_bf16_reference_token_identity") is not False
        or remediation.get("independent_bf16_reference_output_identity") is not False
        or remediation.get("frozen_bf16_merged_token_identity") is not True
        or remediation.get("frozen_bf16_merged_output_identity") is not True
        or remediation.get("passed") is not False
        or gate.get("merged_artifact_saved") is not False
        or gate.get("merged_artifact_allowed") is not False
        or gate.get("constraints") != constraints
        or gate.get("runtime_eligible") is not False
        or gate.get("runtime_eligibility_reason")
        != "deterministic_fp32_merge_output_drift"
        or gate.get("offline") is not True
        or not isinstance(next_action, dict)
        or next_action.get("gate_id")
        != "FC-MVP-001-fp32-merge-drift-analysis-v1"
    ):
        raise GateError("Tool Router merge-remediation acceptance drift")
    return gate


def _validate_fp32_merge_drift(remediation: dict[str, Any]) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_fp32_merge_drift import analyze_path_tokens
    from fullcycle_bridge.tool_router_merge_remediation import token_ids_sha256
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_FP32_MERGE_DRIFT_PATH)
    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    training = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    expected_top_level = {
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
    }
    if (
        set(gate) != expected_top_level
        or file_sha256(TOOL_ROUTER_FP32_MERGE_DRIFT_PATH)
        != "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3"
        or gate.get("fp32_merge_drift_analysis_version") != 1
        or gate.get("experiment_id")
        != "fc-mvp-001-fp32-merge-drift-analysis-v1"
        or gate.get("source_experiment_id") != config.get("experiment_id")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("adapter_files") != training["final_adapter"]["files"]
        or gate.get("stability_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_STABILITY_PATH)
        or gate.get("numerics_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_NUMERICS_PATH)
        or gate.get("remediation_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_REMEDIATION_PATH)
        or gate.get("training_lock_sha256")
        != file_sha256(ROOT / "requirements" / "training.lock")
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("example_id") != "eval-001"
        or gate.get("input_token_count") != 339
        or gate.get("input_token_count") != remediation.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != remediation.get("input_token_ids_sha256")
    ):
        raise GateError("Tool Router FP32 merge-drift source drift")

    expected_storage = {
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
    }
    expected_protocol = {
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
    }
    if gate.get("storage_audit") != expected_storage:
        raise GateError("Tool Router FP32 merge-drift storage audit drift")
    if gate.get("protocol") != expected_protocol:
        raise GateError("Tool Router FP32 merge-drift protocol drift")

    references = gate.get("frozen_references")
    remediation_runs = remediation.get("runs")
    if (
        not isinstance(references, dict)
        or set(references) != {"independent_bf16_adapter", "fp32_safe_merged"}
        or references.get("independent_bf16_adapter")
        != remediation.get("reference")
        or not isinstance(remediation_runs, list)
        or len(remediation_runs) != 2
        or references.get("fp32_safe_merged")
        != {
            "path": "fp32_safe_merged",
            "source_experiment_id": remediation.get("experiment_id"),
            "fresh_runs": 2,
            "token_count": remediation_runs[0].get("token_count"),
            "token_ids_sha256": remediation_runs[0].get("token_ids_sha256"),
            "output_sha256": remediation_runs[0].get("output_sha256"),
        }
    ):
        raise GateError("Tool Router FP32 merge-drift frozen references drift")

    runs = gate.get("runs")
    expected_run_keys = {
        "path",
        "fresh_load",
        "generated_token_ids",
        "token_count",
        "token_ids_sha256",
        "output_sha256",
        "precision_audit",
        "generation_trace",
        "path_protocol_passed",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
    if (
        not isinstance(runs, list)
        or len(runs) != 2
        or any(not isinstance(run, dict) for run in runs)
        or any(set(run) != expected_run_keys for run in runs)
        or [run.get("path") for run in runs]
        != ["independent_bf16_adapter", "fp32_safe_merged"]
        or [run.get("fresh_load") for run in runs] != [1, 1]
    ):
        raise GateError("Tool Router FP32 merge-drift runs are invalid")
    for run in runs:
        path_name = run["path"]
        token_ids = run.get("generated_token_ids")
        reference = references[path_name]
        trace = run.get("generation_trace")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 48
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool)
                for token_id in token_ids
            )
            or run.get("token_count") != len(token_ids)
            or run.get("token_ids_sha256") != token_ids_sha256(token_ids)
            or run.get("token_ids_sha256") != reference.get("token_ids_sha256")
            or run.get("output_sha256") != reference.get("output_sha256")
            or run.get("path_protocol_passed") is not True
            or not isinstance(run.get("peak_gpu_memory_bytes"), int)
            or not isinstance(run.get("memory_allocated_before_load_bytes"), int)
            or not isinstance(run.get("memory_allocated_after_release_bytes"), int)
            or run["memory_allocated_before_load_bytes"]
            > expected_protocol["max_residual_cuda_bytes"]
            or run["memory_allocated_after_release_bytes"]
            > expected_protocol["max_residual_cuda_bytes"]
            or not isinstance(trace, dict)
            or set(trace)
            != {
                "step_count",
                "vocabulary_size",
                "cache_returned",
                "scores",
                "raw_logits",
            }
            or trace.get("step_count") != len(token_ids)
            or trace.get("vocabulary_size") != 151936
            or trace.get("cache_returned") is not True
        ):
            raise GateError("Tool Router FP32 merge-drift run evidence drift")
        for trace_key, evidence_key in (
            ("scores", "selection_score_evidence"),
            ("raw_logits", "raw_logit_evidence"),
        ):
            summary = trace.get(trace_key)
            evidence = gate.get(evidence_key)
            expected_summary_keys = {
                "native_dtypes",
                "shape_per_step",
                "comparison_dtype",
                "all_finite",
                "trace_sha256",
                "divergent_step_index",
                "divergent_step_comparison_vector_sha256",
            }
            if (
                not isinstance(summary, dict)
                or set(summary) != expected_summary_keys
                or summary.get("native_dtypes") != ["float32"]
                or summary.get("shape_per_step") != [1, 151936]
                or summary.get("comparison_dtype") != "float32"
                or summary.get("all_finite") is not True
                or not _is_canonical_sha256(summary.get("trace_sha256"))
                or summary.get("divergent_step_index") != 45
                or not isinstance(evidence, dict)
                or summary.get("divergent_step_comparison_vector_sha256")
                != evidence.get("paths", {}).get(path_name, {}).get(
                    "comparison_vector_sha256"
                )
            ):
                raise GateError("Tool Router FP32 merge-drift trace linkage drift")

    _validate_fp32_drift_precision(runs)
    reproduction = gate.get("reproduction")
    if reproduction != {
        "independent_bf16_token_identity": True,
        "independent_bf16_output_identity": True,
        "fp32_candidate_token_identity": True,
        "fp32_candidate_output_identity": True,
    }:
        raise GateError("Tool Router FP32 merge-drift reproduction drift")

    token_analysis = gate.get("token_analysis")
    recomputed = analyze_path_tokens(
        runs[0]["generated_token_ids"],
        runs[1]["generated_token_ids"],
    )
    if (
        not isinstance(token_analysis, dict)
        or set(token_analysis) != set(recomputed) | {
            "independent_token_text",
            "candidate_token_text",
        }
        or any(token_analysis.get(key) != value for key, value in recomputed.items())
        or token_analysis.get("first_divergent_token_index") != 45
        or token_analysis.get("independent_token_id") != 1866
        or token_analysis.get("candidate_token_id") != 3849
        or token_analysis.get("independent_token_text") != "true"
        or token_analysis.get("candidate_token_text") != "false"
    ):
        raise GateError("Tool Router FP32 merge-drift token analysis drift")

    _validate_fp32_step_evidence(
        gate,
        runs,
        evidence_key="selection_score_evidence",
        source="generated.scores",
        semantics="processed_prediction_scores_after_logits_processors",
        value_key="score",
        expected_values={
            "independent_bf16_adapter": [
                34.54545211791992,
                34.09090805053711,
                25.75,
                21.93181800842285,
                21.5,
            ],
            "fp32_safe_merged": [
                35.61100387573242,
                33.169429779052734,
                24.49706268310547,
                23.684528350830078,
                21.257160186767578,
            ],
        },
        expected_top_ids={
            "independent_bf16_adapter": [1866, 3849, 830, 895, 2641],
            "fp32_safe_merged": [3849, 1866, 830, 895, 2641],
        },
        expected_delta={
            "vocabulary_elements": 151936,
            "nonzero_elements": 151936,
            "max_abs_delta": 1.9437971115112305,
            "mean_abs_delta": 0.2275839000940323,
            "root_mean_square_delta": 0.29325070977211,
        },
    )
    _validate_fp32_step_evidence(
        gate,
        runs,
        evidence_key="raw_logit_evidence",
        source="generated.logits",
        semantics="unprocessed_lm_head_prediction_scores",
        value_key="raw_logit",
        expected_values={
            "independent_bf16_adapter": [38.0, 37.5, 25.75, 24.125, 21.5],
            "fp32_safe_merged": [
                39.17210388183594,
                36.48637390136719,
                26.052982330322266,
                24.49706268310547,
                21.257160186767578,
            ],
        },
        expected_top_ids={
            "independent_bf16_adapter": [1866, 3849, 830, 895, 2641],
            "fp32_safe_merged": [3849, 1866, 895, 830, 2641],
        },
        expected_delta={
            "vocabulary_elements": 151936,
            "nonzero_elements": 151936,
            "max_abs_delta": 1.9437971115112305,
            "mean_abs_delta": 0.22757971286773682,
            "root_mean_square_delta": 0.2932598292827606,
        },
    )

    expected_classification = (
        "deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip"
    )
    analysis_gate = gate.get("analysis_gate")
    remediation_gate = gate.get("remediation_gate")
    acceptance = gate.get("acceptance")
    expected_acceptance = {
        "upstream_evidence_locked",
        "frozen_input_reproduced",
        "independent_bf16_reference_reproduced",
        "fp32_candidate_reproduced",
        "first_divergent_token_located",
        "exact_processed_scores_captured",
        "exact_raw_logits_captured",
        "generation_score_alignment_verified",
        "path_protocols_executed",
        "source_storage_dtypes_locked",
        "fresh_load_memory_isolated",
        "source_adapter_unchanged",
        "source_model_unchanged",
        "eval_digest_unchanged",
        "prompt_digest_unchanged",
    }
    constraints = {
        "failed_candidate_change": False,
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
    next_action = gate.get("locked_next_action")
    if (
        gate.get("classification") != expected_classification
        or analysis_gate
        != {
            "frozen_paths_reproduced": True,
            "first_divergent_token_located": True,
            "exact_cached_generate_step_captured": True,
            "processed_score_argmax_matches_generated_token": True,
            "raw_logits_captured": True,
            "passed": True,
        }
        or remediation_gate
        != {
            "candidate_failure_reproduced": True,
            "independent_bf16_reference_identity": False,
            "passed": False,
        }
        or not isinstance(acceptance, dict)
        or set(acceptance) != expected_acceptance
        or not all(value is True for value in acceptance.values())
        or gate.get("peak_gpu_memory_bytes")
        != max(run["peak_gpu_memory_bytes"] for run in runs)
        or not isinstance(gate.get("elapsed_seconds"), (int, float))
        or isinstance(gate.get("elapsed_seconds"), bool)
        or gate["elapsed_seconds"] <= 0
        or gate.get("merged_artifact_saved") is not False
        or gate.get("merged_artifact_allowed") is not False
        or gate.get("constraints") != constraints
        or not isinstance(next_action, dict)
        or set(next_action) != {"gate_id", "action", "acceptance", "constraints"}
        or next_action.get("gate_id")
        != "FC-MVP-001-fp32-attached-merge-isolation-v1"
        or next_action.get("acceptance")
        != {
            "attached_fp32_repeat_stable": True,
            "fp32_candidate_reproduced": True,
            "same_dtype_exact_step_compared": True,
            "same_dtype_attached_vs_merged_effect_classified": True,
            "source_inputs_unchanged": True,
        }
        or next_action.get("constraints") != constraints
        or gate.get("runtime_eligible") is not False
        or gate.get("runtime_eligibility_reason") != expected_classification
        or gate.get("offline") is not True
    ):
        raise GateError("Tool Router FP32 merge-drift acceptance drift")
    return gate


def _validate_fp32_attached_merge_isolation(
    drift: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_fp32_attached_merge_isolation import (
        analyze_attached_repeat_stability,
        analyze_same_dtype_tokens,
        classify_same_dtype_effect,
        select_comparison_step,
    )
    from fullcycle_bridge.tool_router_merge_remediation import token_ids_sha256
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_FP32_ATTACHED_MERGE_ISOLATION_PATH)
    _validate_finite_json(gate, "$")
    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    training = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    stability = _load_json(TOOL_ROUTER_MERGE_STABILITY_PATH)
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    expected_top_level = {
        "fp32_attached_merge_isolation_version",
        "experiment_id",
        "source_experiment_id",
        "drift_evidence_sha256",
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
        "frozen_fp32_merged_reference",
        "frozen_bf16_context",
        "runs",
        "attached_repeat_stability",
        "merged_candidate_reproduction",
        "same_dtype_token_analysis",
        "comparison_step",
        "selection_score_evidence",
        "raw_logit_evidence",
        "same_dtype_trace_identity",
        "classification",
        "causal_scope",
        "isolation_gate",
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
    if (
        set(gate) != expected_top_level
        or file_sha256(TOOL_ROUTER_FP32_ATTACHED_MERGE_ISOLATION_PATH)
        != "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26"
        or gate.get("fp32_attached_merge_isolation_version") != 1
        or gate.get("experiment_id")
        != "fc-mvp-001-fp32-attached-merge-isolation-v1"
        or gate.get("source_experiment_id") != config.get("experiment_id")
        or gate.get("drift_evidence_sha256")
        != file_sha256(TOOL_ROUTER_FP32_MERGE_DRIFT_PATH)
        or gate.get("remediation_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_REMEDIATION_PATH)
        or gate.get("stability_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_STABILITY_PATH)
        or gate.get("numerics_evidence_sha256")
        != file_sha256(TOOL_ROUTER_MERGE_NUMERICS_PATH)
        or gate.get("training_lock_sha256")
        != file_sha256(ROOT / "requirements" / "training.lock")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("adapter_files") != training["final_adapter"]["files"]
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("example_id") != "eval-001"
        or gate.get("input_token_count") != 339
        or gate.get("input_token_count") != drift.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
        or gate.get("input_token_ids_sha256")
        != drift.get("input_token_ids_sha256")
    ):
        raise GateError("Tool Router FP32 attached/merge source drift")

    expected_storage = {
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
    }
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
    expected_protocol = {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": [
            {
                "run_id": "fp32_attached-r1",
                "path": "fp32_attached_adapter",
                "repeat": 1,
            },
            {
                "run_id": "fp32_attached-r2",
                "path": "fp32_attached_adapter",
                "repeat": 2,
            },
            {
                "run_id": "fp32_safe_merged-r1",
                "path": "fp32_safe_merged",
                "repeat": 1,
            },
        ],
        "fresh_loads_per_path": {
            "fp32_attached_adapter": 2,
            "fp32_safe_merged": 1,
        },
        "max_residual_cuda_bytes": 16777216,
        "paths": {
            "fp32_attached_adapter": {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": False,
                "merge": False,
                "inference_parameter_dtype": "float32",
            },
            "fp32_safe_merged": {
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
    }
    if gate.get("storage_audit") != expected_storage:
        raise GateError("Tool Router FP32 attached/merge storage audit drift")
    if gate.get("protocol") != expected_protocol:
        raise GateError("Tool Router FP32 attached/merge protocol drift")

    drift_runs = drift.get("runs")
    if not isinstance(drift_runs, list):
        raise GateError("Tool Router FP32 drift runs missing")
    drift_merged = [
        run for run in drift_runs if run.get("path") == "fp32_safe_merged"
    ]
    if len(drift_merged) != 1:
        raise GateError("Tool Router FP32 drift merged reference missing")
    drift_merged_run = drift_merged[0]
    expected_merged_reference = {
        "path": "fp32_safe_merged",
        "source_experiment_id": drift["experiment_id"],
        "token_count": drift_merged_run["token_count"],
        "token_ids_sha256": drift_merged_run["token_ids_sha256"],
        "output_sha256": drift_merged_run["output_sha256"],
        "score_trace_sha256": drift_merged_run["generation_trace"]["scores"][
            "trace_sha256"
        ],
        "raw_logit_trace_sha256": drift_merged_run["generation_trace"][
            "raw_logits"
        ]["trace_sha256"],
        "comparison_step_index": 45,
        "comparison_score_vector_sha256": drift_merged_run[
            "generation_trace"
        ]["scores"]["divergent_step_comparison_vector_sha256"],
        "comparison_raw_logit_vector_sha256": drift_merged_run[
            "generation_trace"
        ]["raw_logits"]["divergent_step_comparison_vector_sha256"],
    }
    if gate.get("frozen_fp32_merged_reference") != expected_merged_reference:
        raise GateError("Tool Router FP32 merged reference drift")

    stability_runs = stability.get("runs")
    if not isinstance(stability_runs, list):
        raise GateError("Tool Router BF16 stability runs missing")
    expected_bf16_paths: dict[str, Any] = {}
    for source_path, target_path, boundary_id in (
        ("independent", "bf16_attached_adapter", 1866),
        ("merged", "bf16_safe_merged", 3849),
    ):
        matches = [run for run in stability_runs if run.get("path") == source_path]
        if len(matches) != 2:
            raise GateError("Tool Router BF16 context repeat drift")
        if any(
            matches[index].get(key) != matches[0].get(key)
            for index in (1,)
            for key in ("token_count", "token_ids_sha256", "output_sha256")
        ):
            raise GateError("Tool Router BF16 context identity drift")
        expected_bf16_paths[target_path] = {
            "token_count": matches[0]["token_count"],
            "token_ids_sha256": matches[0]["token_ids_sha256"],
            "output_sha256": matches[0]["output_sha256"],
            "boundary_token_id": boundary_id,
        }
    expected_bf16_context = {
        "context_only": True,
        "gpu_paths_rerun": False,
        "source_experiment_id": stability["experiment_id"],
        "first_divergent_token_index": 45,
        "paths": expected_bf16_paths,
    }
    if gate.get("frozen_bf16_context") != expected_bf16_context:
        raise GateError("Tool Router frozen BF16 context drift")

    runs = gate.get("runs")
    expected_run_keys = {
        "run_id",
        "path",
        "repeat",
        "fresh_load",
        "generated_token_ids",
        "token_count",
        "token_ids_sha256",
        "output_sha256",
        "precision_audit",
        "generation_trace",
        "path_protocol_passed",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
    expected_plan = [
        ("fp32_attached-r1", "fp32_attached_adapter", 1),
        ("fp32_attached-r2", "fp32_attached_adapter", 2),
        ("fp32_safe_merged-r1", "fp32_safe_merged", 1),
    ]
    if (
        not isinstance(runs, list)
        or len(runs) != 3
        or any(not isinstance(run, dict) for run in runs)
        or any(set(run) != expected_run_keys for run in runs)
        or [(run.get("run_id"), run.get("path"), run.get("repeat")) for run in runs]
        != expected_plan
    ):
        raise GateError("Tool Router FP32 attached/merge runs invalid")
    expected_trace_hashes = {
        "fp32_attached-r1": (
            "sha256:e878f06653e43ebf6946a00396fbed7797eecc02dcf25501f0738169a932fdde",
            "sha256:61a891ab427bce3002c3367e2faefd854a11ecb62929d5b187b974a9c3b7f357",
            "sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451",
            "sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a",
        ),
        "fp32_attached-r2": (
            "sha256:e878f06653e43ebf6946a00396fbed7797eecc02dcf25501f0738169a932fdde",
            "sha256:61a891ab427bce3002c3367e2faefd854a11ecb62929d5b187b974a9c3b7f357",
            "sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451",
            "sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a",
        ),
        "fp32_safe_merged-r1": (
            expected_merged_reference["score_trace_sha256"],
            expected_merged_reference["raw_logit_trace_sha256"],
            expected_merged_reference["comparison_score_vector_sha256"],
            expected_merged_reference["comparison_raw_logit_vector_sha256"],
        ),
    }
    expected_token_digest = (
        "sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca"
    )
    expected_output_digest = (
        "sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca"
    )
    for run in runs:
        token_ids = run.get("generated_token_ids")
        trace = run.get("generation_trace")
        if (
            run.get("fresh_load") is not True
            or not isinstance(token_ids, list)
            or len(token_ids) != 48
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool)
                for token_id in token_ids
            )
            or run.get("token_count") != len(token_ids)
            or run.get("token_ids_sha256") != token_ids_sha256(token_ids)
            or run.get("token_ids_sha256") != expected_token_digest
            or run.get("output_sha256") != expected_output_digest
            or run.get("path_protocol_passed") is not True
            or not isinstance(run.get("peak_gpu_memory_bytes"), int)
            or isinstance(run.get("peak_gpu_memory_bytes"), bool)
            or run["peak_gpu_memory_bytes"] < 0
            or not isinstance(run.get("memory_allocated_before_load_bytes"), int)
            or not isinstance(run.get("memory_allocated_after_release_bytes"), int)
            or run["memory_allocated_before_load_bytes"] < 0
            or run["memory_allocated_after_release_bytes"] < 0
            or run["memory_allocated_before_load_bytes"] > 16777216
            or run["memory_allocated_after_release_bytes"] > 16777216
            or not isinstance(trace, dict)
            or set(trace)
            != {
                "step_count",
                "vocabulary_size",
                "cache_returned",
                "scores",
                "raw_logits",
            }
            or trace.get("step_count") != len(token_ids)
            or trace.get("vocabulary_size") != 151936
            or trace.get("cache_returned") is not True
        ):
            raise GateError("Tool Router FP32 attached/merge run evidence drift")
        expected_hashes = expected_trace_hashes[run["run_id"]]
        for trace_key, expected_trace, expected_vector in (
            ("scores", expected_hashes[0], expected_hashes[2]),
            ("raw_logits", expected_hashes[1], expected_hashes[3]),
        ):
            summary = trace.get(trace_key)
            if (
                not isinstance(summary, dict)
                or set(summary)
                != {
                    "native_dtypes",
                    "shape_per_step",
                    "comparison_dtype",
                    "all_finite",
                    "trace_sha256",
                    "comparison_step_index",
                    "comparison_step_vector_sha256",
                }
                or summary.get("native_dtypes") != ["float32"]
                or summary.get("shape_per_step") != [1, 151936]
                or summary.get("comparison_dtype") != "float32"
                or summary.get("all_finite") is not True
                or summary.get("trace_sha256") != expected_trace
                or summary.get("comparison_step_index") != 45
                or summary.get("comparison_step_vector_sha256")
                != expected_vector
            ):
                raise GateError("Tool Router FP32 attached/merge trace linkage drift")

    _validate_fp32_isolation_precision(runs)
    attached = runs[:2]
    merged = runs[2]
    repeat = analyze_attached_repeat_stability(
        attached[0]["generated_token_ids"],
        attached[1]["generated_token_ids"],
        first_output_sha256=attached[0]["output_sha256"],
        second_output_sha256=attached[1]["output_sha256"],
        first_score_trace_sha256=attached[0]["generation_trace"]["scores"][
            "trace_sha256"
        ],
        second_score_trace_sha256=attached[1]["generation_trace"]["scores"][
            "trace_sha256"
        ],
        first_raw_logit_trace_sha256=attached[0]["generation_trace"][
            "raw_logits"
        ]["trace_sha256"],
        second_raw_logit_trace_sha256=attached[1]["generation_trace"][
            "raw_logits"
        ]["trace_sha256"],
        precision_audits_identical=(
            attached[0]["precision_audit"] == attached[1]["precision_audit"]
        ),
    )
    repeat["comparison_score_vector_identity"] = (
        attached[0]["generation_trace"]["scores"][
            "comparison_step_vector_sha256"
        ]
        == attached[1]["generation_trace"]["scores"][
            "comparison_step_vector_sha256"
        ]
    )
    repeat["comparison_raw_logit_vector_identity"] = (
        attached[0]["generation_trace"]["raw_logits"][
            "comparison_step_vector_sha256"
        ]
        == attached[1]["generation_trace"]["raw_logits"][
            "comparison_step_vector_sha256"
        ]
    )
    repeat["passed"] = all(value for key, value in repeat.items() if key != "passed")
    if gate.get("attached_repeat_stability") != repeat or not repeat["passed"]:
        raise GateError("Tool Router attached FP32 repeat stability drift")

    reference = expected_merged_reference
    reproduction = {
        "token_identity": (
            merged["token_count"] == reference["token_count"]
            and merged["token_ids_sha256"] == reference["token_ids_sha256"]
        ),
        "output_identity": merged["output_sha256"] == reference["output_sha256"],
        "score_trace_identity": merged["generation_trace"]["scores"][
            "trace_sha256"
        ]
        == reference["score_trace_sha256"],
        "raw_logit_trace_identity": merged["generation_trace"]["raw_logits"][
            "trace_sha256"
        ]
        == reference["raw_logit_trace_sha256"],
        "comparison_score_vector_identity": merged["generation_trace"]["scores"][
            "comparison_step_vector_sha256"
        ]
        == reference["comparison_score_vector_sha256"],
        "comparison_raw_logit_vector_identity": merged["generation_trace"][
            "raw_logits"
        ]["comparison_step_vector_sha256"]
        == reference["comparison_raw_logit_vector_sha256"],
    }
    reproduction["passed"] = all(reproduction.values())
    if gate.get("merged_candidate_reproduction") != reproduction:
        raise GateError("Tool Router FP32 merged reproduction drift")

    token_analysis = analyze_same_dtype_tokens(
        attached[0]["generated_token_ids"],
        merged["generated_token_ids"],
    )
    if gate.get("same_dtype_token_analysis") != {
        **token_analysis,
        "attached_token_text": None,
        "merged_token_text": None,
    }:
        raise GateError("Tool Router same-dtype token analysis drift")
    comparison_step = select_comparison_step(
        token_analysis,
        frozen_boundary_index=45,
    )
    if gate.get("comparison_step") != comparison_step:
        raise GateError("Tool Router same-dtype comparison step drift")

    _validate_fp32_isolation_step_evidence(
        gate,
        runs,
        evidence_key="selection_score_evidence",
        source="generated.scores",
        semantics="processed_prediction_scores_after_logits_processors",
        value_key="score",
        expected_values={
            "fp32_attached_adapter": [
                35.61114501953125,
                33.16929626464844,
                24.496919631958008,
                23.684675216674805,
                21.257125854492188,
            ],
            "fp32_safe_merged": [
                35.61100387573242,
                33.169429779052734,
                24.49706268310547,
                23.684528350830078,
                21.257160186767578,
            ],
        },
        expected_top_ids={
            "fp32_attached_adapter": [3849, 1866, 830, 895, 2641],
            "fp32_safe_merged": [3849, 1866, 830, 895, 2641],
        },
        expected_delta={
            "vocabulary_elements": 151936,
            "nonzero_elements": 150968,
            "max_abs_delta": 0.0001735687255859375,
            "mean_abs_delta": 2.052841409749817e-05,
            "root_mean_square_delta": 2.6467831048648804e-05,
        },
    )
    _validate_fp32_isolation_step_evidence(
        gate,
        runs,
        evidence_key="raw_logit_evidence",
        source="generated.logits",
        semantics="unprocessed_lm_head_prediction_scores",
        value_key="raw_logit",
        expected_values={
            "fp32_attached_adapter": [
                39.17226028442383,
                36.48622512817383,
                26.053144454956055,
                24.496919631958008,
                21.257125854492188,
            ],
            "fp32_safe_merged": [
                39.17210388183594,
                36.48637390136719,
                26.052982330322266,
                24.49706268310547,
                21.257160186767578,
            ],
        },
        expected_top_ids={
            "fp32_attached_adapter": [3849, 1866, 895, 830, 2641],
            "fp32_safe_merged": [3849, 1866, 895, 830, 2641],
        },
        expected_delta={
            "vocabulary_elements": 151936,
            "nonzero_elements": 150968,
            "max_abs_delta": 0.0001735687255859375,
            "mean_abs_delta": 2.052839772659354e-05,
            "root_mean_square_delta": 2.6469620934221894e-05,
        },
    )

    score = gate["selection_score_evidence"]
    raw = gate["raw_logit_evidence"]
    trace_identity = {
        "token_identity": token_analysis["cross_path_identical"],
        "score_trace_identity": attached[0]["generation_trace"]["scores"][
            "trace_sha256"
        ]
        == merged["generation_trace"]["scores"]["trace_sha256"],
        "raw_logit_trace_identity": attached[0]["generation_trace"]["raw_logits"][
            "trace_sha256"
        ]
        == merged["generation_trace"]["raw_logits"]["trace_sha256"],
        "comparison_score_vector_identity": score["paths"][
            "fp32_attached_adapter"
        ]["comparison_vector_sha256"]
        == score["paths"]["fp32_safe_merged"]["comparison_vector_sha256"],
        "comparison_raw_logit_vector_identity": raw["paths"][
            "fp32_attached_adapter"
        ]["comparison_vector_sha256"]
        == raw["paths"]["fp32_safe_merged"]["comparison_vector_sha256"],
    }
    if gate.get("same_dtype_trace_identity") != trace_identity:
        raise GateError("Tool Router same-dtype trace identity drift")
    classification = classify_same_dtype_effect(
        token_analysis,
        attached_repeat_stable=repeat["passed"],
        merged_candidate_reproduced=reproduction["passed"],
        attached_emitted_token_id=score["paths"]["fp32_attached_adapter"][
            "emitted_token_id"
        ],
        merged_emitted_token_id=score["paths"]["fp32_safe_merged"][
            "emitted_token_id"
        ],
        attached_score_top_token_id=score["paths"]["fp32_attached_adapter"][
            "top_token_ids"
        ][0],
        merged_score_top_token_id=score["paths"]["fp32_safe_merged"][
            "top_token_ids"
        ][0],
        attached_raw_logit_top_token_id=raw["paths"]["fp32_attached_adapter"][
            "top_token_ids"
        ][0],
        merged_raw_logit_top_token_id=raw["paths"]["fp32_safe_merged"][
            "top_token_ids"
        ][0],
        full_score_traces_identical=trace_identity["score_trace_identity"],
        full_raw_logit_traces_identical=trace_identity["raw_logit_trace_identity"],
        comparison_score_vectors_identical=trace_identity[
            "comparison_score_vector_identity"
        ],
        comparison_raw_logit_vectors_identical=trace_identity[
            "comparison_raw_logit_vector_identity"
        ],
    )
    expected_classification = (
        "deterministic_fp32_attached_vs_merged_"
        "numerical_drift_without_token_drift"
    )
    expected_causal_scope = {
        "isolated_variable": (
            "attached_adapter_vs_materialized_safe_merge_execution"
        ),
        "controlled": [
            "base_checkpoint_values",
            "base_and_adapter_runtime_dtype_float32",
            "adapter_weights",
            "eval_001_rendered_input",
            "greedy_decoding",
            "high_level_transformers_sdpa_dispatch",
            "fresh_model_load_lifecycle",
        ],
        "supports": (
            "classification of the observed same-dtype FP32 execution-form effect"
        ),
        "does_not_support": [
            "peft_merge_implementation_bug_claim",
            "low_level_cuda_kernel_identity_or_root_cause",
            "full_eval_generalization",
            "merged_artifact_promotion",
            "runtime_eligibility",
        ],
    }
    expected_isolation_gate = {
        "attached_fp32_repeat_stable": repeat["passed"],
        "fp32_merged_candidate_reproduced": reproduction["passed"],
        "same_dtype_exact_cached_step_compared": True,
        "processed_score_argmax_matches_generated_token": True,
        "raw_logits_captured": True,
        "same_dtype_effect_classified": True,
        "passed": True,
    }
    expected_acceptance_keys = {
        "upstream_evidence_locked",
        "frozen_input_reproduced",
        "attached_fp32_repeat_stable",
        "fp32_candidate_reproduced",
        "same_dtype_exact_step_compared",
        "same_dtype_attached_vs_merged_effect_classified",
        "generation_score_alignment_verified",
        "path_protocols_executed",
        "source_storage_dtypes_locked",
        "fresh_load_memory_isolated",
        "source_adapter_unchanged",
        "source_model_unchanged",
        "eval_digest_unchanged",
        "prompt_digest_unchanged",
        "frozen_bf16_context_only",
    }
    expected_next_action = {
        "gate_id": "FC-MVP-001-fp32-attached-merge-numerics-v1",
        "action": (
            "at frozen comparison step index 45, locate the first module numerical "
            "divergence between repeat-stable FP32 attached LoRA execution and the "
            "unchanged materialized safe-merged execution, without claiming a "
            "same-dtype token boundary"
        ),
        "acceptance": {
            "fp32_paths_reproduced": True,
            "comparison_step_reproduced": True,
            "first_divergent_module_located": True,
            "operation_order_boundary_quantified": True,
            "source_inputs_unchanged": True,
        },
        "constraints": expected_constraints,
    }
    elapsed = gate.get("elapsed_seconds")
    peak = gate.get("peak_gpu_memory_bytes")
    acceptance = gate.get("acceptance")
    if (
        classification != expected_classification
        or gate.get("classification") != classification
        or gate.get("causal_scope") != expected_causal_scope
        or gate.get("isolation_gate") != expected_isolation_gate
        or gate.get("remediation_gate")
        != {
            "source_gate_passed": False,
            "new_remediation_tested": False,
            "passed": False,
        }
        or not isinstance(acceptance, dict)
        or set(acceptance) != expected_acceptance_keys
        or not all(value is True for value in acceptance.values())
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not isinstance(peak, int)
        or isinstance(peak, bool)
        or peak < 0
        or peak != max(run["peak_gpu_memory_bytes"] for run in runs)
        or gate.get("merged_artifact_saved") is not False
        or gate.get("merged_artifact_allowed") is not False
        or gate.get("constraints") != expected_constraints
        or gate.get("locked_next_action") != expected_next_action
        or gate.get("runtime_eligible") is not False
        or gate.get("runtime_eligibility_reason") != expected_classification
        or gate.get("offline") is not True
    ):
        raise GateError("Tool Router FP32 attached/merge acceptance drift")
    return gate


def _validate_fp32_attached_merge_numerics(
    isolation: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_fp32_attached_merge_numerics_archive import (
        validate_frozen_numerics_evidence,
    )
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    gate = _load_json(TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_PATH)
    _validate_finite_json(gate, "$")
    if (
        not TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_TENSORS_PATH.is_file()
        or TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_TENSORS_PATH.is_symlink()
    ):
        raise GateError("Tool Router FP32 numerics tensor archive is unsafe")
    payload = TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_TENSORS_PATH.read_bytes()
    raw_validation = validate_frozen_numerics_evidence(gate, payload)
    if (
        raw_validation.get("frozen_gate_valid") is not True
        or raw_validation.get("record_count") != 138
        or raw_validation.get("comparisons_recomputed") != 35
        or raw_validation.get("weight_elements_recomputed") != 2_359_296
        or raw_validation.get("bias_elements_recomputed") != 1_536
    ):
        raise GateError("Tool Router FP32 numerics raw validation drift")

    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    training = _load_json(
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    expected_lineage = {
        "isolation_evidence_sha256": file_sha256(
            TOOL_ROUTER_FP32_ATTACHED_MERGE_ISOLATION_PATH
        ),
        "drift_evidence_sha256": file_sha256(
            TOOL_ROUTER_FP32_MERGE_DRIFT_PATH
        ),
        "remediation_evidence_sha256": file_sha256(
            TOOL_ROUTER_MERGE_REMEDIATION_PATH
        ),
        "stability_evidence_sha256": file_sha256(
            TOOL_ROUTER_MERGE_STABILITY_PATH
        ),
        "bf16_numerics_context_sha256": file_sha256(
            TOOL_ROUTER_MERGE_NUMERICS_PATH
        ),
    }
    if (
        file_sha256(TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_PATH)
        != "sha256:cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91"
        or gate.get("source_lineage") != expected_lineage
        or gate.get("source_experiment_id") != isolation.get("experiment_id")
        or gate.get("training_lock_sha256")
        != file_sha256(ROOT / "requirements" / "training.lock")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != directory_artifact_manifest(adapter)
        or gate.get("adapter_files") != training["final_adapter"]["files"]
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("eval_digest") != isolation.get("eval_digest")
        or gate.get("input_token_count") != isolation.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != isolation.get("input_token_ids_sha256")
        or gate.get("storage_audit") != isolation.get("storage_audit")
        or gate.get("frozen_bf16_context")
        != isolation.get("frozen_bf16_context")
        or isolation.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-fp32-attached-merge-numerics-v1"
    ):
        raise GateError("Tool Router FP32 numerics source lineage drift")
    expected_environment = {
        **config["environment"],
        "base_and_adapter_runtime_dtype": "float32",
        "autocast": False,
        "tf32": False,
    }
    if gate.get("environment") != expected_environment:
        raise GateError("Tool Router FP32 numerics environment drift")

    expected_references: dict[str, Any] = {}
    for path in ("fp32_attached_adapter", "fp32_safe_merged"):
        matches = [run for run in isolation["runs"] if run.get("path") == path]
        if not matches:
            raise GateError("Tool Router FP32 numerics source run missing")
        run = matches[0]
        expected_references[path] = {
            "token_count": run["token_count"],
            "token_ids_sha256": run["token_ids_sha256"],
            "output_sha256": run["output_sha256"],
            "score_trace_sha256": run["generation_trace"]["scores"][
                "trace_sha256"
            ],
            "raw_logit_trace_sha256": run["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            "comparison_step_index": 45,
            "comparison_score_vector_sha256": run["generation_trace"][
                "scores"
            ]["comparison_step_vector_sha256"],
            "comparison_raw_logit_vector_sha256": run["generation_trace"][
                "raw_logits"
            ]["comparison_step_vector_sha256"],
        }
    if gate.get("frozen_path_references") != expected_references:
        raise GateError("Tool Router FP32 numerics frozen reference drift")
    runs = gate["runs"]
    if (
        gate.get("peak_gpu_memory_bytes")
        != max(run["peak_gpu_memory_bytes"] for run in runs)
        or not isinstance(gate.get("elapsed_seconds"), (int, float))
        or isinstance(gate.get("elapsed_seconds"), bool)
        or not math.isfinite(gate["elapsed_seconds"])
        or gate["elapsed_seconds"] <= 0
    ):
        raise GateError("Tool Router FP32 numerics resource evidence drift")
    return gate


def _validate_attached_dtype_isolation(
    fp32_numerics: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_attached_dtype_isolation_evidence import (
        validate_attached_dtype_isolation_evidence,
    )
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    if (
        not TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH.is_file()
        or TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH.is_symlink()
    ):
        raise GateError("Tool Router attached dtype isolation evidence is unsafe")
    gate = _load_json(TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH)
    _validate_finite_json(gate, "$")

    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    training_path = (
        ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    )
    training = _load_json(training_path)
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    expected_lineage = {
        "stability_evidence_sha256": file_sha256(
            TOOL_ROUTER_MERGE_STABILITY_PATH
        ),
        "drift_evidence_sha256": file_sha256(TOOL_ROUTER_FP32_MERGE_DRIFT_PATH),
        "isolation_evidence_sha256": file_sha256(
            TOOL_ROUTER_FP32_ATTACHED_MERGE_ISOLATION_PATH
        ),
        "fp32_numerics_evidence_sha256": file_sha256(
            TOOL_ROUTER_FP32_ATTACHED_MERGE_NUMERICS_PATH
        ),
        "training_evidence_sha256": file_sha256(training_path),
    }
    adapter_files = directory_artifact_manifest(adapter)
    validation = validate_attached_dtype_isolation_evidence(
        gate,
        expected_source_lineage=expected_lineage,
        expected_adapter_files=adapter_files,
        expected_environment=config["environment"],
    )
    expected_validation = {
        "frozen_gate_valid": True,
        "runs_validated": 4,
        "token_digests_recomputed": 6,
        "comparison_step_manifests_validated": 8,
        "classification": (
            "deterministic_bf16_attached_vs_fp32_attached_"
            "raw_logit_boundary_flip"
        ),
        "delta_statistics_scope": "probe_derived_summary_algebra_only",
    }
    if validation != expected_validation:
        raise GateError("Tool Router attached dtype raw validation drift")

    if (
        file_sha256(TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH)
        != "sha256:7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31"
        or gate.get("source_lineage") != expected_lineage
        or gate.get("source_experiment_id") != training.get("experiment_id")
        or gate.get("training_lock_sha256")
        != file_sha256(ROOT / "requirements" / "training.lock")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != adapter_files
        or gate.get("adapter_files") != training["final_adapter"]["files"]
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("eval_digest") != fp32_numerics.get("eval_digest")
        or gate.get("input_token_count") != fp32_numerics.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != fp32_numerics.get("input_token_ids_sha256")
        or gate.get("storage_audit") != fp32_numerics.get("storage_audit")
        or fp32_numerics.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-isolation-v1"
    ):
        raise GateError("Tool Router attached dtype source lineage drift")
    return gate


def _validate_attached_dtype_numerics(
    attached_dtype_isolation: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_attached_dtype_numerics_evidence import (
        validate_attached_dtype_numerics_evidence,
    )
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    if (
        not TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH.is_file()
        or TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH.is_symlink()
    ):
        raise GateError("Tool Router attached dtype numerics evidence is unsafe")
    gate = _load_json(TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH)
    _validate_finite_json(gate, "$")

    config = _load_json(ROOT / "configs" / "tool_router_lora_sft_v2.json")
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    adapter_files = directory_artifact_manifest(adapter)
    expected_lineage = dict(attached_dtype_isolation["source_lineage"])
    expected_lineage["attached_dtype_isolation_evidence_sha256"] = file_sha256(
        TOOL_ROUTER_ATTACHED_DTYPE_ISOLATION_PATH
    )
    validation = validate_attached_dtype_numerics_evidence(
        gate,
        source_isolation=attached_dtype_isolation,
        expected_source_lineage=expected_lineage,
        expected_adapter_files=adapter_files,
        expected_environment=attached_dtype_isolation["environment"],
    )
    expected_validation = {
        "frozen_gate_valid": True,
        "runs_validated": 4,
        "capture_records_validated": 160,
        "capture_manifests_validated": 4,
        "module_comparisons_validated": 40,
        "first_unequal_module": "model.layers.0.input_layernorm",
        "classification": (
            "deterministic_attached_bf16_vs_fp32_registered_module_"
            "output_drift_reaching_lm_head"
        ),
        "delta_statistics_scope": (
            "probe_derived_summary_algebra_and_frozen_manifest_only"
        ),
    }
    if validation != expected_validation:
        raise GateError("Tool Router attached dtype numerics raw validation drift")
    if (
        file_sha256(TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH)
        != "sha256:de5b048a5d254f61ab3bef1ff23f1484b07808c86a1679bc0de4ee58e8c8d7c5"
        or gate.get("source_lineage") != expected_lineage
        or gate.get("source_experiment_id")
        != attached_dtype_isolation.get("source_experiment_id")
        or gate.get("source_gate_experiment_id")
        != attached_dtype_isolation.get("experiment_id")
        or gate.get("training_lock_sha256")
        != attached_dtype_isolation.get("training_lock_sha256")
        or gate.get("config_sha256") != canonical_config_sha256(config)
        or gate.get("adapter_files") != adapter_files
        or gate.get("model_weight_sha256")
        != f"sha256:{config['model']['weight_sha256']}"
        or gate.get("prompt_sha256")
        != file_sha256(ROOT / config["prompt"]["path"])
        or gate.get("eval_digest") != fixture_digest(evaluation)
        or gate.get("eval_digest") != attached_dtype_isolation.get("eval_digest")
        or gate.get("input_token_count")
        != attached_dtype_isolation.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != attached_dtype_isolation.get("input_token_ids_sha256")
        or gate.get("storage_audit")
        != attached_dtype_isolation.get("storage_audit")
        or attached_dtype_isolation.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-numerics-v1"
    ):
        raise GateError("Tool Router attached dtype numerics source lineage drift")
    return gate


def _validate_attached_dtype_boundary_control(
    attached_dtype_numerics: dict[str, Any],
) -> dict[str, Any]:
    from fullcycle_bridge.tool_router_attached_dtype_boundary_control_evidence import (
        validate_attached_dtype_boundary_control_evidence,
    )
    from fullcycle_bridge.tool_router_sft import (
        directory_artifact_manifest,
        file_sha256,
    )

    if (
        not TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH.is_file()
        or TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH.is_symlink()
    ):
        raise GateError("Tool Router attached dtype boundary-control evidence is unsafe")
    gate = _load_json(TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH)
    _validate_finite_json(gate, "$")

    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    adapter_files = directory_artifact_manifest(adapter)
    source_numerics_sha256 = file_sha256(
        TOOL_ROUTER_ATTACHED_DTYPE_NUMERICS_PATH
    )
    expected_lineage = dict(attached_dtype_numerics["source_lineage"])
    expected_lineage["attached_dtype_numerics_evidence_sha256"] = (
        source_numerics_sha256
    )
    validation = validate_attached_dtype_boundary_control_evidence(
        gate,
        source_numerics=attached_dtype_numerics,
        expected_source_numerics_sha256=source_numerics_sha256,
        expected_source_lineage=expected_lineage,
        expected_adapter_files=adapter_files,
        expected_environment=attached_dtype_numerics["environment"],
    )
    expected_validation = {
        "frozen_gate_valid": True,
        "actual_runs_validated": 4,
        "control_runs_validated": 4,
        "capture_records_validated": 28,
        "actual_comparisons_validated": 4,
        "control_comparisons_validated": 3,
        "protocol_completed": True,
        "current_forward_boundary_sufficiency_observed": True,
        "classification": (
            "deterministic_same_values_rmsnorm_dtype_replay_"
            "reproduces_actual_boundary_drift"
        ),
    }
    if validation != expected_validation:
        raise GateError(
            "Tool Router attached dtype boundary-control raw validation drift"
        )
    if (
        file_sha256(TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH)
        != "sha256:fdf4ab44b1b60853f0d5de9f231ce77557152b47c9ce52156c31c9bbca484bc7"
        or gate.get("source_lineage") != expected_lineage
        or gate.get("source_experiment_id")
        != attached_dtype_numerics.get("source_experiment_id")
        or gate.get("source_gate_experiment_id")
        != attached_dtype_numerics.get("experiment_id")
        or gate.get("training_lock_sha256")
        != attached_dtype_numerics.get("training_lock_sha256")
        or gate.get("config_sha256")
        != attached_dtype_numerics.get("config_sha256")
        or gate.get("adapter_files") != adapter_files
        or gate.get("model_weight_sha256")
        != attached_dtype_numerics.get("model_weight_sha256")
        or gate.get("prompt_sha256")
        != attached_dtype_numerics.get("prompt_sha256")
        or gate.get("eval_digest")
        != attached_dtype_numerics.get("eval_digest")
        or gate.get("input_token_count")
        != attached_dtype_numerics.get("input_token_count")
        or gate.get("input_token_ids_sha256")
        != attached_dtype_numerics.get("input_token_ids_sha256")
        or gate.get("storage_audit")
        != attached_dtype_numerics.get("storage_audit")
        or gate.get("environment") != attached_dtype_numerics.get("environment")
        or attached_dtype_numerics.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-boundary-control-v1"
    ):
        raise GateError("Tool Router attached dtype boundary-control source lineage drift")
    return gate


def _validate_fp32_attached_remediation_eval(
    attached_dtype_boundary_control: dict[str, Any],
) -> dict[str, Any]:
    import inspect

    from fullcycle_bridge.tool_router import fixture_digest, load_fixture
    from fullcycle_bridge.tool_router_decision_compilation import compile_decision
    from fullcycle_bridge.tool_router_fp32_attached_remediation_eval_evidence import (
        validate_fp32_attached_remediation_eval_evidence,
    )
    from fullcycle_bridge.tool_router_sft import (
        canonical_config_sha256,
        directory_artifact_manifest,
        file_sha256,
    )

    paths = (
        TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_PREREGISTRATION_PATH,
        TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_PREDICTIONS_PATH,
        TOOL_ROUTER_FP32_ATTACHED_REMEDIATION_EVAL_PATH,
    )
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise GateError("Tool Router FP32 attached remediation evidence is unsafe")

    preregistration = _load_json(paths[0])
    predictions = _load_json(paths[1])
    gate = _load_json(paths[2])
    for name, value in (
        ("preregistration", preregistration),
        ("predictions", predictions),
        ("gate", gate),
    ):
        _validate_finite_json(value, f"$.{name}")

    config_path = ROOT / "configs" / "tool_router_lora_sft_v2.json"
    config = _load_json(config_path)
    training_lock_path = ROOT / "requirements" / "training.lock"
    training_path = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
    lifecycle_path = TOOL_ROUTER_SFT_V2_BASELINE_PATH
    compiler_path = ROOT / "src" / "fullcycle_bridge" / "tool_router_decision_compilation.py"
    scorer_path = ROOT / "src" / "fullcycle_bridge" / "tool_router_model_eval.py"
    contract_path = (
        ROOT
        / "src"
        / "fullcycle_bridge"
        / "tool_router_fp32_attached_remediation_eval.py"
    )
    runner_path = ROOT / "scripts" / "probe_tool_router_fp32_attached_remediation_eval.py"
    raw_predictions_path = (
        ROOT
        / "baseline"
        / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
    )
    raw_report_path = (
        ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
    )
    preregistration_sha256 = file_sha256(paths[0])
    prediction_sha256 = file_sha256(paths[1])
    compiler_symbol_sha256 = "sha256:" + hashlib.sha256(
        inspect.getsource(compile_decision).encode("utf-8")
    ).hexdigest()
    expected_source_lineage = {
        "base_config_canonical_sha256": canonical_config_sha256(config),
        "base_config_sha256": file_sha256(config_path),
        "bf16_compiled_predictions_sha256": file_sha256(
            TOOL_ROUTER_COMPILED_PREDICTIONS_PATH
        ),
        "bf16_compiled_report_sha256": file_sha256(
            TOOL_ROUTER_COMPILED_REPORT_PATH
        ),
        "bf16_raw_predictions_sha256": file_sha256(raw_predictions_path),
        "bf16_raw_report_sha256": file_sha256(raw_report_path),
        "boundary_control_evidence_sha256": file_sha256(
            TOOL_ROUTER_ATTACHED_DTYPE_BOUNDARY_CONTROL_PATH
        ),
        "comparison_contract_source_sha256": file_sha256(contract_path),
        "decision_compilation_gate_sha256": file_sha256(
            TOOL_ROUTER_DECISION_COMPILATION_PATH
        ),
        "decision_compiler_source_sha256": file_sha256(compiler_path),
        "decision_compiler_source_symbol_source_sha256": (
            compiler_symbol_sha256
        ),
        "lifecycle_evidence_sha256": file_sha256(lifecycle_path),
        "preregistration_sha256": preregistration_sha256,
        "runner_source_sha256": file_sha256(runner_path),
        "scorer_source_sha256": file_sha256(scorer_path),
        "training_evidence_sha256": file_sha256(training_path),
        "training_lock_sha256": file_sha256(training_lock_path),
    }
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise GateError("Tool Router FP32 attached remediation eval drift")
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    validation = validate_fp32_attached_remediation_eval_evidence(
        preregistration,
        predictions,
        gate,
        evaluation=evaluation,
        reference_compiled_report=_load_json(TOOL_ROUTER_COMPILED_REPORT_PATH),
        source_boundary_control=attached_dtype_boundary_control,
        expected_source_lineage=expected_source_lineage,
        expected_model=config["model"],
        expected_tokenizer=config["tokenizer"],
        expected_environment=config["environment"],
        expected_adapter_files=directory_artifact_manifest(adapter),
        expected_preregistration_sha256=preregistration_sha256,
        expected_prediction_artifact_sha256=prediction_sha256,
    )
    expected_validation = {
        "frozen_gate_valid": True,
        "candidate_count": 1,
        "run_count": 1,
        "evaluation_records": 20,
        "raw_outputs_validated": 20,
        "compiled_outputs_validated": 20,
        "classification": (
            "fp32_attached_full_eval_improves_quality_without_safety_or_"
            "resource_regression"
        ),
        "remediation_passed": True,
        "runtime_eligible": False,
    }
    if validation != expected_validation:
        raise GateError("Tool Router FP32 attached remediation validation drift")
    if (
        preregistration_sha256
        != "sha256:5e7b0665f97f5cee760637236f80039c4e621ae0f24915c0ac749d885a683c8b"
        or prediction_sha256
        != "sha256:382071f0689ce4ca41329d689f76fc4c4b06faa68769fb80c99181015e678115"
        or file_sha256(paths[2])
        != "sha256:2dd17f6b1098490034f825d163f48f26eb4093d02f115424eb814cb2c925ad8e"
        or attached_dtype_boundary_control.get("locked_next_action", {}).get(
            "gate_id"
        )
        != "FC-MVP-001-fp32-attached-remediation-eval-v1"
    ):
        raise GateError("Tool Router FP32 attached remediation lineage drift")
    return gate


def _validate_fp32_isolation_precision(runs: list[dict[str, Any]]) -> None:
    expected_generation = {
        "score_dtypes": ["float32"],
        "all_scores_float32": True,
        "raw_logit_dtypes": ["float32"],
        "all_raw_logits_float32": True,
        "dtype_semantics": "transformers_generate_return_tensor_dtype",
        "autocast_enabled": False,
        "training": False,
    }
    expected_pre = {
        "base_parameters": _expected_cuda_inventory("float32", 1543714304, 338),
        "adapter_parameters": _expected_cuda_inventory("float32", 4358144, 224),
        "floating_buffers": _expected_cuda_inventory("float32", 64, 1),
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
    expected_attached = {
        **expected_pre,
        "lora_dropout": {"modules": 112, "training_modules": 0},
        "generation": expected_generation,
    }
    if any(run.get("precision_audit") != expected_attached for run in runs[:2]):
        raise GateError("Tool Router attached FP32 precision audit drift")
    expected_post = {
        "parameters": _expected_cuda_inventory("float32", 1543714304, 338),
        "floating_buffers": _expected_cuda_inventory("float32", 64, 1),
        "lora_target_modules": 0,
        "lora_parameter_tensors": 0,
        "is_peft_model": False,
        "input_output_embeddings_tied": True,
        "attn_implementation": "sdpa",
        "attention_class": "Qwen2Attention",
        "output_attentions": False,
        "hf_device_map": None,
    }
    expected_merged = {
        "pre_merge": expected_pre,
        "post_merge": expected_post,
        "lora_dropout": {"modules": 0, "training_modules": 0},
        "generation": expected_generation,
    }
    if runs[2].get("precision_audit") != expected_merged:
        raise GateError("Tool Router merged FP32 precision audit drift")


def _validate_fp32_isolation_step_evidence(
    gate: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    evidence_key: str,
    source: str,
    semantics: str,
    value_key: str,
    expected_values: dict[str, list[float]],
    expected_top_ids: dict[str, list[int]],
    expected_delta: dict[str, Any],
) -> None:
    evidence = gate.get(evidence_key)
    if (
        not isinstance(evidence, dict)
        or set(evidence)
        != {
            "step_index",
            "comparison_basis",
            "shared_generated_prefix_tokens_before_step",
            "source",
            "semantics",
            "comparison_dtype",
            "top_k",
            "paths",
            "delta",
        }
        or evidence.get("step_index") != 45
        or evidence.get("comparison_basis")
        != "frozen_bf16_token_boundary_context"
        or evidence.get("shared_generated_prefix_tokens_before_step") != 45
        or evidence.get("source") != source
        or evidence.get("semantics") != semantics
        or evidence.get("comparison_dtype") != "float32"
        or evidence.get("top_k") != 5
        or evidence.get("delta") != expected_delta
        or set(evidence.get("paths", {}))
        != {"fp32_attached_adapter", "fp32_safe_merged"}
    ):
        raise GateError(f"Tool Router FP32 isolation {evidence_key} drift")
    representatives = {
        "fp32_attached_adapter": runs[0],
        "fp32_safe_merged": runs[2],
    }
    trace_key = "scores" if source == "generated.scores" else "raw_logits"
    top_values_key = f"top_{value_key}s"
    emitted_value_key = f"emitted_token_{value_key}"
    expected_path_keys = {
        "top_token_ids",
        "top_token_texts",
        top_values_key,
        "top_margin",
        "emitted_token_id",
        "emitted_token_text",
        emitted_value_key,
        "compared_tokens",
        "comparison_vector_sha256",
    }
    for path_name, run in representatives.items():
        path = evidence["paths"].get(path_name)
        top_ids = expected_top_ids[path_name]
        top_values = expected_values[path_name]
        if (
            not isinstance(path, dict)
            or set(path) != expected_path_keys
            or path.get("top_token_ids") != top_ids
            or path.get(top_values_key) != top_values
            or path.get("top_margin") != top_values[0] - top_values[1]
            or path.get("emitted_token_id") != 3849
            or path.get("emitted_token_id") != run["generated_token_ids"][45]
            or path.get("emitted_token_id") != top_ids[0]
            or path.get(emitted_value_key) != top_values[0]
            or path.get("comparison_vector_sha256")
            != run["generation_trace"][trace_key][
                "comparison_step_vector_sha256"
            ]
        ):
            raise GateError(f"Tool Router FP32 isolation {evidence_key} path drift")
        compared = path.get("compared_tokens")
        if (
            not isinstance(compared, list)
            or [item.get("token_id") for item in compared] != [3849, 1866]
        ):
            raise GateError(f"Tool Router FP32 isolation {evidence_key} token drift")
        for item in compared:
            token_id = item["token_id"]
            position = top_ids.index(token_id)
            if (
                item.get(value_key) != top_values[position]
                or item.get("rank") != position + 1
            ):
                raise GateError(f"Tool Router FP32 isolation {evidence_key} rank drift")


def _validate_fp32_drift_precision(runs: list[dict[str, Any]]) -> None:
    expected_generation = {
        "score_dtypes": ["float32"],
        "all_scores_float32": True,
        "raw_logit_dtypes": ["float32"],
        "all_raw_logits_float32": True,
        "autocast_enabled": False,
        "training": False,
    }
    independent = runs[0].get("precision_audit")
    if (
        not isinstance(independent, dict)
        or independent.get("base_parameters")
        != _expected_cuda_inventory("bfloat16", 1543714304, 338)
        or independent.get("adapter_parameters")
        != _expected_cuda_inventory("float32", 4358144, 224)
        or independent.get("floating_buffers")
        != _expected_cuda_inventory("float32", 64, 1)
        or independent.get("lora_target_modules") != 112
        or independent.get("lora_parameter_tensors") != 224
        or independent.get("adapter_parameters_finite") is not True
        or independent.get("active_adapters") != ["default"]
        or independent.get("is_peft_model") is not True
        or independent.get("input_output_embeddings_tied") is not True
        or independent.get("attn_implementation") != "sdpa"
        or independent.get("attention_class") != "Qwen2Attention"
        or independent.get("output_attentions") is not False
        or independent.get("hf_device_map") is not None
        or independent.get("generation") != expected_generation
    ):
        raise GateError("Tool Router independent BF16 precision audit drift")

    candidate = runs[1].get("precision_audit")
    if not isinstance(candidate, dict):
        raise GateError("Tool Router FP32 candidate precision audit missing")
    pre = candidate.get("pre_merge")
    post = candidate.get("post_merge")
    if (
        not isinstance(pre, dict)
        or pre.get("base_parameters")
        != _expected_cuda_inventory("float32", 1543714304, 338)
        or pre.get("adapter_parameters")
        != _expected_cuda_inventory("float32", 4358144, 224)
        or pre.get("floating_buffers")
        != _expected_cuda_inventory("float32", 64, 1)
        or pre.get("lora_target_modules") != 112
        or pre.get("lora_parameter_tensors") != 224
        or pre.get("adapter_parameters_finite") is not True
        or pre.get("active_adapters") != ["default"]
        or pre.get("is_peft_model") is not True
        or pre.get("input_output_embeddings_tied") is not True
        or pre.get("attn_implementation") != "sdpa"
        or pre.get("attention_class") != "Qwen2Attention"
        or pre.get("output_attentions") is not False
        or pre.get("hf_device_map") is not None
        or not isinstance(post, dict)
        or post.get("parameters")
        != _expected_cuda_inventory("float32", 1543714304, 338)
        or post.get("floating_buffers")
        != _expected_cuda_inventory("float32", 64, 1)
        or post.get("lora_target_modules") != 0
        or post.get("lora_parameter_tensors") != 0
        or post.get("is_peft_model") is not False
        or post.get("input_output_embeddings_tied") is not True
        or post.get("attn_implementation") != "sdpa"
        or post.get("attention_class") != "Qwen2Attention"
        or post.get("output_attentions") is not False
        or post.get("hf_device_map") is not None
        or candidate.get("generation") != expected_generation
    ):
        raise GateError("Tool Router FP32 candidate precision audit drift")


def _validate_fp32_step_evidence(
    gate: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    evidence_key: str,
    source: str,
    semantics: str,
    value_key: str,
    expected_values: dict[str, list[float]],
    expected_top_ids: dict[str, list[int]],
    expected_delta: dict[str, Any],
) -> None:
    evidence = gate.get(evidence_key)
    expected_keys = {
        "step_index",
        "common_prefix_generated_tokens",
        "source",
        "semantics",
        "comparison_dtype",
        "top_k",
        "paths",
        "delta",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) != expected_keys
        or evidence.get("step_index") != 45
        or evidence.get("common_prefix_generated_tokens") != 45
        or evidence.get("source") != source
        or evidence.get("semantics") != semantics
        or evidence.get("comparison_dtype") != "float32"
        or evidence.get("top_k") != 5
        or evidence.get("delta") != expected_delta
        or set(evidence.get("paths", {}))
        != {"independent_bf16_adapter", "fp32_safe_merged"}
    ):
        raise GateError(f"Tool Router FP32 merge-drift {evidence_key} drift")
    paths = evidence["paths"]
    top_values_key = f"top_{value_key}s"
    emitted_value_key = f"emitted_token_{value_key}"
    expected_path_keys = {
        "top_token_ids",
        "top_token_texts",
        top_values_key,
        "top_margin",
        "emitted_token_id",
        "emitted_token_text",
        emitted_value_key,
        "compared_tokens",
        "comparison_vector_sha256",
    }
    for run in runs:
        path_name = run["path"]
        path = paths.get(path_name)
        top_ids = expected_top_ids[path_name]
        top_values = expected_values[path_name]
        if (
            not isinstance(path, dict)
            or set(path) != expected_path_keys
            or path.get("top_token_ids") != top_ids
            or path.get(top_values_key) != top_values
            or path.get("top_margin") != top_values[0] - top_values[1]
            or path.get("emitted_token_id") != run["generated_token_ids"][45]
            or path.get("emitted_token_id") != top_ids[0]
            or path.get(emitted_value_key) != top_values[0]
            or not _is_canonical_sha256(path.get("comparison_vector_sha256"))
            or run["generation_trace"][
                "scores" if source == "generated.scores" else "raw_logits"
            ]["divergent_step_comparison_vector_sha256"]
            != path.get("comparison_vector_sha256")
        ):
            raise GateError(
                f"Tool Router FP32 merge-drift {evidence_key} path drift"
            )
        compared = path.get("compared_tokens")
        if (
            not isinstance(compared, list)
            or [item.get("token_id") for item in compared] != [1866, 3849]
        ):
            raise GateError(
                f"Tool Router FP32 merge-drift {evidence_key} token drift"
            )
        for item in compared:
            token_id = item["token_id"]
            position = top_ids.index(token_id)
            if (
                item.get(value_key) != top_values[position]
                or item.get("rank") != position + 1
            ):
                raise GateError(
                    f"Tool Router FP32 merge-drift {evidence_key} rank drift"
                )


def _expected_cuda_inventory(
    dtype: str,
    elements: int,
    tensors: int,
) -> dict[str, Any]:
    return {
        "floating_tensors": tensors,
        "floating_elements": elements,
        "dtypes": {dtype: elements},
        "devices": {"cuda:0": elements},
    }


def _is_canonical_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _fp32_cuda_inventory(value: object, *, expected_elements: int) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("floating_tensors"), int)
        and value["floating_tensors"] > 0
        and value.get("floating_elements") == expected_elements
        and value.get("dtypes") == {"float32": expected_elements}
        and value.get("devices") == {"cuda:0": expected_elements}
    )


def _run_tests() -> int:
    suite = unittest.defaultTestLoader.discover(str(TESTS))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise GateError("unit test suite failed")
    return result.testsRun


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise GateError(f"expected object: {path}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise GateError(f"non-finite JSON constant: {value}")


def _validate_finite_json(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateError(f"non-finite JSON number at {path}: {value!r}")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{path}[{index}]")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(
            json.dumps(
                {"valid": False, "code": "FC_MVP_000_GATE_FAILED", "detail": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
