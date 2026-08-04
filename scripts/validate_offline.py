"""Self-contained FC-MVP-000 validation gate for Python 3.11-3.13."""

from __future__ import annotations

import ast
import hashlib
import json
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
        "tool_router_next_gate": merge_stability["locked_next_action"][
            "gate_id"
        ],
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


def _run_tests() -> int:
    suite = unittest.defaultTestLoader.discover(str(TESTS))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise GateError("unit test suite failed")
    return result.testsRun


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"expected object: {path}")
    return value


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
