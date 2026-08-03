"""Build and score the locked FC-MVP-001 decision-compilation v1 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_decision_compilation import (  # noqa: E402
    COMPILED_EXPERIMENT_ID,
    compile_frozen_v2_outputs,
)
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs  # noqa: E402

BASELINE = ROOT / "baseline"
PREDICTIONS = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
SOURCE_REPORT = BASELINE / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
CLASSIFICATION = BASELINE / "fc-mvp-001-lora-sft-v2-failure-classification.json"
COMPILED_PREDICTIONS = BASELINE / "tool-router-lora-sft-v2-compiled-predictions.json"
COMPILED_REPORT = BASELINE / "tool-router-lora-sft-v2-compiled-report.json"
GATE = BASELINE / "fc-mvp-001-decision-compilation-v1.json"
EVAL = ROOT / "fixtures" / "tool_router_v1" / "eval.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def main() -> int:
    source_predictions = _load(PREDICTIONS)
    source_report = _load(SOURCE_REPORT)
    classification = _load(CLASSIFICATION)
    source_hashes = {
        "predictions": _sha256(PREDICTIONS),
        "report": _sha256(SOURCE_REPORT),
        "classification": _sha256(CLASSIFICATION),
    }
    compiled = compile_frozen_v2_outputs(
        source_predictions,
        source_report,
        classification,
        source_hashes,
    )
    evaluation = load_fixture(EVAL)
    if fixture_digest(evaluation) != compiled["eval_digest"]:
        raise ValueError("frozen evaluation digest changed")
    scoring_outputs = [
        {"example_id": item["example_id"], "raw_output": item["raw_output"]}
        for item in compiled["outputs"]
    ]
    metrics, parsed_outputs = score_raw_outputs(evaluation, scoring_outputs)
    acceptance = {
        "conflicting_decision_flags": metrics["semantic_failure_counts"].get(
            "CONFLICTING_DECISION_FLAGS", 0
        ),
        "false_refusals": metrics["false_refusals"],
        "dangerous_action_candidates": metrics["dangerous_action_candidates"],
        "dangerous_false_approvals": metrics["dangerous_false_approvals"],
        "eval_digest_unchanged": True,
        "raw_predictions_unchanged": compiled["raw_predictions_unchanged"],
    }
    expected_acceptance = classification["locked_next_action"]["acceptance"]
    if acceptance != expected_acceptance:
        raise ValueError(f"decision compilation gate failed: {acceptance!r}")

    compiled_bytes = _json_bytes(compiled)
    report = {
        "report_version": 1,
        "experiment_id": COMPILED_EXPERIMENT_ID,
        "source_hashes": source_hashes,
        "compiled_prediction_sha256": _bytes_sha256(compiled_bytes),
        "eval_digest": compiled["eval_digest"],
        "metrics": metrics,
        "source_metrics": source_report["metrics"],
        "acceptance": acceptance,
        "parsed_outputs": parsed_outputs,
        "runtime_eligible": False,
        "runtime_eligibility_reason": "offline_compilation_gate_and_merge_drift",
    }
    report_bytes = _json_bytes(report)
    gate = {
        "gate_version": 1,
        "experiment_id": COMPILED_EXPERIMENT_ID,
        "offline_only": True,
        "source_hashes": source_hashes,
        "artifact_hashes": {
            str(COMPILED_PREDICTIONS.relative_to(ROOT)).replace("\\", "/"): (
                _bytes_sha256(compiled_bytes)
            ),
            str(COMPILED_REPORT.relative_to(ROOT)).replace("\\", "/"): (
                _bytes_sha256(report_bytes)
            ),
        },
        "eval_digest": compiled["eval_digest"],
        "acceptance": acceptance,
        "metrics": metrics,
        "constraints": classification["locked_next_action"]["constraints"],
        "merge_policy": classification["merge_policy"],
        "locked_next_action": {
            "gate_id": "FC-MVP-001-bf16-merge-stability-v1",
            "action": (
                "reproduce eval-001 from fresh independent and safe-merged BF16 "
                "loads, then locate the first token or logit divergence"
            ),
            "acceptance": {
                "independent_repeats_identical": True,
                "merged_repeats_identical": True,
                "divergence_classified": True,
                "source_adapter_unchanged": True,
                "eval_digest_unchanged": True,
            },
            "constraints": {
                "new_data": False,
                "training": False,
                "eval_answer_tuning": False,
                "runtime_integration": False,
                "merged_artifact_allowed_before_identity": False,
            },
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": "offline_compilation_gate_and_merge_drift",
    }
    for path, value in (
        (COMPILED_PREDICTIONS, compiled_bytes),
        (COMPILED_REPORT, report_bytes),
        (GATE, _json_bytes(gate)),
    ):
        path.write_bytes(value)
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
