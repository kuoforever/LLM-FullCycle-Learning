"""Compile Tool Router decision-v1 terminal fields without model authority."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, NoReturn

from .consumer import canonical_json_bytes
from .tool_router import DECISION_KEYS, RISK_LEVELS, TOOLS, ToolRouterValidationError

DECISION_COMPILER_VERSION = 1
SOURCE_EXPERIMENT_ID = "fc-mvp-001-lora-sft-v2"
COMPILED_EXPERIMENT_ID = "fc-mvp-001-decision-compilation-v1"
NEXT_GATE_ID = "FC-MVP-001-decision-compilation-v1"
SOURCE_HASH_KEYS = frozenset({"predictions", "report", "classification"})
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def compile_decision(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Derive redundant terminal fields from one selected-tool disposition."""

    if frozenset(prediction) != DECISION_KEYS:
        _fail("INVALID_DECISION_KEYS", "$", repr(sorted(prediction)))
    selected_tool = prediction.get("selected_tool")
    if selected_tool not in TOOLS:
        _fail("INVALID_SELECTED_TOOL", "$.selected_tool", repr(selected_tool))
    arguments = prediction.get("arguments")
    if not isinstance(arguments, Mapping):
        _fail("INVALID_ARGUMENTS", "$.arguments", type(arguments).__name__)
    risk_level = prediction.get("risk_level")
    if risk_level not in RISK_LEVELS:
        _fail("INVALID_RISK_LEVEL", "$.risk_level", repr(risk_level))
    requires_approval = prediction.get("requires_approval")
    if not isinstance(requires_approval, bool):
        _fail(
            "INVALID_APPROVAL_FLAG",
            "$.requires_approval",
            repr(requires_approval),
        )

    should_reject = False
    should_fallback = False
    if selected_tool == "reject_request":
        requires_approval = False
        should_reject = True
        expected_result = "rejection"
    elif selected_tool == "fallback_to_strong_model":
        requires_approval = False
        should_fallback = True
        expected_result = "fallback"
    elif selected_tool == "request_clarification":
        requires_approval = False
        expected_result = "clarification"
    else:
        expected_result = (
            "approval_required" if requires_approval else "tool_candidate"
        )
    return {
        "selected_tool": selected_tool,
        "arguments": dict(arguments),
        "risk_level": risk_level,
        "requires_approval": requires_approval,
        "should_reject": should_reject,
        "should_fallback": should_fallback,
        "expected_result": expected_result,
    }


def compile_frozen_v2_outputs(
    predictions: Mapping[str, Any],
    report: Mapping[str, Any],
    classification: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Compile the frozen outputs after verifying exact report provenance."""

    _validate_sources(predictions, report, classification, source_hashes)
    raw_outputs = predictions.get("outputs")
    parsed_outputs = report.get("parsed_outputs")
    if not isinstance(raw_outputs, list) or not isinstance(parsed_outputs, list):
        _fail("INVALID_OUTPUT_COLLECTION", "$", "outputs must be arrays")
    if len(raw_outputs) != len(parsed_outputs) or not raw_outputs:
        _fail(
            "OUTPUT_COUNT_MISMATCH",
            "$",
            f"{len(raw_outputs)}!={len(parsed_outputs)}",
        )

    compiled_outputs: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    for index, (raw_item, parsed_item) in enumerate(zip(raw_outputs, parsed_outputs)):
        raw = _mapping(raw_item, f"$.predictions.outputs[{index}]")
        parsed = _mapping(parsed_item, f"$.report.parsed_outputs[{index}]")
        example_id = raw.get("example_id")
        if example_id != parsed.get("example_id"):
            _fail("OUTPUT_ID_MISMATCH", f"$[{index}]", repr(example_id))
        if parsed.get("valid") is not True or parsed.get("error") is not None:
            _fail("SOURCE_OUTPUT_INVALID", f"$[{index}]", repr(parsed))
        raw_output = raw.get("raw_output")
        if not isinstance(raw_output, str):
            _fail("INVALID_RAW_OUTPUT", f"$[{index}].raw_output", repr(raw_output))
        try:
            decoded = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ToolRouterValidationError(
                "MALFORMED_RAW_OUTPUT", f"$[{index}].raw_output", str(exc)
            ) from exc
        source_prediction = _mapping(
            parsed.get("prediction"), f"$.report.parsed_outputs[{index}].prediction"
        )
        if decoded != source_prediction:
            _fail("PARSED_OUTPUT_DRIFT", f"$[{index}]", repr(example_id))
        compiled = compile_decision(source_prediction)
        changed_fields = [
            f"$.{key}"
            for key in sorted(DECISION_KEYS)
            if source_prediction[key] != compiled[key]
        ]
        if changed_fields:
            changed_ids.append(str(example_id))
        compiled_outputs.append(
            {
                "example_id": example_id,
                "raw_output": json.dumps(
                    compiled,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "changed_fields": changed_fields,
            }
        )

    conflict_ids = _classified_conflict_ids(classification)
    if sorted(changed_ids) != conflict_ids:
        _fail(
            "UNEXPECTED_COMPILATION_COHORT",
            "$.outputs",
            f"changed={sorted(changed_ids)!r},conflicts={conflict_ids!r}",
        )
    return {
        "artifact_version": 1,
        "compiler_version": DECISION_COMPILER_VERSION,
        "experiment_id": COMPILED_EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_hashes": dict(sorted(source_hashes.items())),
        "source_classification_report_digest": classification["report_digest"],
        "eval_digest": report["eval_digest"],
        "raw_predictions_unchanged": True,
        "changed_example_ids": conflict_ids,
        "outputs": compiled_outputs,
    }


def _validate_sources(
    predictions: Mapping[str, Any],
    report: Mapping[str, Any],
    classification: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    if frozenset(source_hashes) != SOURCE_HASH_KEYS or any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
        for value in source_hashes.values()
    ):
        _fail("INVALID_SOURCE_HASHES", "$.source_hashes", repr(source_hashes))
    if (
        predictions.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or report.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or classification.get("experiment_id") != SOURCE_EXPERIMENT_ID
    ):
        _fail("SOURCE_EXPERIMENT_MISMATCH", "$", SOURCE_EXPERIMENT_ID)
    if report.get("prediction_artifact_sha256") != source_hashes["predictions"]:
        _fail("PREDICTION_DIGEST_MISMATCH", "$.report", "prediction artifact")
    classification_sources = _mapping(
        classification.get("source_hashes"), "$.classification.source_hashes"
    )
    if (
        classification_sources.get("predictions") != source_hashes["predictions"]
        or classification_sources.get("report") != source_hashes["report"]
    ):
        _fail("CLASSIFICATION_SOURCE_MISMATCH", "$.classification", "source hash")
    report_digest = classification.get("report_digest")
    digest_payload = dict(classification)
    digest_payload.pop("report_digest", None)
    expected_digest = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
    )
    if report_digest != expected_digest:
        _fail("CLASSIFICATION_DIGEST_MISMATCH", "$.classification", repr(report_digest))
    next_action = _mapping(
        classification.get("locked_next_action"),
        "$.classification.locked_next_action",
    )
    if next_action.get("gate_id") != NEXT_GATE_ID:
        _fail("UNEXPECTED_NEXT_GATE", "$.classification", repr(next_action))
    if classification.get("runtime_eligible") is not False:
        _fail("UNSAFE_RUNTIME_ELIGIBILITY", "$.classification", "true")
    if predictions.get("eval_digest") != report.get("eval_digest"):
        _fail("EVAL_DIGEST_MISMATCH", "$", "prediction/report")


def _classified_conflict_ids(classification: Mapping[str, Any]) -> list[str]:
    groups = classification.get("failure_groups")
    if not isinstance(groups, list):
        _fail("INVALID_FAILURE_GROUPS", "$.classification.failure_groups", repr(groups))
    for index, value in enumerate(groups):
        group = _mapping(value, f"$.classification.failure_groups[{index}]")
        if group.get("failure_id") == "conflicting_decision_flags":
            identifiers = group.get("example_ids")
            if not isinstance(identifiers, list) or any(
                not isinstance(item, str) for item in identifiers
            ):
                _fail("INVALID_CONFLICT_IDS", f"$.failure_groups[{index}]", repr(identifiers))
            return sorted(identifiers)
    _fail("MISSING_CONFLICT_GROUP", "$.classification.failure_groups", "missing")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", path, type(value).__name__)
    return value


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "COMPILED_EXPERIMENT_ID",
    "DECISION_COMPILER_VERSION",
    "compile_decision",
    "compile_frozen_v2_outputs",
]
