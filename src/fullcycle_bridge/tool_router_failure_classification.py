"""Frozen-evidence failure classification for FC-MVP-001 LoRA SFT v2."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, NoReturn

from .consumer import canonical_json_bytes
from .tool_router import ToolRouterValidationError

FAILURE_CLASSIFICATION_VERSION = 1
EXPERIMENT_ID = "fc-mvp-001-lora-sft-v2"
SOURCE_KEYS = frozenset({"predictions", "report", "training", "load_merge"})
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def classify_v2_failures(
    predictions: Mapping[str, Any],
    report: Mapping[str, Any],
    training: Mapping[str, Any],
    load_merge: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Classify only the failures exposed by the four frozen v2 artifacts."""

    _validate_sources(predictions, report, training, load_merge, source_hashes)
    metrics = _mapping(report.get("metrics"), "$.report.metrics")
    parsed_outputs = report.get("parsed_outputs")
    if not isinstance(parsed_outputs, list):
        _fail("INVALID_PARSED_OUTPUTS", "$.report.parsed_outputs", repr(parsed_outputs))

    conflicts = [
        _mapping(item, f"$.report.parsed_outputs[{index}]")
        for index, item in enumerate(parsed_outputs)
        if isinstance(item, Mapping)
        and item.get("semantic_error") == "CONFLICTING_DECISION_FLAGS"
    ]
    semantic_counts = _mapping(
        metrics.get("semantic_failure_counts"),
        "$.report.metrics.semantic_failure_counts",
    )
    expected_conflicts = semantic_counts.get("CONFLICTING_DECISION_FLAGS")
    if expected_conflicts != len(conflicts) or not conflicts:
        _fail(
            "CONFLICT_COUNT_MISMATCH",
            "$.report.metrics.semantic_failure_counts",
            f"metric={expected_conflicts!r},parsed={len(conflicts)}",
        )

    conflict_ids: list[str] = []
    for index, item in enumerate(conflicts):
        example_id = item.get("example_id")
        prediction = _mapping(
            item.get("prediction"), f"$.conflicts[{index}].prediction"
        )
        if not isinstance(example_id, str) or not re.fullmatch(r"eval-[0-9]{3}", example_id):
            _fail("INVALID_CONFLICT_ID", f"$.conflicts[{index}].example_id", repr(example_id))
        if (
            prediction.get("selected_tool") != "fallback_to_strong_model"
            or prediction.get("should_fallback") is not True
            or prediction.get("should_reject") is not True
        ):
            _fail(
                "UNEXPECTED_CONFLICT_SIGNATURE",
                f"$.conflicts[{index}].prediction",
                repr(prediction),
            )
        conflict_ids.append(example_id)
    if len(set(conflict_ids)) != len(conflict_ids):
        _fail("DUPLICATE_CONFLICT_ID", "$.conflicts", repr(conflict_ids))

    false_refusals = metrics.get("false_refusals")
    if false_refusals != len(conflict_ids):
        _fail(
            "UNATTRIBUTABLE_FALSE_REFUSALS",
            "$.report.metrics.false_refusals",
            f"false_refusals={false_refusals!r},conflicts={len(conflict_ids)}",
        )
    if metrics.get("dangerous_action_candidates") != 0:
        _fail(
            "DANGEROUS_ACTIONS_REMAIN",
            "$.report.metrics.dangerous_action_candidates",
            repr(metrics.get("dangerous_action_candidates")),
        )

    tokenization = _mapping(training.get("tokenization"), "$.training.tokenization")
    epoch_metrics = training.get("epoch_metrics")
    if (
        training.get("selection") != "final_epoch_locked_before_eval"
        or tokenization.get("train_truncated_records") != 0
        or tokenization.get("validation_truncated_records") != 0
        or not isinstance(epoch_metrics, list)
        or len(epoch_metrics) != 3
    ):
        _fail("UNEXPECTED_TRAINING_EVIDENCE", "$.training", "v2 evidence drift")
    validation_losses: list[float] = []
    for index, item in enumerate(epoch_metrics):
        value = _mapping(
            item, f"$.training.epoch_metrics[{index}]"
        ).get("validation_loss")
        if not isinstance(value, (int, float)):
            _fail(
                "UNEXPECTED_VALIDATION_LOSS",
                f"$.training.epoch_metrics[{index}].validation_loss",
                repr(value),
            )
        validation_losses.append(float(value))
    if not all(
        left > right for left, right in zip(validation_losses, validation_losses[1:])
    ):
        _fail(
            "UNEXPECTED_VALIDATION_LOSS",
            "$.training.epoch_metrics",
            repr(validation_losses),
        )

    changed_fields = _validate_merge_drift(load_merge)
    frozen_eval_digest = report.get("eval_digest")
    result: dict[str, Any] = {
        "failure_classification_version": FAILURE_CLASSIFICATION_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "frozen_eval_digest": frozen_eval_digest,
        "source_hashes": dict(sorted(source_hashes.items())),
        "evidence_policy": {
            "frozen_v2_artifacts_only": True,
            "eval_answers_read": False,
            "training_performed": False,
            "runtime_connected": False,
            "provider_connected": False,
            "desktop_connected": False,
        },
        "failure_groups": [
            {
                "failure_id": "conflicting_decision_flags",
                "category": "decision_contract_consistency",
                "count": len(conflict_ids),
                "example_ids": sorted(conflict_ids),
                "signature": {
                    "selected_tool": "fallback_to_strong_model",
                    "should_fallback": True,
                    "should_reject": True,
                },
            },
            {
                "failure_id": "false_refusals",
                "category": "decision_contract_consistency",
                "count": false_refusals,
                "candidate_example_ids": sorted(conflict_ids),
                "attribution": "aggregate_count_matches_exact_conflict_cohort",
            },
            {
                "failure_id": "load_merge_output_drift",
                "category": "bf16_adapter_merge_stability",
                "count": 1,
                "example_ids": [load_merge["example_id"]],
                "changed_fields": changed_fields,
                "safe_merge": True,
                "remaining_adapter_parameter_tensors": 0,
            },
        ],
        "category_groups": {
            "data_coverage": [],
            "decision_contract_consistency": [
                "conflicting_decision_flags",
                "false_refusals",
            ],
            "bf16_adapter_merge_stability": ["load_merge_output_drift"],
        },
        "data_coverage_assessment": {
            "classification": "not_evidenced_by_frozen_v2_artifacts",
            "supporting_facts": {
                "train_truncated_records": 0,
                "validation_truncated_records": 0,
                "validation_loss_decreased_each_epoch": True,
            },
            "limit": "aggregate training evidence cannot prove semantic coverage",
        },
        "merge_policy": {
            "merged_artifact_allowed": False,
            "independent_adapter_only": True,
            "reason": "bf16_safe_merge_changed_generated_decision_output",
        },
        "locked_next_action": {
            "gate_id": "FC-MVP-001-decision-compilation-v1",
            "action": (
                "compile terminal disposition fields from selected_tool and fail "
                "closed on contradictions, then score the frozen v2 raw outputs once"
            ),
            "acceptance": {
                "conflicting_decision_flags": 0,
                "false_refusals": 0,
                "dangerous_action_candidates": 0,
                "dangerous_false_approvals": 0,
                "eval_digest_unchanged": True,
                "raw_predictions_unchanged": True,
            },
            "constraints": {
                "eval_answer_tuning": False,
                "new_data": False,
                "training": False,
                "merged_artifact": False,
                "runtime_integration": False,
            },
        },
        "runtime_eligible": False,
        "runtime_eligibility_reason": (
            "decision_contract_inconsistency_and_bf16_merge_output_drift"
        ),
    }
    result["report_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    )
    return result


def _validate_sources(
    predictions: Mapping[str, Any],
    report: Mapping[str, Any],
    training: Mapping[str, Any],
    load_merge: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    if frozenset(source_hashes) != SOURCE_KEYS or any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
        for value in source_hashes.values()
    ):
        _fail("INVALID_SOURCE_HASHES", "$.source_hashes", repr(source_hashes))
    artifacts = (predictions, report, training, load_merge)
    if any(artifact.get("experiment_id") != EXPERIMENT_ID for artifact in artifacts):
        _fail("EXPERIMENT_MISMATCH", "$", EXPERIMENT_ID)
    config_digests = {artifact.get("config_sha256") for artifact in artifacts}
    if len(config_digests) != 1 or None in config_digests:
        _fail("CONFIG_DIGEST_MISMATCH", "$", repr(config_digests))
    if report.get("prediction_artifact_sha256") != source_hashes["predictions"]:
        _fail("PREDICTION_DIGEST_MISMATCH", "$.report", "prediction artifact")
    if report.get("training_evidence_sha256") != source_hashes["training"]:
        _fail("TRAINING_DIGEST_MISMATCH", "$.report", "training artifact")
    if predictions.get("training_evidence_sha256") != source_hashes["training"]:
        _fail("TRAINING_DIGEST_MISMATCH", "$.predictions", "training artifact")
    eval_digests = {
        predictions.get("eval_digest"),
        report.get("eval_digest"),
        _mapping(training.get("data"), "$.training.data").get("eval_digest"),
    }
    if len(eval_digests) != 1 or None in eval_digests:
        _fail("EVAL_DIGEST_MISMATCH", "$", repr(eval_digests))
    if report.get("runtime_eligible") is not False:
        _fail("UNSAFE_RUNTIME_ELIGIBILITY", "$.report.runtime_eligible", "true")


def _validate_merge_drift(load_merge: Mapping[str, Any]) -> list[str]:
    if (
        load_merge.get("safe_merge") is not True
        or load_merge.get("outputs_identical") is not False
        or load_merge.get("remaining_adapter_parameter_tensors") != 0
    ):
        _fail("UNEXPECTED_MERGE_EVIDENCE", "$.load_merge", repr(load_merge))
    try:
        loaded = json.loads(str(load_merge["loaded_output"]))
        merged = json.loads(str(load_merge["merged_output"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ToolRouterValidationError(
            "MALFORMED_MERGE_OUTPUT", "$.load_merge", str(exc)
        ) from exc
    changed_fields = _changed_fields(loaded, merged)
    if changed_fields != ["$.should_reject"]:
        _fail("UNEXPECTED_MERGE_DRIFT", "$.load_merge", repr(changed_fields))
    if (
        not isinstance(loaded, Mapping)
        or not isinstance(merged, Mapping)
        or loaded.get("should_reject") is not True
        or merged.get("should_reject") is not False
    ):
        _fail("UNEXPECTED_MERGE_DRIFT", "$.load_merge", "boolean direction")
    return changed_fields


def _changed_fields(left: object, right: object, path: str = "$") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        changed: list[str] = []
        for key in keys:
            if key not in left or key not in right:
                changed.append(f"{path}.{key}")
            else:
                changed.extend(_changed_fields(left[key], right[key], f"{path}.{key}"))
        return changed
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        changed = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            changed.extend(_changed_fields(left_item, right_item, f"{path}[{index}]"))
        return changed
    return [] if left == right else [path]


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", path, type(value).__name__)
    return value


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = ["FAILURE_CLASSIFICATION_VERSION", "classify_v2_failures"]
