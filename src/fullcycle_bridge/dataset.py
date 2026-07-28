"""Deterministic Lane A mapping for reliability and Verifier datasets."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .consumer import (
    CONSUMER_SCHEMA_VERSION,
    RUNTIME_GIT_COMMIT,
    BridgeValidationError,
    canonical_json_bytes,
    load_validated_files,
)

RELIABILITY_DATASET_SCHEMA_VERSION = 1
EVENT_KINDS = (
    "user_task",
    "model_turn",
    "tool_call",
    "tool_result",
    "observation",
    "policy_decision",
    "recovery",
)
TAG_ORDER = (
    "failure",
    "unknown_outcome",
    "policy_denial",
    "recovery_required",
    "budget_limit",
    "tool_sequence_available",
)


def map_files(manifest_path: Path, run_export_path: Path) -> dict[str, Any]:
    """Validate one bridge pair and map it to one deterministic dataset record."""

    manifest, run_export = load_validated_files(manifest_path, run_export_path)
    return map_validated_run(manifest, run_export)


def map_many(
    manifest_path: Path, run_export_paths: Iterable[Path]
) -> list[dict[str, Any]]:
    """Map ordered run exports and reject duplicate source run identifiers."""

    records: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for path in run_export_paths:
        record = map_files(manifest_path, path)
        run_id = record["source"]["run_id"]
        if run_id in run_ids:
            raise BridgeValidationError(
                "DUPLICATE_DATASET_RUN", "dataset.source.run_id", run_id
            )
        run_ids.add(run_id)
        records.append(record)
    if not records:
        raise BridgeValidationError(
            "EMPTY_DATASET_INPUT", "dataset", "at least one run export is required"
        )
    return records


def map_validated_run(
    manifest: Mapping[str, Any], run_export: Mapping[str, Any]
) -> dict[str, Any]:
    """Map already validated v1 objects without adding semantic information."""

    checkpoint = run_export["checkpoint"]
    events = run_export["events"]
    run_id = run_export["run_id"]
    run_export_digest = "sha256:" + hashlib.sha256(
        canonical_json_bytes(run_export)
    ).hexdigest()
    source_identity = {
        "reliability_dataset_schema_version": RELIABILITY_DATASET_SCHEMA_VERSION,
        "manifest_digest": run_export["manifest_digest"],
        "run_export_digest": run_export_digest,
    }
    record_id = "sha256:" + hashlib.sha256(
        canonical_json_bytes(source_identity)
    ).hexdigest()

    counts = Counter(event["kind"] for event in events)
    event_kind_counts = {kind: counts.get(kind, 0) for kind in EVENT_KINDS}
    tool_sequence = [
        {"sequence": event["sequence"], "tool": event["tool"]}
        for event in events
        if event["kind"] == "tool_call"
    ]
    tool_outcomes = [
        {
            "sequence": event["sequence"],
            "tool": event["tool"],
            "status": event["status"],
            "dispatch": event["dispatch"],
            "code": event.get("code"),
        }
        for event in events
        if event["kind"] == "tool_result"
    ]
    policy_decisions = [
        {"sequence": event["sequence"], "decision": event["decision"]}
        for event in events
        if event["kind"] == "policy_decision"
    ]
    recovery_events = [
        {"sequence": event["sequence"], "status": event["status"]}
        for event in events
        if event["kind"] == "recovery"
    ]

    budget_features = _budget_features(checkpoint["budgets"])
    phase = checkpoint["phase"]
    outcome_class = {
        "SUCCESS": "success",
        "FAILED": "failure",
        "UNKNOWN_OUTCOME": "unknown_outcome",
        "CANCELLED": "cancelled",
        "PAUSED": "paused",
    }.get(phase, "in_progress")
    is_failure = phase in {"FAILED", "UNKNOWN_OUTCOME"}
    is_unknown_outcome = phase == "UNKNOWN_OUTCOME" or any(
        outcome["status"] == "unknown_outcome" or outcome["dispatch"] == "unknown"
        for outcome in tool_outcomes
    )
    policy_denied = any(
        decision["decision"] == "deny" for decision in policy_decisions
    ) or any(outcome["code"] == "POLICY_DENIED" for outcome in tool_outcomes)
    recovery_required = (
        checkpoint["recovery_status"] != "ready"
        or checkpoint["recovery_action"]
        in {
            "human_reobserve_then_start_new_run",
            "inspect_trace_then_start_new_run",
        }
        or any(event["status"] != "ready" for event in recovery_events)
    )
    budget_limit_hit = any(budget_features["limit_hit"].values())
    tag_values = {
        "failure": is_failure,
        "unknown_outcome": is_unknown_outcome,
        "policy_denial": policy_denied,
        "recovery_required": recovery_required,
        "budget_limit": budget_limit_hit,
        "tool_sequence_available": bool(tool_sequence),
    }

    return {
        "reliability_dataset_schema_version": RELIABILITY_DATASET_SCHEMA_VERSION,
        "record_id": record_id,
        "source": {
            "runtime_git_commit": RUNTIME_GIT_COMMIT,
            "consumer_schema_version": CONSUMER_SCHEMA_VERSION,
            "fullcycle_manifest_version": manifest["fullcycle_manifest_version"],
            "fullcycle_run_export_version": run_export[
                "fullcycle_run_export_version"
            ],
            "manifest_digest": run_export["manifest_digest"],
            "run_export_digest": run_export_digest,
            "run_id": run_id,
            "trace_version": manifest["trace_version"],
            "checkpoint_version": checkpoint["checkpoint_version"],
            "policy_version": checkpoint["policy_version"],
        },
        "data_class": run_export["data_class"],
        "training_use": run_export["training_use"],
        "features": {
            "phase": phase,
            "recovery_status": checkpoint["recovery_status"],
            "recovery_action": checkpoint["recovery_action"],
            "failure_code": checkpoint.get("failure_code"),
            "event_count": checkpoint["event_count"],
            "event_kind_counts": event_kind_counts,
            "tool_sequence": tool_sequence,
            "tool_outcomes": tool_outcomes,
            "policy_decisions": policy_decisions,
            "recovery_events": recovery_events,
            "budgets": budget_features,
        },
        "labels": {
            "outcome_class": outcome_class,
            "is_failure": is_failure,
            "is_unknown_outcome": is_unknown_outcome,
            "policy_denied": policy_denied,
            "recovery_required": recovery_required,
            "budget_limit_hit": budget_limit_hit,
            "verifier_tags": [tag for tag in TAG_ORDER if tag_values[tag]],
        },
    }


def _budget_features(budgets: Mapping[str, int]) -> dict[str, Any]:
    pairs = (
        ("model_turns", "max_model_turns", "model_turns_used"),
        ("tool_calls", "max_tool_calls", "tool_calls_used"),
        ("side_effects", "max_side_effects", "side_effects_used"),
        ("input_tokens", "max_input_tokens", "input_tokens_used"),
    )
    limits = {name: budgets[maximum] for name, maximum, _ in pairs}
    used = {name: budgets[consumed] for name, _, consumed in pairs}
    limit_hit = {
        name: limits[name] > 0 and used[name] >= limits[name]
        for name, _, _ in pairs
    }
    return {"limits": limits, "used": used, "limit_hit": limit_hit}


__all__ = [
    "RELIABILITY_DATASET_SCHEMA_VERSION",
    "map_files",
    "map_many",
    "map_validated_run",
]
