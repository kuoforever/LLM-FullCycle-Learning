"""Fail-closed audit for the FC-MVP-001 safety-repair data gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn

from .consumer import canonical_json_bytes
from .tool_router import TOOL_ROUTER_SCHEMA_VERSION, ToolRouterValidationError
from .tool_router_dataset import audit_dataset

SAFETY_REPAIR_DATASET_VERSION = 2
BADCASE_TAXONOMY_VERSION = 1
MAX_TAXONOMY_BYTES = 65_536
REPAIR_TARGETS = frozenset(
    {
        "dangerous_action_candidate",
        "inconsistent_rejection",
        "inconsistent_clarification",
        "conflicting_decision_flags",
    }
)
TAXONOMY_KEYS = frozenset(
    {
        "taxonomy_version",
        "experiment_id",
        "source_eval_digest",
        "source_prediction_sha256",
        "source_report_sha256",
        "badcases",
        "overfitting",
        "repair_policy",
    }
)
BADCASE_KEYS = frozenset(
    {"example_id", "failure_codes", "classification", "repair_target"}
)
OVERFITTING_KEYS = frozenset(
    {
        "lowest_validation_loss_epoch",
        "lowest_validation_loss",
        "final_validation_loss",
        "classification",
    }
)
POLICY_KEYS = frozenset(
    {
        "train_validation_only",
        "eval_answers_included",
        "frozen_eval_must_match",
        "training_performed",
    }
)


def load_badcase_taxonomy(path: Path) -> dict[str, Any]:
    """Load the strict bounded taxonomy that owns repair provenance."""

    if not path.is_file() or path.is_symlink():
        _fail("UNSAFE_BADCASE_TAXONOMY", "$", str(path))
    size = path.stat().st_size
    if size > MAX_TAXONOMY_BYTES:
        _fail("BADCASE_TAXONOMY_TOO_LARGE", "$", f"{size}>{MAX_TAXONOMY_BYTES}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolRouterValidationError(
            "MALFORMED_BADCASE_TAXONOMY", "$", str(exc)
        ) from exc
    taxonomy = _exact_object(value, "$", TAXONOMY_KEYS)
    if taxonomy["taxonomy_version"] != BADCASE_TAXONOMY_VERSION:
        _fail(
            "UNSUPPORTED_BADCASE_TAXONOMY",
            "$.taxonomy_version",
            repr(taxonomy["taxonomy_version"]),
        )
    for key in ("source_eval_digest", "source_prediction_sha256", "source_report_sha256"):
        digest = taxonomy[key]
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            _fail("INVALID_BADCASE_SOURCE_DIGEST", f"$.{key}", repr(digest))
    badcases = taxonomy["badcases"]
    if not isinstance(badcases, list) or len(badcases) != 4:
        _fail("INVALID_BADCASE_COUNT", "$.badcases", repr(badcases))
    identifiers: set[str] = set()
    targets: set[str] = set()
    for index, value in enumerate(badcases):
        path_prefix = f"$.badcases[{index}]"
        badcase = _exact_object(value, path_prefix, BADCASE_KEYS)
        identifier = badcase["example_id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"eval-[0-9]{3}", identifier):
            _fail("INVALID_BADCASE_ID", f"{path_prefix}.example_id", repr(identifier))
        if identifier in identifiers:
            _fail("DUPLICATE_BADCASE_ID", f"{path_prefix}.example_id", identifier)
        identifiers.add(identifier)
        target = badcase["repair_target"]
        if target not in REPAIR_TARGETS:
            _fail("INVALID_REPAIR_TARGET", f"{path_prefix}.repair_target", repr(target))
        targets.add(target)
        failure_codes = badcase["failure_codes"]
        if not isinstance(failure_codes, list) or not failure_codes or any(
            not isinstance(code, str) or not code for code in failure_codes
        ):
            _fail("INVALID_FAILURE_CODES", f"{path_prefix}.failure_codes", repr(failure_codes))
        if not isinstance(badcase["classification"], str) or not badcase["classification"]:
            _fail("INVALID_BADCASE_CLASSIFICATION", f"{path_prefix}.classification", repr(badcase["classification"]))
    if targets != REPAIR_TARGETS:
        _fail("INCOMPLETE_REPAIR_TARGETS", "$.badcases", repr(sorted(targets)))
    _validate_overfitting(taxonomy["overfitting"])
    policy = _exact_object(taxonomy["repair_policy"], "$.repair_policy", POLICY_KEYS)
    expected_policy = {
        "train_validation_only": True,
        "eval_answers_included": False,
        "frozen_eval_must_match": True,
        "training_performed": False,
    }
    if policy != expected_policy:
        _fail("UNSAFE_REPAIR_POLICY", "$.repair_policy", repr(policy))
    return taxonomy


def audit_safety_repair_dataset(
    base_train: Iterable[Mapping[str, Any]],
    base_validation: Iterable[Mapping[str, Any]],
    repaired_train: Iterable[Mapping[str, Any]],
    repaired_validation: Iterable[Mapping[str, Any]],
    evaluation: Iterable[Mapping[str, Any]],
    base_manifest: Mapping[str, Any],
    repaired_manifest: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    eval_digest: str,
) -> dict[str, Any]:
    """Prove the v2 increment is isolated, safe, and evaluation-frozen."""

    base_train_records = list(base_train)
    base_validation_records = list(base_validation)
    train_records = list(repaired_train)
    validation_records = list(repaired_validation)
    evaluation_records = list(evaluation)
    if taxonomy["source_eval_digest"] != eval_digest:
        _fail(
            "BADCASE_EVAL_DIGEST_MISMATCH",
            "$.source_eval_digest",
            str(taxonomy["source_eval_digest"]),
        )
    if train_records[: len(base_train_records)] != base_train_records:
        _fail("BASE_TRAIN_DRIFT", "$.train", "v1 is not the exact v2 prefix")
    if validation_records[: len(base_validation_records)] != base_validation_records:
        _fail(
            "BASE_VALIDATION_DRIFT",
            "$.validation",
            "v1 is not the exact v2 prefix",
        )
    base_families = list(base_manifest["families"])
    repaired_families = list(repaired_manifest["families"])
    if repaired_families[: len(base_families)] != base_families:
        _fail("BASE_FAMILY_DRIFT", "$.families", "v1 is not the exact v2 prefix")

    train_increment = train_records[len(base_train_records) :]
    validation_increment = validation_records[len(base_validation_records) :]
    family_increment = repaired_families[len(base_families) :]
    if len(train_increment) != 16 or len(validation_increment) != 8:
        _fail(
            "INVALID_REPAIR_INCREMENT_SIZE",
            "$",
            f"{len(train_increment)}+{len(validation_increment)}",
        )
    if len(family_increment) != 8:
        _fail("INVALID_REPAIR_FAMILY_COUNT", "$.families", str(len(family_increment)))
    family_targets = Counter(
        str(family["description"]).partition(" | ")[0] for family in family_increment
    )
    if set(family_targets) != REPAIR_TARGETS or set(family_targets.values()) != {2}:
        _fail("INVALID_REPAIR_FAMILY_TARGETS", "$.families", repr(family_targets))

    eval_instructions = {_normalize(record["instruction"]) for record in evaluation_records}
    eval_ids = {record["example_id"] for record in evaluation_records}
    for record in train_increment + validation_increment:
        if record["example_id"] in eval_ids or _normalize(record["instruction"]) in eval_instructions:
            _fail("EVAL_ANSWER_COPY", "$", record["example_id"])
        if "evaluation case" in record["instruction"].lower():
            _fail("EVAL_ANSWER_COPY", "$", record["example_id"])

    combined = audit_dataset(
        train_records,
        validation_records,
        evaluation_records,
        repaired_manifest,
        eval_digest,
    )
    dangerous_action_candidates = sum(
        record["category"] == "dangerous_request"
        and (
            record["decision"]["selected_tool"] != "reject_request"
            or not record["decision"]["should_reject"]
        )
        for record in train_records + validation_records
    )
    if dangerous_action_candidates:
        _fail("DANGEROUS_ACTION_CANDIDATE", "$", str(dangerous_action_candidates))

    report = {
        "safety_repair_dataset_version": SAFETY_REPAIR_DATASET_VERSION,
        "tool_router_schema_version": TOOL_ROUTER_SCHEMA_VERSION,
        "base_train_records": len(base_train_records),
        "base_validation_records": len(base_validation_records),
        "increment_train_records": len(train_increment),
        "increment_validation_records": len(validation_increment),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "eval_records": len(evaluation_records),
        "base_task_families": len(base_families),
        "increment_task_families": len(family_increment),
        "task_families": len(repaired_families),
        "base_prefix_preserved": True,
        "eval_answers_included": False,
        "frozen_eval_digest": eval_digest,
        "repair_target_family_counts": dict(sorted(family_targets.items())),
        "dangerous_action_candidates": dangerous_action_candidates,
        "dangerous_false_approvals": combined["dangerous_false_approvals"],
        "max_cross_split_instruction_jaccard": combined[
            "max_cross_split_instruction_jaccard"
        ],
        "jaccard_failure_threshold": combined["jaccard_failure_threshold"],
        "combined_dataset_report_digest": combined["report_digest"],
    }
    report["report_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    )
    return report


def _validate_overfitting(value: object) -> None:
    overfitting = _exact_object(value, "$.overfitting", OVERFITTING_KEYS)
    if overfitting["lowest_validation_loss_epoch"] != 3:
        _fail(
            "INVALID_OVERFITTING_EPOCH",
            "$.overfitting.lowest_validation_loss_epoch",
            repr(overfitting["lowest_validation_loss_epoch"]),
        )
    lowest = overfitting["lowest_validation_loss"]
    final = overfitting["final_validation_loss"]
    if not isinstance(lowest, (int, float)) or not isinstance(final, (int, float)) or final <= lowest:
        _fail("INVALID_OVERFITTING_EVIDENCE", "$.overfitting", repr(overfitting))
    if overfitting["classification"] != "validation_overfitting_after_epoch_3":
        _fail(
            "INVALID_OVERFITTING_CLASSIFICATION",
            "$.overfitting.classification",
            repr(overfitting["classification"]),
        )


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _exact_object(value: object, path: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("EXPECTED_REPAIR_OBJECT", path, type(value).__name__)
    actual = frozenset(value)
    if actual != keys:
        _fail(
            "INVALID_REPAIR_KEYS",
            path,
            f"missing={sorted(keys - actual)},extra={sorted(actual - keys)}",
        )
    return value


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "BADCASE_TAXONOMY_VERSION",
    "REPAIR_TARGETS",
    "SAFETY_REPAIR_DATASET_VERSION",
    "audit_safety_repair_dataset",
    "load_badcase_taxonomy",
]
