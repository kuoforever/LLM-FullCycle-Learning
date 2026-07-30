"""Dataset-family and leakage audits for Tool Router v1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn

from .consumer import canonical_json_bytes
from .tool_router import (
    CATEGORIES,
    TOOL_ROUTER_SCHEMA_VERSION,
    ToolRouterValidationError,
)

DATASET_MANIFEST_VERSION = 1
MIN_TRAIN_VALIDATION_RECORDS = 200
MAX_TRAIN_VALIDATION_RECORDS = 500
MAX_MANIFEST_BYTES = 262_144
MAX_CROSS_SPLIT_JACCARD = 0.8
MANIFEST_KEYS = frozenset(
    {
        "dataset_manifest_version",
        "tool_router_schema_version",
        "frozen_eval_digest",
        "families",
    }
)
FAMILY_KEYS = frozenset(
    {"family_id", "split", "category", "description", "example_ids"}
)
FAMILY_ID = re.compile(r"^family-[0-9]{3}$")


def load_family_manifest(path: Path) -> dict[str, Any]:
    """Load a strict bounded task-family manifest."""

    if not path.is_file() or path.is_symlink():
        _fail("UNSAFE_FAMILY_MANIFEST", "$", str(path))
    size = path.stat().st_size
    if size > MAX_MANIFEST_BYTES:
        _fail("FAMILY_MANIFEST_TOO_LARGE", "$", f"{size}>{MAX_MANIFEST_BYTES}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolRouterValidationError(
            "MALFORMED_FAMILY_MANIFEST", "$", str(exc)
        ) from exc
    manifest = _exact_object(value, "$", MANIFEST_KEYS)
    _exact_version(manifest["dataset_manifest_version"], DATASET_MANIFEST_VERSION, "$")
    _exact_version(
        manifest["tool_router_schema_version"], TOOL_ROUTER_SCHEMA_VERSION, "$"
    )
    digest = manifest["frozen_eval_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        _fail("INVALID_EVAL_DIGEST", "$.frozen_eval_digest", repr(digest))
    families = manifest["families"]
    if not isinstance(families, list) or not families:
        _fail("INVALID_FAMILIES", "$.families", "expected non-empty array")
    identifiers: set[str] = set()
    example_ids: set[str] = set()
    for index, value in enumerate(families):
        path_prefix = f"$.families[{index}]"
        family = _exact_object(value, path_prefix, FAMILY_KEYS)
        family_id = family["family_id"]
        if not isinstance(family_id, str) or not FAMILY_ID.fullmatch(family_id):
            _fail("INVALID_FAMILY_ID", f"{path_prefix}.family_id", repr(family_id))
        if family_id in identifiers:
            _fail("DUPLICATE_FAMILY_ID", f"{path_prefix}.family_id", family_id)
        identifiers.add(family_id)
        split = family["split"]
        if split not in {"train", "validation"}:
            _fail("INVALID_FAMILY_SPLIT", f"{path_prefix}.split", repr(split))
        if family["category"] not in CATEGORIES:
            _fail(
                "INVALID_FAMILY_CATEGORY",
                f"{path_prefix}.category",
                repr(family["category"]),
            )
        if not isinstance(family["description"], str) or not family["description"]:
            _fail(
                "INVALID_FAMILY_DESCRIPTION",
                f"{path_prefix}.description",
                repr(family["description"]),
            )
        ids = family["example_ids"]
        if not isinstance(ids, list) or not ids:
            _fail("INVALID_FAMILY_EXAMPLES", f"{path_prefix}.example_ids", repr(ids))
        for example_id in ids:
            if not isinstance(example_id, str) or not example_id.startswith(
                f"{split}-"
            ):
                _fail(
                    "INVALID_FAMILY_EXAMPLE_ID",
                    f"{path_prefix}.example_ids",
                    repr(example_id),
                )
            if example_id in example_ids:
                _fail("DUPLICATE_FAMILY_EXAMPLE", path_prefix, example_id)
            example_ids.add(example_id)
    return manifest


def audit_dataset(
    train: Iterable[Mapping[str, Any]],
    validation: Iterable[Mapping[str, Any]],
    evaluation: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    eval_digest: str,
) -> dict[str, Any]:
    """Fail closed on split leakage and return a deterministic distribution."""

    split_records = {
        "train": list(train),
        "validation": list(validation),
        "eval": list(evaluation),
    }
    for split, records in split_records.items():
        if not records or any(record["split"] != split for record in records):
            _fail("WRONG_DATASET_SPLIT", f"$.{split}", "record split mismatch")
    train_validation_count = len(split_records["train"]) + len(
        split_records["validation"]
    )
    if (
        not MIN_TRAIN_VALIDATION_RECORDS
        <= train_validation_count
        <= MAX_TRAIN_VALIDATION_RECORDS
    ):
        _fail(
            "INVALID_DATASET_SIZE",
            "$",
            f"{train_validation_count} not in "
            f"[{MIN_TRAIN_VALIDATION_RECORDS},{MAX_TRAIN_VALIDATION_RECORDS}]",
        )
    if manifest["frozen_eval_digest"] != eval_digest:
        _fail(
            "FROZEN_EVAL_DIGEST_MISMATCH",
            "$.frozen_eval_digest",
            manifest["frozen_eval_digest"],
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    for split, records in split_records.items():
        for record in records:
            identifier = record["example_id"]
            if identifier in by_id:
                _fail("CROSS_SPLIT_EXAMPLE_ID", f"$.{split}", identifier)
            by_id[identifier] = record

    family_by_example: dict[str, Mapping[str, Any]] = {}
    split_families: dict[str, set[str]] = {"train": set(), "validation": set()}
    for family in manifest["families"]:
        split = family["split"]
        split_families[split].add(family["family_id"])
        for example_id in family["example_ids"]:
            matched_record = by_id.get(example_id)
            if matched_record is None:
                _fail("MISSING_FAMILY_RECORD", "$.families", example_id)
            if (
                matched_record["split"] != split
                or matched_record["category"] != family["category"]
            ):
                _fail("FAMILY_RECORD_MISMATCH", "$.families", example_id)
            family_by_example[example_id] = family
    expected_ids = {
        record["example_id"]
        for split in ("train", "validation")
        for record in split_records[split]
    }
    if set(family_by_example) != expected_ids:
        missing = sorted(expected_ids - set(family_by_example))
        _fail("UNMAPPED_DATASET_RECORD", "$.families", missing[0])
    overlap = split_families["train"] & split_families["validation"]
    if overlap:
        _fail("TASK_FAMILY_LEAKAGE", "$.families", sorted(overlap)[0])

    normalized: dict[str, str] = {}
    for split, records in split_records.items():
        for record in records:
            instruction = _normalize(record["instruction"])
            previous = normalized.get(instruction)
            if previous is not None:
                _fail(
                    "EXACT_INSTRUCTION_DUPLICATE",
                    f"$.{split}",
                    f"{record['example_id']} duplicates {previous}",
                )
            normalized[instruction] = record["example_id"]

    max_similarity = 0.0
    cross_pairs = (
        ("train", "validation"),
        ("train", "eval"),
        ("validation", "eval"),
    )
    for left_split, right_split in cross_pairs:
        for left in split_records[left_split]:
            left_tokens = _tokens(left["instruction"])
            for right in split_records[right_split]:
                similarity = _jaccard(left_tokens, _tokens(right["instruction"]))
                max_similarity = max(max_similarity, similarity)
                if similarity > MAX_CROSS_SPLIT_JACCARD:
                    _fail(
                        "NEAR_DUPLICATE_LEAKAGE",
                        "$",
                        f"{left['example_id']}~{right['example_id']}={similarity}",
                    )

    all_records = [record for records in split_records.values() for record in records]
    dangerous_false_approvals = sum(
        record["category"] == "dangerous_request"
        and record["decision"]["requires_approval"]
        for record in all_records
    )
    if dangerous_false_approvals:
        _fail(
            "DANGEROUS_FALSE_APPROVAL",
            "$",
            str(dangerous_false_approvals),
        )

    report = {
        "dataset_manifest_version": DATASET_MANIFEST_VERSION,
        "tool_router_schema_version": TOOL_ROUTER_SCHEMA_VERSION,
        "train_records": len(split_records["train"]),
        "validation_records": len(split_records["validation"]),
        "eval_records": len(split_records["eval"]),
        "train_validation_records": train_validation_count,
        "task_families": len(manifest["families"]),
        "task_family_overlap": 0,
        "exact_instruction_duplicates": 0,
        "max_cross_split_instruction_jaccard": max_similarity,
        "jaccard_failure_threshold": MAX_CROSS_SPLIT_JACCARD,
        "dangerous_false_approvals": dangerous_false_approvals,
        "split_category_counts": {
            split: _count(records, "category")
            for split, records in split_records.items()
        },
        "risk_counts": _count(
            (record["decision"] for record in all_records), "risk_level"
        ),
        "selected_tool_counts": _count(
            (record["decision"] for record in all_records), "selected_tool"
        ),
        "expected_result_counts": _count(
            (record["decision"] for record in all_records), "expected_result"
        ),
    }
    report["report_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    )
    return report


def _count(values: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(value[key] for value in values).items()))


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _exact_object(value: object, path: str, expected: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("EXPECTED_MANIFEST_OBJECT", path, type(value).__name__)
    actual = frozenset(value)
    if actual != expected:
        _fail(
            "INVALID_MANIFEST_KEYS",
            path,
            f"missing={sorted(expected - actual)},extra={sorted(actual - expected)}",
        )
    return value


def _exact_version(value: object, expected: int, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        _fail("UNSUPPORTED_MANIFEST_VERSION", path, repr(value))


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "DATASET_MANIFEST_VERSION",
    "MAX_CROSS_SPLIT_JACCARD",
    "audit_dataset",
    "load_family_manifest",
]
