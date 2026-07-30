"""Deterministic data rendering and evidence helpers for Tool Router SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .consumer import canonical_json_bytes

USER_FIELDS = ("instruction", "available_tools", "state")
DECISION_FIELDS = (
    "arguments",
    "expected_result",
    "requires_approval",
    "risk_level",
    "selected_tool",
    "should_fallback",
    "should_reject",
)


def render_user_payload(record: Mapping[str, Any]) -> str:
    """Render only inference-visible fields as canonical compact JSON."""

    payload = {key: record[key] for key in USER_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def render_target(record: Mapping[str, Any]) -> str:
    """Render the exact decision object as canonical compact JSON."""

    decision = record["decision"]
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be an object")
    if tuple(sorted(decision)) != DECISION_FIELDS:
        raise ValueError("decision fields do not match Tool Router decision v1")
    return json.dumps(
        decision,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Digest a locked config independently of whitespace."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def file_sha256(path: Path) -> str:
    """Digest a frozen binary artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def directory_artifact_manifest(directory: Path) -> list[dict[str, Any]]:
    """Describe regular files in a saved adapter directory."""

    if not directory.is_dir():
        raise ValueError(f"adapter directory does not exist: {directory}")
    artifacts: list[dict[str, Any]] = []
    paths = sorted(
        directory.rglob("*"),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )
    for path in paths:
        if path.is_file() and not path.is_symlink():
            artifacts.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    if not artifacts:
        raise ValueError(f"adapter directory is empty: {directory}")
    return artifacts


__all__ = [
    "canonical_config_sha256",
    "directory_artifact_manifest",
    "file_sha256",
    "render_target",
    "render_user_payload",
]
