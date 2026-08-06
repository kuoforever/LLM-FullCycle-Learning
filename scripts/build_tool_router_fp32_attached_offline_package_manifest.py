"""Build or check the FP32 attached metadata-only offline package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router_fp32_attached_offline_package_manifest import (  # noqa: E402
    ADAPTER_FILE_SPECS,
    BASE_MODEL_FILE_SPECS,
    REPOSITORY_SOURCE_PATHS,
    TOKENIZER_FILE_SPECS,
    build_fp32_attached_offline_package_manifest,
)

OUTPUT = ROOT / "baseline" / "fc-mvp-001-fp32-attached-offline-package-manifest-v1.json"
DEFAULT_BASE_MODEL_DIR = ROOT / "work" / "models" / "Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"

_StatSignature = tuple[int, int, int, int, int, int]
_TreeSnapshot = tuple[tuple[str, str], ...]
_TreeReceipt = tuple[_StatSignature, _TreeSnapshot]


def _source_paths(adapter_dir: Path) -> dict[str, Path]:
    paths = {
        name: ROOT / relative for name, relative in REPOSITORY_SOURCE_PATHS.items()
    }
    paths.update(
        {
            "adapter_config": adapter_dir / "adapter_config.json",
            "adapter_readme": adapter_dir / "README.md",
            "adapter_weights": adapter_dir / "adapter_model.safetensors",
        }
    )
    return paths


def repository_source_paths() -> dict[str, str]:
    """Return the repository-relative small-source roots used by resolution."""

    return dict(REPOSITORY_SOURCE_PATHS)


def load_repository_manifest_inputs(
    *, adapter_dir: Path = DEFAULT_ADAPTER_DIR
) -> dict[str, Any]:
    """Read each small source and the Adapter once from fixed caller roots."""

    inputs, _file_receipts, _directory_receipts, _adapter_tree_receipt = (
        _load_repository_manifest_inputs_with_receipts(adapter_dir=adapter_dir)
    )
    return inputs


def _load_repository_manifest_inputs_with_receipts(
    *, adapter_dir: Path
) -> tuple[
    dict[str, Any],
    dict[Path, _StatSignature],
    dict[Path, _StatSignature],
    _TreeReceipt,
]:
    """Read source bytes and retain receipts for a later whole-build check."""

    adapter_tree_receipt = _require_safe_tree(
        adapter_dir,
        {item[0] for item in ADAPTER_FILE_SPECS},
        allowed_directories=frozenset(),
        label="adapter",
    )
    paths = _source_paths(adapter_dir)
    directory_receipts: dict[Path, _StatSignature] = {}
    for name, path in paths.items():
        if name in REPOSITORY_SOURCE_PATHS:
            _require_repository_source_path(
                REPOSITORY_SOURCE_PATHS[name],
                name,
                directory_receipts,
            )
        else:
            _require_regular_file(path, name)
    source_payloads: dict[str, bytes] = {}
    file_receipts: dict[Path, _StatSignature] = {}
    for name, path in sorted(paths.items()):
        payload, receipt = _read_regular_file_receipt(path, name)
        source_payloads[name] = payload
        file_receipts[path] = receipt
    _require_source_state_unchanged(
        adapter_dir,
        adapter_tree_receipt,
        file_receipts,
        directory_receipts,
    )
    source_hashes = {
        name: "sha256:" + hashlib.sha256(payload).hexdigest()
        for name, payload in source_payloads.items()
    }
    inputs = {
        "upstream_review": _load_json_payload(
            source_payloads["upstream_review"], paths["upstream_review"]
        ),
        "remediation_preregistration": _load_json_payload(
            source_payloads["remediation_preregistration"],
            paths["remediation_preregistration"],
        ),
        "sft_config": _load_json_payload(
            source_payloads["sft_config"], paths["sft_config"]
        ),
        "adapter_config": _load_json_payload(
            source_payloads["adapter_config"], paths["adapter_config"]
        ),
        "adapter_files": _records_from_payloads(
            ADAPTER_FILE_SPECS,
            {
                "README.md": source_payloads["adapter_readme"],
                "adapter_config.json": source_payloads["adapter_config"],
                "adapter_model.safetensors": source_payloads["adapter_weights"],
            },
        ),
        "source_hashes": source_hashes,
        "source_payloads": source_payloads,
    }
    _require_source_state_unchanged(
        adapter_dir,
        adapter_tree_receipt,
        file_receipts,
        directory_receipts,
    )
    return inputs, file_receipts, directory_receipts, adapter_tree_receipt


def build(
    *,
    base_model_dir: Path = DEFAULT_BASE_MODEL_DIR,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
) -> dict[str, Any]:
    """Observe local components and build the deterministic composite manifest."""

    base_tree_receipt = _require_safe_tree(
        base_model_dir,
        {item[0] for item in (*BASE_MODEL_FILE_SPECS, *TOKENIZER_FILE_SPECS)},
        allowed_directories=frozenset({".cache"}),
        label="base_model_and_tokenizer",
    )
    inputs, source_receipts, directory_receipts, adapter_tree_receipt = (
        _load_repository_manifest_inputs_with_receipts(adapter_dir=adapter_dir)
    )
    base_model_files, base_receipts = _stream_file_records(
        base_model_dir, BASE_MODEL_FILE_SPECS
    )
    tokenizer_files, tokenizer_receipts = _stream_file_records(
        base_model_dir, TOKENIZER_FILE_SPECS
    )
    component_receipts = {**base_receipts, **tokenizer_receipts}
    _require_build_state_unchanged(
        base_model_dir,
        base_tree_receipt,
        component_receipts,
        adapter_dir,
        adapter_tree_receipt,
        source_receipts,
        directory_receipts,
    )
    manifest = build_fp32_attached_offline_package_manifest(
        inputs["upstream_review"],
        inputs["remediation_preregistration"],
        inputs["sft_config"],
        inputs["adapter_config"],
        base_model_files=base_model_files,
        tokenizer_files=tokenizer_files,
        adapter_files=inputs["adapter_files"],
        source_hashes=inputs["source_hashes"],
        source_payloads=inputs["source_payloads"],
    )
    _require_build_state_unchanged(
        base_model_dir,
        base_tree_receipt,
        component_receipts,
        adapter_dir,
        adapter_tree_receipt,
        source_receipts,
        directory_receipts,
    )
    return manifest


def _stream_file_records(
    root: Path, specs: tuple[tuple[str, str, int, str], ...]
) -> tuple[list[dict[str, Any]], dict[Path, _StatSignature]]:
    records: list[dict[str, Any]] = []
    receipts: dict[Path, _StatSignature] = {}
    for relative, role, _expected_bytes, _expected_sha256 in specs:
        path = root / relative
        _require_regular_file(path, relative)
        observed_bytes, observed_sha256, receipt = _hash_regular_file_receipt(path)
        receipts[path] = receipt
        records.append(
            {
                "path": relative,
                "role": role,
                "bytes": observed_bytes,
                "sha256": observed_sha256,
            }
        )
    return records, receipts


def _records_from_payloads(
    specs: tuple[tuple[str, str, int, str], ...], payloads: dict[str, bytes]
) -> list[dict[str, Any]]:
    records = []
    for relative, role, _expected_bytes, _expected_sha256 in specs:
        payload = payloads[relative]
        records.append(
            {
                "path": relative,
                "role": role,
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    return records


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or _is_reparse(path):
        raise RuntimeError(f"unsafe or missing {label}: {path}")


def _require_repository_source_path(
    relative: str,
    label: str,
    directory_receipts: dict[Path, _StatSignature],
) -> None:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or str(pure) != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"invalid repository source path {label}: {relative}")
    current = ROOT
    if not current.is_dir() or _is_reparse(current):
        raise RuntimeError(f"unsafe repository root: {current}")
    directory_receipts.setdefault(current, _stat_signature(current.lstat()))
    for part in pure.parts[:-1]:
        current = current / part
        if not current.is_dir() or _is_reparse(current):
            raise RuntimeError(f"unsafe repository source parent {label}: {current}")
        directory_receipts.setdefault(current, _stat_signature(current.lstat()))
    _require_regular_file(current / pure.parts[-1], label)


def _require_safe_tree(
    root: Path,
    expected_files: set[str],
    *,
    allowed_directories: frozenset[str],
    label: str,
) -> _TreeReceipt:
    if not root.is_dir() or _is_reparse(root):
        raise RuntimeError(f"unsafe or missing {label} root: {root}")
    root_before = _stat_signature(root.lstat())
    snapshot_before = _directory_entry_snapshot(root)
    actual_files: set[str] = set()
    for name, kind in snapshot_before:
        if kind == "case_collision":
            raise RuntimeError(f"case-colliding {label} entry: {name}")
        if kind == "reparse":
            raise RuntimeError(f"unsafe reparse {label} entry: {root / name}")
        if kind == "file":
            actual_files.add(name)
        elif kind == "directory" and name in allowed_directories:
            continue
        else:
            raise RuntimeError(f"unexpected {label} entry: {root / name}")
    if actual_files != expected_files:
        raise RuntimeError(
            f"{label} file set mismatch: expected={sorted(expected_files)!r},"
            f"actual={sorted(actual_files)!r}"
        )
    root_after = _stat_signature(root.lstat())
    snapshot_after = _directory_entry_snapshot(root)
    if root_before != root_after or snapshot_before != snapshot_after:
        raise RuntimeError(f"{label} tree changed while inspecting: {root}")
    return root_after, snapshot_after


def _directory_entry_snapshot(root: Path) -> _TreeSnapshot:
    entries: list[tuple[str, str]] = []
    seen_casefold: set[str] = set()
    for entry in root.iterdir():
        folded = entry.name.casefold()
        if folded in seen_casefold:
            kind = "case_collision"
        else:
            seen_casefold.add(folded)
            if _is_reparse(entry):
                kind = "reparse"
            elif entry.is_file():
                kind = "file"
            elif entry.is_dir():
                kind = "directory"
            else:
                kind = "other"
        entries.append((entry.name, kind))
    return tuple(sorted(entries, key=lambda item: (item[0].casefold(), item[0])))


def _read_regular_file_receipt(path: Path, label: str) -> tuple[bytes, _StatSignature]:
    """Read one file once while binding the open handle to its path identity."""

    _require_regular_file(path, label)
    before = path.lstat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _handle_identity_signature(before) != _handle_identity_signature(opened):
            raise RuntimeError(f"{label} identity changed before reading: {path}")
        payload = handle.read()
        handle_after = os.fstat(handle.fileno())
    after = path.lstat()
    if (
        _handle_identity_signature(before) != _handle_identity_signature(handle_after)
        or _handle_identity_signature(handle_after) != _handle_identity_signature(after)
        or _stat_signature(before) != _stat_signature(after)
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while reading: {path}")
    return payload, _stat_signature(after)


def _hash_regular_file_receipt(
    path: Path,
) -> tuple[int, str, _StatSignature]:
    """Hash one file while binding the open handle to its path identity."""

    _require_regular_file(path, path.name)
    before = path.lstat()
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _handle_identity_signature(before) != _handle_identity_signature(opened):
            raise RuntimeError(f"file identity changed before hashing: {path}")
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            digest.update(chunk)
        handle_after = os.fstat(handle.fileno())
    after = path.lstat()
    if (
        _handle_identity_signature(before) != _handle_identity_signature(handle_after)
        or _handle_identity_signature(handle_after) != _handle_identity_signature(after)
        or _stat_signature(before) != _stat_signature(after)
        or observed_bytes != after.st_size
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return observed_bytes, "sha256:" + digest.hexdigest(), _stat_signature(after)


def _require_source_state_unchanged(
    adapter_dir: Path,
    adapter_tree_receipt: _TreeReceipt,
    file_receipts: dict[Path, _StatSignature],
    directory_receipts: dict[Path, _StatSignature],
) -> None:
    _require_tree_receipt_unchanged(
        adapter_dir,
        adapter_tree_receipt,
        {item[0] for item in ADAPTER_FILE_SPECS},
        allowed_directories=frozenset(),
        label="adapter",
    )
    _require_directory_receipts_unchanged(directory_receipts)
    _require_file_receipts_unchanged(file_receipts, "source")


def _require_build_state_unchanged(
    base_model_dir: Path,
    base_tree_receipt: _TreeReceipt,
    component_receipts: dict[Path, _StatSignature],
    adapter_dir: Path,
    adapter_tree_receipt: _TreeReceipt,
    source_receipts: dict[Path, _StatSignature],
    directory_receipts: dict[Path, _StatSignature],
) -> None:
    _require_tree_receipt_unchanged(
        base_model_dir,
        base_tree_receipt,
        {item[0] for item in (*BASE_MODEL_FILE_SPECS, *TOKENIZER_FILE_SPECS)},
        allowed_directories=frozenset({".cache"}),
        label="base_model_and_tokenizer",
    )
    _require_file_receipts_unchanged(component_receipts, "component")
    _require_source_state_unchanged(
        adapter_dir,
        adapter_tree_receipt,
        source_receipts,
        directory_receipts,
    )


def _require_tree_receipt_unchanged(
    root: Path,
    expected_receipt: _TreeReceipt,
    expected_files: set[str],
    *,
    allowed_directories: frozenset[str],
    label: str,
) -> None:
    observed = _require_safe_tree(
        root,
        expected_files,
        allowed_directories=allowed_directories,
        label=label,
    )
    if observed != expected_receipt:
        raise RuntimeError(f"{label} tree changed after observation: {root}")


def _require_directory_receipts_unchanged(
    receipts: dict[Path, _StatSignature],
) -> None:
    for path, receipt in sorted(
        receipts.items(), key=lambda item: str(item[0]).casefold()
    ):
        if (
            not path.is_dir()
            or _is_reparse(path)
            or _stat_signature(path.lstat()) != receipt
        ):
            raise RuntimeError(f"repository directory changed after read: {path}")


def _require_file_receipts_unchanged(
    receipts: dict[Path, _StatSignature], label: str
) -> None:
    for path, receipt in sorted(
        receipts.items(), key=lambda item: str(item[0]).casefold()
    ):
        if (
            not path.is_file()
            or _is_reparse(path)
            or _stat_signature(path.lstat()) != receipt
        ):
            raise RuntimeError(f"{label} file changed after read: {path}")


def _stat_signature(value: os.stat_result) -> _StatSignature:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _handle_identity_signature(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    # Windows handle fstat may expose birth time as ctime while a path stat
    # exposes metadata-change time, so ctime is compared path-to-path only.
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _is_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _load_json_payload(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    _validate_finite(value, f"$.{path.name}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_finite(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON number at {path}: {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _payload(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build(
        base_model_dir=args.base_model_dir,
        adapter_dir=args.adapter_dir,
    )
    payload = _payload(manifest)
    manifest_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    if args.check:
        if not args.output.is_file() or _is_reparse(args.output):
            raise RuntimeError(f"unsafe or missing manifest: {args.output}")
        if args.output.read_bytes() != payload:
            raise RuntimeError(
                f"manifest differs from deterministic rebuild: {args.output}"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(args.output, flags, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    print(
        json.dumps(
            {
                "artifact_kind": manifest["artifact_kind"],
                "bytes": len(payload),
                "check": args.check,
                "component_files": manifest["observed_file_receipts"][
                    "component_files_observed"
                ],
                "manifest_sha256": manifest_sha256,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build",
    "load_repository_manifest_inputs",
    "repository_source_paths",
]
