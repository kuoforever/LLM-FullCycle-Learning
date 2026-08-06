"""Materialize the frozen FP32 attached package into one clean scratch root.

This is a transport-only orchestrator. It permits network access only for a
pre-registered, ephemeral, byte-preserving materialization. It never copies
from the current checkout or model cache, and it never merges, mutates, saves,
publishes, or promotes weights. A successful receipt requires later execution
to run offline and is not remote-origin attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, NoReturn, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge import (  # noqa: E402
    tool_router_fp32_attached_offline_package_reproducibility as repro_contract,
)
from fullcycle_bridge.tool_router_fp32_attached_offline_package_manifest import (  # noqa: E402
    ADAPTER_FILE_SPECS,
    BASE_MODEL_FILE_SPECS,
    REPOSITORY_SOURCE_PATHS,
    TOKENIZER_FILE_SPECS,
)

CLEAN_PARENT = ROOT / "work" / "clean-location"
RECEIPT_PARENT = ROOT / "work" / "test-fixtures"
REPOSITORY_REMOTE_URL = (
    "https://github.com/kuoforever/reliable-agent-model-lifecycle.git"
)
MANIFEST_RELATIVE_PATH = (
    "baseline/fc-mvp-001-fp32-attached-offline-package-manifest-v1.json"
)
DOWNLOADER_RELATIVE_PATH = "scripts/download_pinned_tool_router_model.py"
ADAPTER_RELATIVE_ROOT = "baseline/adapters/fc-mvp-001-lora-sft-v2"
ADAPTER_LFS_RELATIVE_PATH = f"{ADAPTER_RELATIVE_ROOT}/adapter_model.safetensors"
MATERIALIZER_RELATIVE_PATH = (
    "scripts/materialize_tool_router_fp32_attached_offline_package_reproducibility.py"
)
EXPECTED_GROUPS = {
    "base_model_and_tokenizer": (9, 3_098_971_928),
    "adapter": (3, 17_468_332),
    "repository": (15, None),
}
COMMAND_TIMEOUT_SECONDS = 7_200
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GUID_RE = re.compile(r"^[0-9a-f]{32}$")
_StatSignature = tuple[int, int, int, int, int, int]
CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


class MaterializationError(RuntimeError):
    """Fail-closed materialization error with a stable code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class MaterializationContract:
    """Validated transport fields extracted from the frozen preregistration."""

    gate_id: str
    experiment_id: str
    package_id: str
    preregistration_sha256: str
    repository_remote_url: str
    manifest_relative_path: str
    manifest_sha256: str
    downloader_relative_path: str
    downloader_sha256: str
    adapter_lfs_relative_path: str
    adapter_lfs_oid: str
    adapter_lfs_bytes: int
    protocol_sources: Mapping[str, Mapping[str, Any]]
    phase_order: Sequence[str]
    destination_children: Mapping[str, str]
    preregistration: Mapping[str, Any]


@dataclass(frozen=True)
class CleanLayout:
    """All materialized paths, kept out of the receipt itself."""

    destination: Path
    repository: Path
    base_model_and_tokenizer: Path
    adapter: Path
    transport: Path


def _fail(code: str, detail: str = "") -> NoReturn:
    raise MaterializationError(code, detail)


def _is_reparse(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _stat_signature(value: os.stat_result) -> _StatSignature:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(parent)))
    except ValueError:
        return False
    return True


def _require_safe_directory(path: Path, label: str) -> _StatSignature:
    if not path.is_dir() or _is_reparse(path):
        _fail("UNSAFE_DIRECTORY", label)
    return _stat_signature(path.lstat())


def _ensure_owned_parent(path: Path, *, root: Path, label: str) -> None:
    """Create one fixed parent while rejecting traversal and reparse ancestors."""

    expected = root / path.relative_to(root)
    if not _same_path(path, expected):
        _fail("TARGET_ESCAPE", label)
    current = root
    _require_safe_directory(current, f"{label}:root")
    for part in path.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current):
            _require_safe_directory(current, label)
        else:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                _fail("PARENT_CREATE_RACE", label)
            _require_safe_directory(current, label)


def _validate_destination_path(destination: Path) -> None:
    if not destination.is_absolute():
        _fail("DESTINATION_NOT_ABSOLUTE")
    if _GUID_RE.fullmatch(destination.name) is None:
        _fail("DESTINATION_NOT_GUID")
    if not _same_path(destination.parent, CLEAN_PARENT):
        _fail("DESTINATION_ESCAPE")
    if os.path.lexists(destination):
        _fail("DESTINATION_ALREADY_EXISTS")


def _create_destination(destination: Path) -> _StatSignature:
    _ensure_owned_parent(CLEAN_PARENT, root=ROOT, label="clean_parent")
    _validate_destination_path(destination)
    parent_before = _require_safe_directory(CLEAN_PARENT, "clean_parent")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError:
        _fail("DESTINATION_CREATE_RACE")
    destination_receipt = _require_safe_directory(destination, "destination")
    parent_after = _require_safe_directory(CLEAN_PARENT, "clean_parent")
    if parent_after[:3] != parent_before[:3]:
        _fail("CLEAN_PARENT_CHANGED_DURING_CREATE")
    return destination_receipt


def _validate_receipt_path(
    receipt: Path, destination: Path, *, create_parent: bool = True
) -> None:
    if not receipt.is_absolute():
        _fail("RECEIPT_NOT_ABSOLUTE")
    if create_parent:
        _ensure_owned_parent(RECEIPT_PARENT, root=ROOT, label="receipt_parent")
    elif os.path.lexists(RECEIPT_PARENT):
        _require_safe_directory(RECEIPT_PARENT, "receipt_parent")
    else:
        _require_safe_directory(RECEIPT_PARENT.parent, "receipt_parent_parent")
    if not _same_path(receipt.parent, RECEIPT_PARENT):
        _fail("RECEIPT_ESCAPE")
    if receipt.suffix != ".json":
        _fail("INVALID_RECEIPT_SUFFIX")
    if os.path.lexists(receipt):
        _fail("RECEIPT_ALREADY_EXISTS")
    if _is_within(receipt, destination):
        _fail("RECEIPT_INSIDE_EPHEMERAL_ROOT")


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("INVALID_RELATIVE_PATH", label)
    if "\\" in value or "\x00" in value or ":" in value:
        _fail("INVALID_RELATIVE_PATH", label)
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or str(pure) != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("INVALID_RELATIVE_PATH", label)
    return value


def _path_under(root: Path, relative: str, label: str) -> Path:
    safe_relative = _safe_relative_path(relative, label)
    target = root.joinpath(*PurePosixPath(safe_relative).parts)
    if not _is_within(target, root):
        _fail("TARGET_ESCAPE", label)
    return target


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail("INVALID_SHA256", label)
    return value


def _validate_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _COMMIT_RE.fullmatch(value) is None
        or value == "0" * 40
    ):
        _fail("INVALID_FREEZE_COMMIT", label)
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", label)
    return value


def _require_bool(value: object, expected: bool, label: str) -> None:
    if value is not expected:
        _fail("AUTHORIZATION_MISMATCH", label)


def _read_bytes_once(path: Path, label: str, *, max_bytes: int) -> bytes:
    if not path.is_absolute() or not path.is_file() or _is_reparse(path):
        _fail("UNSAFE_REGULAR_FILE", label)
    before = path.lstat()
    if before.st_nlink != 1:
        _fail("HARDLINK_FORBIDDEN", label)
    if before.st_size <= 0 or before.st_size > max_bytes:
        _fail("INVALID_FILE_SIZE", label)
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _stat_signature(before)[:5] != _stat_signature(opened)[:5]:
            _fail("FILE_IDENTITY_CHANGED", label)
        payload = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.lstat()
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(opened)[:5] != _stat_signature(opened_after)[:5]
        or len(payload) != after.st_size
    ):
        _fail("FILE_CHANGED_DURING_READ", label)
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite constant: {value}")


def _file_sha256(path: Path, label: str) -> tuple[int, str]:
    if not path.is_file() or _is_reparse(path):
        _fail("UNSAFE_REGULAR_FILE", label)
    before = path.lstat()
    if before.st_nlink != 1:
        _fail("HARDLINK_FORBIDDEN", label)
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _stat_signature(before)[:5] != _stat_signature(opened)[:5]:
            _fail("FILE_IDENTITY_CHANGED", label)
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    after = path.lstat()
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(opened)[:5] != _stat_signature(opened_after)[:5]
        or observed != after.st_size
    ):
        _fail("FILE_CHANGED_DURING_HASH", label)
    return observed, "sha256:" + digest.hexdigest()


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def _run_checked(
    role: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[bytes]:
    completed = runner(command, cwd=cwd, env=env)
    if completed.returncode != 0:
        _fail("TRANSPORT_COMMAND_FAILED", f"{role}:exit={completed.returncode}")
    return completed


def _git_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key in {"GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"} or key.startswith(
            ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            env.pop(key, None)
    for key in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
    ):
        env.pop(key, None)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": "NUL",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _hf_environment(destination: Path) -> dict[str, str]:
    hf_home = destination / "transport" / "hf-home"
    env = dict(os.environ)
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        env.pop(key, None)
    env.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_ENDPOINT": "https://huggingface.co",
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_OFFLINE": "0",
        }
    )
    return env


def _downloader_local_dir_argument(
    path: Path,
    *,
    platform_name: str | None = None,
) -> str:
    """Use one extended-length prefix only for the Windows downloader argument."""

    value = str(path)
    if (os.name if platform_name is None else platform_name) != "nt":
        return value

    extended_prefix = "\\\\?\\"
    device_prefix = "\\\\.\\"
    normalized = str(PureWindowsPath(value))
    if normalized.startswith(device_prefix):
        _fail("DOWNLOADER_LOCAL_DIR_UNSAFE_PREFIX")
    if normalized.startswith(extended_prefix):
        suffix = normalized[len(extended_prefix) :]
        if suffix.startswith((extended_prefix, "\\?\\")):
            _fail("DOWNLOADER_LOCAL_DIR_DUPLICATE_PREFIX")
        if suffix.upper().startswith("UNC\\"):
            unprefixed = PureWindowsPath("\\\\" + suffix[4:])
        else:
            unprefixed = PureWindowsPath(suffix)
        if not unprefixed.is_absolute() or ".." in unprefixed.parts:
            _fail("DOWNLOADER_LOCAL_DIR_NOT_ABSOLUTE")
        return normalized

    windows_path = PureWindowsPath(normalized)
    if not windows_path.is_absolute() or ".." in windows_path.parts:
        _fail("DOWNLOADER_LOCAL_DIR_NOT_ABSOLUTE")
    if normalized.startswith("\\\\"):
        return f"{extended_prefix}UNC\\{normalized[2:]}"
    if len(windows_path.drive) != 2 or not windows_path.drive.endswith(":"):
        _fail("DOWNLOADER_LOCAL_DIR_UNSUPPORTED_ROOT")
    return extended_prefix + normalized


def _clean_git_command() -> list[str]:
    return ["git", "-c", "credential.helper=", "-c", "core.hooksPath=NUL"]


def _local_lfs_git_command() -> list[str]:
    """Provide an offline-capable LFS filter despite isolated Git config."""

    return [
        *_clean_git_command(),
        "-c",
        "filter.lfs.process=git-lfs filter-process --skip",
        "-c",
        "filter.lfs.required=true",
        "-c",
        "filter.lfs.clean=git-lfs clean -- %f",
        "-c",
        "filter.lfs.smudge=git-lfs smudge --skip -- %f",
    ]


def _require_clean_repository_status(
    layout: CleanLayout,
    *,
    runner: CommandRunner,
) -> None:
    """Reject tracked, untracked, and ignored worktree entries."""

    environment = _git_environment()
    environment.pop("GIT_LFS_SKIP_SMUDGE", None)
    status = _run_checked(
        "git_status",
        [
            *_local_lfs_git_command(),
            "-C",
            str(layout.repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored",
        ],
        cwd=layout.destination,
        env=environment,
        runner=runner,
    ).stdout
    if status:
        _fail("CLEAN_REPOSITORY_DIRTY")


def _checkout_adapter_from_local_lfs(
    layout: CleanLayout,
    contract: MaterializationContract,
    *,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> None:
    """Hydrate only the frozen Adapter path from the verified local LFS object."""

    _run_checked(
        "git_lfs_checkout_adapter",
        [
            *_local_lfs_git_command(),
            "-C",
            str(layout.repository),
            "lfs",
            "checkout",
            contract.adapter_lfs_relative_path,
        ],
        cwd=layout.destination,
        env=environment,
        runner=runner,
    )


def _materialize_repository(
    layout: CleanLayout,
    contract: MaterializationContract,
    freeze_commit: str,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    if os.path.lexists(layout.repository):
        _fail("REPOSITORY_ROOT_ALREADY_EXISTS")
    git_env = _git_environment()
    git = _clean_git_command()
    _run_checked(
        "git_init",
        [*git, "init", str(layout.repository)],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    )
    _require_safe_directory(layout.repository, "repository")
    _require_safe_directory(layout.repository / ".git", "repository_git")
    _run_checked(
        "git_config_autocrlf",
        [*git, "-C", str(layout.repository), "config", "core.autocrlf", "false"],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    )
    _run_checked(
        "git_remote_add",
        [
            *git,
            "-C",
            str(layout.repository),
            "remote",
            "add",
            "origin",
            contract.repository_remote_url,
        ],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    )
    observed_remote = (
        _run_checked(
            "git_remote_verify",
            [*git, "-C", str(layout.repository), "remote", "get-url", "origin"],
            cwd=layout.destination,
            env=git_env,
            runner=runner,
        )
        .stdout.decode("utf-8", errors="strict")
        .strip()
    )
    if observed_remote != contract.repository_remote_url:
        _fail("REPOSITORY_REMOTE_MISMATCH")
    if os.path.lexists(layout.repository / ".lfsconfig"):
        _fail("ALTERNATE_LFS_CONFIGURATION_FORBIDDEN")
    _run_checked(
        "git_fetch_frozen_commit",
        [
            *git,
            "-C",
            str(layout.repository),
            "fetch",
            "--no-tags",
            "--depth=1",
            "origin",
            freeze_commit,
        ],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    )
    _run_checked(
        "git_checkout_detached",
        [
            *git,
            "-C",
            str(layout.repository),
            "checkout",
            "--detach",
            "FETCH_HEAD",
        ],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    )
    observed_commit = (
        _run_checked(
            "git_resolve_head",
            [*git, "-C", str(layout.repository), "rev-parse", "HEAD"],
            cwd=layout.destination,
            env=git_env,
            runner=runner,
        )
        .stdout.decode("ascii", errors="strict")
        .strip()
    )
    if observed_commit != freeze_commit:
        _fail("CHECKED_OUT_COMMIT_MISMATCH")

    protocol_receipts = _verify_protocol_sources(
        layout.repository, contract.protocol_sources
    )
    clean_preregistration = _path_under(
        layout.repository,
        repro_contract.PREREGISTRATION_PATH,
        "clean_preregistration",
    )
    _clean_bytes, clean_preregistration_sha256 = _file_sha256(
        clean_preregistration, "clean_preregistration"
    )
    if clean_preregistration_sha256 != contract.preregistration_sha256:
        _fail("CLEAN_PREREGISTRATION_HASH_MISMATCH")
    pointer = _run_checked(
        "git_read_adapter_pointer",
        [
            *git,
            "-C",
            str(layout.repository),
            "show",
            f"HEAD:{contract.adapter_lfs_relative_path}",
        ],
        cwd=layout.destination,
        env=git_env,
        runner=runner,
    ).stdout
    _validate_lfs_pointer(pointer, contract)

    lfs_env = dict(git_env)
    lfs_env.pop("GIT_LFS_SKIP_SMUDGE", None)
    _run_checked(
        "git_lfs_pull_adapter",
        [
            *git,
            "-C",
            str(layout.repository),
            "lfs",
            "pull",
            f"--include={contract.adapter_lfs_relative_path}",
            "--exclude=",
        ],
        cwd=layout.destination,
        env=lfs_env,
        runner=runner,
    )
    _verify_single_lfs_object(layout.repository, contract)
    _checkout_adapter_from_local_lfs(
        layout,
        contract,
        environment=lfs_env,
        runner=runner,
    )
    adapter_weight = _path_under(
        layout.repository,
        contract.adapter_lfs_relative_path,
        "adapter_lfs_path",
    )
    observed_bytes, observed_sha256 = _file_sha256(
        adapter_weight, "adapter_lfs_worktree_file"
    )
    if (
        observed_bytes != contract.adapter_lfs_bytes
        or observed_sha256 != contract.adapter_lfs_oid
    ):
        _fail("ADAPTER_LFS_OBJECT_MISMATCH")
    _require_clean_repository_status(layout, runner=runner)
    git_version = (
        _run_checked(
            "git_version",
            [*git, "--version"],
            cwd=layout.destination,
            env=lfs_env,
            runner=runner,
        )
        .stdout.decode("utf-8", errors="strict")
        .strip()
    )
    lfs_version = (
        _run_checked(
            "git_lfs_version",
            [*git, "lfs", "version"],
            cwd=layout.destination,
            env=lfs_env,
            runner=runner,
        )
        .stdout.decode("utf-8", errors="strict")
        .strip()
    )
    return {
        "transport": "fresh_https_git_fetch_and_exact_lfs_pull",
        "remote_url": contract.repository_remote_url,
        "requested_commit": freeze_commit,
        "resolved_commit": observed_commit,
        "detached_head": True,
        "lfs_skip_smudge_during_fetch_and_checkout": True,
        "lfs_objects_requested": 1,
        "adapter_lfs": {
            "path": contract.adapter_lfs_relative_path,
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        },
        "protocol_sources": protocol_receipts,
        "git_version": git_version,
        "git_lfs_version": lfs_version,
    }


def _validate_lfs_pointer(payload: bytes, contract: MaterializationContract) -> None:
    expected = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid {contract.adapter_lfs_oid}\n"
        f"size {contract.adapter_lfs_bytes}\n"
    ).encode("ascii")
    if payload.replace(b"\r\n", b"\n") != expected:
        _fail("ADAPTER_LFS_POINTER_MISMATCH")


def _verify_protocol_sources(
    repository: Path, expected_sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, source in sorted(expected_sources.items()):
        relative = str(source["path"])
        expected_sha256 = str(source["sha256"])
        path = _path_under(repository, relative, "protocol_source")
        observed_bytes, observed_sha256 = _file_sha256(path, relative)
        if observed_sha256 != expected_sha256:
            _fail("PROTOCOL_SOURCE_HASH_MISMATCH", relative)
        receipts[name] = {
            "path": relative,
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        }
    return receipts


def _verify_single_lfs_object(
    repository: Path, contract: MaterializationContract
) -> None:
    object_root = repository / ".git" / "lfs" / "objects"
    _require_safe_directory(object_root, "lfs_object_root")
    object_files: list[Path] = []
    for current, directory_names, file_names in os.walk(object_root):
        current_path = Path(current)
        _require_safe_directory(current_path, "lfs_object_directory")
        for directory_name in directory_names:
            directory = current_path / directory_name
            if _is_reparse(directory):
                _fail("UNSAFE_LFS_OBJECT_DIRECTORY")
        for file_name in file_names:
            object_files.append(current_path / file_name)
    if len(object_files) != 1:
        _fail("LFS_OBJECT_COUNT_MISMATCH", str(len(object_files)))
    observed_bytes, observed_sha256 = _file_sha256(
        object_files[0], "adapter_lfs_cached_object"
    )
    if (
        observed_bytes != contract.adapter_lfs_bytes
        or observed_sha256 != contract.adapter_lfs_oid
    ):
        _fail("ADAPTER_LFS_CACHED_OBJECT_MISMATCH")


def _materialize_base(
    layout: CleanLayout,
    contract: MaterializationContract,
    python_executable: Path,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    if os.path.lexists(layout.base_model_and_tokenizer):
        _fail("BASE_ROOT_ALREADY_EXISTS")
    downloader = _path_under(
        layout.repository,
        contract.downloader_relative_path,
        "downloader_source",
    )
    downloader_bytes, downloader_sha256 = _file_sha256(downloader, "downloader_source")
    if downloader_sha256 != contract.downloader_sha256:
        _fail("DOWNLOADER_SOURCE_HASH_MISMATCH")
    hf_env = _hf_environment(layout.destination)
    completed = _run_checked(
        "manifest_bound_downloader",
        [
            str(python_executable),
            "-I",
            str(downloader),
            "--local-dir",
            _downloader_local_dir_argument(layout.base_model_and_tokenizer),
        ],
        cwd=layout.destination,
        env=hf_env,
        runner=runner,
    )
    _require_safe_directory(layout.base_model_and_tokenizer, "base_model_and_tokenizer")
    stdout_sha256 = "sha256:" + hashlib.sha256(completed.stdout).hexdigest()
    return {
        "transport": "manifest_bound_huggingface_snapshot_downloader",
        "official_endpoint": "https://huggingface.co",
        "isolated_hf_home": True,
        "global_cache_reused": False,
        "implicit_token_disabled": True,
        "telemetry_disabled": True,
        "downloader": {
            "path": contract.downloader_relative_path,
            "bytes": downloader_bytes,
            "sha256": downloader_sha256,
            "invoked": True,
            "stdout_sha256": stdout_sha256,
        },
    }


def _validate_python_executable(path: Path) -> Path:
    if not path.is_absolute() or not path.is_file() or _is_reparse(path):
        _fail("UNSAFE_PYTHON_EXECUTABLE")
    return path.resolve(strict=True)


def _validate_clean_package(
    layout: CleanLayout,
    contract: MaterializationContract,
    python_executable: Path,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    _reject_component_links(layout)
    materializer = _path_under(
        layout.repository,
        MATERIALIZER_RELATIVE_PATH,
        "clean_materializer_source",
    )
    clean_preregistration = _path_under(
        layout.repository,
        repro_contract.PREREGISTRATION_PATH,
        "clean_preregistration",
    )
    _clean_preregistration_bytes, clean_preregistration_sha256 = _file_sha256(
        clean_preregistration, "clean_preregistration"
    )
    if clean_preregistration_sha256 != contract.preregistration_sha256:
        _fail("CLEAN_PREREGISTRATION_HASH_MISMATCH")
    command = [
        str(python_executable),
        "-I",
        "-B",
        str(materializer),
        "--internal-validate-clean-package",
        "--preregistration",
        str(clean_preregistration),
        "--base-model-root",
        str(layout.base_model_and_tokenizer),
        "--adapter-root",
        str(layout.adapter),
        "--repository-root",
        str(layout.repository),
    ]
    completed = _run_checked(
        "clean_repository_manifest_resolution",
        command,
        cwd=layout.destination,
        env=_offline_environment(),
        runner=runner,
    )
    try:
        result = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail("INVALID_CLEAN_RESOLUTION_OUTPUT", exc.__class__.__name__)
    if not isinstance(result, dict):
        _fail("INVALID_CLEAN_RESOLUTION_OUTPUT", "expected object")
    _validate_resolution(result, contract)
    return result


def _reject_component_links(layout: CleanLayout) -> None:
    file_roots = (
        (
            layout.base_model_and_tokenizer,
            [item[0] for item in (*BASE_MODEL_FILE_SPECS, *TOKENIZER_FILE_SPECS)],
            "base_model_and_tokenizer",
        ),
        (
            layout.adapter,
            [item[0] for item in ADAPTER_FILE_SPECS],
            "adapter",
        ),
        (
            layout.repository,
            list(REPOSITORY_SOURCE_PATHS.values()),
            "repository",
        ),
    )
    for root, relative_paths, role in file_roots:
        for relative in relative_paths:
            path = _path_under(root, relative, role)
            if not path.is_file() or _is_reparse(path):
                _fail("UNSAFE_COMPONENT_FILE", f"{role}:{relative}")
            if path.lstat().st_nlink != 1:
                _fail("HARDLINK_FORBIDDEN", f"{role}:{relative}")


def _offline_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        env.pop(key, None)
    env.update(
        {
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return env


def _validate_resolution(
    combined: Mapping[str, Any], contract: MaterializationContract
) -> None:
    validation = _mapping(combined.get("validation"), "validation")
    resolution = _mapping(combined.get("resolution"), "resolution")
    if (
        validation.get("frozen_manifest_valid") is not True
        or validation.get("manifest_file_sha256") != contract.manifest_sha256
        or validation.get("metadata_complete") is not True
        or validation.get("offline_package_identity_complete") is not True
        or validation.get("remote_revision_origin_attested") is not False
        or resolution.get("manifest_file_sha256") != contract.manifest_sha256
        or resolution.get("resolved") is not True
        or resolution.get("eligible_for_clean_location_reproducibility_test")
        is not True
        or resolution.get("manifest_machine_paths_used") is not False
        or resolution.get("adapter_local_base_path_used") is not False
        or resolution.get("offline_artifact_eligible") is not False
        or resolution.get("runtime_eligible") is not False
    ):
        _fail("CLEAN_PACKAGE_RESOLUTION_FAILED")
    groups = resolution.get("groups")
    if not isinstance(groups, list) or len(groups) != len(EXPECTED_GROUPS):
        _fail("CLEAN_PACKAGE_GROUPS_MISMATCH")
    by_role = {
        item.get("root_role"): item for item in groups if isinstance(item, Mapping)
    }
    if set(by_role) != set(EXPECTED_GROUPS):
        _fail("CLEAN_PACKAGE_GROUPS_MISMATCH")
    for role, (expected_count, expected_bytes) in EXPECTED_GROUPS.items():
        group = by_role[role]
        if (
            group.get("resolved") is not True
            or group.get("expected_files") != expected_count
            or group.get("matched_files") != expected_count
            or group.get("issues") != []
        ):
            _fail("CLEAN_PACKAGE_GROUP_MISMATCH", role)
        if expected_bytes is not None and group.get("matched_bytes") != expected_bytes:
            _fail("CLEAN_PACKAGE_GROUP_BYTE_MISMATCH", role)


def _mapping_groups(value: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        _fail("CLEAN_PACKAGE_GROUPS_MISMATCH")
    result: dict[str, Mapping[str, Any]] = {}
    for item in value:
        group = _mapping(item, "resolution_group")
        role = group.get("root_role")
        if not isinstance(role, str) or role in result:
            _fail("CLEAN_PACKAGE_GROUPS_MISMATCH")
        result[role] = group
    return result


def _canonical_json_bytes(value: object) -> bytes:
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


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    if value.startswith(("https://", "sha256:")):
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    if _contains_absolute_path(receipt):
        _fail("ABSOLUTE_PATH_IN_RECEIPT")
    payload = _canonical_json_bytes(receipt)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        _fail("RECEIPT_CREATE_RACE")
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _remove_owned_destination(destination: Path) -> None:
    if not _same_path(destination.parent, CLEAN_PARENT):
        _fail("CLEANUP_TARGET_ESCAPE")
    if _GUID_RE.fullmatch(destination.name) is None:
        _fail("CLEANUP_TARGET_NOT_GUID")
    if not os.path.lexists(destination):
        return
    _require_safe_cleanup_parent_chain()
    destination_metadata = _cleanup_lstat(destination, "target")
    if _metadata_is_reparse(destination_metadata) or not stat.S_ISDIR(
        destination_metadata.st_mode
    ):
        _fail("UNSAFE_CLEANUP_TARGET")
    _remove_cleanup_directory(
        destination,
        expected_identity=_cleanup_identity(destination_metadata),
    )
    if os.path.lexists(destination):
        _fail("CLEANUP_INCOMPLETE")


def _remove_cleanup_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    metadata = _cleanup_lstat(path, "directory")
    if (
        _metadata_is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
        or _cleanup_identity(metadata) != expected_identity
    ):
        _fail("UNSAFE_CLEANUP_DIRECTORY")
    directory_descriptor = (
        _open_windows_cleanup_directory_handle(path, expected_identity)
        if os.name == "nt"
        else None
    )
    traversal_complete = False
    try:
        try:
            with os.scandir(path) as entries:
                child_names = sorted(entry.name for entry in entries)
        except OSError as exc:
            raise MaterializationError("CLEANUP_SCAN_FAILED") from exc
        for name in child_names:
            child = path / name
            child_metadata = _cleanup_lstat(child, "entry")
            if _metadata_is_reparse(child_metadata):
                _fail("UNSAFE_CLEANUP_ENTRY")
            if stat.S_ISDIR(child_metadata.st_mode):
                _remove_cleanup_directory(
                    child,
                    expected_identity=_cleanup_identity(child_metadata),
                )
            elif stat.S_ISREG(child_metadata.st_mode):
                _remove_cleanup_regular_file(child, child_metadata)
            else:
                _fail("UNSAFE_CLEANUP_ENTRY")
        traversal_complete = True
    finally:
        if directory_descriptor is not None:
            try:
                os.close(directory_descriptor)
            except OSError as exc:
                if traversal_complete:
                    raise MaterializationError(
                        "CLEANUP_DIRECTORY_HANDLE_CLOSE_FAILED"
                    ) from exc
    final_metadata = _cleanup_lstat(path, "directory")
    if (
        _metadata_is_reparse(final_metadata)
        or not stat.S_ISDIR(final_metadata.st_mode)
        or _cleanup_identity(final_metadata) != expected_identity
    ):
        _fail("CLEANUP_DIRECTORY_CHANGED")
    try:
        path.rmdir()
    except OSError as exc:
        raise MaterializationError("CLEANUP_DIRECTORY_REMOVE_FAILED") from exc


def _require_safe_cleanup_parent_chain() -> None:
    try:
        relative = CLEAN_PARENT.relative_to(ROOT)
    except ValueError:
        _fail("CLEANUP_PARENT_ESCAPE")
    current = ROOT
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        metadata = _cleanup_lstat(current, "parent")
        if _metadata_is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            _fail("UNSAFE_CLEANUP_PARENT")


def _remove_cleanup_regular_file(path: Path, expected: os.stat_result) -> None:
    expected_identity = _cleanup_identity(expected)
    _require_cleanup_regular_identity(expected, expected_identity)
    try:
        path.unlink()
    except PermissionError:
        current = _cleanup_lstat(path, "read_only_file")
        _require_cleanup_regular_identity(current, expected_identity)
        if os.name != "nt":
            _fail("CLEANUP_NOFOLLOW_DELETE_UNAVAILABLE")
        _remove_windows_readonly_file_by_handle(path, expected_identity)
    except OSError as exc:
        raise MaterializationError("CLEANUP_FILE_REMOVE_FAILED") from exc
    if os.path.lexists(path):
        _fail("CLEANUP_FILE_REMOVE_INCOMPLETE")


def _cleanup_lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise MaterializationError("CLEANUP_ENTRY_UNREADABLE", label) from exc


def _require_cleanup_regular_identity(
    metadata: os.stat_result,
    expected_identity: tuple[int, int],
) -> None:
    if (
        _metadata_is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or _cleanup_identity(metadata) != expected_identity
        or int(metadata.st_nlink) != 1
    ):
        _fail("CLEANUP_FILE_CHANGED")


def _open_windows_cleanup_handle(
    path: Path,
    *,
    desired_access: int,
    flags: int,
    failure_code: str,
) -> int:
    """Open one exact Windows entry without following or permitting replacement."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    file_share_read = 0x0001
    open_existing = 3
    invalid_handle = wintypes.HANDLE(-1).value
    raw_handle = create_file(
        str(path),
        desired_access,
        file_share_read,
        None,
        open_existing,
        flags,
        None,
    )
    if raw_handle == invalid_handle:
        error = ctypes.WinError(ctypes.get_last_error())
        raise MaterializationError(failure_code) from error
    try:
        return msvcrt.open_osfhandle(
            int(raw_handle),
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except OSError as exc:
        close_handle(raw_handle)
        raise MaterializationError(f"{failure_code}_WRAP") from exc


def _windows_final_path(descriptor: int) -> Path:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    get_final_path.restype = wintypes.DWORD

    buffer = ctypes.create_unicode_buffer(32_768)
    observed = get_final_path(
        wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
        buffer,
        len(buffer),
        0,
    )
    if observed == 0 or observed >= len(buffer):
        error = ctypes.WinError(ctypes.get_last_error())
        raise MaterializationError("CLEANUP_HANDLE_PATH_FAILED") from error
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _open_windows_cleanup_directory_handle(
    path: Path,
    expected_identity: tuple[int, int],
) -> int:
    file_list_directory = 0x0001
    file_read_attributes = 0x0080
    file_flag_open_reparse_point = 0x00200000
    file_flag_backup_semantics = 0x02000000
    descriptor = _open_windows_cleanup_handle(
        path,
        desired_access=file_list_directory | file_read_attributes,
        flags=file_flag_open_reparse_point | file_flag_backup_semantics,
        failure_code="CLEANUP_DIRECTORY_HANDLE_FAILED",
    )
    try:
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise MaterializationError(
                "CLEANUP_DIRECTORY_HANDLE_INSPECT_FAILED"
            ) from exc
        if (
            _metadata_is_reparse(opened)
            or not stat.S_ISDIR(opened.st_mode)
            or _cleanup_identity(opened) != expected_identity
            or not _same_path(_windows_final_path(descriptor), path)
        ):
            _fail("CLEANUP_DIRECTORY_CHANGED")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return descriptor


def _remove_windows_readonly_file_by_handle(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    """Delete one verified READONLY file without changing its shared attributes."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileDispositionInfoEx(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD)]

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", wintypes.DWORD),
            ("CreationTime", wintypes.FILETIME),
            ("LastAccessTime", wintypes.FILETIME),
            ("LastWriteTime", wintypes.FILETIME),
            ("VolumeSerialNumber", wintypes.DWORD),
            ("FileSizeHigh", wintypes.DWORD),
            ("FileSizeLow", wintypes.DWORD),
            ("NumberOfLinks", wintypes.DWORD),
            ("FileIndexHigh", wintypes.DWORD),
            ("FileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_file_information = kernel32.GetFileInformationByHandle
    get_file_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    get_file_information.restype = wintypes.BOOL
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.BOOL
    delete_access = 0x00010000
    file_read_attributes = 0x0080
    file_flag_open_reparse_point = 0x00200000
    file_attribute_readonly = 0x0001
    file_attribute_directory = 0x0010
    file_attribute_reparse_point = 0x0400
    file_disposition_info_ex_class = 21
    file_disposition_delete = 0x00000001
    file_disposition_posix_semantics = 0x00000002
    file_disposition_ignore_readonly_attribute = 0x00000010
    descriptor = _open_windows_cleanup_handle(
        path,
        desired_access=delete_access | file_read_attributes,
        flags=file_flag_open_reparse_point,
        failure_code="CLEANUP_READONLY_HANDLE_FAILED",
    )

    operation_complete = False
    try:
        native_handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise MaterializationError("CLEANUP_READONLY_INSPECT_FAILED") from exc
        _require_cleanup_regular_identity(opened, expected_identity)
        if not _same_path(_windows_final_path(descriptor), path):
            _fail("CLEANUP_FILE_CHANGED")

        handle_information = ByHandleFileInformation()
        if not get_file_information(
            native_handle,
            ctypes.byref(handle_information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            raise MaterializationError("CLEANUP_READONLY_INSPECT_FAILED") from error
        handle_attributes = int(handle_information.FileAttributes)
        if (
            handle_attributes
            & (file_attribute_directory | file_attribute_reparse_point)
            or int(handle_information.NumberOfLinks) != 1
        ):
            _fail("CLEANUP_FILE_CHANGED")
        if not handle_attributes & file_attribute_readonly:
            _fail("CLEANUP_READONLY_ATTRIBUTE_MISSING")

        try:
            opened_before_delete = os.fstat(descriptor)
        except OSError as exc:
            raise MaterializationError("CLEANUP_READONLY_INSPECT_FAILED") from exc
        _require_cleanup_regular_identity(opened_before_delete, expected_identity)
        if not _same_path(_windows_final_path(descriptor), path):
            _fail("CLEANUP_FILE_CHANGED")
        handle_before_delete = ByHandleFileInformation()
        if not get_file_information(
            native_handle,
            ctypes.byref(handle_before_delete),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            raise MaterializationError("CLEANUP_READONLY_INSPECT_FAILED") from error
        if (
            int(handle_before_delete.FileAttributes)
            & (file_attribute_directory | file_attribute_reparse_point)
            or not int(handle_before_delete.FileAttributes) & file_attribute_readonly
            or int(handle_before_delete.NumberOfLinks) != 1
        ):
            _fail("CLEANUP_FILE_CHANGED")

        disposition = FileDispositionInfoEx()
        disposition.Flags = (
            file_disposition_delete
            | file_disposition_posix_semantics
            | file_disposition_ignore_readonly_attribute
        )
        if not set_information(
            native_handle,
            file_disposition_info_ex_class,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            raise MaterializationError("CLEANUP_FILE_DISPOSITION_FAILED") from error
        operation_complete = True
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            if operation_complete:
                raise MaterializationError("CLEANUP_HANDLE_CLOSE_FAILED") from exc


def _cleanup_identity(value: os.stat_result) -> tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _metadata_is_reparse(value: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & reparse_flag
    )


def _build_layout(destination: Path) -> CleanLayout:
    repository = destination / "repository"
    return CleanLayout(
        destination=destination,
        repository=repository,
        base_model_and_tokenizer=destination / "base_model_and_tokenizer",
        adapter=_path_under(repository, ADAPTER_RELATIVE_ROOT, "adapter_root"),
        transport=destination / "transport",
    )


def _build_receipt(
    contract: MaterializationContract,
    freeze_commit: str,
    destination_id: str,
    repository_transport: Mapping[str, Any],
    combined: Mapping[str, Any],
) -> dict[str, Any]:
    resolution = _mapping(combined.get("resolution"), "resolution")
    groups = [
        dict(group) for group in _mapping_groups(resolution.get("groups")).values()
    ]
    receipt: dict[str, Any] = {
        "receipt_version": repro_contract.RECEIPT_VERSION,
        "gate_id": contract.gate_id,
        "experiment_id": contract.experiment_id,
        "package_id": contract.package_id,
        "preregistration_sha256": contract.preregistration_sha256,
        "protocol_freeze_commit": freeze_commit,
        "manifest_file_sha256": contract.manifest_sha256,
        "phase_order": list(contract.phase_order),
        "destination": {
            "destination_id": destination_id,
            "caller_supplied_root": True,
            "root_was_absent": True,
            "root_created_exclusive": True,
            "children": dict(contract.destination_children),
            "adapter_root_relative_to_repository": ADAPTER_RELATIVE_ROOT,
            "absolute_paths_recorded": False,
            "symlinks_used": False,
            "reparse_points_used": False,
            "hardlinks_used": False,
            "overwrite_used": False,
        },
        "transport": {
            "repository_remote_url": contract.repository_remote_url,
            "fresh_git_checkout": True,
            "git_fetch_used": True,
            "git_lfs_checkout_used": True,
            "model_downloader_path": contract.downloader_relative_path,
            "model_downloader_sha256": contract.downloader_sha256,
            "model_downloader_invoked": True,
            "destination_scoped_hf_home": True,
            "destination_scoped_cache": True,
            "network_used_during_materialization": True,
            "network_used_during_execution": False,
            "alternate_remote_used": False,
            "alternate_revision_fallback_used": False,
            "historical_adapter_base_path_used": False,
        },
        "clean_resolution_digest": resolution["resolution_digest"],
        "clean_groups": groups,
        "protocol_sources": repository_transport["protocol_sources"],
        "materialization_passed": True,
        "issues": [],
    }
    if _contains_absolute_path(receipt):
        _fail("ABSOLUTE_PATH_IN_RECEIPT")
    try:
        validated = repro_contract.validate_materialization_receipt(
            contract.preregistration,
            receipt,
            preregistration_sha256=contract.preregistration_sha256,
            expected_freeze_commit=freeze_commit,
            clean_resolution=resolution,
        )
    except repro_contract.ReproducibilityContractError as exc:
        _fail("MATERIALIZATION_RECEIPT_REJECTED", exc.code)
    return validated


def _contract_from_preregistration(
    value: Mapping[str, Any], preregistration_sha256: str
) -> MaterializationContract:
    """Extract fields from a preregistration already closed by the core API."""

    if value.get("freeze_status") != "frozen":
        _fail("PREREGISTRATION_NOT_FROZEN")
    gate_id = value.get("gate_id")
    experiment_id = value.get("experiment_id")
    package_id = value.get("package_id")
    if not all(isinstance(item, str) for item in (gate_id, experiment_id, package_id)):
        _fail("INVALID_PREREGISTRATION_IDENTITY")
    source_lineage = _mapping(value.get("source_lineage"), "source_lineage")
    repository = _mapping(source_lineage.get("repository"), "repository")
    if repository.get("remote_url") != REPOSITORY_REMOTE_URL:
        _fail("REPOSITORY_REMOTE_MISMATCH")
    manifest = _mapping(source_lineage.get("manifest"), "source_lineage.manifest")
    manifest_path = _safe_relative_path(
        manifest.get("path"), "source_lineage.manifest.path"
    )
    if manifest_path != MANIFEST_RELATIVE_PATH:
        _fail("MANIFEST_PATH_MISMATCH")
    materialization = _mapping(
        value.get("materialization_protocol"), "materialization_protocol"
    )
    _require_bool(
        materialization.get("formal_execution_authorized"),
        True,
        "formal_execution_authorized",
    )
    phase_authority = _mapping(
        materialization.get("phase_authority"), "materialization.phase_authority"
    )
    for key in (
        "network_allowed_during_materialization",
        "download_allowed_during_materialization",
        "git_fetch_allowed_during_materialization",
    ):
        _require_bool(phase_authority.get(key), True, key)
    for key in ("network_allowed_during_execution",):
        _require_bool(phase_authority.get(key), False, key)
    constraints = _mapping(value.get("constraints"), "constraints")
    for key in (
        "adapter_or_weight_mutation",
        "merged_weight_creation",
        "artifact_promotion",
        "serving_integration",
        "runtime_integration",
    ):
        _require_bool(constraints.get(key), False, key)
    downloader = _mapping(source_lineage.get("model_downloader"), "downloader")
    downloader_path = _safe_relative_path(downloader.get("path"), "downloader.path")
    if downloader_path != DOWNLOADER_RELATIVE_PATH:
        _fail("DOWNLOADER_CONTRACT_MISMATCH")
    adapter_lfs = _mapping(source_lineage.get("adapter_lfs"), "adapter_lfs")
    adapter_path = _safe_relative_path(adapter_lfs.get("path"), "adapter_lfs.path")
    if adapter_path != ADAPTER_LFS_RELATIVE_PATH:
        _fail("ADAPTER_LFS_PATH_MISMATCH")
    adapter_bytes = adapter_lfs.get("bytes")
    if not isinstance(adapter_bytes, int) or isinstance(adapter_bytes, bool):
        _fail("ADAPTER_LFS_BYTES_MISMATCH")
    protocol_sources = _mapping(
        source_lineage.get("protocol_sources"), "protocol_sources"
    )
    protocol_source_records: dict[str, dict[str, Any]] = {}
    for name in ("contract_source", "materializer_source", "runner_source"):
        record = _mapping(protocol_sources.get(name), f"protocol_sources.{name}")
        path = _safe_relative_path(record.get("path"), f"protocol_sources.{name}.path")
        protocol_source_records[name] = {
            "path": path,
            "sha256": _validate_sha256(
                record.get("sha256"), f"protocol_sources.{name}.sha256"
            ),
        }
    if (
        protocol_source_records.get("materializer_source", {}).get("path")
        != MATERIALIZER_RELATIVE_PATH
    ):
        _fail("MATERIALIZER_SOURCE_NOT_BOUND")
    destination_policy = _mapping(
        materialization.get("destination_policy"), "destination_policy"
    )
    phase_order = materialization.get("phase_order")
    if not isinstance(phase_order, list) or not all(
        isinstance(item, str) for item in phase_order
    ):
        _fail("INVALID_PHASE_ORDER")
    destination_children = _mapping(
        destination_policy.get("children"), "destination_policy.children"
    )
    return MaterializationContract(
        gate_id=str(gate_id),
        experiment_id=str(experiment_id),
        package_id=str(package_id),
        preregistration_sha256=preregistration_sha256,
        repository_remote_url=REPOSITORY_REMOTE_URL,
        manifest_relative_path=manifest_path,
        manifest_sha256=_validate_sha256(manifest.get("sha256"), "manifest.sha256"),
        downloader_relative_path=downloader_path,
        downloader_sha256=_validate_sha256(
            downloader.get("sha256"), "downloader.sha256"
        ),
        adapter_lfs_relative_path=adapter_path,
        adapter_lfs_oid=_validate_sha256(adapter_lfs.get("oid"), "adapter_lfs.oid"),
        adapter_lfs_bytes=adapter_bytes,
        protocol_sources=protocol_source_records,
        phase_order=tuple(phase_order),
        destination_children={
            str(key): str(item) for key, item in destination_children.items()
        },
        preregistration=dict(value),
    )


def _load_contract(preregistration: Path) -> MaterializationContract:
    loaded = repro_contract.load_and_validate_preregistration(
        preregistration, require_frozen=True
    )
    return _contract_from_preregistration(loaded.data, loaded.sha256)


def materialize(
    *,
    preregistration: Path,
    destination: Path,
    python_executable: Path,
    receipt_output: Path,
    freeze_commit: str,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Perform one authorized remote materialization and write one receipt."""

    contract = _load_contract(preregistration)
    validated_commit = _validate_commit(freeze_commit, "freeze_commit")
    validated_python = _validate_python_executable(python_executable)
    _validate_destination_path(destination)
    _validate_receipt_path(receipt_output, destination)
    _create_destination(destination)
    layout = _build_layout(destination)
    try:
        repository_transport = _materialize_repository(
            layout,
            contract,
            validated_commit,
            runner=runner,
        )
        _materialize_base(
            layout,
            contract,
            validated_python,
            runner=runner,
        )
        combined = _validate_clean_package(
            layout,
            contract,
            validated_python,
            runner=runner,
        )
        _require_clean_repository_status(layout, runner=runner)
        receipt = _build_receipt(
            contract,
            validated_commit,
            destination.name,
            repository_transport,
            combined,
        )
        _write_receipt_exclusive(receipt_output, receipt)
        return receipt
    except BaseException:
        _remove_owned_destination(destination)
        raise


def _internal_validate_clean_package(args: argparse.Namespace) -> int:
    loaded = repro_contract.load_and_validate_preregistration(
        args.preregistration, require_frozen=True
    )
    contract = _contract_from_preregistration(loaded.data, loaded.sha256)
    repository_root = args.repository_root.resolve(strict=True)
    adapter_root = args.adapter_root.resolve(strict=True)
    manifest_sources = repro_contract.load_manifest_source_bundle(
        repository_root=repository_root,
        adapter_root=adapter_root,
    )
    lineage = loaded.data["source_lineage"]
    manifest_payload = _read_lineage_payload(
        repository_root, lineage["manifest"], "manifest"
    )
    reference_predictions_payload = _read_lineage_payload(
        repository_root,
        lineage["reference_predictions"],
        "reference_predictions",
    )
    reference_evidence_payload = _read_lineage_payload(
        repository_root,
        lineage["reference_evidence"],
        "reference_evidence",
    )
    evaluation_payload = _read_lineage_payload(
        repository_root, lineage["evaluation"], "evaluation"
    )
    authenticated = repro_contract.authenticate_manifest_and_references(
        loaded.data,
        manifest_payload=manifest_payload,
        reference_predictions_payload=reference_predictions_payload,
        reference_evidence_payload=reference_evidence_payload,
        evaluation_payload=evaluation_payload,
        manifest_sources=manifest_sources,
    )
    resolution = repro_contract.resolve_clean_roots(
        loaded.data,
        authenticated,
        manifest_sources,
        base_model_root=args.base_model_root.resolve(strict=True),
        adapter_root=adapter_root,
        repository_root=repository_root,
    )
    combined = {
        "validation": authenticated.manifest_validation,
        "resolution": resolution,
    }
    _validate_resolution(combined, contract)
    sys.stdout.buffer.write(
        json.dumps(
            combined,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return 0


def _read_lineage_payload(
    repository_root: Path, reference: Mapping[str, Any], label: str
) -> bytes:
    relative = _safe_relative_path(reference.get("path"), f"{label}.path")
    path = _path_under(repository_root, relative, label)
    return _read_bytes_once(path, label, max_bytes=repro_contract.MAX_JSON_BYTES)


def _build_plan(
    contract: MaterializationContract,
    destination: Path,
    receipt_output: Path,
    freeze_commit: str,
) -> dict[str, Any]:
    return {
        "plan_only": True,
        "evidence": False,
        "gate_id": contract.gate_id,
        "freeze_commit": freeze_commit,
        "destination_role": "ephemeral_clean_location_guid_root",
        "receipt_role": "exclusive_materialization_receipt",
        "steps": [
            "fresh_https_git_fetch_detached_commit",
            "exact_single_adapter_lfs_pull",
            "manifest_bound_huggingface_download_with_isolated_cache",
            "clean_repository_manifest_resolution_9_3_15",
            "exclusive_receipt_then_offline_execution",
        ],
        "absolute_paths_omitted": True,
        "remote_revision_origin_attested": False,
        "destination_initially_absent": not os.path.lexists(destination),
        "receipt_initially_absent": not os.path.lexists(receipt_output),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--internal-validate-clean-package",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--base-model-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--adapter-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repository-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _require_cli_value(value: object, name: str) -> Any:
    if value is None:
        _fail("MISSING_CLI_ARGUMENT", name)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.preregistration = args.preregistration.resolve(strict=True)
    if args.internal_validate_clean_package:
        for name in ("base_model_root", "adapter_root", "repository_root"):
            _require_cli_value(getattr(args, name), name)
        return _internal_validate_clean_package(args)
    destination = Path(_require_cli_value(args.destination, "destination"))
    python_executable = Path(
        _require_cli_value(args.python_executable, "python_executable")
    )
    receipt_output = Path(_require_cli_value(args.receipt_output, "receipt_output"))
    freeze_commit = _validate_commit(
        _require_cli_value(args.freeze_commit, "freeze_commit"), "freeze_commit"
    )
    contract = _load_contract(args.preregistration)
    if args.plan:
        _validate_destination_path(destination)
        _validate_receipt_path(receipt_output, destination, create_parent=False)
        _validate_python_executable(python_executable)
        print(
            json.dumps(
                _build_plan(
                    contract,
                    destination,
                    receipt_output,
                    freeze_commit,
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    receipt = materialize(
        preregistration=args.preregistration,
        destination=destination,
        python_executable=python_executable,
        receipt_output=receipt_output,
        freeze_commit=freeze_commit,
    )
    print(
        json.dumps(
            {
                "materialized": True,
                "receipt_sha256": (
                    "sha256:"
                    + hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
                ),
                "remote_revision_origin_attested": False,
                "offline_execution_required": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
