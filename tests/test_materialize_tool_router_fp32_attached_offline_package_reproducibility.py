from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from fullcycle_bridge import (  # noqa: E402
    tool_router_fp32_attached_offline_package_reproducibility as contract_module,
)
from scripts import (  # noqa: E402
    materialize_tool_router_fp32_attached_offline_package_reproducibility as materializer,
)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _completed(stdout: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=b"")


_EXPECTED_LOCAL_LFS_GIT_PREFIX = [
    "git",
    "-c",
    "credential.helper=",
    "-c",
    "core.hooksPath=NUL",
    "-c",
    "filter.lfs.process=git-lfs filter-process --skip",
    "-c",
    "filter.lfs.required=true",
    "-c",
    "filter.lfs.clean=git-lfs clean -- %f",
    "-c",
    "filter.lfs.smudge=git-lfs smudge --skip -- %f",
]


class MaterializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temporary.name).resolve()
        self.controller_root = self.test_root / "controller"
        self.controller_root.mkdir()
        (self.controller_root / "work").mkdir()
        self.clean_parent = self.controller_root / "work" / "clean-location"
        self.receipt_parent = self.controller_root / "work" / "test-fixtures"
        self.constant_patch = patch.multiple(
            materializer,
            ROOT=self.controller_root,
            CLEAN_PARENT=self.clean_parent,
            RECEIPT_PARENT=self.receipt_parent,
        )
        self.constant_patch.start()
        self.destination = self.clean_parent / ("a" * 32)
        self.receipt = self.receipt_parent / "materialization.json"

    def tearDown(self) -> None:
        self.constant_patch.stop()
        self.temporary.cleanup()

    def _preregistration(self, *, frozen: bool = True) -> dict[str, object]:
        hashes = {
            name: "sha256:" + f"{index + 1:064x}"
            for index, name in enumerate(contract_module.PROTOCOL_SOURCE_PATHS)
        }
        return contract_module.expected_preregistration(
            freeze_status="frozen" if frozen else "draft",
            protocol_source_hashes=(
                hashes
                if frozen
                else {
                    name: contract_module.ZERO_SHA256
                    for name in contract_module.PROTOCOL_SOURCE_PATHS
                }
            ),
        )

    def _contract(
        self,
        *,
        preregistration_payload: bytes | None = None,
        protocol_sources: dict[str, dict[str, object]] | None = None,
        adapter_payload: bytes = b"adapter weights",
        downloader_payload: bytes = b"downloader",
    ) -> materializer.MaterializationContract:
        preregistration = self._preregistration()
        if protocol_sources is None:
            protocol_sources = {
                name: {
                    "path": record["path"],
                    "sha256": record["sha256"],
                }
                for name, record in preregistration["source_lineage"][
                    "protocol_sources"
                ].items()
            }
        if preregistration_payload is None:
            preregistration_payload = json.dumps(preregistration).encode("utf-8")
        return materializer.MaterializationContract(
            gate_id=contract_module.GATE_ID,
            experiment_id=contract_module.EXPERIMENT_ID,
            package_id=contract_module.PACKAGE_ID,
            preregistration_sha256=_sha(preregistration_payload),
            repository_remote_url=materializer.REPOSITORY_REMOTE_URL,
            manifest_relative_path=materializer.MANIFEST_RELATIVE_PATH,
            manifest_sha256=contract_module.MANIFEST_SHA256,
            downloader_relative_path=materializer.DOWNLOADER_RELATIVE_PATH,
            downloader_sha256=_sha(downloader_payload),
            adapter_lfs_relative_path=materializer.ADAPTER_LFS_RELATIVE_PATH,
            adapter_lfs_oid=_sha(adapter_payload),
            adapter_lfs_bytes=len(adapter_payload),
            protocol_sources=protocol_sources,
            phase_order=tuple(
                preregistration["materialization_protocol"]["phase_order"]
            ),
            destination_children=dict(
                preregistration["materialization_protocol"]["destination_policy"][
                    "children"
                ]
            ),
            preregistration=preregistration,
        )

    def _resolution(self) -> dict[str, object]:
        groups = [
            {
                "root_role": "base_model_and_tokenizer",
                "resolved": True,
                "expected_files": 9,
                "matched_files": 9,
                "matched_bytes": 3_098_971_928,
                "issues": [],
            },
            {
                "root_role": "adapter",
                "resolved": True,
                "expected_files": 3,
                "matched_files": 3,
                "matched_bytes": 17_468_332,
                "issues": [],
            },
            {
                "root_role": "repository",
                "resolved": True,
                "expected_files": 15,
                "matched_files": 15,
                "matched_bytes": 123_456,
                "issues": [],
            },
        ]
        resolution: dict[str, object] = {
            "resolution_version": 1,
            "package_id": contract_module.PACKAGE_ID,
            "manifest_file_sha256": contract_module.MANIFEST_SHA256,
            "caller_supplied_roots": True,
            "manifest_machine_paths_used": False,
            "adapter_local_base_path_used": False,
            "resolved": True,
            "eligible_for_clean_location_reproducibility_test": True,
            "offline_artifact_eligible": False,
            "runtime_eligible": False,
            "groups": groups,
            "failure_mode": None,
        }
        resolution["resolution_digest"] = contract_module.sha256_bytes(
            contract_module.canonical_json_bytes(resolution)
        )
        return resolution

    def test_frozen_contract_load_uses_core_validator(self) -> None:
        preregistration = self._preregistration()
        path = self.test_root / "preregistration.json"
        path.write_text(json.dumps(preregistration), encoding="utf-8")
        loaded = materializer._load_contract(path.resolve())
        self.assertEqual(loaded.gate_id, contract_module.GATE_ID)
        self.assertEqual(loaded.preregistration_sha256, _sha(path.read_bytes()))

    def test_draft_preregistration_cannot_authorize_execution(self) -> None:
        path = self.test_root / "draft.json"
        path.write_text(
            json.dumps(self._preregistration(frozen=False)), encoding="utf-8"
        )
        with self.assertRaises(contract_module.ReproducibilityContractError) as caught:
            materializer._load_contract(path.resolve())
        self.assertEqual(caught.exception.code, "PREREGISTRATION_NOT_FROZEN")

    def test_destination_must_be_absent_direct_32_hex_child(self) -> None:
        materializer._ensure_owned_parent(
            self.clean_parent,
            root=self.controller_root,
            label="clean_parent",
        )
        materializer._validate_destination_path(self.destination)
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._validate_destination_path(self.clean_parent / "not-a-guid")
        self.assertEqual(caught.exception.code, "DESTINATION_NOT_GUID")
        self.destination.mkdir()
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._validate_destination_path(self.destination)
        self.assertEqual(caught.exception.code, "DESTINATION_ALREADY_EXISTS")

    def test_receipt_must_be_absent_direct_test_fixture_child(self) -> None:
        materializer._validate_receipt_path(self.receipt, self.destination)
        self.receipt.write_bytes(b"existing")
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._validate_receipt_path(self.receipt, self.destination)
        self.assertEqual(caught.exception.code, "RECEIPT_ALREADY_EXISTS")

    def test_hugging_face_environment_is_destination_scoped(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "secret",
                "HUGGING_FACE_HUB_TOKEN": "secret",
            },
            clear=False,
        ):
            observed = materializer._hf_environment(self.destination)
        self.assertNotIn("HF_TOKEN", observed)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", observed)
        self.assertEqual(observed["HF_ENDPOINT"], "https://huggingface.co")
        self.assertEqual(observed["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertTrue(
            materializer._is_within(Path(observed["HF_HOME"]), self.destination)
        )

    def test_git_environment_disables_smudge_and_external_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "url.file.insteadOf",
                "GIT_CONFIG_VALUE_0": "https://github.com/",
                "GIT_CONFIG_PARAMETERS": "'core.fsmonitor=unsafe'",
                "GIT_EXEC_PATH": "unsafe",
            },
            clear=False,
        ):
            observed = materializer._git_environment()
        self.assertEqual(observed["GIT_LFS_SKIP_SMUDGE"], "1")
        self.assertEqual(observed["GIT_CONFIG_GLOBAL"], "NUL")
        self.assertEqual(observed["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(observed["GIT_TERMINAL_PROMPT"], "0")
        self.assertNotIn("GIT_CONFIG_COUNT", observed)
        self.assertNotIn("GIT_CONFIG_KEY_0", observed)
        self.assertNotIn("GIT_CONFIG_VALUE_0", observed)
        self.assertNotIn("GIT_CONFIG_PARAMETERS", observed)
        self.assertNotIn("GIT_EXEC_PATH", observed)

    def test_lfs_pointer_is_exact(self) -> None:
        payload = b"weights"
        contract = self._contract(adapter_payload=payload)
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid {contract.adapter_lfs_oid}\n"
            f"size {len(payload)}\n"
        ).encode("ascii")
        materializer._validate_lfs_pointer(pointer, contract)
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._validate_lfs_pointer(pointer + b"extra\n", contract)
        self.assertEqual(caught.exception.code, "ADAPTER_LFS_POINTER_MISMATCH")

    def test_repository_transport_fetches_exact_commit_and_one_lfs_object(self) -> None:
        materializer._create_destination(self.destination)
        layout = materializer._build_layout(self.destination)
        preregistration_payload = b"frozen preregistration"
        source_payloads = {
            "contract_source": b"contract",
            "materializer_source": b"materializer",
            "runner_source": b"runner",
        }
        protocol_sources = {
            name: {
                "path": contract_module.PROTOCOL_SOURCE_PATHS[name],
                "sha256": _sha(payload),
            }
            for name, payload in source_payloads.items()
        }
        adapter_payload = b"adapter weights"
        pointer_payload = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid {_sha(adapter_payload)}\n"
            f"size {len(adapter_payload)}\n"
        ).encode("ascii")
        contract = self._contract(
            preregistration_payload=preregistration_payload,
            protocol_sources=protocol_sources,
            adapter_payload=adapter_payload,
        )
        freeze_commit = "1" * 40
        commands: list[tuple[list[str], dict[str, str]]] = []

        def runner(
            command: object, *, cwd: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            del cwd
            argv = list(command)
            commands.append((argv, dict(env)))
            if "init" in argv:
                (layout.repository / ".git").mkdir(parents=True)
            elif "checkout" in argv and "lfs" not in argv:
                for name, source in protocol_sources.items():
                    path = layout.repository / str(source["path"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(source_payloads[name])
                preregistration = (
                    layout.repository / contract_module.PREREGISTRATION_PATH
                )
                preregistration.parent.mkdir(parents=True, exist_ok=True)
                preregistration.write_bytes(preregistration_payload)
                weight = layout.repository / contract.adapter_lfs_relative_path
                weight.parent.mkdir(parents=True, exist_ok=True)
                weight.write_bytes(pointer_payload)
            elif "pull" in argv:
                lfs_object = (
                    layout.repository
                    / ".git"
                    / "lfs"
                    / "objects"
                    / "aa"
                    / "bb"
                    / contract.adapter_lfs_oid.removeprefix("sha256:")
                )
                lfs_object.parent.mkdir(parents=True, exist_ok=True)
                lfs_object.write_bytes(adapter_payload)
            elif "checkout" in argv and "lfs" in argv:
                weight = layout.repository / contract.adapter_lfs_relative_path
                self.assertEqual(weight.read_bytes(), pointer_payload)
                weight.write_bytes(adapter_payload)
            if "get-url" in argv:
                return _completed(contract.repository_remote_url.encode())
            if "rev-parse" in argv:
                return _completed(freeze_commit.encode())
            if "show" in argv:
                return _completed(pointer_payload)
            if "--version" in argv:
                return _completed(b"git version test")
            if "version" in argv and "lfs" in argv:
                return _completed(b"git-lfs/test")
            return _completed()

        receipt = materializer._materialize_repository(
            layout,
            contract,
            freeze_commit,
            runner=runner,
        )
        fetch = next(item for item in commands if "fetch" in item[0])
        pull = [item for item in commands if "pull" in item[0]]
        checkout = [
            item for item in commands if "lfs" in item[0] and "checkout" in item[0]
        ]
        self.assertIn(freeze_commit, fetch[0])
        self.assertEqual(fetch[1]["GIT_LFS_SKIP_SMUDGE"], "1")
        self.assertEqual(len(pull), 1)
        self.assertEqual(len(checkout), 1)
        self.assertLess(commands.index(pull[0]), commands.index(checkout[0]))
        self.assertIn(f"--include={materializer.ADAPTER_LFS_RELATIVE_PATH}", pull[0][0])
        self.assertEqual(
            checkout[0][0][-3:],
            ["lfs", "checkout", materializer.ADAPTER_LFS_RELATIVE_PATH],
        )
        self.assertEqual(
            checkout[0][0][: len(_EXPECTED_LOCAL_LFS_GIT_PREFIX)],
            _EXPECTED_LOCAL_LFS_GIT_PREFIX,
        )
        self.assertNotIn("GIT_LFS_SKIP_SMUDGE", pull[0][1])
        self.assertNotIn("GIT_LFS_SKIP_SMUDGE", checkout[0][1])
        self.assertEqual(receipt["lfs_objects_requested"], 1)

    def test_local_lfs_checkout_targets_only_frozen_adapter_path(self) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        contract = self._contract()
        observed: dict[str, object] = {}
        environment = materializer._git_environment()
        environment.pop("GIT_LFS_SKIP_SMUDGE", None)

        def runner(
            command: object, *, cwd: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            observed.update(command=list(command), cwd=cwd, env=dict(env))
            return _completed()

        materializer._checkout_adapter_from_local_lfs(
            layout,
            contract,
            environment=environment,
            runner=runner,
        )
        self.assertEqual(
            observed["command"][-3:],
            ["lfs", "checkout", contract.adapter_lfs_relative_path],
        )
        self.assertEqual(
            observed["command"][: len(_EXPECTED_LOCAL_LFS_GIT_PREFIX)],
            _EXPECTED_LOCAL_LFS_GIT_PREFIX,
        )
        self.assertEqual(observed["cwd"], layout.destination)
        self.assertEqual(observed["env"], environment)

    def test_local_lfs_checkout_hydrates_verified_object_with_isolated_config(
        self,
    ) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        layout.repository.mkdir()
        adapter_payload = b"locally verified adapter payload"
        contract = self._contract(adapter_payload=adapter_payload)
        pointer_payload = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid {contract.adapter_lfs_oid}\n"
            f"size {contract.adapter_lfs_bytes}\n"
        ).encode("ascii")
        environment = materializer._git_environment()
        environment.pop("GIT_LFS_SKIP_SMUDGE", None)

        def run_git(*arguments: str) -> None:
            completed = materializer._run_command(
                [
                    *materializer._clean_git_command(),
                    "-C",
                    str(layout.repository),
                    *arguments,
                ],
                cwd=layout.destination,
                env=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr.decode("utf-8", errors="replace"),
            )

        run_git("init", "--quiet")
        weight = layout.repository / contract.adapter_lfs_relative_path
        weight.parent.mkdir(parents=True)
        weight.write_bytes(pointer_payload)
        run_git("add", "--", contract.adapter_lfs_relative_path)
        attributes = layout.repository / ".gitattributes"
        attributes.write_text(
            f"{contract.adapter_lfs_relative_path} filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
        run_git("add", "--", ".gitattributes")
        run_git(
            "-c",
            "user.name=Local Fixture",
            "-c",
            "user.email=fixture.invalid@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "local LFS fixture",
        )
        object_oid = contract.adapter_lfs_oid.removeprefix("sha256:")
        local_object = (
            layout.repository
            / ".git"
            / "lfs"
            / "objects"
            / object_oid[:2]
            / object_oid[2:4]
            / object_oid
        )
        local_object.parent.mkdir(parents=True)
        local_object.write_bytes(adapter_payload)
        materializer._verify_single_lfs_object(layout.repository, contract)
        self.assertEqual(weight.read_bytes(), pointer_payload)

        def run_checkout(
            command: object,
            *,
            cwd: Path,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[bytes]:
            completed = materializer._run_command(
                list(command),
                cwd=cwd,
                env=env,
            )
            logs = sorted((layout.repository / ".git" / "lfs" / "logs").glob("*"))
            diagnostic = completed.stderr.decode("utf-8", errors="replace")
            if logs:
                diagnostic += "\n" + logs[-1].read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            self.assertEqual(
                completed.returncode,
                0,
                diagnostic,
            )
            return completed

        materializer._checkout_adapter_from_local_lfs(
            layout,
            contract,
            environment=environment,
            runner=run_checkout,
        )

        self.assertEqual(weight.read_bytes(), adapter_payload)
        materializer._require_clean_repository_status(
            layout,
            runner=materializer._run_command,
        )

    def test_clean_status_rejects_ignored_entries_with_isolated_git(self) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        layout.repository.mkdir()
        observed: dict[str, object] = {}

        def runner(
            command: object, *, cwd: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            observed.update(command=list(command), cwd=cwd, env=dict(env))
            return _completed(b"!! src/torch.pyc\n")

        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.fsmonitor",
                "GIT_CONFIG_VALUE_0": "unsafe",
                "GIT_CONFIG_PARAMETERS": "'core.fsmonitor=unsafe'",
            },
            clear=False,
        ):
            with self.assertRaises(materializer.MaterializationError) as caught:
                materializer._require_clean_repository_status(layout, runner=runner)
        self.assertEqual(caught.exception.code, "CLEAN_REPOSITORY_DIRTY")
        self.assertIn("--ignored", observed["command"])
        self.assertEqual(
            observed["command"][: len(_EXPECTED_LOCAL_LFS_GIT_PREFIX)],
            _EXPECTED_LOCAL_LFS_GIT_PREFIX,
        )
        self.assertEqual(observed["env"]["GIT_CONFIG_GLOBAL"], "NUL")
        self.assertEqual(observed["env"]["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertNotIn("GIT_LFS_SKIP_SMUDGE", observed["env"])
        self.assertNotIn("GIT_CONFIG_COUNT", observed["env"])
        self.assertNotIn("GIT_CONFIG_KEY_0", observed["env"])
        self.assertNotIn("GIT_CONFIG_VALUE_0", observed["env"])
        self.assertNotIn("GIT_CONFIG_PARAMETERS", observed["env"])

    def test_base_transport_invokes_only_clean_manifest_bound_downloader(self) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        layout.repository.mkdir()
        downloader = layout.repository / materializer.DOWNLOADER_RELATIVE_PATH
        downloader.parent.mkdir(parents=True)
        downloader_payload = b"downloader"
        downloader.write_bytes(downloader_payload)
        contract = self._contract(downloader_payload=downloader_payload)
        observed: dict[str, object] = {}

        def runner(
            command: object, *, cwd: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            observed.update(command=list(command), cwd=cwd, env=dict(env))
            layout.base_model_and_tokenizer.mkdir()
            return _completed(b"verified_model=C:\\redacted\n")

        result = materializer._materialize_base(
            layout,
            contract,
            Path(sys.executable),
            runner=runner,
        )
        self.assertEqual(observed["command"][2], str(downloader))
        self.assertEqual(observed["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertTrue(result["isolated_hf_home"])
        self.assertNotIn("C:\\redacted", json.dumps(result))

    def test_clean_validation_disables_child_bytecode_writes(self) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        layout.repository.mkdir()
        preregistration_payload = b"frozen preregistration"
        clean_preregistration = layout.repository / contract_module.PREREGISTRATION_PATH
        clean_preregistration.parent.mkdir(parents=True)
        clean_preregistration.write_bytes(preregistration_payload)
        contract = self._contract(
            preregistration_payload=preregistration_payload,
        )
        combined = {
            "validation": {
                "frozen_manifest_valid": True,
                "manifest_file_sha256": contract.manifest_sha256,
                "metadata_complete": True,
                "offline_package_identity_complete": True,
                "remote_revision_origin_attested": False,
            },
            "resolution": self._resolution(),
        }
        observed: dict[str, object] = {}

        def runner(
            command: object, *, cwd: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            observed.update(command=list(command), cwd=cwd, env=dict(env))
            return _completed(
                json.dumps(combined, separators=(",", ":")).encode("utf-8")
            )

        with patch.object(materializer, "_reject_component_links"):
            materializer._validate_clean_package(
                layout,
                contract,
                Path(sys.executable),
                runner=runner,
            )
        self.assertEqual(observed["command"][1:3], ["-I", "-B"])
        self.assertEqual(observed["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(observed["env"]["HF_HUB_OFFLINE"], "1")

    def test_receipt_matches_core_schema_and_contains_no_absolute_path(self) -> None:
        preregistration = self._preregistration()
        preregistration_payload = json.dumps(preregistration).encode("utf-8")
        contract = materializer._contract_from_preregistration(
            preregistration, _sha(preregistration_payload)
        )
        protocol_sources = {
            name: {
                "path": record["path"],
                "sha256": record["sha256"],
                "bytes": 1,
            }
            for name, record in preregistration["source_lineage"][
                "protocol_sources"
            ].items()
        }
        resolution = self._resolution()
        receipt = materializer._build_receipt(
            contract,
            "1" * 40,
            "a" * 32,
            {"protocol_sources": protocol_sources},
            {"resolution": resolution},
        )
        self.assertEqual(receipt["destination"]["destination_id"], "a" * 32)
        self.assertFalse(materializer._contains_absolute_path(receipt))
        self.assertNotIn("remote_revision_origin_attested", receipt)
        self.assertEqual(receipt["clean_groups"], resolution["groups"])

    def test_receipt_write_is_exclusive(self) -> None:
        materializer._validate_receipt_path(self.receipt, self.destination)
        value = {"relative": "repository", "sha256": "sha256:" + "a" * 64}
        materializer._write_receipt_exclusive(self.receipt, value)
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._write_receipt_exclusive(self.receipt, value)
        self.assertEqual(caught.exception.code, "RECEIPT_CREATE_RACE")

    def test_hardlinked_component_is_rejected(self) -> None:
        self.destination.mkdir(parents=True)
        layout = materializer._build_layout(self.destination)
        layout.base_model_and_tokenizer.mkdir()
        first_name = materializer.BASE_MODEL_FILE_SPECS[0][0]
        source = layout.base_model_and_tokenizer / first_name
        source.write_bytes(b"source")
        linked = layout.base_model_and_tokenizer / "linked"
        try:
            os.link(source, linked)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._reject_component_links(layout)
        self.assertEqual(caught.exception.code, "HARDLINK_FORBIDDEN")

    def test_failure_removes_only_owned_guid_destination(self) -> None:
        preregistration = self.test_root / "preregistration.json"
        preregistration.write_bytes(b"placeholder")
        contract = self._contract()
        with (
            patch.object(materializer, "_load_contract", return_value=contract),
            patch.object(
                materializer,
                "_validate_python_executable",
                return_value=Path(sys.executable),
            ),
            patch.object(
                materializer,
                "_materialize_repository",
                side_effect=materializer.MaterializationError("EXPECTED_FAILURE"),
            ),
        ):
            with self.assertRaises(materializer.MaterializationError) as caught:
                materializer.materialize(
                    preregistration=preregistration,
                    destination=self.destination,
                    python_executable=Path(sys.executable),
                    receipt_output=self.receipt,
                    freeze_commit="1" * 40,
                )
        self.assertEqual(caught.exception.code, "EXPECTED_FAILURE")
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())

    def test_cleanup_removes_nested_read_only_regular_file(self) -> None:
        read_only = (
            self.destination
            / "repository"
            / ".git"
            / "objects"
            / "pack"
            / "frozen-pack.idx"
        )
        read_only.parent.mkdir(parents=True)
        read_only.write_bytes(b"read-only pack index")
        read_only.chmod(stat.S_IREAD)
        self.assertTrue(
            getattr(read_only.lstat(), "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_READONLY
        )
        try:
            materializer._remove_owned_destination(self.destination)
        finally:
            if read_only.exists():
                read_only.chmod(stat.S_IWRITE)
        self.assertFalse(self.destination.exists())

    def test_cleanup_rejects_regular_file_with_external_hardlink(self) -> None:
        regular_file = self.destination / "repository" / "tracked.bin"
        regular_file.parent.mkdir(parents=True)
        regular_file.write_bytes(b"shared bytes")
        external_link = self.test_root / "external-hardlink.bin"
        try:
            os.link(regular_file, external_link)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")

        with self.assertRaises(materializer.MaterializationError) as caught:
            materializer._remove_owned_destination(self.destination)

        self.assertEqual(caught.exception.code, "CLEANUP_FILE_CHANGED")
        self.assertTrue(regular_file.exists())
        self.assertTrue(external_link.exists())

    @unittest.skipUnless(os.name == "nt", "Win32 handle semantics")
    def test_readonly_handle_rejects_identity_mismatch_before_mutation(self) -> None:
        read_only = self.test_root / "identity-mismatch.idx"
        read_only.write_bytes(b"identity-bound bytes")
        read_only.chmod(stat.S_IREAD)
        metadata = read_only.lstat()
        wrong_identity = (int(metadata.st_dev), int(metadata.st_ino) + 1)
        try:
            with self.assertRaises(materializer.MaterializationError) as caught:
                materializer._remove_windows_readonly_file_by_handle(
                    read_only,
                    wrong_identity,
                )
            self.assertEqual(caught.exception.code, "CLEANUP_FILE_CHANGED")
            self.assertTrue(read_only.exists())
            self.assertTrue(
                getattr(read_only.lstat(), "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_READONLY
            )
        finally:
            if read_only.exists():
                read_only.chmod(stat.S_IWRITE)
                read_only.unlink()

    @unittest.skipUnless(os.name == "nt", "Win32 handle semantics")
    def test_cleanup_handles_block_owned_entry_replacement(self) -> None:
        directory = self.destination / "repository"
        directory.mkdir(parents=True)
        directory_metadata = directory.lstat()
        directory_descriptor = materializer._open_windows_cleanup_directory_handle(
            directory,
            materializer._cleanup_identity(directory_metadata),
        )
        moved_directory = self.test_root / "moved-repository"
        try:
            with self.assertRaises(OSError):
                directory.rename(moved_directory)
        finally:
            os.close(directory_descriptor)

        regular_file = directory / "tracked.bin"
        regular_file.write_bytes(b"identity-bound bytes")
        file_descriptor = materializer._open_windows_cleanup_handle(
            regular_file,
            desired_access=0x00010000 | 0x0080 | 0x0100,
            flags=0x00200000,
            failure_code="TEST_HANDLE_FAILED",
        )
        moved_file = self.test_root / "moved-tracked.bin"
        external_link = self.test_root / "concurrent-hardlink.bin"
        try:
            with self.assertRaises(OSError):
                regular_file.rename(moved_file)
            os.link(regular_file, external_link)
            self.assertEqual(regular_file.lstat().st_nlink, 2)
        finally:
            os.close(file_descriptor)

    def test_final_clean_status_runs_after_validation_before_receipt(self) -> None:
        preregistration = self.test_root / "preregistration.json"
        preregistration.write_bytes(b"placeholder")
        contract = self._contract()
        resolution = self._resolution()
        events: list[str] = []

        def validation(*_args: object, **_kwargs: object) -> dict[str, object]:
            events.append("internal_validation")
            return {"resolution": resolution}

        def clean_status(*_args: object, **_kwargs: object) -> None:
            events.append("final_clean_status")

        def build_receipt(*_args: object, **_kwargs: object) -> dict[str, object]:
            events.append("build_receipt")
            return {"materialization_passed": True}

        def write_receipt(*_args: object, **_kwargs: object) -> None:
            events.append("write_receipt")

        with (
            patch.object(materializer, "_load_contract", return_value=contract),
            patch.object(
                materializer,
                "_validate_python_executable",
                return_value=Path(sys.executable),
            ),
            patch.object(
                materializer,
                "_materialize_repository",
                return_value={"protocol_sources": {}},
            ),
            patch.object(materializer, "_materialize_base"),
            patch.object(
                materializer,
                "_validate_clean_package",
                side_effect=validation,
            ),
            patch.object(
                materializer,
                "_require_clean_repository_status",
                side_effect=clean_status,
            ),
            patch.object(materializer, "_build_receipt", side_effect=build_receipt),
            patch.object(
                materializer,
                "_write_receipt_exclusive",
                side_effect=write_receipt,
            ),
        ):
            observed = materializer.materialize(
                preregistration=preregistration,
                destination=self.destination,
                python_executable=Path(sys.executable),
                receipt_output=self.receipt,
                freeze_commit="1" * 40,
            )
        self.assertEqual(observed, {"materialization_passed": True})
        self.assertEqual(
            events,
            [
                "internal_validation",
                "final_clean_status",
                "build_receipt",
                "write_receipt",
            ],
        )

    def test_plan_does_not_invoke_transport(self) -> None:
        preregistration = self._preregistration()
        preregistration_path = self.test_root / "preregistration.json"
        preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
        with (
            patch.object(materializer, "_run_command") as transport,
            redirect_stdout(StringIO()) as output,
        ):
            result = materializer.main(
                [
                    "--preregistration",
                    str(preregistration_path),
                    "--destination",
                    str(self.destination),
                    "--python-executable",
                    sys.executable,
                    "--receipt-output",
                    str(self.receipt),
                    "--freeze-commit",
                    "1" * 40,
                    "--plan",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(transport.called)
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["plan_only"])
        self.assertFalse(plan["evidence"])
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.receipt_parent.exists())


if __name__ == "__main__":
    unittest.main()
