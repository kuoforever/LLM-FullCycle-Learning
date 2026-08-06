from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fullcycle_bridge.tool_router import ToolRouterValidationError
from fullcycle_bridge import (
    tool_router_fp32_attached_offline_package_manifest as contract,
)
from fullcycle_bridge.tool_router_fp32_attached_offline_package_manifest import (
    ADAPTER_FILE_SPECS,
    BASE_MODEL_FILE_SPECS,
    REPOSITORY_SOURCE_PATHS,
    TOKENIZER_FILE_SPECS,
    build_fp32_attached_offline_package_manifest,
    expected_file_records,
    resolve_component_files,
    validate_and_resolve_fp32_attached_offline_package,
    validate_fp32_attached_offline_package_manifest,
)
from scripts import build_tool_router_fp32_attached_offline_package_manifest as builder
from scripts import validate_offline as offline_gate
from scripts.build_tool_router_fp32_attached_offline_package_manifest import (
    DEFAULT_ADAPTER_DIR,
    OUTPUT,
    ROOT,
    load_repository_manifest_inputs,
)


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


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class OfflinePackageManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_repository_manifest_inputs()
        cls.manifest = build_fp32_attached_offline_package_manifest(
            cls.inputs["upstream_review"],
            cls.inputs["remediation_preregistration"],
            cls.inputs["sft_config"],
            cls.inputs["adapter_config"],
            base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
            tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
            adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
            source_hashes=cls.inputs["source_hashes"],
            source_payloads=cls.inputs["source_payloads"],
        )
        cls.manifest_payload = _payload(cls.manifest)
        cls.manifest_sha256 = _sha256(cls.manifest_payload)

    def _validate(
        self,
        *,
        manifest_payload: bytes | None = None,
        expected_manifest_sha256: str | None = None,
        inputs: dict[str, object] | None = None,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> dict[str, object]:
        values = self.inputs if inputs is None else inputs
        payload = (
            self.manifest_payload if manifest_payload is None else manifest_payload
        )
        manifest_sha256 = (
            self.manifest_sha256
            if expected_manifest_sha256 is None
            else expected_manifest_sha256
        )
        roots = (
            values["source_hashes"]
            if expected_source_hashes is None
            else expected_source_hashes
        )
        return validate_fp32_attached_offline_package_manifest(
            payload,
            manifest_sha256,
            values["upstream_review"],
            values["remediation_preregistration"],
            values["sft_config"],
            values["adapter_config"],
            source_hashes=values["source_hashes"],
            source_payloads=values["source_payloads"],
            expected_source_hashes=roots,
        )

    def _assert_code(self, expected: str, callback: object) -> None:
        with self.assertRaises(ToolRouterValidationError) as raised:
            callback()  # type: ignore[operator]
        self.assertEqual(raised.exception.code, expected)

    def test_valid_manifest_derives_safe_decision(self) -> None:
        self.assertEqual(
            self._validate(),
            {
                "frozen_manifest_valid": True,
                "manifest_file_sha256": self.manifest_sha256,
                "metadata_complete": True,
                "offline_package_identity_complete": True,
                "attached_package_identity_bound": True,
                "prior_package_blocker_count_resolved": 6,
                "eligible_for_clean_location_reproducibility_test": True,
                "remote_revision_origin_attested": False,
                "behavioral_reproducibility_established": False,
                "offline_artifact_eligible": False,
                "portable_package_eligible": False,
                "preferred_offline_candidate": False,
                "serving_readiness_established": False,
                "artifact_promotion_allowed": False,
                "merged_artifact_allowed": False,
                "classification": (
                    "fp32_attached_metadata_only_composite_manifest_complete"
                ),
                "remaining_blocking_findings": [
                    "behavioral_reproducibility_unverified",
                    "clean_location_resolution_unverified",
                    "remote_revision_origin_unverified",
                ],
                "remaining_blocking_finding_count": 3,
                "next_gate": (
                    "FC-MVP-001-fp32-attached-offline-package-reproducibility-v1"
                ),
                "runtime_eligible": False,
            },
        )

    def test_frozen_manifest_matches_external_hash_and_rebuild(self) -> None:
        payload = OUTPUT.read_bytes()
        self.assertEqual(len(payload), 17_487)
        self.assertEqual(
            _sha256(payload),
            "sha256:4125f2eef2a4b8f07015169ac7fb77b830514e053a4624aa703e5f5a64943eb0",
        )
        self.assertEqual(payload, self.manifest_payload)
        self.assertEqual(
            self._validate(
                manifest_payload=payload,
                expected_manifest_sha256=_sha256(payload),
            )["classification"],
            "fp32_attached_metadata_only_composite_manifest_complete",
        )

    def test_manifest_contains_facts_not_self_authorized_decisions(self) -> None:
        self.assertNotIn("eligibility_decision", self.manifest)
        self.assertNotIn("report_digest", self.manifest)
        self.assertNotIn("runtime_eligible", self.manifest)
        self.assertNotIn("locked_next_action", self.manifest)
        self.assertEqual(
            self.manifest["artifact_kind"],
            "external_metadata_only_composite_manifest",
        )
        self.assertNotIn(str(ROOT), self.manifest_payload.decode("utf-8"))

    def test_manifest_hash_is_checked_before_content(self) -> None:
        forged = copy.deepcopy(self.manifest)
        forged["eligible"] = True
        forged_payload = _payload(forged)
        self._assert_code(
            "MANIFEST_FILE_HASH_MISMATCH",
            lambda: self._validate(manifest_payload=forged_payload),
        )

    def test_resealed_forgery_cannot_authorize_itself(self) -> None:
        forged = copy.deepcopy(self.manifest)
        forged["eligible"] = True
        forged_payload = _payload(forged)
        self._assert_code(
            "MANIFEST_RECOMPUTATION_MISMATCH",
            lambda: self._validate(
                manifest_payload=forged_payload,
                expected_manifest_sha256=_sha256(forged_payload),
            ),
        )

    def test_resolution_reauthenticates_manifest_and_freezes_repository_roots(
        self,
    ) -> None:
        forged = copy.deepcopy(self.manifest)
        forged["package_id"] = "attacker-controlled-package-id"
        forged["source_artifacts"] = {}
        forged["resolution_contract"]["repository_source_paths"] = {}
        forged_payload = _payload(forged)
        self._assert_code(
            "MANIFEST_RECOMPUTATION_MISMATCH",
            lambda: validate_and_resolve_fp32_attached_offline_package(
                forged_payload,
                _sha256(forged_payload),
                self.inputs["upstream_review"],
                self.inputs["remediation_preregistration"],
                self.inputs["sft_config"],
                self.inputs["adapter_config"],
                source_hashes=self.inputs["source_hashes"],
                source_payloads=self.inputs["source_payloads"],
                expected_source_hashes=self.inputs["source_hashes"],
                base_model_root=ROOT / "work" / "models" / "Qwen2.5-1.5B-Instruct",
                adapter_root=DEFAULT_ADAPTER_DIR,
                repository_root=ROOT,
            ),
        )
        self.assertEqual(
            self.manifest["resolution_contract"]["repository_source_paths"],
            dict(sorted(REPOSITORY_SOURCE_PATHS.items())),
        )
        self.assertGreater(len(REPOSITORY_SOURCE_PATHS), 0)

    def test_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        duplicate = b'{"manifest_version":1,"manifest_version":1}\n'
        self._assert_code(
            "INVALID_SOURCE_JSON",
            lambda: self._validate(
                manifest_payload=duplicate,
                expected_manifest_sha256=_sha256(duplicate),
            ),
        )
        nonfinite = b'{"manifest_version":NaN}\n'
        self._assert_code(
            "INVALID_SOURCE_JSON",
            lambda: self._validate(
                manifest_payload=nonfinite,
                expected_manifest_sha256=_sha256(nonfinite),
            ),
        )

    def test_external_source_roots_cannot_be_resealed(self) -> None:
        values = copy.deepcopy(self.inputs)
        payload = values["source_payloads"]["manifest_contract_source"] + b"\n"
        values["source_payloads"]["manifest_contract_source"] = payload
        values["source_hashes"]["manifest_contract_source"] = _sha256(payload)
        self._assert_code(
            "SOURCE_HASH_ROOT_MISMATCH",
            lambda: self._validate(
                inputs=values,
                expected_source_hashes=self.inputs["source_hashes"],
            ),
        )

    def test_parsed_object_cannot_bypass_source_payload(self) -> None:
        values = copy.deepcopy(self.inputs)
        values["adapter_config"]["revision"] = "forged"
        self._assert_code(
            "SOURCE_OBJECT_PAYLOAD_MISMATCH",
            lambda: build_fp32_attached_offline_package_manifest(
                values["upstream_review"],
                values["remediation_preregistration"],
                values["sft_config"],
                values["adapter_config"],
                base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                source_hashes=values["source_hashes"],
                source_payloads=values["source_payloads"],
            ),
        )

    def test_source_payload_hash_is_independent_of_object(self) -> None:
        values = copy.deepcopy(self.inputs)
        values["source_payloads"]["manifest_builder_source"] += b"\n"
        self._assert_code(
            "SOURCE_PAYLOAD_HASH_MISMATCH",
            lambda: build_fp32_attached_offline_package_manifest(
                values["upstream_review"],
                values["remediation_preregistration"],
                values["sft_config"],
                values["adapter_config"],
                base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                source_hashes=values["source_hashes"],
                source_payloads=values["source_payloads"],
            ),
        )

    def test_each_component_manifest_is_exact(self) -> None:
        cases = [
            (
                "BASE_MODEL_FILE_MANIFEST_MISMATCH",
                "base_model_files",
                expected_file_records(BASE_MODEL_FILE_SPECS)[:-1],
            ),
            (
                "TOKENIZER_FILE_MANIFEST_MISMATCH",
                "tokenizer_files",
                expected_file_records(TOKENIZER_FILE_SPECS)[:-1],
            ),
            (
                "ADAPTER_FILE_MANIFEST_MISMATCH",
                "adapter_files",
                expected_file_records(ADAPTER_FILE_SPECS)[:-1],
            ),
        ]
        for code, name, replacement in cases:
            with self.subTest(name=name):
                arguments = {
                    "base_model_files": expected_file_records(BASE_MODEL_FILE_SPECS),
                    "tokenizer_files": expected_file_records(TOKENIZER_FILE_SPECS),
                    "adapter_files": expected_file_records(ADAPTER_FILE_SPECS),
                }
                arguments[name] = replacement
                self._assert_code(
                    code,
                    lambda arguments=arguments: (
                        build_fp32_attached_offline_package_manifest(
                            self.inputs["upstream_review"],
                            self.inputs["remediation_preregistration"],
                            self.inputs["sft_config"],
                            self.inputs["adapter_config"],
                            source_hashes=self.inputs["source_hashes"],
                            source_payloads=self.inputs["source_payloads"],
                            **arguments,
                        )
                    ),
                )

    def test_generation_drift_is_rejected_after_source_authentication(self) -> None:
        values = copy.deepcopy(self.inputs)
        values["remediation_preregistration"]["protocol"]["generation"]["do_sample"] = (
            True
        )
        payload = _payload(values["remediation_preregistration"])
        digest = _sha256(payload)
        values["source_payloads"]["remediation_preregistration"] = payload
        values["source_hashes"]["remediation_preregistration"] = digest
        with mock.patch.dict(
            contract.STATIC_SOURCE_HASHES,
            {"remediation_preregistration": digest},
        ):
            self._assert_code(
                "EXECUTION_CONTRACT_MISMATCH",
                lambda: build_fp32_attached_offline_package_manifest(
                    values["upstream_review"],
                    values["remediation_preregistration"],
                    values["sft_config"],
                    values["adapter_config"],
                    base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                    tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                    adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                    source_hashes=values["source_hashes"],
                    source_payloads=values["source_payloads"],
                ),
            )

    def test_compiler_import_closure_is_bound(self) -> None:
        compiler = self.manifest["components"]["decision_compiler"]
        self.assertEqual(
            [item["path"] for item in compiler["direct_dependencies"]],
            [
                "src/fullcycle_bridge/__init__.py",
                "src/fullcycle_bridge/consumer.py",
                "src/fullcycle_bridge/tool_router.py",
            ],
        )
        values = copy.deepcopy(self.inputs)
        values["source_payloads"]["package_init_source"] += b"# drift\n"
        values["source_hashes"]["package_init_source"] = _sha256(
            values["source_payloads"]["package_init_source"]
        )
        self._assert_code(
            "STATIC_SOURCE_HASH_MISMATCH",
            lambda: build_fp32_attached_offline_package_manifest(
                values["upstream_review"],
                values["remediation_preregistration"],
                values["sft_config"],
                values["adapter_config"],
                base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                source_hashes=values["source_hashes"],
                source_payloads=values["source_payloads"],
            ),
        )

    def test_adapter_inspector_implementation_is_bound(self) -> None:
        self.assertIn("adapter_inspector_source", self.manifest["source_artifacts"])
        inspector = self.manifest["components"]["adapter"]["tensor_inspector"]
        self.assertEqual(
            [item["path"] for item in inspector["direct_dependencies"]],
            [
                "src/fullcycle_bridge/consumer.py",
                "src/fullcycle_bridge/tool_router.py",
                "src/fullcycle_bridge/tool_router_sft.py",
            ],
        )
        self.assertEqual(
            REPOSITORY_SOURCE_PATHS["adapter_inspector_source"],
            "src/fullcycle_bridge/tool_router_fp32_attached_artifact_eligibility.py",
        )
        values = copy.deepcopy(self.inputs)
        values["source_payloads"]["adapter_inspector_source"] += b"# drift\n"
        values["source_hashes"]["adapter_inspector_source"] = _sha256(
            values["source_payloads"]["adapter_inspector_source"]
        )
        self._assert_code(
            "STATIC_SOURCE_HASH_MISMATCH",
            lambda: build_fp32_attached_offline_package_manifest(
                values["upstream_review"],
                values["remediation_preregistration"],
                values["sft_config"],
                values["adapter_config"],
                base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                source_hashes=values["source_hashes"],
                source_payloads=values["source_payloads"],
            ),
        )
        values = copy.deepcopy(self.inputs)
        values["source_payloads"]["sft_helpers_source"] += b"# drift\n"
        values["source_hashes"]["sft_helpers_source"] = _sha256(
            values["source_payloads"]["sft_helpers_source"]
        )
        self._assert_code(
            "STATIC_SOURCE_HASH_MISMATCH",
            lambda: build_fp32_attached_offline_package_manifest(
                values["upstream_review"],
                values["remediation_preregistration"],
                values["sft_config"],
                values["adapter_config"],
                base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
                tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
                adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
                source_hashes=values["source_hashes"],
                source_payloads=values["source_payloads"],
            ),
        )

    def test_effective_generation_overrides_upstream_sampling_defaults(self) -> None:
        generation = self.manifest["execution_contract"]["generation"]
        self.assertTrue(generation["upstream_file_defaults"]["do_sample"])
        self.assertEqual(generation["upstream_file_defaults"]["temperature"], 0.7)
        self.assertFalse(generation["effective_contract"]["do_sample"])
        self.assertEqual(
            generation["effective_sampling_overrides"],
            {"temperature": None, "top_k": None, "top_p": None},
        )

    def test_structured_use_and_limitations_are_exact(self) -> None:
        use = self.manifest["use_and_limitations"]
        self.assertEqual(
            use["intended_use"], "clean_location_reproducibility_test_only"
        )
        self.assertEqual(use["evidence_scope"]["formal_full_eval_runs"], 1)
        self.assertFalse(use["evidence_scope"]["full_eval_repeat_variance_established"])
        self.assertIn("runtime_integration", use["prohibited_uses"])
        self.assertIn("pristine_fp32_checkpoint", use["unsupported_claims"])
        forged = copy.deepcopy(self.manifest)
        forged["use_and_limitations"]["prohibited_uses"].remove("runtime_integration")
        forged_payload = _payload(forged)
        self._assert_code(
            "MANIFEST_RECOMPUTATION_MISMATCH",
            lambda: self._validate(
                manifest_payload=forged_payload,
                expected_manifest_sha256=_sha256(forged_payload),
            ),
        )

    def test_adapter_local_base_path_is_recorded_but_non_authoritative(self) -> None:
        adapter = self.manifest["components"]["adapter"]
        self.assertEqual(
            adapter["recorded_local_base_path"],
            "work\\models\\Qwen2.5-1.5B-Instruct",
        )
        self.assertFalse(adapter["local_base_path_authoritative"])
        self.assertFalse(
            self.manifest["resolution_contract"][
                "adapter_local_base_path_authoritative"
            ]
        )
        resolution = self.manifest["resolution_contract"]
        receipts = self.manifest["observed_file_receipts"]
        self.assertFalse(
            resolution["absolute_or_caller_supplied_machine_paths_embedded"]
        )
        self.assertTrue(resolution["historical_machine_relative_adapter_path_recorded"])
        self.assertFalse(receipts["caller_supplied_or_absolute_machine_paths_recorded"])

    def test_resolve_component_files_positive_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.bin"
            second = root / "b.json"
            first.write_bytes(b"alpha")
            second.write_bytes(b"beta")
            expected = [
                {
                    "path": "a.bin",
                    "role": "a",
                    "bytes": 5,
                    "sha256": _sha256(b"alpha"),
                },
                {
                    "path": "b.json",
                    "role": "b",
                    "bytes": 4,
                    "sha256": _sha256(b"beta"),
                },
            ]
            resolved = resolve_component_files(root, expected, root_role="fixture")
            self.assertTrue(resolved["resolved"])
            second.write_bytes(b"drift")
            mismatch = resolve_component_files(root, expected, root_role="fixture")
            self.assertFalse(mismatch["resolved"])
            self.assertIn(
                mismatch["issues"][0]["code"],
                {"BYTE_COUNT_MISMATCH", "SHA256_MISMATCH"},
            )
            second.unlink()
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            missing = resolve_component_files(root, expected, root_role="fixture")
            self.assertFalse(missing["resolved"])
            self.assertEqual(
                {item["code"] for item in missing["issues"]},
                {"MISSING_FILE", "UNEXPECTED_FILE"},
            )

    def test_cache_directory_is_explicitly_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"component"
            (root / "component.bin").write_bytes(payload)
            (root / ".cache").mkdir()
            expected = [
                {
                    "path": "component.bin",
                    "role": "fixture",
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            ]
            result = resolve_component_files(
                root,
                expected,
                root_role="fixture",
                allowed_directory_names=frozenset({".cache"}),
            )
            self.assertTrue(result["resolved"])

    def test_component_root_change_after_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"component"
            (root / "component.bin").write_bytes(payload)
            expected = [
                {
                    "path": "component.bin",
                    "role": "fixture",
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            ]
            snapshot = contract._directory_entry_snapshot(root)
            changed = (*snapshot, ("late.bin", "file"))
            with mock.patch.object(
                contract,
                "_directory_entry_snapshot",
                side_effect=[snapshot, changed],
            ):
                result = resolve_component_files(root, expected, root_role="fixture")
        self.assertFalse(result["resolved"])
        self.assertIn(
            "ROOT_CHANGED_DURING_RESOLUTION",
            {item["code"] for item in result["issues"]},
        )

    def test_open_handle_identity_must_match_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "component.bin"
            path.write_bytes(b"component")
            actual = path.lstat()
            changed_identity = mock.Mock(
                st_dev=actual.st_dev,
                st_ino=actual.st_ino + 1,
                st_mode=actual.st_mode,
                st_size=actual.st_size,
                st_mtime_ns=actual.st_mtime_ns,
                st_ctime_ns=actual.st_ctime_ns,
            )
            with mock.patch.object(
                contract.os,
                "fstat",
                return_value=changed_identity,
            ):
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    contract.hash_regular_file(path)

    def test_source_loader_rechecks_earlier_file_after_all_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            adapter = temporary_root / "adapter"
            adapter.mkdir()
            fixture_payloads = {
                "README.md": b"synthetic adapter card\n",
                "adapter_config.json": b'{"fixture":true}\n',
                "adapter_model.safetensors": b"synthetic weights",
            }
            fixture_specs = tuple(
                (
                    name,
                    f"synthetic_{name}",
                    len(payload),
                    _sha256(payload),
                )
                for name, payload in fixture_payloads.items()
            )
            for name, payload in fixture_payloads.items():
                (adapter / name).write_bytes(payload)

            target = adapter / "adapter_config.json"
            replacement = temporary_root / "replacement.json"
            replacement.write_bytes(target.read_bytes())
            original = builder._read_regular_file_receipt
            replaced = False

            def replace_after_later_read(
                path: Path, label: str
            ) -> tuple[bytes, builder._StatSignature]:
                nonlocal replaced
                result = original(path, label)
                if label == "adapter_readme" and not replaced:
                    replacement.replace(target)
                    replaced = True
                return result

            with (
                mock.patch.object(builder, "ADAPTER_FILE_SPECS", fixture_specs),
                mock.patch.object(
                    builder,
                    "_read_regular_file_receipt",
                    side_effect=replace_after_later_read,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed after"):
                    builder.load_repository_manifest_inputs(adapter_dir=adapter)

    def test_builder_rechecks_component_after_manifest_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            base = temporary_root / "base"
            base.mkdir()
            expected_names = {
                item[0] for item in (*BASE_MODEL_FILE_SPECS, *TOKENIZER_FILE_SPECS)
            }
            for name in expected_names:
                (base / name).write_bytes(name.encode("utf-8"))

            target = base / sorted(expected_names)[0]
            replacement = temporary_root / "replacement.bin"
            replacement.write_bytes(target.read_bytes())

            def replace_during_manifest_assembly(
                *_args: object, **_kwargs: object
            ) -> dict[str, object]:
                replacement.replace(target)
                return {}

            with mock.patch.object(
                builder,
                "build_fp32_attached_offline_package_manifest",
                side_effect=replace_during_manifest_assembly,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed after"):
                    builder.build(base_model_dir=base)

    def test_repository_source_set_and_intermediate_reparse_fail_closed(self) -> None:
        empty = contract.resolve_repository_sources(
            ROOT,
            {},
            self.inputs["source_hashes"],
        )
        self.assertFalse(empty["resolved"])
        self.assertEqual(empty["expected_files"], len(REPOSITORY_SOURCE_PATHS))
        self.assertEqual(empty["issues"][0]["code"], "SOURCE_PATH_SET_MISMATCH")

        original = contract._is_reparse

        def pretend_src_is_reparse(path: Path) -> bool:
            return path == ROOT / "src" or original(path)

        with mock.patch.object(
            contract,
            "_is_reparse",
            side_effect=pretend_src_is_reparse,
        ):
            unsafe = contract.resolve_repository_sources(
                ROOT,
                REPOSITORY_SOURCE_PATHS,
                self.inputs["source_hashes"],
            )
        self.assertFalse(unsafe["resolved"])
        self.assertIn(
            "MISSING_OR_UNSAFE_FILE",
            {item["code"] for item in unsafe["issues"]},
        )

    def test_missing_external_base_root_keeps_resolution_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-base"
            combined = validate_and_resolve_fp32_attached_offline_package(
                self.manifest_payload,
                self.manifest_sha256,
                self.inputs["upstream_review"],
                self.inputs["remediation_preregistration"],
                self.inputs["sft_config"],
                self.inputs["adapter_config"],
                source_hashes=self.inputs["source_hashes"],
                source_payloads=self.inputs["source_payloads"],
                expected_source_hashes=self.inputs["source_hashes"],
                base_model_root=missing,
                adapter_root=DEFAULT_ADAPTER_DIR,
                repository_root=ROOT,
            )
            result = combined["resolution"]
        self.assertFalse(result["resolved"])
        self.assertFalse(result["eligible_for_clean_location_reproducibility_test"])
        self.assertFalse(result["offline_artifact_eligible"])
        self.assertFalse(result["runtime_eligible"])
        self.assertEqual(result["failure_mode"], "component_resolution_failed_closed")
        self.assertEqual(result["groups"][0]["issues"][0]["code"], "MISSING_ROOT")

    def test_unified_gate_keeps_missing_local_base_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-base"
            with mock.patch.object(builder, "DEFAULT_BASE_MODEL_DIR", missing):
                combined = (
                    offline_gate._validate_fp32_attached_offline_package_manifest(
                        self.inputs["upstream_review"]
                    )
                )
        self.assertTrue(combined["validation"]["frozen_manifest_valid"])
        self.assertFalse(combined["resolution"]["resolved"])
        self.assertFalse(
            combined["resolution"]["eligible_for_clean_location_reproducibility_test"]
        )
        self.assertFalse(combined["resolution"]["runtime_eligible"])

    def test_unified_gate_uses_external_source_hash_roots(self) -> None:
        expected_roots = dict(
            offline_gate.TOOL_ROUTER_FP32_ATTACHED_OFFLINE_PACKAGE_SOURCE_HASHES
        )
        expected_roots["prompt"] = "sha256:" + "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-base"
            with (
                mock.patch.object(builder, "DEFAULT_BASE_MODEL_DIR", missing),
                mock.patch.object(
                    offline_gate,
                    "TOOL_ROUTER_FP32_ATTACHED_OFFLINE_PACKAGE_SOURCE_HASHES",
                    expected_roots,
                ),
                self.assertRaises(ToolRouterValidationError) as raised,
            ):
                offline_gate._validate_fp32_attached_offline_package_manifest(
                    self.inputs["upstream_review"]
                )
        self.assertEqual(raised.exception.code, "SOURCE_HASH_ROOT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
