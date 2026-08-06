from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge import (  # noqa: E402
    tool_router_fp32_attached_offline_package_reproducibility as contract,
)
from fullcycle_bridge.consumer import canonical_json_bytes  # noqa: E402
from fullcycle_bridge.tool_router_decision_compilation import (  # noqa: E402
    compile_decision,
)
from scripts import (  # noqa: E402
    probe_tool_router_fp32_attached_offline_package_reproducibility as runner,
)


FREEZE_COMMIT = "a" * 40
VALID_RAW_OUTPUT = (
    '{"arguments":{"reason_code":"capability_unavailable"},'
    '"expected_result":"rejection","requires_approval":false,'
    '"risk_level":"medium","selected_tool":"fallback_to_strong_model",'
    '"should_fallback":true,"should_reject":false}'
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _path_binding_preregistration() -> dict[str, object]:
    return {
        "materialization_protocol": {
            "destination_policy": {
                "parent_relative_to_repository": "work/clean-location",
                "children": {
                    "repository": "repository",
                    "base_model_and_tokenizer": "base_model_and_tokenizer",
                },
                "adapter_root_relative_to_repository": (
                    "baseline/adapters/fc-mvp-001-lora-sft-v2"
                ),
            },
            "receipt_policy": {
                "output_root_relative_to_repository": "work/test-fixtures"
            },
        }
    }


def _output_policy_preregistration(
    predictions_name: str,
    result_name: str,
) -> dict[str, object]:
    return {
        "execution_protocol": {
            "output_policy": {
                "root_authority": "caller_supplied",
                "exclusive_create": True,
                "required_parent_relative_to_controller_repository": "baseline",
                "machine_paths_recorded": False,
                "replay_file": predictions_name,
                "evidence_file": result_name,
            }
        }
    }


def _precision_audit() -> dict[str, object]:
    return {
        "base_parameters": {
            "floating_tensors": 338,
            "floating_elements": 1_543_714_304,
            "dtypes": {"float32": 1_543_714_304},
            "devices": {"cuda:0": 1_543_714_304},
        },
        "adapter_parameters": {
            "floating_tensors": 224,
            "floating_elements": 4_358_144,
            "dtypes": {"float32": 4_358_144},
            "devices": {"cuda:0": 4_358_144},
        },
        "floating_buffers": {
            "floating_tensors": 1,
            "floating_elements": 64,
            "dtypes": {"float32": 64},
            "devices": {"cuda:0": 64},
        },
        "lora_target_modules": 112,
        "lora_parameter_tensors": 224,
        "adapter_parameters_finite": True,
        "active_adapters": ["default"],
        "is_peft_model": True,
        "input_output_embeddings_tied": True,
        "attn_implementation": "sdpa",
        "attention_class": "Qwen2Attention",
        "output_attentions": False,
        "hf_device_map": None,
        "training": False,
        "autocast_enabled": False,
        "lora_dropout": {"modules": 112, "training_modules": 0},
        "autocast_adapter_dtype": True,
        "attached_execution_form": "attached_factorized_lora",
    }


def _clean_helpers() -> runner.CleanHelpers:
    return runner.CleanHelpers(
        package_alias="unit_clean_package",
        canonical_json_bytes=canonical_json_bytes,
        compile_decision=compile_decision,
        fixture_digest=lambda _records: "fixture-digest",
        load_fixture=lambda _path: [],
        render_user_payload=lambda record: str(record["example_id"]),
    )


def _rich_record(example_id: str, raw_output: str) -> dict[str, object]:
    compilation = runner._compile_observed_output(raw_output, _clean_helpers())
    return {
        "example_id": example_id,
        "rendered_prompt_sha256": _digest(f"prompt:{example_id}".encode()),
        "input_token_ids_sha256": runner._token_ids_sha256([1, 2]),
        "input_token_count": 2,
        "output_token_ids_sha256": runner._token_ids_sha256([3]),
        "output_token_count": 1,
        "raw_output": raw_output,
        "raw_output_utf8_sha256": _digest(raw_output.encode("utf-8")),
        **compilation,
    }


class _FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)

    def numel(self) -> int:
        return len(self.values)


class _FakeInputIds:
    shape = (1, 2)

    def __getitem__(self, key: object) -> _FakeVector:
        if key != 0:
            raise AssertionError(key)
        return _FakeVector([11, 22])


class _FakeEncoded(dict[str, object]):
    def __init__(self) -> None:
        super().__init__(input_ids=_FakeInputIds())

    def to(self, device: str) -> _FakeEncoded:
        if device != "cuda":
            raise AssertionError(device)
        return self


class _FakeGenerated:
    def __getitem__(self, key: object) -> _FakeVector:
        if key != (0, slice(2, None, None)):
            raise AssertionError(key)
        return _FakeVector([31, 151645])


class _FakeTokenizer:
    eos_token_id = 151643

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "rendered-prompt"

    def __call__(self, *_args: object, **_kwargs: object) -> _FakeEncoded:
        return _FakeEncoded()

    def decode(self, *_args: object, **_kwargs: object) -> str:
        return VALID_RAW_OUTPUT


class RunnerProtocolTests(unittest.TestCase):
    def test_preflight_only_has_zero_runtime_and_zero_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "controller"
            destination_id = "1" * 32
            destination = controller / "work" / "clean-location" / destination_id
            baseline = controller / "baseline"
            baseline.mkdir(parents=True)
            repository = destination / "repository"
            adapter = repository / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
            base = destination / "base_model_and_tokenizer"
            for path in (repository, adapter, base):
                path.mkdir(parents=True, exist_ok=True)
            receipt_path = (
                controller / "work" / "test-fixtures" / "materialization.json"
            )
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(
                json.dumps({"destination": {"destination_id": destination_id}}),
                encoding="utf-8",
            )
            preregistration_path = root / "preregistration.json"
            preregistration_path.write_text("{}", encoding="utf-8")
            predictions = baseline / (
                "tool-router-fp32-attached-offline-package-"
                "reproducibility-v1-predictions.json"
            )
            result = baseline / (
                "fc-mvp-001-fp32-attached-offline-package-reproducibility-v1.json"
            )
            static_environment = {
                "python": "3.12.12",
                "torch": "2.6.0+cu124",
                "transformers": "4.49.0",
                "peft": "0.14.0",
                "accelerate": "1.3.0",
                "huggingface_hub": "0.29.3",
                "safetensors": "0.5.3",
                "tokenizers": "0.21.4",
            }
            preregistration = {
                "source_lineage": {
                    "manifest": {"path": "manifest.json", "sha256": _digest(b"m")},
                    "reference_predictions": {"path": "predictions.json"},
                    "reference_evidence": {"path": "evidence.json"},
                    "evaluation": {"path": "eval.json"},
                },
                "execution_protocol": {
                    "output_policy": {
                        "root_authority": "caller_supplied",
                        "exclusive_create": True,
                        "required_parent_relative_to_controller_repository": "baseline",
                        "machine_paths_recorded": False,
                        "replay_file": predictions.name,
                        "evidence_file": result.name,
                    }
                },
                "materialization_protocol": {
                    "destination_policy": {
                        "parent_relative_to_repository": "work/clean-location",
                        "children": {
                            "repository": "repository",
                            "base_model_and_tokenizer": "base_model_and_tokenizer",
                        },
                        "adapter_root_relative_to_repository": (
                            "baseline/adapters/fc-mvp-001-lora-sft-v2"
                        ),
                    },
                    "receipt_policy": {
                        "output_root_relative_to_repository": "work/test-fixtures"
                    },
                },
            }
            expected_environment = {
                **static_environment,
                "device": "cuda",
                "gpu": "unused-in-static-preflight",
                "gpu_vram_bytes": 1,
                "compute_capability": "8.9",
            }
            authenticated = SimpleNamespace(
                manifest={
                    "components": {
                        "environment": {"recorded_environment": expected_environment}
                    }
                }
            )
            receipt = {
                "materialization_passed": True,
                "destination": {
                    "absolute_paths_recorded": False,
                    "symlinks_used": False,
                    "reparse_points_used": False,
                    "hardlinks_used": False,
                    "overwrite_used": False,
                },
                "transport": {
                    "network_used_during_execution": False,
                    "alternate_remote_used": False,
                    "alternate_revision_fallback_used": False,
                    "historical_adapter_base_path_used": False,
                },
            }
            fake_contract = SimpleNamespace(
                load_and_validate_preregistration=mock.Mock(
                    return_value=SimpleNamespace(
                        data=preregistration,
                        sha256=_digest(b"preregistration"),
                    )
                ),
                load_manifest_source_bundle=mock.Mock(return_value=object()),
                authenticate_manifest_and_references=mock.Mock(
                    return_value=authenticated
                ),
                resolve_clean_roots=mock.Mock(return_value={"resolved": True}),
                validate_materialization_receipt=mock.Mock(return_value=receipt),
            )
            args = argparse.Namespace(
                preregistration=preregistration_path,
                materialization_receipt=receipt_path,
                controller_repository_root=controller,
                clean_base_model_dir=base,
                clean_adapter_dir=adapter,
                clean_repository_root=repository,
                freeze_commit=FREEZE_COMMIT,
                predictions_output=predictions,
                result_output=result,
                preflight_only=True,
            )
            with (
                mock.patch.object(
                    runner, "_clean_repository_head", return_value=FREEZE_COMMIT
                ),
                mock.patch.object(runner, "_validate_clean_protocol_sources"),
                mock.patch.object(
                    runner,
                    "_read_regular_file",
                    side_effect=lambda path, *, label: (
                        receipt_path.read_bytes() if path == receipt_path else b"{}"
                    ),
                ),
                mock.patch.object(
                    runner,
                    "_observe_static_environment",
                    return_value=static_environment,
                ),
                mock.patch.object(runner, "_load_runtime_modules") as runtime_import,
                mock.patch.object(runner, "ROOT", repository),
            ):
                observed = runner.run(args, contract=fake_contract)
            self.assertTrue(observed["preflight_only"])
            self.assertFalse(observed["gpu_runtime_environment_checked"])
            self.assertEqual(observed["generate_calls"], 0)
            runtime_import.assert_not_called()
            self.assertFalse(predictions.exists())
            self.assertFalse(result.exists())

    def test_path_root_and_git_checks_precede_clean_contract_import(self) -> None:
        events: list[str] = []
        repository = runner.ROOT.resolve(strict=True)
        paths = {
            "repository_root": repository,
            "predictions_output": repository / "baseline" / "predictions.json",
            "result_output": repository / "baseline" / "evidence.json",
        }

        def record(name: str, value: object = None) -> object:
            events.append(name)
            return value

        def stop_after_contract_import() -> None:
            events.append("contract-import")
            raise RuntimeError("stop after import")

        with (
            mock.patch.object(
                runner,
                "_resolve_cli_paths",
                side_effect=lambda _args: record("paths", paths),
            ),
            mock.patch.object(
                runner,
                "_validate_output_targets",
                side_effect=lambda *_args: record("output-targets"),
            ),
            mock.patch.object(
                runner,
                "_validate_freeze_commit",
                side_effect=lambda _value: record("freeze", FREEZE_COMMIT),
            ),
            mock.patch.object(
                runner,
                "_require_runner_from_clean_repository",
                side_effect=lambda _root: record("runner-root"),
            ),
            mock.patch.object(
                runner,
                "_clean_repository_head",
                side_effect=lambda _root: record("git", FREEZE_COMMIT),
            ),
            mock.patch.object(
                runner,
                "_force_offline_environment",
                side_effect=lambda: record("offline"),
            ),
            mock.patch.object(
                runner,
                "_load_contract_module",
                side_effect=stop_after_contract_import,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after import"):
                runner.run(argparse.Namespace(freeze_commit=FREEZE_COMMIT))
        self.assertEqual(
            events,
            [
                "paths",
                "output-targets",
                "freeze",
                "runner-root",
                "git",
                "offline",
                "contract-import",
            ],
        )

    def test_contract_import_is_temporary_and_creates_no_ignored_bytecode(
        self,
    ) -> None:
        self.assertTrue(runner.sys.dont_write_bytecode)
        with tempfile.TemporaryDirectory() as temporary:
            clean_root = Path(temporary) / "clean-repository"
            clean_source = clean_root / "src"
            package = clean_source / "clean_contract_probe"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "contract.py").write_text("VALUE = 1\n", encoding="utf-8")
            module_names = ("clean_contract_probe", "clean_contract_probe.contract")
            try:
                with (
                    mock.patch.object(runner, "ROOT", clean_root),
                    mock.patch.object(runner, "SRC", clean_source),
                    mock.patch.object(
                        runner,
                        "CONTRACT_MODULE",
                        "clean_contract_probe.contract",
                    ),
                ):
                    module = runner._load_contract_module()
                self.assertEqual(module.VALUE, 1)
                self.assertNotIn(str(clean_source.resolve()), runner.sys.path)
                self.assertFalse(any(clean_root.rglob("*.pyc")))
                self.assertFalse(any(clean_root.rglob("__pycache__")))
            finally:
                for name in module_names:
                    runner.sys.modules.pop(name, None)

    def test_runtime_modules_have_external_regular_origins_without_clean_src(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_root = root / "clean-repository"
            clean_source = clean_root / "src"
            external = root / "site-packages"
            clean_source.mkdir(parents=True)
            external.mkdir()
            names = (
                "accelerate",
                "peft",
                "torch",
                "transformers",
                "huggingface_hub",
                "safetensors",
                "tokenizers",
                "peft.tuners.tuners_utils",
            )
            modules: dict[str, ModuleType] = {}
            for index, name in enumerate(names):
                origin = external / f"module_{index}.py"
                origin.write_text("", encoding="utf-8")
                module = ModuleType(name)
                module.__file__ = str(origin)
                module.__version__ = "unit"
                modules[name] = module
            modules["transformers"].AutoConfig = object()
            modules["transformers"].AutoModelForCausalLM = object()
            modules["transformers"].AutoTokenizer = object()
            modules["peft"].PeftModel = object()
            modules["peft.tuners.tuners_utils"].BaseTunerLayer = object()
            clean_src_absent: list[bool] = []

            def import_module(name: str) -> ModuleType:
                clean_src_absent.append(
                    str(clean_source.resolve()) not in runner.sys.path
                )
                return modules[name]

            with (
                mock.patch.object(runner, "ROOT", clean_root),
                mock.patch.object(runner, "SRC", clean_source),
                mock.patch.object(
                    runner.importlib,
                    "import_module",
                    side_effect=import_module,
                ),
            ):
                observed = runner._load_runtime_modules()
            self.assertIs(observed.torch, modules["torch"])
            self.assertEqual(clean_src_absent, [True] * len(names))

    def test_runtime_module_origin_inside_clean_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean_root = Path(temporary) / "clean-repository"
            origin = clean_root / "src" / "torch.py"
            origin.parent.mkdir(parents=True)
            origin.write_text("", encoding="utf-8")
            module = ModuleType("torch")
            module.__file__ = str(origin)
            with (
                mock.patch.object(runner, "ROOT", clean_root),
                mock.patch.object(
                    runner.importlib,
                    "import_module",
                    return_value=module,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "inside clean repository"):
                    runner._import_external_runtime_module("torch")

    def test_runtime_module_origin_must_be_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_root = root / "clean-repository"
            external_directory = root / "site-packages" / "torch"
            clean_root.mkdir()
            external_directory.mkdir(parents=True)
            module = ModuleType("torch")
            module.__file__ = str(external_directory)
            with (
                mock.patch.object(runner, "ROOT", clean_root),
                mock.patch.object(
                    runner.importlib,
                    "import_module",
                    return_value=module,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "regular file"):
                    runner._import_external_runtime_module("torch")

    def test_receipt_destination_cannot_be_mixed_with_other_clean_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = Path(temporary).resolve()
            receipt_destination_id = "1" * 32
            other_destination = controller / "work" / "clean-location" / ("2" * 32)
            repository = other_destination / "repository"
            receipt_path = controller / "work" / "test-fixtures" / "receipt.json"
            paths = {
                "controller_root": controller,
                "repository_root": repository,
                "base_model_root": other_destination / "base_model_and_tokenizer",
                "adapter_root": (
                    repository / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
                ),
                "materialization_receipt": receipt_path,
            }
            with self.assertRaisesRegex(RuntimeError, "materialization receipt"):
                runner._validate_materialization_path_bindings(
                    _path_binding_preregistration(),
                    {"destination": {"destination_id": receipt_destination_id}},
                    paths,
                )

    def test_runner_rejects_controller_or_historical_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_repository = root / "clean" / "repository"
            controller_repository = root / "controller"
            historical_repository = root / "historical"
            for path in (
                clean_repository,
                controller_repository,
                historical_repository,
            ):
                path.mkdir(parents=True)
            with mock.patch.object(runner, "ROOT", clean_repository):
                runner._require_runner_from_clean_repository(
                    clean_repository.resolve(strict=True)
                )
                for wrong_root in (controller_repository, historical_repository):
                    with self.subTest(wrong_root=wrong_root):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "runner must execute from",
                        ):
                            runner._require_runner_from_clean_repository(
                                wrong_root.resolve(strict=True)
                            )

    def test_clean_git_state_ignores_external_git_environment_overrides(self) -> None:
        clean = SimpleNamespace(stdout=FREEZE_COMMIT + "\n")
        status = SimpleNamespace(stdout="")
        overrides = {
            "GIT_DIR": "hijack-dir",
            "GIT_WORK_TREE": "hijack-worktree",
            "GIT_COMMON_DIR": "hijack-common",
            "GIT_OBJECT_DIRECTORY": "hijack-objects",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "hijack-alternates",
            "GIT_INDEX_FILE": "hijack-index",
            "GIT_CONFIG": "hijack-config",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "hijack",
        }
        with (
            mock.patch.dict(runner.os.environ, overrides, clear=False),
            mock.patch.object(
                runner.subprocess,
                "run",
                side_effect=[clean, status],
            ) as run_git,
        ):
            self.assertEqual(
                runner._clean_repository_head(Path("clean-repository")),
                FREEZE_COMMIT,
            )
        self.assertEqual(run_git.call_count, 2)
        for call in run_git.call_args_list:
            environment = call.kwargs["env"]
            self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "NUL")
            self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
            self.assertEqual(
                {name for name in environment if name.startswith("GIT_")},
                {
                    "GIT_CONFIG_GLOBAL",
                    "GIT_CONFIG_NOSYSTEM",
                    "GIT_TERMINAL_PROMPT",
                },
            )
            command = call.args[0]
            self.assertIn("core.fsmonitor=false", command)
            self.assertIn("core.untrackedCache=false", command)
        self.assertIn("rev-parse", run_git.call_args_list[0].args[0])
        self.assertIn("status", run_git.call_args_list[1].args[0])
        self.assertIn("--ignored", run_git.call_args_list[1].args[0])
        self.assertIn("--untracked-files=all", run_git.call_args_list[1].args[0])

    def test_clean_git_state_rejects_ignored_bytecode_import_hijack(self) -> None:
        with mock.patch.object(
            runner.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(stdout=FREEZE_COMMIT + "\n"),
                SimpleNamespace(
                    stdout=(
                        "!! src/fullcycle_bridge/__pycache__/consumer.cpython-312.pyc\n"
                    )
                ),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked or untracked"):
                runner._clean_repository_head(Path("clean-repository"))

    def test_output_policy_rejects_outside_repository_baseline_with_same_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            controller = root / "controller"
            outside = root / "outside" / "baseline"
            controller.mkdir()
            outside.mkdir(parents=True)
            predictions = outside / "predictions.json"
            result = outside / "evidence.json"
            with self.assertRaisesRegex(RuntimeError, "output paths violate"):
                runner._validate_formal_output_policy(
                    _output_policy_preregistration(
                        predictions.name,
                        result.name,
                    ),
                    controller,
                    predictions,
                    result,
                )

    def test_raw_reparse_or_symlink_alias_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            alias = root / "alias"
            target.mkdir()
            try:
                alias.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            with self.assertRaisesRegex(RuntimeError, "symlink or reparse"):
                runner._canonical_existing_path(
                    alias,
                    "aliased root",
                    expect_directory=True,
                )

    def test_second_output_create_race_removes_only_unconsumed_reservation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.json"
            result = root / "result.json"
            attempt = runner._ExclusiveOutputAttempt(predictions, result)
            result.write_bytes(b"raced")
            with self.assertRaises(FileExistsError):
                attempt.consume()
            attempt.close()
            self.assertFalse(predictions.exists())
            self.assertEqual(result.read_bytes(), b"raced")

    def test_attempt_consumption_leaves_exclusive_tombstones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.json"
            result = root / "result.json"
            attempt = runner._ExclusiveOutputAttempt(predictions, result)
            self.assertFalse(predictions.exists())
            self.assertFalse(result.exists())
            attempt.consume()
            self.assertTrue(attempt.consumed)
            attempt.close()
            self.assertEqual(predictions.read_bytes(), b"")
            self.assertEqual(result.read_bytes(), b"")
            with self.assertRaisesRegex(RuntimeError, "must not already exist"):
                runner._validate_output_targets(predictions, result)

    def test_raw_drift_fails_even_when_compiled_decisions_match(self) -> None:
        reference_data = json.loads(
            (
                runner.ROOT
                / "baseline"
                / "tool-router-fp32-attached-remediation-v1-predictions.json"
            ).read_text(encoding="utf-8")
        )
        reference = tuple(
            {
                "example_id": item["example_id"],
                "raw_output": item["raw_output"],
            }
            for item in reference_data["outputs"]
        )
        authenticated = contract.AuthenticatedInputs(
            preregistration={"behavior_reference": {"scope": "unit-test"}},
            manifest_payload=b"",
            manifest={},
            manifest_validation={},
            reference_predictions={},
            reference_evidence={},
            evaluation=[],
            reference_outputs=reference,
        )
        exact = [
            _rich_record(item["example_id"], item["raw_output"]) for item in reference
        ]
        exact_comparison = contract.compare_behavioral_replay(authenticated, exact)
        self.assertTrue(exact_comparison["behavioral_reproducibility_established"])

        drifted = list(exact)
        parsed = json.loads(reference[0]["raw_output"])
        formatting_only_drift = json.dumps(parsed, ensure_ascii=False, indent=2)
        drifted[0] = _rich_record("eval-001", formatting_only_drift)
        comparison = contract.compare_behavioral_replay(authenticated, drifted)
        self.assertEqual(comparison["raw_mismatch_example_ids"], ["eval-001"])
        self.assertEqual(comparison["compiled_mismatch_example_ids"], [])
        self.assertFalse(comparison["behavioral_reproducibility_established"])

    def test_resource_failure_classification_is_independent(self) -> None:
        gates = {
            "metadata_validation": True,
            "materialization": True,
            "clean_location_resolution": True,
            "environment": True,
            "execution_contract": True,
            "behavioral_replay": True,
            "resources": False,
        }
        self.assertEqual(
            contract.classify_reproducibility_gates(gates),
            contract.RESOURCE_EXCEEDED_CLASSIFICATION,
        )

    def test_clean_helpers_are_loaded_from_supplied_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            package = repository / "src" / "fullcycle_bridge"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "consumer.py").write_text(
                "def canonical_json_bytes(value): return b'clean'\n",
                encoding="utf-8",
            )
            (package / "tool_router_decision_compilation.py").write_text(
                "def compile_decision(value): return dict(value)\n",
                encoding="utf-8",
            )
            (package / "tool_router.py").write_text(
                "def fixture_digest(value): return 'clean-digest'\n"
                "def load_fixture(path): return []\n",
                encoding="utf-8",
            )
            (package / "tool_router_sft.py").write_text(
                "def render_user_payload(value): return 'clean-render'\n",
                encoding="utf-8",
            )
            with runner._load_clean_helpers(repository) as helpers:
                self.assertEqual(helpers.canonical_json_bytes({}), b"clean")
                self.assertEqual(helpers.render_user_payload({}), "clean-render")
                self.assertEqual(helpers.fixture_digest([]), "clean-digest")
            self.assertNotIn(runner.CLEAN_PACKAGE_ALIAS, runner.sys.modules)

    def test_fake_runtime_executes_exactly_twenty_calls(self) -> None:
        records = [{"example_id": f"eval-{index:03d}"} for index in range(1, 21)]
        helpers = runner.CleanHelpers(
            package_alias="unit_clean_package",
            canonical_json_bytes=canonical_json_bytes,
            compile_decision=compile_decision,
            fixture_digest=lambda _records: "fixture-digest",
            load_fixture=lambda _path: records,
            render_user_payload=lambda record: str(record["example_id"]),
        )
        environment = {"python": "unit"}
        generation = {
            "seed": 20260803,
            "attn_implementation": "sdpa",
            "do_sample": False,
            "max_new_tokens": 256,
            "use_cache": True,
        }
        preregistration = {
            "candidate_id": "fp32-attached-factorized-lora",
            "execution_protocol": {
                "run_id": "fp32-attached-clean-location-full-eval-r1",
                "generation": generation,
            },
            "source_lineage": {
                "evaluation": {
                    "path": "fixtures/eval.json",
                    "order": [record["example_id"] for record in records],
                    "canonical_digest": "fixture-digest",
                }
            },
            "resource_caps": {
                "memory_allocated_before_load_bytes_max": 16_777_216,
            },
        }
        authenticated = SimpleNamespace(
            manifest={
                "components": {
                    "environment": {"recorded_environment": environment},
                    "tokenizer": {"revision": "revision"},
                },
                "resolution_contract": {
                    "repository_source_paths": {"prompt": "prompts/prompt.txt"}
                },
            }
        )
        tokenizer = _FakeTokenizer()
        model = SimpleNamespace(generate=mock.Mock(return_value=_FakeGenerated()))
        cuda = SimpleNamespace(
            manual_seed_all=mock.Mock(),
            empty_cache=mock.Mock(),
            memory_allocated=mock.Mock(side_effect=[0, 8_519_680]),
            synchronize=mock.Mock(),
            reset_peak_memory_stats=mock.Mock(),
            max_memory_allocated=mock.Mock(return_value=6_267_895_296),
        )
        torch = SimpleNamespace(
            cuda=cuda,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True),
            ),
            manual_seed=mock.Mock(),
            inference_mode=lambda: contextlib.nullcontext(),
        )
        runtime = runner.RuntimeModules(
            accelerate=SimpleNamespace(),
            peft=SimpleNamespace(),
            torch=torch,
            transformers=SimpleNamespace(),
            hub_version="unused",
            safetensors_version="unused",
            tokenizers_version="unused",
            AutoConfig=SimpleNamespace(),
            AutoModelForCausalLM=SimpleNamespace(),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=mock.Mock(return_value=tokenizer)
            ),
            PeftModel=SimpleNamespace(),
            BaseTunerLayer=object,
        )
        consume_attempt = mock.Mock()
        runner._force_offline_environment()
        with (
            mock.patch.object(runner, "_observe_environment", return_value=environment),
            mock.patch.object(
                runner,
                "_read_regular_file",
                return_value=b"prompt",
            ),
            mock.patch.object(
                runner,
                "_load_fp32_attached_model",
                return_value=(model, _precision_audit()),
            ),
        ):
            observed = runner._execute_model_replay(
                preregistration=preregistration,
                authenticated=authenticated,
                base_model_root=Path("base"),
                adapter_root=Path("adapter"),
                repository_root=Path("repository"),
                helpers=helpers,
                consume_attempt=consume_attempt,
                runtime=runtime,
            )
        consume_attempt.assert_called_once_with()
        self.assertEqual(model.generate.call_count, 20)
        self.assertEqual(observed["run"]["generate_calls"], 20)
        self.assertEqual(observed["run"]["retries"], 0)
        self.assertEqual(observed["run"]["warmup_calls"], 0)
        self.assertEqual(len(observed["outputs"]), 20)
        self.assertTrue(all(item["compiler_valid"] for item in observed["outputs"]))
        runtime.AutoTokenizer.from_pretrained.assert_called_once_with(
            Path("base"), local_files_only=True, revision="revision"
        )
        for call in model.generate.call_args_list:
            self.assertFalse(call.kwargs["do_sample"])
            self.assertTrue(call.kwargs["use_cache"])
            self.assertEqual(call.kwargs["max_new_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
