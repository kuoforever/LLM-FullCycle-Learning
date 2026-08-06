"""Run one fail-closed clean-location FP32 attached package replay."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from types import ModuleType
from typing import Any, BinaryIO, Callable, NoReturn, cast


sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTRACT_MODULE = (
    "fullcycle_bridge.tool_router_fp32_attached_offline_package_reproducibility"
)
CLEAN_PACKAGE_ALIAS = "_clean_fullcycle_bridge_reproducibility"
EXPECTED_RECORDS = 20
EXPECTED_LORA_TARGETS = 112
EXPECTED_LORA_PARAMETER_TENSORS = 224
EXPECTED_BASE_ELEMENTS = 1_543_714_304
EXPECTED_ADAPTER_ELEMENTS = 4_358_144
EXPECTED_BUFFER_ELEMENTS = 64
MAX_INPUT_BYTES = 16 * 1024 * 1024
REPARSE_POINT_ATTRIBUTE = 0x400
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DESTINATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
}


@dataclass(frozen=True)
class CleanHelpers:
    """Execution helpers loaded only from the authenticated clean repository."""

    package_alias: str
    canonical_json_bytes: Callable[[object], bytes]
    compile_decision: Callable[[Mapping[str, Any]], dict[str, Any]]
    fixture_digest: Callable[[Sequence[Mapping[str, Any]]], str]
    load_fixture: Callable[[Path], list[dict[str, Any]]]
    render_user_payload: Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class RuntimeModules:
    """Lazy third-party runtime imports so unit tests never require a GPU."""

    accelerate: ModuleType
    peft: ModuleType
    torch: ModuleType
    transformers: ModuleType
    hub_version: str
    safetensors_version: str
    tokenizers_version: str
    AutoConfig: Any
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    PeftModel: Any
    BaseTunerLayer: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--controller-repository-root", type=Path, required=True)
    parser.add_argument("--clean-base-model-dir", type=Path, required=True)
    parser.add_argument("--clean-adapter-dir", type=Path, required=True)
    parser.add_argument("--clean-repository-root", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    if result.get("preflight_only") is True:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "behavioral_reproducibility": result["derived_claims"][
                    "behavioral_reproducibility_established"
                ],
                "resource_gate_passed": result["resources"]["passed"],
                "runtime_eligible": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def run(
    args: argparse.Namespace,
    *,
    contract: ModuleType | None = None,
    executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate all inputs, execute once, compare, then write two artifacts."""

    executor = _execute_model_replay if executor is None else executor
    paths = _resolve_cli_paths(args)
    _validate_output_targets(paths["predictions_output"], paths["result_output"])
    freeze_commit = _validate_freeze_commit(args.freeze_commit)
    repository_root = paths["repository_root"]
    _require_runner_from_clean_repository(repository_root)
    if _clean_repository_head(repository_root) != freeze_commit:
        raise RuntimeError("clean repository HEAD does not match --freeze-commit")
    _force_offline_environment()
    contract = _load_contract_module() if contract is None else contract

    loaded = contract.load_and_validate_preregistration(paths["preregistration"])
    preregistration = loaded.data
    _validate_formal_output_policy(
        preregistration,
        paths["controller_root"],
        paths["predictions_output"],
        paths["result_output"],
    )
    adapter_root = paths["adapter_root"]
    receipt_payload = _read_regular_file(
        paths["materialization_receipt"],
        label="materialization receipt",
    )
    receipt = _parse_json_payload(receipt_payload, "materialization receipt")
    _validate_materialization_path_bindings(preregistration, receipt, paths)
    _validate_clean_protocol_sources(preregistration, repository_root)
    manifest_sources = contract.load_manifest_source_bundle(
        repository_root=repository_root,
        adapter_root=adapter_root,
    )
    source_lineage = preregistration["source_lineage"]
    manifest_payload = _read_regular_file(
        repository_root / source_lineage["manifest"]["path"],
        label="manifest",
    )
    reference_predictions_payload = _read_regular_file(
        repository_root / source_lineage["reference_predictions"]["path"],
        label="reference predictions",
    )
    reference_evidence_payload = _read_regular_file(
        repository_root / source_lineage["reference_evidence"]["path"],
        label="reference evidence",
    )
    evaluation_payload = _read_regular_file(
        repository_root / source_lineage["evaluation"]["path"],
        label="evaluation",
    )
    authenticated = contract.authenticate_manifest_and_references(
        preregistration,
        manifest_payload=manifest_payload,
        reference_predictions_payload=reference_predictions_payload,
        reference_evidence_payload=reference_evidence_payload,
        evaluation_payload=evaluation_payload,
        manifest_sources=manifest_sources,
    )
    clean_resolution = contract.resolve_clean_roots(
        preregistration,
        authenticated,
        manifest_sources,
        base_model_root=paths["base_model_root"],
        adapter_root=adapter_root,
        repository_root=repository_root,
    )
    receipt_validation = contract.validate_materialization_receipt(
        preregistration,
        receipt,
        preregistration_sha256=loaded.sha256,
        expected_freeze_commit=freeze_commit,
        clean_resolution=clean_resolution,
    )
    _require_clean_resolution(clean_resolution, receipt_validation)

    static_environment = _observe_static_environment()
    expected_environment = authenticated.manifest["components"]["environment"][
        "recorded_environment"
    ]
    _require_static_environment(static_environment, expected_environment)
    if args.preflight_only:
        return {
            "preflight_only": True,
            "eligible": True,
            "preregistration_sha256": loaded.sha256,
            "protocol_freeze_commit": freeze_commit,
            "manifest_file_sha256": source_lineage["manifest"]["sha256"],
            "clean_location_resolution_checked": True,
            "materialization_receipt_checked": True,
            "static_environment_checked": True,
            "gpu_runtime_environment_checked": False,
            "runtime_imported": False,
            "model_loaded": False,
            "generate_calls": 0,
            "outputs_created": False,
            "absolute_paths_recorded": False,
        }

    historical_hint = authenticated.manifest["components"]["adapter"][
        "recorded_local_base_path"
    ]
    attempt = _ExclusiveOutputAttempt(
        paths["predictions_output"], paths["result_output"]
    )
    try:
        with _clean_execution_context(historical_hint, paths):
            with _load_clean_helpers(repository_root) as helpers:
                execution = executor(
                    preregistration=preregistration,
                    authenticated=authenticated,
                    base_model_root=paths["base_model_root"],
                    adapter_root=adapter_root,
                    repository_root=repository_root,
                    helpers=helpers,
                    consume_attempt=attempt.consume,
                )
        if not attempt.consumed:
            raise RuntimeError("formal executor returned without consuming its attempt")
        contract.compare_behavioral_replay(
            authenticated,
            execution["outputs"],
        )
        predictions = contract.build_replay_artifact(
            preregistration,
            authenticated,
            preregistration_sha256=loaded.sha256,
            protocol_freeze_commit=freeze_commit,
            materialization_receipt=receipt_validation,
            clean_resolution=clean_resolution,
            observed_environment=execution["environment"],
            precision_audit=execution["precision_audit"],
            performance=execution["performance"],
            outputs=execution["outputs"],
        )
        prediction_payload = contract.artifact_json_bytes(predictions)
        evidence = contract.build_reproducibility_evidence(
            preregistration,
            authenticated,
            preregistration_sha256=loaded.sha256,
            protocol_freeze_commit=freeze_commit,
            materialization_receipt=receipt_validation,
            clean_resolution=clean_resolution,
            replay_artifact=predictions,
            replay_artifact_path=preregistration["execution_protocol"]["output_policy"][
                "replay_file"
            ],
        )
        _require_safe_final_evidence(evidence)
        evidence_payload = contract.artifact_json_bytes(evidence)
        attempt.write(prediction_payload, evidence_payload)
        return cast(dict[str, Any], evidence)
    finally:
        attempt.close()


def _load_contract_module() -> ModuleType:
    clean_source = _canonical_existing_path(
        SRC,
        "clean contract source root",
        expect_directory=True,
    )
    _require_clean_source_absent_from_sys_path(clean_source)
    package_name = CONTRACT_MODULE.partition(".")[0]
    if any(
        name == package_name or name.startswith(package_name + ".")
        for name in sys.modules
    ):
        raise RuntimeError("clean contract package was imported before authentication")
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(clean_source))
        module = importlib.import_module(CONTRACT_MODULE)
    finally:
        sys.path[:] = original_path
    expected_origin = clean_source.joinpath(*CONTRACT_MODULE.split(".")).with_suffix(
        ".py"
    )
    actual_origin = _canonical_module_origin(CONTRACT_MODULE, module)
    if actual_origin != expected_origin:
        raise RuntimeError("clean contract module origin mismatch")
    return module


def _resolve_cli_paths(args: argparse.Namespace) -> dict[str, Path]:
    existing_files = {
        "preregistration": args.preregistration,
        "materialization_receipt": args.materialization_receipt,
    }
    existing_directories = {
        "controller_root": args.controller_repository_root,
        "base_model_root": args.clean_base_model_dir,
        "adapter_root": args.clean_adapter_dir,
        "repository_root": args.clean_repository_root,
    }
    absent_outputs = {
        "predictions_output": args.predictions_output,
        "result_output": args.result_output,
    }
    resolved: dict[str, Path] = {}
    for role, value in existing_files.items():
        resolved[role] = _canonical_existing_path(value, role, expect_directory=False)
    for role, value in existing_directories.items():
        resolved[role] = _canonical_existing_path(value, role, expect_directory=True)
    for role, value in absent_outputs.items():
        resolved[role] = _canonical_absent_output(value, role)
    if (
        len(
            {
                resolved["base_model_root"],
                resolved["adapter_root"],
                resolved["repository_root"],
            }
        )
        != 3
    ):
        raise RuntimeError("clean component roots must be distinct")
    return resolved


def _canonical_existing_path(
    value: object,
    label: str,
    *,
    expect_directory: bool,
) -> Path:
    raw = _absolute_lexical_path(value, label)
    _require_non_reparse_chain(raw, label)
    metadata = raw.lstat()
    if expect_directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{label} must be an existing directory")
    elif not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be an existing regular file")
    return raw.resolve(strict=True)


def _canonical_absent_output(value: object, label: str) -> Path:
    raw = _absolute_lexical_path(value, label)
    if not raw.name or os.path.lexists(raw):
        raise RuntimeError(f"{label} must be an absent file path")
    _require_non_reparse_chain(raw.parent, f"{label} parent")
    if not raw.parent.is_dir():
        raise RuntimeError(f"{label} parent must be an existing directory")
    return raw.parent.resolve(strict=True) / raw.name


def _absolute_lexical_path(value: object, label: str) -> Path:
    if not isinstance(value, Path):
        raise RuntimeError(f"invalid {label} path")
    if ".." in value.parts:
        raise RuntimeError(f"{label} path must not contain parent traversal")
    if value.anchor and not value.is_absolute():
        raise RuntimeError(f"{label} path has an invalid drive-relative anchor")
    return value if value.is_absolute() else Path.cwd() / value


def _require_non_reparse_chain(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must resolve from an absolute root")
    chain = (path, *path.parents)
    for current in reversed(chain):
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeError(f"{label} path component is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise RuntimeError(f"{label} path uses a symlink or reparse point")


def _force_offline_environment() -> None:
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


@contextlib.contextmanager
def _clean_execution_context(
    historical_hint: str,
    paths: Mapping[str, Path],
) -> Iterator[None]:
    """Use an empty cwd/cache so the historical Adapter hint cannot resolve."""

    old_cwd = Path.cwd()
    old_environment = {
        name: os.environ.get(name)
        for name in (
            "HF_HOME",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "XDG_CACHE_HOME",
        )
    }
    if not isinstance(historical_hint, str) or not historical_hint:
        raise RuntimeError("historical Adapter base hint is not locked")
    with tempfile.TemporaryDirectory(prefix="fp32-attached-replay-") as temporary:
        temporary_root = Path(temporary).resolve()
        cache_root = temporary_root / "cache"
        cache_root.mkdir()
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
        os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
        os.chdir(temporary_root)
        try:
            hinted = temporary_root / PurePath(historical_hint)
            if hinted.exists() or hinted.resolve(strict=False) in {
                paths["base_model_root"],
                paths["adapter_root"],
            }:
                raise RuntimeError("historical Adapter base hint unexpectedly resolves")
            yield
        finally:
            os.chdir(old_cwd)
            for name, value in old_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


@contextlib.contextmanager
def _load_clean_helpers(repository_root: Path) -> Iterator[CleanHelpers]:
    """Load execution helpers under an alias rooted only in the clean checkout."""

    package_root = repository_root / "src" / "fullcycle_bridge"
    init_path = package_root / "__init__.py"
    _require_regular_file(init_path, "clean package initializer")
    existing = [
        name
        for name in sys.modules
        if name == CLEAN_PACKAGE_ALIAS or name.startswith(CLEAN_PACKAGE_ALIAS + ".")
    ]
    if existing:
        raise RuntimeError("clean execution package alias is already loaded")
    spec = importlib.util.spec_from_file_location(
        CLEAN_PACKAGE_ALIAS,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load clean execution package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[CLEAN_PACKAGE_ALIAS] = package
    try:
        spec.loader.exec_module(package)
        consumer = importlib.import_module(f"{CLEAN_PACKAGE_ALIAS}.consumer")
        compiler = importlib.import_module(
            f"{CLEAN_PACKAGE_ALIAS}.tool_router_decision_compilation"
        )
        tool_router = importlib.import_module(f"{CLEAN_PACKAGE_ALIAS}.tool_router")
        sft = importlib.import_module(f"{CLEAN_PACKAGE_ALIAS}.tool_router_sft")
        for module in (consumer, compiler, tool_router, sft):
            module_path = Path(str(module.__file__)).resolve()
            if not module_path.is_relative_to(repository_root):
                raise RuntimeError("execution helper escaped clean repository")
        yield CleanHelpers(
            package_alias=CLEAN_PACKAGE_ALIAS,
            canonical_json_bytes=consumer.canonical_json_bytes,
            compile_decision=compiler.compile_decision,
            fixture_digest=tool_router.fixture_digest,
            load_fixture=tool_router.load_fixture,
            render_user_payload=sft.render_user_payload,
        )
    finally:
        for name in list(sys.modules):
            if name == CLEAN_PACKAGE_ALIAS or name.startswith(
                CLEAN_PACKAGE_ALIAS + "."
            ):
                sys.modules.pop(name, None)


def _load_runtime_modules() -> RuntimeModules:
    _force_offline_environment()
    clean_source = SRC.resolve(strict=True)
    _require_clean_source_absent_from_sys_path(clean_source)
    accelerate = _import_external_runtime_module("accelerate")
    peft = _import_external_runtime_module("peft")
    torch = _import_external_runtime_module("torch")
    transformers = _import_external_runtime_module("transformers")
    hub = _import_external_runtime_module("huggingface_hub")
    safetensors = _import_external_runtime_module("safetensors")
    tokenizers = _import_external_runtime_module("tokenizers")
    tuners = _import_external_runtime_module("peft.tuners.tuners_utils")
    return RuntimeModules(
        accelerate=accelerate,
        peft=peft,
        torch=torch,
        transformers=transformers,
        hub_version=str(hub.__version__),
        safetensors_version=str(safetensors.__version__),
        tokenizers_version=str(tokenizers.__version__),
        AutoConfig=transformers.AutoConfig,
        AutoModelForCausalLM=transformers.AutoModelForCausalLM,
        AutoTokenizer=transformers.AutoTokenizer,
        PeftModel=peft.PeftModel,
        BaseTunerLayer=tuners.BaseTunerLayer,
    )


def _import_external_runtime_module(name: str) -> ModuleType:
    module = importlib.import_module(name)
    origin = _canonical_module_origin(name, module)
    clean_root = ROOT.resolve(strict=True)
    if origin == clean_root or origin.is_relative_to(clean_root):
        raise RuntimeError(f"runtime module resolved inside clean repository: {name}")
    return module


def _canonical_module_origin(name: str, module: ModuleType) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"runtime module has no regular origin: {name}")
    return _canonical_existing_path(
        Path(value),
        f"{name} module origin",
        expect_directory=False,
    )


def _require_clean_source_absent_from_sys_path(clean_source: Path) -> None:
    for entry in sys.path:
        candidate = Path.cwd() if entry == "" else Path(entry)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved == clean_source:
            raise RuntimeError("clean source root must not remain on sys.path")


def _execute_model_replay(
    *,
    preregistration: Mapping[str, Any],
    authenticated: Any,
    base_model_root: Path,
    adapter_root: Path,
    repository_root: Path,
    helpers: CleanHelpers,
    consume_attempt: Callable[[], None],
    runtime: RuntimeModules | None = None,
) -> dict[str, Any]:
    runtime = _load_runtime_modules() if runtime is None else runtime
    protocol = preregistration["execution_protocol"]
    generation = protocol["generation"]
    expected_environment = authenticated.manifest["components"]["environment"][
        "recorded_environment"
    ]
    observed_environment = _observe_environment(runtime)
    if observed_environment != expected_environment:
        raise RuntimeError("exact execution environment mismatch")
    _verify_offline_environment()

    evaluation_path = (
        repository_root / preregistration["source_lineage"]["evaluation"]["path"]
    )
    prompt_path = (
        repository_root
        / authenticated.manifest["resolution_contract"]["repository_source_paths"][
            "prompt"
        ]
    )
    evaluation = helpers.load_fixture(evaluation_path)
    prompt = _read_regular_file(prompt_path, label="prompt").decode("utf-8")
    expected_order = preregistration["source_lineage"]["evaluation"]["order"]
    if (
        len(evaluation) != EXPECTED_RECORDS
        or [record["example_id"] for record in evaluation] != expected_order
        or helpers.fixture_digest(evaluation)
        != preregistration["source_lineage"]["evaluation"]["canonical_digest"]
    ):
        raise RuntimeError("clean evaluation input drift")

    torch = runtime.torch
    seed = generation["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = runtime.AutoTokenizer.from_pretrained(
        base_model_root,
        local_files_only=True,
        revision=authenticated.manifest["components"]["tokenizer"]["revision"],
    )
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = int(torch.cuda.memory_allocated())
    caps = preregistration["resource_caps"]
    if allocated_before > caps["memory_allocated_before_load_bytes_max"]:
        raise RuntimeError("fresh-load CUDA precondition failed")

    model: Any | None = None
    outputs: list[dict[str, Any]] = []
    generate_calls = 0
    elapsed_seconds = 0.0
    peak_gpu_memory_bytes = 0
    precision_audit: dict[str, Any] = {}
    try:
        consume_attempt()
        model, precision_audit = _load_fp32_attached_model(
            base_model_root,
            adapter_root,
            preregistration,
            runtime,
        )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for record in evaluation:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": helpers.render_user_payload(record),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(rendered, str):
                raise RuntimeError("tokenizer returned a non-text rendered prompt")
            encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
            input_token_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=generation["max_new_tokens"],
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generate_calls += 1
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            output_token_ids = [int(value) for value in new_tokens.tolist()]
            raw_output = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()
            compilation = _compile_observed_output(raw_output, helpers)
            outputs.append(
                {
                    "example_id": record["example_id"],
                    "rendered_prompt_sha256": _sha256(rendered.encode("utf-8")),
                    "input_token_count": len(input_token_ids),
                    "input_token_ids_sha256": _token_ids_sha256(input_token_ids),
                    "output_token_count": len(output_token_ids),
                    "output_token_ids_sha256": _token_ids_sha256(output_token_ids),
                    "raw_output": raw_output,
                    "raw_output_utf8_sha256": _sha256(raw_output.encode("utf-8")),
                    **compilation,
                }
            )
            del encoded, generated, new_tokens
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = int(torch.cuda.memory_allocated())
    if generate_calls != EXPECTED_RECORDS or len(outputs) != EXPECTED_RECORDS:
        raise RuntimeError("one-shot replay did not complete exactly twenty calls")
    performance = {
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "memory_allocated_before_load_bytes": allocated_before,
        "memory_allocated_after_release_bytes": allocated_after,
    }
    return {
        "environment": observed_environment,
        "precision_audit": precision_audit,
        "run": {
            "run_id": preregistration["execution_protocol"]["run_id"],
            "candidate_id": preregistration["candidate_id"],
            "fresh_model_loads": 1,
            "full_eval_replay_runs": 1,
            "generate_calls": generate_calls,
            "retries": 0,
            "warmup_calls": 0,
            "fallback_paths_used": 0,
            "completed": True,
            "historical_adapter_path_used": False,
            "network_used_during_execution": False,
        },
        "performance": performance,
        "outputs": outputs,
    }


def _load_fp32_attached_model(
    base_model_root: Path,
    adapter_root: Path,
    preregistration: Mapping[str, Any],
    runtime: RuntimeModules,
) -> tuple[Any, dict[str, Any]]:
    generation = preregistration["execution_protocol"]["generation"]
    model_config = runtime.AutoConfig.from_pretrained(
        base_model_root,
        local_files_only=True,
    )
    if not getattr(model_config, "use_sliding_window", False):
        model_config.sliding_window = None
    base_model = runtime.AutoModelForCausalLM.from_pretrained(
        base_model_root,
        config=model_config,
        local_files_only=True,
        torch_dtype=runtime.torch.float32,
        attn_implementation=generation["attn_implementation"],
    )
    model = runtime.PeftModel.from_pretrained(
        base_model,
        adapter_root,
        local_files_only=True,
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).to("cuda")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    _verify_generation_semantics(model, generation, runtime.torch)
    precision = _precision_audit(model, runtime)
    precision["lora_dropout"] = _lora_dropout_audit(model, runtime)
    precision["autocast_adapter_dtype"] = True
    precision["attached_execution_form"] = "attached_factorized_lora"
    if not _precision_protocol_passed(precision):
        raise RuntimeError("FP32 attached precision protocol drift")
    return model, precision


def _verify_generation_semantics(
    model: Any,
    protocol: Mapping[str, Any],
    torch: ModuleType,
) -> None:
    generation = model.generation_config
    if (
        model.training
        or protocol["attn_implementation"] != "sdpa"
        or protocol["do_sample"] is not False
        or protocol["max_new_tokens"] != 256
        or protocol["use_cache"] is not True
        or generation.repetition_penalty != 1.1
        or generation.eos_token_id != [151645, 151643]
        or generation.pad_token_id != 151643
        or generation.temperature is not None
        or generation.top_p is not None
        or generation.top_k is not None
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.is_autocast_enabled()
    ):
        raise RuntimeError("frozen greedy SDPA generation semantics drift")


def _precision_audit(model: Any, runtime: RuntimeModules) -> dict[str, Any]:
    base_parameters: list[tuple[str, Any]] = []
    adapter_parameters: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        destination = adapter_parameters if ".lora_" in name else base_parameters
        destination.append((name, parameter))
    causal = model.get_base_model()
    return {
        "base_parameters": _tensor_inventory(base_parameters),
        "adapter_parameters": _tensor_inventory(adapter_parameters),
        "floating_buffers": _tensor_inventory(model.named_buffers()),
        "lora_target_modules": sum(
            isinstance(module, runtime.BaseTunerLayer) for module in model.modules()
        ),
        "lora_parameter_tensors": len(adapter_parameters),
        "adapter_parameters_finite": all(
            bool(runtime.torch.isfinite(parameter).all())
            for _, parameter in adapter_parameters
            if parameter.is_floating_point()
        ),
        "active_adapters": list(model.active_adapters),
        "is_peft_model": isinstance(model, runtime.PeftModel),
        "input_output_embeddings_tied": (
            causal.get_input_embeddings().weight
            is causal.get_output_embeddings().weight
        ),
        "attn_implementation": causal.config._attn_implementation,
        "attention_class": causal.model.layers[0].self_attn.__class__.__name__,
        "output_attentions": causal.config.output_attentions,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "training": model.training,
        "autocast_enabled": runtime.torch.is_autocast_enabled(),
    }


def _tensor_inventory(tensors: Any) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    devices: dict[str, int] = {}
    floating_tensors = 0
    floating_elements = 0
    for _name, tensor in tensors:
        if not tensor.is_floating_point():
            continue
        elements = int(tensor.numel())
        dtype = str(tensor.dtype).removeprefix("torch.")
        device = str(tensor.device)
        floating_tensors += 1
        floating_elements += elements
        dtypes[dtype] = dtypes.get(dtype, 0) + elements
        devices[device] = devices.get(device, 0) + elements
    return {
        "floating_tensors": floating_tensors,
        "floating_elements": floating_elements,
        "dtypes": dict(sorted(dtypes.items())),
        "devices": dict(sorted(devices.items())),
    }


def _lora_dropout_audit(model: Any, runtime: RuntimeModules) -> dict[str, int]:
    modules: list[Any] = []
    for module in model.modules():
        if isinstance(module, runtime.BaseTunerLayer):
            modules.extend(getattr(module, "lora_dropout").values())
    return {
        "modules": len(modules),
        "training_modules": sum(module.training for module in modules),
    }


def _precision_protocol_passed(value: Mapping[str, Any]) -> bool:
    return (
        _inventory_is_float32_cuda(value.get("base_parameters"), EXPECTED_BASE_ELEMENTS)
        and _inventory_is_float32_cuda(
            value.get("adapter_parameters"), EXPECTED_ADAPTER_ELEMENTS
        )
        and _inventory_is_float32_cuda(
            value.get("floating_buffers"), EXPECTED_BUFFER_ELEMENTS
        )
        and value.get("lora_target_modules") == EXPECTED_LORA_TARGETS
        and value.get("lora_parameter_tensors") == EXPECTED_LORA_PARAMETER_TENSORS
        and value.get("adapter_parameters_finite") is True
        and value.get("active_adapters") == ["default"]
        and value.get("is_peft_model") is True
        and value.get("input_output_embeddings_tied") is True
        and value.get("attn_implementation") == "sdpa"
        and value.get("attention_class") == "Qwen2Attention"
        and value.get("output_attentions") is False
        and value.get("hf_device_map") is None
        and value.get("training") is False
        and value.get("autocast_enabled") is False
        and value.get("lora_dropout")
        == {"modules": EXPECTED_LORA_TARGETS, "training_modules": 0}
        and value.get("autocast_adapter_dtype") is True
        and value.get("attached_execution_form") == "attached_factorized_lora"
    )


def _inventory_is_float32_cuda(value: object, expected_elements: int) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("floating_tensors"), int)
        and value["floating_tensors"] > 0
        and value.get("floating_elements") == expected_elements
        and value.get("dtypes") == {"float32": expected_elements}
        and value.get("devices") == {"cuda:0": expected_elements}
    )


def _observe_environment(runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is required")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(runtime.transformers.__version__),
        "peft": str(runtime.peft.__version__),
        "accelerate": str(runtime.accelerate.__version__),
        "huggingface_hub": runtime.hub_version,
        "safetensors": runtime.safetensors_version,
        "tokenizers": runtime.tokenizers_version,
        "device": "cuda",
        "gpu": str(properties.name),
        "gpu_vram_bytes": int(properties.total_memory),
        "compute_capability": f"{capability[0]}.{capability[1]}",
    }


def _observe_static_environment() -> dict[str, str]:
    distributions = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
    }
    result = {"python": platform.python_version()}
    for key, distribution in distributions.items():
        try:
            result[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required distribution is missing: {distribution}"
            ) from exc
    return result


def _require_static_environment(
    observed: Mapping[str, str],
    expected: Mapping[str, Any],
) -> None:
    if expected.get("device") != "cuda":
        raise RuntimeError("frozen environment no longer requires CUDA")
    if any(expected.get(name) != value for name, value in observed.items()):
        raise RuntimeError("static execution environment mismatch")


def _verify_offline_environment() -> None:
    if any(
        os.environ.get(name) != value for name, value in OFFLINE_ENVIRONMENT.items()
    ):
        raise RuntimeError("offline execution flags were not retained")


def _validate_freeze_commit(value: object) -> str:
    if not isinstance(value, str) or GIT_COMMIT_PATTERN.fullmatch(value) is None:
        raise RuntimeError(
            "--freeze-commit must be exactly 40 lowercase hex characters"
        )
    return value


def _clean_repository_head(repository_root: Path) -> str:
    head_result = _run_clean_git(repository_root, "rev-parse", "--verify", "HEAD")
    head = head_result.stdout.strip()
    if GIT_COMMIT_PATTERN.fullmatch(head) is None:
        raise RuntimeError("clean repository returned an invalid HEAD")
    status_result = _run_clean_git(
        repository_root,
        "status",
        "--porcelain=v1",
        "--ignored",
        "--untracked-files=all",
    )
    if status_result.stdout:
        raise RuntimeError("clean repository contains tracked or untracked changes")
    return head


def _run_clean_git(
    repository_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_GLOBAL"] = "NUL"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    try:
        return subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(repository_root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot authenticate clean repository state") from exc


def _validate_clean_protocol_sources(
    preregistration: Mapping[str, Any],
    repository_root: Path,
) -> None:
    lineage = preregistration["source_lineage"]
    records = list(lineage["protocol_sources"].values()) + [lineage["model_downloader"]]
    for record in records:
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("invalid clean protocol source binding")
        payload = _read_regular_file(
            _root_relative_path(repository_root, relative),
            label="clean protocol source",
        )
        if _sha256(payload) != expected:
            raise RuntimeError("clean protocol source hash mismatch")


def _require_runner_from_clean_repository(repository_root: Path) -> None:
    runner_root = ROOT.resolve(strict=True)
    if repository_root != runner_root:
        raise RuntimeError(
            "runner must execute from the supplied clean repository root"
        )


def _validate_materialization_path_bindings(
    preregistration: Mapping[str, Any],
    receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    materialization = preregistration.get("materialization_protocol")
    if not isinstance(materialization, Mapping):
        raise RuntimeError("materialization protocol is missing")
    destination_policy = materialization.get("destination_policy")
    receipt_policy = materialization.get("receipt_policy")
    if not isinstance(destination_policy, Mapping) or not isinstance(
        receipt_policy, Mapping
    ):
        raise RuntimeError("materialization path policy is missing")
    children = destination_policy.get("children")
    if not isinstance(children, Mapping) or set(children) != {
        "repository",
        "base_model_and_tokenizer",
    }:
        raise RuntimeError("materialization destination children are invalid")
    destination = receipt.get("destination")
    if not isinstance(destination, Mapping):
        raise RuntimeError("materialization receipt destination is missing")
    destination_id = destination.get("destination_id")
    if (
        not isinstance(destination_id, str)
        or DESTINATION_ID_PATTERN.fullmatch(destination_id) is None
    ):
        raise RuntimeError("materialization receipt destination id is invalid")

    controller_root = paths["controller_root"]
    clean_parent = _join_policy_path(
        controller_root,
        destination_policy.get("parent_relative_to_repository"),
        "clean-location parent",
    )
    destination_root = clean_parent / destination_id
    expected_roots = {
        "repository_root": _join_policy_path(
            destination_root,
            children.get("repository"),
            "repository child",
        ),
        "base_model_root": _join_policy_path(
            destination_root,
            children.get("base_model_and_tokenizer"),
            "base-model child",
        ),
    }
    expected_roots["adapter_root"] = _join_policy_path(
        expected_roots["repository_root"],
        destination_policy.get("adapter_root_relative_to_repository"),
        "adapter root",
    )
    for role, expected in expected_roots.items():
        if paths[role] != expected:
            raise RuntimeError(f"{role} does not match the materialization receipt")

    expected_receipt_parent = _join_policy_path(
        controller_root,
        receipt_policy.get("output_root_relative_to_repository"),
        "receipt parent",
    )
    receipt_path = paths["materialization_receipt"]
    if receipt_path.parent != expected_receipt_parent or receipt_path.suffix != ".json":
        raise RuntimeError(
            "materialization receipt must be a direct JSON child of the "
            "controller receipt root"
        )


def _join_policy_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise RuntimeError(f"invalid {label} relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError(f"invalid {label} relative path")
    return root.joinpath(*relative.parts)


def _root_relative_path(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    candidate = (root / normalized).resolve(strict=False)
    if candidate == root or not candidate.is_relative_to(root):
        raise RuntimeError("repository-relative path escaped its clean root")
    return candidate


def _require_clean_resolution(
    resolution: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    if resolution.get("resolved") is not True:
        raise RuntimeError("clean package did not resolve exactly")
    if receipt.get("materialization_passed") is not True:
        raise RuntimeError("materialization receipt did not validate")
    destination = receipt.get("destination")
    transport = receipt.get("transport")
    if not isinstance(destination, Mapping) or not isinstance(transport, Mapping):
        raise RuntimeError("materialization safety receipts are missing")
    for unsafe in (
        "absolute_paths_recorded",
        "symlinks_used",
        "reparse_points_used",
        "hardlinks_used",
        "overwrite_used",
    ):
        if destination.get(unsafe) is not False:
            raise RuntimeError(f"unsafe materialization destination flag: {unsafe}")
    for unsafe in (
        "network_used_during_execution",
        "alternate_remote_used",
        "alternate_revision_fallback_used",
        "historical_adapter_base_path_used",
    ):
        if transport.get(unsafe) is not False:
            raise RuntimeError(f"unsafe materialization transport flag: {unsafe}")


def _require_safe_final_evidence(value: Mapping[str, Any]) -> None:
    if value.get("runtime_eligible") is not False:
        raise RuntimeError("unsafe Runtime eligibility in final evidence")
    derived = value.get("derived_claims")
    constraints = value.get("constraints")
    next_action = value.get("locked_next_action")
    if not isinstance(derived, Mapping):
        raise RuntimeError("final evidence derived claims are missing")
    if not isinstance(constraints, Mapping):
        raise RuntimeError("final evidence constraints are missing")
    if not isinstance(next_action, Mapping):
        raise RuntimeError("final evidence safety sections are missing")
    for name in (
        "offline_artifact_eligible",
        "portable_package_eligible",
        "preferred_offline_candidate",
        "serving_readiness_established",
        "artifact_promotion_allowed",
        "merged_artifact_allowed",
        "runtime_eligible",
    ):
        if derived.get(name) is not False:
            raise RuntimeError(f"unsafe final evidence flag: {name}")
    if (
        next_action.get("artifact_promotion_allowed") is not False
        or next_action.get("runtime_integration_allowed") is not False
    ):
        raise RuntimeError("unsafe next-action authority")
    for name in (
        "artifact_promotion",
        "serving_integration",
        "runtime_integration",
        "provider_integration",
        "mcp_integration",
        "desktop_integration",
        "merged_weight_creation",
    ):
        if constraints.get(name) is not False:
            raise RuntimeError(f"unsafe final constraint: {name}")


def _parse_strict_decision(raw_output: str) -> Mapping[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise RuntimeError(f"non-finite JSON constant: {value}")

    try:
        parsed = json.loads(
            raw_output,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("raw replay output is not strict JSON") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError("raw replay output is not a JSON object")
    _require_finite_tree(parsed, "$.raw_output")
    return parsed


def _compile_observed_output(
    raw_output: str,
    helpers: CleanHelpers,
) -> dict[str, Any]:
    try:
        parsed = _parse_strict_decision(raw_output)
        compiler_input = helpers.canonical_json_bytes(parsed)
        compiled = helpers.compile_decision(parsed)
        compiled_bytes = helpers.canonical_json_bytes(compiled)
    except Exception as exc:
        return {
            "compiler_valid": False,
            "compiler_input_canonical_sha256": None,
            "compiled_output": None,
            "compiled_output_canonical_sha256": None,
            "compiler_changed_fields": [],
            "compilation_error": str(getattr(exc, "code", type(exc).__name__)),
        }
    changed_fields = [
        f"$.{name}" for name in sorted(parsed) if parsed[name] != compiled[name]
    ]
    return {
        "compiler_valid": True,
        "compiler_input_canonical_sha256": _sha256(compiler_input),
        "compiled_output": compiled,
        "compiled_output_canonical_sha256": _sha256(compiled_bytes),
        "compiler_changed_fields": changed_fields,
        "compilation_error": None,
    }


def _require_finite_tree(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON number at {path}")
    if isinstance(value, Mapping):
        for name, item in value.items():
            _require_finite_tree(item, f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")


def _token_ids_sha256(values: Sequence[int]) -> str:
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise RuntimeError("invalid token sequence")
    return _sha256(",".join(str(value) for value in values).encode("ascii"))


class _ExclusiveOutputAttempt:
    """Reserve formal outputs exactly when the one-shot model attempt is consumed."""

    def __init__(self, predictions: Path, result: Path) -> None:
        self._predictions_path = predictions
        self._result_path = result
        self._predictions_stream: BinaryIO | None = None
        self._result_stream: BinaryIO | None = None
        self._written = False

    @property
    def consumed(self) -> bool:
        return self._predictions_stream is not None and self._result_stream is not None

    def consume(self) -> None:
        if self.consumed:
            raise RuntimeError("formal execution attempt was consumed more than once")
        if self._predictions_path.parent != self._result_path.parent:
            raise RuntimeError("formal output reservation parents changed")
        parent_metadata = _require_safe_directory(
            self._predictions_path.parent,
            "formal output parent",
        )
        parent_identity = _stat_identity(parent_metadata)
        self._predictions_stream = self._predictions_path.open("xb")
        prediction_metadata = os.fstat(self._predictions_stream.fileno())
        if (
            not stat.S_ISREG(prediction_metadata.st_mode)
            or prediction_metadata.st_size != 0
        ):
            self._predictions_stream.close()
            self._predictions_stream = None
            raise RuntimeError("invalid predictions output reservation")
        prediction_identity = _stat_identity(prediction_metadata)
        try:
            self._result_stream = self._result_path.open("xb")
        except BaseException as exc:
            self._predictions_stream.close()
            self._predictions_stream = None
            try:
                _remove_unconsumed_reservation(
                    self._predictions_path,
                    expected_parent=self._result_path.parent,
                    expected_parent_identity=parent_identity,
                    expected_file_identity=prediction_identity,
                )
            except RuntimeError as cleanup_error:
                raise cleanup_error from exc
            raise

    def write(self, predictions_payload: bytes, result_payload: bytes) -> None:
        predictions_stream = self._predictions_stream
        result_stream = self._result_stream
        if predictions_stream is None or result_stream is None or self._written:
            raise RuntimeError("formal output reservation is not writable")
        predictions_stream.write(predictions_payload)
        predictions_stream.flush()
        os.fsync(predictions_stream.fileno())
        result_stream.write(result_payload)
        result_stream.flush()
        os.fsync(result_stream.fileno())
        self._written = True

    def close(self) -> None:
        if self._result_stream is not None:
            self._result_stream.close()
        if self._predictions_stream is not None:
            self._predictions_stream.close()


def _remove_unconsumed_reservation(
    path: Path,
    *,
    expected_parent: Path,
    expected_parent_identity: tuple[int, int, int],
    expected_file_identity: tuple[int, int, int],
) -> None:
    if path.parent != expected_parent:
        raise RuntimeError("unconsumed reservation parent changed")
    parent_metadata = _require_safe_directory(expected_parent, "formal output parent")
    if _stat_identity(parent_metadata) != expected_parent_identity:
        raise RuntimeError("unconsumed reservation parent identity changed")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("unconsumed reservation disappeared") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
        or _stat_identity(metadata) != expected_file_identity
        or metadata.st_size != 0
    ):
        raise RuntimeError("unconsumed reservation identity changed")
    path.unlink()
    if os.path.lexists(path):
        raise RuntimeError("unconsumed reservation cleanup did not complete")


def _validate_output_targets(predictions: Path, result: Path) -> None:
    if predictions == result:
        raise RuntimeError("predictions and result outputs must be distinct")
    for path in (predictions, result):
        if os.path.lexists(path):
            raise RuntimeError("formal output target must not already exist")
        parent = path.parent
        if not parent.is_dir():
            raise RuntimeError("formal output parent must already exist")
        _require_safe_directory_chain(parent)


def _validate_formal_output_policy(
    preregistration: Mapping[str, Any],
    controller_root: Path,
    predictions: Path,
    result: Path,
) -> None:
    policy = preregistration["execution_protocol"]["output_policy"]
    expected_parent = _join_policy_path(
        controller_root,
        policy.get("required_parent_relative_to_controller_repository"),
        "formal output parent",
    )
    if (
        policy.get("root_authority") != "caller_supplied"
        or policy.get("exclusive_create") is not True
        or policy.get("machine_paths_recorded") is not False
        or predictions.parent != result.parent
        or predictions.parent != expected_parent
        or predictions.name != policy.get("replay_file")
        or result.name != policy.get("evidence_file")
    ):
        raise RuntimeError("formal output paths violate the frozen output policy")


def _write_exclusive_pair(
    predictions_path: Path,
    predictions_payload: bytes,
    result_path: Path,
    result_payload: bytes,
) -> None:
    with predictions_path.open("xb") as predictions_stream:
        with result_path.open("xb") as result_stream:
            predictions_stream.write(predictions_payload)
            result_stream.write(result_payload)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    before = _require_regular_file(path, label)
    if before.st_size > MAX_INPUT_BYTES:
        raise RuntimeError(f"{label} exceeds the input byte limit")
    with path.open("rb") as stream:
        handle = os.fstat(stream.fileno())
        if not stat.S_ISREG(handle.st_mode):
            raise RuntimeError(f"{label} is not a regular file handle")
        payload = stream.read(MAX_INPUT_BYTES + 1)
        after_handle = os.fstat(stream.fileno())
    after = _require_regular_file(path, label)
    if len(payload) > MAX_INPUT_BYTES:
        raise RuntimeError(f"{label} exceeds the input byte limit")
    signatures = {
        _stat_signature(before),
        _stat_signature(handle),
        _stat_signature(after_handle),
        _stat_signature(after),
    }
    if len(signatures) != 1 or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return payload


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"missing or unreadable {label}") from exc
    if not stat.S_ISREG(value.st_mode) or _is_reparse(value):
        raise RuntimeError(f"unsafe {label}")
    return value


def _require_safe_directory_chain(path: Path) -> None:
    resolved = path.resolve(strict=True)
    for current in (resolved, *resolved.parents):
        value = current.lstat()
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
        ):
            raise RuntimeError("unsafe output directory chain")


def _require_safe_directory(path: Path, label: str) -> os.stat_result:
    _require_safe_directory_chain(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"missing {label}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _is_reparse(metadata)
    ):
        raise RuntimeError(f"unsafe {label}")
    return metadata


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(attributes & REPARSE_POINT_ATTRIBUTE)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode))


def _parse_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not UTF-8") from exc
    value = _parse_strict_json(text, label)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _parse_strict_json(text: str, label: str) -> object:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate key in {label}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise RuntimeError(f"non-finite constant in {label}: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed {label}") from exc
    _require_finite_tree(value, f"$.{label}")
    return value


def _json_bytes(value: object) -> bytes:
    _require_finite_tree(value, "$")
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


if __name__ == "__main__":
    raise SystemExit(main())
