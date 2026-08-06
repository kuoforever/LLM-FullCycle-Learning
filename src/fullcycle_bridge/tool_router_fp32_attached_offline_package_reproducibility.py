"""Fail-closed contract for clean-location FP32 attached replay evidence.

This module is deliberately execution-neutral.  It authenticates frozen JSON
inputs, delegates exact package resolution to the already-frozen manifest
contract, compares a replay with the frozen FP32 output, and builds derived
evidence.  It never downloads files, loads a model, or writes an artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from .tool_router import ToolRouterValidationError
from .tool_router_decision_compilation import (
    DECISION_COMPILER_VERSION,
    compile_decision,
)
from .tool_router_fp32_attached_offline_package_manifest import (
    REPOSITORY_SOURCE_PATHS,
    validate_and_resolve_fp32_attached_offline_package,
    validate_fp32_attached_offline_package_manifest,
)

PREREGISTRATION_VERSION = 1
CONTRACT_VERSION = 1
RECEIPT_VERSION = 1
REPLAY_ARTIFACT_VERSION = 1
EVIDENCE_VERSION = 1

GATE_ID = "FC-MVP-001-fp32-attached-offline-package-reproducibility-v1"
EXPERIMENT_ID = "fc-mvp-001-fp32-attached-offline-package-reproducibility-v1"
PACKAGE_ID = "fc-mvp-001-fp32-attached-factorized-lora-package-v1"
CANDIDATE_ID = "fp32-attached-factorized-lora"
RUN_ID = "fp32-attached-clean-location-full-eval-r1"

MANIFEST_PATH = "baseline/fc-mvp-001-fp32-attached-offline-package-manifest-v1.json"
MANIFEST_SHA256 = (
    "sha256:4125f2eef2a4b8f07015169ac7fb77b830514e053a4624aa703e5f5a64943eb0"
)
REFERENCE_PREDICTIONS_PATH = (
    "baseline/tool-router-fp32-attached-remediation-v1-predictions.json"
)
REFERENCE_PREDICTIONS_SHA256 = (
    "sha256:382071f0689ce4ca41329d689f76fc4c4b06faa68769fb80c99181015e678115"
)
REFERENCE_EVIDENCE_PATH = "baseline/fc-mvp-001-fp32-attached-remediation-eval-v1.json"
REFERENCE_EVIDENCE_SHA256 = (
    "sha256:2dd17f6b1098490034f825d163f48f26eb4093d02f115424eb814cb2c925ad8e"
)
EVALUATION_PATH = "fixtures/tool_router_v1/eval.json"
EVALUATION_FILE_SHA256 = (
    "sha256:32ceee99ab1be5672313a8aead91a089238d007dcab1a31d9ca999f28cc24595"
)
EVALUATION_DIGEST = (
    "sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a"
)
REFERENCE_PREREGISTRATION_SHA256 = (
    "sha256:5e7b0665f97f5cee760637236f80039c4e621ae0f24915c0ac749d885a683c8b"
)
REPOSITORY_REMOTE_URL = (
    "https://github.com/kuoforever/reliable-agent-model-lifecycle.git"
)
MODEL_DOWNLOADER_PATH = "scripts/download_pinned_tool_router_model.py"
MODEL_DOWNLOADER_SHA256 = (
    "sha256:1d0d3321a55b185128de020f4b5a2a9c3ecc22f5abb0535c4712c4fd545d3a28"
)
ADAPTER_ROOT_RELATIVE_TO_REPOSITORY = "baseline/adapters/fc-mvp-001-lora-sft-v2"
ADAPTER_LFS_PATH = f"{ADAPTER_ROOT_RELATIVE_TO_REPOSITORY}/adapter_model.safetensors"
ADAPTER_LFS_OID = (
    "sha256:efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342"
)
ADAPTER_LFS_BYTES = 17_462_432

CONTRACT_SOURCE_PATH = (
    "src/fullcycle_bridge/tool_router_fp32_attached_offline_package_reproducibility.py"
)
MATERIALIZER_SOURCE_PATH = (
    "scripts/materialize_tool_router_fp32_attached_offline_package_reproducibility.py"
)
RUNNER_SOURCE_PATH = (
    "scripts/probe_tool_router_fp32_attached_offline_package_reproducibility.py"
)
PREREGISTRATION_PATH = (
    "configs/tool_router_fp32_attached_offline_package_reproducibility_v1.json"
)
ZERO_SHA256 = "sha256:" + "0" * 64

EXPECTED_RECORDS = 20
EXPECTED_EXAMPLE_ORDER = [f"eval-{index:03d}" for index in range(1, 21)]
MAX_ELAPSED_SECONDS = 153.98083879996556
MAX_PEAK_GPU_MEMORY_BYTES = 6_300_631_040
MAX_RESIDUAL_GPU_MEMORY_BYTES = 16_777_216
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024

PASS_CLASSIFICATION = (
    "fp32_attached_same_environment_clean_location_behavior_exactly_reproduced"
)
TRUST_ROOT_INVALID_CLASSIFICATION = "fp32_attached_reproducibility_trust_root_invalid"
MATERIALIZATION_FAILED_CLASSIFICATION = (
    "fp32_attached_clean_location_materialization_failed"
)
RESOLUTION_FAILED_CLASSIFICATION = "fp32_attached_clean_location_resolution_failed"
ENVIRONMENT_MISMATCH_CLASSIFICATION = (
    "fp32_attached_clean_location_environment_mismatch"
)
EXECUTION_FAILED_CLASSIFICATION = (
    "fp32_attached_clean_location_execution_contract_failed"
)
BEHAVIORAL_DRIFT_CLASSIFICATION = "fp32_attached_clean_location_behavioral_drift"
RESOURCE_EXCEEDED_CLASSIFICATION = (
    "fp32_attached_clean_location_resource_budget_exceeded"
)
BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION = (
    "fp32_attached_clean_location_behavioral_and_resource_failure"
)
PASS_NEXT_GATE_ID = "FC-MVP-001-fp32-attached-remote-revision-origin-attestation-v1"
FAILURE_NEXT_GATE_ID = (
    "FC-MVP-001-fp32-attached-offline-package-reproducibility-failure-classification-v1"
)

SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

EXPECTED_MANIFEST_SOURCE_HASHES = {
    "adapter_config": "sha256:8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16",
    "adapter_inspector_source": "sha256:3fa9dca9d5b309b9401be25dd3538ccbdf76df63d0eda67230a45152703c5452",
    "adapter_readme": "sha256:353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522",
    "adapter_weights": ADAPTER_LFS_OID,
    "canonical_json_source": "sha256:05cfe603d4786fb536cc1f99952a55fd211cc0fea2c210b32b575fefda9537d3",
    "decision_compiler_source": "sha256:16f162a84572c7f0782890aef5aafbaafa1862e14938fe08b0ea6e97efa05157",
    "manifest_builder_source": "sha256:7834a35854e14863de4312319fcf109681a14f42bf2d7eee3a385a1376427284",
    "manifest_contract_source": "sha256:8e7b09f914ab45bdbe4841ebf3c06eb75ce9eabf0d2ce9ba2cb8de3ca48d383d",
    "model_downloader_source": MODEL_DOWNLOADER_SHA256,
    "package_documentation": "sha256:a531b0e462aad15a1ec9eb001d05c8cf71b5a72bde66437a499d0c6efba9cb24",
    "package_init_source": "sha256:45cabb5da1c0e7c2c93ef045904cf4555b0c755baf1ec2eaf47330a1aab6008e",
    "prompt": "sha256:4a7d15063b0b074ef999c2848d0fc073a6cc00ed4999ea81f770e2e42cfa6d97",
    "remediation_preregistration": REFERENCE_PREREGISTRATION_SHA256,
    "sft_config": "sha256:110ada11d69f4e83c4b93da0304e62151059115487e90394d32835f6916365c8",
    "sft_helpers_source": "sha256:db881e5e5955341acb735416d93062a40cf512b63ec50eb8c196ddb4371bd020",
    "training_lock": "sha256:e6e23f51834b1815578368ce54c78034e72a7158395892e77fdf75594548931f",
    "upstream_review": "sha256:81977f318c6bcfed8d3844575dc245d4b94c2636a2359165f9aa5553c9b006f8",
    "validation_error_source": "sha256:bb3cda72585bc84bf0cf84c5736cafe29c8dfc8bca5a851d82ecfed35b1b883d",
}

GENERATION_CONTRACT = {
    "attention_backend_claim_scope": "transformers_high_level_dispatch_only",
    "attn_implementation": "sdpa",
    "autocast": False,
    "call_pad_token_source": "tokenizer.eos_token_id",
    "device": "cuda:0",
    "do_sample": False,
    "low_level_cuda_kernel_identity_claimed": False,
    "max_new_tokens": 256,
    "model_eos_token_ids": [151645, 151643],
    "model_pad_token_id": 151643,
    "repetition_penalty": 1.1,
    "seed": 20260803,
    "tf32": False,
    "torch_dtype": "float32",
    "use_cache": True,
}

PROTOCOL_SOURCE_PATHS = {
    "contract_source": CONTRACT_SOURCE_PATH,
    "materializer_source": MATERIALIZER_SOURCE_PATH,
    "runner_source": RUNNER_SOURCE_PATH,
}


class ReproducibilityContractError(ValueError):
    """One deterministic contract validation failure."""

    def __init__(self, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code} at {path}: {detail}")


@dataclass(frozen=True)
class LoadedPreregistration:
    """One raw-byte-bound preregistration snapshot."""

    data: dict[str, Any]
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class ManifestSourceBundle:
    """Exact parsed and raw sources required by the frozen manifest validator."""

    upstream_review: Mapping[str, Any]
    remediation_preregistration: Mapping[str, Any]
    sft_config: Mapping[str, Any]
    adapter_config: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    source_payloads: Mapping[str, bytes]


@dataclass(frozen=True)
class AuthenticatedInputs:
    """Authenticated immutable inputs used by resolution and comparison."""

    preregistration: dict[str, Any]
    manifest_payload: bytes
    manifest: dict[str, Any]
    manifest_validation: dict[str, Any]
    reference_predictions: dict[str, Any]
    reference_evidence: dict[str, Any]
    evaluation: list[dict[str, Any]]
    reference_outputs: tuple[dict[str, str], ...]


def canonical_json_bytes(value: object) -> bytes:
    """Return the repository canonical JSON encoding for any JSON value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def artifact_json_bytes(value: object) -> bytes:
    """Return the tracked artifact encoding used by existing gates."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return one prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def parse_strict_json_bytes(
    payload: bytes,
    *,
    path: str,
    max_bytes: int = MAX_JSON_BYTES,
) -> object:
    """Parse UTF-8 JSON while rejecting duplicates and non-finite values."""

    if not isinstance(payload, bytes):
        _fail("INVALID_JSON_PAYLOAD", path, type(payload).__name__)
    if not payload or len(payload) > max_bytes:
        _fail("INVALID_JSON_SIZE", path, str(len(payload)))
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail("INVALID_UTF8", path, str(exc))
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except ReproducibilityContractError:
        raise
    except json.JSONDecodeError as exc:
        _fail("MALFORMED_JSON", path, str(exc))
    _validate_finite_json(value, path)
    return value


def expected_preregistration(
    *,
    freeze_status: str,
    protocol_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the only accepted preregistration object for given source hashes."""

    if freeze_status not in {"draft", "frozen"}:
        _fail("INVALID_FREEZE_STATUS", "$.freeze_status", repr(freeze_status))
    if set(protocol_source_hashes) != set(PROTOCOL_SOURCE_PATHS):
        _fail(
            "INVALID_PROTOCOL_SOURCE_SET",
            "$.source_lineage.protocol_sources",
            repr(sorted(protocol_source_hashes)),
        )
    for name, digest in protocol_source_hashes.items():
        _validate_sha256(digest, f"$.source_lineage.protocol_sources.{name}.sha256")

    formal_authorized = freeze_status == "frozen"
    protocol_sources: dict[str, Any] = {}
    contract_symbols = [
        "load_and_validate_preregistration",
        "load_manifest_source_bundle",
        "authenticate_manifest_and_references",
        "resolve_clean_roots",
        "validate_materialization_receipt",
        "compare_behavioral_replay",
        "build_replay_artifact",
        "build_reproducibility_evidence",
        "classify_reproducibility_gates",
        "validate_reproducibility_evidence",
    ]
    for name, path in PROTOCOL_SOURCE_PATHS.items():
        record: dict[str, Any] = {
            "path": path,
            "sha256": protocol_source_hashes[name],
        }
        if name == "contract_source":
            record.update({"version": CONTRACT_VERSION, "symbols": contract_symbols})
        else:
            record["symbol"] = "main"
        protocol_sources[name] = record

    return {
        "preregistration_version": PREREGISTRATION_VERSION,
        "freeze_status": freeze_status,
        "gate_id": GATE_ID,
        "experiment_id": EXPERIMENT_ID,
        "package_id": PACKAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "run_count": 1,
        "source_lineage": {
            "repository": {
                "remote_url": REPOSITORY_REMOTE_URL,
                "freeze_commit_authority": "required_formal_cli_argument",
            },
            "manifest": {
                "path": MANIFEST_PATH,
                "sha256": MANIFEST_SHA256,
                "source_hashes": dict(sorted(EXPECTED_MANIFEST_SOURCE_HASHES.items())),
            },
            "reference_predictions": {
                "path": REFERENCE_PREDICTIONS_PATH,
                "sha256": REFERENCE_PREDICTIONS_SHA256,
            },
            "reference_evidence": {
                "path": REFERENCE_EVIDENCE_PATH,
                "sha256": REFERENCE_EVIDENCE_SHA256,
            },
            "evaluation": {
                "path": EVALUATION_PATH,
                "file_sha256": EVALUATION_FILE_SHA256,
                "canonical_digest": EVALUATION_DIGEST,
                "records": EXPECTED_RECORDS,
                "order": list(EXPECTED_EXAMPLE_ORDER),
            },
            "protocol_sources": protocol_sources,
            "model_downloader": {
                "path": MODEL_DOWNLOADER_PATH,
                "sha256": MODEL_DOWNLOADER_SHA256,
                "symbol": "main",
            },
            "adapter_lfs": {
                "path": ADAPTER_LFS_PATH,
                "oid": ADAPTER_LFS_OID,
                "bytes": ADAPTER_LFS_BYTES,
            },
        },
        "materialization_protocol": {
            "formal_execution_authorized": formal_authorized,
            "phase_order": [
                "fresh_repository_checkout",
                "exact_adapter_lfs_checkout",
                "pinned_base_snapshot_download",
                "strict_manifest_resolution",
                "offline_behavioral_execution",
            ],
            "destination_policy": {
                "root_authority": "caller_supplied",
                "must_not_exist_before_materialization": True,
                "exclusive_create": True,
                "parent_relative_to_repository": "work/clean-location",
                "must_be_direct_child_of_parent": True,
                "leaf_name_pattern": "[0-9a-f]{32}",
                "children": {
                    "repository": "repository",
                    "base_model_and_tokenizer": "base_model_and_tokenizer",
                },
                "adapter_root_relative_to_repository": (
                    ADAPTER_ROOT_RELATIVE_TO_REPOSITORY
                ),
                "absolute_paths_recorded_in_artifacts": False,
                "symlinks_allowed": False,
                "reparse_points_allowed": False,
                "hardlinks_allowed": False,
                "overwrite_allowed": False,
            },
            "repository_transport": {
                "method": "fresh_remote_git_checkout",
                "remote_url": REPOSITORY_REMOTE_URL,
                "exact_cli_freeze_commit_required": True,
                "git_fetch_required": True,
                "git_lfs_checkout_required": True,
                "submodules_allowed": False,
                "alternate_remote_allowed": False,
            },
            "base_transport": {
                "method": "pinned_huggingface_snapshot_download",
                "downloader_path": MODEL_DOWNLOADER_PATH,
                "downloader_sha256": MODEL_DOWNLOADER_SHA256,
                "destination_scoped_hf_home_required": True,
                "destination_scoped_cache_required": True,
                "alternate_revision_fallback_allowed": False,
            },
            "adapter_transport": {
                "method": "git_lfs_checkout_in_clean_repository",
                "root_relative_to_repository": (ADAPTER_ROOT_RELATIVE_TO_REPOSITORY),
                "weight_path": ADAPTER_LFS_PATH,
                "lfs_oid": ADAPTER_LFS_OID,
                "bytes": ADAPTER_LFS_BYTES,
                "second_adapter_copy_allowed": False,
            },
            "phase_authority": {
                "network_allowed_during_materialization": True,
                "download_allowed_during_materialization": True,
                "git_fetch_allowed_during_materialization": True,
                "network_allowed_during_execution": False,
                "local_files_only_during_execution": True,
                "historical_adapter_base_path_authoritative": False,
            },
            "receipt_policy": {
                "output_root_relative_to_repository": "work/test-fixtures",
                "protocol_freeze_commit_required": True,
                "source_code_hash_receipts_required": True,
                "clean_resolution_digest_required": True,
                "absolute_paths_prohibited": True,
                "write_before_model_execution": True,
            },
        },
        "execution_protocol": {
            "run_id": RUN_ID,
            "fresh_model_loads": 1,
            "full_eval_runs": 1,
            "generate_calls": EXPECTED_RECORDS,
            "retry_count": 0,
            "fixed_order": True,
            "attempt_consumption": {
                "consumed_when": (
                    "fresh_model_load_started_or_any_generate_call_started"
                ),
                "retry_allowed_before_consumption": True,
                "retry_allowed_after_consumption": False,
            },
            "root_authority": "materialization_receipt_clean_roots",
            "historical_adapter_base_path_used": False,
            "local_files_only": True,
            "network_enabled": False,
            "decision_compilation": (
                "compile_decision_v1_after_raw_parse_before_terminal_consumption"
            ),
            "generation": copy.deepcopy(GENERATION_CONTRACT),
            "candidate": {
                "base_checkpoint_storage_dtype": "bfloat16",
                "base_checkpoint_value_semantics": (
                    "unchanged_bf16_checkpoint_source_values_materialized_as_float32"
                ),
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": True,
                "execution_form": "attached_factorized_lora",
                "merge": False,
                "save_model": False,
                "save_tensors": False,
            },
            "output_policy": {
                "root_authority": "caller_supplied",
                "exclusive_create": True,
                "required_parent_relative_to_controller_repository": "baseline",
                "machine_paths_recorded": False,
                "replay_file": (
                    "tool-router-fp32-attached-offline-package-"
                    "reproducibility-v1-predictions.json"
                ),
                "evidence_file": (
                    "fc-mvp-001-fp32-attached-offline-package-reproducibility-v1.json"
                ),
                "raw_outputs_required": True,
                "summary_required": True,
                "model_artifact_save": False,
                "tensor_payload_save": False,
            },
        },
        "behavior_reference": {
            "scope": (
                "same_recorded_environment_exact_twenty_case_raw_and_compiled_"
                "output_reproduction"
            ),
            "reference_predictions_sha256": REFERENCE_PREDICTIONS_SHA256,
            "reference_evidence_sha256": REFERENCE_EVIDENCE_SHA256,
            "records": EXPECTED_RECORDS,
            "order": list(EXPECTED_EXAMPLE_ORDER),
            "raw_output_comparison": "exact_utf8_bytes",
            "raw_output_normalization_allowed": False,
            "compiler_symbol": "compile_decision",
            "compiler_version": DECISION_COMPILER_VERSION,
            "compiler_recomputed_for_reference_and_replay": True,
            "compiled_output_comparison": "exact_canonical_json_bytes",
            "metric_threshold_comparison_required": False,
        },
        "resource_caps": {
            "elapsed_seconds_max": MAX_ELAPSED_SECONDS,
            "peak_gpu_memory_bytes_max": MAX_PEAK_GPU_MEMORY_BYTES,
            "memory_allocated_before_load_bytes_max": (MAX_RESIDUAL_GPU_MEMORY_BYTES),
            "memory_allocated_after_release_bytes_max": (MAX_RESIDUAL_GPU_MEMORY_BYTES),
        },
        "acceptance_criteria": {
            "required_gates": [
                "metadata_validation",
                "materialization",
                "clean_location_resolution",
                "environment",
                "execution_contract",
                "behavioral_replay",
                "resources",
            ],
            "raw_outputs_exact": EXPECTED_RECORDS,
            "compiled_outputs_exact": EXPECTED_RECORDS,
            "remaining_blocking_findings_on_pass": [
                "remote_revision_origin_unverified"
            ],
        },
        "outcome_classifications": {
            "passed": PASS_CLASSIFICATION,
            "trust_root_invalid": TRUST_ROOT_INVALID_CLASSIFICATION,
            "materialization_failed": MATERIALIZATION_FAILED_CLASSIFICATION,
            "resolution_failed": RESOLUTION_FAILED_CLASSIFICATION,
            "environment_mismatch": ENVIRONMENT_MISMATCH_CLASSIFICATION,
            "execution_contract_failed": EXECUTION_FAILED_CLASSIFICATION,
            "behavioral_drift": BEHAVIORAL_DRIFT_CLASSIFICATION,
            "resource_exceeded": RESOURCE_EXCEEDED_CLASSIFICATION,
            "behavioral_and_resource_failed": (
                BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION
            ),
        },
        "outcome_next_actions": {
            "passed": {
                "gate_id": PASS_NEXT_GATE_ID,
                "action": (
                    "independently attest the pinned remote revision origin while "
                    "keeping all promotion serving and Runtime claims false"
                ),
            },
            "adverse": {
                "gate_id": FAILURE_NEXT_GATE_ID,
                "action": (
                    "classify the frozen clean-location failure before changing any "
                    "package component or execution contract"
                ),
            },
            "pre_execution_invalid_or_incomplete": {
                "gate_id": GATE_ID,
                "action": (
                    "only when model_loads and generate_calls are both zero repair "
                    "the harness or transport and repeat this unchanged protocol"
                ),
            },
            "post_load_invalid_or_incomplete": {
                "gate_id": FAILURE_NEXT_GATE_ID,
                "action": (
                    "after any model load or generate call freeze the incomplete "
                    "attempt and classify it without an automatic rerun"
                ),
            },
        },
        "constraints": {
            "new_data": False,
            "training": False,
            "eval_answer_tuning": False,
            "decision_compiler_change": False,
            "prompt_change": False,
            "generation_change": False,
            "precision_change": False,
            "execution_form_change": False,
            "base_or_tokenizer_substitution": False,
            "adapter_or_weight_mutation": False,
            "merged_weight_creation": False,
            "artifact_promotion": False,
            "serving_integration": False,
            "runtime_integration": False,
            "provider_integration": False,
            "mcp_integration": False,
            "desktop_integration": False,
        },
        "claims": {
            "metadata_validation_is_behavioral_reproducibility": False,
            "materialization_transport_is_remote_origin_attestation": False,
            "clean_location_is_cross_machine_reproducibility": False,
            "transitive_dependency_hashes_pinned": False,
            "full_eval_repeat_variance_established": False,
            "external_execution_count_attested": False,
            "offline_artifact_eligible": False,
            "portable_package_eligible": False,
            "preferred_offline_candidate": False,
            "serving_readiness_established": False,
            "artifact_promotion_allowed": False,
            "merged_artifact_allowed": False,
            "scope": (
                "same_recorded_environment_clean_location_twenty_case_replay_only"
            ),
        },
        "runtime_eligible": False,
    }


def validate_preregistration(
    value: object,
    *,
    require_frozen: bool = True,
) -> dict[str, Any]:
    """Validate a closed preregistration and reject draft execution by default."""

    preregistration = _mapping(value, "$.preregistration")
    _validate_finite_json(preregistration, "$.preregistration")
    status = preregistration.get("freeze_status")
    if status not in {"draft", "frozen"}:
        _fail("INVALID_FREEZE_STATUS", "$.freeze_status", repr(status))
    lineage = _mapping(preregistration.get("source_lineage"), "$.source_lineage")
    sources = _mapping(
        lineage.get("protocol_sources"), "$.source_lineage.protocol_sources"
    )
    if set(sources) != set(PROTOCOL_SOURCE_PATHS):
        _fail(
            "INVALID_PROTOCOL_SOURCE_SET",
            "$.source_lineage.protocol_sources",
            repr(sorted(sources)),
        )
    hashes: dict[str, str] = {}
    for name in sorted(PROTOCOL_SOURCE_PATHS):
        record = _mapping(
            sources.get(name), f"$.source_lineage.protocol_sources.{name}"
        )
        digest = record.get("sha256")
        _validate_sha256(digest, f"$.source_lineage.protocol_sources.{name}.sha256")
        hashes[name] = str(digest)

    expected = expected_preregistration(
        freeze_status=str(status), protocol_source_hashes=hashes
    )
    if preregistration != expected:
        _fail(
            "PREREGISTRATION_RECOMPUTATION_MISMATCH",
            "$.preregistration",
            _first_difference(preregistration, expected),
        )
    if status == "draft":
        if any(value != ZERO_SHA256 for value in hashes.values()):
            _fail(
                "DRAFT_SOURCE_HASH_NOT_PLACEHOLDER",
                "$.source_lineage.protocol_sources",
                repr(hashes),
            )
        if require_frozen:
            _fail(
                "PREREGISTRATION_NOT_FROZEN",
                "$.freeze_status",
                "draft cannot authorize formal execution",
            )
    else:
        if any(value == ZERO_SHA256 for value in hashes.values()):
            _fail(
                "UNBOUND_PROTOCOL_SOURCE",
                "$.source_lineage.protocol_sources",
                repr(hashes),
            )
        if (
            preregistration["materialization_protocol"]["formal_execution_authorized"]
            is not True
        ):
            _fail(
                "FORMAL_EXECUTION_NOT_AUTHORIZED",
                "$.materialization_protocol.formal_execution_authorized",
                "false",
            )
    return copy.deepcopy(dict(preregistration))


def load_and_validate_preregistration(
    path: Path,
    *,
    require_frozen: bool = True,
) -> LoadedPreregistration:
    """Single-read one preregistration and bind its raw bytes."""

    payload = _read_regular_file(path, max_bytes=MAX_JSON_BYTES)
    value = parse_strict_json_bytes(payload, path="$.preregistration_file")
    data = validate_preregistration(value, require_frozen=require_frozen)
    return LoadedPreregistration(
        data=data, payload=payload, sha256=sha256_bytes(payload)
    )


def load_manifest_source_bundle(
    *,
    repository_root: Path,
    adapter_root: Path,
) -> ManifestSourceBundle:
    """Single-read the exact small source closure from clean resolved roots."""

    payloads: dict[str, bytes] = {}
    for name, relative in sorted(REPOSITORY_SOURCE_PATHS.items()):
        payloads[name] = _read_root_relative_file(
            repository_root,
            relative,
            path=f"$.manifest_sources.{name}",
            max_bytes=MAX_SOURCE_BYTES,
        )
    adapter_paths = {
        "adapter_config": "adapter_config.json",
        "adapter_readme": "README.md",
        "adapter_weights": "adapter_model.safetensors",
    }
    for name, relative in adapter_paths.items():
        payloads[name] = _read_root_relative_file(
            adapter_root,
            relative,
            path=f"$.manifest_sources.{name}",
            max_bytes=MAX_SOURCE_BYTES,
        )
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    if hashes != EXPECTED_MANIFEST_SOURCE_HASHES:
        _fail(
            "MANIFEST_SOURCE_HASH_MISMATCH",
            "$.manifest_sources",
            _first_difference(hashes, EXPECTED_MANIFEST_SOURCE_HASHES),
        )
    upstream_review = _json_object_from_payload(
        payloads["upstream_review"], "$.manifest_sources.upstream_review"
    )
    remediation_preregistration = _json_object_from_payload(
        payloads["remediation_preregistration"],
        "$.manifest_sources.remediation_preregistration",
    )
    sft_config = _json_object_from_payload(
        payloads["sft_config"], "$.manifest_sources.sft_config"
    )
    adapter_config = _json_object_from_payload(
        payloads["adapter_config"], "$.manifest_sources.adapter_config"
    )
    return ManifestSourceBundle(
        upstream_review=upstream_review,
        remediation_preregistration=remediation_preregistration,
        sft_config=sft_config,
        adapter_config=adapter_config,
        source_hashes=hashes,
        source_payloads=payloads,
    )


def authenticate_manifest_and_references(
    preregistration: Mapping[str, Any],
    *,
    manifest_payload: bytes,
    reference_predictions_payload: bytes,
    reference_evidence_payload: bytes,
    evaluation_payload: bytes,
    manifest_sources: ManifestSourceBundle,
) -> AuthenticatedInputs:
    """Authenticate external raw roots before parsing or deriving replay inputs."""

    prereg = validate_preregistration(preregistration)
    _require_payload_sha256(manifest_payload, MANIFEST_SHA256, "$.manifest_payload")
    _require_payload_sha256(
        reference_predictions_payload,
        REFERENCE_PREDICTIONS_SHA256,
        "$.reference_predictions_payload",
    )
    _require_payload_sha256(
        reference_evidence_payload,
        REFERENCE_EVIDENCE_SHA256,
        "$.reference_evidence_payload",
    )
    _require_payload_sha256(
        evaluation_payload, EVALUATION_FILE_SHA256, "$.evaluation_payload"
    )
    manifest = _json_object_from_payload(manifest_payload, "$.manifest")
    reference_predictions = _json_object_from_payload(
        reference_predictions_payload, "$.reference_predictions"
    )
    reference_evidence = _json_object_from_payload(
        reference_evidence_payload, "$.reference_evidence"
    )
    evaluation_raw = parse_strict_json_bytes(
        evaluation_payload, path="$.evaluation", max_bytes=MAX_JSON_BYTES
    )
    evaluation = _list_of_objects(evaluation_raw, "$.evaluation")

    expected_source_hashes = prereg["source_lineage"]["manifest"]["source_hashes"]
    if dict(manifest_sources.source_hashes) != expected_source_hashes:
        _fail(
            "MANIFEST_SOURCE_HASH_ROOT_MISMATCH",
            "$.manifest_sources.source_hashes",
            _first_difference(
                dict(manifest_sources.source_hashes), expected_source_hashes
            ),
        )
    manifest_validation = validate_fp32_attached_offline_package_manifest(
        manifest_payload,
        MANIFEST_SHA256,
        manifest_sources.upstream_review,
        manifest_sources.remediation_preregistration,
        manifest_sources.sft_config,
        manifest_sources.adapter_config,
        source_hashes=manifest_sources.source_hashes,
        source_payloads=manifest_sources.source_payloads,
        expected_source_hashes=expected_source_hashes,
    )
    _validate_authenticated_manifest(manifest, manifest_validation, prereg)
    _validate_evaluation(evaluation)
    _validate_reference_predictions(reference_predictions, manifest, prereg)
    _validate_reference_evidence(reference_evidence)
    reference_artifact = _mapping(
        reference_evidence.get("prediction_artifact"),
        "$.reference_evidence.prediction_artifact",
    )
    if reference_artifact.get("sha256") != REFERENCE_PREDICTIONS_SHA256:
        _fail(
            "REFERENCE_EVIDENCE_PREDICTION_MISMATCH",
            "$.reference_evidence.prediction_artifact.sha256",
            repr(reference_artifact.get("sha256")),
        )
    outputs = _validate_output_records(
        reference_predictions.get("outputs"),
        path="$.reference_predictions.outputs",
        require_exact_order=True,
        record_schema="reference",
    )
    return AuthenticatedInputs(
        preregistration=prereg,
        manifest_payload=manifest_payload,
        manifest=manifest,
        manifest_validation=manifest_validation,
        reference_predictions=reference_predictions,
        reference_evidence=reference_evidence,
        evaluation=evaluation,
        reference_outputs=tuple(outputs),
    )


def resolve_clean_roots(
    preregistration: Mapping[str, Any],
    authenticated: AuthenticatedInputs,
    manifest_sources: ManifestSourceBundle,
    *,
    base_model_root: Path,
    adapter_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Re-authenticate the manifest snapshot and resolve caller-supplied clean roots."""

    prereg = validate_preregistration(preregistration)
    if prereg != authenticated.preregistration:
        _fail("PREREGISTRATION_SNAPSHOT_MISMATCH", "$.preregistration", "changed")
    manifest_payload = authenticated.manifest_payload
    if sha256_bytes(manifest_payload) != MANIFEST_SHA256:
        _fail(
            "MANIFEST_SNAPSHOT_ENCODING_MISMATCH",
            "$.manifest",
            sha256_bytes(manifest_payload),
        )
    expected_source_hashes = prereg["source_lineage"]["manifest"]["source_hashes"]
    result = validate_and_resolve_fp32_attached_offline_package(
        manifest_payload,
        MANIFEST_SHA256,
        manifest_sources.upstream_review,
        manifest_sources.remediation_preregistration,
        manifest_sources.sft_config,
        manifest_sources.adapter_config,
        source_hashes=manifest_sources.source_hashes,
        source_payloads=manifest_sources.source_payloads,
        expected_source_hashes=expected_source_hashes,
        base_model_root=base_model_root,
        adapter_root=adapter_root,
        repository_root=repository_root,
    )
    if result.get("validation") != authenticated.manifest_validation:
        _fail(
            "MANIFEST_VALIDATION_SNAPSHOT_MISMATCH",
            "$.resolution.validation",
            "changed",
        )
    resolution = _mapping(result.get("resolution"), "$.resolution")
    _validate_resolution(resolution)
    return copy.deepcopy(dict(resolution))


def validate_materialization_receipt(
    preregistration: Mapping[str, Any],
    receipt: object,
    *,
    preregistration_sha256: str,
    expected_freeze_commit: str,
    clean_resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate path-redacted remote-materialization facts against clean resolution."""

    prereg = validate_preregistration(preregistration)
    _validate_sha256(preregistration_sha256, "$.preregistration_sha256")
    _validate_git_commit(expected_freeze_commit, "$.expected_freeze_commit")
    resolution = _validate_resolution(clean_resolution)
    value = _mapping(receipt, "$.materialization_receipt")
    supplied_destination = _mapping(
        value.get("destination"), "$.materialization_receipt.destination"
    )
    destination_id = supplied_destination.get("destination_id")
    if (
        not isinstance(destination_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", destination_id) is None
    ):
        _fail(
            "INVALID_DESTINATION_ID",
            "$.materialization_receipt.destination.destination_id",
            repr(destination_id),
        )
    protocol_sources = _expected_protocol_source_receipts(prereg, value)
    groups = _normalized_resolution_groups(resolution)
    expected = {
        "receipt_version": RECEIPT_VERSION,
        "gate_id": GATE_ID,
        "experiment_id": EXPERIMENT_ID,
        "package_id": PACKAGE_ID,
        "preregistration_sha256": preregistration_sha256,
        "protocol_freeze_commit": expected_freeze_commit,
        "manifest_file_sha256": MANIFEST_SHA256,
        "phase_order": prereg["materialization_protocol"]["phase_order"],
        "destination": {
            "destination_id": destination_id,
            "caller_supplied_root": True,
            "root_was_absent": True,
            "root_created_exclusive": True,
            "children": prereg["materialization_protocol"]["destination_policy"][
                "children"
            ],
            "adapter_root_relative_to_repository": (
                ADAPTER_ROOT_RELATIVE_TO_REPOSITORY
            ),
            "absolute_paths_recorded": False,
            "symlinks_used": False,
            "reparse_points_used": False,
            "hardlinks_used": False,
            "overwrite_used": False,
        },
        "transport": {
            "repository_remote_url": REPOSITORY_REMOTE_URL,
            "fresh_git_checkout": True,
            "git_fetch_used": True,
            "git_lfs_checkout_used": True,
            "model_downloader_path": MODEL_DOWNLOADER_PATH,
            "model_downloader_sha256": MODEL_DOWNLOADER_SHA256,
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
        "protocol_sources": protocol_sources,
        "materialization_passed": bool(resolution["resolved"]),
        "issues": [] if resolution["resolved"] else _resolution_issues(resolution),
    }
    if value != expected:
        _fail(
            "MATERIALIZATION_RECEIPT_MISMATCH",
            "$.materialization_receipt",
            _first_difference(value, expected),
        )
    return copy.deepcopy(dict(value))


def compare_behavioral_replay(
    authenticated: AuthenticatedInputs,
    replay_outputs: object,
) -> dict[str, Any]:
    """Compare exact UTF-8 raw outputs and independently recompiled decisions."""

    observed = _validate_output_records(
        replay_outputs,
        path="$.replay.outputs",
        require_exact_order=True,
        record_schema="replay",
    )
    reference = list(authenticated.reference_outputs)
    observed_raw = [
        {"example_id": item["example_id"], "raw_output": item["raw_output"]}
        for item in observed
    ]
    raw_mismatches: list[str] = []
    compiled_mismatches: list[str] = []
    compilation_failures: list[dict[str, str]] = []
    reference_compiled: list[dict[str, Any]] = []
    observed_compiled: list[dict[str, Any]] = []
    for index, (expected, actual) in enumerate(zip(reference, observed)):
        example_id = EXPECTED_EXAMPLE_ORDER[index]
        if actual["raw_output"].encode("utf-8") != expected["raw_output"].encode(
            "utf-8"
        ):
            raw_mismatches.append(example_id)
        expected_compilation = _compile_raw_output(
            expected["raw_output"], f"$.reference.outputs[{index}].raw_output"
        )
        actual_compilation = _compile_raw_output(
            actual["raw_output"], f"$.replay.outputs[{index}].raw_output"
        )
        reference_compiled.append({"example_id": example_id, **expected_compilation})
        observed_compiled.append({"example_id": example_id, **actual_compilation})
        if actual_compilation.get("valid") is not True:
            compilation_failures.append(
                {
                    "example_id": example_id,
                    "code": str(actual_compilation.get("error")),
                }
            )
        if canonical_json_bytes(actual_compilation) != canonical_json_bytes(
            expected_compilation
        ):
            compiled_mismatches.append(example_id)
    raw_exact = EXPECTED_RECORDS - len(raw_mismatches)
    compiled_exact = EXPECTED_RECORDS - len(compiled_mismatches)
    return {
        "comparison_version": 1,
        "scope": authenticated.preregistration["behavior_reference"]["scope"],
        "records_expected": EXPECTED_RECORDS,
        "records_observed": len(observed),
        "example_order_exact": True,
        "raw_output_comparison": "exact_utf8_bytes",
        "raw_outputs_exact": raw_exact,
        "raw_outputs_digest_reference": _output_digest(reference),
        "raw_outputs_digest_observed": _output_digest(observed_raw),
        "raw_mismatch_example_ids": raw_mismatches,
        "compiler_symbol": "compile_decision",
        "compiler_version": DECISION_COMPILER_VERSION,
        "compiled_output_comparison": "exact_canonical_json_bytes",
        "compiled_outputs_exact": compiled_exact,
        "compiled_outputs_digest_reference": _output_digest(reference_compiled),
        "compiled_outputs_digest_observed": _output_digest(observed_compiled),
        "compiled_mismatch_example_ids": compiled_mismatches,
        "compilation_failures": compilation_failures,
        "behavioral_reproducibility_established": (
            raw_exact == EXPECTED_RECORDS
            and compiled_exact == EXPECTED_RECORDS
            and not compilation_failures
        ),
    }


def build_replay_artifact(
    preregistration: Mapping[str, Any],
    authenticated: AuthenticatedInputs,
    *,
    preregistration_sha256: str,
    protocol_freeze_commit: str,
    materialization_receipt: Mapping[str, Any],
    clean_resolution: Mapping[str, Any],
    observed_environment: Mapping[str, Any],
    precision_audit: Mapping[str, Any],
    performance: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a path-redacted replay artifact after one completed formal run."""

    prereg = validate_preregistration(preregistration)
    _validate_sha256(preregistration_sha256, "$.preregistration_sha256")
    _validate_git_commit(protocol_freeze_commit, "$.protocol_freeze_commit")
    resolution = _validate_resolution(clean_resolution)
    receipt = validate_materialization_receipt(
        prereg,
        materialization_receipt,
        preregistration_sha256=preregistration_sha256,
        expected_freeze_commit=protocol_freeze_commit,
        clean_resolution=resolution,
    )
    output_records = _validate_output_records(
        list(outputs),
        path="$.outputs",
        require_exact_order=True,
        record_schema="replay",
    )
    environment = _mapping(observed_environment, "$.environment")
    precision = _mapping(precision_audit, "$.precision_audit")
    _validate_precision_audit(precision)
    measured = _validate_performance(performance)
    return {
        "artifact_version": REPLAY_ARTIFACT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "gate_id": GATE_ID,
        "package_id": PACKAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "preregistration_sha256": preregistration_sha256,
        "protocol_freeze_commit": protocol_freeze_commit,
        "manifest_file_sha256": MANIFEST_SHA256,
        "reference_predictions_sha256": REFERENCE_PREDICTIONS_SHA256,
        "reference_evidence_sha256": REFERENCE_EVIDENCE_SHA256,
        "evaluation_file_sha256": EVALUATION_FILE_SHA256,
        "evaluation_digest": EVALUATION_DIGEST,
        "example_order": list(EXPECTED_EXAMPLE_ORDER),
        "materialization_receipt_digest": _object_digest(receipt),
        "clean_resolution_digest": resolution["resolution_digest"],
        "environment": copy.deepcopy(dict(environment)),
        "generation": copy.deepcopy(prereg["execution_protocol"]["generation"]),
        "run": {
            "run_id": RUN_ID,
            "candidate_id": CANDIDATE_ID,
            "order_index": 0,
            "fresh_model_loads": 1,
            "full_eval_runs": 1,
            "generate_calls": EXPECTED_RECORDS,
            "retries": 0,
            "completed": True,
        },
        "precision_audit": copy.deepcopy(dict(precision)),
        "performance": measured,
        "execution_network_used": False,
        "historical_adapter_base_path_used": False,
        "model_artifact_saved": False,
        "tensor_payload_saved": False,
        "outputs": output_records,
    }


def build_reproducibility_evidence(
    preregistration: Mapping[str, Any],
    authenticated: AuthenticatedInputs,
    *,
    preregistration_sha256: str,
    protocol_freeze_commit: str,
    materialization_receipt: Mapping[str, Any],
    clean_resolution: Mapping[str, Any],
    replay_artifact: Mapping[str, Any],
    replay_artifact_path: str,
) -> dict[str, Any]:
    """Strictly derive the gate result; no supplied field may authorize itself."""

    prereg = validate_preregistration(preregistration)
    _validate_sha256(preregistration_sha256, "$.preregistration_sha256")
    _validate_git_commit(protocol_freeze_commit, "$.protocol_freeze_commit")
    resolution = _validate_resolution(clean_resolution)
    receipt = validate_materialization_receipt(
        prereg,
        materialization_receipt,
        preregistration_sha256=preregistration_sha256,
        expected_freeze_commit=protocol_freeze_commit,
        clean_resolution=resolution,
    )
    replay = _validate_replay_artifact(
        prereg,
        authenticated,
        replay_artifact,
        preregistration_sha256=preregistration_sha256,
        protocol_freeze_commit=protocol_freeze_commit,
        materialization_receipt=receipt,
        clean_resolution=resolution,
    )
    _validate_artifact_relative_path(replay_artifact_path, "$.replay_artifact_path")
    replay_payload = artifact_json_bytes(replay)
    comparison = compare_behavioral_replay(authenticated, replay["outputs"])
    expected_environment = authenticated.manifest["components"]["environment"][
        "recorded_environment"
    ]
    environment_passed = replay["environment"] == expected_environment
    execution_passed = _execution_protocol_passed(replay)
    resources = _resource_assessment(replay["performance"], prereg["resource_caps"])
    gates = {
        "metadata_validation": authenticated.manifest_validation.get(
            "frozen_manifest_valid"
        )
        is True,
        "materialization": receipt["materialization_passed"] is True,
        "clean_location_resolution": resolution["resolved"] is True,
        "environment": environment_passed,
        "execution_contract": execution_passed,
        "behavioral_replay": comparison["behavioral_reproducibility_established"]
        is True,
        "resources": resources["passed"] is True,
    }
    classification = classify_reproducibility_gates(gates)
    passed = classification == PASS_CLASSIFICATION
    behavioral_established = (
        gates["metadata_validation"]
        and gates["materialization"]
        and gates["clean_location_resolution"]
        and gates["environment"]
        and gates["execution_contract"]
        and gates["behavioral_replay"]
    )
    remaining = _remaining_blockers(gates, behavioral_established)
    next_action = prereg["outcome_next_actions"]["passed" if passed else "adverse"]
    locked_next_action = {
        **next_action,
        "classification": classification,
        "formal_gate_passed": passed,
        "eligible_to_start": passed,
        "remaining_blocking_findings": remaining,
        "artifact_promotion_allowed": False,
        "runtime_integration_allowed": False,
    }
    derived_claims = {
        "metadata_complete": gates["metadata_validation"],
        "offline_package_identity_complete": gates["metadata_validation"],
        "clean_location_resolution_established": gates["clean_location_resolution"]
        and gates["materialization"],
        "behavioral_reproducibility_established": behavioral_established,
        "behavioral_reproducibility_scope": prereg["behavior_reference"]["scope"],
        "remote_revision_origin_attested": False,
        "offline_artifact_eligible": False,
        "portable_package_eligible": False,
        "preferred_offline_candidate": False,
        "serving_readiness_established": False,
        "artifact_promotion_allowed": False,
        "merged_artifact_allowed": False,
        "runtime_eligible": False,
    }
    return {
        "evidence_version": EVIDENCE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "gate_id": GATE_ID,
        "package_id": PACKAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "preregistration_sha256": preregistration_sha256,
        "protocol_freeze_commit": protocol_freeze_commit,
        "source_lineage": {
            "manifest_file_sha256": MANIFEST_SHA256,
            "reference_predictions_sha256": REFERENCE_PREDICTIONS_SHA256,
            "reference_evidence_sha256": REFERENCE_EVIDENCE_SHA256,
            "evaluation_file_sha256": EVALUATION_FILE_SHA256,
            "evaluation_digest": EVALUATION_DIGEST,
            "protocol_source_hashes": {
                name: value["sha256"]
                for name, value in prereg["source_lineage"]["protocol_sources"].items()
            },
        },
        "replay_artifact": {
            "path": replay_artifact_path,
            "bytes": len(replay_payload),
            "sha256": sha256_bytes(replay_payload),
        },
        "materialization_receipt": receipt,
        "clean_resolution": resolution,
        "comparison": comparison,
        "resources": resources,
        "gates": gates,
        "classification": classification,
        "formal_gate_passed": passed,
        "derived_claims": derived_claims,
        "remaining_blocking_findings": remaining,
        "remaining_blocking_finding_count": len(remaining),
        "locked_next_action": locked_next_action,
        "constraints": prereg["constraints"],
        "claims": prereg["claims"],
        "model_artifact_saved": False,
        "tensor_payload_saved": False,
        "offline_execution": True,
        "runtime_eligible": False,
    }


def classify_reproducibility_gates(gates: Mapping[str, object]) -> str:
    """Derive exactly one outcome classification from non-authoritative gates."""

    expected_keys = {
        "metadata_validation",
        "materialization",
        "clean_location_resolution",
        "environment",
        "execution_contract",
        "behavioral_replay",
        "resources",
    }
    if set(gates) != expected_keys or any(
        not isinstance(gates[key], bool) for key in expected_keys
    ):
        _fail("INVALID_GATE_SET", "$.gates", repr(gates))
    if not gates["metadata_validation"]:
        return TRUST_ROOT_INVALID_CLASSIFICATION
    if not gates["materialization"]:
        return MATERIALIZATION_FAILED_CLASSIFICATION
    if not gates["clean_location_resolution"]:
        return RESOLUTION_FAILED_CLASSIFICATION
    if not gates["environment"]:
        return ENVIRONMENT_MISMATCH_CLASSIFICATION
    if not gates["execution_contract"]:
        return EXECUTION_FAILED_CLASSIFICATION
    behavior = bool(gates["behavioral_replay"])
    resources = bool(gates["resources"])
    if not behavior and not resources:
        return BEHAVIORAL_AND_RESOURCE_FAILED_CLASSIFICATION
    if not behavior:
        return BEHAVIORAL_DRIFT_CLASSIFICATION
    if not resources:
        return RESOURCE_EXCEEDED_CLASSIFICATION
    return PASS_CLASSIFICATION


def validate_reproducibility_evidence(
    preregistration_payload: bytes,
    replay_artifact_payload: bytes,
    evidence_payload: bytes,
    *,
    expected_preregistration_sha256: str,
    expected_replay_artifact_sha256: str,
    expected_evidence_sha256: str,
    expected_protocol_freeze_commit: str,
    replay_artifact_path: str,
    manifest_payload: bytes,
    reference_predictions_payload: bytes,
    reference_evidence_payload: bytes,
    evaluation_payload: bytes,
    manifest_sources: ManifestSourceBundle,
) -> dict[str, Any]:
    """Authenticate frozen artifacts and independently rebuild all gate decisions."""

    _require_payload_sha256(
        preregistration_payload,
        expected_preregistration_sha256,
        "$.preregistration_payload",
    )
    _require_payload_sha256(
        replay_artifact_payload,
        expected_replay_artifact_sha256,
        "$.replay_artifact_payload",
    )
    _require_payload_sha256(
        evidence_payload, expected_evidence_sha256, "$.evidence_payload"
    )
    _validate_git_commit(
        expected_protocol_freeze_commit, "$.expected_protocol_freeze_commit"
    )
    preregistration_raw = parse_strict_json_bytes(
        preregistration_payload, path="$.preregistration"
    )
    preregistration = validate_preregistration(preregistration_raw)
    replay_raw = parse_strict_json_bytes(
        replay_artifact_payload, path="$.replay_artifact"
    )
    replay = _mapping(replay_raw, "$.replay_artifact")
    evidence_raw = parse_strict_json_bytes(evidence_payload, path="$.evidence")
    evidence = _mapping(evidence_raw, "$.evidence")
    authenticated = authenticate_manifest_and_references(
        preregistration,
        manifest_payload=manifest_payload,
        reference_predictions_payload=reference_predictions_payload,
        reference_evidence_payload=reference_evidence_payload,
        evaluation_payload=evaluation_payload,
        manifest_sources=manifest_sources,
    )
    receipt = _mapping(
        evidence.get("materialization_receipt"),
        "$.evidence.materialization_receipt",
    )
    resolution = _mapping(
        evidence.get("clean_resolution"), "$.evidence.clean_resolution"
    )
    expected = build_reproducibility_evidence(
        preregistration,
        authenticated,
        preregistration_sha256=expected_preregistration_sha256,
        protocol_freeze_commit=expected_protocol_freeze_commit,
        materialization_receipt=receipt,
        clean_resolution=resolution,
        replay_artifact=replay,
        replay_artifact_path=replay_artifact_path,
    )
    if evidence != expected:
        _fail(
            "EVIDENCE_RECOMPUTATION_MISMATCH",
            "$.evidence",
            _first_difference(evidence, expected),
        )
    if artifact_json_bytes(replay) != replay_artifact_payload:
        _fail(
            "REPLAY_ARTIFACT_ENCODING_MISMATCH",
            "$.replay_artifact",
            "not canonical tracked encoding",
        )
    if artifact_json_bytes(evidence) != evidence_payload:
        _fail(
            "EVIDENCE_ENCODING_MISMATCH",
            "$.evidence",
            "not canonical tracked encoding",
        )
    return {
        "frozen_gate_valid": True,
        "classification": expected["classification"],
        "formal_gate_passed": expected["formal_gate_passed"],
        "clean_location_resolution_established": expected["derived_claims"][
            "clean_location_resolution_established"
        ],
        "behavioral_reproducibility_established": expected["derived_claims"][
            "behavioral_reproducibility_established"
        ],
        "remaining_blocking_findings": expected["remaining_blocking_findings"],
        "next_gate": expected["locked_next_action"]["gate_id"],
        "runtime_eligible": False,
    }


def _validate_authenticated_manifest(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    if (
        manifest.get("manifest_version") != 1
        or manifest.get("gate_id")
        != "FC-MVP-001-fp32-attached-offline-package-manifest-v1"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("artifact_kind") != "external_metadata_only_composite_manifest"
        or manifest.get("source_artifacts")
        != preregistration["source_lineage"]["manifest"]["source_hashes"]
    ):
        _fail("AUTHENTICATED_MANIFEST_IDENTITY_MISMATCH", "$.manifest", "identity")
    if (
        validation.get("frozen_manifest_valid") is not True
        or validation.get("manifest_file_sha256") != MANIFEST_SHA256
        or validation.get("metadata_complete") is not True
        or validation.get("offline_package_identity_complete") is not True
        or validation.get("eligible_for_clean_location_reproducibility_test")
        is not True
        or validation.get("behavioral_reproducibility_established") is not False
        or validation.get("runtime_eligible") is not False
    ):
        _fail("MANIFEST_VALIDATION_DECISION_MISMATCH", "$.manifest_validation", "drift")
    execution = _mapping(manifest.get("execution_contract"), "$.manifest.execution")
    generation = _mapping(
        execution.get("generation"), "$.manifest.execution.generation"
    )
    if (
        generation.get("effective_contract") != GENERATION_CONTRACT
        or execution.get("execution_form") != "attached_factorized_lora"
        or execution.get("merge") is not False
        or execution.get("save_model") is not False
        or execution.get("save_tensors") is not False
        or execution.get("local_files_only") is not True
    ):
        _fail("MANIFEST_EXECUTION_CONTRACT_MISMATCH", "$.manifest.execution", "drift")


def _validate_evaluation(evaluation: Sequence[Mapping[str, Any]]) -> None:
    if len(evaluation) != EXPECTED_RECORDS:
        _fail("EVALUATION_COUNT_MISMATCH", "$.evaluation", str(len(evaluation)))
    order: list[str] = []
    for index, record in enumerate(evaluation):
        example_id = record.get("example_id")
        if not isinstance(example_id, str):
            _fail(
                "INVALID_EVALUATION_ID",
                f"$.evaluation[{index}].example_id",
                repr(example_id),
            )
        order.append(example_id)
    if order != EXPECTED_EXAMPLE_ORDER:
        _fail("EVALUATION_ORDER_MISMATCH", "$.evaluation", repr(order))
    if _object_digest(list(evaluation)) != EVALUATION_DIGEST:
        _fail(
            "EVALUATION_CANONICAL_DIGEST_MISMATCH",
            "$.evaluation",
            _object_digest(list(evaluation)),
        )


def _validate_reference_predictions(
    predictions: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    expected_keys = {
        "artifact_version",
        "experiment_id",
        "gate_id",
        "preregistration_sha256",
        "source_lineage",
        "model",
        "tokenizer",
        "environment",
        "generation",
        "prompt_sha256",
        "eval_digest",
        "example_order",
        "adapter_files",
        "storage_audit",
        "run",
        "precision_audit",
        "performance",
        "outputs",
    }
    if set(predictions) != expected_keys:
        _fail(
            "REFERENCE_PREDICTION_SCHEMA_MISMATCH",
            "$.reference_predictions",
            repr(sorted(predictions)),
        )
    components = _mapping(manifest.get("components"), "$.manifest.components")
    environment = _mapping(components.get("environment"), "$.manifest.environment")
    if (
        predictions.get("artifact_version") != 1
        or predictions.get("experiment_id")
        != "fc-mvp-001-fp32-attached-remediation-eval-v1"
        or predictions.get("gate_id") != "FC-MVP-001-fp32-attached-remediation-eval-v1"
        or predictions.get("preregistration_sha256") != REFERENCE_PREREGISTRATION_SHA256
        or predictions.get("environment") != environment.get("recorded_environment")
        or predictions.get("generation")
        != preregistration["execution_protocol"]["generation"]
        or predictions.get("eval_digest") != EVALUATION_DIGEST
        or predictions.get("example_order") != EXPECTED_EXAMPLE_ORDER
    ):
        _fail(
            "REFERENCE_PREDICTION_IDENTITY_MISMATCH", "$.reference_predictions", "drift"
        )
    run = _mapping(predictions.get("run"), "$.reference_predictions.run")
    if (
        run.get("fresh_model_loads") != 1
        or run.get("full_eval_runs") != 1
        or run.get("generate_calls") != EXPECTED_RECORDS
        or run.get("retries") != 0
        or run.get("completed") is not True
    ):
        _fail("REFERENCE_RUN_MISMATCH", "$.reference_predictions.run", repr(run))
    outputs = _validate_output_records(
        predictions.get("outputs"),
        path="$.reference_predictions.outputs",
        require_exact_order=True,
        record_schema="reference",
    )
    for index, output in enumerate(outputs):
        compiled = _compile_raw_output(
            output["raw_output"],
            f"$.reference_predictions.outputs[{index}].raw_output",
        )
        if compiled["valid"] is not True:
            _fail(
                "REFERENCE_COMPILATION_FAILED",
                f"$.reference_predictions.outputs[{index}]",
                str(compiled["error"]),
            )


def _validate_reference_evidence(evidence: Mapping[str, Any]) -> None:
    expected_keys = {
        "gate_version",
        "experiment_id",
        "gate_id",
        "preregistration_sha256",
        "source_lineage",
        "prediction_artifact",
        "raw_metrics",
        "raw_parsed_outputs",
        "compilation",
        "compiled_metrics",
        "compiled_parsed_outputs",
        "comparison",
        "assessment",
        "gates",
        "resources",
        "constraints",
        "claims",
        "locked_next_action",
        "compiled_model_saved",
        "tensor_payload_saved",
        "runtime_eligible",
        "runtime_eligibility_reason",
        "offline",
    }
    if set(evidence) != expected_keys:
        _fail(
            "REFERENCE_EVIDENCE_SCHEMA_MISMATCH",
            "$.reference_evidence",
            repr(sorted(evidence)),
        )
    assessment = _mapping(evidence.get("assessment"), "$.reference_evidence.assessment")
    if (
        evidence.get("gate_version") != 1
        or evidence.get("experiment_id")
        != "fc-mvp-001-fp32-attached-remediation-eval-v1"
        or evidence.get("gate_id") != "FC-MVP-001-fp32-attached-remediation-eval-v1"
        or evidence.get("preregistration_sha256") != REFERENCE_PREREGISTRATION_SHA256
        or assessment.get("classification")
        != (
            "fp32_attached_full_eval_improves_quality_without_safety_or_"
            "resource_regression"
        )
        or assessment.get("evaluation_gate_passed") is not True
        or evidence.get("compiled_model_saved") is not False
        or evidence.get("tensor_payload_saved") is not False
        or evidence.get("runtime_eligible") is not False
        or evidence.get("offline") is not True
    ):
        _fail("REFERENCE_EVIDENCE_IDENTITY_MISMATCH", "$.reference_evidence", "drift")


def _validate_output_records(
    value: object,
    *,
    path: str,
    require_exact_order: bool,
    record_schema: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail("INVALID_OUTPUT_COLLECTION", path, type(value).__name__)
    if record_schema not in {"reference", "replay"}:
        _fail("INVALID_OUTPUT_RECORD_SCHEMA", path, record_schema)
    records: list[dict[str, Any]] = []
    reference_keys = {"example_id", "raw_output"}
    replay_keys = {
        "example_id",
        "rendered_prompt_sha256",
        "input_token_ids_sha256",
        "input_token_count",
        "output_token_ids_sha256",
        "output_token_count",
        "raw_output",
        "raw_output_utf8_sha256",
        "compiler_valid",
        "compiler_input_canonical_sha256",
        "compiled_output",
        "compiled_output_canonical_sha256",
        "compiler_changed_fields",
        "compilation_error",
    }
    for index, raw in enumerate(value):
        item = _mapping(raw, f"{path}[{index}]")
        expected_keys = reference_keys if record_schema == "reference" else replay_keys
        if set(item) != expected_keys:
            _fail("INVALID_OUTPUT_SCHEMA", f"{path}[{index}]", repr(sorted(item)))
        example_id = item.get("example_id")
        raw_output = item.get("raw_output")
        if not isinstance(example_id, str) or not isinstance(raw_output, str):
            _fail(
                "INVALID_OUTPUT_VALUE",
                f"{path}[{index}]",
                f"{type(example_id).__name__}/{type(raw_output).__name__}",
            )
        try:
            encoded = raw_output.encode("utf-8")
        except UnicodeEncodeError as exc:
            _fail("INVALID_OUTPUT_UTF8", f"{path}[{index}].raw_output", str(exc))
        if len(encoded) > 64 * 1024:
            _fail(
                "RAW_OUTPUT_TOO_LARGE", f"{path}[{index}].raw_output", str(len(encoded))
            )
        if record_schema == "reference":
            records.append({"example_id": example_id, "raw_output": raw_output})
            continue
        for key in (
            "rendered_prompt_sha256",
            "input_token_ids_sha256",
            "output_token_ids_sha256",
            "raw_output_utf8_sha256",
        ):
            _validate_sha256(item.get(key), f"{path}[{index}].{key}")
            if item.get(key) == ZERO_SHA256:
                _fail("ZERO_OUTPUT_DIGEST", f"{path}[{index}].{key}", ZERO_SHA256)
        input_count = _nonnegative_int(
            item.get("input_token_count"), f"{path}[{index}].input_token_count"
        )
        _nonnegative_int(
            item.get("output_token_count"), f"{path}[{index}].output_token_count"
        )
        if input_count == 0:
            _fail("EMPTY_INPUT_TOKENS", f"{path}[{index}].input_token_count", "0")
        if item.get("raw_output_utf8_sha256") != sha256_bytes(encoded):
            _fail(
                "RAW_OUTPUT_DIGEST_MISMATCH",
                f"{path}[{index}].raw_output_utf8_sha256",
                repr(item.get("raw_output_utf8_sha256")),
            )
        compilation = _compile_raw_output(raw_output, f"{path}[{index}].raw_output")
        expected_compiler_fields = {
            "compiler_valid": compilation["valid"],
            "compiler_input_canonical_sha256": compilation["input_digest"],
            "compiled_output": compilation["compiled"],
            "compiled_output_canonical_sha256": compilation["compiled_digest"],
            "compiler_changed_fields": compilation["changed_fields"],
            "compilation_error": compilation["error"],
        }
        observed_compiler_fields = {
            key: item.get(key) for key in expected_compiler_fields
        }
        if observed_compiler_fields != expected_compiler_fields:
            _fail(
                "REPLAY_COMPILER_RECEIPT_MISMATCH",
                f"{path}[{index}]",
                _first_difference(observed_compiler_fields, expected_compiler_fields),
            )
        records.append(copy.deepcopy(dict(item)))
    if require_exact_order:
        order = [item["example_id"] for item in records]
        if order != EXPECTED_EXAMPLE_ORDER:
            _fail("OUTPUT_ORDER_MISMATCH", path, repr(order))
    return records


def _compile_raw_output(raw_output: str, path: str) -> dict[str, Any]:
    try:
        decoded = parse_strict_json_bytes(raw_output.encode("utf-8"), path=path)
        source = _mapping(decoded, path)
        compiled = compile_decision(source)
    except (ReproducibilityContractError, ToolRouterValidationError) as exc:
        return {
            "valid": False,
            "input_digest": None,
            "compiled": None,
            "compiled_digest": None,
            "changed_fields": [],
            "error": getattr(exc, "code", type(exc).__name__),
        }
    changed_fields = [
        f"$.{key}" for key in sorted(source) if source.get(key) != compiled.get(key)
    ]
    return {
        "valid": True,
        "input_digest": _object_digest(dict(source)),
        "compiled": compiled,
        "compiled_digest": _object_digest(compiled),
        "changed_fields": changed_fields,
        "error": None,
    }


def _output_digest(value: object) -> str:
    return _object_digest(value)


def _object_digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _validate_resolution(value: Mapping[str, Any]) -> dict[str, Any]:
    resolution = _mapping(value, "$.clean_resolution")
    expected_keys = {
        "resolution_version",
        "package_id",
        "manifest_file_sha256",
        "caller_supplied_roots",
        "manifest_machine_paths_used",
        "adapter_local_base_path_used",
        "resolved",
        "eligible_for_clean_location_reproducibility_test",
        "offline_artifact_eligible",
        "runtime_eligible",
        "groups",
        "failure_mode",
        "resolution_digest",
    }
    if set(resolution) != expected_keys:
        _fail(
            "INVALID_RESOLUTION_SCHEMA", "$.clean_resolution", repr(sorted(resolution))
        )
    _validate_sha256(
        resolution.get("resolution_digest"), "$.clean_resolution.resolution_digest"
    )
    resolved = resolution.get("resolved")
    if not isinstance(resolved, bool):
        _fail("INVALID_RESOLUTION_FLAG", "$.clean_resolution.resolved", repr(resolved))
    if (
        resolution.get("resolution_version") != 1
        or resolution.get("package_id") != PACKAGE_ID
        or resolution.get("manifest_file_sha256") != MANIFEST_SHA256
        or resolution.get("caller_supplied_roots") is not True
        or resolution.get("manifest_machine_paths_used") is not False
        or resolution.get("adapter_local_base_path_used") is not False
        or resolution.get("eligible_for_clean_location_reproducibility_test")
        is not resolved
        or resolution.get("offline_artifact_eligible") is not False
        or resolution.get("runtime_eligible") is not False
        or resolution.get("failure_mode")
        != (None if resolved else "component_resolution_failed_closed")
    ):
        _fail("RESOLUTION_CONTRACT_MISMATCH", "$.clean_resolution", "drift")
    groups = resolution.get("groups")
    if not isinstance(groups, list) or len(groups) != 3:
        _fail("INVALID_RESOLUTION_GROUPS", "$.clean_resolution.groups", repr(groups))
    expected_roles = ["base_model_and_tokenizer", "adapter", "repository"]
    for index, (group_raw, role) in enumerate(zip(groups, expected_roles)):
        group = _mapping(group_raw, f"$.clean_resolution.groups[{index}]")
        if set(group) != {
            "root_role",
            "resolved",
            "expected_files",
            "matched_files",
            "matched_bytes",
            "issues",
        }:
            _fail(
                "INVALID_RESOLUTION_GROUP_SCHEMA",
                f"$.clean_resolution.groups[{index}]",
                repr(sorted(group)),
            )
        if group.get("root_role") != role or not isinstance(
            group.get("resolved"), bool
        ):
            _fail(
                "RESOLUTION_GROUP_ROLE_MISMATCH",
                f"$.clean_resolution.groups[{index}]",
                role,
            )
        for key in ("expected_files", "matched_files", "matched_bytes"):
            _nonnegative_int(
                group.get(key), f"$.clean_resolution.groups[{index}].{key}"
            )
        issues = group.get("issues")
        if not isinstance(issues, list):
            _fail(
                "INVALID_RESOLUTION_ISSUES",
                f"$.clean_resolution.groups[{index}].issues",
                repr(issues),
            )
        for issue_index, issue_raw in enumerate(issues):
            issue = _mapping(
                issue_raw, f"$.clean_resolution.groups[{index}].issues[{issue_index}]"
            )
            if set(issue) != {"code", "path"} or not all(
                isinstance(issue.get(k), str) and issue.get(k) for k in ("code", "path")
            ):
                _fail(
                    "INVALID_RESOLUTION_ISSUE",
                    f"$.clean_resolution.groups[{index}].issues[{issue_index}]",
                    repr(issue),
                )
        if group["resolved"] is not (
            not issues and group["matched_files"] == group["expected_files"]
        ):
            _fail(
                "RESOLUTION_GROUP_ALGEBRA_MISMATCH",
                f"$.clean_resolution.groups[{index}]",
                repr(group),
            )
    if resolved is not all(
        bool(_mapping(item, "$.group")["resolved"]) for item in groups
    ):
        _fail(
            "RESOLUTION_ALGEBRA_MISMATCH", "$.clean_resolution.resolved", repr(resolved)
        )
    digest_source = dict(resolution)
    observed_digest = str(digest_source.pop("resolution_digest"))
    if _object_digest(digest_source) != observed_digest:
        _fail(
            "RESOLUTION_DIGEST_MISMATCH",
            "$.clean_resolution.resolution_digest",
            observed_digest,
        )
    return copy.deepcopy(dict(resolution))


def _validate_replay_artifact(
    preregistration: Mapping[str, Any],
    authenticated: AuthenticatedInputs,
    value: Mapping[str, Any],
    *,
    preregistration_sha256: str,
    protocol_freeze_commit: str,
    materialization_receipt: Mapping[str, Any],
    clean_resolution: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _mapping(value, "$.replay_artifact")
    expected_keys = {
        "artifact_version",
        "experiment_id",
        "gate_id",
        "package_id",
        "candidate_id",
        "preregistration_sha256",
        "protocol_freeze_commit",
        "manifest_file_sha256",
        "reference_predictions_sha256",
        "reference_evidence_sha256",
        "evaluation_file_sha256",
        "evaluation_digest",
        "example_order",
        "materialization_receipt_digest",
        "clean_resolution_digest",
        "environment",
        "generation",
        "run",
        "precision_audit",
        "performance",
        "execution_network_used",
        "historical_adapter_base_path_used",
        "model_artifact_saved",
        "tensor_payload_saved",
        "outputs",
    }
    if set(replay) != expected_keys:
        _fail(
            "INVALID_REPLAY_ARTIFACT_SCHEMA", "$.replay_artifact", repr(sorted(replay))
        )
    if (
        replay.get("artifact_version") != REPLAY_ARTIFACT_VERSION
        or replay.get("experiment_id") != EXPERIMENT_ID
        or replay.get("gate_id") != GATE_ID
        or replay.get("package_id") != PACKAGE_ID
        or replay.get("candidate_id") != CANDIDATE_ID
        or replay.get("preregistration_sha256") != preregistration_sha256
        or replay.get("protocol_freeze_commit") != protocol_freeze_commit
        or replay.get("manifest_file_sha256") != MANIFEST_SHA256
        or replay.get("reference_predictions_sha256") != REFERENCE_PREDICTIONS_SHA256
        or replay.get("reference_evidence_sha256") != REFERENCE_EVIDENCE_SHA256
        or replay.get("evaluation_file_sha256") != EVALUATION_FILE_SHA256
        or replay.get("evaluation_digest") != EVALUATION_DIGEST
        or replay.get("example_order") != EXPECTED_EXAMPLE_ORDER
        or replay.get("materialization_receipt_digest")
        != _object_digest(materialization_receipt)
        or replay.get("clean_resolution_digest")
        != clean_resolution.get("resolution_digest")
        or replay.get("generation")
        != preregistration["execution_protocol"]["generation"]
        or replay.get("execution_network_used") is not False
        or replay.get("historical_adapter_base_path_used") is not False
        or replay.get("model_artifact_saved") is not False
        or replay.get("tensor_payload_saved") is not False
    ):
        _fail("REPLAY_ARTIFACT_IDENTITY_MISMATCH", "$.replay_artifact", "drift")
    environment = _mapping(replay.get("environment"), "$.replay_artifact.environment")
    _validate_finite_json(environment, "$.replay_artifact.environment")
    run = _mapping(replay.get("run"), "$.replay_artifact.run")
    expected_run = {
        "run_id": RUN_ID,
        "candidate_id": CANDIDATE_ID,
        "order_index": 0,
        "fresh_model_loads": 1,
        "full_eval_runs": 1,
        "generate_calls": EXPECTED_RECORDS,
        "retries": 0,
        "completed": True,
    }
    if run != expected_run:
        _fail(
            "REPLAY_RUN_MISMATCH",
            "$.replay_artifact.run",
            _first_difference(run, expected_run),
        )
    precision = _mapping(
        replay.get("precision_audit"), "$.replay_artifact.precision_audit"
    )
    _validate_precision_audit(precision)
    performance = _mapping(replay.get("performance"), "$.replay_artifact.performance")
    _validate_performance(performance)
    outputs = _validate_output_records(
        replay.get("outputs"),
        path="$.replay_artifact.outputs",
        require_exact_order=True,
        record_schema="replay",
    )
    result = copy.deepcopy(dict(replay))
    result["environment"] = copy.deepcopy(dict(environment))
    result["precision_audit"] = copy.deepcopy(dict(precision))
    result["performance"] = copy.deepcopy(dict(performance))
    result["outputs"] = outputs
    if authenticated.preregistration != preregistration:
        _fail("AUTHENTICATED_PREREGISTRATION_MISMATCH", "$.preregistration", "drift")
    return result


def _execution_protocol_passed(replay: Mapping[str, Any]) -> bool:
    run = replay.get("run")
    return (
        isinstance(run, Mapping)
        and run
        == {
            "run_id": RUN_ID,
            "candidate_id": CANDIDATE_ID,
            "order_index": 0,
            "fresh_model_loads": 1,
            "full_eval_runs": 1,
            "generate_calls": EXPECTED_RECORDS,
            "retries": 0,
            "completed": True,
        }
        and replay.get("generation") == GENERATION_CONTRACT
        and replay.get("execution_network_used") is False
        and replay.get("historical_adapter_base_path_used") is False
        and replay.get("model_artifact_saved") is False
        and replay.get("tensor_payload_saved") is False
        and _precision_protocol_passed(replay.get("precision_audit"))
    )


def _validate_precision_audit(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "base_parameters",
        "adapter_parameters",
        "floating_buffers",
        "lora_target_modules",
        "lora_parameter_tensors",
        "adapter_parameters_finite",
        "active_adapters",
        "is_peft_model",
        "input_output_embeddings_tied",
        "attn_implementation",
        "attention_class",
        "output_attentions",
        "hf_device_map",
        "training",
        "autocast_enabled",
        "lora_dropout",
        "autocast_adapter_dtype",
        "attached_execution_form",
    }
    if set(value) != expected_keys:
        _fail(
            "INVALID_PRECISION_AUDIT_SCHEMA", "$.precision_audit", repr(sorted(value))
        )
    if not _precision_protocol_passed(value):
        _fail("PRECISION_PROTOCOL_MISMATCH", "$.precision_audit", "drift")


def _precision_protocol_passed(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        _inventory_is_float32_cuda(value.get("base_parameters"), 1_543_714_304)
        and _inventory_is_float32_cuda(value.get("adapter_parameters"), 4_358_144)
        and _inventory_is_float32_cuda(value.get("floating_buffers"), 64)
        and value.get("lora_target_modules") == 112
        and value.get("lora_parameter_tensors") == 224
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
        and value.get("lora_dropout") == {"modules": 112, "training_modules": 0}
        and value.get("autocast_adapter_dtype") is True
        and value.get("attached_execution_form") == "attached_factorized_lora"
    )


def _inventory_is_float32_cuda(value: object, expected_elements: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    tensors = value.get("floating_tensors")
    return (
        set(value) == {"floating_tensors", "floating_elements", "dtypes", "devices"}
        and isinstance(tensors, int)
        and not isinstance(tensors, bool)
        and tensors > 0
        and value.get("floating_elements") == expected_elements
        and value.get("dtypes") == {"float32": expected_elements}
        and value.get("devices") == {"cuda:0": expected_elements}
    )


def _validate_performance(value: Mapping[str, Any]) -> dict[str, int | float]:
    expected_keys = {
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
    if set(value) != expected_keys:
        _fail("INVALID_PERFORMANCE_SCHEMA", "$.performance", repr(sorted(value)))
    elapsed = _nonnegative_number(
        value.get("elapsed_seconds"), "$.performance.elapsed_seconds"
    )
    peak = _nonnegative_int(
        value.get("peak_gpu_memory_bytes"), "$.performance.peak_gpu_memory_bytes"
    )
    before = _nonnegative_int(
        value.get("memory_allocated_before_load_bytes"),
        "$.performance.memory_allocated_before_load_bytes",
    )
    after = _nonnegative_int(
        value.get("memory_allocated_after_release_bytes"),
        "$.performance.memory_allocated_after_release_bytes",
    )
    return {
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak,
        "memory_allocated_before_load_bytes": before,
        "memory_allocated_after_release_bytes": after,
    }


def _resource_assessment(
    performance: Mapping[str, Any], caps: Mapping[str, Any]
) -> dict[str, Any]:
    measured = _validate_performance(performance)
    expected_caps = {
        "elapsed_seconds_max": MAX_ELAPSED_SECONDS,
        "peak_gpu_memory_bytes_max": MAX_PEAK_GPU_MEMORY_BYTES,
        "memory_allocated_before_load_bytes_max": MAX_RESIDUAL_GPU_MEMORY_BYTES,
        "memory_allocated_after_release_bytes_max": MAX_RESIDUAL_GPU_MEMORY_BYTES,
    }
    if dict(caps) != expected_caps:
        _fail("RESOURCE_CAP_MISMATCH", "$.resource_caps", repr(caps))
    within = {
        "elapsed_seconds": measured["elapsed_seconds"] <= MAX_ELAPSED_SECONDS,
        "peak_gpu_memory_bytes": measured["peak_gpu_memory_bytes"]
        <= MAX_PEAK_GPU_MEMORY_BYTES,
        "memory_allocated_before_load_bytes": measured[
            "memory_allocated_before_load_bytes"
        ]
        <= MAX_RESIDUAL_GPU_MEMORY_BYTES,
        "memory_allocated_after_release_bytes": measured[
            "memory_allocated_after_release_bytes"
        ]
        <= MAX_RESIDUAL_GPU_MEMORY_BYTES,
    }
    return {
        "performance": measured,
        "caps": expected_caps,
        "within_caps": within,
        "passed": all(within.values()),
    }


def _remaining_blockers(
    gates: Mapping[str, bool], behavioral_established: bool
) -> list[str]:
    blockers: list[str] = []
    if not behavioral_established:
        blockers.append("behavioral_reproducibility_unverified")
    if not (gates["materialization"] and gates["clean_location_resolution"]):
        blockers.append("clean_location_resolution_unverified")
    if not gates["resources"]:
        blockers.append("resource_budget_exceeded")
    blockers.append("remote_revision_origin_unverified")
    return blockers


def _expected_protocol_source_receipts(
    preregistration: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    observed = _mapping(
        receipt.get("protocol_sources"),
        "$.materialization_receipt.protocol_sources",
    )
    expected_sources = preregistration["source_lineage"]["protocol_sources"]
    if set(observed) != set(expected_sources):
        _fail(
            "MATERIALIZED_PROTOCOL_SOURCE_SET_MISMATCH",
            "$.materialization_receipt.protocol_sources",
            repr(sorted(observed)),
        )
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_sources):
        item = _mapping(
            observed.get(name),
            f"$.materialization_receipt.protocol_sources.{name}",
        )
        if set(item) != {"path", "sha256", "bytes"}:
            _fail(
                "INVALID_PROTOCOL_SOURCE_RECEIPT",
                f"$.materialization_receipt.protocol_sources.{name}",
                repr(sorted(item)),
            )
        size = _nonnegative_int(
            item.get("bytes"),
            f"$.materialization_receipt.protocol_sources.{name}.bytes",
        )
        expected = expected_sources[name]
        expected_record = {
            "path": expected["path"],
            "sha256": expected["sha256"],
            "bytes": size,
        }
        if item != expected_record:
            _fail(
                "PROTOCOL_SOURCE_RECEIPT_MISMATCH",
                f"$.materialization_receipt.protocol_sources.{name}",
                _first_difference(item, expected_record),
            )
        result[name] = expected_record
    return result


def _normalized_resolution_groups(
    resolution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups = resolution.get("groups")
    if not isinstance(groups, list):
        _fail("INVALID_RESOLUTION_GROUPS", "$.clean_resolution.groups", repr(groups))
    return [
        copy.deepcopy(dict(_mapping(group, "$.clean_resolution.groups")))
        for group in groups
    ]


def _resolution_issues(resolution: Mapping[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    groups = resolution.get("groups")
    if not isinstance(groups, list):
        return [{"code": "INVALID_RESOLUTION_GROUPS", "path": "clean_resolution"}]
    for group_raw in groups:
        group = _mapping(group_raw, "$.clean_resolution.groups")
        group_issues = group.get("issues")
        if not isinstance(group_issues, list):
            continue
        for issue_raw in group_issues:
            issue = _mapping(issue_raw, "$.clean_resolution.groups.issues")
            issues.append({"code": str(issue["code"]), "path": str(issue["path"])})
    return issues


def _validate_artifact_relative_path(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("INVALID_ARTIFACT_PATH", path, repr(value))
    if (
        "\\" in value
        or "\x00" in value
        or ":" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        _fail("INVALID_ARTIFACT_PATH", path, value)
    expected_name = expected_preregistration(
        freeze_status="draft",
        protocol_source_hashes={name: ZERO_SHA256 for name in PROTOCOL_SOURCE_PATHS},
    )["execution_protocol"]["output_policy"]["replay_file"]
    if value != expected_name:
        _fail("UNEXPECTED_REPLAY_ARTIFACT_PATH", path, value)
    return value


def _read_root_relative_file(
    root: Path,
    relative: str,
    *,
    path: str,
    max_bytes: int,
) -> bytes:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail("INVALID_ROOT_RELATIVE_PATH", path, repr(relative))
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail("INVALID_ROOT_RELATIVE_PATH", path, relative)
    root_resolved = root.resolve()
    if not root_resolved.is_dir() or _is_reparse(root_resolved):
        _fail("INVALID_SOURCE_ROOT", path, str(root))
    target = root_resolved.joinpath(*parts)
    try:
        target_resolved = target.resolve(strict=True)
    except OSError as exc:
        _fail("MISSING_SOURCE_FILE", path, str(exc))
    if not target_resolved.is_relative_to(root_resolved):
        _fail("SOURCE_PATH_ESCAPE", path, relative)
    return _read_regular_file(target_resolved, max_bytes=max_bytes)


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        _fail("MISSING_REGULAR_FILE", str(path), str(exc))
    if not stat.S_ISREG(before.st_mode) or _is_reparse(path):
        _fail("UNSAFE_REGULAR_FILE", str(path), "not a plain regular file")
    try:
        with path.open("rb") as stream:
            handle = os.fstat(stream.fileno())
            if _file_identity(before) != _file_identity(handle):
                _fail("FILE_HANDLE_IDENTITY_MISMATCH", str(path), "open race")
            payload = stream.read(max_bytes + 1)
    except OSError as exc:
        _fail("SOURCE_FILE_READ_FAILED", str(path), str(exc))
    if len(payload) > max_bytes:
        _fail("SOURCE_FILE_TOO_LARGE", str(path), str(len(payload)))
    try:
        after = path.lstat()
    except OSError as exc:
        _fail("SOURCE_FILE_CHANGED", str(path), str(exc))
    if _file_identity(before) != _file_identity(after):
        _fail("SOURCE_FILE_CHANGED", str(path), "identity changed")
    return payload


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _json_object_from_payload(payload: bytes, path: str) -> dict[str, Any]:
    return copy.deepcopy(
        dict(_mapping(parse_strict_json_bytes(payload, path=path), path))
    )


def _list_of_objects(value: object, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        _fail("INVALID_JSON_ARRAY", path, type(value).__name__)
    return [
        copy.deepcopy(dict(_mapping(item, f"{path}[{index}]")))
        for index, item in enumerate(value)
    ]


def _require_payload_sha256(payload: bytes, expected: str, path: str) -> None:
    if not isinstance(payload, bytes):
        _fail("INVALID_RAW_PAYLOAD", path, type(payload).__name__)
    _validate_sha256(expected, f"{path}.expected_sha256")
    observed = sha256_bytes(payload)
    if observed != expected:
        _fail("RAW_PAYLOAD_HASH_MISMATCH", path, observed)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail("INVALID_JSON_OBJECT", path, type(value).__name__)
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail("INVALID_NONNEGATIVE_INTEGER", path, repr(value))
    return value


def _nonnegative_number(value: object, path: str) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        _fail("INVALID_NONNEGATIVE_NUMBER", path, repr(value))
    return value


def _validate_sha256(value: object, path: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail("INVALID_SHA256", path, repr(value))


def _validate_git_commit(value: object, path: str) -> None:
    if not isinstance(value, str) or GIT_COMMIT_PATTERN.fullmatch(value) is None:
        _fail("INVALID_GIT_COMMIT", path, repr(value))


def _validate_finite_json(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("NONFINITE_JSON_NUMBER", path, repr(value))
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                _fail("INVALID_JSON_KEY", path, repr(key))
            _validate_finite_json(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_finite_json(nested, f"{path}[{index}]")
        return
    _fail("INVALID_JSON_TYPE", path, type(value).__name__)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("DUPLICATE_JSON_KEY", "$", key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    _fail("NONFINITE_JSON_NUMBER", "$", value)


def _first_difference(actual: object, expected: object, path: str = "$") -> str:
    if type(actual) is not type(expected):
        return f"{path}:type={type(actual).__name__}!={type(expected).__name__}"
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            return f"{path}:keys={sorted(actual_keys)}!={sorted(expected_keys)}"
        for key in sorted(actual_keys):
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return f"{path}:length={len(actual)}!={len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    if actual != expected:
        return f"{path}:{actual!r}!={expected!r}"
    return ""


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ReproducibilityContractError(code, path, detail)


__all__ = [
    "ADAPTER_ROOT_RELATIVE_TO_REPOSITORY",
    "AuthenticatedInputs",
    "CANDIDATE_ID",
    "CONTRACT_VERSION",
    "EVALUATION_FILE_SHA256",
    "EVIDENCE_VERSION",
    "EXPERIMENT_ID",
    "GATE_ID",
    "GENERATION_CONTRACT",
    "LoadedPreregistration",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "MATERIALIZER_SOURCE_PATH",
    "ManifestSourceBundle",
    "PACKAGE_ID",
    "PASS_CLASSIFICATION",
    "PASS_NEXT_GATE_ID",
    "PREREGISTRATION_PATH",
    "REFERENCE_EVIDENCE_SHA256",
    "REFERENCE_PREDICTIONS_SHA256",
    "REPLAY_ARTIFACT_VERSION",
    "RUNNER_SOURCE_PATH",
    "RUN_ID",
    "ReproducibilityContractError",
    "artifact_json_bytes",
    "authenticate_manifest_and_references",
    "build_replay_artifact",
    "build_reproducibility_evidence",
    "canonical_json_bytes",
    "classify_reproducibility_gates",
    "compare_behavioral_replay",
    "expected_preregistration",
    "load_and_validate_preregistration",
    "load_manifest_source_bundle",
    "parse_strict_json_bytes",
    "resolve_clean_roots",
    "sha256_bytes",
    "validate_materialization_receipt",
    "validate_preregistration",
    "validate_reproducibility_evidence",
]
