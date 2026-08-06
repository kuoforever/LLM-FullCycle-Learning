"""Strict metadata contract for the FP32 attached offline package identity."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

from .consumer import canonical_json_bytes
from .tool_router import ToolRouterValidationError
from .tool_router_fp32_attached_artifact_eligibility import (
    inspect_adapter_safetensors_bytes,
)

MANIFEST_VERSION = 1
EXPERIMENT_ID = "fc-mvp-001-fp32-attached-offline-package-manifest-v1"
GATE_ID = "FC-MVP-001-fp32-attached-offline-package-manifest-v1"
PACKAGE_ID = "fc-mvp-001-fp32-attached-factorized-lora-package-v1"
CANDIDATE_ID = "fp32-attached-factorized-lora"
UPSTREAM_GATE_ID = "FC-MVP-001-fp32-attached-artifact-eligibility-review-v1"
NEXT_GATE_ID = "FC-MVP-001-fp32-attached-offline-package-reproducibility-v1"

BASE_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
BASE_LICENSE = "apache-2.0"
BASE_PARAMETERS = 1_543_714_304
COMPILER_SYMBOL = "compile_decision"
COMPILER_VERSION = 1

SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

# path, role, bytes, sha256. These roots were observed before the manifest
# contract was frozen; the manifest cannot replace or broaden them.
BASE_MODEL_FILE_SPECS = (
    (
        "LICENSE",
        "license",
        11_343,
        "sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    (
        "README.md",
        "upstream_model_card",
        4_917,
        "sha256:2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada",
    ),
    (
        "config.json",
        "model_config",
        660,
        "sha256:98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    ),
    (
        "generation_config.json",
        "upstream_generation_defaults",
        242,
        "sha256:e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    ),
    (
        "model.safetensors",
        "base_weights",
        3_087_467_144,
        "sha256:dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    ),
)

TOKENIZER_FILE_SPECS = (
    (
        "merges.txt",
        "tokenizer_merges",
        1_671_839,
        "sha256:599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    (
        "tokenizer.json",
        "tokenizer_definition",
        7_031_645,
        "sha256:c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    (
        "tokenizer_config.json",
        "tokenizer_config",
        7_305,
        "sha256:5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    (
        "vocab.json",
        "tokenizer_vocabulary",
        2_776_833,
        "sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)

ADAPTER_FILE_SPECS = (
    (
        "README.md",
        "historical_adapter_card",
        5_107,
        "sha256:353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522",
    ),
    (
        "adapter_config.json",
        "peft_adapter_config",
        793,
        "sha256:8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16",
    ),
    (
        "adapter_model.safetensors",
        "fp32_lora_weights",
        17_462_432,
        "sha256:efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342",
    ),
)

SOURCE_HASH_KEYS = frozenset(
    {
        "adapter_config",
        "adapter_inspector_source",
        "adapter_readme",
        "adapter_weights",
        "canonical_json_source",
        "decision_compiler_source",
        "manifest_builder_source",
        "manifest_contract_source",
        "model_downloader_source",
        "package_init_source",
        "package_documentation",
        "prompt",
        "remediation_preregistration",
        "sft_config",
        "sft_helpers_source",
        "training_lock",
        "upstream_review",
        "validation_error_source",
    }
)

PAYLOAD_SOURCE_KEYS = SOURCE_HASH_KEYS

STATIC_SOURCE_HASHES = {
    "adapter_config": (
        "sha256:8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16"
    ),
    "adapter_inspector_source": (
        "sha256:3fa9dca9d5b309b9401be25dd3538ccbdf76df63d0eda67230a45152703c5452"
    ),
    "adapter_readme": (
        "sha256:353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522"
    ),
    "adapter_weights": (
        "sha256:efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342"
    ),
    "canonical_json_source": (
        "sha256:05cfe603d4786fb536cc1f99952a55fd211cc0fea2c210b32b575fefda9537d3"
    ),
    "decision_compiler_source": (
        "sha256:16f162a84572c7f0782890aef5aafbaafa1862e14938fe08b0ea6e97efa05157"
    ),
    "model_downloader_source": (
        "sha256:1d0d3321a55b185128de020f4b5a2a9c3ecc22f5abb0535c4712c4fd545d3a28"
    ),
    "package_init_source": (
        "sha256:45cabb5da1c0e7c2c93ef045904cf4555b0c755baf1ec2eaf47330a1aab6008e"
    ),
    "package_documentation": (
        "sha256:a531b0e462aad15a1ec9eb001d05c8cf71b5a72bde66437a499d0c6efba9cb24"
    ),
    "prompt": (
        "sha256:4a7d15063b0b074ef999c2848d0fc073a6cc00ed4999ea81f770e2e42cfa6d97"
    ),
    "remediation_preregistration": (
        "sha256:5e7b0665f97f5cee760637236f80039c4e621ae0f24915c0ac749d885a683c8b"
    ),
    "sft_config": (
        "sha256:110ada11d69f4e83c4b93da0304e62151059115487e90394d32835f6916365c8"
    ),
    "sft_helpers_source": (
        "sha256:db881e5e5955341acb735416d93062a40cf512b63ec50eb8c196ddb4371bd020"
    ),
    "training_lock": (
        "sha256:e6e23f51834b1815578368ce54c78034e72a7158395892e77fdf75594548931f"
    ),
    "upstream_review": (
        "sha256:81977f318c6bcfed8d3844575dc245d4b94c2636a2359165f9aa5553c9b006f8"
    ),
    "validation_error_source": (
        "sha256:bb3cda72585bc84bf0cf84c5736cafe29c8dfc8bca5a851d82ecfed35b1b883d"
    ),
}

REPOSITORY_SOURCE_PATHS = {
    "adapter_inspector_source": (
        "src/fullcycle_bridge/tool_router_fp32_attached_artifact_eligibility.py"
    ),
    "canonical_json_source": "src/fullcycle_bridge/consumer.py",
    "decision_compiler_source": (
        "src/fullcycle_bridge/tool_router_decision_compilation.py"
    ),
    "manifest_builder_source": (
        "scripts/build_tool_router_fp32_attached_offline_package_manifest.py"
    ),
    "manifest_contract_source": (
        "src/fullcycle_bridge/tool_router_fp32_attached_offline_package_manifest.py"
    ),
    "model_downloader_source": "scripts/download_pinned_tool_router_model.py",
    "package_documentation": (
        "docs/FC-MVP-001-fp32-attached-offline-package-use-v1.md"
    ),
    "package_init_source": "src/fullcycle_bridge/__init__.py",
    "prompt": "prompts/tool_router_v1.txt",
    "remediation_preregistration": (
        "configs/tool_router_fp32_attached_remediation_eval_v1.json"
    ),
    "sft_config": "configs/tool_router_lora_sft_v2.json",
    "sft_helpers_source": "src/fullcycle_bridge/tool_router_sft.py",
    "training_lock": "requirements/training.lock",
    "upstream_review": (
        "baseline/fc-mvp-001-fp32-attached-artifact-eligibility-review-v1.json"
    ),
    "validation_error_source": "src/fullcycle_bridge/tool_router.py",
}

PRIOR_PACKAGE_BLOCKERS = (
    "base_model_revision_binding_missing",
    "composite_manifest_missing",
    "package_use_and_limitations_documentation_incomplete",
    "portable_base_model_binding_missing",
    "required_compiler_binding_missing",
    "tokenizer_file_manifest_missing",
)

REMAINING_BLOCKERS = (
    "behavioral_reproducibility_unverified",
    "clean_location_resolution_unverified",
    "remote_revision_origin_unverified",
)

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


def expected_file_records(
    specs: Sequence[tuple[str, str, int, str]],
) -> list[dict[str, Any]]:
    """Return canonical file records for one component group."""

    return [
        {"path": path, "role": role, "bytes": size, "sha256": digest}
        for path, role, size, digest in specs
    ]


def build_fp32_attached_offline_package_manifest(
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    base_model_files: Sequence[Mapping[str, Any]],
    tokenizer_files: Sequence[Mapping[str, Any]],
    adapter_files: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Build the exact metadata-only composite manifest from external roots."""

    _validate_finite_json(upstream_review, "$.upstream_review")
    _validate_finite_json(remediation_preregistration, "$.preregistration")
    _validate_finite_json(sft_config, "$.sft_config")
    _validate_finite_json(adapter_config, "$.adapter_config")
    _validate_source_bindings(
        upstream_review,
        remediation_preregistration,
        sft_config,
        adapter_config,
        source_hashes=source_hashes,
        source_payloads=source_payloads,
    )

    expected_base = expected_file_records(BASE_MODEL_FILE_SPECS)
    expected_tokenizer = expected_file_records(TOKENIZER_FILE_SPECS)
    expected_adapter = expected_file_records(ADAPTER_FILE_SPECS)
    if list(base_model_files) != expected_base:
        _fail("BASE_MODEL_FILE_MANIFEST_MISMATCH", "$.base_model_files", "exact roots")
    if list(tokenizer_files) != expected_tokenizer:
        _fail("TOKENIZER_FILE_MANIFEST_MISMATCH", "$.tokenizer_files", "exact roots")
    if list(adapter_files) != expected_adapter:
        _fail("ADAPTER_FILE_MANIFEST_MISMATCH", "$.adapter_files", "exact roots")

    frozen_inputs = _mapping(
        remediation_preregistration.get("frozen_inputs"),
        "$.preregistration.frozen_inputs",
    )
    model = _mapping(frozen_inputs.get("model"), "$.preregistration.model")
    tokenizer = _mapping(frozen_inputs.get("tokenizer"), "$.preregistration.tokenizer")
    environment = _mapping(
        frozen_inputs.get("environment"), "$.preregistration.environment"
    )
    prompt = _mapping(frozen_inputs.get("prompt"), "$.preregistration.prompt")
    protocol = _mapping(
        remediation_preregistration.get("protocol"), "$.preregistration.protocol"
    )
    generation = _mapping(protocol.get("generation"), "$.preregistration.generation")
    _validate_candidate_contract(
        upstream_review,
        remediation_preregistration,
        sft_config,
        adapter_config,
        model=model,
        tokenizer=tokenizer,
        environment=environment,
        prompt=prompt,
        generation=generation,
        source_hashes=source_hashes,
        source_payloads=source_payloads,
    )
    adapter_tensor_audit = inspect_adapter_safetensors_bytes(
        source_payloads["adapter_weights"]
    )
    _validate_adapter_tensor_audit(adapter_tensor_audit)

    base_bytes = sum(item[2] for item in BASE_MODEL_FILE_SPECS)
    tokenizer_bytes = sum(item[2] for item in TOKENIZER_FILE_SPECS)
    adapter_bytes = sum(item[2] for item in ADAPTER_FILE_SPECS)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "gate_id": GATE_ID,
        "package_id": PACKAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "artifact_kind": "external_metadata_only_composite_manifest",
        "source_artifacts": dict(sorted(source_hashes.items())),
        "upstream_review": {
            "gate_id": upstream_review["gate_id"],
            "file_sha256": source_hashes["upstream_review"],
            "report_digest": upstream_review["report_digest"],
            "classification": upstream_review["eligibility_decision"]["classification"],
            "blocking_findings": list(
                upstream_review["packaging_review"]["blocking_findings"]
            ),
        },
        "components": {
            "base_model": {
                "repo_id": BASE_REPO_ID,
                "revision": BASE_REVISION,
                "license": BASE_LICENSE,
                "parameters": BASE_PARAMETERS,
                "checkpoint_storage_dtype": "bfloat16",
                "load_dtype": "float32",
                "files": expected_base,
                "total_bytes": base_bytes,
            },
            "tokenizer": {
                "repo_id": BASE_REPO_ID,
                "revision": BASE_REVISION,
                "files": expected_tokenizer,
                "total_bytes": tokenizer_bytes,
            },
            "adapter": {
                "adapter_id": "fc-mvp-001-lora-sft-v2",
                "execution_form": "attached_factorized_lora",
                "storage_dtype": "float32",
                "runtime_dtype": "float32",
                "merge": False,
                "files": expected_adapter,
                "total_bytes": adapter_bytes,
                "tensor_count": 224,
                "parameter_count": 4_358_144,
                "tensor_audit": copy.deepcopy(adapter_tensor_audit),
                "tensor_inspector": {
                    "path": (
                        "src/fullcycle_bridge/"
                        "tool_router_fp32_attached_artifact_eligibility.py"
                    ),
                    "file_sha256": source_hashes["adapter_inspector_source"],
                    "symbol": "inspect_adapter_safetensors_bytes",
                    "direct_dependencies": [
                        {
                            "path": "src/fullcycle_bridge/consumer.py",
                            "sha256": source_hashes["canonical_json_source"],
                        },
                        {
                            "path": "src/fullcycle_bridge/tool_router.py",
                            "sha256": source_hashes["validation_error_source"],
                        },
                        {
                            "path": "src/fullcycle_bridge/tool_router_sft.py",
                            "sha256": source_hashes["sft_helpers_source"],
                        },
                    ],
                },
                "local_base_path_authoritative": False,
                "recorded_local_base_path": adapter_config["base_model_name_or_path"],
                "recorded_revision": adapter_config.get("revision"),
            },
            "decision_compiler": {
                "path": "src/fullcycle_bridge/tool_router_decision_compilation.py",
                "file_sha256": source_hashes["decision_compiler_source"],
                "symbol": COMPILER_SYMBOL,
                "symbol_source_sha256": (
                    "sha256:1fee8097efd70242e33c57c2f4a11a2096bb089bba033f34c48dea58c8ffa8c5"
                ),
                "version": COMPILER_VERSION,
                "direct_dependencies": [
                    {
                        "path": "src/fullcycle_bridge/__init__.py",
                        "sha256": source_hashes["package_init_source"],
                    },
                    {
                        "path": "src/fullcycle_bridge/consumer.py",
                        "sha256": source_hashes["canonical_json_source"],
                    },
                    {
                        "path": "src/fullcycle_bridge/tool_router.py",
                        "sha256": source_hashes["validation_error_source"],
                    },
                ],
            },
            "prompt": {
                "path": "prompts/tool_router_v1.txt",
                "bytes": len(source_payloads["prompt"]),
                "sha256": source_hashes["prompt"],
            },
            "environment": {
                "lock_path": "requirements/training.lock",
                "lock_bytes": len(source_payloads["training_lock"]),
                "lock_sha256": source_hashes["training_lock"],
                "recorded_environment": copy.deepcopy(dict(environment)),
                "transitive_dependency_hashes_pinned": False,
            },
            "documentation": {
                "path": "docs/FC-MVP-001-fp32-attached-offline-package-use-v1.md",
                "bytes": len(source_payloads["package_documentation"]),
                "sha256": source_hashes["package_documentation"],
            },
        },
        "execution_contract": {
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
            "local_files_only": True,
            "prompt_sha256": source_hashes["prompt"],
            "generation": {
                "upstream_file_defaults": {
                    "bos_token_id": 151643,
                    "do_sample": True,
                    "eos_token_id": [151645, 151643],
                    "pad_token_id": 151643,
                    "repetition_penalty": 1.1,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "transformers_version": "4.37.0",
                },
                "effective_contract": copy.deepcopy(GENERATION_CONTRACT),
                "effective_sampling_overrides": {
                    "temperature": None,
                    "top_k": None,
                    "top_p": None,
                },
            },
            "decision_compilation": {
                "required": True,
                "order": "after_raw_parse_and_before_terminal_decision_consumption",
                "symbol": COMPILER_SYMBOL,
                "version": COMPILER_VERSION,
            },
        },
        "resolution_contract": {
            "root_roles": ["base_model_and_tokenizer", "adapter", "repository"],
            "repository_source_paths": dict(sorted(REPOSITORY_SOURCE_PATHS.items())),
            "absolute_or_caller_supplied_machine_paths_embedded": False,
            "historical_machine_relative_adapter_path_recorded": True,
            "caller_supplied_roots_required": True,
            "adapter_local_base_path_authoritative": False,
            "exact_regular_file_sets_required": True,
            "symlinks_allowed": False,
            "cache_directory_authoritative": False,
            "missing_component_fails_closed": True,
            "mismatched_component_fails_closed": True,
            "alternate_revision_fallback_allowed": False,
        },
        "provenance_scope": {
            "base_identity_roots": [
                "upstream_review",
                "remediation_preregistration",
                "observed_local_file_receipts",
            ],
            "tokenizer_hash_origin": (
                "local_snapshot_observation_frozen_by_external_manifest_sha256"
            ),
            "remote_revision_origin_attestation": "not_established",
            "clean_location_resolution_attestation": "not_established",
            "manifest_statement_authority": "component_identity_only",
        },
        "observed_file_receipts": {
            "base_model_files_observed": len(expected_base),
            "tokenizer_files_observed": len(expected_tokenizer),
            "adapter_files_observed": len(expected_adapter),
            "component_files_observed": (
                len(expected_base) + len(expected_tokenizer) + len(expected_adapter)
            ),
            "component_bytes_observed": base_bytes + tokenizer_bytes + adapter_bytes,
            "caller_supplied_or_absolute_machine_paths_recorded": False,
        },
        "use_and_limitations": {
            "contract_version": 1,
            "intended_use": "clean_location_reproducibility_test_only",
            "required_components": [
                "pinned_base_model",
                "pinned_tokenizer",
                "unchanged_fp32_adapter",
                "exact_prompt",
                "decision_compiler_v1",
                "locked_environment",
            ],
            "required_controls": [
                "schema_validation",
                "policy",
                "approval",
                "wal",
                "grounding",
                "budgets",
                "sole_desktop_boundary",
            ],
            "evidence_scope": {
                "formal_full_eval_runs": 1,
                "formal_full_eval_records": 20,
                "compiled_argument_exact_match": 0.25,
                "compiled_argument_field_f1": 0.29787234042553196,
                "raw_semantic_validity_bf16": 0.85,
                "raw_semantic_validity_fp32": 0.8,
                "fp32_to_bf16_peak_memory_ratio": 1.9896087411587269,
                "full_eval_repeat_variance_established": False,
                "external_execution_count_attested": False,
            },
            "prohibited_uses": [
                "artifact_promotion",
                "base_or_tokenizer_substitution",
                "compiler_substitution",
                "desktop_integration",
                "merged_weight_creation",
                "mcp_integration",
                "provider_integration",
                "runtime_integration",
                "serving_integration",
                "weight_copy_or_mutation",
            ],
            "unsupported_claims": [
                "behavioral_reproducibility",
                "clean_location_resolution",
                "cross_machine_reproducibility",
                "full_eval_repeatability",
                "generalization",
                "portable_package_availability",
                "pristine_fp32_checkpoint",
                "production_safety",
                "remote_revision_origin_attestation",
                "serving_capacity_latency_or_cost",
            ],
            "failure_policy": (
                "missing_unsafe_unexpected_or_mismatched_component_fails_closed"
            ),
        },
        "construction_constraints": {
            "permitted_operations": [
                "read_existing_component_bytes",
                "hash_existing_component_bytes",
                "write_metadata_manifest",
            ],
            "forbidden_operations": [
                "adapter_mutation",
                "compiler_change",
                "eval_answer_tuning",
                "full_eval_run",
                "generation_change",
                "merge_or_save_weights",
                "model_or_tokenizer_copy_or_mutation",
                "new_data",
                "prompt_change",
                "runtime_or_serving_integration",
                "training",
            ],
        },
    }
    return manifest


def validate_fp32_attached_offline_package_manifest(
    manifest_payload: bytes,
    expected_manifest_sha256: str,
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate raw bytes, then recompute against external source roots."""

    _manifest, validation = _authenticate_manifest(
        manifest_payload,
        expected_manifest_sha256,
        upstream_review,
        remediation_preregistration,
        sft_config,
        adapter_config,
        source_hashes=source_hashes,
        source_payloads=source_payloads,
        expected_source_hashes=expected_source_hashes,
    )
    return validation


def validate_and_resolve_fp32_attached_offline_package(
    manifest_payload: bytes,
    expected_manifest_sha256: str,
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
    expected_source_hashes: Mapping[str, str],
    base_model_root: Path,
    adapter_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Authenticate one raw manifest snapshot before resolving caller roots."""

    manifest, validation = _authenticate_manifest(
        manifest_payload,
        expected_manifest_sha256,
        upstream_review,
        remediation_preregistration,
        sft_config,
        adapter_config,
        source_hashes=source_hashes,
        source_payloads=source_payloads,
        expected_source_hashes=expected_source_hashes,
    )
    resolution = _build_local_resolution(
        manifest,
        expected_manifest_sha256,
        base_model_root=base_model_root,
        adapter_root=adapter_root,
        repository_root=repository_root,
    )
    return {"validation": validation, "resolution": resolution}


def _authenticate_manifest(
    manifest_payload: bytes,
    expected_manifest_sha256: str,
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
    expected_source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(manifest_payload, bytes):
        _fail("INVALID_MANIFEST_PAYLOAD", "$.manifest", type(manifest_payload).__name__)
    if len(manifest_payload) > 1024 * 1024:
        _fail("MANIFEST_TOO_LARGE", "$.manifest", str(len(manifest_payload)))
    _validate_sha256(expected_manifest_sha256, "$.expected_manifest_sha256")
    observed_manifest_sha256 = "sha256:" + hashlib.sha256(manifest_payload).hexdigest()
    if observed_manifest_sha256 != expected_manifest_sha256:
        _fail(
            "MANIFEST_FILE_HASH_MISMATCH",
            "$.manifest",
            observed_manifest_sha256,
        )
    manifest_data = _parse_json_payload(manifest_payload, "$.manifest")
    _validate_hash_map(expected_source_hashes, "$.expected_source_hashes")
    if source_hashes != expected_source_hashes:
        _fail(
            "SOURCE_HASH_ROOT_MISMATCH",
            "$.source_hashes",
            "observed hashes differ from external roots",
        )
    expected = build_fp32_attached_offline_package_manifest(
        upstream_review,
        remediation_preregistration,
        sft_config,
        adapter_config,
        base_model_files=expected_file_records(BASE_MODEL_FILE_SPECS),
        tokenizer_files=expected_file_records(TOKENIZER_FILE_SPECS),
        adapter_files=expected_file_records(ADAPTER_FILE_SPECS),
        source_hashes=source_hashes,
        source_payloads=source_payloads,
    )
    if manifest_data != expected:
        _fail(
            "MANIFEST_RECOMPUTATION_MISMATCH",
            "$.manifest",
            "frozen manifest differs",
        )
    validation = {
        "frozen_manifest_valid": True,
        "manifest_file_sha256": observed_manifest_sha256,
        "metadata_complete": True,
        "offline_package_identity_complete": True,
        "attached_package_identity_bound": True,
        "prior_package_blocker_count_resolved": len(PRIOR_PACKAGE_BLOCKERS),
        "eligible_for_clean_location_reproducibility_test": True,
        "remote_revision_origin_attested": False,
        "behavioral_reproducibility_established": False,
        "offline_artifact_eligible": False,
        "portable_package_eligible": False,
        "preferred_offline_candidate": False,
        "serving_readiness_established": False,
        "artifact_promotion_allowed": False,
        "merged_artifact_allowed": False,
        "classification": "fp32_attached_metadata_only_composite_manifest_complete",
        "remaining_blocking_findings": list(REMAINING_BLOCKERS),
        "remaining_blocking_finding_count": len(REMAINING_BLOCKERS),
        "next_gate": NEXT_GATE_ID,
        "runtime_eligible": False,
    }
    return manifest_data, validation


def resolve_component_files(
    root: Path,
    expected_files: Sequence[Mapping[str, Any]],
    *,
    root_role: str,
    allowed_directory_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Resolve one caller-supplied component root and return a fail-closed result."""

    issues: list[dict[str, str]] = []
    if not root.exists():
        issues.append({"code": "MISSING_ROOT", "path": root_role})
        return _resolution_result(root_role, expected_files, issues, 0, 0)
    if not root.is_dir() or _is_reparse(root):
        issues.append({"code": "UNSAFE_ROOT", "path": root_role})
        return _resolution_result(root_role, expected_files, issues, 0, 0)

    root_before = _stat_signature(root.lstat())
    entry_snapshot_before = _directory_entry_snapshot(root)
    expected_names = {str(item.get("path")) for item in expected_files}
    actual_file_names: set[str] = set()
    for name, kind in entry_snapshot_before:
        if kind == "reparse":
            issues.append({"code": "UNSAFE_SYMLINK", "path": f"{root_role}/{name}"})
        elif kind == "file":
            actual_file_names.add(name)
        elif kind == "directory" and name not in allowed_directory_names:
            issues.append(
                {
                    "code": "UNEXPECTED_DIRECTORY",
                    "path": f"{root_role}/{name}",
                }
            )
        elif kind not in {"directory", "file"}:
            issues.append({"code": "UNSAFE_ENTRY", "path": f"{root_role}/{name}"})
    for name in sorted(expected_names - actual_file_names):
        issues.append({"code": "MISSING_FILE", "path": f"{root_role}/{name}"})
    for name in sorted(actual_file_names - expected_names):
        issues.append({"code": "UNEXPECTED_FILE", "path": f"{root_role}/{name}"})

    matched_files = 0
    matched_bytes = 0
    records_by_name = {str(item["path"]): item for item in expected_files}
    file_receipts: dict[str, tuple[int, int, int, int, int, int]] = {}
    for name in sorted(expected_names & actual_file_names):
        path = root / name
        if _is_reparse(path) or not path.is_file():
            issues.append({"code": "UNSAFE_FILE", "path": f"{root_role}/{name}"})
            continue
        observed_bytes, observed_sha256, receipt = _hash_regular_file_receipt(path)
        file_receipts[name] = receipt
        expected = records_by_name[name]
        if observed_bytes != expected.get("bytes"):
            issues.append(
                {"code": "BYTE_COUNT_MISMATCH", "path": f"{root_role}/{name}"}
            )
        elif observed_sha256 != expected.get("sha256"):
            issues.append({"code": "SHA256_MISMATCH", "path": f"{root_role}/{name}"})
        else:
            matched_files += 1
            matched_bytes += observed_bytes
    if (
        _stat_signature(root.lstat()) != root_before
        or _directory_entry_snapshot(root) != entry_snapshot_before
    ):
        issues.append({"code": "ROOT_CHANGED_DURING_RESOLUTION", "path": root_role})
    for name, receipt in sorted(file_receipts.items()):
        path = root / name
        if (
            not path.is_file()
            or _is_reparse(path)
            or _stat_signature(path.lstat()) != receipt
        ):
            issues.append(
                {"code": "FILE_CHANGED_AFTER_HASH", "path": f"{root_role}/{name}"}
            )
    return _resolution_result(
        root_role, expected_files, issues, matched_files, matched_bytes
    )


def _directory_entry_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
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


def resolve_repository_sources(
    repository_root: Path,
    source_paths: Mapping[str, str],
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve the small repository-owned package sources by external roots."""

    issues: list[dict[str, str]] = []
    expected_count = len(REPOSITORY_SOURCE_PATHS)
    if dict(source_paths) != REPOSITORY_SOURCE_PATHS:
        issues.append({"code": "SOURCE_PATH_SET_MISMATCH", "path": "repository"})
    if not repository_root.exists():
        issues.append({"code": "MISSING_ROOT", "path": "repository"})
    elif not repository_root.is_dir() or _is_reparse(repository_root):
        issues.append({"code": "UNSAFE_ROOT", "path": "repository"})
    matched_files = 0
    matched_bytes = 0
    if not issues:
        directory_receipts = {repository_root: _stat_signature(repository_root.lstat())}
        file_receipts: dict[Path, tuple[int, int, int, int, int, int]] = {}
        casefold_paths: set[str] = set()
        for name, relative in sorted(REPOSITORY_SOURCE_PATHS.items()):
            path = _safe_repository_source_path(
                repository_root,
                relative,
                directory_receipts,
                casefold_paths,
            )
            if path is None or not path.is_file() or _is_reparse(path):
                issues.append({"code": "MISSING_OR_UNSAFE_FILE", "path": relative})
                continue
            observed_bytes, observed_sha256, receipt = _hash_regular_file_receipt(path)
            file_receipts[path] = receipt
            if observed_sha256 != expected_source_hashes.get(name):
                issues.append({"code": "SHA256_MISMATCH", "path": relative})
            else:
                matched_files += 1
                matched_bytes += observed_bytes
        for path, receipt in sorted(
            directory_receipts.items(), key=lambda item: str(item[0]).casefold()
        ):
            if (
                not path.is_dir()
                or _is_reparse(path)
                or _stat_signature(path.lstat()) != receipt
            ):
                issues.append(
                    {
                        "code": "REPOSITORY_DIRECTORY_CHANGED_DURING_RESOLUTION",
                        "path": path.relative_to(repository_root).as_posix() or ".",
                    }
                )
        for path, receipt in sorted(
            file_receipts.items(), key=lambda item: str(item[0]).casefold()
        ):
            if (
                not path.is_file()
                or _is_reparse(path)
                or _stat_signature(path.lstat()) != receipt
            ):
                issues.append(
                    {
                        "code": "REPOSITORY_FILE_CHANGED_AFTER_HASH",
                        "path": path.relative_to(repository_root).as_posix(),
                    }
                )
    return {
        "root_role": "repository",
        "resolved": not issues and matched_files == expected_count,
        "expected_files": expected_count,
        "matched_files": matched_files,
        "matched_bytes": matched_bytes,
        "issues": issues,
    }


def _safe_repository_source_path(
    repository_root: Path,
    relative: str,
    directory_receipts: dict[Path, tuple[int, int, int, int, int, int]],
    casefold_paths: set[str],
) -> Path | None:
    if not relative or "\\" in relative or "\x00" in relative or ":" in relative:
        return None
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or str(pure) != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        return None
    folded = relative.casefold()
    if folded in casefold_paths:
        return None
    casefold_paths.add(folded)
    current = repository_root
    for part in pure.parts[:-1]:
        current = current / part
        if not current.is_dir() or _is_reparse(current):
            return None
        directory_receipts.setdefault(current, _stat_signature(current.lstat()))
    return current / pure.parts[-1]


def _build_local_resolution(
    manifest_data: Mapping[str, Any],
    expected_manifest_sha256: str,
    *,
    base_model_root: Path,
    adapter_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Resolve all local roots without treating manifest path hints as authority."""

    _validate_sha256(expected_manifest_sha256, "$.expected_manifest_sha256")
    components = manifest_data.get("components")
    sources = manifest_data.get("source_artifacts")
    if not isinstance(components, Mapping) or not isinstance(sources, Mapping):
        _fail("INVALID_MANIFEST_FOR_RESOLUTION", "$.manifest", "missing roots")
    expected_base = expected_file_records(BASE_MODEL_FILE_SPECS)
    expected_tokenizer = expected_file_records(TOKENIZER_FILE_SPECS)
    expected_adapter = expected_file_records(ADAPTER_FILE_SPECS)
    base_component = _mapping(components.get("base_model"), "$.components.base_model")
    tokenizer_component = _mapping(
        components.get("tokenizer"), "$.components.tokenizer"
    )
    adapter_component = _mapping(components.get("adapter"), "$.components.adapter")
    resolution_contract = _mapping(
        manifest_data.get("resolution_contract"), "$.resolution_contract"
    )
    if (
        base_component.get("files") != expected_base
        or tokenizer_component.get("files") != expected_tokenizer
        or adapter_component.get("files") != expected_adapter
        or resolution_contract.get("repository_source_paths")
        != dict(sorted(REPOSITORY_SOURCE_PATHS.items()))
    ):
        _fail("INVALID_MANIFEST_FOR_RESOLUTION", "$.components", "file roots drift")

    base = resolve_component_files(
        base_model_root,
        [*expected_base, *expected_tokenizer],
        root_role="base_model_and_tokenizer",
        allowed_directory_names=frozenset({".cache"}),
    )
    adapter = resolve_component_files(
        adapter_root,
        expected_adapter,
        root_role="adapter",
    )
    repository = resolve_repository_sources(
        repository_root,
        REPOSITORY_SOURCE_PATHS,
        sources,
    )
    groups = [base, adapter, repository]
    resolved = all(group["resolved"] for group in groups)
    result: dict[str, Any] = {
        "resolution_version": 1,
        "package_id": manifest_data.get("package_id"),
        "manifest_file_sha256": expected_manifest_sha256,
        "caller_supplied_roots": True,
        "manifest_machine_paths_used": False,
        "adapter_local_base_path_used": False,
        "resolved": resolved,
        "eligible_for_clean_location_reproducibility_test": resolved,
        "offline_artifact_eligible": False,
        "runtime_eligible": False,
        "groups": groups,
        "failure_mode": None if resolved else "component_resolution_failed_closed",
    }
    result["resolution_digest"] = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    )
    return result


def hash_regular_file(path: Path) -> tuple[int, str]:
    """Hash one regular non-symlink file through one stable stream."""

    observed_bytes, observed_sha256, _signature = _hash_regular_file_receipt(path)
    return observed_bytes, observed_sha256


def _hash_regular_file_receipt(
    path: Path,
) -> tuple[int, str, tuple[int, int, int, int, int, int]]:
    """Hash one open file and bind its handle identity to the path before/after."""

    if not path.is_file() or _is_reparse(path):
        raise ValueError(f"unsafe or missing regular file: {path}")
    before = path.lstat()
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if _handle_identity_signature(before) != _handle_identity_signature(opened):
            raise ValueError(f"file identity changed before hashing: {path}")
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
        raise ValueError(f"file changed while hashing: {path}")
    return (
        observed_bytes,
        "sha256:" + digest.hexdigest(),
        _stat_signature(after),
    )


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
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
    # On Windows, handle fstat may report birth time in st_ctime_ns while a path
    # stat reports metadata-change time. Compare ctime only path-to-path.
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


def _validate_source_bindings(
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
) -> None:
    _validate_hash_map(source_hashes, "$.source_hashes")
    if frozenset(source_payloads) != PAYLOAD_SOURCE_KEYS:
        _fail("INVALID_SOURCE_PAYLOAD_KEYS", "$.source_payloads", repr(source_payloads))
    for name, expected in STATIC_SOURCE_HASHES.items():
        if source_hashes.get(name) != expected:
            _fail(
                "STATIC_SOURCE_HASH_MISMATCH", f"$.source_hashes.{name}", repr(expected)
            )
    for name, payload in source_payloads.items():
        if not isinstance(payload, bytes):
            _fail(
                "INVALID_SOURCE_PAYLOAD",
                f"$.source_payloads.{name}",
                type(payload).__name__,
            )
        observed = "sha256:" + hashlib.sha256(payload).hexdigest()
        if observed != source_hashes[name]:
            _fail("SOURCE_PAYLOAD_HASH_MISMATCH", f"$.source_payloads.{name}", observed)

    parsed_objects = {
        "upstream_review": upstream_review,
        "remediation_preregistration": remediation_preregistration,
        "sft_config": sft_config,
        "adapter_config": adapter_config,
    }
    for name, value in parsed_objects.items():
        parsed = _parse_json_payload(source_payloads[name], f"$.source_payloads.{name}")
        if parsed != value:
            _fail("SOURCE_OBJECT_PAYLOAD_MISMATCH", f"$.{name}", "parsed object drift")


def _validate_candidate_contract(
    upstream_review: Mapping[str, Any],
    remediation_preregistration: Mapping[str, Any],
    sft_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    environment: Mapping[str, Any],
    prompt: Mapping[str, Any],
    generation: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    source_payloads: Mapping[str, bytes],
) -> None:
    decision = _mapping(
        upstream_review.get("eligibility_decision"), "$.upstream_review.decision"
    )
    packaging = _mapping(
        upstream_review.get("packaging_review"), "$.upstream_review.packaging"
    )
    locked = _mapping(
        upstream_review.get("locked_next_action"), "$.upstream_review.next"
    )
    if (
        upstream_review.get("gate_id") != UPSTREAM_GATE_ID
        or upstream_review.get("report_digest")
        != "sha256:285d5e5e25dfd16de5adc6cb760fe54588af68d8580308b54ccfaf612d51636b"
        or decision.get("classification")
        != (
            "fp32_attached_fixed_compiler_favorable_eval_but_offline_artifact_"
            "package_incomplete"
        )
        or decision.get("repository_local_evidence_usable") is not True
        or decision.get("offline_artifact_eligible") is not False
        or tuple(packaging.get("blocking_findings", ())) != PRIOR_PACKAGE_BLOCKERS
        or locked.get("gate_id") != GATE_ID
        or upstream_review.get("runtime_eligible") is not False
    ):
        _fail("UPSTREAM_REVIEW_MISMATCH", "$.upstream_review", "negative review")

    expected_model = {
        "repo_id": BASE_REPO_ID,
        "revision": BASE_REVISION,
        "license": BASE_LICENSE,
        "parameters": BASE_PARAMETERS,
        "weight_file": "model.safetensors",
        "weight_bytes": 3_087_467_144,
        "weight_sha256": BASE_MODEL_FILE_SPECS[-1][3].removeprefix("sha256:"),
    }
    expected_tokenizer = {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION}
    if model != expected_model or tokenizer != expected_tokenizer:
        _fail("BASE_OR_TOKENIZER_IDENTITY_MISMATCH", "$.preregistration", "identity")
    if sft_config.get("model") != model or sft_config.get("tokenizer") != tokenizer:
        _fail("SFT_IDENTITY_MISMATCH", "$.sft_config", "model/tokenizer")
    if sft_config.get("environment") != environment:
        _fail("ENVIRONMENT_IDENTITY_MISMATCH", "$.sft_config.environment", "frozen")

    candidate = _mapping(
        remediation_preregistration.get("candidate"), "$.preregistration.candidate"
    )
    expected_candidate = {
        "adapter_runtime_dtype": "float32",
        "adapter_storage_dtype": "float32",
        "autocast_adapter_dtype": True,
        "base_checkpoint_storage_dtype": "bfloat16",
        "base_checkpoint_value_semantics": (
            "unchanged_bf16_checkpoint_source_values_materialized_as_float32"
        ),
        "base_load_dtype": "float32",
        "candidate_id": CANDIDATE_ID,
        "execution_form": "attached_factorized_lora",
        "merge": False,
        "run_id": "fp32-attached-full-eval-r1",
        "save_model": False,
        "save_tensors": False,
    }
    if candidate != expected_candidate or dict(generation) != GENERATION_CONTRACT:
        _fail(
            "EXECUTION_CONTRACT_MISMATCH", "$.preregistration", "candidate/generation"
        )
    if (
        prompt.get("path") != "prompts/tool_router_v1.txt"
        or prompt.get("sha256") != source_hashes["prompt"]
        or adapter_config.get("base_model_name_or_path")
        != "work\\models\\Qwen2.5-1.5B-Instruct"
        or adapter_config.get("revision") is not None
    ):
        _fail("PROMPT_OR_ADAPTER_METADATA_MISMATCH", "$.preregistration", "metadata")

    compiler_text = _decode_utf8(
        source_payloads["decision_compiler_source"], "$.decision_compiler_source"
    )
    if _symbol_sha256(compiler_text, COMPILER_SYMBOL) != (
        "sha256:1fee8097efd70242e33c57c2f4a11a2096bb089bba033f34c48dea58c8ffa8c5"
    ):
        _fail("COMPILER_SYMBOL_HASH_MISMATCH", "$.decision_compiler", COMPILER_SYMBOL)

    documentation = _decode_utf8(
        source_payloads["package_documentation"], "$.package_documentation"
    )
    if not documentation:
        _fail("PACKAGE_DOCUMENTATION_EMPTY", "$.package_documentation", "empty")

    training_lock = _decode_utf8(source_payloads["training_lock"], "$.training_lock")
    package_names = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
    }
    for environment_name, lock_name in package_names.items():
        expected_line = f"{lock_name}=={environment[environment_name]}"
        if expected_line not in training_lock.splitlines():
            _fail("ENVIRONMENT_LOCK_MISMATCH", "$.training_lock", expected_line)

    downloader = _decode_utf8(
        source_payloads["model_downloader_source"], "$.model_downloader_source"
    )
    for marker in (BASE_REPO_ID, BASE_REVISION, "model.safetensors", "tokenizer.json"):
        if marker not in downloader:
            _fail("MODEL_DOWNLOADER_MISMATCH", "$.model_downloader_source", marker)


def _validate_adapter_tensor_audit(audit: Mapping[str, Any]) -> None:
    if (
        audit.get("format") != "safetensors"
        or audit.get("file_bytes") != 17_462_432
        or audit.get("data_bytes") != 17_432_576
        or audit.get("tensor_count") != 224
        or audit.get("parameter_count") != 4_358_144
        or audit.get("dtype_tensor_counts") != {"F32": 224}
        or audit.get("dtype_element_counts") != {"F32": 4_358_144}
        or audit.get("layer_count") != 28
        or audit.get("layers") != list(range(28))
        or audit.get("module_count") != 112
        or audit.get("shape_counts") != {"1536x16": 56, "16x1536": 112, "256x16": 56}
        or audit.get("target_module_tensor_counts")
        != {"k_proj": 56, "o_proj": 56, "q_proj": 56, "v_proj": 56}
        or audit.get("lora_matrix_tensor_counts") != {"A": 112, "B": 112}
        or audit.get("data_offsets_contiguous") is not True
        or audit.get("topology_complete") is not True
    ):
        _fail("ADAPTER_TENSOR_AUDIT_MISMATCH", "$.adapter.tensor_audit", repr(audit))


def _validate_hash_map(value: Mapping[str, str], path: str) -> None:
    if frozenset(value) != SOURCE_HASH_KEYS or any(
        not isinstance(item, str) or SHA256_PATTERN.fullmatch(item) is None
        for item in value.values()
    ):
        _fail("INVALID_SOURCE_HASHES", path, repr(value))


def _validate_sha256(value: object, path: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail("INVALID_SHA256", path, repr(value))


def _symbol_sha256(source: str, symbol: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolRouterValidationError(
            "INVALID_COMPILER_SOURCE", "$.decision_compiler", str(exc)
        ) from exc
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        _fail("COMPILER_SYMBOL_MISMATCH", "$.decision_compiler", symbol)
    lines = source.splitlines(keepends=True)
    node = nodes[0]
    payload = "".join(lines[node.lineno - 1 : node.end_lineno]).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _resolution_result(
    root_role: str,
    expected_files: Sequence[Mapping[str, Any]],
    issues: list[dict[str, str]],
    matched_files: int,
    matched_bytes: int,
) -> dict[str, Any]:
    return {
        "root_role": root_role,
        "resolved": not issues and matched_files == len(expected_files),
        "expected_files": len(expected_files),
        "matched_files": matched_files,
        "matched_bytes": matched_bytes,
        "issues": issues,
    }


def _parse_json_payload(payload: bytes, path: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolRouterValidationError("INVALID_SOURCE_JSON", path, str(exc)) from exc
    if not isinstance(value, dict):
        _fail("EXPECTED_OBJECT", path, type(value).__name__)
    _validate_finite_json(value, path)
    return value


def _decode_utf8(payload: bytes, path: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolRouterValidationError("INVALID_SOURCE_TEXT", path, str(exc)) from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_finite_json(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("NON_FINITE_NUMBER", path, repr(value))
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EXPECTED_OBJECT", path, type(value).__name__)
    return value


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = [
    "ADAPTER_FILE_SPECS",
    "BASE_MODEL_FILE_SPECS",
    "EXPERIMENT_ID",
    "GATE_ID",
    "NEXT_GATE_ID",
    "PACKAGE_ID",
    "REPOSITORY_SOURCE_PATHS",
    "SOURCE_HASH_KEYS",
    "TOKENIZER_FILE_SPECS",
    "build_fp32_attached_offline_package_manifest",
    "expected_file_records",
    "hash_regular_file",
    "resolve_component_files",
    "validate_and_resolve_fp32_attached_offline_package",
    "validate_fp32_attached_offline_package_manifest",
]
