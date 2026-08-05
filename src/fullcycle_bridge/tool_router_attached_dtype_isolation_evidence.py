"""Fail-closed JSON evidence validation for attached dtype isolation.

The frozen artifact deliberately does not contain raw score/logit vectors.  This
validator therefore recomputes token, manifest-linkage, repeat, classification,
gate, and resource claims, while limiting vector-delta validation to finite
algebraic bounds and values exposed in the JSON summary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from .tool_router import ToolRouterValidationError
from .tool_router_attached_dtype_isolation import (
    analyze_attached_dtype_tokens,
    analyze_path_repeat_stability,
    classify_attached_dtype_effect,
    select_locked_comparison_step,
)
from .tool_router_merge_remediation import token_ids_sha256

BF16_PATH = "bf16_attached_adapter"
FP32_PATH = "fp32_attached_adapter"
PATHS = (BF16_PATH, FP32_PATH)
TARGET_STEP_INDEX = 45
TOKEN_COUNT = 48
VOCABULARY_SIZE = 151_936
BF16_TOKEN_ID = 1866
FP32_TOKEN_ID = 3849
TOP_K = 5
MAX_RESIDUAL_CUDA_BYTES = 16 * 1024 * 1024

RUN_PLAN = (
    ("bf16-attached-dtype-r1", BF16_PATH, 1, 0),
    ("fp32-attached-dtype-r1", FP32_PATH, 1, 1),
    ("fp32-attached-dtype-r2", FP32_PATH, 2, 2),
    ("bf16-attached-dtype-r2", BF16_PATH, 2, 3),
)

EXPECTED_SOURCE_LINEAGE = {
    "stability_evidence_sha256": (
        "sha256:82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc"
    ),
    "drift_evidence_sha256": (
        "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3"
    ),
    "isolation_evidence_sha256": (
        "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26"
    ),
    "fp32_numerics_evidence_sha256": (
        "sha256:cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91"
    ),
    "training_evidence_sha256": (
        "sha256:641b1a7ef3dc0de0d9f2124b9122cb2c4be46b42de9265d558ab6f5b25b41a30"
    ),
}

EXPECTED_STEP_MANIFEST_SHA256 = {
    BF16_PATH: {
        "scores": (
            "sha256:91e18d967889e83f996afa992169465583356b543ba06b592018f2fbbbd9b696"
        ),
        "raw_logits": (
            "sha256:02ef5804633291107c2a2f1ee4b2fca58561eef6dfa5839c787263dbe9e06bd6"
        ),
    },
    FP32_PATH: {
        "scores": (
            "sha256:fccefa0a44b8dfa9d93d3db7b575c7e520edcf41e9f10564e12bd5587e3226c0"
        ),
        "raw_logits": (
            "sha256:39bc9224964233181e552091579b2b23977c3f41cf806aa082fc841d958d016e"
        ),
    },
}

EXPECTED_ADAPTER_FILES = [
    {
        "path": "adapter_config.json",
        "bytes": 793,
        "sha256": (
            "sha256:8eb104c3af2f4deb3abe5e471b3d3a74cb306683c1fdadb95488de981ba14c16"
        ),
    },
    {
        "path": "adapter_model.safetensors",
        "bytes": 17_462_432,
        "sha256": (
            "sha256:efb62471e105b8ef25641200967d447b8cc2f3ff565937bc47193fbf79f4f342"
        ),
    },
    {
        "path": "README.md",
        "bytes": 5_107,
        "sha256": (
            "sha256:353053cad9659d849cbf1fdacc7d9b86b82fb72197e2d101785843a4109bc522"
        ),
    },
]

EXPECTED_ENVIRONMENT = {
    "python": "3.12.12",
    "torch": "2.6.0+cu124",
    "transformers": "4.49.0",
    "peft": "0.14.0",
    "accelerate": "1.3.0",
    "huggingface_hub": "0.29.3",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.4",
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
    "gpu_vram_bytes": 17_170_956_288,
    "compute_capability": "8.9",
}

EXPECTED_REFERENCES = {
    BF16_PATH: {
        "source_experiment_ids": [
            "fc-mvp-001-bf16-merge-stability-v1",
            "fc-mvp-001-fp32-merge-drift-analysis-v1",
        ],
        "token_ids_sha256": (
            "sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173"
        ),
        "output_sha256": (
            "sha256:b3bef0f22aad858ad94275466af0cc082dbb7fe42a7814320aa4a8995dff0bc5"
        ),
        "score_trace_sha256": (
            "sha256:116cc75b8420f5e95188818583e490a689de9392d34438c1d569e66822d7ee49"
        ),
        "raw_logit_trace_sha256": (
            "sha256:a7aab2daf284030ff1eab20b01b77f59064b1fc341ba59765e3d0b957c8174cf"
        ),
        "comparison_score_vector_sha256": (
            "sha256:5b78c36066365bb9c52a4894b6f642006fe891552ebc0d6a294f82aa9a8a80db"
        ),
        "comparison_raw_logit_vector_sha256": (
            "sha256:aa7ae2fab3c2be5b0ddeecb7e4a10d01dcfd8636a6a404d7e48e9ef19eb9bf9e"
        ),
        "boundary_token_id": BF16_TOKEN_ID,
    },
    FP32_PATH: {
        "source_experiment_ids": [
            "fc-mvp-001-fp32-attached-merge-isolation-v1",
            "fc-mvp-001-fp32-attached-merge-numerics-v1",
        ],
        "token_ids_sha256": (
            "sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca"
        ),
        "output_sha256": (
            "sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca"
        ),
        "score_trace_sha256": (
            "sha256:e878f06653e43ebf6946a00396fbed7797eecc02dcf25501f0738169a932fdde"
        ),
        "raw_logit_trace_sha256": (
            "sha256:61a891ab427bce3002c3367e2faefd854a11ecb62929d5b187b974a9c3b7f357"
        ),
        "comparison_score_vector_sha256": (
            "sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451"
        ),
        "comparison_raw_logit_vector_sha256": (
            "sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a"
        ),
        "boundary_token_id": FP32_TOKEN_ID,
    },
}

_TOP_LEVEL_KEYS = frozenset(
    {
        "attached_dtype_isolation_version",
        "experiment_id",
        "source_experiment_id",
        "source_lineage",
        "training_lock_sha256",
        "config_sha256",
        "adapter_files",
        "model_weight_sha256",
        "prompt_sha256",
        "eval_digest",
        "example_id",
        "input_token_count",
        "input_token_ids_sha256",
        "storage_audit",
        "environment",
        "protocol",
        "frozen_path_references",
        "runs",
        "path_repeat_stability",
        "path_reproduction",
        "cross_dtype_token_analysis",
        "comparison_step",
        "selection_score_evidence",
        "raw_logit_evidence",
        "cross_dtype_trace_identity",
        "classification",
        "causal_scope",
        "dtype_isolation_gate",
        "remediation_gate",
        "acceptance",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "merged_artifact_saved",
        "merged_artifact_allowed",
        "constraints",
        "locked_next_action",
        "runtime_eligible",
        "runtime_eligibility_reason",
        "offline",
    }
)

_RUN_KEYS = frozenset(
    {
        "run_id",
        "path",
        "repeat",
        "order_index",
        "fresh_load",
        "base_load_dtype",
        "generated_token_ids",
        "token_count",
        "token_ids_sha256",
        "output_sha256",
        "precision_audit",
        "generation_trace",
        "target_alignment",
        "target_alignment_passed",
        "lm_head_raw_logit_linked",
        "path_protocol_passed",
        "elapsed_seconds",
        "peak_gpu_memory_bytes",
        "memory_allocated_before_load_bytes",
        "memory_allocated_after_release_bytes",
    }
)

_TRACE_VECTOR_KEYS = frozenset(
    {
        "native_dtypes",
        "shape_per_step",
        "comparison_dtype",
        "all_finite",
        "trace_sha256",
        "comparison_vector_sha256_per_step",
        "comparison_step_index",
        "comparison_step_vector_sha256",
    }
)

_ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "deterministic_bf16_attached_vs_fp32_attached_raw_logit_boundary_flip",
        (
            "deterministic_bf16_attached_vs_fp32_attached_"
            "logits_processor_boundary_flip"
        ),
        (
            "deterministic_bf16_attached_vs_fp32_attached_"
            "mixed_raw_logit_and_logits_processor_boundary_flip"
        ),
    }
)


def validate_attached_dtype_isolation_evidence(
    data: Mapping[str, Any],
    *,
    expected_source_lineage: Mapping[str, str] | None = None,
    expected_adapter_files: Sequence[Mapping[str, Any]] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_step_manifest_sha256: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate the complete frozen JSON artifact without loading model code."""

    evidence = _mapping(data, "$", _TOP_LEVEL_KEYS, "INVALID_EVIDENCE_SCHEMA")
    _finite_tree(evidence, "$")
    lineage = (
        EXPECTED_SOURCE_LINEAGE
        if expected_source_lineage is None
        else dict(expected_source_lineage)
    )
    adapters = (
        EXPECTED_ADAPTER_FILES
        if expected_adapter_files is None
        else [dict(item) for item in expected_adapter_files]
    )
    environment = (
        EXPECTED_ENVIRONMENT
        if expected_environment is None
        else dict(expected_environment)
    )
    manifest_locks = _validate_manifest_locks(
        EXPECTED_STEP_MANIFEST_SHA256
        if expected_step_manifest_sha256 is None
        else expected_step_manifest_sha256
    )
    _require_exact(
        evidence["attached_dtype_isolation_version"],
        1,
        "INVALID_FROZEN_LOCK",
        "$.attached_dtype_isolation_version",
    )
    for key, expected in {
        "experiment_id": "fc-mvp-001-attached-dtype-isolation-v1",
        "source_experiment_id": "fc-mvp-001-lora-sft-v2",
        "training_lock_sha256": (
            "sha256:e6e23f51834b1815578368ce54c78034e72a7158395892e77fdf75594548931f"
        ),
        "config_sha256": (
            "sha256:5a038ea786526f188c796a6e5eea4c4d3aa47fc66977dc4f6ff16f52999236d8"
        ),
        "model_weight_sha256": (
            "sha256:dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
        ),
        "prompt_sha256": (
            "sha256:4a7d15063b0b074ef999c2848d0fc073a6cc00ed4999ea81f770e2e42cfa6d97"
        ),
        "eval_digest": (
            "sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a"
        ),
        "example_id": "eval-001",
        "input_token_count": 339,
        "input_token_ids_sha256": (
            "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
        ),
    }.items():
        _require_exact(evidence[key], expected, "INVALID_FROZEN_LOCK", f"$.{key}")
    lineage_keys = frozenset(EXPECTED_SOURCE_LINEAGE)
    expected_lineage = _mapping(
        lineage, "$expected_source_lineage", lineage_keys, "INVALID_EXPECTED_LINEAGE"
    )
    actual_lineage = _mapping(
        evidence["source_lineage"],
        "$.source_lineage",
        lineage_keys,
        "INVALID_SOURCE_LINEAGE",
    )
    for key in lineage_keys:
        _sha256(expected_lineage[key], f"$expected_source_lineage.{key}")
        _sha256(actual_lineage[key], f"$.source_lineage.{key}")
    _require_exact(
        actual_lineage, expected_lineage, "INVALID_SOURCE_LINEAGE", "$.source_lineage"
    )
    _validate_adapter_files(evidence["adapter_files"], adapters)
    _require_exact(
        evidence["environment"], environment, "INVALID_ENVIRONMENT", "$.environment"
    )
    _require_exact(
        evidence["storage_audit"], _expected_storage_audit(), "INVALID_STORAGE_AUDIT", "$.storage_audit"
    )
    _require_exact(
        evidence["protocol"], _expected_protocol(), "INVALID_PROTOCOL", "$.protocol"
    )

    references = _validate_references(evidence["frozen_path_references"])
    runs = _validate_runs(evidence["runs"], references, manifest_locks)
    by_path = {path: [run for run in runs if run["path"] == path] for path in PATHS}
    _validate_repeat_manifest_identity(by_path)
    representatives = {path: by_path[path][0] for path in PATHS}

    repeat = _recompute_repeat_stability(by_path)
    _require_exact(
        evidence["path_repeat_stability"],
        repeat,
        "REPEAT_STABILITY_MISMATCH",
        "$.path_repeat_stability",
    )
    reproduction = _recompute_path_reproduction(by_path, references)
    _require_exact(
        evidence["path_reproduction"],
        reproduction,
        "PATH_REPRODUCTION_MISMATCH",
        "$.path_reproduction",
    )

    token_analysis = analyze_attached_dtype_tokens(
        representatives[BF16_PATH]["generated_token_ids"],
        representatives[FP32_PATH]["generated_token_ids"],
    )
    if (
        token_analysis.get("first_divergent_token_index") != TARGET_STEP_INDEX
        or token_analysis.get("bf16_token_id") != BF16_TOKEN_ID
        or token_analysis.get("fp32_token_id") != FP32_TOKEN_ID
    ):
        _fail("FROZEN_TOKEN_BOUNDARY_MISMATCH", "$.runs", repr(token_analysis))
    stored_token_analysis = {
        **token_analysis,
        "bf16_token_text": "true",
        "fp32_token_text": "false",
    }
    _require_exact(
        evidence["cross_dtype_token_analysis"],
        stored_token_analysis,
        "TOKEN_ANALYSIS_MISMATCH",
        "$.cross_dtype_token_analysis",
    )
    comparison = select_locked_comparison_step(
        token_analysis, frozen_boundary_index=TARGET_STEP_INDEX
    )
    _require_exact(
        evidence["comparison_step"],
        comparison,
        "COMPARISON_STEP_MISMATCH",
        "$.comparison_step",
    )

    score_tops = _validate_step_evidence(
        evidence["selection_score_evidence"],
        representatives,
        kind="scores",
    )
    raw_tops = _validate_step_evidence(
        evidence["raw_logit_evidence"],
        representatives,
        kind="raw_logits",
    )
    trace_identity = _recompute_cross_dtype_identity(representatives)
    _require_exact(
        evidence["cross_dtype_trace_identity"],
        trace_identity,
        "TRACE_IDENTITY_MISMATCH",
        "$.cross_dtype_trace_identity",
    )
    classification = classify_attached_dtype_effect(
        token_analysis,
        bf16_repeat_stable=repeat[BF16_PATH]["passed"],
        fp32_repeat_stable=repeat[FP32_PATH]["passed"],
        bf16_reference_reproduced=reproduction[BF16_PATH]["passed"],
        fp32_reference_reproduced=reproduction[FP32_PATH]["passed"],
        bf16_emitted_token_id=BF16_TOKEN_ID,
        fp32_emitted_token_id=FP32_TOKEN_ID,
        bf16_score_top_token_id=score_tops[BF16_PATH],
        fp32_score_top_token_id=score_tops[FP32_PATH],
        bf16_raw_logit_top_token_id=raw_tops[BF16_PATH],
        fp32_raw_logit_top_token_id=raw_tops[FP32_PATH],
    )
    if classification not in _ALLOWED_CLASSIFICATIONS:
        _fail("INVALID_CLASSIFICATION", "$.classification", classification)
    _require_exact(
        evidence["classification"],
        classification,
        "CLASSIFICATION_MISMATCH",
        "$.classification",
    )
    _validate_derived_claims(evidence, runs, repeat, reproduction, classification)
    _validate_resources(evidence, runs)
    _validate_frozen_policy(evidence)
    return {
        "frozen_gate_valid": True,
        "runs_validated": len(runs),
        "token_digests_recomputed": len(runs) + len(references),
        "comparison_step_manifests_validated": len(runs) * 2,
        "classification": classification,
        "delta_statistics_scope": "probe_derived_summary_algebra_only",
    }


def _validate_manifest_locks(
    value: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    locks = _mapping(
        value,
        "$.expected_step_manifest_sha256",
        frozenset(PATHS),
        "INVALID_EXPECTED_STEP_MANIFEST",
    )
    result: dict[str, dict[str, str]] = {}
    for path_name in PATHS:
        path = f"$.expected_step_manifest_sha256.{path_name}"
        path_locks = _mapping(
            locks[path_name],
            path,
            frozenset({"scores", "raw_logits"}),
            "INVALID_EXPECTED_STEP_MANIFEST",
        )
        result[path_name] = {
            kind: _sha256(path_locks[kind], f"{path}.{kind}")
            for kind in ("scores", "raw_logits")
        }
    return result


def _manifest_sha256(value: list[Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_adapter_files(value: object, expected: list[dict[str, Any]]) -> None:
    items = _list(value, "$.adapter_files")
    for index, item in enumerate(items):
        record = _mapping(
            item,
            f"$.adapter_files[{index}]",
            frozenset({"path", "bytes", "sha256"}),
            "INVALID_ADAPTER_MANIFEST",
        )
        if not isinstance(record["path"], str) or not record["path"]:
            _fail("INVALID_ADAPTER_MANIFEST", f"$.adapter_files[{index}].path", repr(record["path"]))
        _positive_int(record["bytes"], f"$.adapter_files[{index}].bytes", "INVALID_ADAPTER_MANIFEST")
        _sha256(record["sha256"], f"$.adapter_files[{index}].sha256")
    _require_exact(items, expected, "INVALID_ADAPTER_MANIFEST", "$.adapter_files")


def _validate_references(value: object) -> dict[str, Mapping[str, Any]]:
    refs = _mapping(value, "$.frozen_path_references", frozenset(PATHS), "INVALID_REFERENCE")
    result: dict[str, Mapping[str, Any]] = {}
    keys = frozenset(
        {
            "path",
            "source_experiment_ids",
            "generated_token_ids",
            "token_count",
            "token_ids_sha256",
            "output_sha256",
            "score_trace_sha256",
            "raw_logit_trace_sha256",
            "comparison_step_index",
            "comparison_score_vector_sha256",
            "comparison_raw_logit_vector_sha256",
            "boundary_token_id",
        }
    )
    for path in PATHS:
        item_path = f"$.frozen_path_references.{path}"
        ref = _mapping(refs[path], item_path, keys, "INVALID_REFERENCE")
        tokens = _tokens(ref["generated_token_ids"], f"{item_path}.generated_token_ids")
        expected = EXPECTED_REFERENCES[path]
        if ref["path"] != path or ref["source_experiment_ids"] != expected["source_experiment_ids"]:
            _fail("INVALID_REFERENCE", item_path, repr(ref))
        if ref["token_count"] != TOKEN_COUNT or ref["token_count"] != len(tokens):
            _fail("INVALID_REFERENCE", f"{item_path}.token_count", repr(ref["token_count"]))
        digest = token_ids_sha256(tokens)
        if ref["token_ids_sha256"] != digest:
            _fail("INVALID_TOKEN_DIGEST", f"{item_path}.token_ids_sha256", repr(ref["token_ids_sha256"]))
        for key in (
            "token_ids_sha256",
            "output_sha256",
            "score_trace_sha256",
            "raw_logit_trace_sha256",
            "comparison_score_vector_sha256",
            "comparison_raw_logit_vector_sha256",
            "boundary_token_id",
        ):
            _require_exact(ref[key], expected[key], "FROZEN_REFERENCE_MISMATCH", f"{item_path}.{key}")
        _require_exact(ref["comparison_step_index"], TARGET_STEP_INDEX, "INVALID_REFERENCE", f"{item_path}.comparison_step_index")
        if tokens[TARGET_STEP_INDEX] != expected["boundary_token_id"]:
            _fail("FROZEN_REFERENCE_MISMATCH", f"{item_path}.generated_token_ids[{TARGET_STEP_INDEX}]", repr(tokens[TARGET_STEP_INDEX]))
        result[path] = ref
    return result


def _validate_runs(
    value: object,
    refs: Mapping[str, Mapping[str, Any]],
    manifest_locks: Mapping[str, Mapping[str, str]],
) -> list[Mapping[str, Any]]:
    items = _list(value, "$.runs")
    if len(items) != len(RUN_PLAN):
        _fail("INVALID_RUN_PLAN", "$.runs", repr(len(items)))
    runs: list[Mapping[str, Any]] = []
    for index, (item, plan) in enumerate(zip(items, RUN_PLAN)):
        run_path = f"$.runs[{index}]"
        run = _mapping(item, run_path, _RUN_KEYS, "INVALID_RUN_SCHEMA")
        run_id, path, repeat, order = plan
        for key, expected in {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order,
            "fresh_load": True,
            "base_load_dtype": "bfloat16" if path == BF16_PATH else "float32",
        }.items():
            _require_exact(run[key], expected, "INVALID_RUN_PLAN", f"{run_path}.{key}")
        tokens = _tokens(run["generated_token_ids"], f"{run_path}.generated_token_ids")
        _require_exact(run["token_count"], TOKEN_COUNT, "INVALID_RUN_TOKEN_COUNT", f"{run_path}.token_count")
        if len(tokens) != TOKEN_COUNT:
            _fail("INVALID_RUN_TOKEN_COUNT", f"{run_path}.generated_token_ids", repr(len(tokens)))
        digest = token_ids_sha256(tokens)
        if run["token_ids_sha256"] != digest:
            _fail("INVALID_TOKEN_DIGEST", f"{run_path}.token_ids_sha256", repr(run["token_ids_sha256"]))
        if tokens != refs[path]["generated_token_ids"]:
            _fail("FROZEN_REFERENCE_MISMATCH", f"{run_path}.generated_token_ids", "run/reference mismatch")
        for key in ("token_ids_sha256", "output_sha256"):
            _require_exact(run[key], refs[path][key], "FROZEN_REFERENCE_MISMATCH", f"{run_path}.{key}")
        _validate_precision(run["precision_audit"], path, f"{run_path}.precision_audit")
        _validate_generation_trace(
            run["generation_trace"],
            path,
            refs[path],
            manifest_locks[path],
            f"{run_path}.generation_trace",
        )
        _validate_target_alignment(run["target_alignment"], path, refs[path], f"{run_path}.target_alignment")
        for key in ("target_alignment_passed", "lm_head_raw_logit_linked", "path_protocol_passed"):
            _require_exact(run[key], True, "INVALID_RUN_CLAIM", f"{run_path}.{key}")
        _positive_finite(run["elapsed_seconds"], f"{run_path}.elapsed_seconds", "INVALID_RUN_RESOURCE")
        peak = _positive_int(run["peak_gpu_memory_bytes"], f"{run_path}.peak_gpu_memory_bytes", "INVALID_RUN_RESOURCE")
        before = _nonnegative_int(run["memory_allocated_before_load_bytes"], f"{run_path}.memory_allocated_before_load_bytes", "INVALID_RUN_RESOURCE")
        after = _nonnegative_int(run["memory_allocated_after_release_bytes"], f"{run_path}.memory_allocated_after_release_bytes", "INVALID_RUN_RESOURCE")
        if before > MAX_RESIDUAL_CUDA_BYTES or after > MAX_RESIDUAL_CUDA_BYTES or peak < max(before, after):
            _fail("INVALID_RUN_RESOURCE", run_path, repr({"peak": peak, "before": before, "after": after}))
        runs.append(run)
    return runs


def _validate_precision(value: object, path_name: str, path: str) -> None:
    expected = _expected_precision(path_name)
    _require_exact(value, expected, "INVALID_PRECISION_AUDIT", path)


def _validate_generation_trace(
    value: object,
    path_name: str,
    ref: Mapping[str, Any],
    manifest_locks: Mapping[str, str],
    path: str,
) -> None:
    trace = _mapping(
        value,
        path,
        frozenset({"step_count", "vocabulary_size", "cache_returned", "scores", "raw_logits", "lm_head_output"}),
        "INVALID_GENERATION_TRACE",
    )
    for key, expected in {"step_count": TOKEN_COUNT, "vocabulary_size": VOCABULARY_SIZE, "cache_returned": True}.items():
        _require_exact(trace[key], expected, "INVALID_GENERATION_TRACE", f"{path}.{key}")
    for kind, ref_key, vector_key in (
        ("scores", "score_trace_sha256", "comparison_score_vector_sha256"),
        ("raw_logits", "raw_logit_trace_sha256", "comparison_raw_logit_vector_sha256"),
    ):
        vector_path = f"{path}.{kind}"
        vector = _mapping(trace[kind], vector_path, _TRACE_VECTOR_KEYS, "INVALID_GENERATION_TRACE")
        _require_exact(vector["native_dtypes"], ["float32"], "INVALID_GENERATION_TRACE", f"{vector_path}.native_dtypes")
        _require_exact(vector["shape_per_step"], [1, VOCABULARY_SIZE], "INVALID_GENERATION_TRACE", f"{vector_path}.shape_per_step")
        _require_exact(vector["comparison_dtype"], "float32", "INVALID_GENERATION_TRACE", f"{vector_path}.comparison_dtype")
        _require_exact(vector["all_finite"], True, "INVALID_GENERATION_TRACE", f"{vector_path}.all_finite")
        _require_exact(vector["comparison_step_index"], TARGET_STEP_INDEX, "INVALID_GENERATION_TRACE", f"{vector_path}.comparison_step_index")
        _sha256(vector["trace_sha256"], f"{vector_path}.trace_sha256")
        _require_exact(vector["trace_sha256"], ref[ref_key], "FROZEN_REFERENCE_MISMATCH", f"{vector_path}.trace_sha256")
        manifest = _list(vector["comparison_vector_sha256_per_step"], f"{vector_path}.comparison_vector_sha256_per_step")
        if len(manifest) != TOKEN_COUNT:
            _fail("INVALID_STEP_MANIFEST", f"{vector_path}.comparison_vector_sha256_per_step", repr(len(manifest)))
        for index, digest in enumerate(manifest):
            _sha256(digest, f"{vector_path}.comparison_vector_sha256_per_step[{index}]")
        _require_exact(vector["comparison_step_vector_sha256"], manifest[TARGET_STEP_INDEX], "STEP_VECTOR_LINK_MISMATCH", f"{vector_path}.comparison_step_vector_sha256")
        _require_exact(vector["comparison_step_vector_sha256"], ref[vector_key], "FROZEN_REFERENCE_MISMATCH", f"{vector_path}.comparison_step_vector_sha256")
        _require_exact(
            _manifest_sha256(manifest),
            manifest_locks[kind],
            "STEP_MANIFEST_LOCK_MISMATCH",
            f"{vector_path}.comparison_vector_sha256_per_step",
        )
    lm = _mapping(
        trace["lm_head_output"],
        f"{path}.lm_head_output",
        frozenset({"native_dtype", "shape", "comparison_dtype", "all_finite", "comparison_step_index", "comparison_vector_sha256"}),
        "INVALID_GENERATION_TRACE",
    )
    expected_dtype = "bfloat16" if path_name == BF16_PATH else "float32"
    expected_lm = {
        "native_dtype": expected_dtype,
        "shape": [1, 1, VOCABULARY_SIZE],
        "comparison_dtype": "float32",
        "all_finite": True,
        "comparison_step_index": TARGET_STEP_INDEX,
        "comparison_vector_sha256": trace["raw_logits"]["comparison_step_vector_sha256"],
    }
    _require_exact(lm, expected_lm, "LM_HEAD_RAW_LOGIT_LINK_MISMATCH", f"{path}.lm_head_output")


def _validate_target_alignment(value: object, path_name: str, ref: Mapping[str, Any], path: str) -> None:
    expected_dtype = "bfloat16" if path_name == BF16_PATH else "float32"
    expected = {
        "call_index": TARGET_STEP_INDEX,
        "generation_step_index": TARGET_STEP_INDEX,
        "input_token_ids": [788],
        "input_shape": [1, 1],
        "cache_position": [383],
        "position_ids": [383],
        "past_length": 383,
        "causal_forward_calls": TOKEN_COUNT,
        "lm_head_output_shape": [1, 1, VOCABULARY_SIZE],
        "lm_head_output_native_dtype": expected_dtype,
        "lm_head_output_comparison_vector_sha256": ref["comparison_raw_logit_vector_sha256"],
        "generated_raw_logit_comparison_vector_sha256": ref["comparison_raw_logit_vector_sha256"],
    }
    _require_exact(value, expected, "INVALID_TARGET_ALIGNMENT", path)


def _recompute_repeat_stability(by_path: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATHS:
        first, second = by_path[path]
        stable = analyze_path_repeat_stability(
            first["generated_token_ids"],
            second["generated_token_ids"],
            first_output_sha256=first["output_sha256"],
            second_output_sha256=second["output_sha256"],
            first_score_trace_sha256=first["generation_trace"]["scores"]["trace_sha256"],
            second_score_trace_sha256=second["generation_trace"]["scores"]["trace_sha256"],
            first_raw_logit_trace_sha256=first["generation_trace"]["raw_logits"]["trace_sha256"],
            second_raw_logit_trace_sha256=second["generation_trace"]["raw_logits"]["trace_sha256"],
            first_score_vector_sha256=first["generation_trace"]["scores"]["comparison_step_vector_sha256"],
            second_score_vector_sha256=second["generation_trace"]["scores"]["comparison_step_vector_sha256"],
            first_raw_logit_vector_sha256=first["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"],
            second_raw_logit_vector_sha256=second["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"],
            precision_audits_identical=first["precision_audit"] == second["precision_audit"],
        )
        stable["target_alignment_identity"] = first["target_alignment"] == second["target_alignment"]
        stable["lm_head_output_identity"] = (
            first["target_alignment"]["lm_head_output_comparison_vector_sha256"]
            == second["target_alignment"]["lm_head_output_comparison_vector_sha256"]
        )
        stable["passed"] = all(value for key, value in stable.items() if key != "passed")
        result[path] = stable
    result["passed"] = all(result[path]["passed"] for path in PATHS)
    return result


def _validate_repeat_manifest_identity(
    by_path: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    """Bind every JSON-visible step digest across the two repeats per dtype."""

    for path in PATHS:
        first, second = by_path[path]
        for kind in ("scores", "raw_logits"):
            left = first["generation_trace"][kind][
                "comparison_vector_sha256_per_step"
            ]
            right = second["generation_trace"][kind][
                "comparison_vector_sha256_per_step"
            ]
            if left != right:
                _fail(
                    "STEP_MANIFEST_REPEAT_MISMATCH",
                    f"$.runs[{second['order_index']}].generation_trace.{kind}."
                    "comparison_vector_sha256_per_step",
                    path,
                )


def _recompute_path_reproduction(
    by_path: Mapping[str, list[Mapping[str, Any]]], refs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATHS:
        ref = refs[path]
        runs = by_path[path]
        item = {
            "token_identity": all(run["generated_token_ids"] == ref["generated_token_ids"] and run["token_count"] == ref["token_count"] and run["token_ids_sha256"] == ref["token_ids_sha256"] for run in runs),
            "output_identity": all(run["output_sha256"] == ref["output_sha256"] for run in runs),
            "score_trace_identity": all(run["generation_trace"]["scores"]["trace_sha256"] == ref["score_trace_sha256"] for run in runs),
            "raw_logit_trace_identity": all(run["generation_trace"]["raw_logits"]["trace_sha256"] == ref["raw_logit_trace_sha256"] for run in runs),
            "comparison_score_vector_identity": all(run["generation_trace"]["scores"]["comparison_step_vector_sha256"] == ref["comparison_score_vector_sha256"] for run in runs),
            "comparison_raw_logit_vector_identity": all(run["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"] == ref["comparison_raw_logit_vector_sha256"] for run in runs),
            "boundary_token_identity": all(run["generated_token_ids"][TARGET_STEP_INDEX] == ref["boundary_token_id"] for run in runs),
        }
        item["passed"] = all(item.values())
        result[path] = item
    result["passed"] = all(result[path]["passed"] for path in PATHS)
    return result


def _validate_step_evidence(
    value: object, runs: Mapping[str, Mapping[str, Any]], *, kind: str
) -> dict[str, int]:
    path = "$.selection_score_evidence" if kind == "scores" else "$.raw_logit_evidence"
    record = _mapping(
        value,
        path,
        frozenset({"step_index", "comparison_basis", "shared_generated_prefix_tokens_before_step", "source", "semantics", "comparison_dtype", "top_k", "paths", "delta"}),
        "INVALID_STEP_EVIDENCE",
    )
    is_score = kind == "scores"
    expected_header = {
        "step_index": TARGET_STEP_INDEX,
        "comparison_basis": "frozen_first_cross_dtype_generated_token_divergence",
        "shared_generated_prefix_tokens_before_step": TARGET_STEP_INDEX,
        "source": "generated.scores" if is_score else "generated.logits",
        "semantics": "processed_prediction_scores_after_logits_processors" if is_score else "unprocessed_lm_head_prediction_scores",
        "comparison_dtype": "float32",
        "top_k": TOP_K,
    }
    for key, expected in expected_header.items():
        _require_exact(record[key], expected, "INVALID_STEP_EVIDENCE", f"{path}.{key}")
    paths = _mapping(record["paths"], f"{path}.paths", frozenset(PATHS), "INVALID_STEP_EVIDENCE")
    top_ids: dict[str, int] = {}
    compared_values: dict[str, dict[int, float]] = {}
    vector_digests: dict[str, str] = {}
    value_key = "score" if is_score else "raw_logit"
    top_values_key = "top_scores" if is_score else "top_raw_logits"
    emitted_value_key = "emitted_token_score" if is_score else "emitted_token_raw_logit"
    path_keys = frozenset({"top_token_ids", "top_token_texts", top_values_key, "top_margin", "emitted_token_id", "emitted_token_text", emitted_value_key, "compared_tokens", "decision_contrast_true_minus_false", "comparison_vector_sha256"})
    for path_name in PATHS:
        item_path = f"{path}.paths.{path_name}"
        item = _mapping(paths[path_name], item_path, path_keys, "INVALID_TOP_K_EVIDENCE")
        ids = _list(item["top_token_ids"], f"{item_path}.top_token_ids")
        texts = _list(item["top_token_texts"], f"{item_path}.top_token_texts")
        values = _list(item[top_values_key], f"{item_path}.{top_values_key}")
        if len(ids) != TOP_K or len(set(ids)) != TOP_K or len(texts) != TOP_K or len(values) != TOP_K:
            _fail("INVALID_TOP_K_EVIDENCE", item_path, "top-k cardinality or uniqueness")
        for index, token_id in enumerate(ids):
            _token_id(token_id, f"{item_path}.top_token_ids[{index}]")
            if not isinstance(texts[index], str):
                _fail("INVALID_TOP_K_EVIDENCE", f"{item_path}.top_token_texts[{index}]", repr(texts[index]))
            _known_text(token_id, texts[index], f"{item_path}.top_token_texts[{index}]")
        numeric_values = [_finite_number(item_value, f"{item_path}.{top_values_key}[{index}]", "INVALID_TOP_K_EVIDENCE") for index, item_value in enumerate(values)]
        if any(left < right for left, right in zip(numeric_values, numeric_values[1:])) or numeric_values[0] <= numeric_values[1]:
            _fail("INVALID_TOP_K_EVIDENCE", f"{item_path}.{top_values_key}", repr(numeric_values))
        expected_emitted = BF16_TOKEN_ID if path_name == BF16_PATH else FP32_TOKEN_ID
        _require_exact(item["emitted_token_id"], expected_emitted, "INVALID_TOP_K_EVIDENCE", f"{item_path}.emitted_token_id")
        _require_exact(ids[0], expected_emitted, "INVALID_TOP_K_EVIDENCE", f"{item_path}.top_token_ids[0]")
        _require_exact(item["emitted_token_text"], "true" if path_name == BF16_PATH else "false", "INVALID_TOP_K_EVIDENCE", f"{item_path}.emitted_token_text")
        emitted_value = _finite_number(item[emitted_value_key], f"{item_path}.{emitted_value_key}", "INVALID_TOP_K_EVIDENCE")
        _float_exact(emitted_value, numeric_values[0], "INVALID_TOP_K_EVIDENCE", f"{item_path}.{emitted_value_key}")
        margin = _finite_number(item["top_margin"], f"{item_path}.top_margin", "INVALID_TOP_K_EVIDENCE")
        _float_exact(margin, numeric_values[0] - numeric_values[1], "INVALID_TOP_K_EVIDENCE", f"{item_path}.top_margin")
        compared = _list(item["compared_tokens"], f"{item_path}.compared_tokens")
        if len(compared) != 2:
            _fail("INVALID_TOP_K_EVIDENCE", f"{item_path}.compared_tokens", repr(len(compared)))
        values_by_id: dict[int, float] = {}
        for index, expected_id in enumerate((BF16_TOKEN_ID, FP32_TOKEN_ID)):
            compared_path = f"{item_path}.compared_tokens[{index}]"
            compared_item = _mapping(compared[index], compared_path, frozenset({"token_id", "token_text", value_key, "rank"}), "INVALID_TOP_K_EVIDENCE")
            _require_exact(compared_item["token_id"], expected_id, "INVALID_TOP_K_EVIDENCE", f"{compared_path}.token_id")
            _require_exact(compared_item["token_text"], "true" if expected_id == BF16_TOKEN_ID else "false", "INVALID_TOP_K_EVIDENCE", f"{compared_path}.token_text")
            token_value = _finite_number(compared_item[value_key], f"{compared_path}.{value_key}", "INVALID_TOP_K_EVIDENCE")
            rank = _positive_int(compared_item["rank"], f"{compared_path}.rank", "INVALID_TOP_K_EVIDENCE")
            if rank > VOCABULARY_SIZE:
                _fail("INVALID_TOP_K_EVIDENCE", f"{compared_path}.rank", repr(rank))
            if expected_id in ids:
                top_index = ids.index(expected_id)
                _require_exact(rank, top_index + 1, "INVALID_TOP_K_EVIDENCE", f"{compared_path}.rank")
                _float_exact(token_value, numeric_values[top_index], "INVALID_TOP_K_EVIDENCE", f"{compared_path}.{value_key}")
            elif rank <= TOP_K:
                _fail("INVALID_TOP_K_EVIDENCE", f"{compared_path}.rank", repr(rank))
            values_by_id[expected_id] = token_value
        contrast = _finite_number(item["decision_contrast_true_minus_false"], f"{item_path}.decision_contrast_true_minus_false", "INVALID_TOP_K_EVIDENCE")
        _float_exact(contrast, values_by_id[BF16_TOKEN_ID] - values_by_id[FP32_TOKEN_ID], "INVALID_TOP_K_EVIDENCE", f"{item_path}.decision_contrast_true_minus_false")
        if (path_name == BF16_PATH and contrast <= 0) or (path_name == FP32_PATH and contrast >= 0):
            _fail("INVALID_TOP_K_EVIDENCE", f"{item_path}.decision_contrast_true_minus_false", repr(contrast))
        digest = _sha256(item["comparison_vector_sha256"], f"{item_path}.comparison_vector_sha256")
        _require_exact(digest, runs[path_name]["generation_trace"][kind]["comparison_step_vector_sha256"], "STEP_VECTOR_LINK_MISMATCH", f"{item_path}.comparison_vector_sha256")
        top_ids[path_name] = ids[0]
        compared_values[path_name] = values_by_id
        vector_digests[path_name] = digest
    _validate_delta(record["delta"], compared_values, vector_digests, f"{path}.delta")
    return top_ids


def _validate_delta(
    value: object,
    observed: Mapping[str, Mapping[int, float]],
    digests: Mapping[str, str],
    path: str,
) -> None:
    item = _mapping(value, path, frozenset({"vocabulary_elements", "nonzero_elements", "max_abs_delta", "mean_abs_delta", "root_mean_square_delta"}), "INVALID_DELTA_SUMMARY")
    _require_exact(item["vocabulary_elements"], VOCABULARY_SIZE, "INVALID_DELTA_SUMMARY", f"{path}.vocabulary_elements")
    nonzero = _nonnegative_int(item["nonzero_elements"], f"{path}.nonzero_elements", "INVALID_DELTA_SUMMARY")
    if nonzero > VOCABULARY_SIZE:
        _fail("INVALID_DELTA_SUMMARY", f"{path}.nonzero_elements", repr(nonzero))
    maximum = _nonnegative_finite(item["max_abs_delta"], f"{path}.max_abs_delta", "INVALID_DELTA_SUMMARY")
    mean = _nonnegative_finite(item["mean_abs_delta"], f"{path}.mean_abs_delta", "INVALID_DELTA_SUMMARY")
    rms = _nonnegative_finite(item["root_mean_square_delta"], f"{path}.root_mean_square_delta", "INVALID_DELTA_SUMMARY")
    identical = digests[BF16_PATH] == digests[FP32_PATH]
    if identical:
        if nonzero != 0 or any(metric != 0.0 for metric in (maximum, mean, rms)):
            _fail("INVALID_DELTA_SUMMARY", path, "identical vectors require zero summary")
        return
    if nonzero <= 0 or maximum <= 0 or mean <= 0 or rms <= 0:
        _fail("INVALID_DELTA_SUMMARY", path, "different vectors require positive summary")
    tolerance = 1e-12 * max(1.0, maximum, mean, rms)
    if mean > rms + tolerance or rms > maximum + tolerance:
        _fail("INVALID_DELTA_SUMMARY", path, "requires mean <= RMS <= max")
    density = nonzero / VOCABULARY_SIZE
    if mean > maximum * density + tolerance or rms > maximum * math.sqrt(density) + tolerance:
        _fail("INVALID_DELTA_SUMMARY", path, "summary exceeds nonzero-density bound")
    if rms * rms > maximum * mean + tolerance:
        _fail("INVALID_DELTA_SUMMARY", path, "RMS^2 exceeds max*mean")
    observed_deltas = [
        abs(observed[BF16_PATH][token_id] - observed[FP32_PATH][token_id])
        for token_id in (BF16_TOKEN_ID, FP32_TOKEN_ID)
    ]
    if max(observed_deltas) > maximum + tolerance:
        _fail("INVALID_DELTA_SUMMARY", path, "max is below an exposed token delta")
    if sum(delta > 0 for delta in observed_deltas) > nonzero:
        _fail("INVALID_DELTA_SUMMARY", path, "nonzero count is below exposed deltas")


def _recompute_cross_dtype_identity(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    left = runs[BF16_PATH]
    right = runs[FP32_PATH]
    return {
        "token_identity": left["generated_token_ids"] == right["generated_token_ids"],
        "output_identity": left["output_sha256"] == right["output_sha256"],
        "score_trace_identity": left["generation_trace"]["scores"]["trace_sha256"] == right["generation_trace"]["scores"]["trace_sha256"],
        "raw_logit_trace_identity": left["generation_trace"]["raw_logits"]["trace_sha256"] == right["generation_trace"]["raw_logits"]["trace_sha256"],
        "comparison_score_vector_identity": left["generation_trace"]["scores"]["comparison_step_vector_sha256"] == right["generation_trace"]["scores"]["comparison_step_vector_sha256"],
        "comparison_raw_logit_vector_identity": left["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"] == right["generation_trace"]["raw_logits"]["comparison_step_vector_sha256"],
    }


def _validate_derived_claims(
    evidence: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    repeat: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    classification: str,
) -> None:
    gate = {
        "bf16_attached_repeat_stable": repeat[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": repeat[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": reproduction[FP32_PATH]["passed"],
        "first_token_boundary_reproduced": True,
        "target_forward_aligned": all(run["target_alignment_passed"] for run in runs),
        "lm_head_raw_logit_linked": all(run["lm_head_raw_logit_linked"] for run in runs),
        "processed_score_argmax_matches_emitted_token": True,
        "raw_logits_captured": all(run["generation_trace"]["raw_logits"]["all_finite"] is True for run in runs),
        "attached_dtype_effect_classified": classification in _ALLOWED_CLASSIFICATIONS,
    }
    gate["passed"] = all(gate.values())
    _require_exact(evidence["dtype_isolation_gate"], gate, "GATE_CLAIM_MISMATCH", "$.dtype_isolation_gate")
    acceptance = {
        "upstream_evidence_locked": True,
        "frozen_input_reproduced": True,
        "attached_execution_form_fixed": True,
        "base_dtype_only_treatment": True,
        "bf16_attached_repeat_stable": repeat[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": repeat[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": reproduction[FP32_PATH]["passed"],
        "first_token_boundary_reproduced": True,
        "target_forward_aligned": all(run["target_alignment_passed"] for run in runs),
        "full_generation_traces_captured": all(run["generation_trace"]["step_count"] == run["token_count"] and run["generation_trace"]["scores"]["all_finite"] is True and run["generation_trace"]["raw_logits"]["all_finite"] is True for run in runs),
        "lm_head_raw_logit_linked": all(run["lm_head_raw_logit_linked"] for run in runs),
        "generation_score_alignment_verified": True,
        "path_protocols_executed": all(run["path_protocol_passed"] for run in runs),
        "source_storage_dtypes_locked": True,
        "fresh_load_memory_isolated": all(run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES and run["memory_allocated_after_release_bytes"] <= MAX_RESIDUAL_CUDA_BYTES for run in runs),
        "source_adapter_unchanged": True,
        "source_model_unchanged": True,
        "eval_digest_unchanged": True,
        "prompt_digest_unchanged": True,
    }
    _require_exact(evidence["acceptance"], acceptance, "ACCEPTANCE_CLAIM_MISMATCH", "$.acceptance")
    if not gate["passed"] or not all(acceptance.values()):
        _fail("FROZEN_GATE_NOT_PASSED", "$", "derived gate or acceptance failed")


def _validate_resources(evidence: Mapping[str, Any], runs: list[Mapping[str, Any]]) -> None:
    elapsed = _positive_finite(evidence["elapsed_seconds"], "$.elapsed_seconds", "FROZEN_RESOURCE_CLAIM_MISMATCH")
    peak = _positive_int(evidence["peak_gpu_memory_bytes"], "$.peak_gpu_memory_bytes", "FROZEN_RESOURCE_CLAIM_MISMATCH")
    run_elapsed = math.fsum(float(run["elapsed_seconds"]) for run in runs)
    run_peak = max(int(run["peak_gpu_memory_bytes"]) for run in runs)
    if elapsed + 1e-12 < run_elapsed or peak != run_peak:
        _fail("FROZEN_RESOURCE_CLAIM_MISMATCH", "$", repr({"elapsed": elapsed, "sum": run_elapsed, "peak": peak, "run_peak": run_peak}))


def _validate_frozen_policy(evidence: Mapping[str, Any]) -> None:
    constraints = _expected_constraints()
    for key, expected in {
        "causal_scope": _expected_causal_scope(),
        "remediation_gate": {"new_remediation_tested": False, "passed": False},
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _expected_next_action(constraints),
        "runtime_eligible": False,
        "runtime_eligibility_reason": evidence["classification"],
        "offline": True,
    }.items():
        _require_exact(evidence[key], expected, "INVALID_FROZEN_POLICY", f"$.{key}")


def _expected_storage_audit() -> dict[str, Any]:
    return {
        "base_checkpoint": {"tensors": 338, "elements": 1_543_714_304, "dtype_tensors": {"bfloat16": 338}, "dtype_elements": {"bfloat16": 1_543_714_304}},
        "adapter": {"tensors": 224, "elements": 4_358_144, "dtype_tensors": {"float32": 224}, "dtype_elements": {"float32": 4_358_144}},
    }


def _expected_protocol() -> dict[str, Any]:
    plan = [{"run_id": run_id, "path": path, "repeat": repeat, "order_index": order} for run_id, path, repeat, order in RUN_PLAN]
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": plan,
        "run_order_design": "ABBA",
        "fresh_loads_per_path": {BF16_PATH: 2, FP32_PATH: 2},
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "target_forward": {"generation_step_index": 45, "input_generated_token_index": 44, "input_token_id": 788, "predicted_token_ids": {BF16_PATH: BF16_TOKEN_ID, FP32_PATH: FP32_TOKEN_ID}, "past_length": 383, "cache_position": [383], "causal_forward_calls": 48},
        "treatment": {"isolated_variable": "attached_path_base_and_inference_dtype", "bf16_condition": "bfloat16", "fp32_condition": "float32", "controlled_adapter_runtime_dtype": "float32", "attached_execution_form_fixed": True},
        "paths": {
            BF16_PATH: {"checkpoint_storage_dtype": "bfloat16", "base_load_dtype": "bfloat16", "adapter_storage_dtype": "float32", "adapter_runtime_dtype": "float32", "autocast_adapter_dtype": True, "merge": False, "inference_parameter_dtypes": ["bfloat16", "float32"]},
            FP32_PATH: {"checkpoint_storage_dtype": "bfloat16", "base_load_dtype": "float32", "adapter_storage_dtype": "float32", "adapter_runtime_dtype": "float32", "autocast_adapter_dtype": True, "merge": False, "inference_parameter_dtypes": ["float32"]},
        },
        "generation": {"attn_implementation": "sdpa", "attention_class": "Qwen2Attention", "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']", "low_level_cuda_kernel_identity_claimed": False, "output_attentions": False, "do_sample": False, "max_new_tokens": 256, "use_cache": True, "repetition_penalty": 1.1, "model_eos_token_ids": [151645, 151643], "model_pad_token_id": 151643, "call_pad_token_id": 151645, "return_dict_in_generate": True, "output_scores": True, "output_logits": True, "generate_return_dtype_semantics": "return_tensor_dtype_not_internal_compute_dtype", "tf32": False, "autocast": False, "device": "cuda:0"},
        "sdp_kernel_flags": {"flash_sdp_enabled": True, "math_sdp_enabled": True, "mem_efficient_sdp_enabled": True, "cudnn_sdp_enabled": True, "fp16_bf16_reduction_math_sdp_allowed": False},
    }


def _expected_precision(path: str) -> dict[str, Any]:
    base_dtype = "bfloat16" if path == BF16_PATH else "float32"
    return {
        "base_parameters": {"floating_tensors": 338, "floating_elements": 1_543_714_304, "dtypes": {base_dtype: 1_543_714_304}, "devices": {"cuda:0": 1_543_714_304}},
        "adapter_parameters": {"floating_tensors": 224, "floating_elements": 4_358_144, "dtypes": {"float32": 4_358_144}, "devices": {"cuda:0": 4_358_144}},
        "floating_buffers": {"floating_tensors": 1, "floating_elements": 64, "dtypes": {"float32": 64}, "devices": {"cuda:0": 64}},
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
        "lora_dropout": {"modules": 112, "training_modules": 0},
        "generation": {"score_dtypes": ["float32"], "all_scores_float32": True, "raw_logit_dtypes": ["float32"], "all_raw_logits_float32": True, "dtype_semantics": "transformers_generate_return_tensor_dtype", "autocast_enabled": False, "training": False},
    }


def _expected_causal_scope() -> dict[str, Any]:
    return {
        "isolated_variable": "attached_path_base_and_inference_dtype_bfloat16_vs_float32",
        "controlled": ["same_bfloat16_checkpoint_source_values", "same_fp32_adapter_source_and_runtime_values", "same_attached_factorized_lora_execution_form", "same_eval_001_rendered_input_and_generation_prefix", "same_greedy_decoding_and_high_level_sdpa_dispatch", "same_fresh_model_load_lifecycle"],
        "supports": "classification of the repeat-stable total dtype effect on the frozen attached generation path at token boundary 45",
        "does_not_support": ["all_bf16_versus_all_fp32_path_claim", "pristine_fp32_checkpoint_comparison", "earliest_temporal_or_module_root_cause", "low_level_cuda_kernel_identity_or_unique_root_cause", "peft_bug_claim", "full_eval_generalization", "merged_artifact_promotion", "runtime_eligibility"],
    }


def _expected_constraints() -> dict[str, bool]:
    return {"attached_execution_form_change": False, "adapter_runtime_dtype_change": False, "source_checkpoint_values_change": False, "target_step_change": False, "locked_path_backend_change": False, "locked_path_decoding_change": False, "new_data": False, "training": False, "eval_answer_tuning": False, "runtime_integration": False, "full_eval_run": False, "merged_artifact_save": False, "merged_artifact_promotion": False, "module_tensor_sidecar": False}


def _expected_next_action(constraints: dict[str, bool]) -> dict[str, Any]:
    return {
        "gate_id": "FC-MVP-001-attached-dtype-numerics-v1",
        "action": "on the same frozen target forward, locate the first registered BF16-versus-FP32 attached module-output difference and quantify its propagation without changing execution form or claiming a unique low-level root cause",
        "acceptance": {"fresh_attached_paths_reproduced": True, "target_forward_reproduced": True, "first_registered_module_difference_located": True, "dtype_effect_quantified": True, "source_inputs_unchanged": True},
        "constraints": constraints,
    }


def _mapping(value: object, path: str, keys: frozenset[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        _fail(code, path, repr(value))
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("INVALID_LIST", path, repr(value))
    return value


def _tokens(value: object, path: str) -> list[int]:
    items = _list(value, path)
    for index, item in enumerate(items):
        _token_id(item, f"{path}[{index}]")
    return items


def _token_id(value: object, path: str) -> int:
    result = _nonnegative_int(value, path, "INVALID_TOKEN_ID")
    if result >= VOCABULARY_SIZE:
        _fail("INVALID_TOKEN_ID", path, repr(value))
    return result


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:") or any(character not in "0123456789abcdef" for character in value[7:]):
        _fail("INVALID_SHA256", path, repr(value))
    return value


def _positive_int(value: object, path: str, code: str) -> int:
    result = _nonnegative_int(value, path, code)
    if result <= 0:
        _fail(code, path, repr(value))
    return result


def _nonnegative_int(value: object, path: str, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code, path, repr(value))
    return value


def _finite_number(value: object, path: str, code: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        _fail(code, path, repr(value))
    return float(value)


def _positive_finite(value: object, path: str, code: str) -> float:
    result = _finite_number(value, path, code)
    if result <= 0:
        _fail(code, path, repr(value))
    return result


def _nonnegative_finite(value: object, path: str, code: str) -> float:
    result = _finite_number(value, path, code)
    if result < 0:
        _fail(code, path, repr(value))
    return result


def _float_exact(actual: float, expected: float, code: str, path: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        _fail(code, path, f"expected {expected!r}, got {actual!r}")


def _known_text(token_id: object, text: object, path: str) -> None:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        return
    expected = {BF16_TOKEN_ID: "true", FP32_TOKEN_ID: "false"}.get(token_id)
    if expected is not None and text != expected:
        _fail("INVALID_TOKEN_TEXT", path, repr(text))


def _finite_tree(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("NONFINITE_JSON_NUMBER", path, repr(value))
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _require_exact(actual: object, expected: object, code: str, path: str) -> None:
    if not _strict_equal(actual, expected):
        _fail(code, path, f"expected {expected!r}, got {actual!r}")


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise ToolRouterValidationError(code, path, detail)


__all__ = ["validate_attached_dtype_isolation_evidence"]
