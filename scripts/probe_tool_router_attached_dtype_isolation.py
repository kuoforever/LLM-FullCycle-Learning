"""Isolate BF16 versus FP32 dtype on the frozen attached-Adapter path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import accelerate  # type: ignore[import-not-found,import-untyped]
import peft  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import transformers  # type: ignore[import-not-found,import-untyped]
from huggingface_hub import (  # type: ignore[import-not-found]
    __version__ as hub_version,
)
from peft import PeftModel  # type: ignore[import-not-found]
from peft.tuners.tuners_utils import (  # type: ignore[import-not-found]
    BaseTunerLayer,
)
from safetensors import safe_open  # type: ignore[import-not-found]
from safetensors import __version__ as safetensors_version  # type: ignore[attr-defined,import-not-found]
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found,import-untyped]
from transformers import (  # type: ignore[import-not-found]
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_attached_dtype_isolation import (  # noqa: E402
    ATTACHED_DTYPE_ISOLATION_VERSION,
    analyze_attached_dtype_tokens,
    analyze_path_repeat_stability,
    classify_attached_dtype_effect,
    select_locked_comparison_step,
)
from fullcycle_bridge.tool_router_merge_remediation import (  # noqa: E402
    token_ids_sha256,
)
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)

BF16_PATH = "bf16_attached_adapter"
FP32_PATH = "fp32_attached_adapter"
PATH_ORDER = (BF16_PATH, FP32_PATH)
RUN_PLAN = (
    (BF16_PATH, 1, "bf16-attached-dtype-r1"),
    (FP32_PATH, 1, "fp32-attached-dtype-r1"),
    (FP32_PATH, 2, "fp32-attached-dtype-r2"),
    (BF16_PATH, 2, "bf16-attached-dtype-r2"),
)
TARGET_STEP_INDEX = 45
TARGET_INPUT_TOKEN_ID = 788
TARGET_CACHE_POSITION = 383
TARGET_FORWARD_CALLS = 48
BF16_BOUNDARY_TOKEN_ID = 1866
FP32_BOUNDARY_TOKEN_ID = 3849
TOP_K = 5
MAX_RESIDUAL_CUDA_BYTES = 16 * 1024 * 1024
EXPECTED_LORA_TARGETS = 112
EXPECTED_LORA_PARAMETER_TENSORS = 224
EXPECTED_BASE_ELEMENTS = 1_543_714_304
EXPECTED_ADAPTER_ELEMENTS = 4_358_144
EXPECTED_BUFFER_ELEMENTS = 64
EXPECTED_INPUT_TOKEN_COUNT = 339
EXPECTED_INPUT_TOKEN_SHA256 = (
    "sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f"
)
EXPECTED_CONFIG_SHA256 = (
    "sha256:5a038ea786526f188c796a6e5eea4c4d3aa47fc66977dc4f6ff16f52999236d8"
)
EXPECTED_TRAINING_LOCK_SHA256 = (
    "sha256:e6e23f51834b1815578368ce54c78034e72a7158395892e77fdf75594548931f"
)
EXPECTED_TRAINING_EVIDENCE_SHA256 = (
    "sha256:641b1a7ef3dc0de0d9f2124b9122cb2c4be46b42de9265d558ab6f5b25b41a30"
)
EXPECTED_STABILITY_SHA256 = (
    "sha256:82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc"
)
EXPECTED_DRIFT_SHA256 = (
    "sha256:ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3"
)
EXPECTED_ISOLATION_SHA256 = (
    "sha256:37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26"
)
EXPECTED_FP32_NUMERICS_SHA256 = (
    "sha256:cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91"
)
EXPECTED_REFERENCES = {
    BF16_PATH: {
        "token_count": 48,
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
        "boundary_token_id": BF16_BOUNDARY_TOKEN_ID,
    },
    FP32_PATH: {
        "token_count": 48,
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
        "boundary_token_id": FP32_BOUNDARY_TOKEN_ID,
    },
}
ALLOWED_CLASSIFICATIONS = frozenset(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--stability-evidence", type=Path, required=True)
    parser.add_argument("--drift-evidence", type=Path, required=True)
    parser.add_argument("--isolation-evidence", type=Path, required=True)
    parser.add_argument("--fp32-numerics-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _verify_output_boundary(args.output)

    config = _load_json(args.config)
    training = _load_json(args.training_evidence)
    stability = _load_json(args.stability_evidence)
    drift = _load_json(args.drift_evidence)
    isolation = _load_json(args.isolation_evidence)
    fp32_numerics = _load_json(args.fp32_numerics_evidence)
    source_lineage = _verify_sources(
        config,
        training,
        stability,
        drift,
        isolation,
        fp32_numerics,
        args,
    )
    environment = _verify_environment(config)
    frozen_references = _frozen_path_references(
        stability,
        drift,
        isolation,
        fp32_numerics,
    )

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("evaluation digest mismatch")
    record = evaluation[0]
    if record["example_id"] != "eval-001":
        raise RuntimeError("locked dtype probe requires eval-001 first")
    prompt_path = ROOT / config["prompt"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": render_user_payload(record)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded_cpu = tokenizer(rendered, return_tensors="pt")
    input_token_ids = [int(value) for value in encoded_cpu["input_ids"][0]]
    input_digest = token_ids_sha256(input_token_ids)
    if (
        len(input_token_ids) != EXPECTED_INPUT_TOKEN_COUNT
        or input_digest != EXPECTED_INPUT_TOKEN_SHA256
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")

    storage_audit = {
        "base_checkpoint": _safetensor_storage_audit(
            args.model_dir / config["model"]["weight_file"]
        ),
        "adapter": _safetensor_storage_audit(
            args.adapter_dir / "adapter_model.safetensors"
        ),
    }
    source_storage_dtypes_locked = storage_audit == _expected_storage_audit()
    if not source_storage_dtypes_locked:
        raise RuntimeError(f"source storage dtype drift: {storage_audit!r}")

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    target_vectors: dict[str, dict[str, Any]] = {}
    for order_index, (path, repeat, run_id) in enumerate(RUN_PLAN):
        run, vectors = _run_attached_path(
            path=path,
            repeat=repeat,
            run_id=run_id,
            order_index=order_index,
            model_dir=args.model_dir,
            adapter_dir=args.adapter_dir,
            config=config,
            encoded_cpu=encoded_cpu,
            tokenizer=tokenizer,
        )
        runs.append(run)
        target_vectors[run_id] = vectors

    by_path = {
        path: [run for run in runs if run["path"] == path]
        for path in PATH_ORDER
    }
    if any(len(path_runs) != 2 for path_runs in by_path.values()):
        raise RuntimeError("fixed ABBA attached dtype run plan was not executed")

    path_repeat_stability: dict[str, Any] = {}
    for path in PATH_ORDER:
        first, second = by_path[path]
        stability_result = analyze_path_repeat_stability(
            first["generated_token_ids"],
            second["generated_token_ids"],
            first_output_sha256=first["output_sha256"],
            second_output_sha256=second["output_sha256"],
            first_score_trace_sha256=first["generation_trace"]["scores"][
                "trace_sha256"
            ],
            second_score_trace_sha256=second["generation_trace"]["scores"][
                "trace_sha256"
            ],
            first_raw_logit_trace_sha256=first["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            second_raw_logit_trace_sha256=second["generation_trace"]["raw_logits"][
                "trace_sha256"
            ],
            first_score_vector_sha256=first["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ],
            second_score_vector_sha256=second["generation_trace"]["scores"][
                "comparison_step_vector_sha256"
            ],
            first_raw_logit_vector_sha256=first["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ],
            second_raw_logit_vector_sha256=second["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ],
            precision_audits_identical=(
                first["precision_audit"] == second["precision_audit"]
            ),
        )
        stability_result["target_alignment_identity"] = (
            first["target_alignment"] == second["target_alignment"]
        )
        stability_result["lm_head_output_identity"] = (
            first["target_alignment"][
                "lm_head_output_comparison_vector_sha256"
            ]
            == second["target_alignment"][
                "lm_head_output_comparison_vector_sha256"
            ]
        )
        stability_result["passed"] = all(
            value
            for key, value in stability_result.items()
            if key != "passed"
        )
        path_repeat_stability[path] = stability_result
    path_repeat_stability["passed"] = all(
        path_repeat_stability[path]["passed"] for path in PATH_ORDER
    )
    if not path_repeat_stability["passed"]:
        raise RuntimeError(
            f"attached dtype repeat stability failed: {path_repeat_stability!r}"
        )

    path_reproduction: dict[str, Any] = {
        path: _path_reproduction(by_path[path], frozen_references[path])
        for path in PATH_ORDER
    }
    path_reproduction["passed"] = all(
        path_reproduction[path]["passed"] for path in PATH_ORDER
    )
    if not path_reproduction["passed"]:
        raise RuntimeError(
            f"attached dtype frozen reference reproduction failed: "
            f"{path_reproduction!r}"
        )

    representatives = {path: by_path[path][0] for path in PATH_ORDER}
    token_analysis = analyze_attached_dtype_tokens(
        representatives[BF16_PATH]["generated_token_ids"],
        representatives[FP32_PATH]["generated_token_ids"],
    )
    comparison_step = select_locked_comparison_step(
        token_analysis,
        frozen_boundary_index=TARGET_STEP_INDEX,
    )
    step_index = comparison_step["step_index"]
    compared_token_ids = [BF16_BOUNDARY_TOKEN_ID, FP32_BOUNDARY_TOKEN_ID]
    representative_vectors = {
        path: target_vectors[representatives[path]["run_id"]]
        for path in PATH_ORDER
    }
    selection_score_evidence = _step_evidence(
        source="generated.scores",
        semantics="processed_prediction_scores_after_logits_processors",
        value_key="score",
        vectors={path: representative_vectors[path]["score"] for path in PATH_ORDER},
        runs=representatives,
        step_index=step_index,
        comparison_basis=comparison_step["basis"],
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    raw_logit_evidence = _step_evidence(
        source="generated.logits",
        semantics="unprocessed_lm_head_prediction_scores",
        value_key="raw_logit",
        vectors={path: representative_vectors[path]["raw_logit"] for path in PATH_ORDER},
        runs=representatives,
        step_index=step_index,
        comparison_basis=comparison_step["basis"],
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    cross_dtype_trace_identity = {
        "token_identity": token_analysis["cross_dtype_identical"],
        "output_identity": (
            representatives[BF16_PATH]["output_sha256"]
            == representatives[FP32_PATH]["output_sha256"]
        ),
        "score_trace_identity": (
            representatives[BF16_PATH]["generation_trace"]["scores"][
                "trace_sha256"
            ]
            == representatives[FP32_PATH]["generation_trace"]["scores"][
                "trace_sha256"
            ]
        ),
        "raw_logit_trace_identity": (
            representatives[BF16_PATH]["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
            == representatives[FP32_PATH]["generation_trace"]["raw_logits"][
                "trace_sha256"
            ]
        ),
        "comparison_score_vector_identity": (
            selection_score_evidence["paths"][BF16_PATH][
                "comparison_vector_sha256"
            ]
            == selection_score_evidence["paths"][FP32_PATH][
                "comparison_vector_sha256"
            ]
        ),
        "comparison_raw_logit_vector_identity": (
            raw_logit_evidence["paths"][BF16_PATH][
                "comparison_vector_sha256"
            ]
            == raw_logit_evidence["paths"][FP32_PATH][
                "comparison_vector_sha256"
            ]
        ),
    }

    classification = classify_attached_dtype_effect(
        token_analysis,
        bf16_repeat_stable=path_repeat_stability[BF16_PATH]["passed"],
        fp32_repeat_stable=path_repeat_stability[FP32_PATH]["passed"],
        bf16_reference_reproduced=path_reproduction[BF16_PATH]["passed"],
        fp32_reference_reproduced=path_reproduction[FP32_PATH]["passed"],
        bf16_emitted_token_id=representatives[BF16_PATH]["generated_token_ids"][
            step_index
        ],
        fp32_emitted_token_id=representatives[FP32_PATH]["generated_token_ids"][
            step_index
        ],
        bf16_score_top_token_id=selection_score_evidence["paths"][BF16_PATH][
            "top_token_ids"
        ][0],
        fp32_score_top_token_id=selection_score_evidence["paths"][FP32_PATH][
            "top_token_ids"
        ][0],
        bf16_raw_logit_top_token_id=raw_logit_evidence["paths"][BF16_PATH][
            "top_token_ids"
        ][0],
        fp32_raw_logit_top_token_id=raw_logit_evidence["paths"][FP32_PATH][
            "top_token_ids"
        ][0],
    )
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError(f"unexpected dtype classification: {classification}")

    target_forward_aligned = all(run["target_alignment_passed"] for run in runs)
    lm_head_raw_logit_linked = all(run["lm_head_raw_logit_linked"] for run in runs)
    score_argmax_aligned = all(
        selection_score_evidence["paths"][path]["top_token_ids"][0]
        == representatives[path]["generated_token_ids"][step_index]
        for path in PATH_ORDER
    )
    raw_logits_captured = all(
        run["generation_trace"]["raw_logits"]["all_finite"] is True
        for run in runs
    )
    full_generation_traces_captured = all(
        run["generation_trace"]["step_count"] == run["token_count"]
        and run["generation_trace"]["scores"]["all_finite"] is True
        and run["generation_trace"]["raw_logits"]["all_finite"] is True
        for run in runs
    )
    first_token_boundary_reproduced = (
        token_analysis["first_divergent_token_index"] == TARGET_STEP_INDEX
        and token_analysis["bf16_token_id"] == BF16_BOUNDARY_TOKEN_ID
        and token_analysis["fp32_token_id"] == FP32_BOUNDARY_TOKEN_ID
    )
    path_protocols_executed = all(run["path_protocol_passed"] for run in runs)
    fresh_load_memory_isolated = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"]
        <= MAX_RESIDUAL_CUDA_BYTES
        for run in runs
    )
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    model_weight_path = args.model_dir / config["model"]["weight_file"]
    source_adapter_unchanged = adapter_manifest == training["final_adapter"]["files"]
    source_model_unchanged = (
        file_sha256(model_weight_path)
        == f"sha256:{config['model']['weight_sha256']}"
    )
    eval_digest_unchanged = fixture_digest(evaluation) == config["data"]["eval_digest"]
    prompt_digest_unchanged = (
        file_sha256(prompt_path) == f"sha256:{config['prompt']['sha256']}"
    )
    dtype_isolation_gate = {
        "bf16_attached_repeat_stable": path_repeat_stability[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": path_repeat_stability[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": path_reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": path_reproduction[FP32_PATH]["passed"],
        "first_token_boundary_reproduced": first_token_boundary_reproduced,
        "target_forward_aligned": target_forward_aligned,
        "lm_head_raw_logit_linked": lm_head_raw_logit_linked,
        "processed_score_argmax_matches_emitted_token": score_argmax_aligned,
        "raw_logits_captured": raw_logits_captured,
        "attached_dtype_effect_classified": classification in ALLOWED_CLASSIFICATIONS,
    }
    dtype_isolation_gate["passed"] = all(dtype_isolation_gate.values())
    acceptance = {
        "upstream_evidence_locked": True,
        "frozen_input_reproduced": True,
        "attached_execution_form_fixed": True,
        "base_dtype_only_treatment": True,
        "bf16_attached_repeat_stable": path_repeat_stability[BF16_PATH]["passed"],
        "fp32_attached_repeat_stable": path_repeat_stability[FP32_PATH]["passed"],
        "bf16_frozen_reference_reproduced": path_reproduction[BF16_PATH]["passed"],
        "fp32_frozen_reference_reproduced": path_reproduction[FP32_PATH]["passed"],
        "first_token_boundary_reproduced": first_token_boundary_reproduced,
        "target_forward_aligned": target_forward_aligned,
        "full_generation_traces_captured": full_generation_traces_captured,
        "lm_head_raw_logit_linked": lm_head_raw_logit_linked,
        "generation_score_alignment_verified": score_argmax_aligned,
        "path_protocols_executed": path_protocols_executed,
        "source_storage_dtypes_locked": source_storage_dtypes_locked,
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": source_model_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": prompt_digest_unchanged,
    }
    if not dtype_isolation_gate["passed"] or not all(acceptance.values()):
        raise RuntimeError(
            "attached dtype isolation evidence invalid: "
            f"gate={dtype_isolation_gate!r}, acceptance={acceptance!r}"
        )

    constraints = _constraints()
    result = {
        "attached_dtype_isolation_version": ATTACHED_DTYPE_ISOLATION_VERSION,
        "experiment_id": "fc-mvp-001-attached-dtype-isolation-v1",
        "source_experiment_id": config["experiment_id"],
        "source_lineage": source_lineage,
        "training_lock_sha256": file_sha256(ROOT / "requirements" / "training.lock"),
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "model_weight_sha256": file_sha256(model_weight_path),
        "prompt_sha256": file_sha256(prompt_path),
        "eval_digest": config["data"]["eval_digest"],
        "example_id": record["example_id"],
        "input_token_count": len(input_token_ids),
        "input_token_ids_sha256": input_digest,
        "storage_audit": storage_audit,
        "environment": environment,
        "protocol": _protocol(config, tokenizer),
        "frozen_path_references": frozen_references,
        "runs": runs,
        "path_repeat_stability": path_repeat_stability,
        "path_reproduction": path_reproduction,
        "cross_dtype_token_analysis": {
            **token_analysis,
            "bf16_token_text": _optional_decode(
                tokenizer, token_analysis["bf16_token_id"]
            ),
            "fp32_token_text": _optional_decode(
                tokenizer, token_analysis["fp32_token_id"]
            ),
        },
        "comparison_step": comparison_step,
        "selection_score_evidence": selection_score_evidence,
        "raw_logit_evidence": raw_logit_evidence,
        "cross_dtype_trace_identity": cross_dtype_trace_identity,
        "classification": classification,
        "causal_scope": _causal_scope(),
        "dtype_isolation_gate": dtype_isolation_gate,
        "remediation_gate": {
            "new_remediation_tested": False,
            "passed": False,
        },
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(run["peak_gpu_memory_bytes"] for run in runs),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _locked_next_action(constraints),
        "runtime_eligible": False,
        "runtime_eligibility_reason": classification,
        "offline": True,
    }
    _require_finite_json(result, "$")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _run_attached_path(
    *,
    path: str,
    repeat: int,
    run_id: str,
    order_index: int,
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    encoded_cpu: Any,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} load exceeded residual CUDA ceiling: "
            f"{allocated_before} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model: Any | None = None
    run: dict[str, Any] | None = None
    vectors: dict[str, Any] | None = None
    try:
        model, precision = _load_attached_model(
            model_dir,
            adapter_dir,
            config,
            path=path,
        )
        (
            token_ids,
            generation_trace,
            vectors,
            target_alignment,
            lm_head_linked,
        ) = _generate_with_target_evidence(
            model,
            encoded_cpu,
            config,
            tokenizer,
        )
        precision["generation"] = {
            "score_dtypes": generation_trace["scores"]["native_dtypes"],
            "all_scores_float32": (
                generation_trace["scores"]["native_dtypes"] == ["float32"]
            ),
            "raw_logit_dtypes": generation_trace["raw_logits"]["native_dtypes"],
            "all_raw_logits_float32": (
                generation_trace["raw_logits"]["native_dtypes"] == ["float32"]
            ),
            "dtype_semantics": "transformers_generate_return_tensor_dtype",
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        path_protocol_passed = _path_protocol_passed(precision, path=path)
        target_alignment_passed = _target_alignment_passed(
            target_alignment,
            path=path,
        )
        if not path_protocol_passed or not target_alignment_passed:
            raise RuntimeError(
                f"{run_id} protocol failed: precision={precision!r}, "
                f"target={target_alignment!r}"
            )
        if not lm_head_linked:
            raise RuntimeError(f"{run_id} LM-head output does not match raw logits")
        torch.cuda.synchronize()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        run = {
            "run_id": run_id,
            "path": path,
            "repeat": repeat,
            "order_index": order_index,
            "fresh_load": True,
            "base_load_dtype": _dtype_name_for_path(path),
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": _sha256(decoded.encode("utf-8")),
            "precision_audit": precision,
            "generation_trace": generation_trace,
            "target_alignment": target_alignment,
            "target_alignment_passed": target_alignment_passed,
            "lm_head_raw_logit_linked": lm_head_linked,
            "path_protocol_passed": path_protocol_passed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "memory_allocated_before_load_bytes": allocated_before,
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {run_id} retained model-scale CUDA memory: "
            f"{allocated_after} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    if run is None or vectors is None:
        raise RuntimeError(f"{run_id} did not produce evidence")
    run["memory_allocated_after_release_bytes"] = allocated_after
    return run, vectors


def _load_attached_model(
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    *,
    path: str,
) -> tuple[Any, dict[str, Any]]:
    seed = config["generation"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if not getattr(model_config, "use_sliding_window", False):
        model_config.sliding_window = None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=model_config,
        local_files_only=True,
        torch_dtype=_torch_dtype_for_path(path),
        attn_implementation=config["generation"]["attn_implementation"],
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).to("cuda")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    _verify_generation_semantics(model, config)
    precision = _precision_audit(model)
    precision["lora_dropout"] = _lora_dropout_audit(model)
    return model, precision


def _generate_with_target_evidence(
    model: Any,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[int], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    if torch.is_autocast_enabled():
        raise RuntimeError("generation must not run under autocast")
    causal = _causal_model(model)
    lm_head = causal.get_output_embeddings()
    state: dict[str, Any] = {
        "call_index": -1,
        "active": False,
        "target_count": 0,
        "alignment": {},
        "lm_head_vector": None,
        "lm_head_shape": None,
        "lm_head_dtype": None,
    }

    def causal_pre_hook(_module: Any, hook_args: Any, kwargs: Any) -> None:
        state["call_index"] += 1
        state["active"] = state["call_index"] == TARGET_STEP_INDEX
        if state["active"]:
            state["target_count"] += 1
            state["alignment"] = _target_alignment(
                hook_args,
                kwargs,
                state["call_index"],
            )

    def causal_post_hook(
        _module: Any,
        _args: Any,
        _kwargs: Any,
        _output: Any,
    ) -> None:
        if state["active"]:
            state["active"] = False

    def lm_head_hook(_module: Any, _args: Any, output: Any) -> None:
        if not state["active"]:
            return
        tensor = _first_tensor(output)
        native = tensor.detach().contiguous().cpu()
        if native.ndim < 2 or native.shape[-1] != int(model.config.vocab_size):
            raise RuntimeError(f"unexpected LM-head output shape: {list(native.shape)!r}")
        vector = native.reshape(-1, native.shape[-1])[-1].float().clone()
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError("non-finite target LM-head output")
        if state["lm_head_vector"] is not None:
            raise RuntimeError("target LM-head output captured more than once")
        state["lm_head_vector"] = vector
        state["lm_head_shape"] = list(native.shape)
        state["lm_head_dtype"] = str(native.dtype).removeprefix("torch.")

    handles = [
        causal.register_forward_pre_hook(causal_pre_hook, with_kwargs=True),
        causal.register_forward_hook(causal_post_hook, with_kwargs=True),
        lm_head.register_forward_hook(lm_head_hook),
    ]
    encoded = {key: value.to("cuda") for key, value in encoded_cpu.items()}
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["generation"]["max_new_tokens"],
                use_cache=config["generation"]["use_cache"],
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    new_tokens = generated.sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.cpu()]
    if generated.scores is None or generated.logits is None:
        raise RuntimeError("generate did not return scores and raw logits")
    if len(generated.scores) != len(token_ids) or len(generated.logits) != len(
        token_ids
    ):
        raise RuntimeError("generated trace count does not match token count")
    expected_shape = [1, int(model.config.vocab_size)]
    score_summary, target_score = _summarize_trace(
        generated.scores,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    raw_summary, target_raw_logit = _summarize_trace(
        generated.logits,
        expected_shape=expected_shape,
        target_step=TARGET_STEP_INDEX,
    )
    cache_returned = generated.past_key_values is not None
    if not cache_returned:
        raise RuntimeError("cached generation did not return past_key_values")
    lm_head_vector = state["lm_head_vector"]
    if not torch.is_tensor(lm_head_vector):
        raise RuntimeError("target LM-head output was not captured")
    lm_head_sha256 = _float32_vector_sha256(lm_head_vector)
    raw_logit_sha256 = _float32_vector_sha256(target_raw_logit)
    lm_head_linked = torch.equal(lm_head_vector, target_raw_logit)
    alignment = dict(state["alignment"])
    alignment.update(
        {
            "causal_forward_calls": state["call_index"] + 1,
            "lm_head_output_shape": state["lm_head_shape"],
            "lm_head_output_native_dtype": state["lm_head_dtype"],
            "lm_head_output_comparison_vector_sha256": lm_head_sha256,
            "generated_raw_logit_comparison_vector_sha256": raw_logit_sha256,
        }
    )
    if state["target_count"] != 1:
        raise RuntimeError(f"target forward count drift: {state['target_count']!r}")
    generation_trace = {
        "step_count": len(token_ids),
        "vocabulary_size": expected_shape[1],
        "cache_returned": cache_returned,
        "scores": score_summary,
        "raw_logits": raw_summary,
        "lm_head_output": {
            "native_dtype": state["lm_head_dtype"],
            "shape": state["lm_head_shape"],
            "comparison_dtype": "float32",
            "all_finite": True,
            "comparison_step_index": TARGET_STEP_INDEX,
            "comparison_vector_sha256": lm_head_sha256,
        },
    }
    return (
        token_ids,
        generation_trace,
        {"score": target_score, "raw_logit": target_raw_logit},
        alignment,
        lm_head_linked,
    )


def _summarize_trace(
    tensors: Iterable[Any],
    *,
    expected_shape: list[int],
    target_step: int,
) -> tuple[dict[str, Any], Any]:
    digest = hashlib.sha256()
    dtypes: set[str] = set()
    comparison_vector_sha256_per_step: list[str] = []
    target_vector: Any | None = None
    steps = list(tensors)
    for index, tensor in enumerate(steps):
        shape = list(tensor.shape)
        if shape != expected_shape:
            raise RuntimeError(
                f"unexpected generation vector shape at {index}: {shape!r}"
            )
        native_cpu = tensor.detach().contiguous().cpu()
        dtype = str(native_cpu.dtype).removeprefix("torch.")
        dtypes.add(dtype)
        if not bool(torch.isfinite(native_cpu).all()):
            raise RuntimeError(f"non-finite generation vector at step {index}")
        header = json.dumps(
            {"index": index, "dtype": dtype, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        payload = native_cpu.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        comparison_vector = native_cpu[0].float()
        comparison_vector_sha256_per_step.append(
            _float32_vector_sha256(comparison_vector)
        )
        if index == target_step:
            target_vector = comparison_vector.clone()
    if target_vector is None:
        raise RuntimeError(f"target generation step missing: {target_step}")
    return (
        {
            "native_dtypes": sorted(dtypes),
            "shape_per_step": expected_shape,
            "comparison_dtype": "float32",
            "all_finite": True,
            "trace_sha256": "sha256:" + digest.hexdigest(),
            "comparison_vector_sha256_per_step": (
                comparison_vector_sha256_per_step
            ),
            "comparison_step_index": target_step,
            "comparison_step_vector_sha256": _float32_vector_sha256(
                target_vector
            ),
        },
        target_vector,
    )


def _target_alignment(args: Any, kwargs: Any, call_index: int) -> dict[str, Any]:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args and torch.is_tensor(args[0]):
        input_ids = args[0]
    cache_position = kwargs.get("cache_position")
    position_ids = kwargs.get("position_ids")
    past = kwargs.get("past_key_values")
    if input_ids is None or not torch.is_tensor(input_ids):
        raise RuntimeError("target forward input_ids missing")
    if cache_position is None or not torch.is_tensor(cache_position):
        raise RuntimeError("target forward cache_position missing")
    if position_ids is None or not torch.is_tensor(position_ids):
        raise RuntimeError("target forward position_ids missing")
    if past is None or not hasattr(past, "get_seq_length"):
        raise RuntimeError("target forward cache object missing")
    return {
        "call_index": call_index,
        "generation_step_index": TARGET_STEP_INDEX,
        "input_token_ids": [int(value) for value in input_ids.detach().cpu().view(-1)],
        "input_shape": list(input_ids.shape),
        "cache_position": [
            int(value) for value in cache_position.detach().cpu().view(-1)
        ],
        "position_ids": [
            int(value) for value in position_ids.detach().cpu().view(-1)
        ],
        "past_length": int(past.get_seq_length()),
    }


def _target_alignment_passed(
    value: dict[str, Any],
    *,
    path: str,
) -> bool:
    return (
        value.get("call_index") == TARGET_STEP_INDEX
        and value.get("generation_step_index") == TARGET_STEP_INDEX
        and value.get("input_token_ids") == [TARGET_INPUT_TOKEN_ID]
        and value.get("input_shape") == [1, 1]
        and value.get("cache_position") == [TARGET_CACHE_POSITION]
        and value.get("position_ids") == [TARGET_CACHE_POSITION]
        and value.get("past_length") == TARGET_CACHE_POSITION
        and value.get("causal_forward_calls") == TARGET_FORWARD_CALLS
        and value.get("lm_head_output_shape") == [1, 1, 151936]
        and value.get("lm_head_output_native_dtype") == _dtype_name_for_path(path)
        and value.get("lm_head_output_comparison_vector_sha256")
        == value.get("generated_raw_logit_comparison_vector_sha256")
    )


def _precision_audit(model: Any) -> dict[str, Any]:
    base_parameters: list[tuple[str, Any]] = []
    adapter_parameters: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        destination = adapter_parameters if ".lora_" in name else base_parameters
        destination.append((name, parameter))
    causal = _causal_model(model)
    return {
        "base_parameters": _tensor_inventory(base_parameters),
        "adapter_parameters": _tensor_inventory(adapter_parameters),
        "floating_buffers": _tensor_inventory(model.named_buffers()),
        "lora_target_modules": sum(
            isinstance(module, BaseTunerLayer) for module in model.modules()
        ),
        "lora_parameter_tensors": len(adapter_parameters),
        "adapter_parameters_finite": all(
            bool(torch.isfinite(parameter).all())
            for _, parameter in adapter_parameters
            if parameter.is_floating_point()
        ),
        "active_adapters": list(model.active_adapters),
        "is_peft_model": isinstance(model, PeftModel),
        "input_output_embeddings_tied": (
            causal.get_input_embeddings().weight
            is causal.get_output_embeddings().weight
        ),
        "attn_implementation": causal.config._attn_implementation,
        "attention_class": causal.model.layers[0].self_attn.__class__.__name__,
        "output_attentions": causal.config.output_attentions,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }


def _path_protocol_passed(value: dict[str, Any], *, path: str) -> bool:
    generation = value.get("generation")
    expected_base_dtype = _dtype_name_for_path(path)
    return (
        _inventory_is_dtype_cuda(
            value.get("base_parameters"),
            dtype=expected_base_dtype,
            expected_elements=EXPECTED_BASE_ELEMENTS,
        )
        and _inventory_is_dtype_cuda(
            value.get("adapter_parameters"),
            dtype="float32",
            expected_elements=EXPECTED_ADAPTER_ELEMENTS,
        )
        and _inventory_is_dtype_cuda(
            value.get("floating_buffers"),
            dtype="float32",
            expected_elements=EXPECTED_BUFFER_ELEMENTS,
        )
        and value.get("lora_target_modules") == EXPECTED_LORA_TARGETS
        and value.get("lora_parameter_tensors")
        == EXPECTED_LORA_PARAMETER_TENSORS
        and value.get("adapter_parameters_finite") is True
        and value.get("active_adapters") == ["default"]
        and value.get("is_peft_model") is True
        and value.get("input_output_embeddings_tied") is True
        and value.get("attn_implementation") == "sdpa"
        and value.get("attention_class") == "Qwen2Attention"
        and value.get("output_attentions") is False
        and value.get("hf_device_map") is None
        and value.get("lora_dropout")
        == {"modules": EXPECTED_LORA_TARGETS, "training_modules": 0}
        and isinstance(generation, dict)
        and generation.get("score_dtypes") == ["float32"]
        and generation.get("all_scores_float32") is True
        and generation.get("raw_logit_dtypes") == ["float32"]
        and generation.get("all_raw_logits_float32") is True
        and generation.get("dtype_semantics")
        == "transformers_generate_return_tensor_dtype"
        and generation.get("autocast_enabled") is False
        and generation.get("training") is False
    )


def _tensor_inventory(tensors: Iterable[tuple[str, Any]]) -> dict[str, Any]:
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


def _inventory_is_dtype_cuda(
    value: object,
    *,
    dtype: str,
    expected_elements: int,
) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("floating_tensors"), int)
        and value["floating_tensors"] > 0
        and value.get("floating_elements") == expected_elements
        and value.get("dtypes") == {dtype: expected_elements}
        and value.get("devices") == {"cuda:0": expected_elements}
    )


def _lora_dropout_audit(model: Any) -> dict[str, int]:
    dropout_modules: list[Any] = []
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            lora_dropout = getattr(module, "lora_dropout")
            dropout_modules.extend(lora_dropout.values())
    return {
        "modules": len(dropout_modules),
        "training_modules": sum(module.training for module in dropout_modules),
    }


def _verify_generation_semantics(model: Any, config: dict[str, Any]) -> None:
    generation = model.generation_config
    if (
        model.training
        or config["generation"]["attn_implementation"] != "sdpa"
        or config["generation"]["do_sample"] is not False
        or config["generation"]["use_cache"] is not True
        or config["generation"]["max_new_tokens"] != 256
        or generation.repetition_penalty != 1.1
        or generation.eos_token_id != [151645, 151643]
        or generation.pad_token_id != 151643
        or generation.temperature is not None
        or generation.top_p is not None
        or generation.top_k is not None
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("frozen greedy SDPA generation semantics drift")


def _path_reproduction(
    runs: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, bool]:
    result = {
        "token_identity": all(
            run["generated_token_ids"] == reference["generated_token_ids"]
            and run["token_count"] == reference["token_count"]
            and run["token_ids_sha256"] == reference["token_ids_sha256"]
            for run in runs
        ),
        "output_identity": all(
            run["output_sha256"] == reference["output_sha256"] for run in runs
        ),
        "score_trace_identity": all(
            run["generation_trace"]["scores"]["trace_sha256"]
            == reference["score_trace_sha256"]
            for run in runs
        ),
        "raw_logit_trace_identity": all(
            run["generation_trace"]["raw_logits"]["trace_sha256"]
            == reference["raw_logit_trace_sha256"]
            for run in runs
        ),
        "comparison_score_vector_identity": all(
            run["generation_trace"]["scores"]["comparison_step_vector_sha256"]
            == reference["comparison_score_vector_sha256"]
            for run in runs
        ),
        "comparison_raw_logit_vector_identity": all(
            run["generation_trace"]["raw_logits"][
                "comparison_step_vector_sha256"
            ]
            == reference["comparison_raw_logit_vector_sha256"]
            for run in runs
        ),
        "boundary_token_identity": all(
            run["generated_token_ids"][TARGET_STEP_INDEX]
            == reference["boundary_token_id"]
            for run in runs
        ),
    }
    result["passed"] = all(result.values())
    return result


def _step_evidence(
    *,
    source: str,
    semantics: str,
    value_key: str,
    vectors: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    step_index: int,
    comparison_basis: str,
    compared_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for path in PATH_ORDER:
        vector = vectors[path]
        top = torch.topk(vector, k=TOP_K)
        top_ids = [int(value) for value in top.indices]
        top_values = [float(value) for value in top.values]
        if top_values[0] <= top_values[1]:
            raise RuntimeError(f"non-unique top {value_key} for {path}")
        emitted_token_id = runs[path]["generated_token_ids"][step_index]
        paths[path] = {
            "top_token_ids": top_ids,
            "top_token_texts": [
                _decode_token(tokenizer, token_id) for token_id in top_ids
            ],
            f"top_{value_key}s": top_values,
            "top_margin": top_values[0] - top_values[1],
            "emitted_token_id": emitted_token_id,
            "emitted_token_text": _decode_token(tokenizer, emitted_token_id),
            f"emitted_token_{value_key}": float(vector[emitted_token_id]),
            "compared_tokens": [
                {
                    "token_id": token_id,
                    "token_text": _decode_token(tokenizer, token_id),
                    value_key: float(vector[token_id]),
                    "rank": int((vector > vector[token_id]).sum()) + 1,
                }
                for token_id in compared_token_ids
            ],
            "decision_contrast_true_minus_false": float(
                vector[BF16_BOUNDARY_TOKEN_ID]
                - vector[FP32_BOUNDARY_TOKEN_ID]
            ),
            "comparison_vector_sha256": _float32_vector_sha256(vector),
        }
    delta = (vectors[BF16_PATH] - vectors[FP32_PATH]).abs()
    return {
        "step_index": step_index,
        "comparison_basis": comparison_basis,
        "shared_generated_prefix_tokens_before_step": step_index,
        "source": source,
        "semantics": semantics,
        "comparison_dtype": "float32",
        "top_k": TOP_K,
        "paths": paths,
        "delta": {
            "vocabulary_elements": int(delta.numel()),
            "nonzero_elements": int(torch.count_nonzero(delta)),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "root_mean_square_delta": float(torch.sqrt(torch.mean(delta.square()))),
        },
    }


def _frozen_path_references(
    stability: dict[str, Any],
    drift: dict[str, Any],
    isolation: dict[str, Any],
    fp32_numerics: dict[str, Any],
) -> dict[str, Any]:
    stability_bf16 = [
        run for run in stability.get("runs", []) if run.get("path") == "independent"
    ]
    drift_bf16 = [
        run
        for run in drift.get("runs", [])
        if run.get("path") == "independent_bf16_adapter"
    ]
    isolation_fp32 = [
        run
        for run in isolation.get("runs", [])
        if run.get("path") == FP32_PATH
    ]
    if len(stability_bf16) != 2 or len(drift_bf16) != 1 or len(isolation_fp32) != 2:
        raise RuntimeError("frozen attached path references are incomplete")
    bf16_run = drift_bf16[0]
    fp32_run = isolation_fp32[0]
    bf16 = {
        "path": BF16_PATH,
        "source_experiment_ids": [
            stability["experiment_id"],
            drift["experiment_id"],
        ],
        "generated_token_ids": bf16_run["generated_token_ids"],
        "token_count": bf16_run["token_count"],
        "token_ids_sha256": bf16_run["token_ids_sha256"],
        "output_sha256": bf16_run["output_sha256"],
        "score_trace_sha256": bf16_run["generation_trace"]["scores"][
            "trace_sha256"
        ],
        "raw_logit_trace_sha256": bf16_run["generation_trace"]["raw_logits"][
            "trace_sha256"
        ],
        "comparison_step_index": TARGET_STEP_INDEX,
        "comparison_score_vector_sha256": bf16_run["generation_trace"]["scores"][
            "divergent_step_comparison_vector_sha256"
        ],
        "comparison_raw_logit_vector_sha256": bf16_run["generation_trace"][
            "raw_logits"
        ]["divergent_step_comparison_vector_sha256"],
        "boundary_token_id": BF16_BOUNDARY_TOKEN_ID,
    }
    fp32 = {
        "path": FP32_PATH,
        "source_experiment_ids": [
            isolation["experiment_id"],
            fp32_numerics["experiment_id"],
        ],
        "generated_token_ids": fp32_run["generated_token_ids"],
        "token_count": fp32_run["token_count"],
        "token_ids_sha256": fp32_run["token_ids_sha256"],
        "output_sha256": fp32_run["output_sha256"],
        "score_trace_sha256": fp32_run["generation_trace"]["scores"][
            "trace_sha256"
        ],
        "raw_logit_trace_sha256": fp32_run["generation_trace"]["raw_logits"][
            "trace_sha256"
        ],
        "comparison_step_index": TARGET_STEP_INDEX,
        "comparison_score_vector_sha256": fp32_run["generation_trace"]["scores"][
            "comparison_step_vector_sha256"
        ],
        "comparison_raw_logit_vector_sha256": fp32_run["generation_trace"][
            "raw_logits"
        ]["comparison_step_vector_sha256"],
        "boundary_token_id": FP32_BOUNDARY_TOKEN_ID,
    }
    for path, reference in ((BF16_PATH, bf16), (FP32_PATH, fp32)):
        expected = EXPECTED_REFERENCES[path]
        if any(reference[key] != value for key, value in expected.items()):
            raise RuntimeError(f"frozen {path} reference drift: {reference!r}")
    if any(
        run["token_count"] != bf16["token_count"]
        or run["token_ids_sha256"] != bf16["token_ids_sha256"]
        or run["output_sha256"] != bf16["output_sha256"]
        for run in stability_bf16
    ):
        raise RuntimeError("BF16 stability and drift references disagree")
    if any(
        run["generated_token_ids"] != fp32["generated_token_ids"]
        or run["token_ids_sha256"] != fp32["token_ids_sha256"]
        or run["output_sha256"] != fp32["output_sha256"]
        or run["generation_trace"]["scores"]["trace_sha256"]
        != fp32["score_trace_sha256"]
        or run["generation_trace"]["raw_logits"]["trace_sha256"]
        != fp32["raw_logit_trace_sha256"]
        for run in isolation_fp32
    ):
        raise RuntimeError("FP32 isolation attached references are not stable")
    numerics_fp32 = fp32_numerics.get("frozen_path_references", {}).get(FP32_PATH)
    if not isinstance(numerics_fp32, dict) or any(
        numerics_fp32.get(key) != fp32[key]
        for key in (
            "token_count",
            "token_ids_sha256",
            "output_sha256",
            "score_trace_sha256",
            "raw_logit_trace_sha256",
            "comparison_step_index",
            "comparison_score_vector_sha256",
            "comparison_raw_logit_vector_sha256",
        )
    ):
        raise RuntimeError("FP32 numerics attached reference drift")
    return {BF16_PATH: bf16, FP32_PATH: fp32}


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    drift: dict[str, Any],
    isolation: dict[str, Any],
    fp32_numerics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, str]:
    source_lineage = {
        "stability_evidence_sha256": file_sha256(args.stability_evidence),
        "drift_evidence_sha256": file_sha256(args.drift_evidence),
        "isolation_evidence_sha256": file_sha256(args.isolation_evidence),
        "fp32_numerics_evidence_sha256": file_sha256(
            args.fp32_numerics_evidence
        ),
        "training_evidence_sha256": file_sha256(args.training_evidence),
    }
    expected_lineage = {
        "stability_evidence_sha256": EXPECTED_STABILITY_SHA256,
        "drift_evidence_sha256": EXPECTED_DRIFT_SHA256,
        "isolation_evidence_sha256": EXPECTED_ISOLATION_SHA256,
        "fp32_numerics_evidence_sha256": EXPECTED_FP32_NUMERICS_SHA256,
        "training_evidence_sha256": EXPECTED_TRAINING_EVIDENCE_SHA256,
    }
    if source_lineage != expected_lineage:
        raise RuntimeError(f"attached dtype source lineage drift: {source_lineage!r}")
    if canonical_config_sha256(config) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("locked config digest drift")
    if training.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("training evidence config mismatch")
    if (
        stability.get("experiment_id")
        != "fc-mvp-001-bf16-merge-stability-v1"
        or stability.get("classification")
        != "deterministic_bf16_merge_logit_boundary_flip"
        or stability.get("acceptance", {}).get("independent_repeats_identical")
        is not True
        or stability.get("token_analysis", {}).get(
            "first_divergent_token_index"
        )
        != TARGET_STEP_INDEX
    ):
        raise RuntimeError("required BF16 stability source is invalid")
    if (
        drift.get("experiment_id")
        != "fc-mvp-001-fp32-merge-drift-analysis-v1"
        or drift.get("classification")
        != (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "raw_logit_boundary_flip"
        )
        or drift.get("analysis_gate", {}).get("passed") is not True
        or drift.get("stability_evidence_sha256") != EXPECTED_STABILITY_SHA256
    ):
        raise RuntimeError("required BF16 raw-logit drift source is invalid")
    if (
        isolation.get("experiment_id")
        != "fc-mvp-001-fp32-attached-merge-isolation-v1"
        or isolation.get("classification")
        != (
            "deterministic_fp32_attached_vs_merged_"
            "numerical_drift_without_token_drift"
        )
        or isolation.get("isolation_gate", {}).get("passed") is not True
        or isolation.get("drift_evidence_sha256") != EXPECTED_DRIFT_SHA256
        or isolation.get("stability_evidence_sha256") != EXPECTED_STABILITY_SHA256
    ):
        raise RuntimeError("required FP32 attached isolation source is invalid")
    numerics_lineage = fp32_numerics.get("source_lineage")
    if (
        fp32_numerics.get("experiment_id")
        != "fc-mvp-001-fp32-attached-merge-numerics-v1"
        or fp32_numerics.get("classification")
        != (
            "deterministic_fp32_factorized_lora_and_"
            "materialized_linear_execution_form_drift"
        )
        or fp32_numerics.get("numerics_gate", {}).get("passed") is not True
        or fp32_numerics.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-attached-dtype-isolation-v1"
        or not isinstance(numerics_lineage, dict)
        or numerics_lineage.get("isolation_evidence_sha256")
        != EXPECTED_ISOLATION_SHA256
        or numerics_lineage.get("drift_evidence_sha256") != EXPECTED_DRIFT_SHA256
        or numerics_lineage.get("stability_evidence_sha256")
        != EXPECTED_STABILITY_SHA256
    ):
        raise RuntimeError("required FP32 attached numerics source is invalid")
    training_lock_sha256 = file_sha256(ROOT / "requirements" / "training.lock")
    if training_lock_sha256 != EXPECTED_TRAINING_LOCK_SHA256:
        raise RuntimeError("training dependency lock drift")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if adapter_manifest != training.get("final_adapter", {}).get("files"):
        raise RuntimeError("adapter artifact mismatch")
    model_weight_path = args.model_dir / config["model"]["weight_file"]
    if model_weight_path.stat().st_size != config["model"]["weight_bytes"]:
        raise RuntimeError("model weight size mismatch")
    if file_sha256(model_weight_path) != f"sha256:{config['model']['weight_sha256']}":
        raise RuntimeError("model weight digest mismatch")
    prompt_path = ROOT / config["prompt"]["path"]
    if file_sha256(prompt_path) != f"sha256:{config['prompt']['sha256']}":
        raise RuntimeError("prompt digest mismatch")
    return source_lineage


def _verify_environment(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["environment"]
    actual = {
        "python": ".".join(str(item) for item in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": hub_version,
        "safetensors": safetensors_version,
        "tokenizers": tokenizers_version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 0
        ),
        "compute_capability": (
            ".".join(str(item) for item in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else ""
        ),
    }
    if actual != expected:
        raise RuntimeError(
            f"environment mismatch: actual={actual}, expected={expected}"
        )
    return actual


def _protocol(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "run_plan": [
            {
                "run_id": run_id,
                "path": path,
                "repeat": repeat,
                "order_index": order_index,
            }
            for order_index, (path, repeat, run_id) in enumerate(RUN_PLAN)
        ],
        "run_order_design": "ABBA",
        "fresh_loads_per_path": {BF16_PATH: 2, FP32_PATH: 2},
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "target_forward": {
            "generation_step_index": TARGET_STEP_INDEX,
            "input_generated_token_index": TARGET_STEP_INDEX - 1,
            "input_token_id": TARGET_INPUT_TOKEN_ID,
            "predicted_token_ids": {
                BF16_PATH: BF16_BOUNDARY_TOKEN_ID,
                FP32_PATH: FP32_BOUNDARY_TOKEN_ID,
            },
            "past_length": TARGET_CACHE_POSITION,
            "cache_position": [TARGET_CACHE_POSITION],
            "causal_forward_calls": TARGET_FORWARD_CALLS,
        },
        "treatment": {
            "isolated_variable": "attached_path_base_and_inference_dtype",
            "bf16_condition": "bfloat16",
            "fp32_condition": "float32",
            "controlled_adapter_runtime_dtype": "float32",
            "attached_execution_form_fixed": True,
        },
        "paths": {
            BF16_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "bfloat16",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": True,
                "merge": False,
                "inference_parameter_dtypes": ["bfloat16", "float32"],
            },
            FP32_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "autocast_adapter_dtype": True,
                "merge": False,
                "inference_parameter_dtypes": ["float32"],
            },
        },
        "generation": {
            "attn_implementation": config["generation"]["attn_implementation"],
            "attention_class": "Qwen2Attention",
            "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
            "low_level_cuda_kernel_identity_claimed": False,
            "output_attentions": False,
            "do_sample": False,
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "use_cache": config["generation"]["use_cache"],
            "repetition_penalty": 1.1,
            "model_eos_token_ids": [151645, 151643],
            "model_pad_token_id": 151643,
            "call_pad_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_logits": True,
            "generate_return_dtype_semantics": (
                "return_tensor_dtype_not_internal_compute_dtype"
            ),
            "tf32": False,
            "autocast": False,
            "device": "cuda:0",
        },
        "sdp_kernel_flags": _sdp_kernel_flags(),
    }


def _sdp_kernel_flags() -> dict[str, bool]:
    return {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "fp16_bf16_reduction_math_sdp_allowed": (
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        ),
    }


def _causal_scope() -> dict[str, Any]:
    return {
        "isolated_variable": (
            "attached_path_base_and_inference_dtype_bfloat16_vs_float32"
        ),
        "controlled": [
            "same_bfloat16_checkpoint_source_values",
            "same_fp32_adapter_source_and_runtime_values",
            "same_attached_factorized_lora_execution_form",
            "same_eval_001_rendered_input_and_generation_prefix",
            "same_greedy_decoding_and_high_level_sdpa_dispatch",
            "same_fresh_model_load_lifecycle",
        ],
        "supports": (
            "classification of the repeat-stable total dtype effect on the "
            "frozen attached generation path at token boundary 45"
        ),
        "does_not_support": [
            "all_bf16_versus_all_fp32_path_claim",
            "pristine_fp32_checkpoint_comparison",
            "earliest_temporal_or_module_root_cause",
            "low_level_cuda_kernel_identity_or_unique_root_cause",
            "peft_bug_claim",
            "full_eval_generalization",
            "merged_artifact_promotion",
            "runtime_eligibility",
        ],
    }


def _constraints() -> dict[str, bool]:
    return {
        "attached_execution_form_change": False,
        "adapter_runtime_dtype_change": False,
        "source_checkpoint_values_change": False,
        "target_step_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_save": False,
        "merged_artifact_promotion": False,
        "module_tensor_sidecar": False,
    }


def _locked_next_action(constraints: dict[str, bool]) -> dict[str, Any]:
    return {
        "gate_id": "FC-MVP-001-attached-dtype-numerics-v1",
        "action": (
            "on the same frozen target forward, locate the first registered "
            "BF16-versus-FP32 attached module-output difference and quantify "
            "its propagation without changing execution form or claiming a "
            "unique low-level root cause"
        ),
        "acceptance": {
            "fresh_attached_paths_reproduced": True,
            "target_forward_reproduced": True,
            "first_registered_module_difference_located": True,
            "dtype_effect_quantified": True,
            "source_inputs_unchanged": True,
        },
        "constraints": constraints,
    }


def _safetensor_storage_audit(path: Path) -> dict[str, Any]:
    dtype_names = {"BF16": "bfloat16", "F32": "float32"}
    dtype_tensors: dict[str, int] = {}
    dtype_elements: dict[str, int] = {}
    tensor_count = 0
    element_count = 0
    with safe_open(path, framework="pt", device="cpu") as source:
        for key in source.keys():
            metadata = source.get_slice(key)
            raw_dtype = metadata.get_dtype()
            dtype = dtype_names.get(raw_dtype, raw_dtype.lower())
            elements = 1
            for dimension in metadata.get_shape():
                elements *= dimension
            tensor_count += 1
            element_count += elements
            dtype_tensors[dtype] = dtype_tensors.get(dtype, 0) + 1
            dtype_elements[dtype] = dtype_elements.get(dtype, 0) + elements
    return {
        "tensors": tensor_count,
        "elements": element_count,
        "dtype_tensors": dict(sorted(dtype_tensors.items())),
        "dtype_elements": dict(sorted(dtype_elements.items())),
    }


def _expected_storage_audit() -> dict[str, Any]:
    return {
        "base_checkpoint": {
            "tensors": 338,
            "elements": EXPECTED_BASE_ELEMENTS,
            "dtype_tensors": {"bfloat16": 338},
            "dtype_elements": {"bfloat16": EXPECTED_BASE_ELEMENTS},
        },
        "adapter": {
            "tensors": EXPECTED_LORA_PARAMETER_TENSORS,
            "elements": EXPECTED_ADAPTER_ELEMENTS,
            "dtype_tensors": {"float32": EXPECTED_LORA_PARAMETER_TENSORS},
            "dtype_elements": {"float32": EXPECTED_ADAPTER_ELEMENTS},
        },
    }


def _torch_dtype_for_path(path: str) -> Any:
    if path == BF16_PATH:
        return torch.bfloat16
    if path == FP32_PATH:
        return torch.float32
    raise RuntimeError(f"unknown attached dtype path: {path}")


def _dtype_name_for_path(path: str) -> str:
    if path == BF16_PATH:
        return "bfloat16"
    if path == FP32_PATH:
        return "float32"
    raise RuntimeError(f"unknown attached dtype path: {path}")


def _causal_model(model: Any) -> Any:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def _first_tensor(output: Any) -> Any:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _first_tensor(output[0])
    raise RuntimeError("module output does not contain a tensor")


def _float32_vector_sha256(vector: Any) -> str:
    if str(vector.dtype) != "torch.float32" or vector.ndim != 1:
        raise RuntimeError("comparison vector must be one-dimensional float32")
    payload = vector.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return _sha256(payload)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _optional_decode(tokenizer: Any, token_id: object) -> str | None:
    if token_id is None:
        return None
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise RuntimeError(f"invalid optional token ID: {token_id!r}")
    return _decode_token(tokenizer, token_id)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verify_output_boundary(output: Path) -> None:
    resolved = output.resolve()
    candidate_root = (ROOT / "work" / "test-fixtures").resolve()
    if resolved == candidate_root or not resolved.is_relative_to(candidate_root):
        raise RuntimeError("probe output must stay under work/test-fixtures")
    if resolved.exists():
        raise RuntimeError("probe output must be a new candidate artifact")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=resolved.parent,
            prefix=".attached-dtype-probe-",
        ):
            pass
    except OSError as exc:
        raise RuntimeError("probe output directory is not writable") from exc


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    _require_finite_json(value, "$")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise RuntimeError(f"non-finite JSON constant: {value}")


def _require_finite_json(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{path}[{index}]")


if __name__ == "__main__":
    raise SystemExit(main())
