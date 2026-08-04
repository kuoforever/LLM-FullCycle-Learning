"""Locate exact FP32-safe-merge drift against the independent BF16 path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import accelerate  # type: ignore[import-not-found]
import peft  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import transformers  # type: ignore[import-not-found]
from huggingface_hub import (  # type: ignore[import-not-found]
    __version__ as hub_version,
)
from peft import PeftModel  # type: ignore[import-not-found]
from peft.tuners.tuners_utils import (  # type: ignore[import-not-found]
    BaseTunerLayer,
)
from safetensors import safe_open  # type: ignore[import-not-found]
from safetensors import __version__ as safetensors_version  # type: ignore[import-not-found]
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found]
from transformers import AutoTokenizer  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_fp32_merge_drift import (  # noqa: E402
    FP32_MERGE_DRIFT_VERSION,
    analyze_path_tokens,
    classify_generation_boundary,
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
from scripts import (  # noqa: E402
    probe_tool_router_bf16_merge_remediation as remediation_probe,
)
from scripts import (  # noqa: E402
    probe_tool_router_bf16_merge_stability as stability_probe,
)

INDEPENDENT_PATH = "independent_bf16_adapter"
CANDIDATE_PATH = "fp32_safe_merged"
PATH_ORDER = (INDEPENDENT_PATH, CANDIDATE_PATH)
TOP_K = 5
MAX_RESIDUAL_CUDA_BYTES = remediation_probe.MAX_RESIDUAL_CUDA_BYTES
EXPECTED_LORA_TARGETS = remediation_probe.EXPECTED_LORA_TARGETS
EXPECTED_LORA_PARAMETER_TENSORS = (
    remediation_probe.EXPECTED_LORA_PARAMETER_TENSORS
)
EXPECTED_STABILITY_SHA256 = (
    "sha256:82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc"
)
EXPECTED_NUMERICS_SHA256 = (
    "sha256:eb39674127ac93fea2ce6415b3a2fea0d20f6da916b76f1532392533db3e805f"
)
EXPECTED_REMEDIATION_SHA256 = (
    "sha256:7f3c5aff55e69c08a7676d33636a52a5a2bb43f025dae8a2db362041354050b3"
)
ALLOWED_CLASSIFICATIONS = frozenset(
    {
        (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "raw_logit_boundary_flip"
        ),
        (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "logits_processor_boundary_flip"
        ),
        (
            "deterministic_bf16_attached_vs_fp32_merged_"
            "mixed_logit_score_drift"
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
    parser.add_argument("--numerics-evidence", type=Path, required=True)
    parser.add_argument("--remediation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _verify_output_boundary(args.output, args.model_dir, args.adapter_dir)

    config = _load_json(args.config)
    training = _load_json(args.training_evidence)
    stability = _load_json(args.stability_evidence)
    numerics = _load_json(args.numerics_evidence)
    remediation = _load_json(args.remediation_evidence)
    _verify_sources(config, training, stability, numerics, remediation, args)
    _verify_environment(config)

    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    record = evaluation[0]
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
        len(input_token_ids) != stability["input_token_count"]
        or input_digest != stability["input_token_ids_sha256"]
        or len(input_token_ids) != remediation["input_token_count"]
        or input_digest != remediation["input_token_ids_sha256"]
    ):
        raise RuntimeError("frozen eval-001 rendered input drift")

    frozen_references = {
        INDEPENDENT_PATH: _frozen_independent_reference(remediation),
        CANDIDATE_PATH: _frozen_candidate_reference(remediation),
    }
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
    traces: dict[str, dict[str, list[Any]]] = {}
    for path_name in PATH_ORDER:
        run, trace = _run_path_once(
            path_name,
            args.model_dir,
            args.adapter_dir,
            config,
            encoded_cpu,
            tokenizer,
        )
        runs.append(run)
        traces[path_name] = trace

    reproduction = _reproduction_evidence(runs, frozen_references)
    frozen_paths_reproduced = all(reproduction.values())
    if not frozen_paths_reproduced:
        raise RuntimeError(f"frozen path reproduction failed: {reproduction!r}")

    independent_tokens = runs[0]["generated_token_ids"]
    candidate_tokens = runs[1]["generated_token_ids"]
    token_analysis = analyze_path_tokens(independent_tokens, candidate_tokens)
    divergence_index = token_analysis["first_divergent_token_index"]
    if (
        token_analysis["classification"] != "cross_path_token_drift"
        or not isinstance(divergence_index, int)
    ):
        raise RuntimeError(f"expected a comparable token drift: {token_analysis!r}")

    independent_token_id = token_analysis["independent_token_id"]
    candidate_token_id = token_analysis["candidate_token_id"]
    if not isinstance(independent_token_id, int) or not isinstance(
        candidate_token_id, int
    ):
        raise RuntimeError("divergent token IDs are missing")
    compared_token_ids = [independent_token_id, candidate_token_id]
    selection_score_evidence = _step_evidence(
        source="generated.scores",
        semantics="processed_prediction_scores_after_logits_processors",
        value_key="score",
        step_index=divergence_index,
        traces=traces,
        runs=runs,
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    raw_logit_evidence = _step_evidence(
        source="generated.logits",
        semantics="unprocessed_lm_head_prediction_scores",
        value_key="raw_logit",
        step_index=divergence_index,
        traces=traces,
        runs=runs,
        compared_token_ids=compared_token_ids,
        tokenizer=tokenizer,
    )
    classification = classify_generation_boundary(
        token_analysis,
        frozen_paths_reproduced=frozen_paths_reproduced,
        independent_score_top_token_id=selection_score_evidence["paths"][
            INDEPENDENT_PATH
        ]["top_token_ids"][0],
        candidate_score_top_token_id=selection_score_evidence["paths"][
            CANDIDATE_PATH
        ]["top_token_ids"][0],
        independent_raw_logit_top_token_id=raw_logit_evidence["paths"][
            INDEPENDENT_PATH
        ]["top_token_ids"][0],
        candidate_raw_logit_top_token_id=raw_logit_evidence["paths"][
            CANDIDATE_PATH
        ]["top_token_ids"][0],
    )
    if classification == "generation_score_alignment_failure":
        raise RuntimeError("processed generation score does not select emitted token")
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError(f"unexpected drift classification: {classification}")

    _link_divergent_step_hashes(
        runs,
        selection_score_evidence,
        raw_logit_evidence,
    )

    model_weight_path = args.model_dir / config["model"]["weight_file"]
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    source_adapter_unchanged = adapter_manifest == training["final_adapter"]["files"]
    source_model_unchanged = (
        file_sha256(model_weight_path)
        == f"sha256:{config['model']['weight_sha256']}"
    )
    eval_digest_unchanged = (
        fixture_digest(load_fixture(ROOT / config["data"]["eval_path"]))
        == config["data"]["eval_digest"]
    )
    prompt_digest_unchanged = (
        file_sha256(prompt_path) == f"sha256:{config['prompt']['sha256']}"
    )
    path_protocols_executed = all(
        run["path_protocol_passed"] is True for run in runs
    )
    fresh_load_memory_isolated = all(
        run["memory_allocated_before_load_bytes"] <= MAX_RESIDUAL_CUDA_BYTES
        and run["memory_allocated_after_release_bytes"]
        <= MAX_RESIDUAL_CUDA_BYTES
        for run in runs
    )
    score_argmax_aligned = all(
        selection_score_evidence["paths"][run["path"]]["top_token_ids"][0]
        == run["generated_token_ids"][divergence_index]
        for run in runs
    )
    processed_scores_captured = _trace_evidence_linked(
        runs,
        selection_score_evidence,
        trace_key="scores",
    )
    raw_logits_captured = _trace_evidence_linked(
        runs,
        raw_logit_evidence,
        trace_key="raw_logits",
    )
    exact_cached_step_captured = (
        processed_scores_captured
        and raw_logits_captured
        and all(
            run["generation_trace"]["cache_returned"] is True
            and divergence_index < run["generation_trace"]["step_count"]
            for run in runs
        )
    )
    analysis_gate = {
        "frozen_paths_reproduced": frozen_paths_reproduced,
        "first_divergent_token_located": divergence_index >= 0,
        "exact_cached_generate_step_captured": exact_cached_step_captured,
        "processed_score_argmax_matches_generated_token": score_argmax_aligned,
        "raw_logits_captured": raw_logits_captured,
    }
    analysis_gate["passed"] = all(analysis_gate.values())
    candidate_failure_reproduced = (
        reproduction["fp32_candidate_token_identity"]
        and reproduction["fp32_candidate_output_identity"]
        and not token_analysis["cross_path_identical"]
    )
    remediation_passed = (
        reproduction["fp32_candidate_token_identity"]
        and reproduction["fp32_candidate_output_identity"]
        and token_analysis["cross_path_identical"]
    )
    remediation_gate = {
        "candidate_failure_reproduced": candidate_failure_reproduced,
        "independent_bf16_reference_identity": token_analysis[
            "cross_path_identical"
        ],
        "passed": remediation_passed,
    }
    acceptance = {
        "upstream_evidence_locked": True,
        "frozen_input_reproduced": True,
        "independent_bf16_reference_reproduced": (
            reproduction["independent_bf16_token_identity"]
            and reproduction["independent_bf16_output_identity"]
        ),
        "fp32_candidate_reproduced": (
            reproduction["fp32_candidate_token_identity"]
            and reproduction["fp32_candidate_output_identity"]
        ),
        "first_divergent_token_located": divergence_index >= 0,
        "exact_processed_scores_captured": processed_scores_captured,
        "exact_raw_logits_captured": raw_logits_captured,
        "generation_score_alignment_verified": score_argmax_aligned,
        "path_protocols_executed": path_protocols_executed,
        "source_storage_dtypes_locked": source_storage_dtypes_locked,
        "fresh_load_memory_isolated": fresh_load_memory_isolated,
        "source_adapter_unchanged": source_adapter_unchanged,
        "source_model_unchanged": source_model_unchanged,
        "eval_digest_unchanged": eval_digest_unchanged,
        "prompt_digest_unchanged": prompt_digest_unchanged,
    }
    if not analysis_gate["passed"] or not all(acceptance.values()):
        raise RuntimeError(
            "FP32 merge drift evidence invalid: "
            f"analysis={analysis_gate!r}, acceptance={acceptance!r}"
        )

    constraints = _constraints()
    result = {
        "fp32_merge_drift_analysis_version": FP32_MERGE_DRIFT_VERSION,
        "experiment_id": "fc-mvp-001-fp32-merge-drift-analysis-v1",
        "source_experiment_id": config["experiment_id"],
        "remediation_evidence_sha256": file_sha256(args.remediation_evidence),
        "stability_evidence_sha256": file_sha256(args.stability_evidence),
        "numerics_evidence_sha256": file_sha256(args.numerics_evidence),
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
        "protocol": _protocol(config, tokenizer),
        "frozen_references": frozen_references,
        "runs": runs,
        "reproduction": reproduction,
        "token_analysis": {
            **token_analysis,
            "independent_token_text": _decode_token(
                tokenizer, independent_token_id
            ),
            "candidate_token_text": _decode_token(tokenizer, candidate_token_id),
        },
        "selection_score_evidence": selection_score_evidence,
        "raw_logit_evidence": raw_logit_evidence,
        "classification": classification,
        "analysis_gate": analysis_gate,
        "remediation_gate": remediation_gate,
        "acceptance": acceptance,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max(
            run["peak_gpu_memory_bytes"] for run in runs
        ),
        "merged_artifact_saved": False,
        "merged_artifact_allowed": False,
        "constraints": constraints,
        "locked_next_action": _locked_next_action(classification),
        "runtime_eligible": False,
        "runtime_eligibility_reason": classification,
        "offline": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _run_path_once(
    path_name: str,
    model_dir: Path,
    adapter_dir: Path,
    config: dict[str, Any],
    encoded_cpu: Any,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    if allocated_before > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {path_name} load exceeded residual CUDA ceiling: "
            f"{allocated_before} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    torch.cuda.reset_peak_memory_stats()
    model: Any | None = None
    evidence: dict[str, Any] | None = None
    trace: dict[str, list[Any]] | None = None
    try:
        if path_name == INDEPENDENT_PATH:
            model = stability_probe._load_model(  # noqa: SLF001
                model_dir,
                adapter_dir,
                config,
                merge=False,
            )
            precision = _independent_precision(model)
        elif path_name == CANDIDATE_PATH:
            model, precision = remediation_probe._load_fp32_merged_model(  # noqa: SLF001
                model_dir,
                adapter_dir,
                config,
            )
        else:
            raise RuntimeError(f"unknown path: {path_name}")
        remediation_probe._verify_generation_semantics(model, config)  # noqa: SLF001
        token_ids, trace, generation_trace = _generate_trace(
            model,
            encoded_cpu,
            config,
            tokenizer,
        )
        precision["generation"] = {
            "score_dtypes": generation_trace["scores"]["native_dtypes"],
            "all_scores_float32": generation_trace["scores"]["native_dtypes"]
            == ["float32"],
            "raw_logit_dtypes": generation_trace["raw_logits"]["native_dtypes"],
            "all_raw_logits_float32": generation_trace["raw_logits"]
            ["native_dtypes"]
            == ["float32"],
            "autocast_enabled": torch.is_autocast_enabled(),
            "training": model.training,
        }
        if path_name == INDEPENDENT_PATH:
            protocol_passed = _independent_protocol_passed(precision)
        else:
            protocol_passed = (
                remediation_probe._precision_protocol_passed(precision)  # noqa: SLF001
                and precision["generation"]["all_raw_logits_float32"] is True
            )
        if not protocol_passed:
            raise RuntimeError(f"{path_name} protocol failed: {precision!r}")
        torch.cuda.synchronize()
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        evidence = {
            "path": path_name,
            "fresh_load": 1,
            "generated_token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
            "output_sha256": "sha256:"
            + hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "precision_audit": precision,
            "generation_trace": generation_trace,
            "path_protocol_passed": protocol_passed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()
    if allocated_after > MAX_RESIDUAL_CUDA_BYTES:
        raise RuntimeError(
            f"fresh {path_name} load retained model-scale CUDA memory: "
            f"{allocated_after} > {MAX_RESIDUAL_CUDA_BYTES}"
        )
    if evidence is None or trace is None:
        raise RuntimeError(f"{path_name} run did not produce evidence")
    evidence["memory_allocated_before_load_bytes"] = allocated_before
    evidence["memory_allocated_after_release_bytes"] = allocated_after
    return evidence, trace


def _generate_trace(
    model: Any,
    encoded_cpu: Any,
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[int], dict[str, list[Any]], dict[str, Any]]:
    if torch.is_autocast_enabled():
        raise RuntimeError("generation must not run under autocast")
    encoded = {key: value.to("cuda") for key, value in encoded_cpu.items()}
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
    new_tokens = generated.sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = [int(value) for value in new_tokens.cpu()]
    if generated.scores is None or generated.logits is None:
        raise RuntimeError("generate did not return scores and raw logits")
    if len(generated.scores) != len(token_ids) or len(generated.logits) != len(
        token_ids
    ):
        raise RuntimeError("generated trace count does not match token count")
    expected_shape = [1, int(model.config.vocab_size)]
    scores, score_summary = _copy_trace(
        generated.scores,
        expected_shape=expected_shape,
    )
    raw_logits, raw_logit_summary = _copy_trace(
        generated.logits,
        expected_shape=expected_shape,
    )
    cache_returned = generated.past_key_values is not None
    if not cache_returned:
        raise RuntimeError("cached generation did not return past_key_values")
    return (
        token_ids,
        {"generated.scores": scores, "generated.logits": raw_logits},
        {
            "step_count": len(token_ids),
            "vocabulary_size": expected_shape[1],
            "cache_returned": cache_returned,
            "scores": score_summary,
            "raw_logits": raw_logit_summary,
        },
    )


def _copy_trace(
    tensors: Iterable[Any],
    *,
    expected_shape: list[int],
) -> tuple[list[Any], dict[str, Any]]:
    digest = hashlib.sha256()
    comparison: list[Any] = []
    dtypes: set[str] = set()
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
        comparison.append(native_cpu[0].float())
    return comparison, {
        "native_dtypes": sorted(dtypes),
        "shape_per_step": expected_shape,
        "comparison_dtype": "float32",
        "all_finite": True,
        "trace_sha256": "sha256:" + digest.hexdigest(),
    }


def _step_evidence(
    *,
    source: str,
    semantics: str,
    value_key: str,
    step_index: int,
    traces: dict[str, dict[str, list[Any]]],
    runs: list[dict[str, Any]],
    compared_token_ids: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    vectors = {path: traces[path][source][step_index] for path in PATH_ORDER}
    paths: dict[str, Any] = {}
    for run in runs:
        path_name = run["path"]
        vector = vectors[path_name]
        top = torch.topk(vector, k=TOP_K)
        top_ids = [int(value) for value in top.indices]
        top_values = [float(value) for value in top.values]
        if top_values[0] <= top_values[1]:
            raise RuntimeError(f"non-unique top {value_key} for {path_name}")
        emitted_token_id = run["generated_token_ids"][step_index]
        paths[path_name] = {
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
            "comparison_vector_sha256": _float32_vector_sha256(vector),
        }
    delta = (vectors[INDEPENDENT_PATH] - vectors[CANDIDATE_PATH]).abs()
    return {
        "step_index": step_index,
        "common_prefix_generated_tokens": step_index,
        "source": source,
        "semantics": semantics,
        "comparison_dtype": "float32",
        "top_k": TOP_K,
        "paths": paths,
        "delta": {
            "vocabulary_elements": delta.numel(),
            "nonzero_elements": int(torch.count_nonzero(delta)),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "root_mean_square_delta": float(torch.sqrt(torch.mean(delta.square()))),
        },
    }


def _float32_vector_sha256(vector: Any) -> str:
    if str(vector.dtype) != "torch.float32" or vector.ndim != 1:
        raise RuntimeError("comparison vector must be one-dimensional float32")
    payload = vector.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _link_divergent_step_hashes(
    runs: list[dict[str, Any]],
    selection_score_evidence: dict[str, Any],
    raw_logit_evidence: dict[str, Any],
) -> None:
    step_index = selection_score_evidence["step_index"]
    if raw_logit_evidence["step_index"] != step_index:
        raise RuntimeError("score and raw-logit evidence steps do not match")
    for run in runs:
        path_name = run["path"]
        for trace_key, evidence in (
            ("scores", selection_score_evidence),
            ("raw_logits", raw_logit_evidence),
        ):
            trace = run["generation_trace"][trace_key]
            trace["divergent_step_index"] = step_index
            trace["divergent_step_comparison_vector_sha256"] = evidence["paths"][
                path_name
            ]["comparison_vector_sha256"]


def _trace_evidence_linked(
    runs: list[dict[str, Any]],
    step_evidence: dict[str, Any],
    *,
    trace_key: str,
) -> bool:
    step_index = step_evidence.get("step_index")
    return isinstance(step_index, int) and all(
        run["generation_trace"]["step_count"] == run["token_count"]
        and run["generation_trace"][trace_key]["all_finite"] is True
        and run["generation_trace"][trace_key]["shape_per_step"]
        == [1, run["generation_trace"]["vocabulary_size"]]
        and run["generation_trace"][trace_key]["divergent_step_index"]
        == step_index
        and run["generation_trace"][trace_key][
            "divergent_step_comparison_vector_sha256"
        ]
        == step_evidence["paths"][run["path"]]["comparison_vector_sha256"]
        for run in runs
    )


def _independent_precision(model: Any) -> dict[str, Any]:
    base_parameters: list[tuple[str, Any]] = []
    adapter_parameters: list[tuple[str, Any]] = []
    for name, parameter in model.named_parameters():
        destination = adapter_parameters if ".lora_" in name else base_parameters
        destination.append((name, parameter))
    causal = _causal_model(model)
    return {
        "base_parameters": remediation_probe._tensor_inventory(  # noqa: SLF001
            base_parameters
        ),
        "adapter_parameters": remediation_probe._tensor_inventory(  # noqa: SLF001
            adapter_parameters
        ),
        "floating_buffers": remediation_probe._tensor_inventory(  # noqa: SLF001
            model.named_buffers()
        ),
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


def _independent_protocol_passed(value: dict[str, Any]) -> bool:
    generation = value.get("generation")
    return (
        _inventory_is_dtype_cuda(
            value["base_parameters"],
            dtype="bfloat16",
            expected_elements=1543714304,
        )
        and _inventory_is_dtype_cuda(
            value["adapter_parameters"],
            dtype="float32",
            expected_elements=4358144,
        )
        and _inventory_is_dtype_cuda(
            value["floating_buffers"],
            dtype="float32",
            expected_elements=64,
        )
        and value["lora_target_modules"] == EXPECTED_LORA_TARGETS
        and value["lora_parameter_tensors"] == EXPECTED_LORA_PARAMETER_TENSORS
        and value["adapter_parameters_finite"] is True
        and value["active_adapters"] == ["default"]
        and value["is_peft_model"] is True
        and value["input_output_embeddings_tied"] is True
        and value["attn_implementation"] == "sdpa"
        and value["attention_class"] == "Qwen2Attention"
        and value["output_attentions"] is False
        and value["hf_device_map"] is None
        and isinstance(generation, dict)
        and generation.get("score_dtypes") == ["float32"]
        and generation.get("all_scores_float32") is True
        and generation.get("raw_logit_dtypes") == ["float32"]
        and generation.get("all_raw_logits_float32") is True
        and generation.get("autocast_enabled") is False
        and generation.get("training") is False
    )


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


def _reproduction_evidence(
    runs: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    by_path = {run["path"]: run for run in runs}
    independent = by_path[INDEPENDENT_PATH]
    candidate = by_path[CANDIDATE_PATH]
    return {
        "independent_bf16_token_identity": (
            independent["token_count"]
            == references[INDEPENDENT_PATH]["token_count"]
            and independent["token_ids_sha256"]
            == references[INDEPENDENT_PATH]["token_ids_sha256"]
        ),
        "independent_bf16_output_identity": (
            independent["output_sha256"]
            == references[INDEPENDENT_PATH]["output_sha256"]
        ),
        "fp32_candidate_token_identity": (
            candidate["token_count"] == references[CANDIDATE_PATH]["token_count"]
            and candidate["token_ids_sha256"]
            == references[CANDIDATE_PATH]["token_ids_sha256"]
        ),
        "fp32_candidate_output_identity": (
            candidate["output_sha256"]
            == references[CANDIDATE_PATH]["output_sha256"]
        ),
    }


def _frozen_independent_reference(remediation: dict[str, Any]) -> dict[str, Any]:
    reference = remediation.get("reference")
    if not isinstance(reference, dict) or reference.get("path") != INDEPENDENT_PATH:
        raise RuntimeError("frozen independent BF16 reference is invalid")
    return dict(reference)


def _frozen_candidate_reference(remediation: dict[str, Any]) -> dict[str, Any]:
    runs = [
        run for run in remediation.get("runs", []) if run.get("path") == CANDIDATE_PATH
    ]
    if len(runs) != 2 or [run.get("repeat") for run in runs] != [1, 2]:
        raise RuntimeError("frozen FP32 candidate runs are invalid")
    identities = {
        (
            run.get("token_count"),
            run.get("token_ids_sha256"),
            run.get("output_sha256"),
        )
        for run in runs
    }
    if len(identities) != 1:
        raise RuntimeError("frozen FP32 candidate is not repeat-stable")
    token_count, token_digest, output_digest = identities.pop()
    if not isinstance(token_count, int) or not isinstance(token_digest, str):
        raise RuntimeError("frozen FP32 candidate token evidence is invalid")
    if not isinstance(output_digest, str):
        raise RuntimeError("frozen FP32 candidate output evidence is invalid")
    return {
        "path": CANDIDATE_PATH,
        "source_experiment_id": remediation["experiment_id"],
        "fresh_runs": 2,
        "token_count": token_count,
        "token_ids_sha256": token_digest,
        "output_sha256": output_digest,
    }


def _protocol(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    return {
        "freshness_scope": "fresh_model_load_lifecycle_in_fixed_process",
        "path_order": list(PATH_ORDER),
        "fresh_loads_per_path": 1,
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
        "paths": {
            INDEPENDENT_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "bfloat16",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "merge": False,
                "inference_parameter_dtypes": ["bfloat16", "float32"],
            },
            CANDIDATE_PATH: {
                "checkpoint_storage_dtype": "bfloat16",
                "base_load_dtype": "float32",
                "adapter_storage_dtype": "float32",
                "adapter_runtime_dtype": "float32",
                "merge_dtype": "float32",
                "inference_dtype": "float32",
                "merge": True,
                "safe_merge": True,
                "adapter_names": ["default"],
            },
        },
        "generation": {
            "attn_implementation": config["generation"]["attn_implementation"],
            "attention_class": "Qwen2Attention",
            "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
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
        "mem_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "fp16_bf16_reduction_math_sdp_allowed": (
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        ),
    }


def _constraints() -> dict[str, bool]:
    return {
        "failed_candidate_change": False,
        "locked_path_dtype_change": False,
        "locked_path_backend_change": False,
        "locked_path_decoding_change": False,
        "new_data": False,
        "training": False,
        "eval_answer_tuning": False,
        "runtime_integration": False,
        "full_eval_run": False,
        "merged_artifact_promotion": False,
    }


def _locked_next_action(classification: str) -> dict[str, Any]:
    constraints = _constraints()
    if classification == (
        "deterministic_bf16_attached_vs_fp32_merged_"
        "logits_processor_boundary_flip"
    ):
        return {
            "gate_id": "FC-MVP-001-fp32-logits-processor-analysis-v1",
            "action": (
                "on the same frozen eval-001 exact generation step, explain the "
                "raw-logit versus processed-score flip under the unchanged "
                "repetition penalty before testing another remediation"
            ),
            "acceptance": {
                "frozen_paths_reproduced": True,
                "processor_transform_reproduced": True,
                "selected_token_scores_reconciled": True,
                "source_inputs_unchanged": True,
            },
            "constraints": constraints,
        }
    return {
        "gate_id": "FC-MVP-001-fp32-attached-merge-isolation-v1",
        "action": (
            "add only a fresh independent FP32 Adapter path on frozen eval-001 "
            "and compare it with the unchanged FP32 safe-merged candidate, while "
            "retaining the frozen BF16 attached and merged token controls, so the "
            "same-dtype attached-versus-merged effect can be classified"
        ),
        "acceptance": {
            "attached_fp32_repeat_stable": True,
            "fp32_candidate_reproduced": True,
            "same_dtype_exact_step_compared": True,
            "same_dtype_attached_vs_merged_effect_classified": True,
            "source_inputs_unchanged": True,
        },
        "constraints": constraints,
    }


def _verify_sources(
    config: dict[str, Any],
    training: dict[str, Any],
    stability: dict[str, Any],
    numerics: dict[str, Any],
    remediation: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    config_digest = canonical_config_sha256(config)
    if any(
        evidence.get("config_sha256") != config_digest
        for evidence in (training, stability, numerics, remediation)
    ):
        raise RuntimeError("upstream config evidence mismatch")
    if (
        file_sha256(args.stability_evidence) != EXPECTED_STABILITY_SHA256
        or file_sha256(args.numerics_evidence) != EXPECTED_NUMERICS_SHA256
        or file_sha256(args.remediation_evidence) != EXPECTED_REMEDIATION_SHA256
        or remediation.get("candidate_protocol")
        != _expected_remediation_candidate_protocol()
        or stability.get("classification")
        != "deterministic_bf16_merge_logit_boundary_flip"
        or not all(stability.get("acceptance", {}).values())
        or numerics.get("classification") != "bf16_safe_merge_weight_rounding"
        or not all(numerics.get("acceptance", {}).values())
        or numerics.get("stability_evidence_sha256")
        != file_sha256(args.stability_evidence)
        or remediation.get("stability_evidence_sha256")
        != file_sha256(args.stability_evidence)
        or remediation.get("numerics_evidence_sha256")
        != file_sha256(args.numerics_evidence)
        or remediation.get("classification")
        != "deterministic_fp32_merge_output_drift"
        or not all(remediation.get("acceptance", {}).values())
        or remediation.get("remediation_gate", {}).get("passed") is not False
        or remediation.get("remediation_gate", {}).get(
            "candidate_repeats_identical"
        )
        is not True
        or remediation.get("remediation_gate", {}).get(
            "frozen_bf16_merged_token_identity"
        )
        is not True
        or remediation.get("locked_next_action", {}).get("gate_id")
        != "FC-MVP-001-fp32-merge-drift-analysis-v1"
        or remediation.get("merged_artifact_saved") is not False
        or remediation.get("merged_artifact_allowed") is not False
        or remediation.get("runtime_eligible") is not False
    ):
        raise RuntimeError("required FP32 remediation source chain is invalid")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if (
        training["final_adapter"].get("files") != adapter_manifest
        or stability.get("adapter_files") != adapter_manifest
        or numerics.get("adapter_files") != adapter_manifest
        or remediation.get("adapter_files") != adapter_manifest
    ):
        raise RuntimeError("adapter artifact mismatch")
    weight_path = args.model_dir / config["model"]["weight_file"]
    if weight_path.stat().st_size != config["model"]["weight_bytes"]:
        raise RuntimeError("model weight size mismatch")
    if file_sha256(weight_path) != f"sha256:{config['model']['weight_sha256']}":
        raise RuntimeError("model weight digest mismatch")
    prompt_path = ROOT / config["prompt"]["path"]
    if file_sha256(prompt_path) != f"sha256:{config['prompt']['sha256']}":
        raise RuntimeError("prompt digest mismatch")
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    if fixture_digest(evaluation) != config["data"]["eval_digest"]:
        raise RuntimeError("eval digest mismatch")
    if (
        evaluation[0]["example_id"] != "eval-001"
        or stability.get("example_id") != "eval-001"
        or numerics.get("example_id") != "eval-001"
        or remediation.get("example_id") != "eval-001"
    ):
        raise RuntimeError("locked drift probe requires eval-001")


def _verify_environment(config: dict[str, Any]) -> None:
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


def _verify_output_boundary(output: Path, model_dir: Path, adapter_dir: Path) -> None:
    resolved = output.resolve()
    for prohibited in (model_dir.resolve(), adapter_dir.resolve()):
        if resolved == prohibited or resolved.is_relative_to(prohibited):
            raise RuntimeError("probe output must not modify model or Adapter artifacts")


def _expected_storage_audit() -> dict[str, Any]:
    return {
        "base_checkpoint": {
            "tensors": 338,
            "elements": 1543714304,
            "dtype_tensors": {"bfloat16": 338},
            "dtype_elements": {"bfloat16": 1543714304},
        },
        "adapter": {
            "tensors": 224,
            "elements": 4358144,
            "dtype_tensors": {"float32": 224},
            "dtype_elements": {"float32": 4358144},
        },
    }


def _expected_remediation_candidate_protocol() -> dict[str, Any]:
    return {
        "checkpoint_storage_dtype": "bfloat16",
        "base_load_dtype": "float32",
        "adapter_storage_dtype": "float32",
        "adapter_load_dtype": "float32",
        "merge_dtype": "float32",
        "inference_dtype": "float32",
        "safe_merge": True,
        "adapter_names": ["default"],
        "attn_implementation": "sdpa",
        "attention_class": "Qwen2Attention",
        "attention_dispatch": "ALL_ATTENTION_FUNCTIONS['sdpa']",
        "output_attentions": False,
        "do_sample": False,
        "max_new_tokens": 256,
        "use_cache": True,
        "repetition_penalty": 1.1,
        "model_eos_token_ids": [151645, 151643],
        "model_pad_token_id": 151643,
        "call_pad_token_id": 151645,
        "tf32": False,
        "autocast": False,
        "device": "cuda:0",
        "fresh_loads": 2,
        "max_residual_cuda_bytes": MAX_RESIDUAL_CUDA_BYTES,
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


def _causal_model(model: Any) -> Any:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
