"""Build or check the frozen FP32 attached artifact-eligibility review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router_fp32_attached_artifact_eligibility import (  # noqa: E402
    build_fp32_attached_artifact_eligibility_review,
    inspect_adapter_safetensors_bytes,
)

OUTPUT = (
    ROOT / "baseline" / "fc-mvp-001-fp32-attached-artifact-eligibility-review-v1.json"
)


def _source_paths() -> dict[str, Path]:
    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    return {
        "adapter_config": adapter / "adapter_config.json",
        "adapter_readme": adapter / "README.md",
        "adapter_verifier_source": ROOT
        / "scripts"
        / "verify_tool_router_lora_adapter.py",
        "adapter_weights": adapter / "adapter_model.safetensors",
        "attached_dtype_isolation": ROOT
        / "baseline"
        / "fc-mvp-001-attached-dtype-isolation-v1.json",
        "canonical_json_source": ROOT / "src" / "fullcycle_bridge" / "consumer.py",
        "decision_compiler_source": ROOT
        / "src"
        / "fullcycle_bridge"
        / "tool_router_decision_compilation.py",
        "evaluation_fixture": ROOT / "fixtures" / "tool_router_v1" / "eval.json",
        "gitattributes": ROOT / ".gitattributes",
        "inference_runner_source": ROOT
        / "scripts"
        / "run_tool_router_lora_inference.py",
        "lifecycle_evidence": ROOT / "baseline" / "fc-mvp-001-lora-sft-v2.json",
        "load_merge_evidence": ROOT
        / "baseline"
        / "fc-mvp-001-lora-sft-v2-load-merge.json",
        "model_downloader_source": ROOT
        / "scripts"
        / "download_pinned_tool_router_model.py",
        "prompt": ROOT / "prompts" / "tool_router_v1.txt",
        "remediation_gate": ROOT
        / "baseline"
        / "fc-mvp-001-fp32-attached-remediation-eval-v1.json",
        "remediation_predictions": ROOT
        / "baseline"
        / "tool-router-fp32-attached-remediation-v1-predictions.json",
        "remediation_preregistration": ROOT
        / "configs"
        / "tool_router_fp32_attached_remediation_eval_v1.json",
        "review_builder_source": ROOT
        / "scripts"
        / "review_tool_router_fp32_attached_artifact_eligibility.py",
        "review_contract_source": ROOT
        / "src"
        / "fullcycle_bridge"
        / "tool_router_fp32_attached_artifact_eligibility.py",
        "sft_config": ROOT / "configs" / "tool_router_lora_sft_v2.json",
        "sft_helpers_source": ROOT / "src" / "fullcycle_bridge" / "tool_router_sft.py",
        "training_evidence": ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json",
        "training_lock": ROOT / "requirements" / "training.lock",
        "training_runner_source": ROOT / "scripts" / "train_tool_router_lora.py",
        "validation_error_source": ROOT / "src" / "fullcycle_bridge" / "tool_router.py",
    }


def build() -> dict[str, Any]:
    """Recompute the review from immutable repository-local evidence."""

    return build_fp32_attached_artifact_eligibility_review(**load_review_inputs())


def load_review_inputs() -> dict[str, Any]:
    """Read each source once, then derive content and hashes from those bytes."""

    paths = _source_paths()
    for name, path in paths.items():
        _require_regular_file(path, name)

    adapter = ROOT / "baseline" / "adapters" / "fc-mvp-001-lora-sft-v2"
    _require_safe_adapter_tree(adapter)
    source_payloads = {name: path.read_bytes() for name, path in sorted(paths.items())}
    _require_safe_adapter_tree(adapter)
    source_hashes = {
        name: "sha256:" + hashlib.sha256(payload).hexdigest()
        for name, payload in source_payloads.items()
    }
    adapter_sources = {
        "adapter_config.json": "adapter_config",
        "adapter_model.safetensors": "adapter_weights",
        "README.md": "adapter_readme",
    }
    adapter_files = [
        {
            "path": path,
            "bytes": len(source_payloads[source_name]),
            "sha256": source_hashes[source_name],
        }
        for path, source_name in sorted(
            adapter_sources.items(), key=lambda item: item[0].casefold()
        )
    ]
    return {
        "remediation_preregistration": _load_json_payload(
            source_payloads["remediation_preregistration"],
            paths["remediation_preregistration"],
        ),
        "remediation_predictions": _load_json_payload(
            source_payloads["remediation_predictions"],
            paths["remediation_predictions"],
        ),
        "remediation_gate": _load_json_payload(
            source_payloads["remediation_gate"], paths["remediation_gate"]
        ),
        "attached_dtype_isolation": _load_json_payload(
            source_payloads["attached_dtype_isolation"],
            paths["attached_dtype_isolation"],
        ),
        "sft_config": _load_json_payload(
            source_payloads["sft_config"], paths["sft_config"]
        ),
        "training_evidence": _load_json_payload(
            source_payloads["training_evidence"], paths["training_evidence"]
        ),
        "lifecycle_evidence": _load_json_payload(
            source_payloads["lifecycle_evidence"], paths["lifecycle_evidence"]
        ),
        "load_merge_evidence": _load_json_payload(
            source_payloads["load_merge_evidence"], paths["load_merge_evidence"]
        ),
        "adapter_config": _load_json_payload(
            source_payloads["adapter_config"], paths["adapter_config"]
        ),
        "adapter_readme": _decode_text(
            source_payloads["adapter_readme"], paths["adapter_readme"]
        ),
        "adapter_files": adapter_files,
        "adapter_tensor_audit": inspect_adapter_safetensors_bytes(
            source_payloads["adapter_weights"]
        ),
        "training_lock": _decode_text(
            source_payloads["training_lock"], paths["training_lock"]
        ),
        "gitattributes": _decode_text(
            source_payloads["gitattributes"], paths["gitattributes"]
        ),
        "source_hashes": source_hashes,
        "source_payloads": source_payloads,
    }


def _require_regular_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"unsafe or missing {name}: {path}")


def _require_safe_adapter_tree(adapter: Path) -> None:
    if not adapter.is_dir() or adapter.is_symlink():
        raise RuntimeError(f"unsafe or missing adapter directory: {adapter}")
    entries = list(adapter.rglob("*"))
    expected_files = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    actual_files: set[str] = set()
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe adapter entry: {path}")
        actual_files.add(path.relative_to(adapter).as_posix())
    if actual_files != expected_files:
        raise RuntimeError(
            f"adapter tree mismatch: expected={sorted(expected_files)!r},"
            f"actual={sorted(actual_files)!r}"
        )


def _load_json_payload(payload: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    _validate_finite(value, f"$.{path.name}")
    return value


def _decode_text(payload: bytes, path: Path) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"invalid UTF-8 source: {path}: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise RuntimeError(f"non-finite JSON constant: {value}")


def _validate_finite(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON number at {path}: {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def _payload(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and require the existing output to match byte-for-byte",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected = build()
    payload = _payload(expected)
    output = args.output.resolve()
    if args.check:
        _require_regular_file(output, "review output")
        if output.read_bytes() != payload:
            raise RuntimeError(f"frozen review differs from recomputation: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(payload)
    print(
        json.dumps(
            {
                "valid": True,
                "check": args.check,
                "output": str(output),
                "bytes": len(payload),
                "report_digest": expected["report_digest"],
                "classification": expected["eligibility_decision"]["classification"],
                "offline_artifact_eligible": expected["eligibility_decision"][
                    "offline_artifact_eligible"
                ],
                "runtime_eligible": expected["runtime_eligible"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
