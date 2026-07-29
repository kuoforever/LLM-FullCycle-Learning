"""Score a frozen raw Tool Router prediction artifact without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_model_eval import score_raw_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_json(args.config)
    artifact = _load_json(args.predictions)
    evaluation = load_fixture(ROOT / config["eval"]["path"])
    if fixture_digest(evaluation) != config["eval"]["canonical_digest"]:
        raise RuntimeError("eval digest mismatch")
    for key in (
        "model",
        "tokenizer",
        "environment",
        "generation",
    ):
        if artifact[key] != config[key]:
            raise RuntimeError(f"prediction artifact config mismatch: {key}")
    if artifact["prompt_sha256"] != config["prompt"]["sha256"]:
        raise RuntimeError("prediction prompt mismatch")
    if artifact["eval_digest"] != config["eval"]["canonical_digest"]:
        raise RuntimeError("prediction eval mismatch")
    raw_outputs = [
        {
            "example_id": item["example_id"],
            "raw_output": item["raw_output"],
        }
        for item in artifact["outputs"]
    ]
    report, parsed = score_raw_outputs(evaluation, raw_outputs)
    prediction_digest = (
        "sha256:" + hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    )
    output = {
        "report_version": 2,
        "prediction_artifact_sha256": prediction_digest,
        "eval_digest": config["eval"]["canonical_digest"],
        "metrics": report,
        "parsed_outputs": parsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    print(f"prediction_artifact_sha256={prediction_digest}")
    print(f"output={args.output}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
