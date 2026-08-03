"""Build the frozen FC-MVP-001 LoRA SFT v2 failure-classification report."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fullcycle_bridge.tool_router_failure_classification import (  # noqa: E402
    classify_v2_failures,
)

PREDICTIONS = (
    ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-predictions.json"
)
REPORT = ROOT / "baseline" / "tool-router-qwen2.5-1.5b-lora-sft-v2-report.json"
TRAINING = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-training.json"
LOAD_MERGE = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-load-merge.json"
OUTPUT = ROOT / "baseline" / "fc-mvp-001-lora-sft-v2-failure-classification.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = classify_v2_failures(
        _load(PREDICTIONS),
        _load(REPORT),
        _load(TRAINING),
        _load(LOAD_MERGE),
        {
            "predictions": _sha256(PREDICTIONS),
            "report": _sha256(REPORT),
            "training": _sha256(TRAINING),
            "load_merge": _sha256(LOAD_MERGE),
        },
    )
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
