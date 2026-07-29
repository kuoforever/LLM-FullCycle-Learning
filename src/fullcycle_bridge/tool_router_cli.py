"""Offline CLI for Tool Router fixture validation and baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .consumer import canonical_json_bytes
from .tool_router import (
    ToolRouterValidationError,
    baseline_predict,
    evaluate,
    fixture_digest,
    load_fixture,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, required=True)
    args = parser.parse_args()
    try:
        seed = load_fixture(args.seed)
        evaluation = load_fixture(args.eval_path)
        if any(record["split"] != "seed" for record in seed):
            raise ToolRouterValidationError("WRONG_SPLIT", "$.seed", "expected seed")
        if any(record["split"] != "eval" for record in evaluation):
            raise ToolRouterValidationError("WRONG_SPLIT", "$.eval", "expected eval")
        report = {
            "valid": True,
            "seed_records": len(seed),
            "eval_records": len(evaluation),
            "seed_digest": fixture_digest(seed),
            "eval_digest": fixture_digest(evaluation),
            "baseline": evaluate(
                evaluation, [baseline_predict(record) for record in evaluation]
            ),
        }
        print(canonical_json_bytes(report).decode("utf-8"))
        return 0
    except ToolRouterValidationError as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "code": exc.code,
                    "path": exc.path,
                    "detail": exc.detail,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
