"""Offline CLI for Tool Router train/validation family audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .consumer import canonical_json_bytes
from .tool_router import ToolRouterValidationError, fixture_digest, load_fixture
from .tool_router_dataset import audit_dataset, load_family_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        train = load_fixture(args.train)
        validation = load_fixture(args.validation)
        evaluation = load_fixture(args.eval_path)
        manifest = load_family_manifest(args.family_manifest)
        report = audit_dataset(
            train,
            validation,
            evaluation,
            manifest,
            fixture_digest(evaluation),
        )
        print(canonical_json_bytes({"valid": True, "report": report}).decode("utf-8"))
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
