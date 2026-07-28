"""JSONL command surface for the deterministic Lane A dataset mapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .consumer import BridgeValidationError, canonical_json_bytes
from .dataset import map_many


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map validated Lane A run exports to deterministic JSONL records."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--run-export", required=True, action="append", type=Path, dest="run_exports"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        records = map_many(args.manifest, args.run_exports)
    except BridgeValidationError as exc:
        error = canonical_json_bytes(
            {"valid": False, "code": exc.code, "location": exc.location}
        )
        sys.stderr.buffer.write(error + b"\n")
        return 2
    for record in records:
        sys.stdout.buffer.write(canonical_json_bytes(record) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
