"""Command-line entry point for the offline bridge consumer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .consumer import BridgeValidationError, validate_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a manifest v1 and redacted run-export v1 fully offline."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-export", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = validate_files(args.manifest, args.run_export)
    except BridgeValidationError as exc:
        print(
            json.dumps(
                {"valid": False, "code": exc.code, "location": exc.location},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"valid": True, **summary.to_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
