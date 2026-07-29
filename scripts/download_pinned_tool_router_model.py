"""Download and verify the pinned local Tool Router baseline model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHT_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=args.local_dir,
        allow_patterns=list(FILES),
        max_workers=4,
    )
    weight_path = Path(path) / "model.safetensors"
    actual = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if actual != WEIGHT_SHA256:
        raise RuntimeError(f"weight digest mismatch: {actual}")
    print(f"verified_model={path}")
    print(f"revision={REVISION}")
    print(f"weight_sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
