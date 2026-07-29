"""Load the frozen LoRA adapter locally and freeze Tool Router eval outputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import accelerate  # type: ignore[import-not-found]
import peft  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import transformers  # type: ignore[import-not-found]
from huggingface_hub import (  # type: ignore[import-not-found]
    __version__ as hub_version,
)
from peft import PeftModel  # type: ignore[import-not-found]
from safetensors import __version__ as safetensors_version  # type: ignore[import-not-found]
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_user_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_json(args.config)
    evidence = _load_json(args.training_evidence)
    _verify_environment(config)
    _verify_inputs(config, evidence, args)
    prompt_path = ROOT / config["prompt"]["path"]
    prompt = prompt_path.read_text(encoding="utf-8")
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])

    seed = config["generation"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        revision=config["tokenizer"]["revision"],
    )
    model_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    if not getattr(model_config, "use_sliding_window", False):
        model_config.sliding_window = None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=model_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["generation"]["attn_implementation"],
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=True,
        is_trainable=False,
    ).to("cuda")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    torch.cuda.reset_peak_memory_stats()

    outputs: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, record in enumerate(evaluation, start=1):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": render_user_payload(record)},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=config["generation"]["do_sample"],
                max_new_tokens=config["generation"]["max_new_tokens"],
                use_cache=config["generation"]["use_cache"],
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append({"example_id": record["example_id"], "raw_output": raw_output})
        print(
            f"{index:02d}/{len(evaluation)} {record['example_id']} "
            f"tokens={new_tokens.numel()}",
            flush=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    artifact = {
        "artifact_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": canonical_config_sha256(config),
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "environment": config["environment"],
        "generation": config["generation"],
        "prompt_sha256": config["prompt"]["sha256"],
        "eval_digest": config["data"]["eval_digest"],
        "training_evidence_sha256": file_sha256(args.training_evidence),
        "adapter_files": directory_artifact_manifest(args.adapter_dir),
        "performance": {
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak_memory,
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"peak_gpu_memory_bytes={peak_memory}")
    print(f"output={args.output}")
    return 0


def _verify_inputs(
    config: dict[str, Any], evidence: dict[str, Any], args: argparse.Namespace
) -> None:
    if evidence["experiment_id"] != config["experiment_id"]:
        raise RuntimeError("training evidence experiment mismatch")
    if evidence["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("training evidence config mismatch")
    if evidence["runtime_eligible"] is not False:
        raise RuntimeError("training evidence must remain runtime ineligible")
    adapter_files = directory_artifact_manifest(args.adapter_dir)
    if adapter_files != evidence["final_adapter"]["files"]:
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
    if len(evaluation) != config["data"]["eval_records"]:
        raise RuntimeError("eval record count mismatch")


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
