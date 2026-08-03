"""Verify that the frozen Tool Router LoRA loads and safely merges offline."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch  # type: ignore[import-not-found]
from peft import PeftModel  # type: ignore[import-not-found]
from transformers import (  # type: ignore[import-not-found]
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router import load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
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
    if evidence["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("training evidence config mismatch")
    adapter_manifest = directory_artifact_manifest(args.adapter_dir)
    if adapter_manifest != evidence["final_adapter"]["files"]:
        raise RuntimeError("adapter artifact mismatch")

    seed = config["generation"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
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
    trainable_before_merge = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    evaluation = load_fixture(ROOT / config["data"]["eval_path"])
    record = evaluation[0]
    prompt = (ROOT / config["prompt"]["path"]).read_text(encoding="utf-8")
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
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    before = _generate(model, encoded, tokenizer, config)
    merged = model.merge_and_unload(safe_merge=True)
    merged.eval()
    after = _generate(merged, encoded, tokenizer, config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs_identical = before == after
    remaining_adapter_parameters = sum(
        ".lora_" in name for name, _ in merged.named_parameters()
    )
    if remaining_adapter_parameters:
        raise RuntimeError("merged model still contains LoRA parameters")

    result = {
        "verification_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": canonical_config_sha256(config),
        "adapter_files": adapter_manifest,
        "example_id": record["example_id"],
        "loaded_output": before,
        "merged_output": after,
        "outputs_identical": outputs_identical,
        "safe_merge": True,
        "trainable_parameters_before_merge": trainable_before_merge,
        "remaining_adapter_parameter_tensors": remaining_adapter_parameters,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "merged_model_saved": False,
        "offline": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if not outputs_identical:
        print(
            "merged model output differs from loaded adapter output",
            file=sys.stderr,
        )
        return 2
    return 0


def _generate(
    model: Any,
    encoded: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> str:
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=config["generation"]["do_sample"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            use_cache=config["generation"]["use_cache"],
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
