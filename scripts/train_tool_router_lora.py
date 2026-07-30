"""Train one locked, fully local BF16 LoRA adapter for Tool Router v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import accelerate  # type: ignore[import-not-found]  # noqa: E402
import peft  # type: ignore[import-not-found]  # noqa: E402
import torch  # type: ignore[import-not-found]  # noqa: E402
import transformers  # type: ignore[import-not-found]  # noqa: E402
from huggingface_hub import (  # type: ignore[import-not-found]  # noqa: E402
    __version__ as hub_version,
)
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]  # noqa: E402
from safetensors import __version__ as safetensors_version  # type: ignore[import-not-found]  # noqa: E402
from tokenizers import __version__ as tokenizers_version  # type: ignore[import-not-found]  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]  # noqa: E402
from transformers import (  # type: ignore[import-not-found]  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fullcycle_bridge.tool_router import fixture_digest, load_fixture  # noqa: E402
from fullcycle_bridge.tool_router_sft import (  # noqa: E402
    canonical_config_sha256,
    directory_artifact_manifest,
    file_sha256,
    render_target,
    render_user_payload,
)


class TokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-tokenized, fixed-shape causal LM records with prompt loss masked."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        prompt: str,
        max_length: int,
    ):
        self.items: list[dict[str, torch.Tensor]] = []
        self.lengths: list[int] = []
        self.prompt_lengths: list[int] = []
        self.truncated_records = 0
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        for record in records:
            prefix_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": render_user_payload(record)},
            ]
            full_messages = [
                *prefix_messages,
                {"role": "assistant", "content": render_target(record)},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if len(input_ids) > max_length:
                self.truncated_records += 1
                input_ids = input_ids[:max_length]
            if len(prompt_ids) >= len(input_ids):
                raise RuntimeError(
                    f"target has no trainable tokens: {record['example_id']}"
                )
            length = len(input_ids)
            padding = max_length - length
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
            labels += [-100] * padding
            attention_mask = [1] * length + [0] * padding
            input_ids += [pad_token_id] * padding
            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
            self.lengths.append(length)
            self.prompt_lengths.append(len(prompt_ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()

    config = _load_json(args.config)
    _verify_environment(config)
    _verify_model(config, args.model_dir)
    prompt_path = ROOT / config["prompt"]["path"]
    if file_sha256(prompt_path) != f"sha256:{config['prompt']['sha256']}":
        raise RuntimeError("prompt digest mismatch")
    prompt = prompt_path.read_text(encoding="utf-8")
    train_records = _load_locked_split(config, "train")
    validation_records = _load_locked_split(config, "validation")

    seed = config["training"]["seed"]
    _seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        revision=config["tokenizer"]["revision"],
    )
    tokenizer.padding_side = "right"
    max_length = config["training"]["max_sequence_length"]
    train_dataset = TokenizedDataset(
        train_records, tokenizer, prompt, max_length
    )
    validation_dataset = TokenizedDataset(
        validation_records, tokenizer, prompt, max_length
    )
    if train_dataset.truncated_records or validation_dataset.truncated_records:
        raise RuntimeError("locked SFT config must have zero truncated records")

    generator = torch.Generator()
    generator.manual_seed(seed)
    micro_batch_size = config["training"]["micro_batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    if not getattr(model_config, "use_sliding_window", False):
        model_config.sliding_window = None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=model_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["training"]["attn_implementation"],
    )
    lora = config["lora"]
    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
    )
    model = get_peft_model(base_model, adapter_config)
    if config["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    setattr(model.config, "use_cache", False)
    model.to("cuda")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    accumulation = config["training"]["gradient_accumulation_steps"]
    epochs = config["training"]["epochs"]
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = round(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    epoch_metrics: list[dict[str, Any]] = []
    checkpoint_manifests: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            epoch_loss += float(loss.detach())
            batch_count += 1
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        validation_loss = _evaluate_loss(model, validation_loader)
        torch.cuda.synchronize()
        epoch_elapsed = time.perf_counter() - epoch_started
        checkpoint_dir = checkpoints_dir / f"epoch-{epoch:02d}"
        model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        manifest = directory_artifact_manifest(checkpoint_dir)
        checkpoint_manifests.append(
            {
                "epoch": epoch,
                "relative_path": checkpoint_dir.relative_to(args.output_dir).as_posix(),
                "files": manifest,
                "total_bytes": sum(item["bytes"] for item in manifest),
            }
        )
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / batch_count,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": epoch_elapsed,
        }
        epoch_metrics.append(metrics)
        print(json.dumps(metrics, sort_keys=True, separators=(",", ":")), flush=True)

    final_adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(
        final_adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    adapter_manifest = directory_artifact_manifest(final_adapter_dir)
    evidence = {
        "evidence_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": canonical_config_sha256(config),
        "model": config["model"],
        "environment": config["environment"],
        "data": config["data"],
        "prompt": config["prompt"],
        "lora": config["lora"],
        "training": config["training"],
        "tokenization": {
            "train_min_tokens": min(train_dataset.lengths),
            "train_max_tokens": max(train_dataset.lengths),
            "train_mean_tokens": sum(train_dataset.lengths)
            / len(train_dataset.lengths),
            "validation_min_tokens": min(validation_dataset.lengths),
            "validation_max_tokens": max(validation_dataset.lengths),
            "validation_mean_tokens": sum(validation_dataset.lengths)
            / len(validation_dataset.lengths),
            "train_truncated_records": train_dataset.truncated_records,
            "validation_truncated_records": validation_dataset.truncated_records,
        },
        "parameters": {
            "trainable": trainable_parameters,
            "all_with_adapter": all_parameters,
            "trainable_percent": 100 * trainable_parameters / all_parameters,
        },
        "optimization": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "completed_steps": global_step,
        },
        "epoch_metrics": epoch_metrics,
        "checkpoints": checkpoint_manifests,
        "final_adapter": {
            "relative_path": "adapter",
            "files": adapter_manifest,
            "total_bytes": sum(item["bytes"] for item in adapter_manifest),
        },
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "selection": "final_epoch_locked_before_eval",
        "runtime_eligible": False,
    }
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"peak_gpu_memory_bytes={peak_memory}")
    print(f"trainable_parameters={trainable_parameters}")
    print(f"adapter_bytes={evidence['final_adapter']['total_bytes']}")
    print(f"evidence={args.evidence_output}")
    return 0


def _evaluate_loss(model: Any, loader: DataLoader[Any]) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            total += float(model(**batch).loss)
            count += 1
    return total / count


def _load_locked_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    data = config["data"]
    records = load_fixture(ROOT / data[f"{split}_path"])
    if fixture_digest(records) != data[f"{split}_digest"]:
        raise RuntimeError(f"{split} digest mismatch")
    if len(records) != data[f"{split}_records"]:
        raise RuntimeError(f"{split} record count mismatch")
    if any(record["split"] != split for record in records):
        raise RuntimeError(f"{split} fixture contains another split")
    return records


def _verify_model(config: dict[str, Any], model_dir: Path) -> None:
    weight_path = model_dir / config["model"]["weight_file"]
    if weight_path.stat().st_size != config["model"]["weight_bytes"]:
        raise RuntimeError("model weight size mismatch")
    if file_sha256(weight_path) != f"sha256:{config['model']['weight_sha256']}":
        raise RuntimeError("model weight digest mismatch")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


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
        raise RuntimeError(f"expected config object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
