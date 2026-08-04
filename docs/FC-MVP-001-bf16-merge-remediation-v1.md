# FC-MVP-001 BF16 merge remediation v1

> **Result: COMPLETE LOCALLY — the pre-registered FP32 safe-merge candidate is
> repeat-stable but does not restore the frozen independent BF16 Adapter token
> output. Full eval and merged-artifact promotion remain prohibited.**

## Locked candidate

The only candidate loads the pinned BF16 checkpoint values as FP32, attaches
the frozen FP32 Adapter, safe-merges the Adapter into FP32 base weights, and
retains the merged model in FP32 for greedy SDPA generation. The prompt, seed,
tokenizer, generation semantics, Adapter, eval digest, and exact `eval-001`
input remain frozen.

Loading the checkpoint with `torch_dtype=torch.float32` materializes its saved
BF16 values in FP32. It does not recover base-weight precision that was not
present in the checkpoint. The purpose of this candidate is narrower: prevent
the FP32 LoRA delta from being materialized back into BF16 during merge.

Each candidate run has an independent load/merge/generate lifecycle. Before
the second load, the first model is destroyed and CUDA allocated memory must
remain below a locked 16 MiB process-residual ceiling. No reference model and
candidate model coexist on the GPU.

## Precision and protocol audit

Safetensors metadata confirms that all 338 base-checkpoint tensors are stored
as BF16 and all 224 Adapter tensors are stored as FP32. No tensor payload needs
to be copied merely to establish the storage dtype.

Both fresh runs satisfy the same audit:

| Stage | Floating tensors | Floating elements | Dtype/device | LoRA targets |
|---|---:|---:|---|---:|
| Base before merge | 338 | 1,543,714,304 | FP32 / `cuda:0` | — |
| Adapter before merge | 224 | 4,358,144 | FP32 / `cuda:0` | 112 |
| Model after merge | 338 | 1,543,714,304 | FP32 / `cuda:0` | 0 |
| Generation scores | one per generated token | vocabulary logits | FP32 / `cuda:0` | — |

The Adapter parameters are finite, active Adapter name is exactly `default`,
`merge_and_unload(safe_merge=True, adapter_names=["default"])` removes every
LoRA tensor and `BaseTunerLayer`, and input/output embeddings remain tied.
Transformers 4.49 dispatches the unified `Qwen2Attention` implementation
through `ALL_ATTENTION_FUNCTIONS['sdpa']`; `output_attentions=false` prevents
the documented eager fallback. Autocast and both TF32 flags are disabled.

The effective generation contract stays identical to the frozen reference:
`do_sample=false`, `use_cache=true`, `max_new_tokens=256`, repetition penalty
1.1, model EOS IDs `[151645, 151643]`, and call-time PAD ID 151645.

## Repeat and reference evidence

| Path | Runs | Tokens | Token digest | Output digest |
|---|---:|---:|---|---|
| Independent BF16 Adapter reference | 2 frozen | 48 | `sha256:e23b3f5e…54e173` | `sha256:b3bef0f2…f0bc5` |
| FP32 safe-merged candidate | 2 fresh | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` |
| Prior safe-merged BF16 control | 2 frozen | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` |

The two FP32 candidate runs are token-identical to each other, so the result is
not within-path nondeterminism. They do not match the independent BF16 Adapter
reference and therefore fail the pre-registered remediation requirement. Their
token and decoded-output digests happen to equal the prior safe-merged BF16
control. That is a token-level observation only; this gate does not claim that
the FP32 and BF16 merged logits, activations, or weights are identical.

The classification is `deterministic_fp32_merge_output_drift` and
`remediation_gate.passed=false`.

## Resource and lifecycle evidence

The final two-run probe took `18.06072970002424` seconds. Peak allocated GPU
memory was `6,248,754,688` bytes. The first run began at zero allocated bytes
and released to `8,519,680` bytes; the second began and ended at the same
`8,519,680`-byte process residual. This is far below one model's parameter
footprint and demonstrates fresh model-load isolation without requiring
PyTorch's process-level CUDA workspace to disappear.

The unified offline gate passes 87 tests on Python 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, and mypy 2.3.0 reports no issues in all 33
source/script files.

## Boundary and next gate

This result completes the pre-registered candidate experiment, not the merge
repair. It adds no data, performs no training or eval-answer tuning, runs only
`eval-001`, saves no merged weights, and does not connect Runtime, Provider,
MCP, or Desktop. `merged_artifact_allowed=false` and
`runtime_eligible=false`.

The canonical evidence is
[`baseline/fc-mvp-001-bf16-merge-remediation-v1.json`](../baseline/fc-mvp-001-bf16-merge-remediation-v1.json),
with SHA-256
`7f3c5aff55e69c08a7676d33636a52a5a2bb43f025dae8a2db362041354050b3`.

The single next objective is `FC-MVP-001-fp32-merge-drift-analysis-v1`:
reproduce the stable FP32 candidate and a fresh independent BF16 Adapter on
the unchanged `eval-001`, then locate the first token and quantify the exact
generation-step logit divergence. The candidate, backend, generation settings,
data, and artifact-prohibition boundaries remain locked.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_bf16_merge_remediation.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --numerics-evidence .\baseline\fc-mvp-001-bf16-merge-numerics-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-bf16-merge-remediation-v1.json

python -I .\scripts\validate_offline.py
```
