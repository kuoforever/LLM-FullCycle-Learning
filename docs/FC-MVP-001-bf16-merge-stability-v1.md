# FC-MVP-001 BF16 merge stability v1

> **Result: COMPLETE LOCALLY — both load paths are repeat-stable, and the
> cross-path failure is a deterministic BF16 merge logit-boundary flip. The
> merged artifact remains prohibited and Runtime eligibility remains false.**

## Locked probe

The probe uses the frozen LoRA SFT v2 model, Adapter, prompt, seed, BF16 dtype,
SDPA generation settings, eval digest, and exact `eval-001` input. The rendered
input contains 339 tokens and has token-ID digest
`sha256:3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f`.

Each path is loaded from disk twice. The independent path keeps the Adapter
attached. The merged path calls PEFT `merge_and_unload(safe_merge=True)` and
verifies that no LoRA parameter tensors remain. No merged weights are saved.

## Repeat evidence

| Path | Repeat 1 token digest | Repeat 2 token digest | Stable |
|---|---|---|---:|
| Independent Adapter | `sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173` | same | Yes |
| Safe-merged BF16 | `sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca` | same | Yes |

Both outputs contain 48 generated tokens. Their first 45 generated tokens are
identical. At zero-based token index 45, the independent path chooses token
1866 (`true`) and the merged path chooses token 3849 (`false`).

## Exact generation-step logits

The logits come directly from the same `generate(use_cache=True)` calls that
produced the frozen tokens; they are not reconstructed with a different cache
path.

| Path | Top token | Runner-up | Top score | Runner-up score | Margin |
|---|---|---|---:|---:|---:|
| Independent Adapter | `true` (1866) | `false` (3849) | 34.545452 | 34.090908 | 0.454544 |
| Safe-merged BF16 | `false` (3849) | `true` (1866) | 36.590908 | 32.500000 | 4.090908 |

At this boundary, the maximum absolute vocabulary-logit delta is `3.0` and the
mean absolute delta is `0.3340962529`. The classification is therefore
`deterministic_bf16_merge_logit_boundary_flip`, not within-path nondeterminism.

## Acceptance and boundary

All locked acceptance checks pass: repeat identity for each path, exact logit
argmax agreement with the generated tokens, classified divergence, unchanged
source Adapter, and unchanged eval digest. The run took `13.6114153` seconds
and peaked at `6,246,685,696` allocated GPU bytes on the locked RTX 4090 Laptop
GPU environment.

The gate does not claim a repaired merge. `merged_artifact_allowed=false`, no
merged artifact was saved, and `runtime_eligible=false`. It adds no data, does
not train, does not run the full eval, and does not connect Runtime, Provider,
MCP, or Desktop.

The canonical evidence is
[`baseline/fc-mvp-001-bf16-merge-stability-v1.json`](../baseline/fc-mvp-001-bf16-merge-stability-v1.json),
with SHA-256
`82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc`.

## Next gate

The single next objective is `FC-MVP-001-bf16-merge-numerics-v1`: on the same
frozen `eval-001` common prefix, locate the earliest module-level divergence
between independent and merged BF16 execution, then quantify the Adapter update
against safe-merged-weight rounding. The same no-data, no-training, no-Runtime,
and no-merged-artifact-promotion boundaries remain in force.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_bf16_merge_stability.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --output .\work\test-fixtures\fc-mvp-001-bf16-merge-stability-v1.json

python -I .\scripts\validate_offline.py
```
