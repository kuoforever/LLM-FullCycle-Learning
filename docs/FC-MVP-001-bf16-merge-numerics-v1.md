# FC-MVP-001 BF16 merge numerics v1

> **Result: COMPLETE LOCALLY — the deterministic output flip originates at
> the first LoRA projection and is explained by BF16 safe-merge weight
> rounding, not a PEFT merge implementation mismatch.**

## Locked execution point

The probe reproduces both frozen merge-stability outputs and their logits on
the exact cached generation step for zero-based token index 45. It attaches
paired hooks to the independent Adapter and safe-merged BF16 paths without
changing the prompt, tokens, seed, model, Adapter, dtype, or generation mode.

The independent path again selects `true` (token 1866) and the merged path
again selects `false` (token 3849). Their complete 48-token digests remain
identical to the preceding stability gate.

## Earliest module divergence

| Execution stage | Different elements | Maximum absolute delta | Result |
|---|---:|---:|---|
| `model.embed_tokens` | 0 | 0 | Identical |
| `model.layers.0.input_layernorm` | 0 | 0 | Identical |
| `model.layers.0.self_attn.q_proj` | 569 / 1,536 | 0.0625 | First divergence |
| `model.layers.0.self_attn.k_proj` | 83 / 256 | 0.5 | Diverged |
| `model.layers.0.self_attn.v_proj` | 134 / 256 | 0.00390625 | Diverged |
| `model.layers.0.self_attn.o_proj` | 1,191 / 1,536 | 0.0078125 | Diverged |
| `lm_head` | 148,563 / 151,936 | 3.0 | Diverged |

Because the embedding and first input normalization are exactly equal, the
first difference at layer 0 `q_proj` is reached with an identical activation.
That isolates the divergence to how the LoRA update is represented after
merge, before attention state or later-layer differences can accumulate.

## Weight-rounding audit

The audit recomputes the ideal FP32 LoRA update for every Q/K/V/O projection,
applies PEFT's BF16 in-place safe-merge rule, and compares the result with the
actual merged model.

| Measure | Result |
|---|---:|
| LoRA target modules | 112 |
| Target weights | 154,140,672 |
| Ideal nonzero updates | 154,140,672 |
| Effective changed BF16 weights | 123,499,678 |
| Nonzero updates rounded back to the base value | 30,640,994 |
| Rounded-away fraction | 19.878591% |
| Actual merged weights differing from reproduced PEFT merge | 0 |
| Maximum absolute rounding error | 0.0011988399783149362 |
| Mean absolute rounding error | 0.000029496931764409027 |

For the first divergent layer 0 `q_proj`, 717,204 of 2,359,296 ideal nonzero
updates are rounded back to the base BF16 value. Its generated-step output then
differs in 569 of 1,536 elements.

The classification is `bf16_safe_merge_weight_rounding`. This does not mean
PEFT executed a different algorithm than expected: the reproduced algorithm
matches all 154,140,672 actual merged target weights exactly. The failure is
the loss of effective Adapter updates when the update is materialized into
BF16 base weights.

## Boundary and next gate

The run took `15.5990462` seconds and peaked at `9,360,067,072` allocated GPU
bytes. It did not train, alter data, run the full eval, save a merged model, or
connect Runtime, Provider, MCP, or Desktop. The merged artifact remains
prohibited and `runtime_eligible=false`.

The canonical evidence is
[`baseline/fc-mvp-001-bf16-merge-numerics-v1.json`](../baseline/fc-mvp-001-bf16-merge-numerics-v1.json),
with SHA-256
`eb39674127ac93fea2ce6415b3a2fea0d20f6da916b76f1532392533db3e805f`.

The next gate is `FC-MVP-001-bf16-merge-remediation-v1`. Its sole
pre-registered candidate loads the pinned base and Adapter in FP32, performs
safe merge in FP32, and retains FP32 for greedy SDPA inference on frozen
`eval-001`. Two fresh candidate loads must agree with each other and match the
frozen independent BF16 Adapter output exactly before any full-eval run or
artifact promotion.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_bf16_merge_numerics.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-bf16-merge-numerics-v1.json

python -I .\scripts\validate_offline.py
```
