# FC-MVP-001 FP32 attached/merge isolation v1

> **Result: COMPLETE LOCALLY — two fresh FP32 attached-Adapter runs are
> exactly repeat-stable, the unchanged FP32 safe-merged path reproduces, and
> all three runs emit the same tokens and decoded output. Attached and merged
> FP32 nevertheless have deterministic numerical trace drift at the frozen
> comparison step. The isolation gate passes; remediation and Runtime
> eligibility remain false.**

## Locked isolation

This gate changes one comparison factor from the prior drift analysis: both
new paths materialize the same pinned BF16 checkpoint values as FP32 and retain
FP32 base and Adapter values for inference.

The fixed execution order is:

1. fresh FP32 base plus attached FP32 Adapter, repeat 1;
2. fresh FP32 base plus attached FP32 Adapter, repeat 2;
3. the unchanged FP32 safe-merged candidate, repeat 1.

The attached path loads the frozen Adapter with
`autocast_adapter_dtype=false` and never merges it. The candidate uses the
previously frozen FP32 `safe_merge` implementation without modification. Each
run receives a fresh model-load lifecycle, and only CPU copies of generated
tokens and numerical traces survive between paths.

The model checkpoint, Adapter files, 339-token rendered `eval-001` input,
prompt digest, eval digest, seed, greedy decoding, repetition penalty 1.1,
EOS/PAD IDs, `use_cache=true`, high-level Transformers `sdpa` dispatch, and
generation call that returns both processed scores and raw logits remain
locked. The frozen BF16 results are context only and were not rerun.

## Fresh path reproduction

| Run | Form | Tokens | Token digest | Output digest | Peak GPU bytes |
|---|---|---:|---|---|---:|
| `fp32_attached-r1` | FP32 attached Adapter | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,285,651,968` |
| `fp32_attached-r2` | FP32 attached Adapter | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,285,127,680` |
| `fp32_safe_merged-r1` | FP32 safe-merged | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,268,076,032` |

The two attached runs are identical in generated tokens, decoded output, all
48 processed-score vectors, all 48 raw-logit vectors, precision audit, and the
two index-45 comparison vectors. Their processed-score trace digest is
`sha256:e878f06653e43ebf6946a00396fbed7797eecc02dcf25501f0738169a932fdde`;
their raw-logit trace digest is
`sha256:61a891ab427bce3002c3367e2faefd854a11ecb62929d5b187b974a9c3b7f357`.

The fresh merged run exactly reproduces the frozen token/output digests, full
processed-score trace
`sha256:1b7b93ba7cba872cfb8dd4d50e452df8fb76b76e4ddda5c513ec9697e67e1fe9`,
full raw-logit trace
`sha256:a0a03e4bf73123db942a0e44122b3d312d3885aa195a49782d0dd92a9d4a65ee`,
and both frozen index-45 vector digests.

## Same-dtype token result

The FP32 attached and FP32 merged paths have a common prefix of all 48
generated tokens. There is therefore no same-dtype generated-token boundary
in this run. Both emit token `3849` (`false`) at zero-based index `45`, and
both finish with the same decoded output.

Index `45` remains the pre-registered comparison step because it is the frozen
BF16 contextual boundary. The gate does not search for a more favorable step
after seeing the FP32 result. In frozen BF16 context, the attached path emitted
`true` and the safe-merged path emitted `false` at this index; those BF16 GPU
paths were not rerun here.

## Processed generation scores at index 45

`generated.scores[45]` contains scores after configured logits processors.

| FP32 path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| Attached Adapter | `false`: `35.61114501953125` | `true`: `33.16929626464844` | `2.4418487548828125` |
| Safe-merged | `false`: `35.61100387573242` | `true`: `33.169429779052734` | `2.4415740966796875` |

Both argmax values align with the emitted token. Across the `151,936`-element
vocabulary vector, `150,968` elements differ. Maximum absolute delta is
`0.0001735687255859375`, mean absolute delta is
`0.00002052841409749817`, and RMS delta is
`0.000026467831048648804`.

The attached and merged comparison-vector digests are respectively
`sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451`
and
`sha256:c645fd357a4d34fc94dab70978e90143886051e898edfc0049ad67a370a14d8b`.

## Raw LM-head logits at index 45

The same cached `generate(use_cache=True)` call returns unprocessed LM-head
prediction scores in `generated.logits[45]`.

| FP32 path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| Attached Adapter | `false`: `39.17226028442383` | `true`: `36.48622512817383` | `2.68603515625` |
| Safe-merged | `false`: `39.17210388183594` | `true`: `36.48637390136719` | `2.68572998046875` |

Again, `150,968` of `151,936` elements differ. Maximum absolute delta is
`0.0001735687255859375`, mean absolute delta is
`0.00002052839772659354`, and RMS delta is
`0.000026469620934221894`.

The attached and merged raw-vector digests are respectively
`sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a`
and
`sha256:87d3bee7986814bb3a9bb22b249247235cc52e8444ba96b22b82b409ed1e0c93`.

## Precision, lifecycle, and resource audit

The attached path contains `338` FP32 base parameter tensors and `224` FP32
Adapter tensors over `112` LoRA target modules. All `112` LoRA dropout modules
are in evaluation mode. The merged path has the same FP32 tensor inventory
before merge and `338` FP32 base tensors with zero LoRA tensors after merge.
Both paths retain tied input/output embeddings, use `Qwen2Attention` through
the high-level Transformers `sdpa` dispatch, and disable autocast and TF32.

Returned processed scores and raw logits are finite FP32 tensors with shape
`[1, 151936]` at each of 48 steps. Their return dtype is evidence about the
returned tensors, not proof of every internal operation's compute dtype or of
low-level CUDA kernel identity.

Allocated CUDA memory begins at zero for the first run and releases to
`8,519,680` bytes. The next two runs both begin and end at `8,519,680`, below
the locked 16 MiB residual ceiling. The complete three-path probe takes
`29.405898299999535` seconds and peaks at `6,285,651,968` allocated GPU bytes.

## Gate result and causal boundary

The classification is
`deterministic_fp32_attached_vs_merged_numerical_drift_without_token_drift`.
It proves the observed attached and materialized FP32 execution forms can
produce small, repeatable numerical differences under the locked case while
preserving the entire generated token sequence and output.

It does not prove a PEFT merge bug, identify the first divergent module,
identify a low-level CUDA kernel or root cause, generalize to the full eval,
repair the frozen BF16 behavior, authorize a merged artifact, or establish
Runtime eligibility. In particular, the result must not be described as a
same-dtype token boundary because none exists.

`isolation_gate.passed=true`, but `remediation_gate.passed=false` and
`runtime_eligible=false`. No data was added, no training or eval-answer tuning
occurred, only `eval-001` ran, no merged weights were saved, and Runtime,
Provider, MCP, and Desktop remain disconnected.

The canonical evidence is
[`baseline/fc-mvp-001-fp32-attached-merge-isolation-v1.json`](../baseline/fc-mvp-001-fp32-attached-merge-isolation-v1.json),
with SHA-256
`37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26`.

The unified offline gate passes 106 tests on Python 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, and mypy 2.3.0 reports no issues in all 37
source/script files.

## Single next objective

`FC-MVP-001-fp32-attached-merge-numerics-v1` must reproduce both FP32 paths
and the frozen index-45 comparison step, then use paired execution-order hooks
to locate the first module-level numerical divergence between repeat-stable
attached LoRA execution and the unchanged materialized safe-merged execution.
It must quantify the operation-order boundary without claiming a same-dtype
token boundary.

The source checkpoint values, Adapter, failed candidate, generation backend,
decoding, input, and artifact prohibitions remain locked. The BF16 paths stay
context-only. No new data, training, full eval, Runtime integration, or merged
artifact promotion is allowed.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_fp32_attached_merge_isolation.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --numerics-evidence .\baseline\fc-mvp-001-bf16-merge-numerics-v1.json `
  --remediation-evidence .\baseline\fc-mvp-001-bf16-merge-remediation-v1.json `
  --drift-evidence .\baseline\fc-mvp-001-fp32-merge-drift-analysis-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-fp32-attached-merge-isolation-v1.json

python -I .\scripts\validate_offline.py
```
