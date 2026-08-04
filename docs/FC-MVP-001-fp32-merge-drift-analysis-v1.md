# FC-MVP-001 FP32 merge drift analysis v1

> **Result: COMPLETE LOCALLY — both frozen paths reproduce, and the first
> generated-token boundary contains a raw-logit argmax flip. The experiment
> does not isolate dtype from attached-versus-merged execution; the failed
> candidate remains Runtime ineligible.**

## Locked comparison

The probe runs exactly one new model-load lifecycle for each already stable
path, in fixed order:

1. the frozen independent BF16 base plus attached FP32 Adapter;
2. the failed candidate that materializes the BF16 checkpoint values as FP32,
   attaches the FP32 Adapter, safe-merges it in FP32, and retains FP32 for
   inference.

Upstream evidence already established two-run repeat stability separately for
the independent BF16 path and the FP32 candidate. This gate therefore requires
each new path to reproduce its frozen token and decoded-output digests before
it permits any cross-path analysis.

The prompt, tokenizer, rendered 339-token `eval-001` input, seed, greedy
decoding, repetition penalty 1.1, EOS/PAD IDs, `use_cache=true`, SDPA dispatch,
Adapter, checkpoint, and source evidence remain locked. No path changes its
previously registered dtype or merge behavior.

## Fresh reproduction

| Path | Prior stability | New tokens | Token digest | Output digest |
|---|---:|---:|---|---|
| Independent BF16 + attached Adapter | 2 frozen runs | 48 | `sha256:e23b3f5e…54e173` | `sha256:b3bef0f2…f0bc5` |
| FP32 safe-merged candidate | 2 frozen runs | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` |

Both new runs match their respective frozen token and output digests. They
share the first 45 generated tokens and first differ at zero-based generated
token index `45`: the independent BF16 path emits token `1866` (`true`), while
the FP32 safe-merged path emits token `3849` (`false`).

This is the first **generated-token boundary**. The probe does not claim that
the two logit vectors are numerically identical at every earlier generation
step, because it does not compare those earlier vectors element by element.

## Processed generation scores

`generated.scores[45]` is the processed prediction vector used by greedy
selection after Transformers applies configured `LogitsProcessor` operations.
It is not a raw model logit vector.

| Path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| Independent BF16 + attached Adapter | `true`: `34.54545211791992` | `false`: `34.09090805053711` | `0.4545440673828125` |
| FP32 safe-merged candidate | `false`: `35.61100387573242` | `true`: `33.169429779052734` | `2.4415740966796875` |

Across all `151,936` vocabulary elements at this one step, every processed
score differs. The maximum absolute delta is `1.9437971115112305`, the mean
absolute delta is `0.2275839000940323`, and RMS delta is
`0.29325070977211`.

## Raw LM-head logits

The same cached `generate(use_cache=True)` call also requests
`output_logits=True`. Transformers defines `generated.logits` as the
unprocessed LM-head prediction scores, before logits processors.

| Path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| Independent BF16 + attached Adapter | `true`: `38.0` | `false`: `37.5` | `0.5` |
| FP32 safe-merged candidate | `false`: `39.17210388183594` | `true`: `36.48637390136719` | `2.68572998046875` |

The raw vectors therefore already contain the argmax flip. Both compared
tokens had appeared in the common generated prefix, and their positive logits
are divided by the unchanged repetition penalty without reversing either
path's ordering. The logits processor does not create this observed flip.

At this step, all `151,936` raw-logit elements differ. Maximum absolute delta
is `1.9437971115112305`, mean absolute delta is
`0.22757971286773682`, and RMS delta is `0.2932598292827606`.

The classification is
`deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip`. Its name
describes the two measured paths rather than assigning a root cause.

## Precision, trace, and lifecycle audit

The independent path contains `338` BF16 base parameter tensors and `224`
FP32 Adapter tensors across `112` LoRA targets. The candidate contains `338`
FP32 base tensors and `224` FP32 Adapter tensors before merge, then `338` FP32
base tensors and no LoRA tensors after merge. Both paths retain tied
input/output embeddings, use `Qwen2Attention` through the Transformers `sdpa`
dispatch, disable autocast and TF32, and return 48 finite score vectors plus 48
finite raw-logit vectors of shape `[1, 151936]`.

The recorded `native_dtypes=["float32"]` applies to the tensors returned by
Transformers generation. It does not mean that the independent path's BF16
base parameters, attention, or LM-head computation ran in FP32. Likewise, the
matching SDPA dispatch and global enabled flags do not prove that BF16 and
FP32 selected the same low-level CUDA kernel.

Full-trace hashes and the index-45 comparison-vector hashes are linked for
each path and each source (`generated.scores` and `generated.logits`). Only CPU
copies cross path lifecycles. The independent path begins at zero allocated
CUDA bytes and releases to `8,519,680`; the candidate begins and ends at
`8,519,680`, below the locked 16 MiB residual ceiling. Peak allocated memory is
`3,186,198,528` bytes for the independent path and `6,268,076,032` bytes for
the FP32 candidate. The two-path probe takes `13.201921800035052` seconds.

Because this probe retains both raw logits and processed scores, its peak
memory should not be compared directly with the earlier remediation probe as
a performance regression.

## Gate result and boundary

The evidence satisfies `analysis_gate.passed=true`: both frozen paths
reproduce, the first generated-token boundary is located, the exact cached
step is captured, and processed-score argmax aligns with each emitted token.
It simultaneously records `remediation_gate.passed=false`, because the FP32
candidate still does not match the independent BF16 reference.

The comparison changes two factors at once: BF16 versus FP32 compute and
attached versus materialized Adapter execution. It therefore cannot attribute
the raw-logit difference to either factor alone. Distribution-wide drift at
index 45 also does not establish drift at every generation step.

No data was added, no training or eval-answer tuning occurred, only
`eval-001` ran, no merged weights were saved, and Runtime, Provider, MCP, and
Desktop remain disconnected. `merged_artifact_allowed=false` and
`runtime_eligible=false`.

The canonical evidence is
[`baseline/fc-mvp-001-fp32-merge-drift-analysis-v1.json`](../baseline/fc-mvp-001-fp32-merge-drift-analysis-v1.json),
with SHA-256
`ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3`.

The unified offline gate passes 95 tests on Python 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, and mypy 2.3.0 reports no issues in all 35
source/script files.

## Single next objective

`FC-MVP-001-fp32-attached-merge-isolation-v1` adds only a fresh independent
FP32 attached-Adapter control. It must establish two-run stability for that new
path, reproduce the unchanged FP32 safe-merged candidate, and compare the two
FP32 paths at the exact cached generation step. The frozen BF16 attached and
merged token controls remain available for context, but the new causal claim
is deliberately narrower: classify the same-dtype attached-versus-merged
effect.

The failed candidate, existing paths, backend, decoding, data, and artifact
prohibitions remain locked. Full eval and Runtime promotion are still outside
scope.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_fp32_merge_drift.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --numerics-evidence .\baseline\fc-mvp-001-bf16-merge-numerics-v1.json `
  --remediation-evidence .\baseline\fc-mvp-001-bf16-merge-remediation-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-fp32-merge-drift-analysis-v1.json

python -I .\scripts\validate_offline.py
```
