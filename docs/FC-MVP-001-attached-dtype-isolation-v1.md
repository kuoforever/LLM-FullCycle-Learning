# FC-MVP-001 attached dtype isolation v1

> **Result: COMPLETE LOCALLY — four fresh ABBA-ordered attached-Adapter runs
> reproduce the frozen BF16 and FP32 paths and remain exactly repeat-stable
> within each dtype. With the execution form fixed, the paths share 45
> generated tokens and then select `true` versus `false` from different raw
> LM-head argmax values at index 45. The dtype-isolation gate passes;
> remediation and Runtime eligibility remain false.**

## Locked experiment

This gate holds the factorized attached-LoRA execution form fixed and changes
only the base/inference dtype condition. The fresh-load order is ABBA:

1. `bf16-attached-dtype-r1`;
2. `fp32-attached-dtype-r1`;
3. `fp32-attached-dtype-r2`;
4. `bf16-attached-dtype-r2`.

Both paths load the same BF16 checkpoint source values and the same FP32
Adapter, keep the Adapter attached with `merge=false`, retain 112 LoRA target
modules and 224 LoRA parameter tensors, and run the same frozen `eval-001`
input and generation protocol. The checkpoint storage remains BF16 in both
conditions:

- `bf16_attached_adapter` materializes the base and performs base inference in
  BF16 while the attached Adapter remains FP32;
- `fp32_attached_adapter` widens the same stored BF16 checkpoint values into an
  FP32 base and performs base inference in FP32 while the same attached Adapter
  remains FP32.

This is therefore an attached-path base/inference dtype comparison, not an
all-BF16 versus all-FP32 comparison and not a comparison with a pristine FP32
checkpoint.

Both path records set `autocast_adapter_dtype=true`. This locks PEFT's
load-time policy, which would upcast FP16/BF16 Adapter weights to FP32; the
stored Adapter values in this experiment are already FP32 and remain FP32.
The option is not generation autocast. The generated forward runs with
`autocast=false`, TF32 disabled, greedy decoding, the locked repetition
penalty and EOS/PAD IDs, `use_cache=true`, and the same high-level Transformers
`sdpa` dispatch.

The model, Adapter, training lock and evidence, prompt, 339-token rendered
input, eval digest, seed, decoding, attention configuration, and source
lineage remain pinned. No merged path is executed, and no model or Adapter
artifact is modified.

## Fresh path reproduction and repeat stability

| Run | Base/inference dtype | Tokens | Token digest | Output digest | Peak GPU bytes |
|---|---|---:|---|---|---:|
| `bf16-attached-dtype-r1` | BF16 + FP32 Adapter | 48 | `sha256:e23b3f5e…4e173` | `sha256:b3bef0f2…0bc5` | `3,186,198,528` |
| `fp32-attached-dtype-r1` | FP32 + FP32 Adapter | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,285,127,680` |
| `fp32-attached-dtype-r2` | FP32 + FP32 Adapter | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,285,127,680` |
| `bf16-attached-dtype-r2` | BF16 + FP32 Adapter | 48 | `sha256:e23b3f5e…4e173` | `sha256:b3bef0f2…0bc5` | `3,186,198,528` |

Within each dtype, the two fresh runs are identical in all 48 generated
tokens, decoded output, full processed-score and raw-logit trace digests, all
48 per-step comparison-vector digests, index-45 comparison vectors, precision
audit, target-forward alignment, and LM-head output linkage. Both paths also
reproduce their frozen token, output, trace, target-vector, and boundary-token
references.

The BF16 and FP32 token digests are respectively
`sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173`
and
`sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca`.

## Frozen token boundary

Both paths generate 48 tokens and share exactly the first 45 generated tokens.
Their first divergence is the pre-registered zero-based index `45`:

- BF16 attached emits token `1866` (`true`);
- FP32 attached emits token `3849` (`false`).

The compared cached forward consumes generated token index `44`, token `788`,
with `past_length=383` and `cache_position=[383]`. All four runs record call
and generation index `45`, input shape `[1, 1]`, position ID `[383]`, and 48
causal forward calls. The captured `lm_head` output is exactly linked to
`generated.logits[45]` after canonical FP32 comparison.

The gate does not search for a later or more favorable step after observing
the result. It requires the full frozen token paths and the original index-45
boundary to reproduce before assigning a dtype classification.

## Processed generation scores at index 45

`generated.scores[45]` contains the prediction scores after the configured
logits processors.

| Attached path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| BF16 base/inference | `true`: `34.54545211791992` | `false`: `34.09090805053711` | `0.4545440673828125` |
| FP32 base/inference | `false`: `35.61114501953125` | `true`: `33.16929626464844` | `2.4418487548828125` |

Both processed-score argmax values match the emitted tokens. All `151,936`
elements of the canonical FP32 comparison vectors differ. Maximum absolute
delta is `1.943955421447754`, mean absolute delta is
`0.2276017963886261`, and RMS delta is `0.29327404499053955`.

The BF16 and FP32 processed-vector digests are respectively
`sha256:5b78c36066365bb9c52a4894b6f642006fe891552ebc0d6a294f82aa9a8a80db`
and
`sha256:47055d7f7614955154ce736de5fd79b8e1636aacb80e214377f7faa6e4767451`.

## Raw LM-head logits at index 45

The same cached `generate(use_cache=true)` call returns unprocessed LM-head
prediction scores in `generated.logits[45]`.

| Attached path | Rank 1 | Rank 2 | Top-1 margin |
|---|---:|---:|---:|
| BF16 base/inference | `true`: `38.0` | `false`: `37.5` | `0.5` |
| FP32 base/inference | `false`: `39.17226028442383` | `true`: `36.48622512817383` | `2.68603515625` |

The raw argmax already flips from `true` to `false`, before the processed-score
selection. Again, all `151,936` comparison-vector elements differ. Maximum
absolute delta is `1.943955421447754`, mean absolute delta is
`0.22759762406349182`, and RMS delta is `0.2932831645011902`.

The BF16 and FP32 raw-vector digests are respectively
`sha256:aa7ae2fab3c2be5b0ddeecb7e4a10d01dcfd8636a6a404d7e48e9ef19eb9bf9e`
and
`sha256:14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a`.

These observations support the classification
`deterministic_bf16_attached_vs_fp32_attached_raw_logit_boundary_flip`.

## Precision, lifecycle, and resource audit

The checkpoint contains 338 BF16 tensors and `1,543,714,304` elements. The
Adapter contains 224 FP32 tensors and `4,358,144` elements. The BF16 condition
retains BF16 base parameters plus FP32 Adapter parameters; the FP32 condition
retains FP32 base and Adapter parameters. All Adapter tensors are finite, all
112 LoRA dropout modules are in evaluation mode, embeddings remain tied, and
both conditions retain `Qwen2Attention` under the same high-level `sdpa`
configuration.

Returned processed scores and raw logits are finite FP32 tensors with shape
`[1, 151936]` at each of 48 steps. Their returned dtype does not prove that all
internal BF16-path operations execute in FP32 or that both dtype conditions
select the same low-level CUDA kernels.

The complete four-run probe takes `37.87893820001045` seconds and peaks at
`6,285,127,680` allocated GPU bytes. The first lifecycle begins with zero
allocated CUDA bytes. Every lifecycle releases to `8,519,680` bytes, below
the locked 16 MiB residual ceiling.

## JSON-only evidence and strict validation

The artifact is JSON-only. It intentionally contains no raw target vectors,
module-tensor payloads, or tensor sidecar. Each run instead binds:

- the complete 48-token sequence and recomputable token digest;
- native full-trace digests for processed scores and raw logits;
- an ordered 48-entry manifest of canonical FP32 vector digests for each
  trace;
- the exact index-45 vector digests and exposed top-k scalar evidence;
- precision, target-forward, LM-head linkage, lifecycle, and resource records.

The strict standard-library evidence validator closes the exact top-level,
source-lineage, Adapter, environment, storage, protocol, run, trace,
target-alignment, classification, gate, resource, and frozen-policy schemas.
It validates four runs, recomputes six token digests, validates eight 48-step
comparison manifests, recomputes within-dtype repeat identity and frozen-path
reproduction, verifies target-vector and LM-head/raw-logit links, reconstructs
the token boundary and classification, and checks the exposed top-k values,
margins, ranks, and decision contrasts.

The full-vector delta statistics are probe-derived summary algebra only. The
validator checks finiteness, nonzero-density and RMS/mean/max algebraic bounds,
vector-identity consistency, and lower bounds implied by exposed token values.
Because the artifact stores neither the raw vectors nor a sidecar, the
validator cannot independently recompute all `151,936` elementwise deltas or
their exact aggregate statistics. The report does not claim otherwise.

The canonical artifact is
[`baseline/fc-mvp-001-attached-dtype-isolation-v1.json`](../baseline/fc-mvp-001-attached-dtype-isolation-v1.json),
with SHA-256
`7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31`.

## Gate result and causal boundary

`dtype_isolation_gate.passed=true`. No remediation is tested, so
`remediation_gate.passed=false`; `runtime_eligible=false`, no merged artifact
is saved or allowed, and the Adapter remains Runtime ineligible.

The evidence supports a repeat-stable total effect of changing the attached
path's base/inference dtype from BF16 to FP32 on this one frozen generation
path. The same BF16 checkpoint source values, FP32 Adapter source/runtime
values, attached factorized-LoRA form, input and token prefix, greedy decoding,
high-level SDPA dispatch, and fresh-load lifecycle are controlled.

It does not establish:

- an all-BF16 versus all-FP32 comparison;
- a comparison against pristine FP32 checkpoint values;
- the earliest temporal or module-level numerical divergence;
- a unique rounding, floating-point, PEFT, or CUDA-kernel root cause;
- that index 45 is where the accumulated dtype effect originates;
- improved correctness, safety, or model quality from either token;
- full-eval generalization, merged-artifact promotion, or Runtime eligibility.

The cached state at index 45 has accumulated the preceding dtype-conditioned
forwards. This gate therefore classifies the total effect along the locked
attached generation path, not an isolated one-step local compute effect.

No data was added, no training or eval-answer tuning occurred, and only frozen
`eval-001` ran. Runtime, Provider, MCP, and Desktop remain disconnected.

## Validation

The unified offline gate passes 144 tests on CPython 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, py_compile passes the new source/script/
test files, and mypy 2.3.0 reports no issues in all 43 source/script files.

## Single next objective

`FC-MVP-001-attached-dtype-numerics-v1` must reproduce both fresh attached
paths and the same frozen target forward, then locate the first difference in
a pre-registered paired module plan and quantify propagation of the dtype
effect. It must keep the attached execution form, source values, Adapter,
input, target step, backend, and decoding locked, and it must not claim a
unique low-level root cause from registered module evidence alone.

No new data, training, full eval, Runtime integration, merged-artifact save or
promotion, or module tensor sidecar is authorized by this isolation result.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_attached_dtype_isolation.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --drift-evidence .\baseline\fc-mvp-001-fp32-merge-drift-analysis-v1.json `
  --isolation-evidence .\baseline\fc-mvp-001-fp32-attached-merge-isolation-v1.json `
  --fp32-numerics-evidence .\baseline\fc-mvp-001-fp32-attached-merge-numerics-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-attached-dtype-isolation-v1.json

python -I .\scripts\validate_offline.py
```
