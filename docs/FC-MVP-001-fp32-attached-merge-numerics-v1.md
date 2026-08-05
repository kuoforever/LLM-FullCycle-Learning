# FC-MVP-001 FP32 attached/merge numerics v1

> **Result: COMPLETE LOCALLY — four fresh ABBA-ordered FP32 runs reproduce
> both frozen paths and remain bitwise repeat-stable within each path. Within
> the pre-registered paired module capture plan, the first unequal output is
> layer 0 `q_proj`. Raw tensor replay confirms deterministic drift between the
> factorized attached-LoRA and materialized-linear execution forms. The
> numerics gate passes; remediation and Runtime eligibility remain false.**

## Locked experiment

This gate keeps the FP32 attached and FP32 safe-merged paths from the prior
isolation gate unchanged. It reruns the exact 339-token `eval-001` prompt and
captures the cached forward that predicts zero-based generated token index
`45`: input generated token index `44` is token `788`, the cache/past position
is `383`, and both paths emit token `3849` (`false`).

The fresh-load order is ABBA:

1. `fp32-attached-numerics-r1`;
2. `fp32-safe-merged-numerics-r1`;
3. `fp32-safe-merged-numerics-r2`;
4. `fp32-attached-numerics-r2`.

The model checkpoint, Adapter, prompt and eval digests, seed, greedy decoding,
repetition penalty, EOS/PAD IDs, `use_cache=true`, high-level Transformers
`sdpa` dispatch, disabled autocast/TF32, and FP32 model/Adapter parameter and
captured-tensor dtypes are locked. Each run has a fresh model-load lifecycle.
Only archived CPU evidence is intentionally retained between paths; the fixed
process and bounded `8,519,680`-byte CUDA allocator residual remain. No BF16
path is rerun by this gate.

## Path reproduction and repeat stability

| Run | Form | Tokens | Token digest | Output digest | Peak GPU bytes |
|---|---|---:|---|---|---:|
| `fp32-attached-numerics-r1` | Attached factorized LoRA | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,286,505,472` |
| `fp32-safe-merged-numerics-r1` | Materialized linear | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,268,910,592` |
| `fp32-safe-merged-numerics-r2` | Materialized linear | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,268,910,592` |
| `fp32-attached-numerics-r2` | Attached factorized LoRA | 48 | `sha256:9dfd817e…43dca` | `sha256:b37939d2…5cc7ca` | `6,285,981,184` |

All four runs reproduce their frozen full token, decoded-output,
processed-score-trace, raw-logit-trace, target-vector, and target-forward
references. Within each path, all captured tensor payloads are bitwise
repeat-stable and event-sequence digests match. Across paths, all 48 tokens and
decoded output remain identical while processed scores and raw logits retain
the frozen numerical drift. This gate therefore has no same-dtype token
boundary.

The complete probe takes `29.912574299960397` seconds and peaks at
`6,286,505,472` allocated GPU bytes. The residual allocated memory after each
fresh lifecycle remains `8,519,680` bytes, below the locked 16 MiB ceiling.

## Paired common-module boundary

The pre-registered capture plan observes one tensor leaf at the input and
output of 13 common semantic stages in execution order. Each run records 26
common input/output events. It covers the embedding, layer 0 normalization,
Q/K/V/O projections, the layer 0 MLP and layer output, final normalization,
and `lm_head`.

The first paired unequal output is capture-plan index `2`:
`model.layers.0.self_attn.q_proj`. The preceding `model.embed_tokens` and
`model.layers.0.input_layernorm` outputs are bitwise identical, and the
`q_proj` inputs are bitwise identical.

| Paired output | Different / total elements | Maximum absolute delta | Mean absolute delta | RMS delta |
|---|---:|---:|---:|---:|
| Layer 0 `q_proj` | `1,261 / 1,536` | `3.814697265625e-06` | `1.6505699325837972e-07` | `2.990363292409807e-07` |
| `lm_head` at step 45 | `150,968 / 151,936` | `0.0001735687255859375` | `2.0528396424031623e-05` | `2.6469620496529053e-05` |

This is the first divergence inside the registered paired capture plan, not a
claim that it is the first temporal difference in the full 48-step generation
history or the first unregistered functional operation.

## Registered `q_proj` execution forms

The attached operation sequence is audited as:

```text
base_layer -> dropout_output -> lora_a_output -> lora_b_output
```

The locked Adapter has rank `16`, alpha `32`, scaling `2.0`, and dropout
probability `0.05`; all dropout modules are in evaluation mode. The compared
execution forms are:

```text
attached: base_linear(x) + lora_B(lora_A(dropout(x))) * scale
merged:   linear(x, base_weight + (lora_B_weight @ lora_A_weight) * scale)
```

The probe-captured linear replays match both actual outputs exactly. The
stdlib validator recomputes their comparisons and the registered scalar/add
identities, separating two counterfactual numerical differences that remain
nonzero at the `q_proj` output boundary:

| Replay comparison | Different elements | Maximum absolute delta | Interpretation |
|---|---:|---:|---|
| `q_proj` input vs attached dropout output | `0` | `0` | Eval-mode dropout is identity |
| Attached actual vs captured factorized replay | `0` | `0` | Captured attached replay matches actual output |
| Merged actual vs captured materialized-linear replay | `0` | `0` | Captured in-memory linear replay matches actual output |
| Expected materialized vs actual merged output | `0` | `0` | In-memory safe-merged output is reproduced |
| Factorized LoRA term vs delta-weight linear | `1,355` | `3.259629011154175e-09` | Factorized and pre-multiplied delta paths differ |
| Factorized post-add vs split-delta post-add | `53` | `5.960464477539063e-08` | The first axis survives base addition |
| Split base-plus-delta vs materialized-weight linear | `1,260` | `3.814697265625e-06` | The replay differs at the materialized-weight boundary |
| Attached actual vs merged actual | `1,261` | `3.814697265625e-06` | Observed paired boundary |

The classification is
`deterministic_fp32_factorized_lora_and_materialized_linear_execution_form_drift`.
“Execution-form drift” is intentionally narrower than “operation-order root
cause”: the evidence links the registered graphs to captured replay tensors,
verifies scalar/add identities, and compares their observed differences, but
the stdlib validator does not independently recompute CUDA `F.linear`/GEMM,
identify a low-level CUDA kernel, or prove a unique floating-point mechanism.
The counterfactual branches are not individually
injected through the network after `q_proj`, so their independent propagation
beyond this boundary is not isolated.

## Weight materialization audit

For layer 0 `q_proj`, the archive stores all `2,359,296` elements of the base,
delta, expected merged, and actual merged weights, plus both 1,536-element
bias vectors. The validator recomputes FP32 `base + delta` from the raw
records.

- Expected and actual merged weights are bitwise identical, with digest
  `sha256:06e176006e6eb480f9f119af53e27b227d15abf9c0bc86da134d9591f3c5496c`.
- Actual merged-weight mismatches and bias mismatches are both zero.
- `2,359,275` weights change effectively; 21 nonzero archived FP32 delta
  updates round back to the base FP32 value.
- Maximum and mean absolute materialization error are
  `4.062894731760025e-08` and `7.122678848880609e-10`.

This verifies the representative safe-merge materialization. It does not turn
the small rounding count into a complete explanation of the output delta.

## Raw tensor archive and fail-closed validation

The JSON manifest binds a Git LFS-managed little-endian IEEE-754 float32
sidecar containing 138 records and `46,069,904` bytes. It includes all four
runs' captured activations, operation-graph diagnostics, target processed
scores and raw logits, and the representative weights and biases.

The standard-library validator checks exact descriptor keys, sidecar basename,
size and digest; unique record and semantic identities; contiguous exhaustive
offsets; shapes and finite values; per-record raw, canonical, and
metadata-bound digests; run/event/comparison closure; all 35 paired and replay
comparisons; ABBA repeat stability; LM-head/raw-logit linkage; scalar/add replay
reconstruction plus captured linear-replay linkage; and the full representative
weight audit. All 138 record provenance and tensor descriptors, run schemas,
FP32 precision inventories, generation-trace structures, materialization forms,
and per-run resources are exact-closed. Mutation tests cover descriptor,
range, duplicate, rebound metadata, payload, NaN, event, comparison, repeat,
generation-link, trace, precision, resource, weight, target-alignment,
acceptance, and final-gate forgeries.

Canonical artifacts:

- [`baseline/fc-mvp-001-fp32-attached-merge-numerics-v1.json`](../baseline/fc-mvp-001-fp32-attached-merge-numerics-v1.json),
  SHA-256 `cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91`;
- `baseline/fc-mvp-001-fp32-attached-merge-numerics-v1-tensors.bin`,
  SHA-256 `550175dfcfe14b0739aabf17573825a124180a6e21826e25d4b5ff733fb298a9`.

## Gate result and causal boundary

`numerics_gate.passed=true`, but `remediation_gate.passed=false` and
`runtime_eligible=false`. The evidence supports only the first divergence in
the pre-registered paired common-module capture plan at frozen step `45` and
the registered FP32 `q_proj` replays.

It does not support claims about:

- the earliest temporal divergence across the full generation history;
- every tensor leaf or unregistered functional boundary;
- a same-dtype token boundary;
- a specific CUDA kernel or unique root cause;
- pure non-associativity as the only mechanism;
- independent propagation of either counterfactual beyond `q_proj`;
- a PEFT implementation bug;
- full-eval generalization, merged-artifact promotion, or Runtime eligibility.

No data was added, no training or eval-answer tuning occurred, and only the
frozen case ran. No full or deployable merged model artifact was saved or
promoted; only the registered full `q_proj` merged-weight snapshot is retained
as diagnostic evidence. Runtime, Provider, MCP, and Desktop remain
disconnected.

The unified offline gate passes 124 tests on Python 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, py_compile passes the new source/test
files, and mypy 2.3.0 reports no issues in all 40 source/script files.

## Single next objective

`FC-MVP-001-attached-dtype-isolation-v1` will hold the Adapter-attached
execution form fixed and compare fresh repeat-stable BF16 and FP32 paths on the
same frozen `eval-001` cached step `45`. Its purpose is to isolate the
remaining token-level precision effect without changing merge form. The
source model, Adapter, prompt, eval digest, backend, decoding, and artifact
prohibitions remain locked; no new data, training, full eval, Runtime
integration, or merged-artifact promotion is allowed.

## Reproduction

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_fp32_attached_merge_numerics.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --training-evidence .\baseline\fc-mvp-001-lora-sft-v2-training.json `
  --stability-evidence .\baseline\fc-mvp-001-bf16-merge-stability-v1.json `
  --numerics-evidence .\baseline\fc-mvp-001-bf16-merge-numerics-v1.json `
  --remediation-evidence .\baseline\fc-mvp-001-bf16-merge-remediation-v1.json `
  --drift-evidence .\baseline\fc-mvp-001-fp32-merge-drift-analysis-v1.json `
  --isolation-evidence .\baseline\fc-mvp-001-fp32-attached-merge-isolation-v1.json `
  --tensor-output .\work\test-fixtures\fc-mvp-001-fp32-attached-merge-numerics-v1-tensors.bin `
  --output .\work\test-fixtures\fc-mvp-001-fp32-attached-merge-numerics-v1.json

python -I .\scripts\validate_offline.py
```
