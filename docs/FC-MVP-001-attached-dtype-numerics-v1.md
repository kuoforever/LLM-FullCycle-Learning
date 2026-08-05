# FC-MVP-001 Attached Dtype Numerics v1 Evidence

Status: completed locally on 2026-08-05; Runtime ineligible.

## Result

The frozen experiment classifies the observed behavior as
`deterministic_attached_bf16_vs_fp32_registered_module_output_drift_reaching_lm_head`.
On the same attached factorized-LoRA execution form and frozen target forward,
the BF16-base path and FP32-base path are independently repeat-stable. Their
canonical FP32 embedding values are identical, while the first unequal output
inside an outcome-neutral, pre-registered 40-output plan is registered index
`1`, `model.layers.0.input_layernorm`. Every one of the 38 later registered
outputs remains unequal through the linked LM head.

This is a descriptive total-dtype delta trajectory. It does not establish an
independent causal propagation chain from RMSNorm, the first unregistered
operation, the earliest difference across generation history, or a unique
floating-point, CUDA, or PEFT root cause.

The canonical record is
[`baseline/fc-mvp-001-attached-dtype-numerics-v1.json`](../baseline/fc-mvp-001-attached-dtype-numerics-v1.json),
is `393,662` bytes, and has SHA-256
`de5b048a5d254f61ab3bef1ff23f1484b07808c86a1679bc0de4ee58e8c8d7c5`.

## Frozen source and treatment controls

The immediate source gate is
[`FC-MVP-001-attached-dtype-isolation-v1`](../baseline/fc-mvp-001-attached-dtype-isolation-v1.json),
locked by SHA-256
`7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31`.
The complete frozen evidence lineage is:

| Evidence | SHA-256 |
|---|---|
| BF16 merge stability | `82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc` |
| FP32 merge drift | `ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3` |
| FP32 attached/merge isolation | `37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26` |
| FP32 attached/merge numerics | `cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91` |
| LoRA SFT v2 training | `641b1a7ef3dc0de0d9f2124b9122cb2c4be46b42de9265d558ab6f5b25b41a30` |
| Attached dtype isolation | `7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31` |

The treatment changes only the attached path's base load and inference dtype:

| Control | BF16 attached path | FP32 attached path |
|---|---|---|
| Base checkpoint storage | BF16 | BF16 |
| Base load/inference dtype | BF16 | FP32 |
| Adapter storage/runtime dtype | FP32 / FP32 | FP32 / FP32 |
| Execution form | attached factorized LoRA | attached factorized LoRA |
| Merge | false | false |
| Generation autocast | false | false |
| Attention dispatch | high-level SDPA | high-level SDPA |

`autocast_adapter_dtype=true` is the locked PEFT load-time Adapter policy on
both paths; it is not generation autocast. The checkpoint contains 338 BF16
tensors and `1,543,714,304` BF16 elements. The Adapter contains 224 FP32
tensors and `4,358,144` FP32 elements.

The source input is frozen to `eval-001`, 339 input tokens, eval digest
`02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a`,
and input-token digest
`3bd24b9f36966889e543dda2aea25f5c0f29db40a8fccf1453d0657a06a4429f`.
No source checkpoint, Adapter values, prompt, target step, decoding parameters,
or backend control changed.

## Fresh ABBA reproduction and target alignment

Four fresh model-load lifecycles ran in fixed-process ABBA order. Both paths
reproduced their prior frozen 48-token output, processed-score trace, raw-logit
trace, comparison vectors, boundary token, and precision audit exactly.

| Order | Run | Path | Fresh | Target token | Capture manifest | Elapsed (s) | Peak allocated bytes | After release bytes |
|---:|---|---|---|---:|---|---:|---:|---:|
| 0 | `bf16-attached-numerics-r1` | BF16 attached | true | 1866 | `e40227233795e440fb9138c542a843d3e3915e0f985347355c64717af62ba630` | `8.47314750001533` | `3,186,647,040` | `8,519,680` |
| 1 | `fp32-attached-numerics-r1` | FP32 attached | true | 3849 | `f0893f13c404a433005c3a7a3ac9fc0e29bc1a47b95423b0d1b2667111b93163` | `7.4342252999777` | `6,286,024,192` | `8,519,680` |
| 2 | `fp32-attached-numerics-r2` | FP32 attached | true | 3849 | `f0893f13c404a433005c3a7a3ac9fc0e29bc1a47b95423b0d1b2667111b93163` | `7.58322359999875` | `6,286,024,192` | `8,519,680` |
| 3 | `bf16-attached-numerics-r2` | BF16 attached | true | 1866 | `e40227233795e440fb9138c542a843d3e3915e0f985347355c64717af62ba630` | `6.78154140000697` | `3,186,647,040` | `8,519,680` |

The compared target is generation step/cached causal call `45`, whose input is
generated-token index `44`, token ID `788`, with past length and cache position
`383`. Each captured run reports 48 causal forward calls and exact target
alignment. BF16 predicts token `1866` (`true`) and FP32 predicts token `3849`
(`false`) at this boundary, matching the upstream frozen references.

## Outcome-neutral 40-output plan

The capture plan was fixed before observing these outputs. Its selection basis
is `existing_layer0_causal_spine_plus_all_decoder_block_outputs`, and its scope
is `target_forward_pre_registered_attached_dtype_module_outputs`.

| Pre-registered group | Outputs |
|---|---:|
| `model.embed_tokens` | 1 |
| Layer 0 detailed spine: input RMSNorm; Q/K/V/O projections; post-attention RMSNorm; gate/up/down projections | 9 |
| Decoder block outputs `model.layers.0` through `model.layers.27` | 28 |
| `model.norm` | 1 |
| `lm_head` | 1 |
| Total | 40 |

The canonical plan SHA-256 is
`945dc2b468edf361b73189e7adf1f4ef61599da4fd942942591fdc13c073b38a`.
For each module, occurrence `0` and the first tensor leaf at the specified
output path are cloned on GPU, then summarized after generation as contiguous
canonical FP32. Signed zero is normalized, non-finite values are forbidden,
comparison is exact with no tolerance, and moments use fixed flatten order with
stdlib `math.fsum` float64 reductions.

The plan does not cover module inputs, every tensor leaf, unregistered
functional operations, internal modules in decoder layers 1 through 27, the
earliest difference over the preceding generation history, an isolated
dtype-conditioned KV-cache intervention, or low-level CUDA-kernel identity.

## Capture records and manifest locks

Each run emitted 40 summary-only capture records, for 160 records total. No
module tensor payload was serialized. The following locks close ordering,
path-repeat, and paired-comparison identity:

- capture-plan SHA-256 on all four runs:
  `945dc2b468edf361b73189e7adf1f4ef61599da4fd942942591fdc13c073b38a`;
- capture-event-sequence SHA-256 on all four runs:
  `875edb689b4afef1472a746953ea581de42c4a55b05d810dcbf3d8a05c870ef9`;
- BF16 capture-manifest SHA-256 on both BF16 repeats:
  `e40227233795e440fb9138c542a843d3e3915e0f985347355c64717af62ba630`;
- FP32 capture-manifest SHA-256 on both FP32 repeats:
  `f0893f13c404a433005c3a7a3ac9fc0e29bc1a47b95423b0d1b2667111b93163`;
- representative and repeat module-comparison manifest SHA-256:
  `f136842f6754030a07a29d7d5172ee6c1192e82b123f391a9768eb2ceac9befe`.

Within each dtype, native-payload digests, canonical-FP32 digests, capture
records, event order, and capture manifests are exact across repeats. The
paired 40-comparison manifest is also exact across representative and repeat
pairs.

The strict JSON validator returns four validated runs, 160 validated capture
records, four validated capture manifests, and 40 validated module
comparisons. It closes the source lineage, schema, ABBA identity, trace and
target links, capture plan and event order, repeat identity, comparison
manifest, LM-head links, gate state, and no-payload policy.

Its statistics scope is
`probe_derived_summary_algebra_and_frozen_manifest_only`. Because the artifact
contains digests and summaries rather than module tensors, the validator can
check exact digests, links, repeat identity, frozen manifests, and summary
algebra, but it cannot independently recompute intermediate full-tensor
different-element counts or moments. That is an explicit JSON-only evidence
limit, not a claim that raw module tensors were revalidated offline.

## First registered difference and selected trajectory

The embedding has different native output dtypes, BF16 and FP32, but all 1,536
values are identical after canonical FP32 conversion. Its two canonical vector
digests are both
`8ebfe9e9c6947d1295014958f313cd761f84c7bbaa23312d27aa7cdca0b01d3c`.

The first unequal registered output is index `1`,
`model.layers.0.input_layernorm`. All `1,536/1,536` elements differ. The table
shows that boundary, the selected detailed-layer endpoint, representative
decoder outputs, final norm, and LM head:

| Index | Registered output | Different / total | Max abs delta | Mean abs delta | RMS delta | RMS / first |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `model.embed_tokens` | `0 / 1,536` | `0` | `0` | `0` | `0` |
| 1 | `model.layers.0.input_layernorm` | `1,536 / 1,536` | `0.012537479400634766` | `0.0005440729593146898` | `0.0009270972900508952` | `1.0` |
| 10 | `model.layers.0` | `1,536 / 1,536` | `0.044994354248046875` | `0.005065801998474247` | `0.006520985730796157` | `7.033766359556689` |
| 23 | `model.layers.13` | `1,536 / 1,536` | `0.09090137481689453` | `0.011829206477462625` | `0.01581094963521535` | `17.054250729551125` |
| 37 | `model.layers.27` | `1,536 / 1,536` | `1.5991058349609375` | `0.26510887151622836` | `0.33549556305265993` | `361.87740666812016` |
| 38 | `model.norm` | `1,536 / 1,536` | `1.3796119689941406` | `0.21995875258350375` | `0.283224604729097` | `305.49609816414056` |
| 39 | `lm_head` | `151,936 / 151,936` | `1.943955421447754` | `0.22759762689943575` | `0.29328314971734404` | `316.34560133515606` |

In total, 39 of 40 registered outputs are unequal, including the first one;
all 38 registered outputs downstream of it are unequal. The ratios summarize
cross-stage RMS magnitudes under different tensor shapes and operations. They
are not monotonic, and they are not amplification factors or estimates of
independent causal propagation.

## LM-head and frozen raw-logit linkage

The canonical LM-head vectors link exactly to the prior frozen target-forward
raw-logit vectors:

| Path | LM-head canonical vector / frozen raw-logit SHA-256 |
|---|---|
| BF16 attached | `aa7ae2fab3c2be5b0ddeecb7e4a10d01dcfd8636a6a404d7e48e9ef19eb9bf9e` |
| FP32 attached | `14b7b48cfb9012388762d0d9925c0c19ea737b7459bcf637ea31f880e731654a` |

Both per-run `lm_head_raw_logit_linked` flags are true. The frozen upstream
delta contains `151,936/151,936` nonzero elements and the same maximum absolute
delta, `1.943955421447754`. Its torch-reduced mean/RMS are
`0.22759762406349182` and `0.2932831645011902`; the current fixed-order
`math.fsum` reductions produce `0.22759762689943575` and
`0.29328314971734404`. The validator requires exact vector-digest, element,
nonzero-count, and maximum links, plus close mean/RMS links across the two
documented reduction methods.

This proves that registered index 39 is the frozen raw-logit vector being
discussed. It does not convert the preceding registered-output profile into an
interventional causal chain.

## KV-cache history and causal boundary

The evidence supports only the first exact canonical-FP32 inequality inside
the pre-registered 40-output plan at this frozen target forward, plus a
descriptive downstream registered total-dtype profile reaching the linked LM
head.

The target state already includes the prompt and the dtype-conditioned cached
generation prefix through input generated-token index 44. The current-forward
embedding values are canonically equal, but their native dtypes differ; the
probe did not feed one identical native tensor through RMSNorm under a bounded
intervention. Later attention and decoder outputs also consume accumulated
dtype-conditioned KV-cache history. Consequently, this record does not
separate current-forward arithmetic from historical cached state, locate the
earliest difference in generation history, or prove that the registered
`input_layernorm` output independently causes each later delta.

It also does not support a pristine-FP32-checkpoint comparison, a PEFT bug
claim, a unique floating-point/CUDA root cause, full-eval generalization,
artifact promotion, or Runtime eligibility. No data, training, full eval,
Runtime integration, module tensor sidecar, merged artifact, or second
intervention was added or authorized.

## Resources and lifecycle isolation

The probe ran under Python 3.12.12, PyTorch 2.6.0+cu124, Transformers 4.49.0,
PEFT 0.14.0, and Accelerate 1.3.0 on one NVIDIA GeForce RTX 4090 Laptop GPU
with compute capability 8.9 and `17,170,956,288` reported VRAM bytes.

Total elapsed time was `35.3808942000032` seconds. Peak allocated GPU memory
was `6,286,024,192` bytes. Each fresh lifecycle released to `8,519,680`
allocated bytes, below the locked `16,777,216`-byte residual ceiling. No
module tensor payload or sidecar was saved; no merged artifact was saved or
allowed.

## Gate, remediation, and Runtime status

`numerics_gate.passed=true`. Both paths are repeat-stable, all captures repeat
exactly, both frozen references reproduce, target alignment and the capture
plan pass, event order is identical, the paired comparison repeats exactly,
the first registered unequal output is located with its predecessor equal, and
the registered LM-head difference links to the frozen raw logits and delta.

No new remediation was tested, so `remediation_gate.passed=false` and
`new_remediation_tested=false`. `runtime_eligible=false`, with the frozen
classification above as the eligibility reason. The gate is diagnosis, not a
promotion decision.

## Validation

The unified offline gate passes 181 tests on CPython 3.11.15, 3.12.12, and
3.13.7. The targeted attached dtype numerics helper/evidence tests pass, Ruff
passes the repository, and mypy reports no issues in all 46 source/script
files.

## Single next objective

`FC-MVP-001-attached-dtype-boundary-control-v1` must pre-register one bounded
control that separates accumulated dtype-conditioned target-forward state from
current-forward computation at the observed first registered output. It must:

- reproduce fresh BF16/FP32 attached paths and the same frozen target forward;
- reproduce the layer-0 `input_layernorm` boundary and capture its input;
- run one pre-registered same-values RMSNorm control under two locked dtypes;
- link the actual boundary result to the control result; and
- keep the attached execution form, source values, backend, decoding, and
  target identity fixed.

It must not add a second intervention, change Adapter dtype or execution form,
or claim a unique low-level root cause from the control.

## Exact reproduction commands

```powershell
.\work\training-env\Scripts\python.exe `
  .\scripts\probe_tool_router_attached_dtype_numerics.py `
  --config .\configs\tool_router_lora_sft_v2.json `
  --model-dir .\work\models\Qwen2.5-1.5B-Instruct `
  --adapter-dir .\baseline\adapters\fc-mvp-001-lora-sft-v2 `
  --isolation-evidence .\baseline\fc-mvp-001-attached-dtype-isolation-v1.json `
  --output .\work\test-fixtures\fc-mvp-001-attached-dtype-numerics-v1.json

.\work\training-env\Scripts\python.exe -I .\scripts\validate_offline.py
```

The probe implementation is
[`scripts/probe_tool_router_attached_dtype_numerics.py`](../scripts/probe_tool_router_attached_dtype_numerics.py).
The capture/comparison helpers and strict JSON validator are
[`tool_router_attached_dtype_numerics.py`](../src/fullcycle_bridge/tool_router_attached_dtype_numerics.py)
and
[`tool_router_attached_dtype_numerics_evidence.py`](../src/fullcycle_bridge/tool_router_attached_dtype_numerics_evidence.py).
