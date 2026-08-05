# Project status

> Updated: 2026-08-05.
> This is the operational entry point for a new Reliable Agent Model Lifecycle
> session.

## Current phase

MVP-0 remains frozen. `FC-MVP-001` now has a strict Tool Router decision v1
contract, a frozen balanced eval set, 200 task-family-disjoint
train/validation records, deterministic validation, leakage/distribution
audits, an offline rule baseline, a reproducible local prompt-only model
baseline, two reproducible local LoRA SFT adapters, and a passed v2
safety-repair data gate. LoRA SFT v2 passed the narrow dangerous-action gate,
and its frozen failure-classification gate now separates decision-contract
inconsistency from BF16 load/merge output drift. Decision compilation v1
removes the former without changing raw model output. Merge stability v1
proves the latter is a deterministic BF16 logit-boundary flip rather than
within-path nondeterminism, and merge numerics v1 traces it to BF16
materialization of LoRA updates at the first projection. The pre-registered
FP32 remediation is repeat-stable but fails output identity and reproduces the
safe-merged BF16 token sequence instead of the independent BF16 Adapter
reference. FP32 merge drift analysis now reproduces both frozen paths and
confirms that their first generated-token boundary contains a raw-logit argmax
flip, not merely a logits-processor flip. Because that comparison couples
compute dtype with attached-versus-merged execution, FP32 attached/merge
isolation adds the missing same-dtype control. Two fresh attached runs are
exactly repeat-stable, the unchanged merged path reproduces, and both forms
emit identical tokens and output while retaining small deterministic numerical
trace drift. FP32 attached/merge numerics v1 now locates the first unequal
paired captured output at layer 0 `q_proj` and replays the factorized LoRA and
materialized-linear forms from a raw tensor sidecar. Both registered
counterfactual differences remain nonzero at the `q_proj` output boundary,
but their independent propagation beyond `q_proj` is not isolated. Attached
dtype isolation v1 now adds the missing fixed-form BF16/FP32 control: four
fresh ABBA-ordered attached runs reproduce both frozen paths, and the raw
LM-head argmax flips from `true` to `false` at the locked token boundary when
only the base/inference dtype changes while the Adapter remains FP32. This
isolates a repeat-stable total dtype effect on the one frozen example, not its
first module boundary or a unique low-level root cause. The Adapter remains
Runtime ineligible; the next gate must locate the first registered attached
module-output difference and quantify its propagation.

## Single active objective

Complete `FC-MVP-001-attached-dtype-numerics-v1`:

```text
fresh repeat-stable BF16 and FP32 attached paths + frozen target forward
        -> capture the same pre-registered module-output sequence
paired BF16-versus-FP32 comparisons in execution order
        -> locate the first registered unequal module output
registered downstream comparisons on the same cached forward
        -> quantify propagation without claiming a unique low-level root cause
```

Keep both frozen attached paths and the exact `eval-001` cached forward that
predicts generated token index `45`. Pre-register a bounded common module
capture plan, reproduce both 48-token paths in fresh ABBA lifecycles, locate
the first unequal output inside that plan, and quantify registered propagation
to the LM head. Keep the BF16 checkpoint source values, FP32 Adapter values,
prompt, eval digest, high-level SDPA backend, greedy decoding, and target step
unchanged. Do not add data; train; tune against eval answers; run the full
eval; connect Runtime/Provider/MCP/Desktop; change execution form; save or
promote merged weights; add a module tensor sidecar; or claim the first
unregistered operation, a unique CUDA/floating-point root cause, or a PEFT bug.

The `FC-MVP-001-attached-dtype-isolation-v1` gate completed locally on
2026-08-05. Four fresh runs execute in ABBA order: BF16 attached, FP32
attached, FP32 attached, BF16 attached. Each dtype is exactly repeat-stable
and reproduces its frozen 48-token, decoded-output, processed-score,
raw-logit, and target-forward references. Both paths keep the same attached
factorized LoRA form and FP32 Adapter runtime values; PEFT
`autocast_adapter_dtype=true` is a locked load-time policy that would upcast
FP16/BF16 Adapter weights, while these stored FP32 values remain FP32.
Generation autocast remains disabled. The paths share generated indices
`0`–`44`; at index `45`, BF16 emits token `1866` (`true`) and FP32 emits token
`3849` (`false`). Processed-score margins are `0.4545440673828125` and
`2.4418487548828125`; raw-logit margins are `0.5` and `2.68603515625`.
All `151,936` vocabulary elements differ in both comparison vectors, with
maximum absolute delta `1.943955421447754`. The classification is
`deterministic_bf16_attached_vs_fp32_attached_raw_logit_boundary_flip`.
`dtype_isolation_gate.passed=true`, `remediation_gate.passed=false`, and
`runtime_eligible=false`. The probe took `37.87893820001045` seconds and
peaked at `6,285,127,680` allocated GPU bytes; every released lifecycle stays
at `8,519,680` bytes, below the 16 MiB ceiling. The JSON-only validator closes
the source chain, ABBA protocol, 48-step manifests, target/LM-head linkage,
top-k algebra, classification, gates, and policy. Without raw vector payloads,
its full-vector delta checks are limited to probe-derived summary algebra and
do not independently recompute maximum/mean/RMS statistics. The evidence does
not support a pristine-FP32 checkpoint claim, first module or unique CUDA root
cause, PEFT bug, full-eval generalization, artifact promotion, or Runtime
eligibility. The canonical record SHA-256 is
`7eaedee1d6f7ea27b2fc083f82bc8df620612e84095f4a66a2ba7dfec791ce31`.
The unified offline gate passes 144 tests on Python 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, py_compile passes the new source/script/
test files, and mypy 2.3.0 reports no issues in all 43 source/script files.
[Evidence](docs/FC-MVP-001-attached-dtype-isolation-v1.md).

The `FC-MVP-001-fp32-attached-merge-numerics-v1` gate completed locally on
2026-08-05. Four fresh runs execute in ABBA order: attached, merged, merged,
attached. Every run reproduces its frozen 48-token, decoded-output,
processed-score, raw-logit, and target-forward references, and all captured
tensors are bitwise repeat-stable within each path. The 13-stage paired output
comparison first differs at index `2`,
`model.layers.0.self_attn.q_proj`, after bitwise-identical embedding, layer 0
input normalization, and `q_proj` input tensors. Its output differs in
`1,261/1,536` elements, with maximum/mean/RMS absolute deltas
`3.814697265625e-06`/`1.6505699325837972e-07`/
`2.990363292409807e-07`. Probe-captured linear replays match both actual
outputs, while the stdlib gate independently recomputes their comparisons and
the scalar/add identities.
The factorized LoRA term versus delta-weight linear comparison differs in
`1,355` elements with maximum delta `3.259629011154175e-09`, and that axis
survives base addition in 53 elements. Split base-plus-delta versus the
materialized-weight linear differs in `1,260` elements with maximum delta
`3.814697265625e-06`. The representative `q_proj` audit recomputes all
`2,359,296` merged weights with zero mismatches; 21 nonzero archived FP32 delta
updates round back to the base value. The classification is
`deterministic_fp32_factorized_lora_and_materialized_linear_execution_form_drift`.
The Git LFS tensor sidecar contains 138 bound records and `46,069,904` bytes,
with SHA-256
`550175dfcfe14b0739aabf17573825a124180a6e21826e25d4b5ff733fb298a9`;
the JSON record SHA-256 is
`cb1c2b4255ebc5c38aa2ff66436804cca55dc088e39ca8fe8959654488e41a91`.
`numerics_gate.passed=true`, `remediation_gate.passed=false`, and
`runtime_eligible=false`. The probe took `29.912574299960397` seconds and
peaked at `6,286,505,472` allocated GPU bytes. It does not claim the earliest
temporal or unregistered functional divergence, independent propagation of
either counterfactual beyond `q_proj`, a unique floating-point or CUDA root
cause, a PEFT bug, full-eval generalization, merged-artifact promotion, or
Runtime eligibility. The unified offline gate passes 124 tests on Python
3.11.15, 3.12.12, and 3.13.7; Ruff passes the repository, py_compile passes the
new source/test files, and mypy 2.3.0 reports no issues in all 40 source/script
files.
[Evidence](docs/FC-MVP-001-fp32-attached-merge-numerics-v1.md).

The `FC-MVP-001-fp32-attached-merge-isolation-v1` gate completed locally on
2026-08-04. Two fresh independent FP32 attached-Adapter loads are exactly
repeat-stable across 48 generated tokens, decoded output, all processed-score
and raw-logit vectors, precision audit, and frozen-step vectors. One fresh
unchanged FP32 safe-merged load reproduces its frozen token/output/full-trace
and index-45 vector digests. All three runs share token digest
`sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca`
and output digest
`sha256:b37939d2e8014afcc92b094d9c63715aa28d91f02504bf8b56186dd2dd5cc7ca`,
so there is no same-dtype token boundary. At the pre-registered BF16 context
step `45`, both FP32 forms emit `false`, but `150,968/151,936` processed-score
and raw-logit elements differ. Both have maximum absolute delta
`0.0001735687255859375`; processed-score mean/RMS deltas are
`0.00002052841409749817`/`0.000026467831048648804`, and raw-logit mean/RMS
deltas are `0.00002052839772659354`/`0.000026469620934221894`. The
classification is
`deterministic_fp32_attached_vs_merged_numerical_drift_without_token_drift`,
`isolation_gate.passed=true`, `remediation_gate.passed=false`, and
`runtime_eligible=false`. The run took `29.405898299999535` seconds and peaked
at `6,285,651,968` allocated GPU bytes. The gate record SHA-256 is
`37d8d35bc3802a76bd7e0ab484f3b86e01b03852212ae9aaf3d9cec318fb5e26`.
No BF16 GPU path was rerun, and no merged artifact was saved or permitted. The
unified offline gate passes 106 tests on Python 3.11.15, 3.12.12, and 3.13.7;
Ruff passes the repository and mypy reports no issues in 37 source/script
files.
[Evidence](docs/FC-MVP-001-fp32-attached-merge-isolation-v1.md).

The `FC-MVP-001-fp32-merge-drift-analysis-v1` gate completed locally on
2026-08-04. One fresh independent BF16 attached-Adapter path and one fresh
locked FP32 safe-merged path each reproduce their frozen 48-token and decoded
output digests. Their first generated-token boundary remains zero-based index
`45`: BF16 attached chooses token `1866` (`true`), while FP32 merged chooses
token `3849` (`false`). Processed generation-score margins are
`0.4545440673828125` and `2.4415740966796875`; raw-logit margins from the same
cached `generate` call are `0.5` and `2.68572998046875`. The raw-logit vector's
maximum absolute delta is `1.9437971115112305`, mean absolute delta is
`0.22757971286773682`, and all `151,936` vocabulary elements differ at this
step. The classification is
`deterministic_bf16_attached_vs_fp32_merged_raw_logit_boundary_flip`,
`analysis_gate.passed=true`, and `remediation_gate.passed=false`. This locates
the first token/argmax boundary but does not isolate dtype from merge execution
or claim earlier logits are identical. The run took `13.201921800035052`
seconds and peaked at `6,268,076,032` allocated GPU bytes. The gate record
SHA-256 is
`ae5d1c7ace24c6cfcfed0eca60354cd3dfa9579fa0aea4e1f64c66eb73e41ea3`.
No merged artifact was saved or permitted, and Runtime eligibility remains
false. The unified offline gate passes 95 tests on Python 3.11.15, 3.12.12,
and 3.13.7; Ruff passes the repository and mypy reports no issues in 35
source/script files.
[Evidence](docs/FC-MVP-001-fp32-merge-drift-analysis-v1.md).

The `FC-MVP-001-bf16-merge-remediation-v1` gate completed locally on
2026-08-04 and falsified its sole pre-registered candidate. Two fresh FP32
safe-merged loads are token-identical at 48 generated tokens, both with digest
`sha256:9dfd817e59df5c0278fdd9da20feb3664fade5d354040bbd5b3b4c650ca43dca`.
They do not match the frozen independent BF16 Adapter digest
`sha256:e23b3f5ed71ec57f44ccacfadf8d79abfb21be622f13cae83cf14274cc54e173`;
instead, they match the prior safe-merged BF16 token and output digests. Both
runs verify 338 FP32 base tensors, 224 FP32 LoRA tensors across 112 targets,
zero LoRA tensors after FP32 safe merge, FP32 generation scores, greedy SDPA,
disabled autocast/TF32, tied embeddings, and isolated fresh-load lifecycles.
The classification is `deterministic_fp32_merge_output_drift` and
`remediation_gate.passed=false`. The run took `18.06072970002424` seconds and
peaked at `6,248,754,688` allocated GPU bytes. The gate record SHA-256 is
`7f3c5aff55e69c08a7676d33636a52a5a2bb43f025dae8a2db362041354050b3`.
No merged artifact was saved or permitted, and Runtime eligibility remains
false. The unified offline gate passes 87 tests on Python 3.11.15, 3.12.12,
and 3.13.7; Ruff passes the repository and mypy reports no issues in 33
source/script files.
[Evidence](docs/FC-MVP-001-bf16-merge-remediation-v1.md).

The `FC-MVP-001-bf16-merge-numerics-v1` gate completed locally on 2026-08-04.
At the exact cached generation step for token index `45`, the embedding and
layer 0 input normalization are element-identical across paths. The first
difference is layer 0 `q_proj`: `569/1536` output elements differ, with maximum
absolute delta `0.0625`. Across all `112` Q/K/V/O LoRA targets and
`154,140,672` weights, the reproduced PEFT safe-merge algorithm has zero
weight mismatches with the actual merged model. However, `30,640,994` nonzero
ideal updates round back to their base BF16 values, or
`0.19878591161195924` of all nonzero updates. The classification is
`bf16_safe_merge_weight_rounding`. The gate record SHA-256 is
`eb39674127ac93fea2ce6415b3a2fea0d20f6da916b76f1532392533db3e805f`.
No merged artifact was saved or permitted, and Runtime eligibility remains
false. The unified offline gate passes 81 tests; Ruff passes the repository
and mypy reports no issues in 31 source/script files.
[Evidence](docs/FC-MVP-001-bf16-merge-numerics-v1.md).

The `FC-MVP-001-bf16-merge-stability-v1` gate completed locally on 2026-08-04.
Two fresh independent Adapter loads are token-identical to each other, and two
fresh safe-merged BF16 loads are token-identical to each other. The paths first
diverge at zero-based generated token index `45`: independent loading chooses
token `1866` (`true`) while safe merge chooses token `3849` (`false`). Scores
captured from the exact cached generation step confirm a deterministic argmax
flip: the independent margin is `0.4545440673828125`, the merged margin is
`4.090908050537109`, maximum absolute logit delta is `3.0`, and mean absolute
delta is `0.3340962529182434`. The gate record SHA-256 is
`82bc73310625855770d6cc90aab6b5ed0e78fc1cd3c7684fd007ac8379c67abc`.
No merged artifact was saved or permitted, and Runtime eligibility remains
false. The unified offline gate passes 77 tests; Ruff passes the repository
and mypy reports no issues in 29 source/script files.
[Evidence](docs/FC-MVP-001-bf16-merge-stability-v1.md).

The `FC-MVP-001-decision-compilation-v1` gate completed locally on 2026-08-03.
It compiles redundant terminal fields from `selected_tool` without modifying
the frozen raw prediction artifact. Exactly `eval-001`, `eval-014`, and
`eval-020` change `expected_result` and `should_reject`; no selected tool,
argument, risk, approval, instruction, or source artifact changes. On the same
eval digest, decision semantic validity rises from `0.85` to `1.0`, rejection
accuracy from `0.85` to `1.0`, false refusals fall from three to zero, tool
accuracy remains `0.95`, and dangerous action candidates and false approvals
remain zero. The gate record SHA-256 is
`0e798d3404acd4fc6965d773a5ee2f8b3c593eb7865774a0acaadf7d2073a6de`.
The unified offline gate passes 71 tests on Python 3.11.15, 3.12.12, and
3.13.7; Ruff passes the repository and mypy reports no issues in 27
source/script files. The compiled output remains Runtime ineligible because
merge drift is still unresolved.
[Evidence](docs/FC-MVP-001-decision-compilation-v1.md).

The `FC-MVP-001` v2 failure-classification gate completed locally on
2026-08-03. The three semantic conflicts are exactly `eval-001`, `eval-014`,
and `eval-020`; all select fallback while setting both fallback and rejection
flags. The aggregate false-refusal count is also three, so both failure groups
belong to decision-contract consistency without opening per-case eval labels.
The safe BF16 merge separately changes only `$.should_reject` on `eval-001`
despite removing all adapter tensors, so merged output remains prohibited and
belongs to adapter-merge stability. Frozen aggregate evidence does not support
a data-coverage diagnosis. The canonical classification report digest is
`sha256:671e4fad7e2b9987b0cbf3f3fdb078c11431efa5887109a204874ec136316a9a`.
The unified offline gate passes 67 tests on Python 3.11.15, 3.12.12, and
3.13.7; Ruff passes the repository and mypy reports no issues in 25
source/script files. [Evidence](docs/FC-MVP-001-v2-failure-classification.md).

The `FC-MVP-001` LoRA SFT v2 gate completed locally on 2026-08-03. The locked
three-epoch BF16 LoRA run trained for 66 optimizer steps on the passed 176/48
data in `169.527236` seconds with `5,217,494,016` peak allocated GPU bytes.
On the unchanged eval, Tool Accuracy improved from `0.8` to `0.95`, dangerous
action candidates fell from one to zero, and dangerous false approvals stayed
zero. The narrow safety gate passed. Decision semantic validity is only
`0.85`, with three `CONFLICTING_DECISION_FLAGS` outputs and three false
refusals. `safe_merge` removed all adapter tensors but changed one boolean on
`eval-001`, so output identity failed. The adapter and all raw evidence are
frozen, and `runtime_eligible=false`. The unified offline gate passed `63`
tests on Python 3.11.15, 3.12.12, and 3.13.7; Ruff and mypy passed.

The `FC-MVP-001` safety-repair data gate completed locally on 2026-08-03. The
frozen SFT v1 diagnosis records one dangerous action candidate, four semantic
inconsistencies across four eval cases, and validation overfitting after epoch
3. A reviewed train/validation-only increment added 16/8 records and eight
disjoint repair families, producing 176/48 records across 68 families while
preserving v1 as the exact prefix and keeping the eval digest unchanged. Eval
answers are excluded, maximum cross-split instruction token Jaccard is
`0.4166666666666667` under the `0.8` rejection threshold, and dangerous action
candidates and dangerous false approvals are both zero. The pinned repair
report digest is
`sha256:2383731556a66ba81de670378c18afcd0493d368dc157d6a5a4e51e5904ee4b2`.

The first `FC-MVP-001` SFT gate completed locally on 2026-07-29. BF16 LoRA
rank 16 / alpha 32 targeted Q/K/V/O projections for 5 epochs and 100 optimizer
steps on the frozen 160/40 data. Training took `216.825720` seconds, peak
allocated GPU memory was `5,217,494,016` bytes, and 4,358,144 parameters
(`0.281521%`) were trainable. The independent Adapter directory is 17,468,332
bytes; its 17,462,432-byte weight file has SHA-256
`1c58a3d08598250cc01bd35a3367fbcc778c551782e6117f686394ede3d65659`.
Independent loading and safe merge produced identical verification output.
On the unchanged eval, Tool Accuracy improved from `0.2` to `0.8`, argument
exact match from `0.0` to `0.35`, and risk Macro F1 from
`0.4257518796992481` to `0.7373015873015873`. One dangerous action candidate
remains, so `safety_gate_passed=false` and `runtime_eligible=false`.

The `FC-MVP-001` schema/eval gate completed locally on 2026-07-29:
`tool_router_schema_version=1`, 20 reviewed seed records, 20 frozen eval
records, ten categories with two eval cases each, and canonical eval digest
`sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a`.
The deterministic non-model baseline produced tool accuracy `1.0`, argument
exact match/F1 `0.0/0.0`, risk Macro F1 `0.8641148325358852`, approval,
rejection, and fallback accuracy `1.0`, and zero dangerous false approvals.
The unified offline gate passed `31 tests` on Python 3.11.15, 3.12.12, and
3.13.7; Ruff passed and mypy passed all nine source/script files.

The `FC-MVP-001` data-expansion gate completed locally on 2026-07-29 with 160
train and 40 validation records across 60 explicit task families. Every
category contributes 16 train and four validation records; task-family overlap
and exact instruction duplicates are zero. Maximum cross-split instruction
token Jaccard is `0.4166666666666667` under the `0.8` rejection threshold,
dangerous false approvals remain zero, and the frozen eval digest is unchanged.
The pinned data report digest is
`sha256:b58af24bdc3cfd34eb4309f91e977f2f4fc6f76a53a229eaa8d3f757d1ebf9a4`.
The unified offline gate passed `40 tests` on Python 3.11.15, 3.12.12, and
3.13.7; Ruff passed and mypy passed all 12 source/script files.

The `FC-MVP-001` inference-baseline gate completed locally on 2026-07-29 with
`Qwen/Qwen2.5-1.5B-Instruct` at Hub revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` under Apache-2.0. BF16 greedy SDPA
inference on an RTX 4090 Laptop GPU completed 20 cases in `74.492267` seconds
with `3,132,882,944` peak allocated GPU bytes and empty stderr. A second run
was byte-identical to prediction artifact SHA-256
`6182e70cdab772597a68d6b7e0bcbbff8b74c20626fa197c68dbced82e0d5f0d`.
JSON validity was `1.0`, but decision semantic validity was `0.7`, tool
accuracy was `0.2`, rejection recall was `0.0`, and both dangerous cases
produced dangerous action candidates. The model is explicitly
`runtime_eligible=false`. Frozen prediction scoring is reproduced by the
standard-library gate. The unified offline gate passed `45 tests` on Python
3.11.15, 3.12.12, and 3.13.7; Ruff passed and mypy passed all 16
source/script files.

`FC-BRIDGE-001` completed on 2026-07-28 with consumer schema `1.0.0`, Runtime
commit `8ace897f746a4aa3dd3f8b10af392ea9ba81941d`, one valid producer-pinned
manifest, one minimal valid run export, and eight invalid fixtures. Validation
on Python 3.13.7: `12 tests` passed, Ruff passed, mypy passed, and the offline
CLI accepted the valid fixture with the pinned manifest digest. The repository
is published as the GitHub repository
`kuoforever/reliable-agent-model-lifecycle`.

`FC-BRIDGE-002` completed on 2026-07-28 with
`reliability_dataset_schema_version=1`, a strict Draft 2020-12 JSON Schema, a
canonical JSONL mapper, two exact input/output fixtures, and deterministic
failure, unknown-outcome, policy-denial, recovery, budget-limit, and
tool-sequence signals. Validation on Python 3.13.7: `21 tests` passed, Ruff
passed, mypy passed, the JSON Schema and two records validated, and the offline
script reproduced both JSONL records byte-for-byte.

`FC-BRIDGE-004` completed locally on 2026-08-02. The canonical
`baseline/runtime-freeze-v1.json` pins Runtime commit
`324ff2fb5911e332ddb5c5f90eb41296e8faf7a9`, package `0.1.0`, every Lane A
contract version, consumer schema `1.0.0`, and manifest digest
`sha256:6abe3431ea0e6b4065f21e9a6c6fe34de772f9c3c86a2437f8d14f95a5d6f522`.
The Runtime clean preflight passed `1566` tests with `8` skips on CPython
3.13.7; its sanitized report SHA-256 is
`dc78f08030b4d3c4fac255a91fb7badf2b06fdb0eb0c487073e1f825260c6d0e`.
The coordinated consumer offline gate passed `53 tests`, all seven frozen
artifact hashes, and exact bridge/dataset reproduction on Python 3.13.7; Ruff
passed and mypy reported no issues in 21 source/script files.
The old `8ace897` fixture pin remains immutable generation provenance. Lane B
is explicitly deferred to `FC-BRIDGE-003` and remains disabled by default.

`FC-MVP-000` local gates completed on 2026-07-28 at implementation commit
`01167034d797d4d6855b1ba916b60564d29ba210`: Python 3.11.15, 3.12.12, and
3.13.7 each passed `21 tests`, seven artifact hashes, five source import-boundary
audits, and two exact dataset records with zero runtime dependencies. Ruff
0.15.22 and mypy 2.3.0 also passed.

`FC-MVP-000` remote gate completed on 2026-07-28. The repository is
`kuoforever/reliable-agent-model-lifecycle`; Actions run `30369941536` at head
`80bafb4a5bd5039115519ad7239584be39acb037` passed the Python 3.11, 3.12, and
3.13 matrix jobs. The exact run and job IDs are recorded in
`baseline/validation-2026-07-28.json`.

## Project backlog

| ID | Status | Deliverable |
|---|---|---|
| `FC-PM-000` | Complete | Project structure, MVP roadmap, scenario matrix, Project H, cross-repo management |
| `FC-BRIDGE-001` | Complete | Strict manifest/run-export consumer and offline compatibility fixtures |
| `FC-BRIDGE-002` | Complete | Lane A reliability/Verifier dataset mapping |
| `FC-BRIDGE-003` | Pending review | Explicit-consent rich multimodal capture contract |
| `FC-BRIDGE-004` | Complete locally | Runtime freeze pin, contract compatibility, and cross-repository handoff |
| `FC-MVP-000` | Complete | Runtime consumer baseline, locked environment, local/remote Python matrix |
| `FC-MVP-001` | In progress | Text Tool Router closed loop; FP32 execution-form drift frozen at layer 0 `q_proj`, attached BF16/FP32 dtype isolation next |
| `FC-MVP-002` | Pending | Multimodal GUI Action Model |

Detailed technical tasks remain in
`AI_Infra_LLM_Agent_待做任务清单.md`. This file owns only sequencing and the
single active objective.

## Session start

1. Read `AGENTS.md` or `CLAUDE.md`.
2. Read this file.
3. Read `README.md`.
4. Read only the roadmap or task section required by the active item.
5. Inspect the target repository's status before editing.

## Session end

1. Record modified files and validation results.
2. Update one backlog status.
3. Set one exact next objective.
4. Do not report planned capabilities as implemented.
5. If Runtime contracts changed, update
   `Desktop_Runtime_依赖与集成.md` and the consumer fixture together.

## Current decisions

- The official display name is `Reliable Agent Model Lifecycle`; it names the
  complete target system while implementation claims remain evidence-backed.
- The local directory and the remote repository slug are
  `reliable-agent-model-lifecycle`, renamed on 2026-07-31 from
  `LLM-FullCycle-Learning` to match the display name. GitHub redirects the
  former slug, so frozen evidence URLs such as the `FC-MVP-000` Actions run
  URL in `baseline/validation-2026-07-28.json` stay resolvable and are left
  unedited.
- The remote repository is public. Earlier status and environment lines
  described it as private; that claim was found to contradict the repository
  on 2026-07-31 and was corrected rather than the visibility being changed.
- Existing `FC-*` IDs, `fullcycle_*` contracts, and package/CLI names remain
  unchanged for compatibility.
- One flagship project and four depth Labs.
- Desktop GUI is the first environment, not the permanent product boundary.
- Runtime owns execution safety; Reliable Agent Model Lifecycle owns models and
  datasets.
- Automatic Runtime export is redacted reliability evidence only.
- Rich multimodal episodes require explicit consent and a separate review.
- Runtime freeze commit `324ff2fb5911e332ddb5c5f90eb41296e8faf7a9`
  is pinned by `baseline/runtime-freeze-v1.json`; Lane B was deferred without
  changing the immutable Lane A fixture provenance.
- Multi-Agent is formal Project H but does not block the first closed loop.
- Runtime Lane A producer v1 passed `1428` tests plus Ruff, mypy, docs, wheel
  build/install, and offline release gates, then PR #219 passed the Python
  3.11-3.13 and wheel CI gates and merged as `8ace897` on 2026-07-28.
