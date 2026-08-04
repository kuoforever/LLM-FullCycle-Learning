# Project status

> Updated: 2026-08-04.
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
removes the former without changing raw model output. Merge stability v1 now
proves the latter is a deterministic BF16 logit-boundary flip rather than
within-path nondeterminism. The Adapter remains Runtime ineligible while the
underlying merge numerics are unresolved.

## Single active objective

Complete `FC-MVP-001-bf16-merge-numerics-v1`:

```text
frozen eval-001 common prefix at token index 45
        -> capture paired module outputs
        -> locate earliest module-level divergence
        -> quantify adapter-update versus merged-weight rounding
```

Use the pinned model, Adapter, prompt, seed, dtype, exact `eval-001` input, and
the already frozen 45-token common generated prefix. Locate the earliest
module whose output differs and measure how BF16 safe-merge rounding changes
the effective LoRA update. Do not add data, train, tune against eval answers,
run the full eval, connect Runtime/Provider/MCP/Desktop, or promote a merged
artifact.

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
| `FC-MVP-001` | In progress | Text Tool Router closed loop; BF16 merge boundary flip frozen, merge numerics next |
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
