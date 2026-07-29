# Project status

> Updated: 2026-07-29.
> This is the operational entry point for a new Full Cycle session.

## Current phase

MVP-0 remains frozen. `FC-MVP-001` now has a strict Tool Router decision v1
contract, a frozen balanced eval set, 200 task-family-disjoint
train/validation records, deterministic validation, leakage/distribution
audits, an offline rule baseline, a reproducible local prompt-only model
baseline, and a reproducible first local LoRA SFT adapter. SFT materially
improved routing quality but still failed the dangerous-action gate and is not
Runtime eligible.

## Single active objective

Complete the safety-repair data gate of `FC-MVP-001`:

```text
frozen SFT v1 badcase taxonomy
        -> reviewed train/validation-only hard negatives v2
        -> leakage and dangerous-action pre-training gate
```

Classify the one remaining dangerous action candidate, four semantic
inconsistencies, and validation overfitting without changing the frozen eval.
Create a reviewed v2 train/validation-only hard-negative increment with
explicit provenance and task families; reject exact/near eval leakage and
preserve the canonical eval digest. Do not retrain or connect Runtime,
Provider, MCP, Desktop, Memory, Continuation, or Lane B in this gate. The next
training config must be locked only after this data gate passes.

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
is published as the private GitHub repository
`kuoforever/LLM-FullCycle-Learning`.

`FC-BRIDGE-002` completed on 2026-07-28 with
`reliability_dataset_schema_version=1`, a strict Draft 2020-12 JSON Schema, a
canonical JSONL mapper, two exact input/output fixtures, and deterministic
failure, unknown-outcome, policy-denial, recovery, budget-limit, and
tool-sequence signals. Validation on Python 3.13.7: `21 tests` passed, Ruff
passed, mypy passed, the JSON Schema and two records validated, and the offline
script reproduced both JSONL records byte-for-byte.

`FC-MVP-000` local gates completed on 2026-07-28 at implementation commit
`01167034d797d4d6855b1ba916b60564d29ba210`: Python 3.11.15, 3.12.12, and
3.13.7 each passed `21 tests`, seven artifact hashes, five source import-boundary
audits, and two exact dataset records with zero runtime dependencies. Ruff
0.15.22 and mypy 2.3.0 also passed.

`FC-MVP-000` remote gate completed on 2026-07-28. The private repository is
`kuoforever/LLM-FullCycle-Learning`; Actions run `30369941536` at head
`80bafb4a5bd5039115519ad7239584be39acb037` passed the Python 3.11, 3.12, and
3.13 matrix jobs. The exact run and job IDs are recorded in
`baseline/validation-2026-07-28.json`.

## Full Cycle backlog

| ID | Status | Deliverable |
|---|---|---|
| `FC-PM-000` | Complete | Project structure, MVP roadmap, scenario matrix, Project H, cross-repo management |
| `FC-BRIDGE-001` | Complete | Strict manifest/run-export consumer and offline compatibility fixtures |
| `FC-BRIDGE-002` | Complete | Lane A reliability/Verifier dataset mapping |
| `FC-BRIDGE-003` | Pending review | Explicit-consent rich multimodal capture contract |
| `FC-MVP-000` | Complete | Runtime consumer baseline, locked environment, local/remote Python matrix |
| `FC-MVP-001` | In progress | Text Tool Router closed loop; first LoRA SFT complete, safety-repair data gate next |
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

- One flagship project and four depth Labs.
- Desktop GUI is the first environment, not the permanent product boundary.
- Runtime owns execution safety; Full Cycle owns models and datasets.
- Automatic Runtime export is redacted reliability evidence only.
- Rich multimodal episodes require explicit consent and a separate review.
- Multi-Agent is formal Project H but does not block the first closed loop.
- Runtime Lane A producer v1 passed `1428` tests plus Ruff, mypy, docs, wheel
  build/install, and offline release gates, then PR #219 passed the Python
  3.11-3.13 and wheel CI gates and merged as `8ace897` on 2026-07-28.
