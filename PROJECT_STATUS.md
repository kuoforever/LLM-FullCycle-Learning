# Project status

> Updated: 2026-07-29.
> This is the operational entry point for a new Full Cycle session.

## Current phase

MVP-0 remains frozen. `FC-MVP-001` now has a strict Tool Router decision v1
contract, a frozen balanced eval set, 200 task-family-disjoint
train/validation records, deterministic validation, leakage/distribution
audits, and an offline rule baseline. No model, Provider, Runtime execution, or
rich trace channel has been opened.

## Single active objective

Complete the inference-baseline gate of `FC-MVP-001`:

```text
separately locked local inference environment + exact base model revision
        -> JSON-only prompt adapter
        -> frozen prompt-only/base predictions and eval report
```

Select and pin one appropriately licensed open-weight text model and tokenizer,
record environment/hardware/seed/generation settings, and evaluate it only
against the unchanged 20-record eval fixture. Freeze raw predictions before
scoring with the existing metrics. Do not train, connect Runtime, Provider,
MCP, Desktop, Memory, Continuation, or Lane B under this next gate.

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
| `FC-MVP-001` | In progress | Text Tool Router closed loop; data gate complete, local base-model evaluation next |
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
