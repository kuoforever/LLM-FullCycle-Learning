# Project status

> Updated: 2026-07-29.
> This is the operational entry point for a new Full Cycle session.

## Current phase

MVP-0 remains frozen. The first `FC-MVP-001` training-preparation gate now has
a strict Tool Router decision v1 contract, reviewed seed records, a frozen
balanced eval set, deterministic validation, and an offline rule baseline.
No model, Provider, Runtime execution, or rich trace channel has been opened.

## Single active objective

Complete the data-expansion gate of `FC-MVP-001`:

```text
frozen eval v1 (unchanged)
        -> task-family-disjoint train/validation records >= 200
        -> distribution, duplicate, leakage, and schema audits
```

Expand reviewed and deterministically verifiable examples without changing the
20-record eval answers or their digest. Split task families, not paraphrases,
between train and validation; keep dangerous false approvals at zero and
publish a category/risk/tool distribution report. Do not start model inference,
training, Provider, MCP, Desktop, network, Memory, Continuation, or Lane B under
this next gate.

The `FC-MVP-001` schema/eval gate completed locally on 2026-07-29:
`tool_router_schema_version=1`, 20 reviewed seed records, 20 frozen eval
records, ten categories with two eval cases each, and canonical eval digest
`sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a`.
The deterministic non-model baseline produced tool accuracy `1.0`, argument
exact match/F1 `0.0/0.0`, risk Macro F1 `0.8641148325358852`, approval,
rejection, and fallback accuracy `1.0`, and zero dangerous false approvals.
The unified offline gate passed `31 tests` on Python 3.11.15, 3.12.12, and
3.13.7; Ruff passed and mypy passed all nine source/script files.

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
| `FC-MVP-001` | In progress | Text Tool Router closed loop; schema/eval gate complete, data expansion next |
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
