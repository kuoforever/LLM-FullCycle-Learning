# Project status

> Updated: 2026-07-28.  
> This is the operational entry point for a new Full Cycle session.

## Current phase

The Desktop Runtime Lane A producer and the Full Cycle strict offline consumer
now form a pinned, tested bridge baseline. The next phase may map only this
validated redacted evidence into a reliability/Verifier dataset.

## Single active objective

Complete `FC-BRIDGE-002` in this repository:

```text
validated Lane A run-export v1
        -> versioned reliability/Verifier dataset schema
        -> deterministic mapping and dataset fixtures
```

Use only records accepted by `fullcycle_bridge` and preserve
`training_use=reliability_and_verifier_only`. Define deterministic examples for
failure, unknown outcome, denial, recovery, budget, and tool-sequence analysis.
Do not add raw task/model/tool-result text, images, Provider, MCP, Desktop,
network, Approval, Memory, Continuation, or Lane B capture.

`FC-BRIDGE-001` completed on 2026-07-28 with consumer schema `1.0.0`, Runtime
commit `8ace897f746a4aa3dd3f8b10af392ea9ba81941d`, one valid producer-pinned
manifest, one minimal valid run export, and eight invalid fixtures. Validation
on Python 3.13.7: `12 tests` passed, Ruff passed, mypy passed, and the offline
CLI accepted the valid fixture with the pinned manifest digest. The repository
is local-only with no configured remote.

## Full Cycle backlog

| ID | Status | Deliverable |
|---|---|---|
| `FC-PM-000` | Complete | Project structure, MVP roadmap, scenario matrix, Project H, cross-repo management |
| `FC-BRIDGE-001` | Complete | Strict manifest/run-export consumer and offline compatibility fixtures |
| `FC-BRIDGE-002` | Next | Lane A reliability/Verifier dataset mapping |
| `FC-BRIDGE-003` | Pending review | Explicit-consent rich multimodal capture contract |
| `FC-MVP-000` | Pending | Freeze Runtime consumer baseline and environment |
| `FC-MVP-001` | Pending | Text Tool Router closed loop |
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
