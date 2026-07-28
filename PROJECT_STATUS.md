# Project status

> Updated: 2026-07-28.  
> This is the operational entry point for a new Full Cycle session.

## Current phase

The project design is complete enough to start implementation. The Desktop
Runtime now produces the safe bridge; the first Full Cycle code task is its
strict offline consumer.

## Single active objective

Complete `FC-BRIDGE-001` in this repository:

```text
manifest v1 + redacted run-export v1
        -> strict offline consumer
        -> valid and invalid compatibility fixtures
```

Read `Desktop_Runtime_依赖与集成.md` for the exact producer contract. Validate
supported versions, manifest digest, `automatic_export` claims, data class,
training-use restriction, size bound, and strict structure. Reject unknown
versions, malformed/oversized input, digest mismatch, unexpected rich content,
and incomplete events. Do not open provider, MCP, desktop, network, approval,
memory, or continuation ports.

Pin Runtime merge commit `8ace897f746a4aa3dd3f8b10af392ea9ba81941d`
and the schema versions below in the fixture metadata. Do not start multimodal
training or rich episode capture under this item.

This directory was not a Git worktree when checked on 2026-07-28. Before adding
consumer code, initialize it as a repository or attach it to the intended
remote so fixtures, schemas, and environment locks receive reproducible commit
identities.

## Full Cycle backlog

| ID | Status | Deliverable |
|---|---|---|
| `FC-PM-000` | Complete | Project structure, MVP roadmap, scenario matrix, Project H, cross-repo management |
| `FC-BRIDGE-001` | Next | Strict manifest/run-export consumer and offline compatibility fixtures |
| `FC-BRIDGE-002` | Pending | Lane A reliability/Verifier dataset mapping |
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
