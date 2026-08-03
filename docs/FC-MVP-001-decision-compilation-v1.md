# FC-MVP-001 decision compilation v1

> **Result: COMPLETE LOCALLY — the frozen v2 decision inconsistencies are
> removed without changing model output or training; Runtime eligibility
> remains false.**

## Contract

The compiler treats `selected_tool` as the model's one proposed terminal
disposition and derives the redundant v1 fields deterministically:

| Selected tool class | `expected_result` | `should_reject` | `should_fallback` | Approval |
|---|---|---:|---:|---|
| `reject_request` | `rejection` | true | false | false |
| `fallback_to_strong_model` | `fallback` | false | true | false |
| `request_clarification` | `clarification` | false | false | false |
| Other tool, approval required | `approval_required` | false | false | true |
| Other tool, no approval | `tool_candidate` | false | false | false |

Contradictory independently generated flags are not authority. Compilation
fails closed if the source prediction, frozen parsed report, classification
digest, eval digest, or expected changed cohort drifts.

## Frozen transformation

The source prediction artifact remains byte-unchanged at
`sha256:8262e68a100d0ecf888a337b57631a3b3e42b54ce02f5274e4e7b77c1fb56d46`.
Compilation creates a separate derived artifact. Exactly three cases change:

| Example | Changed fields |
|---|---|
| `eval-001` | `$.expected_result`, `$.should_reject` |
| `eval-014` | `$.expected_result`, `$.should_reject` |
| `eval-020` | `$.expected_result`, `$.should_reject` |

No instruction, argument, selected tool, risk, approval flag, or raw source
artifact changes.

## Result

| Metric | Frozen LoRA SFT v2 | Compiled v1 |
|---|---:|---:|
| Decision semantic validity | 0.85 | 1.00 |
| Tool accuracy | 0.95 | 0.95 |
| Rejection accuracy | 0.85 | 1.00 |
| False refusals | 3 | 0 |
| Dangerous action candidates | 0 | 0 |
| Dangerous false approvals | 0 | 0 |

The unchanged 20-case eval digest is
`sha256:02221fedccb331466eb7b4a354c725a0ab31ac121f022298f42d3782b5e56a7a`.
The compiler was fixed before this one frozen scoring pass; it did not read eval
answers or iterate against their labels.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Compiled predictions | `sha256:2888ef53ca5888f85fba0f28a54d68a865d478ca7d3cf01dc8ca9c96444455cf` |
| Compiled report | `sha256:c35f1608a7de4da4991347131082385866af942acb7f7dc905612402d406602b` |
| Gate record | `sha256:0e798d3404acd4fc6965d773a5ee2f8b3c593eb7865774a0acaadf7d2073a6de` |

The canonical record is
[`baseline/fc-mvp-001-decision-compilation-v1.json`](../baseline/fc-mvp-001-decision-compilation-v1.json).

## Boundary and next gate

This is an offline contract compiler, not model improvement and not Runtime
integration. The independently loaded adapter remains the only allowed v2
adapter form. The safe-merged output is still prohibited because it changed a
generated boolean.

The single next objective is `FC-MVP-001-bf16-merge-stability-v1`: reproduce
`eval-001` from fresh independent and safe-merged BF16 loads, prove repeat
stability on each path, and locate the first token or logit divergence. It may
not add data, train, tune against eval answers, connect Runtime, or permit a
merged artifact before output identity is restored.

## Reproduction

```powershell
python -I .\scripts\compile_tool_router_v2_decisions.py
python -I .\scripts\validate_offline.py
```

The unified offline gate passes 71 tests on CPython 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, and mypy reports no issues in all 27
source/script files.
