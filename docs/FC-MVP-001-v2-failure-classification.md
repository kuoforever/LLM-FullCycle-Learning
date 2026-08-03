# FC-MVP-001 LoRA SFT v2 failure classification

> **Result: COMPLETE LOCALLY — one bounded decision-compilation gate is
> locked; the adapter remains Runtime ineligible.**

## Scope

This gate reads only four frozen v2 artifacts:

| Artifact | SHA-256 |
|---|---|
| Predictions | `sha256:8262e68a100d0ecf888a337b57631a3b3e42b54ce02f5274e4e7b77c1fb56d46` |
| Scoring report | `sha256:ce534f8471e72fd053590c3be838bad0e82f438875912a69c2e8e9e4cacc97fa` |
| Training evidence | `sha256:641b1a7ef3dc0de0d9f2124b9122cb2c4be46b42de9265d558ab6f5b25b41a30` |
| Load/merge evidence | `sha256:115b58b210d6d0a49b0e86385f8ef02eb07b2a0392a227762e8fa7f31ab373d5` |

The classifier does not open the evaluation answers, add data, train, call a
provider, connect Runtime, or touch MCP/Desktop. It recomputes the report from
the exact artifacts and fails closed if their provenance or observed failure
signature changes.

## Classification

| Failure | Evidence | Classification |
|---|---|---|
| Three `CONFLICTING_DECISION_FLAGS` outputs | `eval-001`, `eval-014`, and `eval-020` each select `fallback_to_strong_model` while setting both `should_fallback=true` and `should_reject=true` | `decision_contract_consistency` |
| Three false refusals | The aggregate count is exactly equal to the three-case conflict cohort; per-case eval labels were deliberately not opened | `decision_contract_consistency` |
| One load/merge output drift | `safe_merge=true`, zero adapter tensors remain, but `eval-001` changes only `$.should_reject` from `true` to `false` | `bf16_adapter_merge_stability` |

No observed failure is assigned to `data_coverage`. The training evidence
records zero truncation and monotonically decreasing validation loss, but
aggregate training evidence cannot prove semantic coverage. The correct
classification is therefore “not evidenced by the frozen v2 artifacts,” not
“coverage is sufficient.”

The canonical machine-readable result is
[`baseline/fc-mvp-001-lora-sft-v2-failure-classification.json`](../baseline/fc-mvp-001-lora-sft-v2-failure-classification.json),
with report digest
`sha256:671e4fad7e2b9987b0cbf3f3fdb078c11431efa5887109a204874ec136316a9a`.

## Locked next action

The single next gate is `FC-MVP-001-decision-compilation-v1`:

1. compile the terminal disposition fields from `selected_tool` under the
   existing v1 semantic contract;
2. fail closed on contradictions instead of treating independently generated
   booleans as separate authority; and
3. score the unchanged frozen v2 raw outputs exactly once.

Acceptance is zero conflicting flags, zero false refusals, zero dangerous
action candidates, zero dangerous false approvals, the same eval digest, and
byte-unchanged raw predictions. The gate may not add data, train, tune against
eval answers, use the merged artifact, or connect Runtime.

The merged artifact remains prohibited. Only the independently loaded adapter
may be used for this offline diagnostic; this does not make it Runtime
eligible.

## Reproduction

```powershell
python -I .\scripts\classify_tool_router_v2_failures.py
python -I .\scripts\validate_offline.py
```

The unified offline gate passes 67 tests on CPython 3.11.15, 3.12.12, and
3.13.7. Ruff passes the repository, and mypy reports no issues in all 25
source/script files.
