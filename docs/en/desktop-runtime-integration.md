# Desktop Runtime dependency and integration

中文：[Desktop_Runtime_依赖与集成.md](../../Desktop_Runtime_依赖与集成.md)

## Dependency

The Runtime repository is located at:

```text
C:\Users\Alienware\guarded-desktop-agent
```

Its own `PROJECT_STATUS.md` controls Runtime sequencing. This repository must
not infer Runtime state from chat history or branch names.

## Ownership

| Capability | Desktop Runtime | Full Cycle |
|---|---:|---:|
| UIA, screenshots, OCR, document text, desktop actions | Owner | Consumer |
| Grounding, policy, approval, WAL, recovery | Owner | Must not bypass |
| Safe traces, checkpoints, runtime metrics | Owner | Consumer |
| Multimodal datasets and registry | Boundary provider | Owner |
| Model post-training and serving | Not responsible | Owner |
| Agentic RL and multi-agent work | Not responsible | Owner |
| Consent/redaction/retention for rich episodes | Safety constraints | Owner and separate review |

## Two data lanes

Lane A is automatic redacted reliability evidence. It can support reliability
evaluation, failure classification, policy-denial analysis, recovery signals,
tool sequences, and verifier hard negatives. It excludes raw user tasks,
model text, tool-result bodies, screenshots, memory, and continuation data.

Lane B is explicitly consented rich training capture. It is off by default,
visibly indicated, locally redacted, independently retained/deleted, and
verified through post-state rather than model self-report. Lane B must never
turn the Runtime safety trace into a hidden rich log.

## Current pins

```text
runtime_git_commit=324ff2fb5911e332ddb5c5f90eb41296e8faf7a9
agent_contract_version=0.1.0
driver_contract_version=1.0.0
fullcycle_manifest_version=1
fullcycle_run_export_version=1
consumer_schema_version=1.0.0
reliability_dataset_schema_version=1
```

The canonical record is `baseline/runtime-freeze-v1.json`. The immutable
`FC-BRIDGE-001` fixture still names `8ace897` as its generation provenance;
the later freeze pin does not rewrite that dated evidence.

The consumer validates canonical manifest SHA-256 digests and fails closed on
schema, digest, size, or completeness violations. Runtime contract changes
must update the pin and compatibility fixtures together.

`runtime_git_commit=8ace897f` was re-verified on 2026-07-31 as an ancestor of
the Runtime working tree HEAD, so this pin is valid. Three cross-repository
status conflicts found during the same check had to be corrected in the Runtime
repository, not here; they are recorded in
[Desktop_Runtime_依赖与集成.md](../../Desktop_Runtime_依赖与集成.md) under
「已解决的跨仓状态冲突」.

All three were corrected in the Runtime repository on 2026-08-01, under
`PROJECT_STATUS.md` section "Cross-repository correction (2026-08-01)":
`GDA-FC-002` became `Complete`, and `GDA-FC-004` was demoted from
`Complete locally` to `Next` with the unreachable `45bee82` replaced by its
squash merge `8ace897`. That correction established the safe resume point for
the freeze completed below.

The same check regenerated a manifest from the Runtime HEAD and reproduced the
digest pinned in `fixtures/bridge_v1` byte for byte, so the Lane A contract has
not drifted since `8ace897`, so the immutable fixture needs no change.
`FC-BRIDGE-004` completed locally on 2026-08-02: both repositories pin
branch-reachable Runtime commit
`324ff2fb5911e332ddb5c5f90eb41296e8faf7a9`. Its CPython 3.13.7 clean release
preflight passed `1566` tests with `8` skips, all independent offline gates,
and clean wheel build/install. Lane B is explicitly deferred to the separate
`FC-BRIDGE-003` consent, security, and privacy review and remains disabled by
default. This does not add provider, desktop, application, or release evidence.
