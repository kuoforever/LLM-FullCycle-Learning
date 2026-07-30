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
runtime_git_commit=8ace897f746a4aa3dd3f8b10af392ea9ba81941d
agent_contract_version=0.1.0
driver_contract_version=1.0.0
fullcycle_manifest_version=1
fullcycle_run_export_version=1
consumer_schema_version=1.0.0
reliability_dataset_schema_version=1
```

The consumer validates canonical manifest SHA-256 digests and fails closed on
schema, digest, size, or completeness violations. Runtime contract changes
must update the pin and compatibility fixtures together.

