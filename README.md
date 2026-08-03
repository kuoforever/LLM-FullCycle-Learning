# Reliable Agent Model Lifecycle

> A complete target system for multimodal data, post-training, evaluation,
> serving, reliable agent execution, and bad-case-driven model iteration.

中文：[README.zh-CN.md](README.zh-CN.md)

## Documentation

| Document | Purpose |
|---|---|
| [Project status](PROJECT_STATUS.md) | Current phase, single active objective, and latest validation evidence |
| [Documentation index](docs/README.md) | Implemented evidence, maintenance contracts, and development references |
| [Task checklist](docs/en/task-checklist.md) | English map of task IDs, dependencies, and Definition of Done |
| [Desktop Runtime integration](docs/en/desktop-runtime-integration.md) | Cross-repository ownership, safety boundaries, and version pins |

## Positioning

`Reliable Agent Model Lifecycle` names the complete system being built, not a
single MVP. It is not tied to autonomous driving or one business scenario.
Desktop GUI is the first verifiable environment; the architecture extends to
documents, browsers, charts, audio/video, and optional simulation as the
corresponding stages are implemented.

The goal is a reproducible lifecycle:

```text
multimodal data and agent traces
→ cleaning, redaction, quality, and versioning
→ SFT / QLoRA / distillation / preference optimization
→ action model / tool router / retriever / verifier
→ quantization / serving / routing / fallback
→ reliable agent runtime
→ traces / evaluation / bad cases / human review
→ dataset vN+1 / model vN+1
```

This repository owns model and dataset work. The separate
`guarded-desktop-agent` repository owns execution policy, approvals,
grounding, write-ahead logging, budgets, recovery, and the sole desktop
boundary. Models may propose actions but never bypass those controls.

## One flagship project and four depth labs

| Area | What it demonstrates |
|---|---|
| Flagship lifecycle | Data, post-training, evaluation, serving, runtime integration, and bad-case feedback |
| Tiny Transformer & Pretraining | Decoder internals, operator graphs, MHA/MQA/GQA, RoPE, KV cache, continued pretraining, and resumable training |
| Multimodal Post-training & Agentic RL | QLoRA, DPO/GRPO, verifiable rewards, verifier models, and ablations |
| Distributed Training & Inference | DDP/FSDP, collectives, vLLM, quantization, profiling, and correctness-gated Triton kernel experiments |
| Multi-Agent Systems | Coordination, typed handoffs, leases, conflict handling, recovery, and single-agent controls |

Only capabilities backed by code, tests, metrics, and artifacts are described
as implemented. Planned labs remain roadmap items.

The existing `FC-*` task IDs, `fullcycle_*` contract fields, Python package,
and CLI names remain unchanged for compatibility with frozen artifacts.

## MVP progression

Each MVP keeps a vertical loop and adds one primary variable:

| Version | Goal |
|---|---|
| MVP-0 | Freeze the reliable execution baseline |
| MVP-1 | Close the text Tool Router loop |
| MVP-2 | Add a screenshot/UIA/OCR GUI action model |
| MVP-3 | Add multimodal post-training and a verifier |
| MVP-4 | Add multi-model serving, quantization, routing, and rollout controls |
| MVP-5 | Use the Runtime as an Agentic RL environment |
| MVP-6 | Add environments and modalities |
| MVP-7 | Deepen model architecture, operator/kernel, distributed training, and inference systems work |
| MVP-8 | Add reliable multi-agent coordination |

See the [English roadmap companion](docs/en/mvp-roadmap.md) and
[scenario coverage companion](docs/en/scenario-coverage.md).

## Four gates

Every version must pass:

1. Functional gate: the new capability works on fixed tasks.
2. Regression gate: existing behavior and safety contracts stay intact.
3. Safety gate: false approvals, unauthorized actions, and duplicate side
   effects do not increase.
4. Performance gate: memory, latency, throughput, and cost stay within budget.

An experiment must bind the code commit, dataset version, model version,
configuration, seed, hardware, evaluation report, serving benchmark, failure
report, and demonstration evidence.

## Implemented evidence

### Runtime Lane A bridge

`FC-BRIDGE-001` implements a strict offline consumer for versioned Runtime
manifests and redacted run exports. It validates schema versions and digests,
rejects unknown or incomplete inputs, and never starts the provider, MCP,
desktop, or network layers. See [FC-BRIDGE-001](docs/en/FC-BRIDGE-001.md).

The offline Runtime dependency is frozen locally at
`324ff2fb5911e332ddb5c5f90eb41296e8faf7a9`; the exact contract and preflight
pin is retained in `baseline/runtime-freeze-v1.json`. Lane B rich capture is
not part of that freeze and remains deferred, separately reviewed, and off by
default.

### Reliability/Verifier Dataset v1

`FC-BRIDGE-002` deterministically maps accepted Runtime evidence to canonical
JSONL. Version 1 emits only signals supported by Runtime facts: failure,
unknown outcome, policy denial, recovery, budget limits, and tool
sequence/outcome features. See [ADR-0001](docs/en/adr/ADR-0001-lane-a-reliability-dataset-v1.md).

### Tool Router schema and evaluation

`FC-MVP-001` defines Tool Router decision schema v1, 20 reviewed seed records,
20 frozen evaluation records, and 200 train/validation records across 60
explicit task families. Offline audits enforce family separation, duplicate
and near-duplicate rejection, distribution checks, and dangerous-action
checks. See [schema/eval gate](docs/FC-MVP-001-schema-eval.md).

### Local base-model baseline

The baseline pins `Qwen/Qwen2.5-1.5B-Instruct`, its Hub revision, inference
configuration, raw predictions, and an independent scorer. Measured on the
frozen 20-case evaluation set, JSON validity was 1.0, but tool accuracy was
0.20 and both dangerous cases produced dangerous action candidates. The model
is explicitly not Runtime eligible. See
[base-model baseline](docs/FC-MVP-001-base-model-v1.md).

### First local LoRA SFT

BF16 LoRA SFT on the frozen 160/40 train/validation data improved tool
accuracy from 0.20 to 0.80, argument exact match from 0.00 to 0.35, and risk
Macro F1 from 0.4258 to 0.7373, measured on the same unchanged 20-case
evaluation set with the same independent scorer. One dangerous-action
candidate remained, so `safety_gate_passed=false` and
`runtime_eligible=false`. The repository keeps the independent adapter,
configuration, raw predictions, reports, and safe-merge verification. See
[LoRA SFT v1](docs/FC-MVP-001-lora-sft-v1.md).

### Safety-repair data gate

The frozen SFT v1 bad cases are classified into four repair targets without
copying eval answers into training data. A reviewed train/validation-only v2
increment adds 16 train and eight validation examples across eight disjoint
families. The combined 176/48 data preserves v1 as an exact prefix, keeps the
20-case eval digest unchanged, and passes fail-closed leakage and
dangerous-action audits. No retraining occurred in this gate. See
[safety-repair data v2](docs/FC-MVP-001-safety-repair-data-v2.md).

### Safety-repair LoRA SFT v2

The locked three-epoch v2 run trained locally on the passed 176/48 data and
scored once on the unchanged eval. Tool accuracy reached 0.95 and dangerous
action candidates fell to zero. The adapter is still not Runtime eligible:
three decisions contain conflicting flags, three are false refusals, and a
safe merge changed one generated boolean on `eval-001`. These failures are
frozen as evidence rather than waived. See
[LoRA SFT v2](docs/FC-MVP-001-lora-sft-v2.md).

### LoRA SFT v2 failure classification

The frozen v2 evidence assigns the three conflicting decisions and the
count-aligned false refusals to decision-contract consistency. The one-field
load/merge drift is a separate BF16 adapter-merge stability failure. The
frozen artifacts do not justify a data-coverage diagnosis. One offline-only
decision-compilation gate is locked next; no v3 data or training has started.
See [failure classification](docs/FC-MVP-001-v2-failure-classification.md).

### Scale boundary of the current model evidence

Every model number above comes from a single RTX 4090 Laptop GPU, a 1.5B
base model, LoRA rank 16 over Q/K/V/O projections, 100 optimizer steps, and a
20-case evaluation set built as ten categories with two cases each. These
results are reproducible and were produced under frozen data and evaluation
contracts, but at this sample size they indicate direction only and do not
establish generalization. They are not a claim about large-scale pretraining,
post-training, or serving infrastructure, none of which is implemented.

## Reproducible offline gate

```powershell
python -I .\scripts\validate_offline.py
```

The gate validates frozen artifact hashes, dependency boundaries, unit tests,
bridge fixtures, and exact dataset JSONL on Python 3.11–3.13. Environment
details are in [environment.md](docs/environment.md).

## Current boundary

The current work is `FC-MVP-001-decision-compilation-v1`. Compile terminal
disposition fields under the existing v1 semantic contract, fail closed on
contradictions, and score the unchanged frozen v2 raw outputs exactly once.
No new data, training, merged artifact, eval-answer tuning, or Runtime
integration is allowed in this gate.
