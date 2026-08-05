# Project documentation index

中文：[README.zh-CN.md](README.zh-CN.md)

The repository keeps project descriptions, implementation evidence,
maintenance contracts, and documents used during development. English-native
technical reports are shared directly; Chinese operational sources have
English companion documents.

## Project and maintenance

| Document | English default | Chinese / source |
|---|---|---|
| Main README | [README.md](../README.md) | [README.zh-CN.md](../README.zh-CN.md) |
| Project status | [PROJECT_STATUS](../PROJECT_STATUS.md) | English source |
| Desktop Runtime integration | [English companion](en/desktop-runtime-integration.md) | [Chinese source](../Desktop_Runtime_依赖与集成.md) |
| Agent instructions | [AGENTS](../AGENTS.md) | English source |

## Implemented technical evidence

| Topic | English default | Chinese / source |
|---|---|---|
| Runtime bridge consumer | [English version](en/FC-BRIDGE-001.md) | [Chinese source](FC-BRIDGE-001.md) |
| Runtime freeze pin | [Desktop Runtime integration](en/desktop-runtime-integration.md) | [Canonical record](../baseline/runtime-freeze-v1.json) |
| Lane A reliability dataset | [English version](en/adr/ADR-0001-lane-a-reliability-dataset-v1.md) | [Chinese source](adr/ADR-0001-lane-a-reliability-dataset-v1.md) |
| Standard-library offline baseline | [ADR-0002](adr/ADR-0002-stdlib-offline-baseline.md) | English source |
| Environment baseline | [environment.md](environment.md) | English source |
| Tool Router schema/eval | [FC-MVP-001 schema/eval](FC-MVP-001-schema-eval.md) | English source |
| Base-model baseline | [FC-MVP-001 base model](FC-MVP-001-base-model-v1.md) | English source |
| LoRA SFT v1 | [FC-MVP-001 LoRA SFT](FC-MVP-001-lora-sft-v1.md) | English source |
| Safety-repair data v2 | [FC-MVP-001 safety repair](FC-MVP-001-safety-repair-data-v2.md) | English source |
| LoRA SFT v2 | [FC-MVP-001 LoRA SFT v2](FC-MVP-001-lora-sft-v2.md) | English source |
| LoRA SFT v2 failure classification | [FC-MVP-001 failure classification](FC-MVP-001-v2-failure-classification.md) | English source |
| Decision compilation v1 | [FC-MVP-001 decision compilation](FC-MVP-001-decision-compilation-v1.md) | English source |
| BF16 merge stability v1 | [FC-MVP-001 BF16 merge stability](FC-MVP-001-bf16-merge-stability-v1.md) | English source |
| BF16 merge numerics v1 | [FC-MVP-001 BF16 merge numerics](FC-MVP-001-bf16-merge-numerics-v1.md) | English source |
| BF16 merge remediation v1 | [FC-MVP-001 BF16 merge remediation](FC-MVP-001-bf16-merge-remediation-v1.md) | English source |
| FP32 merge drift analysis v1 | [FC-MVP-001 FP32 merge drift analysis](FC-MVP-001-fp32-merge-drift-analysis-v1.md) | English source |
| FP32 attached/merge isolation v1 | [FC-MVP-001 FP32 attached/merge isolation](FC-MVP-001-fp32-attached-merge-isolation-v1.md) | English source |
| FP32 attached/merge numerics v1 | [FC-MVP-001 FP32 attached/merge numerics](FC-MVP-001-fp32-attached-merge-numerics-v1.md) | English source |
| Attached BF16/FP32 dtype isolation v1 | [FC-MVP-001 attached dtype isolation](FC-MVP-001-attached-dtype-isolation-v1.md) | English source |
| Attached BF16/FP32 dtype numerics v1 | [FC-MVP-001 attached dtype numerics](FC-MVP-001-attached-dtype-numerics-v1.md) | English source |
| Attached BF16/FP32 dtype boundary control v1 | [FC-MVP-001 attached dtype boundary control](FC-MVP-001-attached-dtype-boundary-control-v1.md) | English source |
| FP32 attached remediation eval v1 | [FC-MVP-001 FP32 attached remediation eval](FC-MVP-001-fp32-attached-remediation-eval-v1.md) | English source |
| FP32 attached artifact eligibility review v1 | [FC-MVP-001 FP32 attached artifact eligibility review](FC-MVP-001-fp32-attached-artifact-eligibility-review-v1.md) | English source |

## Development references

| English companion | Chinese authoritative source |
|---|---|
| [Roadmap companion](en/mvp-roadmap.md) | [MVP roadmap](../多模态LLM全周期_MVP演进路线.md) |
| [Scenario companion](en/scenario-coverage.md) | [Scenario coverage matrix](../多模态与业务场景覆盖矩阵.md) |
| [Task map](en/task-checklist.md) | [Task checklist](../AI_Infra_LLM_Agent_待做任务清单.md) |
| [Template guide](en/writing-execution-templates.md) | [Writing and execution templates](../AI_Infra_LLM_Agent_写作与执行模块模板.md) |

`PROJECT_STATUS.md` remains the source of truth for sequencing and the single
active objective. The Chinese checklist remains authoritative for exact task
acceptance wording; its English companion is a stable navigation and usage
guide, not a second tracker.
