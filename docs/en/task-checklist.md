# Multimodal LLM × Agent Infra task checklist

中文权威版：[AI_Infra_LLM_Agent_待做任务清单.md](../../AI_Infra_LLM_Agent_待做任务清单.md)

This document is the English navigation companion for the detailed checklist.
It intentionally does not create a second status tracker. Exact requirements
and acceptance wording remain in the Chinese source; sequencing and the single
active objective come from [PROJECT_STATUS](../../PROJECT_STATUS.md).

## Workstreams

| IDs | Workstream |
|---|---|
| `ENV-*` | WSL, Python, CUDA, and experiment-tracking baseline |
| `TT-*` | Tiny decoder-only Transformer, architecture/operator graph, generation, KV cache, training/recovery, profiling, and one verified kernel experiment |
| `DATA-*`, `CPT-*` | Data licensing, parsing, cleaning, deduplication, packing, continued pretraining |
| `MM-*`, `TOOL-*` | Multimodal traces, GUI grounding, Tool Router, SFT, distillation, preference optimization |
| `RET-*` | Embeddings, retrieval baselines, fine-tuning, reranking, joint evaluation |
| `VER-*` | Trajectory schemas, hard negatives, verifier/reward models, calibration, release gates |
| `SERV-*`, `MLOPS-*` | Gateway, vLLM, quantization, concurrency, cache, overload control, packaging and cold start, multi-LoRA hot swap, constrained decoding, scheduler tuning, capacity/SLO/cost, performance gates, tiered degradation, registry, rollout |
| `RUN-*` | Reliable Runtime contracts, state, recovery, idempotency, policy, memory, observability |
| `MA-*` | Identity, typed handoffs, coordination, durable state, leases, budgets, verification |
| `SCN-*` | Desktop, documents, browser, coding, enterprise, DevOps, security, media, simulation |
| `EVAL-*` | Frozen evaluation sets, unified reports, and evidence-backed portfolio metrics |

## Execution order

1. Establish environment and reproducibility.
2. Complete one text Tool Router vertical loop.
3. Add image-text GUI actions without weakening Runtime controls.
4. Add post-training comparisons and verifier gates.
5. Add serving, deployment/inference optimization, MLOps, and bad-case feedback.
6. Expand environments, infrastructure depth, and multi-agent work only after
   the earlier gates have evidence.

## Definition of Done

A task is not complete until it has:

- code and versioned artifacts;
- deterministic or appropriately repeated tests;
- recorded metrics and raw evidence;
- dependency, privacy, and safety checks;
- failure/bad-case documentation;
- exact reproduction commands;
- a clear statement of limitations and the next objective.

Planned functionality must remain marked planned until all required evidence
exists.

