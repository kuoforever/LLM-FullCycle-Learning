# Multimodal LLM full-cycle MVP roadmap

中文权威版：[多模态 LLM 全周期 MVP 演进路线](../../多模态LLM全周期_MVP演进路线.md)

This companion preserves the roadmap structure and decision boundaries. The
Chinese source remains authoritative for detailed schemas, metrics, and exit
criteria.

## Scope

The project demonstrates a post-training-to-deployment lifecycle rather than
claiming full foundation-model pretraining at scale. Desktop GUI is the first
environment, not a permanent product boundary. Reliable execution remains a
separate deterministic layer.

## System loop

```text
data and traces → governance → post-training → evaluation
→ serving and routing → reliable runtime → bad-case review
→ dataset/model revision
```

## MVP sequence

| MVP | Primary increment | Exit principle |
|---|---|---|
| 0 | Freeze Runtime behavior and evidence | Reproducible safety/recovery baseline |
| 1 | Text Tool Router | Dataset, training, frozen eval, and safety-gated candidate output |
| 2 | Image-text GUI Action Model | Screenshot/UIA/OCR input, grounding, risk, and fallback |
| 3 | Multimodal post-training and verifier | SFT/distillation/preference comparisons and trajectory gates |
| 4 | Serving and MLOps | vLLM, quantization, cache, routing, rollout, and rollback evidence |
| 5 | Agentic RL | Runtime-backed environment and verifiable rewards |
| 6 | Multiple environments/modalities | Reuse contracts across documents, browser, media, or simulation |
| 7 | AI infrastructure depth | Distributed training, recovery, profiling, and inference optimization |
| 8 | Multi-agent systems | Typed delegation, durable state, leases, conflicts, and single-agent control |

## Change discipline

- Introduce one primary variable per stage.
- Freeze evaluation inputs before comparing models.
- Bind code, data, model, config, seed, hardware, metrics, and failures.
- Do not soften Runtime policy or approval contracts to improve model metrics.
- Do not describe planned capability as implemented.

## Portfolio slices

The same evidence can support role-specific narratives:

- multimodal/post-training;
- applied LLM and agents;
- AI infrastructure and serving;
- ML systems and training;
- multi-agent/distributed agents.

Each narrative must use only completed evidence recorded in
[PROJECT_STATUS](../../PROJECT_STATUS.md).

