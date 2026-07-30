# Multimodal and business-scenario coverage

中文权威版：[多模态与业务场景覆盖矩阵](../../多模态与业务场景覆盖矩阵.md)

## Orthogonal dimensions

Every scenario is described across five independent dimensions:

1. Modality: text, structured UI, image, document/chart, audio/video, or
   simulation state.
2. Environment: desktop, browser, document workspace, developer tools,
   enterprise systems, or simulation.
3. Model task: routing, grounding, action prediction, retrieval, verification,
   reward modeling, or planning.
4. Business goal: productivity, research, coding, operations, security,
   media understanding, or physical-world transfer.
5. Execution topology: single agent, model router, verifier-gated agent, or
   coordinated multi-agent system.

## Unified scenario contract

A scenario must define:

- identity, owner, version, and status;
- input observation and allowed action contracts;
- environment adapter and Runtime boundary;
- dataset source, consent, redaction, and retention;
- state-based success verifier;
- fixed evaluation set and metrics;
- failure taxonomy and safety gates;
- reproducible demo evidence.

## Delivery waves

| Wave | Purpose | Representative scenarios |
|---|---|---|
| 1 | Prove the vertical loop | Desktop GUI and Tool Router |
| 2 | Prove cross-environment reuse | Documents/charts, browser research, coding |
| 3 | Prove enterprise safety and systems depth | Workflow/data, DevOps, security |
| 4 | Prove native multimodality and transfer | Audio/video and optional robotics/driving simulation |
| Project H | Prove coordination | Multi-agent coding with a fixed single-agent control |

## Reuse rules

Scenarios reuse versioned observation, action, trace, evaluation, and safety
contracts. They do not bypass policy, approval, budgets, grounding, or
write-ahead logging. Rich multimodal capture is opt-in and separate from
automatic redacted reliability evidence.

## Selection rule

Prefer a scenario only when it adds a new measurable capability, reuses the
existing vertical loop, has deterministic acceptance evidence, and does not
expand multiple uncontrolled variables at once.

