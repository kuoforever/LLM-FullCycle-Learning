# Writing and execution template guide

中文权威版：[AI_Infra_LLM_Agent_写作与执行模块模板.md](../../AI_Infra_LLM_Agent_写作与执行模块模板.md)

The source file contains copy-ready Chinese templates used during
implementation. This English companion maps their purpose and required
structure without duplicating a thousand-line operational template.

## Template catalog

| Module | Required content |
|---|---|
| Project overview | Goals, problem, users/roles, system loop, success criteria, non-goals |
| Scenario specification | Modality/environment, observation/action contracts, adapter, dataset, verifier, eval, failures, demo |
| Knowledge note | Definition, motivation, principles, formulas/data flow, implementation, failures, evaluation, mastery state |
| Task specification | Context, known facts, goal, non-goals, allowed scope, requirements, prohibitions, deliverables, validation, DoD |
| Codex implementation task | Required reading, bounded edits, quality gates, commands, and handoff format |
| Architecture/code review | Correctness, model/data, infrastructure/safety, engineering quality, evidence, blocking findings |
| Experiment record | Question, hypothesis, variables, environment, data, config, commands, results, bad cases, limits |
| Dataset card | Purpose, provenance/license, schema, cleaning, PII, quality, split, leakage, bias, deletion, versions |
| Model card | Base model, objective, data, method, config, tasks, evaluation, safety, resources, limitations, rollback |
| ADR | Context, decision, alternatives, rationale, consequences, risks, reconsideration triggers |
| Evaluation report | Dataset/model versions, metrics, environment, grouped results, ablations, cost, safety, gate decision |
| Failure experiment | Fault, expected behavior, injection, observed state, side effects, evidence, recovery, root cause, regression |
| Completion record | Goal, implementation, files, tests, metrics, reviews, limitations, follow-up, Git identity |
| Portfolio extraction | Positioning, responsibility, design, difficulty, verified metrics, evidence links, claim boundaries |

## Usage rules

- Start from `PROJECT_STATUS.md`, then read only the active checklist section.
- Keep one task ID and one primary variable per implementation unit.
- Use exact executable validation commands.
- Separate automated evidence, human acceptance, and unverified assumptions.
- Record failures and negative results, not only successful demonstrations.
- Never turn a template into a competing roadmap or status tracker.

