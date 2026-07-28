# ADR-0001：Lane A Reliability/Verifier Dataset v1

- Status: Accepted
- Date: 2026-07-28
- Task: `FC-BRIDGE-002`

## Context

Runtime Lane A 只提供脱敏的 checkpoint、事件、工具安全摘要和公开契约版本。
它明确不提供原始任务、模型文本、tool-result 正文、图像、Memory 或
Continuation。Dataset mapping 必须利用这些事实服务可靠性评测和 Verifier，
但不能把缺失语义伪装成训练标签。

## Decision

采用“一份已验证 run-export snapshot 对应一条 record”的 canonical JSONL
schema，版本为：

```text
reliability_dataset_schema_version=1
```

机器可读结构由 `schemas/reliability_dataset_v1.schema.json` 固定，所有对象层
均声明 `additionalProperties=false`。

所有输入必须先通过 `fullcycle_bridge.consumer`。Mapper 不接受未验证 dict，
CLI 也不提供跳过校验的选项。

Record 分为四部分：

1. `source`：Runtime/consumer/schema pin、manifest digest、run ID、policy
   version，以及 canonical run-export SHA-256；
2. `features`：phase、recovery、failure code、事件计数、tool sequence、
   tool outcomes、policy decisions、recovery events 和预算使用；
3. `labels`：仅包含可由 Lane A 事实确定的标签；
4. 原样保留 `data_class` 与 `training_use` 安全限制。

`record_id` 对以下 canonical identity 求 SHA-256：

```text
reliability_dataset_schema_version
+ manifest_digest
+ run_export_digest
```

因此同一 run ID 的不同 checkpoint snapshot 不会碰撞，schema 升级也会生成
新的 record ID。

## Deterministic label rules

| Label | v1 规则 |
|---|---|
| `outcome_class` | terminal phase 映射为 success/failure/unknown_outcome/cancelled/paused；其他为 in_progress |
| `is_failure` | phase 为 `FAILED` 或 `UNKNOWN_OUTCOME` |
| `is_unknown_outcome` | checkpoint phase 为 unknown，或任一 tool result 的 status/dispatch 为 unknown |
| `policy_denied` | 存在 `decision=deny` 或 `code=POLICY_DENIED` |
| `recovery_required` | recovery status 非 ready、recovery action 要求检查/重观察，或 recovery event 非 ready |
| `budget_limit_hit` | 任一正数 limit 的 used 值达到 limit |

`tool_sequence` 与 `tool_outcomes` 按原始连续 event sequence 保序，不进行工具
名称改写或语义补全。`verifier_tags` 使用固定顺序，避免集合迭代造成输出漂移。

## Rejected alternatives

- 不从 `failure_code` 猜测 `should_retry` 或 `should_rollback`：这需要独立的
  policy/operational contract。
- 不输出 `unsafe_action=true`：Policy denial 是可观察事实，但不能证明候选
  动作在所有上下文中语义不安全。
- 不生成 pairwise preference、最终答案质量或“工具失败但声称成功”标签：
  Lane A 没有模型文本或最终答案。
- 不做困难负样本扰动：删除步骤、打乱顺序或替换工具属于后续冻结 Eval
  设计，不能混入原始 evidence mapping。
- 不引入第三方 dataframe/schema library：v1 规模下标准库 canonical JSONL
  更容易离线复现。

## Consequences

优点：

- 输出完全确定、内容寻址、可逐字节回归；
- producer contract 变化会先在 bridge gate 失败；
- dataset 不扩大 Lane A 数据权限。

限制：

- 这是 reliability/Verifier feature dataset，不是 SFT episode；
- 标签是规则生成的 operational signals，不是人工审阅 ground truth；
- 还没有 train/validation/test split、sampling、class balance 或模型 baseline；
- cancelled/paused 是否属于特定训练任务的负类，应由后续 Eval contract 决定。
