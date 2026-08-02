# Desktop Runtime 依赖与集成

English: [docs/en/desktop-runtime-integration.md](docs/en/desktop-runtime-integration.md)

> `guarded-desktop-agent` 是本项目的可靠桌面环境和安全执行依赖，不是模型训练仓库。

## 依赖位置

```text
C:\Users\Alienware\guarded-desktop-agent
```

Runtime 的当前任务、冻结范围和新会话入口：

- `C:\Users\Alienware\guarded-desktop-agent\PROJECT_STATUS.md`
- `C:\Users\Alienware\guarded-desktop-agent\AGENTS.md`
- `C:\Users\Alienware\guarded-desktop-agent\CLAUDE.md`

规范集成契约：

- `C:\Users\Alienware\guarded-desktop-agent\docs\FULLCYCLE_INTEGRATION.md`

不要从聊天记录、分支名或 Runtime 的大路线图推断当前任务。

## 项目分工

| 能力 | Desktop Runtime | Full Cycle |
|---|---:|---:|
| UIA、截图、OCR、文档文本和桌面动作 | Owner | Consumer |
| Grounding、Policy、Approval、WAL、恢复 | Owner | 不得绕过 |
| 安全 Trace、Checkpoint 和运行指标 | Owner | Consumer |
| 多模态 Dataset 和 Registry | 提供边界 | Owner |
| VLM/LLM 训练和后训练 | 不负责 | Owner |
| vLLM Serving 和模型路由 | 不负责 | Owner |
| Agentic RL / Multi-Agent | 不负责 | Owner |
| 富训练轨迹的同意、脱敏、留存和删除 | 提供安全约束 | Owner，单独评审 |

## 两条数据通道

### Lane A：自动安全证据

数据来自现有脱敏 Trace、Checkpoint 和 reviewed tool registry。

用途：

- Runtime 可靠性和安全 Eval；
- Failure / Unknown Outcome 分类；
- Tool sequence、预算和恢复分析；
- Verifier 困难负样本；
- Runtime/Policy/Schema 兼容门禁。

不包含：

- 原始用户任务；
- 模型文本；
- Tool result 正文；
- Screenshot；
- Memory 或 Continuation。

因此不能用于 GUI Grounding 或多模态行为模仿训练。

### Lane B：显式同意的富训练轨迹

由 Full Cycle 侧独立设计 Capture Adapter：

- 默认关闭；
- 每次运行显式同意；
- 可见采集指示；
- 本地脱敏和图像遮罩；
- 独立目录、Retention 和 Delete；
- 记录 instruction、observation、candidate action、policy、result 和
  post-state；
- 使用状态 Verifier 标注成功，不接受模型自报。

Lane B 不得把 Runtime 的安全 Trace 改成秘密富日志。

## 跨仓库 Backlog

| ID | Owner | Status | 任务 |
|---|---|---|---|
| `GDA-FC-001` | Runtime | Complete | manifest v1 和 redacted run export v1 已实现并通过离线门禁 |
| `FC-BRIDGE-001` | Full Cycle | Complete | 严格 consumer、合法/非法 fixture 和兼容性失败行为已通过离线验证 |
| `FC-BRIDGE-002` | Full Cycle | Complete | Lane A 已确定性映射到版本化 Reliability/Verifier Dataset v1 |
| `FC-BRIDGE-003` | Full Cycle | Pending review | 设计 Lane B consent/capture/security contract |
| `FC-BRIDGE-004` | Both | Complete locally | Pin Runtime commit、contract version 和兼容性测试 |

### ID 对照

两个仓库各自维护一套 ID 且独立更新状态，因此漂移是结构性的，不是偶发的。
跨仓任务的对应关系如下，任何一侧改状态都必须同时核对另一侧：

| Full Cycle | Runtime | 同一件事 |
|---|---|---|
| `FC-BRIDGE-001` | `GDA-FC-002` | Full Cycle 侧的离线 consumer 与 fixture |
| `FC-BRIDGE-003` | `GDA-FC-003` | Lane B 显式同意采集契约 |
| `FC-BRIDGE-004` | `GDA-FC-004` | Pin 冻结与交接关闭 |

### 已解决的跨仓状态冲突

以下三条于 2026-07-31 核对 Runtime 工作区后记录，**均需在 Runtime 仓库内修
正**，本仓库不代改。三条已于 2026-08-01 在 Runtime 仓库内修正，记录见该仓库
`PROJECT_STATUS.md` 的「Cross-repository correction (2026-08-01)」小节：
`GDA-FC-002` 改为 `Complete`，`GDA-FC-004` 由 `Complete locally` 降为 `Next`
并将不可达的 `45bee82` 更正为其 squash merge `8ace897`。原始记录保留如下，
以便追溯：

1. Runtime `PROJECT_STATUS.md` 的 `GDA-FC-002` 仍为 `Next`，且其「Exact
   active task」整节指示下一个会话到本仓库来实现该 consumer；但本仓库
   `FC-BRIDGE-001` 已 Complete 并通过离线门禁。**新会话会被指向一件已完成
   的工作。**
2. Runtime `GDA-FC-004` 记为 `Complete locally`，完成证据引用 producer
   candidate `45bee82`；但在 Runtime 当前 HEAD（`7001375`）上
   `45bee82` **不是 HEAD 的祖先**（HEAD 的提交信息为 "Recalibrate HUD
   handoff commit identities after rebase"，该提交已被 rebase 移出历史）。
   本仓库固定的 `8ace897f` 经核实**是** HEAD 的祖先，因此本仓库的 pin 有
   效，需要更正的是 Runtime 侧的引用。
3. 同一件事在两侧状态不一致：Runtime `GDA-FC-004` 为 `Complete locally`，
   本仓库 `FC-BRIDGE-004` 为 `Pending`。在第 2 条的 commit 引用更正之前，
   不得按「已完成」处理。

两侧已于 2026-08-02 在同一次关闭变更中完成：`FC-BRIDGE-004` 与
`GDA-FC-004` 共同 pin 可从 Runtime 本地 `main` 到达的 commit
`324ff2fb5911e332ddb5c5f90eb41296e8faf7a9`。用该候选重新生成的 manifest 与
本仓库 `fixtures/bridge_v1` 固定的
`sha256:6abe3431ea0e6b4065f21e9a6c6fe34de772f9c3c86a2437f8d14f95a5d6f522`
逐字节一致，说明 Lane A 契约自 `8ace897` 以来没有漂移。新的冻结 pin 记录在
`baseline/runtime-freeze-v1.json`，旧 fixture 继续保留 `8ace897` 作为不可改写的
生成来源。

## Consumer Fixture 验收

`FC-BRIDGE-001` 必须：

- 完全离线；
- 不启动 Provider、MCP、Desktop 或网络；
- 校验 manifest/run export 版本和 digest；
- 拒绝未知字段、超大文件、错误 digest 和不完整事件；
- 明确标记 `training_use=reliability_and_verifier_only`；
- 保存一个最小合法 fixture 和多个非法 fixture；
- 在 Runtime contract 变化时自动失败。

当前 Producer 事实：

```text
agent_contract_version=0.1.0
driver_contract_version=1.0.0
fullcycle_manifest_version=1
fullcycle_run_export_version=1
trace_version=1
checkpoint_version=1
plan_contract_version=1
```

Runtime 的 manifest 必须通过 canonical JSON 重新计算 SHA-256，并与 run
bundle 的 `manifest_digest=sha256:<hex>` 精确匹配。Producer 已通过 PR
#219 合并并固定为：

```text
runtime_git_commit=8ace897f746a4aa3dd3f8b10af392ea9ba81941d
runtime_pull_request=219
```

`FC-BRIDGE-004` 的标准记录 `baseline/runtime-freeze-v1.json` 固定
`consumer_schema_version=1.0.0`、`reliability_dataset_schema_version=1`、
Runtime package `0.1.0`、所有 Lane A contract 版本和相同 manifest digest。
Runtime 在 CPython 3.13.7 上的 clean release preflight 通过 `1566 passed,
8 skipped`、Ruff、E1/E2、crash reconstruction、stateless replay 和 wheel
build/install；报告 SHA-256 为
`dc78f08030b4d3c4fac255a91fb7badf2b06fdb0eb0c487073e1f825260c6d0e`。
这只关闭离线 Runtime freeze，不新增 provider、desktop、application 或 release
证据。Lane B 明确 defer 到 `FC-BRIDGE-003` 独立评审，默认关闭。

## 新会话入口

在 Full Cycle 新会话中：

1. 阅读根 `README.md`；
2. 阅读本文件；
3. 阅读 `AI_Infra_LLM_Agent_待做任务清单.md`；
4. 若任务属于 Runtime，切换到 Runtime 仓库并以其
   `PROJECT_STATUS.md` 为唯一任务源；
5. 完成后更新对应 Backlog 状态和唯一下一任务。

## Pin 规则

`GDA-FC-004` / `FC-BRIDGE-004` 已在本地关闭。标准记录必须保留：

```text
runtime_git_commit
agent_contract_version
driver_contract_version
fullcycle_manifest_version
fullcycle_run_export_version
consumer_schema_version
validation_date
```

当前标准记录是 `baseline/runtime-freeze-v1.json`；不得用后续工作区或分支名
静默替换其中的精确 SHA。
