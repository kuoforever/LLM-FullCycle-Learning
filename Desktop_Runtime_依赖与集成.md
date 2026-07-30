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
| `FC-BRIDGE-004` | Both | Pending | Pin Runtime commit、contract version 和兼容性测试 |

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

`FC-BRIDGE-004` 仍需在 consumer 测试通过后补齐
`consumer_schema_version` 和跨仓兼容性结果。

Full Cycle 本地侧已固定 `consumer_schema_version=1.0.0`、
`reliability_dataset_schema_version=1` 和 Runtime commit `8ace897f`，并在
Python 3.11-3.13 本地矩阵通过。Full Cycle 远程 Actions run
`30369941536` 也通过三版本矩阵；consumer 侧 pin 与兼容门禁已具备远程证据。
`FC-BRIDGE-004` 仍需 Runtime 侧记录/确认该 consumer 结果后共同关闭。

## 新会话入口

在 Full Cycle 新会话中：

1. 阅读根 `README.md`；
2. 阅读本文件；
3. 阅读 `AI_Infra_LLM_Agent_待做任务清单.md`；
4. 若任务属于 Runtime，切换到 Runtime 仓库并以其
   `PROJECT_STATUS.md` 为唯一任务源；
5. 完成后更新对应 Backlog 状态和唯一下一任务。

## Pin 规则

在 `GDA-FC-004` 完成前，不把当前分支或工作区描述为稳定 Runtime
版本。关闭时必须记录：

```text
runtime_git_commit
agent_contract_version
driver_contract_version
fullcycle_manifest_version
fullcycle_run_export_version
consumer_schema_version
validation_date
```
