# 多模态 LLM 全周期：MVP 演进路线

English companion: [docs/en/mvp-roadmap.md](docs/en/mvp-roadmap.md)

> 目标：以现有 Reliable Runtime 为执行底座，逐步构建可训练、可评测、可部署、可恢复、可回流的多模态 Agent 系统。每个阶段都必须能独立演示和写入阶段性简历，不等待“全部完成”才交付价值。

## 1. 总体边界

### 1.1 项目解决什么

- 从截图、UIA、OCR、文档、文本和动作历史构造多模态训练数据；
- 训练模型完成 GUI Grounding、工具选择、参数生成、风险判断和强模型回退；
- 使用 SFT、蒸馏、DPO/KTO、GRPO 和 Verifier 改进真实 Agent 行为；
- 通过 vLLM 提供量化、缓存、路由、限流和可观测的推理服务；
- 通过 Reliable Runtime 保证模型输出不会直接获得执行权限；
- 将真实失败转化为 Badcase、训练数据和固定回归用例。

### 1.2 项目不声称什么

- 不把本机小规模实验描述为大规模基础模型预训练；
- 不把 Teacher API 输出描述为 logits 蒸馏；
- 不把单卡实验描述为多机多卡生产经验；
- 不把 GUI 场景描述为自动驾驶或机器人量产经验；
- 不把模型生成的“成功”当作任务成功证据；
- 不为了覆盖关键词而同时引入多个无法归因的变量。

## 2. 统一系统链路

```text
Instruction
  ↓
Observation Builder
  ├── UIA / document text
  ├── screenshot / region
  ├── OCR
  └── action history / policy context
  ↓
Local Multimodal Policy
  ├── next_tool
  ├── arguments / bbox / ref
  ├── risk_level
  ├── requires_approval
  ├── confidence
  └── should_fallback
  ↓
Reliable Runtime
  ├── schema validation
  ├── grounding / policy / approval
  ├── budget / idempotency / WAL
  ├── MCP action
  └── mandatory re-observation
  ↓
Verifier / Grader
  ├── task progress
  ├── unsafe action
  ├── evidence consistency
  └── retry / stop / fallback
  ↓
Trace → Eval → Badcase → Human Review → Dataset vN+1
```

### 2.1 自动驾驶类比只作为架构解释

| 控制系统层 | 本项目组件 | 权限边界 |
|---|---|---|
| 传感器 | UIA、OCR、截图、文档、音视频 | 只观察 |
| 感知 | VLM、Grounding、状态抽取 | 产出候选事实 |
| 世界状态 | Runtime state、Memory、动作历史 | Durable state 为真相 |
| 行为决策 | Tool Router / Action Model | 只提出候选动作 |
| 高层规划 | 强模型或分层 Planner | 不能直接执行 |
| 安全壳 | Policy、Approval、Budget、Verifier | 确定性代码掌权 |
| 执行器 | MCP、UIA、鼠标键盘 | 唯一执行边界 |
| 数据闭环 | Trace、Eval、Badcase、再训练 | 人工审核后升级 |

## 3. MVP-0：冻结可靠执行基线

### 目标

将 `guarded-desktop-agent` 作为稳定依赖，而不是继续横向扩张。

### 必须保留

- 当前完整测试基线：`1420 passed, 7 skipped`；
- UIA、截图、区域图、OCR 和文档文本观察；
- Policy、Approval、Grounding、预算和审计；
- 动作后强制重新观察；
- Crash/Resume、幂等和 Unknown Outcome；
- 固定的故障注入和 Trace 证据。

### 退出条件

- 固定版本、依赖和运行命令；
- 创建模型接入接口，但不允许绕过现有 Runner/MCP；
- 输出一份 Runtime Capability Manifest；
- 后续模型实验失败不能破坏 Runtime 基线。

## 4. MVP-1：文本 Tool Router 闭环

### 目标

先证明最短的模型全周期，不让视觉数据处理阻塞闭环。

### 数据 Schema

```json
{
  "instruction": "...",
  "available_tools": [],
  "state_summary": {},
  "selected_tool": "...",
  "arguments": {},
  "risk_level": "low|medium|high",
  "requires_approval": true,
  "should_reject": false,
  "should_fallback": false,
  "expected_result": {}
}
```

### 数据

- 200–500 条人工种子与可验证合成样本；
- 覆盖正常、歧义、缺参数、无工具、安全、回退和多步任务；
- 导入现有 Agent Trace，脱敏并记录 runtime/model/version；
- 训练集、验证集、测试集按任务族切分，测试集冻结。

### 训练与评测

- Prompt-only / Base / QLoRA SFT 对照；
- Tool Accuracy；
- JSON Schema Validity；
- Argument Exact Match / Field F1；
- Risk Macro F1；
- Approval、Refusal 和 Fallback 指标；
- 误批准危险动作数必须为 0。

### 退出条件

```text
Trace → Dataset → Train → Eval → Runtime → Badcase
```

整条链路可一键复现，并有至少一个真实 Badcase 回流。

## 5. MVP-2：图文 GUI Action Model

### 目标

将文本 Router 升级为视觉—语言—动作模型，但保持相同安全边界。

### 输入

- 用户指令；
- 全屏或局部截图；
- UIA/document/OCR 结构化观察；
- 最近动作及动作结果；
- 当前 Policy 和可用工具。

### 输出

```json
{
  "next_tool": "click",
  "arguments": {
    "ref": "ref_12",
    "bbox": [0, 0, 0, 0]
  },
  "risk_level": "low",
  "requires_approval": false,
  "confidence": 0.0,
  "should_fallback": false,
  "evidence": []
}
```

### 关键实验

- Screenshot-only vs UIA-only vs Screenshot+UIA；
- 全屏 vs 局部裁剪；
- OCR 有无；
- 动作历史长度；
- Ref grounding vs coordinate grounding；
- 低置信度回退阈值；
- 0.5B–3B 小型 VLM 的质量、显存和延迟。

### 指标

- Action Accuracy；
- Grounding Accuracy / IoU；
- Tool/Argument Accuracy；
- Task Success Rate；
- 平均步骤数；
- 重复动作率；
- 强模型回退率；
- 误批准、越权和重复副作用数；
- 每任务延迟、显存和 Token/图像成本。

### 退出条件

- 至少一个固定 GUI 任务集端到端通过；
- 模型不能直接调用桌面 Driver；
- 所有动作仍经过 Runtime 的 Grounding、Policy 和 Approval；
- 失败可以归因到感知、grounding、规划、工具或验证层。

## 6. MVP-3：多模态后训练与 Verifier

### 目标

证明模型行为可以通过训练方法和困难负样本得到可量化改进。

### 训练矩阵

| 阶段 | 必做 | 主要问题 |
|---|---|---|
| Baseline | Base / Prompt-only | 原始能力 |
| SFT | LoRA / QLoRA | 格式、动作和边界 |
| Distillation | Teacher 输出/序列蒸馏 | 扩大行为覆盖 |
| Preference | DPO 或 KTO | 安全、回退和偏好 |
| Agentic RL | GRPO 最小实验 | 可验证任务奖励 |
| Verifier | 分类、排序、校准 | 轨迹门禁 |

### 困难负样本

- 错误工具或危险参数；
- bbox/ref 指向错误控件；
- 删除必要步骤、打乱顺序；
- 重复产生副作用；
- 绕过审批；
- 使用过期 Memory；
- 工具失败却声称成功；
- 最终答案与动作后证据矛盾；
- 看似合理但证据不足。

### 可验证奖励

```text
任务状态验证通过              +1.0
工具和参数完全正确            +0.3
正确触发审批或拒绝            +0.3
在预算内完成                  +0.1
无效或重复动作                -0.2
超出步骤/成本预算             -0.3
未经审批尝试危险动作          -1.0
重复副作用或伪造成功          -1.0
```

模型奖励不能替代 Runtime 的安全规则。

### Verifier 指标

- Accuracy、Macro F1、AUROC；
- Pairwise Accuracy；
- ECE / Reliability Diagram；
- 错误轨迹漏放率；
- 正确轨迹误杀率；
- 启用门禁前后的任务成功率、延迟和成本。

### 退出条件

- Base/SFT/Distilled/Preference/RL 使用同一冻结测试集；
- 至少一个阶段带来统计上明确、可解释的提升；
- 失败实验和负结果同样进入报告；
- Verifier 自动阻止只能用于预先定义的安全边界。

## 7. MVP-4：多模型 Serving 与 MLOps

### 目标

将 Notebook 模型升级为可路由、可压测、可回滚的服务。

### 服务能力

- OpenAI-compatible API；
- vLLM 模型池；
- API Key、Tenant、Quota、Rate Limit；
- Timeout、Circuit Breaker、Admission Control；
- 本地小模型、远端强模型和规则 fallback；
- Dataset/Model/Config/Runtime 版本绑定；
- Eval Gate、Canary 和 Rollback。

### 多模态性能实验

| 实验 | 变量 | 指标 |
|---|---|---|
| 并发 | 1/4/8/16/32 | throughput、p50/p95、TTFT、TPOT |
| 图像 | 分辨率、张数、裁剪策略 | encoder latency、VRAM、质量 |
| 视频 | 帧数、采样策略 | latency、VRAM、任务成功率 |
| Cache | 相同/不同前缀 | hit rate、TTFT、GPU 利用率 |
| 量化 | BF16/FP16/4-bit/AWQ/GPTQ | VRAM、速度、质量变化 |
| 路由 | 全强模型 vs 分层路由 | 成功率、成本、强模型比例 |
| 过载 | 队列上限、OOM、进程退出 | 拒绝、恢复、丢失请求 |

### 退出条件

- 有容量边界，不只给出单请求速度；
- 新模型未过 Eval Gate 不能发布；
- 可一键回滚；
- 线上 Trace 能关联数据、模型、配置和 Runtime 版本。

## 8. MVP-5：Agentic RL 环境

### 目标

把 Reliable Runtime 转换为可训练环境，而不是另写一个无法代表生产行为的模拟器。

### 环境职责

- 生成固定任务和初始状态；
- 公开受限 Observation/Action Schema；
- 执行动作但不放宽 Runtime 安全规则；
- 使用最终状态和过程事件计算可验证奖励；
- 保存完整 episode 和失败原因；
- 支持环境重置、种子和版本固定。

### 必做对照

- SFT only；
- SFT + rule-based retry；
- SFT + GRPO；
- SFT + GRPO + Verifier；
- 小模型全执行 vs 低置信度强模型 fallback。

### 退出条件

- Reward 与真实任务状态相关，不奖励模型自报成功；
- 环境版本、任务版本和奖励版本固定；
- Reward hacking 有专门测试；
- 训练收益能转移到至少一个未见任务族。

## 9. MVP-6：多环境和更多模态

详细场景契约、覆盖等级和 `SCN-001～009` 见：[多模态与业务场景覆盖矩阵](多模态与业务场景覆盖矩阵.md)。

扩展波次：

1. `SCN-001` Desktop GUI：证明主闭环；
2. `SCN-002～004` Document、Browser、Coding：选择两个证明跨环境复用；
3. `SCN-005～007` Workflow/Data、DevOps、Security：证明企业治理；
4. `SCN-008～009` Audio/Video、Simulator：证明时序多模态和物理环境迁移。

每个新环境只能新增：

- Environment Adapter；
- Observation/Action 扩展；
- 专属任务集；
- 状态 Verifier；
- 数据和评测报告。

不得复制出另一套训练、Serving、审批或恢复系统。

Multi-Agent 不是独立业务场景，而是正式的跨场景执行拓扑与深度 Lab。它使用 `SCN-004 Coding Agent` 作为首个验证载体，并与 Single-Agent 使用相同任务集对照。

## 10. MVP-7：AI Infra 深化

### 训练系统 Lab

- DDP/FSDP/DeepSpeed 最小对照；
- mixed precision、gradient checkpointing、offload；
- sharded checkpoint 和中断恢复；
- 数据加载、通信和 GPU 利用率 profiling；
- 本地单卡完成基线，租用 2–4 GPU 完成一次严格多卡实验。

### 推理系统 Lab

- vLLM/SGLang 对照；
- continuous batching、prefix/KV cache；
- speculative decoding；
- PyTorch Profiler / Nsight；
- TensorRT-LLM 或 LMDeploy 最小验证；
- 一个 Triton kernel 或热点算子实验。

### 退出条件

- 报告性能瓶颈、假设、实验和反例；
- 不把“跑过命令”描述为框架二次开发；
- 至少一个优化有可复现的速度/显存收益；
- 正确性测试与性能测试同时保留。

## 11. MVP-8：Multi-Agent Coordination & Distributed Agent Systems

### 目标

将 Multi-Agent 做成具备明确身份、受限委派、共享 Durable State、Worker 恢复和量化评测的正式系统，而不是多个模型自由聊天。

### 系统结构

```text
Task / Goal
  ↓
Coordinator
  ↓
Capability / Authority / Budget Routing
  ├── Repository Researcher
  ├── Implementation Worker
  ├── Test / Review Worker
  └── Final Integrator
  ↓
Shared Durable State / Artifact Store
  ↓
Reliable Runtime / Verifier / Eval
```

### 核心契约

- Agent Identity、Role、Capability、Authority 和 expiry；
- Typed Task、Result、Artifact、Review 和 Handoff；
- 委派不能扩大父任务权限；
- CAS/version、冲突仲裁和 tombstone；
- Lease、Heartbeat、stale-owner recovery；
- 父子取消传播、预算、背压和循环保护；
- Reviewer 无默认执行权限；
- 所有副作用仍经过唯一 Runtime/Runner/MCP 边界。

### 首个验证场景

使用 `SCN-004 Coding Agent`：

```text
Issue
→ Coordinator
→ Researcher 定位文件和约束
→ Implementation Worker 生成补丁
→ Test / Review Worker 运行验证并审查
→ Final Integrator 绑定最终 Artifact 和证据
```

### 统一指标

- Task Success / Test Pass Rate；
- cost、tokens、provider calls；
- end-to-end latency；
- coordination overhead；
- duplicate work / conflict rate；
- recovery rate；
- safety violations；
- reviewer recall / false reject。

### 退出条件

- Multi-Agent 与 Single-Agent 使用同一冻结测试集和预算报告；
- Worker Crash、重复消息、冲突更新和取消传播测试通过；
- 没有第二个工具执行入口；
- Multi-Agent 在复杂任务上的收益足以覆盖额外成本和故障面；
- 不满足收益门禁时，系统保留 Single-Agent 路径。

## 12. 统一门禁与版本策略

### 四个门禁

1. **功能门禁**：新增能力在固定任务上通过。
2. **回归门禁**：旧任务和安全契约不能退化。
3. **安全门禁**：误批准、越权和重复副作用不能上升。
4. **性能门禁**：显存、延迟、吞吐和成本不超预算。

### 版本绑定

```text
experiment_id
├── git_commit
├── dataset_version + manifest digest
├── model_base + adapter + tokenizer
├── training_config + seed
├── runtime_version + policy_version
├── environment_version + reward_version
├── hardware + software lock
└── eval_report + serving_report + failure_report
```

### 变更纪律

- 一轮实验只改变一个主要变量；
- 看过测试结果后不得直接修改答案；
- 修改评测集必须新建版本并解释原因；
- 旧报告保留，不能用新结果覆盖历史事实；
- 所有指标必须标明样本数、硬件和运行日期。

## 13. 求职输出

### 旗舰项目

**Reliable Agent Model Lifecycle**

证明完整的数据、训练、评测、Serving、执行和回流闭环。

### 五个简历切片

#### Multimodal / Post-training

- 多模态轨迹数据与质量门禁；
- VLM QLoRA、DPO/GRPO、Reward/Verifier；
- Base/SFT/Preference/RL 对照；
- Grounding、Task Success 和安全指标。

#### Agent / Applied LLM

- GUI/文档多源 Observation；
- Tool Use、长程执行和强模型 fallback；
- Crash/Resume、Approval、Unknown Outcome；
- 固定任务集和真实 Badcase。

#### AI Infra / ML Systems

- vLLM、量化、缓存、路由和容量测试；
- Checkpoint、DDP/FSDP 和故障恢复；
- Model Registry、Eval Gate、Canary、Rollback；
- GPU profiling 和一项底层优化。

#### Multi-Agent / Distributed Agents

- Coordinator、能力路由和受限委派；
- Typed Message、共享 Durable State 和 Artifact；
- Lease/Heartbeat、Worker Crash 和取消传播；
- Single-Agent 对照、协调开销、冲突和安全指标。

#### Scene-specific Applied LLM

- 根据目标岗位选择 Document、Browser、Coding、Workflow/Data、DevOps 或 Security；
- 复用同一 Model Lifecycle、Serving、Runtime 和 Eval；
- 展示跨环境迁移而不是复制另一套系统。

## 14. 近期优先级

```text
P0  冻结 Runtime 基线
P1  建立多模态 Trace/Dataset Schema
P2  固定文本与 GUI Eval
P3  跑通文本 Tool Router 闭环
P4  升级图文 GUI Action Model
P5  做 SFT / DPO 或 GRPO / Verifier 对照
P6  接入 vLLM 和性能门禁
P7  扩展第二环境
P8  完成多卡或推理性能 Lab
P9  完成 Coding 场景的 Single-Agent / Multi-Agent 对照
```

原则：先完成一条闭环，再增加模态；先有固定 Eval，再开始训练；先证明正确性，再优化吞吐。
