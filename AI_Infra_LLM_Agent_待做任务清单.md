# Multimodal LLM × Agent Infra：待做任务清单

English companion: [docs/en/task-checklist.md](docs/en/task-checklist.md)

> 用途：作为后续 Codex / Claude Code 的统一执行源。  
> 当前状态：`guarded-desktop-agent` 可靠执行基线已存在且完整测试为 `1420 passed, 7 skipped`；模型训练、多模态后训练和 Serving 主线尚待实施。  
> 主目标：以现有 Runtime 为交互环境和安全执行底座，形成可写简历、可演示、可复现的多模态数据 + 后训练 + Eval + Serving + Reliable Agent 闭环。  
> 权威路线：[多模态 LLM 全周期 MVP 演进路线](多模态LLM全周期_MVP演进路线.md)和[多模态与业务场景覆盖矩阵](多模态与业务场景覆盖矩阵.md)。若本清单旧任务顺序与路线冲突，以两份路线文档和本文件第 2、14 节为准。
> Runtime 跨仓库依赖、数据通道和 Pin 规则见：[Desktop Runtime 依赖与集成](Desktop_Runtime_依赖与集成.md)。

## 0. 已确认环境

- 设备：Alienware m18 R1
- GPU：RTX 4090 Laptop GPU，16GB VRAM
- CPU / 内存：i9 / 64GB
- 存储：约 1.5TB；WSL 根目录当前约 1TB 可用
- Windows：NVIDIA 驱动工作正常
- WSL：WSL2 + Ubuntu 24.04，GPU 透传正常
- Ubuntu 当前缺少：PyTorch、pip、nvcc
- Windows 侧已装过 PyTorch，但后续训练和 vLLM 主环境使用 Ubuntu/WSL2

### 禁止事项

- 不在 WSL 内安装 Linux NVIDIA 驱动。
- 暂不执行 `sudo apt install nvidia-cuda-toolkit`。
- 不把训练仓库放到 `/mnt/c`。
- 不把 training、vLLM、runtime 全部塞进同一个 Python 环境。
- 不在没有基线和评测集前开始大规模训练。
- 不把“教师 API 生成数据”描述为 logits 蒸馏。
- 不在本机做 14B+ 训练或完整多机多卡方案。
- 不把 GUI 项目描述为自动驾驶或机器人量产经验。
- 不让模型绕过 Runtime、Policy、Approval 或唯一 MCP 执行边界。
- 不把模型自报成功作为任务完成证据。

---

# 1. 最终仓库结构

```text
agent-model-factory/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
├── labs/
│   ├── tiny-transformer/
│   ├── multimodal-post-training/
│   ├── distributed-training-inference/
│   └── multi-agent-distributed-systems/
├── data-engine/
├── post-training/
│   ├── tool-router/
│   ├── gui-action-model/
│   └── agentic-rl/
├── retrieval/
├── verifier/
├── serving/
├── agent-runtime/
├── environments/
│   ├── desktop-gui/
│   ├── document/
│   ├── browser/
│   └── optional-simulation/
├── evals/
├── infra/
├── scripts/
└── docs/
    ├── architecture.md
    ├── roadmap.md
    ├── adr/
    ├── experiments/
    ├── model-cards/
    ├── dataset-cards/
    └── failure-tests.md
```

## 统一工程规范

- Python 3.11 或 3.12，具体由依赖兼容性决定。
- 依赖必须锁定，禁止仅提供一串临时 `pip install` 命令。
- 配置使用 YAML/TOML，禁止关键超参数散落在代码中。
- 所有实验必须记录：Git commit、数据版本、模型版本、配置、随机种子、硬件和结果。
- 所有模块必须有最小单元测试；关键链路必须有集成测试。
- 每个项目至少保留一份 ADR、一份实验报告和一段可复制运行命令。
- 评测集必须固定版本，不能在看过测试结果后直接修改答案。

---

# 2. 里程碑

| 里程碑 | 目标 | 预计完成条件 |
|---|---|---|
| M0 | 冻结 Runtime 与训练环境基线 | Runtime 测试固定；PyTorch CUDA 通过；环境锁定 |
| M1 | 文本 Tool Router 最短闭环 | Trace→Dataset→QLoRA→Eval→Runtime→Badcase |
| M2 | 图文 GUI Action Model | Screenshot/UIA/OCR 联合输入；Grounding、动作、风险和回退可评测 |
| M3 | 多模态后训练与 Verifier | SFT/蒸馏/DPO 或 GRPO/Verifier 使用同一冻结测试集对照 |
| M4 | 多模型推理平台 | vLLM、量化、缓存、路由、灰度、回滚与性能报告 |
| M5 | Agentic RL 环境 | Runtime 状态产生可验证奖励；有 reward-hacking 测试 |
| M6 | 多环境扩展 | 文档或浏览器成为第二环境，不复制训练和执行系统 |
| M7 | 四个深度 Lab | Tiny Transformer、多模态后训练、训练/推理性能、Multi-Agent 均有独立报告 |
| M8 | 简历与演示 | 五种岗位叙事、架构图、3 分钟视频和真实数字 |

---

# 3. Phase 0：环境与基线

## ENV-001：检查并配置 WSL 资源

**产出**

- `.wslconfig` 建议值与修改说明
- `free -h`、磁盘和 GPU 快照
- WSL 网络/代理诊断结果

**验收**

- WSL 可稳定看到 GPU
- 可用内存符合配置
- PyPI、PyTorch、Hugging Face 下载可用

## ENV-002：建立独立 Python 环境

建议：

```text
.venv-dev       Tiny Transformer / 数据处理
.venv-training  Transformers / PEFT / TRL / bitsandbytes
.venv-serving   vLLM / Gateway
.venv-runtime   FastAPI / Temporal / OTel
```

**验收**

- 每个环境均有锁文件
- 环境之间不共享 site-packages
- `make env-check` 输出完整版本矩阵

## ENV-003：安装并验证 PyTorch CUDA

**测试内容**

- `torch.cuda.is_available()`
- GPU 名称和 VRAM
- BF16 支持
- FP16/BF16 矩阵乘法
- 峰值显存
- 10 分钟稳定性测试

**验收**

- CUDA 矩阵测试通过
- 无驱动冲突
- 保存 `docs/environment.md`

## ENV-004：建立实验追踪骨架

**第一版**

- MLflow 本地模式或 W&B offline
- 统一 `run_id`
- Git commit、config、metrics、artifacts 自动记录

**验收**

- 一个 smoke experiment 能被完整重现

---

# 4. Project A：Tiny Transformer & Training Systems Lab

## TT-001：最小 Decoder-only Transformer

实现：

- Token embedding
- Multi-head causal attention
- Causal mask
- RoPE
- RMSNorm
- MLP
- Residual
- LM head
- Cross entropy

**验收**

- 关键张量 shape 测试
- causal mask 不泄漏未来 token
- 小 batch 可前向、反向并更新参数

## TT-002：生成策略

实现：

- greedy
- temperature
- top-k
- top-p
- repetition penalty
- stop token

**验收**

- 固定 seed 可复现
- 保存不同策略输出和延迟对照

## TT-003：KV Cache

**验收**

- 缓存与无缓存输出一致
- 记录逐 token 延迟和显存变化
- 文档说明 prefill / decode 区别

## TT-004：训练工程

实现：

- BF16 / FP16
- gradient accumulation
- gradient clipping
- warmup + scheduler
- gradient checkpointing
- train/validation split

**验收**

- 显存与 tokens/s 对照表
- OOM 后可通过配置调整恢复

## TT-005：Checkpoint 与中断恢复

保存：

- model
- optimizer
- scheduler
- scaler
- global step
- RNG states
- data sampler state（可行时）

**故障注入**

- 训练中途强制结束进程
- 从 checkpoint 恢复

**验收**

- step 连续
- 学习率连续
- loss 无异常跳变
- 固定条件下结果可复现

---

# 5. Project B：Data Engine & Continued Pretraining

## DATA-001：数据清单和许可证字段

每条来源记录：

- source_id
- URL / repository / file
- license
- retrieved_at
- checksum
- deletion_policy
- usage_scope

## DATA-002：解析与标准化

支持：

- Markdown
- HTML
- PDF 文本
- JSONL
- Git 仓库文档
- Agent Trace

**验收**

- 统一 schema
- 解析失败可追踪
- 原始数据与处理数据分离

## DATA-003：清洗、去重、质量和 PII

实现：

- exact dedup
- near dedup（MinHash 或等价方法）
- 长度、字符、语言和模板过滤
- PII 检测/脱敏
- 质量评分

**验收**

- `quality_report.md`
- 去重比例和过滤原因分布

## DATA-004：切分与污染检查

实现：

- train / validation / test
- 文档级隔离
- 与 eval 集的 exact / near duplicate 检查

**验收**

- `contamination_report.md`
- 测试集可冻结和版本化

## DATA-005：Packing 与 Token 统计

**产出**

- 长度分布
- token 总量
- packing 利用率
- 截断比例

## CPT-001：0.5B-0.6B 领域继续预训练

对照：

- Base
- CPT
- SFT
- CPT + SFT

**指标**

- validation loss / perplexity
- 领域任务指标
- 通用能力保留
- 显存和 tokens/s

**验收**

- 不仅展示 loss
- 有灾难性遗忘分析

---

# 6. Project C：Multimodal Tool-Use Post-Training & Distillation

## MM-001：多模态轨迹 Schema

输入至少包含：

- instruction
- screenshot / cropped region
- UIA / document text / OCR
- previous actions and results
- available tools
- policy context

输出至少包含：

- next_tool
- arguments / bbox / ref
- risk_level
- requires_approval
- confidence
- should_reject / should_fallback
- evidence

**验收**

- 文本样本与图文样本使用可兼容的版本化 Schema
- 每条轨迹绑定 runtime/model/policy/environment 版本
- 动作前观察与动作后观察可以关联
- 模型输出不包含任何直接执行权限

## MM-002：GUI Grounding 数据与评测

构造：

- ref grounding
- bbox grounding
- UIA-only / screenshot-only / fused observation
- OCR 缺失或噪声
- 控件移动、遮挡、过期 ref

指标：

- Grounding Accuracy / IoU
- Action Accuracy
- Tool / Argument Accuracy
- stale-ref rejection
- coordinate/ref disagreement

## MM-003：图文 GUI Action Model

实验：

- 0.5B–3B 小型 VLM
- 全屏 vs 局部裁剪
- UIA-only vs screenshot-only vs fusion
- 不同动作历史长度
- 不同置信度 fallback 阈值

**验收**

- QLoRA Adapter 可独立加载
- 固定 GUI 任务集可重复运行
- 有任务成功率、步骤数、回退率、显存和延迟报告
- 所有动作仍经过既有 Runtime/Runner/MCP 边界

## MM-004：多模态困难负样本

- bbox/ref 指向错误控件
- 图像与结构化观察冲突
- 忽略动作后状态
- 重复动作或重复副作用
- 绕过审批
- 工具失败但声称成功
- 看似合理但证据不足

## MM-005：多模态环境适配

扩展顺序：

1. Desktop GUI
2. Document / Chart / PDF
3. Browser Research
4. Audio / Video
5. Robotics / Autonomous Driving Simulation（可选）

每个环境只新增 Adapter、任务集、Verifier 和数据，不复制训练、Serving、审批或恢复系统。

## TOOL-001：工具 Schema 与任务定义

首批工具建议：

- file_read
- file_write
- browser_search
- browser_extract
- shell_readonly
- database_query
- computer_use
- request_approval
- reject_request
- fallback_to_strong_model

每条任务标注：

- selected_tool
- arguments
- risk_level
- requires_approval
- should_reject
- should_fallback
- expected_result

## TOOL-002：人工种子集

**目标**

- 200-500 条高质量样本
- 覆盖正常、歧义、安全、无工具可用、参数缺失和多步任务

**验收**

- Schema 全部合法
- 工具可实际执行或被模拟器验证
- 类别分布报告

## TOOL-003：教师模型合成流水线

实现：

- 提示模板
- 多样性生成
- Schema 验证
- 去重
- 工具执行验证
- 一致性检查
- 难度分类
- 人工抽检

**验收**

- 每条数据可追溯 teacher/model/prompt/version
- 不合格样本有明确拒绝原因

## TOOL-004：真实 Agent Trace 导入

**要求**

- 脱敏
- 工具和参数规范化
- 成功/失败标注
- runtime/model/version 记录

## TOOL-005：统一 Baseline Eval

指标：

- tool accuracy
- JSON validity
- argument exact match / field F1
- risk Macro F1
- approval accuracy
- refusal / false refusal
- fallback rate

**验收**

- Base 模型结果冻结
- 每次训练自动运行同一评测

## TOOL-006：LoRA / QLoRA SFT

实验：

- LoRA vs QLoRA
- rank
- alpha
- target modules
- data size
- sequence length

**验收**

- 记录峰值显存、训练时间、Adapter 大小和最终指标
- Adapter 可独立加载与合并

## TOOL-007：输出/序列蒸馏

对照：

- 人工数据 SFT
- 合成数据 SFT
- 人工 + 合成
- 人工 + 合成 + 真实 Trace

**验收**

- 明确称为 output/sequence distillation
- 分析合成数据噪声和收益

## TOOL-008：Logits 蒸馏最小实验

仅在教师 logits 可访问且 tokenizer 兼容时执行。

实现：

- temperature
- KL divergence
- hard loss + soft loss
- loss weight

**验收**

- 小数据实验报告
- 解释 tokenizer 不一致问题

## TOOL-009：DPO

构造 chosen/rejected：

- 正确 vs 错误工具
- 安全参数 vs 危险参数
- 需要审批 vs 绕过审批
- 简洁轨迹 vs 无效循环
- 有证据完成 vs 虚假完成

**验收**

- Base / SFT / Distilled / DPO 同集对照
- 安全和任务效果均评估

## TOOL-010：ORPO/KTO/GRPO 最小对照

- ORPO 或 KTO 二选一
- GRPO 仅用于可自动验证任务

奖励示例：

- JSON 合法
- 工具正确
- 参数正确
- 风险判断正确
- 未绕过审批

**验收**

- 分析 reward hacking 和长度偏置
- 不阻塞首版简历项目

---

# 7. Project D：Embedding & Reranker

## RET-001：检索数据集

格式：

```json
{
  "query": "...",
  "positive": "...",
  "hard_negatives": ["..."],
  "metadata": {}
}
```

场景优先级：

1. Agent 任务 → 项目 Memory
2. Issue → 代码文件
3. 告警 → Runbook
4. 技术问题 → 文档章节

## RET-002：基线

- BM25
- Base embedding
- Hybrid search

## RET-003：Embedding 微调

覆盖：

- contrastive learning
- in-batch negatives
- hard-negative mining

## RET-004：Cross-Encoder Reranker

**验收指标**

- Recall@5/10
- MRR
- nDCG@10
- Top-1
- latency

## RET-005：Agent 联合评测

**必须证明**

- 检索指标提升是否提高最终任务成功率
- 过期/冲突 Memory 是否被正确过滤

---

# 8. Project E：Verifier / Reward Model

## VER-001：轨迹数据 Schema

输入：

- user request
- plan
- steps
- tool calls
- tool results
- approvals
- final answer

输出：

- success
- error_type
- unsafe_action
- should_retry
- should_rollback
- pairwise preference

## VER-002：困难负样本生成

- 错误工具替换
- 参数扰动
- 删除关键步骤
- 顺序打乱
- 重复副作用
- 绕过审批
- 过期 Memory
- 工具失败但声称成功
- 最终答案与证据矛盾

## VER-003：分类与排序模型

任务：

- binary classification
- error multiclass
- pairwise ranking
- reward score

## VER-004：校准与阈值

指标：

- Accuracy
- Macro F1
- AUROC
- ECE
- pairwise accuracy
- false accept / false reject

## VER-005：Agent 门禁

实验：

- 无 Verifier
- Verifier 仅告警
- Verifier 自动阻止/重试

**验收**

- Agent 成功率和成本变化
- 错误轨迹漏放率可量化

---

# 9. Project F：Serving & MLOps

## SERV-001：OpenAI-compatible Gateway

能力：

- API key / tenant
- rate limit
- token quota
- timeout
- retry
- circuit breaker
- model routing
- fallback
- metrics / trace

## SERV-002：vLLM 服务

部署：

- Tool Router
- Verifier（若为生成模型）
- Strong model fallback（本地或外部）

## SERV-003：量化基准

对照：

- BF16 / FP16
- bitsandbytes 8-bit
- bitsandbytes 4-bit
- GPTQ
- AWQ

记录：

- VRAM
- model size
- TTFT
- TPOT
- tokens/s
- quality delta

## SERV-004：并发与缓存实验

变量：

- concurrency 1/4/8/16/32
- same prefix / random prefix
- prompt length
- output length

记录：

- throughput
- p50/p95
- TTFT/TPOT
- cache hit
- GPU utilization
- error rate

## SERV-005：Admission Control 与过载

故障：

- 队列积压
- OOM
- 模型进程退出
- 上游超时

**验收**

- 有明确拒绝/排队策略
- 能自动恢复或 fallback

## MLOPS-001：模型和数据注册

- Dataset Registry
- Model Registry
- Model Card
- Dataset Card
- config / commit / artifact binding

## MLOPS-002：Eval 门禁、灰度与回滚

**验收**

- 新模型未过阈值不能发布
- 灰度流量可配置
- 一键回滚旧模型

## MLOPS-003：Badcase 回流

```text
Trace → Badcase 分类 → 人工审核 → Dataset vN+1
→ Train → Eval → Canary → Release
```

---

# 10. Project G：Reliable Agent Runtime

> 复用并冻结现有 `guarded-desktop-agent`，不在新仓库中重写第二套 Runtime。以下任务首先执行 capability mapping：已实现能力标记为基线，只为多模态模型接入补最小接口和缺失证据。

## RUN-001：任务数据模型

至少包含：

- Task
- Step
- Attempt
- ToolCall
- Checkpoint
- Approval
- Event
- Memory
- ModelVersion

## RUN-002：状态机与 Worker

状态：

```text
CREATED → QUEUED → RUNNING
→ WAITING_APPROVAL / RETRYING / SUSPENDED
→ COMPLETED / FAILED / CANCELLED / TIMED_OUT
```

## RUN-003：Checkpoint / Resume

**故障注入**

- 任务执行中杀死 Worker
- 新 Worker 接管

**验收**

- 从持久化状态继续
- 已完成步骤不重复

## RUN-004：工具幂等和 Unknown Outcome

实现：

- idempotency key
- unique constraint
- duplicate delivery handling
- success-before-persist crash handling

## RUN-005：权限、审批、沙箱与审计

高风险动作：

- 删除
- 发送
- 提交
- 修改生产数据
- 高危 Shell
- 最终确认

**验收**

- 必须暂停并等待人工批准
- 审批消息重复到达不重复执行

## RUN-006：Context / Memory

覆盖：

- token budget
- context compaction
- tool result cleanup
- working / semantic / episodic / procedural memory
- scope
- version conflict
- expiry
- deletion

## RUN-007：可观测性

Trace：

- HTTP request
- Agent task
- LLM call
- planning
- tool call
- approval wait
- sandbox
- memory read/write
- persist

Metrics：

- success rate
- p95
- queue wait
- retry
- tool errors
- tokens
- TTFT
- memory stale hit

## RUN-008：Kubernetes 与弹性

- Docker Compose
- Helm
- Deployment / Service
- Secret / ConfigMap
- Probe
- HPA / KEDA

**演示**

- 任务积压时 Worker 扩容
- 清空后缩容

## RUN-009：10 个故障实验

1. Worker 被杀
2. Tool 成功后写库前被杀
3. 同一任务重复投递
4. LLM 超时
5. Tool 超时 / unknown outcome
6. PostgreSQL 短暂不可用
7. 用户取消
8. Agent 无限循环
9. 新旧 Memory 冲突
10. Context 压缩丢约束

---

# 11. Project H：Multi-Agent Coordination & Distributed Agent Systems

> Multi-Agent 是跨业务场景的执行拓扑和正式深度 Lab，不是第十个业务场景。首个验证载体使用 `SCN-004 Coding Agent`，并始终与 Single-Agent 使用同一任务集对照。

## MA-001：Agent Identity、Role、Capability 与 Authority

定义：

- agent_id / run_id / parent_agent_id
- role / capabilities / allowed_tools
- delegated_scope / policy_version / budget
- created_at / expires_at / revoked_at

**验收**

- 委派不能扩大父任务权限
- Reviewer 默认无执行权限
- 每个 Agent 的行为可独立审计
- 权限过期和撤销立即生效

## MA-002：Typed Message、Task、Result、Artifact 与 Handoff

禁止依赖自由文本聊天作为系统真相。至少定义：

- TaskAssignment
- ObservationRequest / ObservationResult
- WorkResult
- ArtifactReference
- ReviewDecision
- Handoff
- Cancellation
- AttentionRequired

**验收**

- Schema 和版本可校验
- 消息具有 correlation_id 和 idempotency_key
- 大型 Artifact 使用引用和 digest，不复制进所有上下文
- 非法或过期消息 fail closed

## MA-003：Coordinator、任务图与能力路由

能力：

- 将目标编译为有依赖关系的任务图
- 按能力、权限、预算和负载选择 Worker
- 限制 fan-out、深度和总步骤
- 记录每次路由的输入事实和理由

**验收**

- 无可用 Worker 时明确阻塞
- 不因模型建议自动创建无限 Agent
- Coordinator 不成为第二个工具执行入口
- 固定路由和学习型路由可使用同一评测

## MA-004：共享 Durable State、Memory 与冲突仲裁

实现：

- 工作状态、Artifact、Memory 分离
- optimistic version / CAS
- single writer 或显式 merge policy
- superseded_by / conflict / tombstone

**验收**

- 并发更新不会静默覆盖
- 过期 Memory 不进入活动上下文
- 冲突由确定性规则或显式 Reviewer 处理
- 聊天历史不作为 durable source of truth

## MA-005：Lease、Heartbeat 与 Worker Crash Recovery

覆盖：

- Worker claim / lease / heartbeat
- stale owner 检测
- 已知未执行任务重新分配
- 已知完成结果复用
- Unknown Outcome 停止并请求人工处理

**验收**

- Worker 被杀后可恢复
- 不重复产生副作用
- lease 转移有 durable evidence
- 同一前台桌面始终只有一个执行权持有者

## MA-006：预算、取消传播、背压与循环保护

预算至少包含：

- Agent 数量
- 总步骤和每 Agent 步骤
- Token / provider calls
- side effects
- wall-clock deadline
- queue depth

**验收**

- 父任务取消传播到所有子任务
- 超预算任务不继续派生
- 队列积压有拒绝或降级策略
- Agent 之间互相委派的循环可检测

## MA-007：Reviewer、Verifier 与最终决策

Reviewer 输入必须包括：

- 原始任务
- 候选结果和 Artifact digest
- 测试/环境状态证据
- Policy 和预算结果
- 分歧点

**验收**

- Reviewer 不能只根据自然语言自报成功
- 评审结论绑定具体 Artifact/version
- 分歧无法解决时进入人工注意状态
- Reviewer 不直接执行修复

## MA-008：Single-Agent vs Multi-Agent 固定评测

首个任务集使用 Coding Agent：

```text
Coordinator
├── Repository Researcher
├── Implementation Worker
├── Test / Review Worker
└── Final Integrator
```

统一指标：

- Task Success Rate
- Test Pass Rate
- cost / tokens / provider calls
- end-to-end latency
- coordination overhead
- duplicate work rate
- conflict rate
- recovery rate
- safety violations
- reviewer recall / false reject

**发布门禁**

Multi-Agent 必须在固定复杂任务上显著优于 Single-Agent，且收益足以覆盖成本、延迟和新增故障面；否则保留 Single-Agent。

## MA-009：多进程/容器 Worker 与弹性调度

实现：

- 独立 Worker 进程或容器
- 队列、租约、健康检查和优雅退出
- 按任务类型和队列深度扩缩
- Artifact Store 和 Trace 关联

**验收**

- Worker 扩缩不改变任务语义
- 进程重启不丢失 durable state
- 负载、排队和恢复有指标
- 不以并行 Worker 推导并行桌面执行权

## MA-010：学习型路由或 Multi-Agent RL 最小实验

对照：

- 固定规则路由
- 模型能力路由
- bandit / supervised router
- 可选的可验证奖励优化

奖励同时考虑：

- 任务成功
- 成本和延迟
- 重复工作
- 冲突
- 安全和权限

**验收**

- 路由策略版本固定并可回滚
- 有 held-out 任务族
- 防止通过少做必要步骤获得虚假成本奖励
- 学习型路由不能改变 Authority Contract

---

# 12. 场景项目

> 每个场景复用同一 Model Lifecycle、Serving、Reliable Runtime、Registry 和 Eval Hub。场景只新增 Environment Adapter、Observation/Action Schema、任务集、Verifier 和 Policy。

## SCN-001：Desktop GUI Agent

**范围**

- UIA + Screenshot + Region + OCR + action history
- GUI Grounding、Tool Use、风险、审批和 fallback
- 固定低风险任务、动作后状态验证、Crash/Resume

**验收**

- `manifest.yaml`、Observation/Action Schema、Policy 和 Verifier
- 固定 GUI 训练/验证/测试集
- stale ref、遮挡、焦点变化和工具失败用例
- 达到 L3 Environment，作为首个主场景

## SCN-002：Document / Chart / PDF

**范围**

- 文档结构、表格、图表、扫描页和引用定位
- 抽取、问答、摘要、编辑建议

**验收**

- Extraction F1、Table/Chart QA、Citation Accuracy
- 版面变化、OCR 噪声、长文档和跨页引用用例
- 文档 Verifier 根据原文件内容核验

## SCN-003：Browser Research

**范围**

- DOM + Screenshot + page text
- 搜索、翻页、提取、来源记录和下载前确认

**验收**

- 来源覆盖、Citation Accuracy、重复结果率和任务成功率
- 页面变化、登录态失效、分页和下载失败用例
- 最终提交、下载或外部写入必须经过审批

## SCN-004：Coding Agent

**范围**

- Issue → 仓库检索 → 计划 → 指定文件编辑 → 测试 → 补丁摘要
- 受控 Shell / Git、恢复和回归

**验收**

- 固定 Repo Fixture、Issue 集和隐藏测试
- Patch Apply Rate、Test Pass Rate、无关文件修改率
- 禁止未经审批 commit、push、PR 或破坏性 Shell

## SCN-005：Enterprise Workflow / Data Agent

**范围**

- 表单、规则校验、文档解析、条件路由和人工审批
- Schema、只读 SQL、结果解释和受控回写

**验收**

- RBAC、幂等键、重复消息、流程版本和补偿用例
- SQL Validity、扫描预算、结果正确率和越权率
- 生产写入默认禁止；批准后仍需精确 effect binding

## SCN-006：DevOps Agent

**范围**

- Metric / Log / Trace → Runbook → 故障假设 → 只读诊断
- 审批后的扩容、重启或回滚只作为进阶

**验收**

- 固定日志、指标、告警和 Runbook Fixture
- Diagnosis Accuracy、Evidence Coverage、误操作率
- 生产变更有审批、审计、回滚和 Unknown Outcome 处理

## SCN-007：Security Agent

**范围**

- 告警关联、资产和历史事件、风险分级、调查清单和证据建议
- Prompt Injection、工具越权和数据泄漏测试

**验收**

- Risk Macro F1、Evidence Accuracy、false accept / false reject
- 最小权限、敏感信息脱敏和审计
- 自主隔离、攻击或破坏性动作不在首版范围

## SCN-008：Audio / Video Agent

**范围**

- 短音频、短视频、动态界面和同步提示
- ASR、事件检测、时序状态、跨模态动作

**验收**

- WER、Intent Accuracy、Event F1、Temporal Accuracy
- 不同采样率、帧数、噪声和跨模态干扰实验
- encoder latency、流式延迟、显存和端到端成本报告

## SCN-009：Robotics / Driving Simulator

**范围**

- 仿真传感器/世界状态、离散或受限连续动作
- VLA 候选动作、闭环状态验证和安全约束

**验收**

- Simulator Adapter、reset/seed/version
- 任务成功率、碰撞/越界率、控制延迟和分布外场景
- 不连接真实车辆或机器人；安全壳不由模型控制

# 13. 统一评测与报告

## EVAL-001：固定评测集

至少覆盖：

- 正常工具任务
- 参数缺失
- 工具歧义
- 危险请求
- 审批
- 模型超时
- 工具失败
- 重复消息
- 过期 Memory
- 冲突 Memory
- 无限循环

## EVAL-002：统一报告格式

每次报告必须包含：

- 数据/模型/代码版本
- 运行环境
- 主要指标
- Base 对照
- 置信区间或重复运行
- Badcase 分类
- 性能和成本
- 已知限制
- 下一步实验

## EVAL-003：简历指标表

最终需要真实填写：

| 模块 | 指标 |
|---|---|
| Tool Router | Tool Accuracy、JSON 合法率、风险 F1、fallback rate |
| Retriever | Recall@10、MRR、nDCG、stale memory hit |
| Verifier | Macro F1、AUROC、ECE、false accept |
| Serving | TTFT、TPOT、tokens/s、p95、throughput、VRAM |
| Runtime | 恢复率、恢复时间、重复副作用、审批命中、故障覆盖 |
| End-to-end | 任务成功率、平均步骤、Token、延迟、成本 |

---

# 14. MVP-first 执行顺序

> 不同时铺开所有模块。先完成一条可运行、可评测的垂直闭环，再逐轮增加模态、训练方法和系统深度。

## 第一阶段：14 天可演示 MVP

| 天数 | 任务 | 必须产出 |
|---|---|---|
| 1-2 | ENV + Runtime baseline | 环境锁定、Capability Manifest、Runtime 测试与故障证据 |
| 3-4 | TOOL-001~005 + EVAL-001 | 文本 Schema、种子集、冻结测试集和 Base 结果 |
| 5-7 | TOOL-006 | Tool Router QLoRA、Adapter、训练和显存报告 |
| 8 | Runtime 接入 | 小模型候选动作经过既有 Policy/Approval/MCP |
| 9-10 | Trace / Badcase 回流 | 至少一个真实失败进入 Dataset vN+1 |
| 11-12 | MM-001~002 | 多模态轨迹 Schema、GUI Grounding 数据和评测 |
| 13-14 | MM-003 最小版 | 图文 GUI Action Model 基线与端到端演示 |

## 第二阶段：第 3-4 周

- MM-003~005：图文 Action Model、融合消融和第二环境准备；
- TOOL-007~010：输出蒸馏、DPO/KTO、GRPO 最小对照；
- VER-001~005：困难负样本、轨迹 Verifier、校准和门禁；
- 统一比较 Base / SFT / Distilled / Preference / RL。

## 第三阶段：第 5 周

- SERV-001~005：vLLM、量化、缓存、路由、过载和性能报告；
- MLOPS-001~003：Registry、Eval Gate、Canary、Rollback、Badcase 回流；
- 完成多模态图像分辨率、张数、并发和显存实验。

## 第四阶段：第 6 周及以后

- TT-001~005 + CPT-001：Tiny Transformer / Pretraining Lab；
- RET-001~005：Embedding / Reranker，仅在第二环境需要检索时进入主线；
- DDP/FSDP/DeepSpeed、Profiler、Nsight、Triton 最小实验；
- 文档/浏览器成为第二环境，音视频和仿真按需求继续扩展；
- Docker/K8s/Helm/KEDA 在本地闭环稳定后再实施。

## 第五阶段：Project H Multi-Agent

进入条件：

- SCN-004 Coding Agent 的 Single-Agent 基线已经冻结；
- 复杂任务确实存在专业化、独立审查或并行探索收益；
- Runtime、Artifact Store、Trace 和预算接口稳定。

执行：

- MA-001~007：身份、消息、协调、共享状态、恢复、预算和 Reviewer；
- MA-008：同任务集、同验证器的 Single-Agent / Multi-Agent 对照；
- MA-009：多进程或容器 Worker；
- MA-010：学习型路由作为最后的可选增量。

Multi-Agent 不阻塞前四个阶段，也不能替换未通过收益门禁的 Single-Agent。

## 时间分配

- 60%：旗舰母项目完整闭环；
- 25%：当前目标岗位对应的四个深度 Lab 之一；
- 15%：其他方向的最低可信证据。

---

# 15. Definition of Done

一个任务只有同时满足以下条件才算完成：

- 代码已提交，目录和命名清晰。
- 有自动化测试，且测试真实覆盖该能力。
- 有可复制命令，不依赖未记录的手工步骤。
- 有固定输入和输出样例。
- 有指标或故障证据，不只展示“成功运行”。
- 有 README/ADR 解释选择、替代方案和 Trade-off。
- 失败场景有明确行为。
- 未实现部分明确标记，禁止在简历中夸大。

---

# 16. 后续 Codex / Claude Code 分派准备

暂不直接分派，后续按以下方式拆任务：

- 每次只分一个可在 0.5-2 天内验收的 Task ID。
- 提示词必须包含：背景、已有文件、允许修改范围、禁止事项、验收命令、Definition of Done。
- Agent 完成后必须先运行测试和基准，再提交变更摘要。
- 架构决策、实验设计和跨模块重构需要双重审查。
- 不允许 Agent 自行扩大范围或替换核心技术路线。

建议倾向：

- Codex：实现、测试、脚本、性能基准、修复具体问题。
- Claude Code：架构梳理、复杂跨文件改造、ADR、评测设计、代码审查。
- 关键任务可采用“一方实现，另一方审查”的方式，而不是两边重复写同一代码。
