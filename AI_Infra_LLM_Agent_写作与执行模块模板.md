# Multimodal LLM × Agent Infra：写作与执行模块模板

> 用途：作为项目正式开工后的统一写作、任务分派、实验记录、代码审查和验收模板。  
> 配套文件：
> - `AI_Infra_LLM_Agent系统研发行动手册_模型开发项目完整版.pdf`：知识地图、项目路线和学习参考。
> - `AI_Infra_LLM_Agent_待做任务清单.md`：任务编号、依赖关系、阶段顺序和 Definition of Done。
> - `多模态LLM全周期_MVP演进路线.md`：当前项目定位、MVP 版本、指标和扩展边界。
> - `多模态与业务场景覆盖矩阵.md`：模态、环境、模型任务、业务场景和统一场景契约。
> - 本文件：把路线与任务转换为 Codex、Claude Code 和人工执行时可直接复制使用的模块。

---

## 1. 建议仓库中的保存位置

```text
agent-model-factory/
├── README.md
├── environments/
│   ├── desktop-gui/
│   ├── document/
│   └── browser/
├── labs/
│   ├── tiny-transformer/
│   ├── multimodal-post-training/
│   ├── distributed-training-inference/
│   └── multi-agent-distributed-systems/
├── docs/
│   ├── project-overview.md
│   ├── knowledge-map.md
│   ├── architecture.md
│   ├── roadmap.md
│   ├── environment.md
│   ├── adr/
│   ├── experiments/
│   ├── evaluations/
│   ├── failure-tests/
│   ├── model-cards/
│   ├── dataset-cards/
│   └── resume/
├── tasks/
│   ├── backlog.md
│   ├── current.md
│   ├── completed.md
│   └── specs/
└── .ai/
    ├── codex/
    └── claude/
```

### 文件命名

```text
tasks/specs/ENV-001-wsl-environment.md
.ai/codex/ENV-001-implementation.md
.ai/claude/ENV-001-review.md
docs/experiments/EXP-001-pytorch-cuda-baseline.md
docs/adr/ADR-001-python-environment-strategy.md
docs/failure-tests/FAIL-001-training-resume.md
```

---

# 2. 项目总览模块

保存为 `docs/project-overview.md`。

```markdown
# Multimodal Agent Model Factory × Reliable Runtime

## 1. 项目目标

以桌面 GUI 作为第一个可验证环境，构建覆盖多模态数据、模型训练、后训练、检索、评测、推理服务、Reliable Agent Runtime 和 MLOps 的完整闭环，并逐步扩展到文档、浏览器、音视频和可选仿真场景。

## 2. 要解决的问题

- 如何从截图、UIA、OCR、文档、文本和 Agent Trace 构造可训练的视觉—语言—动作数据？
- 如何进行继续预训练、SFT、LoRA/QLoRA、蒸馏和偏好优化？
- 如何训练 Tool Router、GUI Action Model、Embedding、Reranker 和 Verifier？
- 如何使用可验证环境与奖励进行 Agentic RL？
- 如何量化并通过 vLLM 部署多模态模型？
- 如何让 Agent 长任务支持恢复、幂等、审批、审计和可观测性？
- 如何让多个 Agent 在受限权限下委派、共享状态、处理冲突并从 Worker 故障中恢复？
- 如何将线上 Badcase 回流为新的训练与评测数据？

## 3. 目标用户与求职方向

- Agent 开发工程师
- Agent Runtime / Agent Infra
- AI Platform / AI Infra
- LLM/VLM 后训练与多模态模型应用
- LLM Serving / 推理平台
- Eval / Data / Alignment
- ML Systems / Training Infrastructure
- Multi-Agent Platform / Distributed Agent Systems
- Python / 后端系统研发

## 4. 核心系统链路

多模态原始数据与 Agent Trace
→ 数据清洗、脱敏、质量、切分和版本管理
→ CPT / SFT / QLoRA / 蒸馏 / DPO / GRPO
→ Tool Router / GUI Action Model / Retriever / Reranker / Verifier
→ 量化、多模态 vLLM Serving
→ 模型路由、Fallback、限流和缓存
→ Reliable Agent Runtime
→ 可选 Multi-Agent Coordinator / Specialist Workers / Reviewer
→ Trace、Eval、Badcase
→ 数据审核与再训练

## 5. 成功标准

- 所有实验绑定代码、数据、模型和配置版本。
- 关键模型均有 Base 对照和固定评测集。
- 训练过程可以中断恢复。
- 推理服务有 TTFT、TPOT、吞吐、图像编码延迟和显存报告。
- Agent 任务可以在 Worker 崩溃后恢复。
- 高风险工具必须经过人工审批。
- 新模型必须通过功能、回归、安全和性能四个门禁。
- Multi-Agent 必须有 Single-Agent 基线、共享状态和 Worker 故障验证。
- 最终形成一个旗舰母项目和四个可独立讲述的深度 Lab。

## 6. 非目标

- 不自行实现大型分布式训练框架。
- 不在本机进行 14B 以上模型训练。
- 不为展示技术而强行使用多 Agent。
- 不以“跑通框架示例”替代真实评测和故障验证。
- 不把 GUI 场景包装为自动驾驶或机器人量产经验。
- 不让模型输出绕过 Runtime 的确定性安全和执行边界。
- 不把小规模实验描述为完整基础模型预训练经验。
```

---

# 2.5 场景规格模块

保存为 `environments/<scenario>/README.md`，并与对应的 `manifest.yaml` 一起评审。

```markdown
# SCN-[编号]：[场景名称]

## 1. 场景目标

说明要验证的真实任务，以及为什么需要该场景。

## 2. 模态与运行环境

- Modalities：
- Environment：
- Model Tasks：
- Side-effect Class：

## 3. Observation Contract

- 允许输入：
- 结构化字段：
- 图像/音频/视频预算：
- 新鲜度要求：
- 禁止暴露的数据：

## 4. Action Contract

- 允许动作：
- 参数 Schema：
- Grounding 要求：
- 必须审批的动作：
- 永久禁止的动作：

## 5. Environment Adapter

- 初始化：
- Reset：
- Seed：
- Version：
- 与 Runtime 的连接边界：

## 6. Dataset

- 训练集：
- 验证集：
- 冻结测试集：
- 困难负样本：
- 数据和许可证：

## 7. State Verifier

- 成功状态：
- 失败状态：
- 部分完成：
- 不确定结果：
- 为什么不依赖模型自报成功：

## 8. Eval

- 质量指标：
- 安全指标：
- 可靠性指标：
- 延迟/显存/成本：
- 跨场景回归：

## 9. Failure Cases

- 环境变化：
- 工具失败：
- 超时：
- Worker Crash：
- 重复消息：
- Unknown Outcome：

## 10. Demo

- 固定任务：
- 固定版本：
- 执行命令：
- 预期证据：

## 11. 非目标

列出首版明确不做的能力。

## 12. Definition of Done

- [ ] Manifest、Schema、Policy 和 Verifier 已评审
- [ ] 数据集和测试集版本冻结
- [ ] 自动化测试和故障实验通过
- [ ] Serving Benchmark 已生成
- [ ] Demo 可重复
- [ ] 已知限制已记录
```

---

# 3. 知识模块模板

保存到 `docs/knowledge-map.md`，每学习一个主题就填写一节。

```markdown
## [知识主题名称]

### 一句话定义

### 为什么项目需要它

### 核心原理

### 关键公式或数据流

### 工程实现位置

### 最小实践

### 常见失败方式

### 评测与验证方法

### 与相邻概念的区别

### 面试问题

### 当前掌握状态

- [ ] 只听说过
- [ ] 能解释原理
- [ ] 跑通过示例
- [ ] 在项目中实现
- [ ] 做过对照实验
- [ ] 能解释 Trade-off
```

## 必须覆盖的知识主题

```text
Tokenizer / Embedding / Attention / QKV / Causal Mask / RoPE
RMSNorm / MLP / Residual / LM Head / Logits
Greedy / Temperature / Top-k / Top-p / KV Cache
数据清洗 / 去重 / PII / Packing / 泄漏与污染
Causal LM / Optimizer / Warmup / Weight Decay / 梯度裁剪
BF16 / FP16 / 梯度累积 / Gradient Checkpointing
Checkpoint / RNG / 可复现 / OOM 处理
DDP / FSDP / ZeRO / TP / PP 基本边界
继续预训练 / 灾难性遗忘
SFT / Chat Template / Loss Mask
LoRA / QLoRA / Rank / Alpha / Target Modules
输出蒸馏 / Logits 蒸馏 / 特征蒸馏
DPO / ORPO / KTO / Reward Model / RLHF / GRPO
Embedding / 对比学习 / Hard Negative
Reranker / Cross-Encoder
Verifier / Process Reward / Outcome Reward
量化 / GPTQ / AWQ / bitsandbytes
vLLM / Prefill / Decode / Continuous Batching / Prefix Cache
TTFT / TPOT / Tokens/s / Admission Control
Model Router / Confidence / Fallback
MLflow / Dataset Version / Model Registry / Model Card
Tool Use / MCP / Context / Memory / RAG
Durable Execution / Worker / Checkpoint / 幂等 / Unknown Outcome
HITL / 沙箱 / 权限 / 审计 / Prompt Injection
OpenTelemetry / Prometheus / Grafana / Kubernetes / KEDA
```

---

# 4. 单项任务规格模块

每个任务开始前，从待做清单中复制一个 Task ID，保存到 `tasks/specs/`。

```markdown
---
task_id: ENV-001
title: 配置并验证 WSL 模型开发环境
status: planned
priority: P0
owner: unassigned
reviewer: unassigned
depends_on: []
estimated_hours: 4
created_at: YYYY-MM-DD
updated_at: YYYY-MM-DD
---

# ENV-001：配置并验证 WSL 模型开发环境

## 1. 背景

说明当前项目为何需要此任务，以及它阻塞哪些后续任务。

## 2. 已知条件

- 设备：Alienware m18 R1
- GPU：RTX 4090 Laptop GPU，16GB VRAM
- CPU / 内存：i9 / 64GB
- WSL：WSL2 + Ubuntu 24.04
- GPU 透传：已通过 `nvidia-smi` 验证
- Ubuntu 当前状态：未安装 PyTorch，未安装 pip，未安装 nvcc

## 3. 目标

用可验证的结果描述任务完成后系统具备的能力。

## 4. 非目标

明确本任务不处理什么，防止 Agent 擅自扩大范围。

## 5. 允许修改范围

- 可以修改：列出目录和文件。
- 不可修改：列出核心配置、已有模块或外部资源。

## 6. 实施要求

逐项列出必须实现的功能，不写模糊的“完善”“优化”。

## 7. 禁止事项

- 不在 WSL 内安装 Linux NVIDIA 驱动。
- 未经确认不安装系统级 CUDA Toolkit。
- 不删除 Windows 已有 Python/PyTorch 环境。
- 不覆盖用户已有 `.wslconfig`。
- 不把项目放到 `/mnt/c`。

## 8. 交付物

- 代码或脚本：
- 配置文件：
- 文档：
- 测试：
- 日志或截图：

## 9. 自动验收命令

```bash
# 在这里写可直接复制执行的命令
```

## 10. 人工验收步骤

1. 
2. 
3. 

## 11. 失败场景

| 场景 | 预期行为 | 验证方法 |
|---|---|---|
| 网络无法访问 PyPI | 给出诊断，不破坏环境 | 断网或错误代理测试 |
| PyTorch 无法识别 GPU | 输出明确错误和环境快照 | CUDA smoke test |

## 12. Definition of Done

- [ ] 所有验收命令通过
- [ ] 测试覆盖主要成功与失败路径
- [ ] 文档包含完整复现步骤
- [ ] 配置和依赖已锁定
- [ ] 产生可保存的指标或证据
- [ ] 未实现部分已明确记录

## 13. 完成记录

- 完成时间：
- Git commit：
- 执行者：
- 审查者：
- 关键结果：
- 剩余问题：
```

---

# 5. Codex 实现指令模块

保存到 `.ai/codex/<TASK-ID>-implementation.md`，实际执行时将括号内容替换完整。

```markdown
# Codex Implementation Task: [TASK-ID] [任务名称]

## 角色

你是本仓库的实现工程师。严格按照任务规格完成编码、测试和文档，不自行扩大范围。

## 必读文件

1. `tasks/specs/[TASK-ID]-*.md`
2. `docs/architecture.md`
3. `docs/environment.md`
4. `docs/adr/` 中与本任务相关的 ADR
5. 任务涉及目录中的现有 README 和测试

## 当前背景

[复制任务规格中的背景、已知条件和依赖。]

## 本次目标

[列出 3-8 个可验证目标。]

## 允许修改

- `[目录/文件]`

## 禁止修改

- `[目录/文件]`
- 不更换既定技术路线。
- 不删除或绕过现有测试。
- 不加入没有必要的新框架。
- 不把密钥、模型 Token 或本地绝对路径提交到仓库。

## 实施要求

1. [具体要求]
2. [具体要求]
3. [具体要求]

## 质量要求

- 代码有类型标注和必要错误处理。
- 配置不得硬编码在业务逻辑中。
- 新增行为必须有自动化测试。
- 日志不能泄露密钥、完整敏感数据或超长模型输出。
- 命令必须可在全新终端中复现。

## 必须执行的验收命令

```bash
[lint]
[type-check]
[unit tests]
[integration/smoke test]
[benchmark if applicable]
```

## 输出格式

完成后只提交以下内容：

1. 修改文件列表。
2. 核心实现摘要。
3. 已运行命令及结果。
4. 指标或基准结果。
5. 已知限制和未完成项。
6. 建议由 Claude Code 重点审查的风险点。

未通过测试时不得声称任务完成。
```

---

# 6. Claude Code 架构与代码审查模块

保存到 `.ai/claude/<TASK-ID>-review.md`。

```markdown
# Claude Code Review Task: [TASK-ID] [任务名称]

## 角色

你是本仓库的架构和质量审查者。不要为了展示能力重写整个实现；先验证任务目标、正确性、故障行为和可复现性。

## 必读文件

- `tasks/specs/[TASK-ID]-*.md`
- Codex 的实现变更
- 相关测试、ADR、实验记录和运行日志

## 审查目标

1. 实现是否满足任务规格和 Definition of Done。
2. 测试是否真正验证能力，而不是只验证函数能运行。
3. 是否存在数据泄漏、评测污染、错误指标或不公平对照。
4. 是否存在恢复、并发、幂等、权限、安全或资源泄漏问题。
5. 文档和命令能否在干净环境复现。
6. 简历可使用的数字是否由真实实验支持。

## 重点检查清单

### 正确性

- [ ] 输入、输出和异常边界明确
- [ ] 关键状态迁移合法
- [ ] 随机种子和数据切分可复现
- [ ] Checkpoint 恢复包含必要状态

### 模型与数据

- [ ] Base、训练模型和对照条件公平
- [ ] 训练集与评测集无明显泄漏
- [ ] 蒸馏类型描述准确
- [ ] 指标计算正确

### Infra 与安全

- [ ] 重试不会造成重复副作用
- [ ] Unknown Outcome 不会被盲目重试
- [ ] 高风险操作进入审批
- [ ] 日志和配置不泄露密钥

### 工程质量

- [ ] 依赖已锁定
- [ ] 测试可重复运行
- [ ] 无无关大范围重构
- [ ] 文档与实现一致

## 必须亲自运行

```bash
[复制任务验收命令]
```

## 审查输出格式

### 结论

- `APPROVE`
- `APPROVE WITH MINOR FIXES`
- `REQUEST CHANGES`

### 阻塞问题

按严重程度列出，必须给出文件、位置、原因和可验证修复标准。

### 非阻塞建议

只列真正能提高可维护性、性能或可信度的建议。

### 验证结果

记录实际执行过的命令和结果，不接受只阅读代码后的推断。

### 简历可信性检查

指出哪些结果已经能够写入简历，哪些仍缺少实验数据。
```

---

# 7. 实验记录模块

保存到 `docs/experiments/EXP-xxx-*.md`。

```markdown
---
experiment_id: EXP-001
title: QLoRA Tool Router 基线实验
status: planned
code_commit: 
dataset_version: 
model_version: 
config_file: 
run_id: 
hardware: RTX 4090 Laptop 16GB
created_at: YYYY-MM-DD
---

# EXP-001：QLoRA Tool Router 基线实验

## 1. 实验问题

要回答的具体问题是什么？

## 2. 假设

明确预期结果及原因。

## 3. 自变量

- 模型版本：
- 数据版本：
- LoRA rank：
- LoRA alpha：
- Target modules：

## 4. 控制变量

- 随机种子：
- 训练步数：
- 序列长度：
- Batch / Gradient Accumulation：
- 评测集版本：

## 5. 环境

- OS / WSL：
- GPU / VRAM：
- Driver：
- Python：
- PyTorch / CUDA Runtime：
- Transformers / PEFT / TRL：

## 6. 数据

- 来源：
- 许可证：
- 样本数量：
- 长度分布：
- 去重策略：
- Train/Validation/Test 切分：
- 泄漏检查：

## 7. 运行配置

```yaml
# 粘贴最终配置
```

## 8. 执行命令

```bash
# 可复制命令
```

## 9. 结果

| 版本 | Tool Accuracy | JSON 合法率 | 参数 F1 | 风险 Macro F1 | 显存峰值 | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| Base | | | | | | |
| LoRA | | | | | | |
| QLoRA | | | | | | |

## 10. Badcase

| 样本 ID | 预期 | 实际 | 错误类型 | 推测原因 |
|---|---|---|---|---|

## 11. 结论

实验实际支持了什么结论？禁止把相关性写成因果。

## 12. 已知限制

## 13. 下一步

## 14. 可写入简历的结果

仅填写有真实数据支撑的描述。
```

---

# 8. 数据集卡模块

保存到 `docs/dataset-cards/<dataset-name>.md`。

```markdown
# Dataset Card: [名称与版本]

## 用途

## 不适用范围

## 数据来源与许可证

| source_id | 来源 | 许可证 | 获取时间 | 使用范围 |
|---|---|---|---|---|

## 数据结构

## 生成与清洗流程

## 去重方法

## PII 与敏感数据处理

## 数据质量评分

## 数据切分

## 泄漏与污染检查

## 类别和长度分布

## 已知偏差

## 删除与更新策略

## 版本历史
```

---

# 9. 模型卡模块

保存到 `docs/model-cards/<model-name>.md`。

```markdown
# Model Card: [模型名称与版本]

## 基础模型

## 训练目标

## 训练数据

## 训练方式

- CPT / SFT / LoRA / QLoRA / Distillation / DPO / GRPO

## 关键配置

## 适用任务

## 不适用任务

## 评测结果

## 与 Base 的对照

## 安全与权限边界

## 推理资源需求

## 量化版本

## 已知限制

## 发布和回滚信息
```

---

# 10. ADR 架构决策模块

保存到 `docs/adr/ADR-xxx-*.md`。

```markdown
# ADR-[编号]：[决策名称]

- 状态：Proposed / Accepted / Superseded
- 日期：YYYY-MM-DD
- 决策者：

## 背景

## 决策

## 备选方案

### 方案 A

### 方案 B

### 方案 C

## 选择理由

## 正面影响

## 负面影响

## 风险与缓解措施

## 重新评估条件
```

---

# 11. 统一评测报告模块

保存到 `docs/evaluations/EVAL-xxx-*.md`。

```markdown
# EVAL-[编号]：[评测名称]

## 评测目标

## 评测集版本

## 被测版本

## 指标定义

## 运行环境

## 总体结果

## 分组结果

## Base / Ablation 对照

## 置信区间或重复运行

## Badcase 分类

## 延迟、吞吐、显存和成本

## 安全评测

## 是否通过发布门禁

## 结论与限制
```

---

# 12. 故障实验模块

保存到 `docs/failure-tests/FAIL-xxx-*.md`。

```markdown
# FAIL-[编号]：[故障名称]

## 故障目标

## 系统预期行为

## 注入方式

## 执行前状态

## 执行步骤

## 实际行为

## 状态与数据一致性

## 是否产生重复副作用

## Trace、日志和指标证据

## 恢复时间

## 根本原因

## 修复或改进

## 回归测试
```

## 必须记录的故障实验

```text
训练进程中断并恢复
Worker 执行中被杀
Tool 成功后、结果持久化前被杀
同一任务重复投递
LLM 请求超时
Tool 超时与 Unknown Outcome
PostgreSQL 短暂不可用
用户取消任务
Agent 无限循环
新旧 Memory 冲突
Context 压缩遗漏约束
模型服务 OOM 或过载
模型版本灰度失败并回滚
```

---

# 13. 完成与验收记录模块

用于任务从 `current` 移入 `completed`。

```markdown
# [TASK-ID] 完成记录

## 任务目标

## 最终实现

## 修改文件

## 测试结果

## 指标与证据

## Codex 实现摘要

## Claude Code 审查结论

## 人工验收结论

## 已知限制

## 后续任务

## Git 信息

- Branch：
- Commit：
- Tag：

## 状态

- [ ] 实现完成
- [ ] 自动测试通过
- [ ] Claude Code 审查通过
- [ ] 人工验收通过
- [ ] 文档同步完成
- [ ] 可以进入下一个任务
```

---

# 14. 简历项目提取模块

保存到 `docs/resume/`。项目完成后再填写，禁止提前编造数字。

```markdown
# 简历项目：[项目名称]

## 一句话定位

## 背景问题

## 我的职责

## 核心设计

## 关键难点

## 量化结果

- 准确率：
- 延迟：
- 吞吐：
- 显存：
- 成本：
- 恢复时间：
- 故障覆盖：

## 技术栈

## 中文简历描述

## 英文简历描述

## 面试展开故事

问题 → 约束 → 方案 → 验证 → 权衡 → 结果

## 证据链接

- 仓库：
- 演示视频：
- 实验报告：
- 架构图：
```

---

# 15. 项目状态工作流

```text
BACKLOG
  ↓ 编写任务规格
READY
  ↓ 分派 Codex
IMPLEMENTING
  ↓ Codex 运行测试并提交摘要
REVIEWING
  ↓ Claude Code 审查和复测
CHANGES_REQUESTED ──→ IMPLEMENTING
  ↓ 审查通过
HUMAN_ACCEPTANCE
  ↓ 人工验收
DONE
```

每个任务必须满足：

1. 先有任务规格，再开始写代码。
2. Codex 负责实现和自测。
3. Claude Code 负责架构、正确性、故障和可复现性审查。
4. 审查通过不等于最终完成，仍需人工验收关键输出。
5. 实验结果必须写入实验记录，不只保留在终端日志里。
6. 能写入简历的数字必须能够从固定评测或基准脚本复现。

---

# 16. 开工时的最小文件集合

开始第一个任务前，只需先创建：

```text
docs/project-overview.md
docs/architecture.md
docs/environment.md
docs/roadmap.md
tasks/backlog.md
tasks/current.md
tasks/completed.md
tasks/specs/ENV-001-wsl-environment.md
.ai/codex/ENV-001-implementation.md
.ai/claude/ENV-001-review.md
```

其他模块在第一次使用时再创建，避免空文件泛滥。

---

# 17. 当前第一个任务建议

```text
ENV-001：确认 WSL 资源、网络和代理
ENV-002：建立独立 Python 环境
ENV-003：安装并验证 PyTorch CUDA
ENV-004：建立实验追踪骨架
```

执行 ENV-001 前，应先把 PowerShell 和 WSL 的现有检测结果写入 `docs/environment.md`，作为后续环境变化的基线。
