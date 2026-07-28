# Multimodal Agent Model Factory × Reliable Runtime

> 一个面向国内外大模型工程岗位的旗舰母项目：从多模态数据、后训练、评测和推理服务，一直走到可恢复、安全受控的真实 Agent 执行与 Badcase 回流。

## 一句话定位

本项目不绑定自动驾驶或单一业务场景。它以桌面 GUI 作为第一个可验证环境，逐步扩展到文档、浏览器、图表、音频、视频，以及可选的机器人或自动驾驶仿真场景。

核心目标不是堆砌框架，而是证明一条完整且可重复的生产闭环：

```text
多模态数据与 Agent Trace
→ 清洗、脱敏、质量与版本治理
→ SFT / QLoRA / Distillation / DPO / GRPO
→ VLM Action Model / Tool Router / Retriever / Verifier
→ Quantization / vLLM / Model Router / Fallback
→ Reliable Agent Runtime
→ Trace / Eval / Badcase / 人工审核
→ Dataset vN+1 / Model vN+1
```

## 已确认的项目决策

| 决策 | 结论 |
|---|---|
| 是否先做现有项目 MVP | 是；冻结可靠 Runtime，优先补模型全周期闭环 |
| 是否绑定自动驾驶 | 否；自动驾驶只是可选环境，主线是通用多模态 LLM/Agent Infra |
| 本地小模型的角色 | 负责感知、Grounding、路由、风险和 Verifier；强模型负责复杂长程规划 |
| 模型能否直接执行 | 不能；模型只产生候选动作，确定性 Runtime 掌握权限 |
| 如何兼顾岗位广度和技术深度 | 一个旗舰母项目负责广度，四个独立 Lab 负责深度 |
| MVP 是否一次性 | 不是；每一版保持完整闭环并逐步扩展 |
| 如何面向国内外岗位 | 共用代码和证据，按 Post-training、Agent、Serving、ML Systems 生成不同简历切片 |
| 如何描述 Full Cycle | 使用“post-training-to-deployment lifecycle”；无大规模证据时不声称完整基础模型预训练 |

## 为什么从现有 Runtime 出发

现有 `guarded-desktop-agent` 已经具备：

- UIA、截图、区域图、OCR、文档文本等多源观察；
- 工具调用、动作后重新观察和固定执行边界；
- Policy、Approval、Grounding、预算和审计；
- Worker 崩溃恢复、幂等与 Unknown Outcome 治理；
- Trace、固定评测和故障证据；
- 当前完整测试基线：`1420 passed, 7 skipped`。

因此不应重写 Runtime。它在本项目中承担三个角色：

1. 多模态交互环境；
2. 真实视觉—语言—动作轨迹的数据生产器；
3. 训练后模型的安全执行、在线评测和 Badcase 回流平台。

## 项目结构：一个母项目，四个深度实验

```text
Multimodal Agent Model Factory
├── 主干：Data / Post-training / Eval / Serving / Runtime
├── Lab A：Tiny Transformer & Pretraining
├── Lab B：Multimodal Post-training & Agentic RL
├── Lab C：Distributed Training & Inference Performance
└── Lab D：Multi-Agent Coordination & Distributed Agent Systems

Environments
├── Desktop GUI
├── Document / Chart / PDF
├── Browser Research
├── Audio / Video
└── Robotics / Autonomous Driving Simulation（可选）
```

场景不只按模态划分，还按运行环境和业务目标组合。正式定义见：[多模态与业务场景覆盖矩阵](多模态与业务场景覆盖矩阵.md)。

### 母项目负责广度

- 多模态数据工程与数据版本；
- VLM/LLM 后训练和统一 Eval；
- Tool Use、GUI Grounding、风险与回退；
- vLLM Serving、量化、路由和灰度；
- Agent Runtime、审批、恢复、Trace；
- Trace → Badcase → 再训练闭环。

### 四个 Lab 负责深度

| Lab | 证明什么 | 关键产物 |
|---|---|---|
| Tiny Transformer & Pretraining | 理解模型结构、训练状态和推理缓存 | Decoder、RoPE、KV Cache、CPT、可恢复训练 |
| Multimodal Post-training & Agentic RL | 具备当前 VLM/Agent 后训练能力 | QLoRA、DPO/GRPO、可验证奖励、Verifier、消融 |
| Distributed Training & Inference Performance | 具备 AI Infra 与性能分析深度 | DDP/FSDP、vLLM、量化、Profiler、Triton 最小实验 |
| Multi-Agent Coordination & Distributed Agent Systems | 具备多 Agent 调度、共享状态、可靠恢复和安全委派能力 | Coordinator、Typed Message、Lease、冲突仲裁、Single-Agent 对照 |

## MVP 不是一次性 Demo

每个 MVP 都保持完整闭环，只增加一个主要变量：

| 版本 | 目标 | 完成标准 |
|---|---|---|
| MVP-0 | 冻结可靠执行基线 | 测试、故障恢复、安全边界和证据可复现 |
| MVP-1 | 文本 Tool Router | Trace→Dataset→QLoRA→Eval→Runtime→Badcase |
| MVP-2 | 图文 GUI Action Model | Screenshot/UIA/OCR 联合输入，输出动作、风险与回退 |
| MVP-3 | 多模态后训练与 Verifier | SFT、蒸馏、DPO/GRPO 对照和轨迹门禁 |
| MVP-4 | 多模型 Serving | vLLM、量化、缓存、路由、灰度和性能报告 |
| MVP-5 | Agentic RL | Runtime 作为环境，使用可验证奖励训练 |
| MVP-6 | 多环境、多模态 | 文档、浏览器、音视频或仿真环境适配 |
| MVP-7 | AI Infra 深化 | 多卡、故障恢复、性能分析和底层优化实验 |
| MVP-8 | Multi-Agent 系统 | 在 Coding 场景完成协调、委派、恢复和 Single-Agent 对照 |

详见：[多模态 LLM 全周期 MVP 演进路线](多模态LLM全周期_MVP演进路线.md)。

## 每一版的四个门禁

1. **功能门禁**：新增能力在固定任务上跑通。
2. **回归门禁**：旧任务和安全契约不能被改软。
3. **安全门禁**：误批准、越权和重复副作用不能上升。
4. **性能门禁**：显存、延迟、吞吐和成本在预算内。

每次实验必须绑定：

```text
code commit
+ dataset version
+ model version
+ config / seed / hardware
+ eval report
+ serving benchmark
+ failure report
+ demo evidence
```

## 求职定位

这是一项母项目，而不是一段对所有岗位都使用相同措辞的简历经历。

| 简历版本 | 重点 |
|---|---|
| Multimodal / Post-training | VLM 数据、SFT、DPO/GRPO、Reward/Verifier、Eval |
| Agent / Applied LLM | GUI Grounding、Tool Use、长程任务、安全与恢复 |
| AI Infra / Serving | vLLM、量化、缓存、路由、灰度、性能与可观测性 |
| ML Systems / Training | Checkpoint、DDP/FSDP、故障恢复、Profiler 和吞吐 |
| Multi-Agent / Distributed Agents | Coordinator、能力委派、共享状态、Worker 恢复、冲突和预算 |

推荐英文定位：

> End-to-end multimodal post-training, evaluation, serving, and reliable agent deployment lifecycle.

在没有大规模预训练证据前，不写成 “full foundation-model pretraining experience”。纯 Research Scientist、CUDA Kernel 专家和超大规模训练岗位仍需要论文或独立的系统深度证据。

## 推荐阅读顺序

1. [项目状态](PROJECT_STATUS.md)：新 session 的唯一当前目标和跨仓库顺序。
2. 本页：项目定位和整体结构。
3. [多模态 LLM 全周期 MVP 演进路线](多模态LLM全周期_MVP演进路线.md)：具体版本、指标和扩展规则。
4. [多模态与业务场景覆盖矩阵](多模态与业务场景覆盖矩阵.md)：模态、环境、模型任务、业务场景和 SCN-001～009。
5. [Desktop Runtime 依赖与集成](Desktop_Runtime_依赖与集成.md)：跨仓库分工、安全数据通道、Backlog 和 Pin 规则。
6. [待做任务清单](AI_Infra_LLM_Agent_待做任务清单.md)：可执行任务、依赖和 Definition of Done。
7. [写作与执行模块模板](AI_Infra_LLM_Agent_写作与执行模块模板.md)：实验、ADR、评测和任务分派格式。
8. [系统研发行动手册 PDF](AI_Infra_LLM_Agent系统研发行动手册_模型开发项目完整版.pdf)：知识地图和背景材料。
9. [Dify / LangGraph 两天补齐与对照计划](Dify_LangGraph_两天补齐与对照计划.md)与[观察记录表](观察记录表.md)：主流框架补齐和自研 Runtime 对照。

## 当前原则

- 先完成可运行、可评测的垂直闭环，再拓展模态和场景。
- 结构化观察优先，视觉模型用于补足 UIA/OCR 无法表达的信息。
- 模型负责提出动作，不获得执行权限；确定性 Runtime 始终掌握安全边界。
- 一个阶段只引入一个主要变量，避免模型、数据、Prompt 和 Runtime 同时变化。
- 任何没有真实运行证据的能力，不写成“熟练掌握”或“生产经验”。
