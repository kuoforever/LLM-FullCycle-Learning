# Reliable Agent Model Lifecycle（可靠 Agent 模型全周期）

> 一个覆盖多模态数据、后训练、评测、Serving、可靠 Agent 执行与 Badcase
> 驱动模型迭代的完整目标系统。

English (default): [README.md](README.md)

## 文档入口

| 文档 | 用途 |
|---|---|
| [项目状态](PROJECT_STATUS.md) | 当前阶段、唯一活动目标和最新验证结果 |
| [项目文档索引](docs/README.zh-CN.md) | 已实现技术证据、维护文档和开发资料 |
| [待做任务清单](AI_Infra_LLM_Agent_待做任务清单.md) | 详细任务、依赖和 Definition of Done |
| [Desktop Runtime 依赖与集成](Desktop_Runtime_依赖与集成.md) | 跨仓所有权、安全边界和版本 pin |

## 一句话定位

`Reliable Agent Model Lifecycle` 是完整系统的正式名称，不局限于某一个 MVP。
项目不绑定自动驾驶或单一业务场景，以桌面 GUI 作为第一个可验证环境，并随着
对应阶段落地逐步扩展到文档、浏览器、图表、音频、视频，以及可选的机器人或
自动驾驶仿真场景。

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
| 如何描述完整系统 | 使用“post-training-to-deployment lifecycle”；名称覆盖完整目标架构，具体完成度以状态和证据为准 |

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
Reliable Agent Model Lifecycle
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

现有 `FC-*` 任务号、`fullcycle_*` 契约字段、Python 包名和 CLI 名称为兼容冻结
产物继续保留，不随展示名称一起修改。

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

## 文档导航

- 项目状态与维护：[项目状态](PROJECT_STATUS.md)。
- 已实现技术证据：[项目文档索引](docs/README.zh-CN.md)。
- 后续开发：[待做任务清单](AI_Infra_LLM_Agent_待做任务清单.md)。
- 项目路线：[多模态 LLM 全周期 MVP 演进路线](多模态LLM全周期_MVP演进路线.md)。
- 场景边界：[多模态与业务场景覆盖矩阵](多模态与业务场景覆盖矩阵.md)。
- 实验与验收：[写作与执行模块模板](AI_Infra_LLM_Agent_写作与执行模块模板.md)。

## 当前原则

- 先完成可运行、可评测的垂直闭环，再拓展模态和场景。
- 结构化观察优先，视觉模型用于补足 UIA/OCR 无法表达的信息。
- 模型负责提出动作，不获得执行权限；确定性 Runtime 始终掌握安全边界。
- 一个阶段只引入一个主要变量，避免模型、数据、Prompt 和 Runtime 同时变化。
- 任何没有真实运行证据的能力，不写成“熟练掌握”或“生产经验”。

## 已实现：Runtime Lane A bridge consumer

`FC-BRIDGE-001` 在本仓库提供严格、完全离线的 manifest v1 与 redacted
run-export v1 consumer、固定合法/非法 compatibility fixtures 和可复制验证
脚本。契约、边界和错误行为见
[FC-BRIDGE-001 离线消费契约](docs/FC-BRIDGE-001.md)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\validate_bridge.ps1
```

## 已实现：Lane A Reliability/Verifier Dataset v1

`FC-BRIDGE-002` 将通过 bridge gate 的脱敏 run export 确定性映射为 canonical
JSONL。v1 只生成可由 Runtime 事实确定的 failure、unknown outcome、policy
denial、recovery 和 budget signals，并保留 tool sequence/outcome 特征；不生成
SFT 文本、富多模态 episode 或需要语义猜测的 retry/rollback 标签。设计决策见
[ADR-0001](docs/adr/ADR-0001-lane-a-reliability-dataset-v1.md)。

## MVP-0 离线基线门禁

Bridge 与 Reliability Dataset 基线使用 Python 3.11–3.13 标准库，不包含
runtime 第三方依赖。统一验证命令：

```powershell
python -I .\scripts\validate_offline.py
```

该 gate 会核对冻结 artifact hashes、依赖边界、全部单测、bridge fixture 和
exact dataset JSONL。环境证据见 [Environment baseline](docs/environment.md)。

## MVP-1 Tool Router schema/eval gate

`FC-MVP-001` 的首个训练前门禁已定义严格 Tool Router decision v1，保存
20 条人工种子和 20 条冻结 eval gold fixtures，并提供确定性规则 baseline。
数据扩展门禁另保存 160 条 train、40 条 validation 和 60 个显式 task
families；类别完全平衡，并对 family overlap、exact/near duplicate、分布与
危险误审批执行离线审计。冻结 eval digest 保持不变。
该门禁只生成候选路由决定，不执行工具，也不打开 Provider、MCP、Desktop、
网络、Memory、Continuation 或训练。契约、固定 digest、指标与已知限制见
[FC-MVP-001 schema/eval gate](docs/FC-MVP-001-schema-eval.md)。

## MVP-1 本地 Base Model baseline

已固定 `Qwen/Qwen2.5-1.5B-Instruct` 的 Apache-2.0 Hub revision、权重
SHA-256、本地 BF16/SDPA greedy generation 配置、原始 predictions 和独立
scorer report。20 条 eval 的 JSON validity 为 1.0，但 Tool Accuracy 仅
0.20，且两个危险请求均产生危险动作候选，因此该模型明确不可接入 Runtime。
完整环境、命令、指标和限制见
[FC-MVP-001 local base-model baseline](docs/FC-MVP-001-base-model-v1.md)。

## MVP-1 首次本地 LoRA SFT

已在冻结的 160/40 train/validation v1 上完成一次 BF16 LoRA SFT，并用未改动
的 20 条 eval 和同一 scorer 对比。Tool Accuracy 从 0.20 提升到 0.80，
argument exact match 从 0 提升到 0.35，危险动作候选从 2 降到 1；但安全门禁
仍未通过，因此 Adapter 明确不可接入 Runtime。仓库保存可独立加载的 Adapter、
训练证据、原始 predictions、对比报告和 safe-merge 验证。完整配置、命令、
指标与限制见
[FC-MVP-001 local LoRA SFT v1](docs/FC-MVP-001-lora-sft-v1.md)。
