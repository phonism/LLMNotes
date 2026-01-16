---
layout: post
title: "Train Long, Think Short：LLM 推理长度控制方法综述"
date: 2025-12-31 12:00:00
author: Qi Lu
tags: [RL, GRPO, Reasoning, Efficiency]
lang: zh
translation: /en/train-long-think-short/
---

在训练 RLVR 的过程中，即使是一些简单的问题，模型的思维链动辄数千甚至上万 token，但 ChatGPT、Claude 这些主流商业模型的推理过程却相当简洁。这中间到底差了什么？

带着这个问题，调研了**推理长度控制**方面的研究进展，发现这个方向已经有不少工作，大致可以分为训练期优化和推理期控制两类。

---

## 1. 问题背景

### 1.1 Overthinking 现象

在 RLVR（Reinforcement Learning with Verifiable Rewards）场景下，推理模型常出现这些问题：

- **冗余验证**：答案已经正确，但模型继续"Wait, let me verify..."
- **反复犹豫**：使用"Hmm"、"Alternatively"等词反复切换思路
- **长度膨胀**：小模型需要数千 token 才能完成中等难度推理

### 1.2 优化目标

在不牺牲正确率的前提下，最小化推理 token 数：

$$\min_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot\mid x)}[\text{len}(y)] \quad \text{s.t.} \quad \text{Acc}(\pi) \geq \text{Acc}(\pi_0)$$

评估指标包括：
- **Accuracy-Length Pareto Front**：同正确率下更短，或同长度下更准
- **正确样本的长度分布**：关注长尾而非仅均值

---

## 2. 训练期方法

### 2.1 硬截断：ThinkPrune

**论文**: [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296)
**时间**: 2025-04
**机构**: UCSB
**代码**: [GitHub](https://github.com/UCSB-NLP-Chang/ThinkPrune)

**思路**：训练时设置 token 上限，超过上限的未完成推理被截断，reward 直接归零。通过多轮迭代逐步收紧上限，迫使模型学会更简洁的推理。

**方法**：
1. 设置初始长度上限 $L_0$
2. 超过上限的样本无法得到有效答案 → reward = 0
3. 迭代收紧：$L_{t+1} = \alpha \cdot L_t$，其中 $\alpha < 1$

**实验结果**：
- DeepSeek-R1-Distill-Qwen-1.5B 在 AIME24 上长度减半，正确率仅下降 2%
- DeepScaleR-1.5B-Preview：5,914 → 3,370 tokens
- QwQ-32B：8,763 → 4,494 tokens

**优点**：不需要复杂的 reward 工程，目标清晰
**风险**：上限过紧会截断正确解

---

### 2.2 长度奖励：GRPO-LEAD

**论文**: [GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning](https://arxiv.org/abs/2504.09696)
**时间**: 2025-04
**代码**: [GitHub](https://github.com/aeroplanepaper/GRPO-LEAD)

**LEAD = Length-dependent rewards + Explicit penalties + Advantage reweighting for Difficulty**

这个方法包含三个改动：

1. **Length-dependent accuracy reward**：对答对样本按长度排序打分，鼓励更短的正确解

2. **显式惩罚错误**：对答错样本额外施加负向约束

3. **Difficulty-aware advantage reweighting**：根据问题的经验正确率确定难度权重，对更难的问题放大学习信号

值得注意的是，长度排序只在正确样本内进行，错误样本另用惩罚项处理。

**实验结果**：14B 规模模型达到 SOTA 性能，显著提升推理准确性、简洁性和效率。

---

### 2.3 分段 Shaping：LASER

**论文**: [Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](https://arxiv.org/abs/2505.15612)
**时间**: 2025-05
**代码**: [GitHub](https://github.com/hkust-nlp/Laser)

这篇工作提出了一个统一框架，将各种高效推理方法形式化为长度相关的 reward shaping。基于此框架，作者提出 **LASER（Length-bAsed StEp Reward shaping）**，使用阶跃函数作为奖励：

$$r_{\text{shaped}}(y) = r_{\text{task}}(y) + f(\text{len}(y))$$

**LASER-D（Dynamic and Difficulty-aware）扩展**：
1. 模型训练过程中推理行为会演化，奖励规格也需要自适应和动态调整
2. 长度奖励应该是难度感知的——对简单问题更多惩罚长 CoT

**实验结果**：LASER-D 在 AIME2024 上提升 +6.1 分，同时减少 63% token 使用。

---

### 2.4 自适应约束：LEASH

**论文**: [Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model](https://arxiv.org/abs/2512.21540)
**时间**: 2025-12

LEASH 把长度控制写成约束优化问题，使用 **Lagrangian Primal-Dual** 方法动态调整惩罚系数：

$$\max_\pi \mathbb{E}[r_{\text{task}}] \quad \text{s.t.} \quad \mathbb{E}[\text{len}(y)] \leq L_{\text{target}}$$

**动态调整机制**：
- 生成超过目标长度 → 惩罚增强
- 生成短于目标长度 → 惩罚放松

**One-sided penalty**：只惩罚"过长"，避免激励模型无限变短。

**实验结果**：在 Deepseek-R1-Distill-Qwen-1.5B 和 Qwen3-4B-Thinking-2507 上，跨任务平均推理长度减少 60%（包括分布内数学推理和分布外代码、指令遵循任务），同时保持竞争性性能。

---

### 2.5 课程学习：Train Long, Think Short

**论文**: [Train Long, Think Short: Curriculum Learning for Efficient Reasoning](https://arxiv.org/abs/2508.08940)
**时间**: 2025-08
**代码**: [GitHub](https://github.com/hammoudhasan/curriculum_grpo)

采用 curriculum 方式，先让模型"学会做题"，再逐步压缩预算：

1. **Phase 1**：慷慨的 token 预算，让模型探索有效的解题策略
2. **Phase 2**：逐步收紧预算，鼓励模型将策略蒸馏为更简洁的推理链
3. **组合训练信号**：正确性（验证器反馈）+ 长度效率 + 格式遵循

**实验结果**：在 GSM8K、MATH500、SVAMP、College Math、GSM+ 上，课程式训练在相同最终预算下始终优于固定预算基线。

---

### 2.6 提示可控：L1 / LCPO

**论文**: [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697)
**时间**: 2025-03
**主页**: [CMU L3 Lab](https://cmu-l3.github.io/l1)

**LCPO（Length Controlled Policy Optimization）** 把目标长度写入 prompt：

- **LCPO-Exact**："Think for exactly N tokens"
- **LCPO-Max**："Think for maximum N tokens"

RL 目标中加入长度偏差项，实现可控预算推理。

**实验结果**：
- 1.5B 的 L1 模型在相同推理长度下超越 GPT-4o
- 优于 s1（Budget Forcing）基线
- 可导出 Short Reasoning Models (SRMs)：CoT 长度与非推理模型相当，但保留推理模式

---

### 2.7 难度自适应长度惩罚：Just Enough Thinking

**论文**: [Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning](https://arxiv.org/abs/2506.05256)
**时间**: 2025-06

LRM 经常对简单问题"过度思考"——比如 DeepSeek-R1 和 Qwen-QwQ32B 回答"2+3=?"居然生成超过 10,000 tokens。

这篇工作提出 **Adaptive Length Penalty (ALP)**，根据每个 prompt 的**在线求解率**调整惩罚幅度：
- 高求解率（简单）prompt → 高额外 token 成本
- 低求解率（困难）prompt → 惩罚不变

简单来说，就是让模型在简单问题上节省 token，把预算留给困难问题。

**实验结果**：
- DeepScaleR-1.5B 使用 ALP 后训练，**平均 token 使用减少 50%**，性能基本不降
- 相比固定预算和均匀惩罚基线，ALP 在最难问题上准确率更高

---

### 2.8 Long2Short：Kimi k1.5

**论文**: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
**时间**: 2025-01
**机构**: Moonshot AI
**代码**: [GitHub](https://github.com/MoonshotAI/Kimi-k1.5)

长 CoT 推理准确率高但计算开销大。Kimi k1.5 提出 **Long2Short** 技术，将长 CoT 策略压缩为更高效的短 CoT 表示。

**三种 Long2Short 方法**：

| 方法 | 描述 |
|------|------|
| **Model Merging** | 长 CoT 和短 CoT 模型权重平均 |
| **Shortest Rejection Sampling** | 从多个正确响应中选择最短的做 SFT |
| **Preference-based RL** | 训练模型在保持正确性的前提下偏好简洁 |

**实验结果**（短 CoT SOTA）：
- AIME 2024: **60.8**
- MATH500: **94.6**
- LiveCodeBench: **47.3**
- 超越 GPT-4o 和 Claude Sonnet 3.5 高达 **+550%**

---

### 2.9 长度协调微调：O1-Pruner

**论文**: [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570)
**时间**: 2025-01
**代码**: [GitHub](https://github.com/StarDewXXX/O1-Pruner)

O1-like 长思考模型难以根据问题难度和推理冗余有效分配 token 预算。O1-Pruner 提出 **Length-Harmonizing Fine-Tuning** 来解决这个问题：
1. **Pre-sampling**：估计模型在不同问题上的基线性能
2. **RL-style Fine-tuning**：在准确性约束下，鼓励模型生成更短的推理过程

**实验结果**：
- 推理开销减少 **50%**
- 准确率不降反升
- 适用于各种数学推理基准

---

### 2.10 简洁性引导 RL：ConciseRL

**论文**: [ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models](https://arxiv.org/abs/2505.17250)
**时间**: 2025-05

推理轨迹常常在得出正确答案后继续延伸，造成计算浪费、可读性下降甚至幻觉。ConciseRL 引入**无超参数的简洁性评分**作为 RL 奖励信号：
- 使用 LLM-as-judge 评估简洁性
- 动态、上下文感知的反馈（不仅仅是 token 数量）

**实验结果**：
- TheoremQA: 准确率 **+2.2%**，同时使用 **12.5x 更少** tokens
- 根据问题难度动态调整推理长度
- 更强的 judge 模型带来更大收益

---

## 3. 推理期方法

### 3.1 答案收敛：Answer Convergence

**论文**: [Answer Convergence as a Signal for Early Stopping in Reasoning](https://arxiv.org/abs/2506.02536)
**时间**: 2025-06

一个有意思的发现：在 MATH 等数学推理任务上，模型通常在 **60% 推理步骤后**就已收敛到最终答案，剩余内容基本是冗余。

基于这个观察，作者提出了三类推理期策略：
1. **Answer Consistency 早停**：连续推理块产生相同答案时停止
2. **Think Token Adjustment**：提高生成结束推理信号的概率
3. **Learn-to-Stop**：基于内部激活训练"何时停止"分类器

**实验结果**：
- Learn-to-Stop 在 NQ + QwQ-32B 上减少 48% token，有时甚至提升准确率
- Answer Consistency 在 NaturalQuestions 上减少 40%+ token 并提升准确率

---

### 3.2 Step Answer 监测：ES-CoT

**论文**: [Early Stopping Chain-of-thoughts in Large Language Models](https://arxiv.org/abs/2509.14004)
**时间**: 2025-09

几个关键概念：

- **Step Answer**：模型在每个推理步骤的当前答案猜测
- **Run**：连续相同答案的步骤序列
- **Run-Jump Test**：当相同 step answer 的 run length 出现统计显著跳变时，终止推理

思路很直接："stop thinking when the answer stabilizes"——无需额外模型或重训练。

**实验结果**：在 5 个推理数据集、3 个 LLM 上，ES-CoT 平均减少 **41%** 生成 token，同时保持与原始 CoT 相当的准确率。

---

### 3.3 思路切换点监测：DEER

**论文**: [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)
**时间**: 2025-04
**代码**: [GitHub](https://github.com/iie-ycx/DEER)

DEER 的观察是：长 CoT 中存在 "pearl reasoning"——足够但不冗余的关键位置。

具体做法：
1. 监测 **Action Transition Points（ATP）**：如 "Wait", "Alternatively" 等思路切换点
2. 在 ATP 诱导试答
3. 用置信度决定是否提前结束——推理不完整时试答置信度低，推理充分时置信度高

**优点**：无需额外训练，可无缝集成到现有 o1-like 推理 LLM。

**实验结果**：在 10 个推理基准（GSM8K、MATH-500、AMC、GPQA、AIME、LiveCodeBench）、11 个前沿推理 LLM 上：
- CoT 长度平均减少 19.1% - 80.1%
- 准确率提升 0.3% - 5.0%

---

### 3.4 推理三阶段理论：Stop Spinning Wheels

**论文**: [Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit](https://arxiv.org/abs/2508.17627)
**时间**: 2025-08

这篇工作将推理过程分为**三个阶段**：
1. **Insufficient Exploration Stage**：探索不足阶段
2. **Compensatory Reasoning Stage**：补偿推理阶段——通常在此阶段产生正确答案
3. **Reasoning Convergence Stage**：推理收敛阶段——常触发 overthinking

关键是找到 **Reasoning Completion Point (RCP)** —— 补偿推理阶段结束的位置，通常出现在第一个完整推理周期末尾。

检测 RCP 的方法包括：
- 逐句查询 LLM
- 监测 `</think>` 等结束思考 token 的概率
- 挖掘更敏感一致的 RCP 模式 + 轻量级阈值策略

**实验结果**：在 AIME24、AIME25、GPQA-D 上减少 token 消耗，同时保持或提升推理准确率。

---

### 3.5 Budget Forcing：s1

**论文**: [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
**时间**: 2025-01
**代码**: [GitHub](https://github.com/simplescaling/s1)

s1 的做法很简洁：
- 精选 1,000 条问题+推理轨迹的小数据集 **s1K**
- 在 Qwen2.5-32B-Instruct 上做 SFT（仅需 26 分钟，16×H100）
- **Budget Forcing**：通过强制终止或反复追加 "Wait" 控制推理长度

**效果**：
- s1-32B 在竞赛数学题上比 o1-preview 高 27%（MATH 和 AIME24）
- Budget forcing 可将 AIME24 从 50% 提升到 57%

---

### 3.6 抑制反思词：NoWait

**论文**: [Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency](https://aclanthology.org/2025.findings-emnlp.394/)
**时间**: EMNLP 2025
**arXiv**: [2506.08343](https://arxiv.org/abs/2506.08343)

Budget forcing 在很多推理模型上并不总是有效。这篇工作的观察是：显式自我反思（"Wait", "Hmm", "Alternatively"）可能并非必要。

做法很简单：推理期对特定"反思/迟疑" token 做 **logit 抑制**：
1. 识别关键反思词（通过 32 次独立运行统计最频繁的单语词）
2. 推理时抑制这些 token 的生成

**实验结果**：在 5 个 R1 风格模型系列（QwQ、Phi4、Qwen3、Kimi-VL、QvQ）上：
- CoT 长度减少 **27%-51%**
- 跨文本、视觉、视频推理任务保持模型效用
- 即插即用，无需训练

---

### 3.7 动态预算：ABF

**论文**: [Reasoning at the Right Length: Adaptive Budget Forcing for Efficient and Accurate LLM Inference](https://openreview.net/forum?id=ieBgxTG7Mt)
**时间**: 2025-09

**Adaptive Budget Forcing (ABF)** 通过监测实时确定性信号（token 级置信度、熵、语义一致性）动态调整推理长度：
- 置信度足够 → 停止生成
- 置信度不足 → 继续推理

**与传统 Budget Forcing 的区别**：传统方法使用固定长度约束或预定控制 token，ABF 实时监测模型的"思考轨迹"并自适应做出停止决策。

---

## 4. 方法分类总结

### 训练期方法

| 类别 | 核心思想 | 代表工作 |
|------|----------|----------|
| **Reward Shaping** | 在 RL 奖励中加入长度惩罚项，鼓励模型生成更短的正确推理 | ThinkPrune, GRPO-LEAD, LASER, LEASH, Just Enough Thinking, ConciseRL |
| **课程/蒸馏** | 先让模型学会解题，再逐步压缩推理长度或从长 CoT 蒸馏到短 CoT | Train Long Think Short, Kimi k1.5, O1-Pruner |
| **提示可控** | 训练模型根据 prompt 中的预算指令控制推理长度 | L1/LCPO |

### 推理期方法

| 类别 | 核心思想 | 代表工作 |
|------|----------|----------|
| **早停检测** | 监测答案收敛、置信度或推理完成信号，提前终止生成 | Answer Convergence, ES-CoT, DEER, Stop Spinning Wheels |
| **Token 干预** | 通过强制预算、抑制反思词或动态阈值控制生成长度 | s1, NoWait, ABF |

---

## 5. 开放问题

1. **正确性与效率的 Trade-off**：如何保证压缩不伤正确率？
2. **难度感知**：简单题压缩多、难题保留长思考
3. **泛化性**：训练期方法能否泛化到 OOD 任务？
4. **推理期 vs 训练期**：两类方法能否有效结合？

---

## 参考文献

### 训练期方法
- ThinkPrune: [arXiv:2504.01296](https://arxiv.org/abs/2504.01296)
- GRPO-LEAD: [arXiv:2504.09696](https://arxiv.org/abs/2504.09696)
- LASER: [arXiv:2505.15612](https://arxiv.org/abs/2505.15612)
- LEASH: [arXiv:2512.21540](https://arxiv.org/abs/2512.21540)
- Train Long, Think Short: [arXiv:2508.08940](https://arxiv.org/abs/2508.08940)
- L1/LCPO: [arXiv:2503.04697](https://arxiv.org/abs/2503.04697)
- Just Enough Thinking: [arXiv:2506.05256](https://arxiv.org/abs/2506.05256)
- Kimi k1.5: [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
- O1-Pruner: [arXiv:2501.12570](https://arxiv.org/abs/2501.12570)
- ConciseRL: [arXiv:2505.17250](https://arxiv.org/abs/2505.17250)

### 推理期方法
- Answer Convergence: [arXiv:2506.02536](https://arxiv.org/abs/2506.02536)
- ES-CoT: [arXiv:2509.14004](https://arxiv.org/abs/2509.14004)
- DEER: [arXiv:2504.15895](https://arxiv.org/abs/2504.15895)
- Stop Spinning Wheels: [arXiv:2508.17627](https://arxiv.org/abs/2508.17627)
- s1: [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)
- NoWait: [arXiv:2506.08343](https://arxiv.org/abs/2506.08343)
- ABF: [OpenReview](https://openreview.net/forum?id=ieBgxTG7Mt)
