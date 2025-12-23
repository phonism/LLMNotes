---
layout: post
title: "LLM-RL 中的熵控制：从 Entropy Collapse 到 Exploration 的系统性综述"
date: 2025-12-23 12:00:00
author: Qi Lu
tags: [RL, Entropy, GRPO, RLVR, Negative Samples]
lang: zh
---

## 引言

2025 年，LLM 强化学习（特别是 RLVR: Reinforcement Learning with Verifiable Rewards）领域爆发了大量关于 **Entropy Collapse**（熵崩溃）问题的研究。这个问题的核心是：RL 训练过程中，模型的输出多样性会不可避免地下降，导致探索能力丧失、过早收敛到次优解。

本文按**论文发表时间线**，系统梳理这一领域的关键工作，提取核心观点和公式，最后进行统一分析。

---

## 论文时间线

### 2025年3月：DAPO — 工业级 RL 系统的熵控制实践

**论文**: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
**机构**: ByteDance Seed
**发表时间**: 2025-03-18

#### 核心问题

DAPO 团队在大规模 RL 训练中观察到：
> "PPO and GRPO suffer from entropy collapse — entropy of policy decreases quickly with training, causing sampled responses to become identical."

#### 核心方法：Clip-Higher（解耦裁剪）

标准 GRPO 使用单一参数 $\epsilon$ 控制裁剪范围：

$$\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$$

DAPO 将其**解耦**为 $\epsilon_{\text{low}}$ 和 $\epsilon_{\text{high}}$：

$$\text{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$$

**关键配置**: $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$

> "By increasing the value of $\epsilon_{\text{high}}$, we leave more room for the increase of low-probability tokens. This adjustment effectively enhances the policy's entropy."

#### 四大技术

| 技术 | 作用 |
|------|------|
| **Clip-Higher** | 增加低概率 token 的上升空间，缓解熵崩溃 |
| **Dynamic Sampling** | 提高训练效率和稳定性 |
| **Token-Level PG Loss** | 适配长 CoT 场景 |
| **Overlong Reward Shaping** | 减少截断样本的奖励噪声 |

#### 实验结果

基于 Qwen2.5-32B，AIME 2024 达到 50 分，超过 DeepSeek-R1-Zero-Qwen-32B（47 分），且只用 50% 训练步数。

---

### 2025年5月：SEED-GRPO — 语义熵引导的不确定性感知优化

**论文**: [SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization](https://arxiv.org/abs/2505.12346)
**发表时间**: 2025-05-18

#### 核心问题

> "Vanilla GRPO treats all prompts equally during policy updates, ignoring important information about the model's knowledge boundaries."

#### 核心观点：语义熵 vs Token 熵

- **Token 熵**: 衡量单个 token 位置的生成不确定性
- **语义熵**: 衡量多个回答的**语义多样性**，聚类基于含义而非形式

> "Semantic entropy clusters responses based on meaning rather than form. This makes semantic entropy a more faithful and robust indicator of a model's uncertainty."

#### 核心公式

给定 prompt $q$，采样 $G$ 个 response $\{o_1, ..., o_G\}$，计算语义熵后调制 advantage：

$$\hat{A}_i = f(\text{SE}(q)) \cdot (r_i - \bar{r})$$

其中 $\text{SE}(q)$ 是语义熵，$f$ 是调制函数。

**策略**：语义熵高（模型不确定）→ 保守更新；语义熵低（模型确定）→ 激进更新。

#### 实验结果

| Benchmark | SEED-GRPO |
|-----------|-----------|
| AIME24 | 56.7 |
| AMC | 68.7 |
| MATH | 83.4 |

---

### 2025年5月：Unearthing Gems from Stones — 从负样本中挖掘正确步骤

**论文**: [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning](https://arxiv.org/abs/2505.14403)
**机构**: 中科院自动化所、StepFun
**发表时间**: 2025-05-20

#### 核心问题

> "Negative responses contain valuable components such as self-reflection and error-correction steps, yet existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL)."

#### 核心方法：BCPG-NSA

**三阶段流程**：

1. **样本分割**: 使用 SAT 模型将长推理轨迹分割为独立步骤
2. **共识评估**: LLM Judge + PRM 双重判断步骤正确性
3. **NSA 优化**: 对负样本中的正确步骤给予正向奖励

#### 核心思想

> "Mining positive steps within negative samples" — 不是简单惩罚整个负样本，而是**捞起**其中思路正确的 token。

#### 实验结果

基于 DeepSeek-R1-Distill-Qwen-14B：

| Method | AIME24 | AIME25 |
|--------|--------|--------|
| BCPG | 70.50% | 52.00% |
| **BCPG-NSA** | **72.17%** | **54.42%** |

---

### 2025年5月：The Entropy Mechanism — 熵-性能交换的数学定律

**论文**: [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617)
**机构**: PRIME-RL (上海 AI Lab)
**发表时间**: 2025-05-28

#### 核心发现：R = -a·exp(H) + b

这是本领域最重要的理论发现之一：

$$R = -a \cdot e^H + b$$

其中 $R$ 是下游性能，$H$ 是策略熵，$a, b$ 是拟合系数。

> "This empirical law strongly indicates that policy performance is traded from policy entropy, thus bottlenecked by its exhaustion. The ceiling is fully predictable: $H=0 \Rightarrow R = -a + b$."

**含义**: 性能以熵消耗为代价，熵耗尽则性能触顶。

#### 核心定理：协方差驱动熵变化

对于 vanilla Policy Gradient，logit 差分为：

$$\Delta z_{s,a} = \eta \cdot \pi_\theta(a|s) \cdot A(s,a)$$

**熵变化公式（Theorem 1）**:

$$H(\pi^{k+1}_\theta|s) - H(\pi^k_\theta|s) \approx -\eta \cdot \text{Cov}_{a \sim \pi^k_\theta}[\log \pi^k_\theta(a|s), \pi^k_\theta(a|s) \cdot A(s,a)]$$

#### 关键洞见

> "A high-probability action with high advantage would **reduce** policy entropy, while a rare action with high advantage would **increase** policy entropy."

| 情况 | 熵变化 |
|------|--------|
| 高概率 + 高 advantage | 熵下降 |
| 低概率 + 高 advantage | 熵上升 |

经验上，协方差项始终为正 → 熵单调下降。

#### 解决方案：Clip-Cov 和 KL-Cov

- **Clip-Cov**: 随机选择一部分正协方差 token，detach 其梯度
- **KL-Cov**: 对最大协方差 token 施加 KL 惩罚

$$\mathcal{L}_{\text{KL-Cov}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})_{\text{high-cov tokens}}$$

#### 实验结果

| Model | Method | AIME24 | AIME25 |
|-------|--------|--------|--------|
| Qwen2.5-32B | GRPO | baseline | baseline |
| Qwen2.5-32B | **KL-Cov** | **+15.0%** | **+14.6%** |

已合并到 [verl 框架](https://verl.readthedocs.io/en/latest/algo/entropy.html)。

---

### 2025年6月：Beyond the 80/20 Rule — 高熵少数 Token 驱动有效 RL

**论文**: [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)
**机构**: Qwen/Alibaba、清华大学
**发表时间**: 2025-06-02
**会议**: NeurIPS 2025

#### 核心发现：20% 高熵 Token 决定一切

> "Only a small fraction of tokens exhibit high entropy, and these tokens act as **critical forks** that steer the model toward diverse reasoning pathways."

**Token 分布**:
- ~80% token: 低熵（完成句子、确定性元素）
- ~20% token: 高熵（逻辑连接词如 "however", "because", "thus"）

#### 核心实验

只用 top 20% 高熵 token 的梯度训练：

| Model | Full Gradient | **Top 20% Only** |
|-------|---------------|------------------|
| Qwen3-32B (AIME'25) | baseline | **+11.04** |
| Qwen3-32B (AIME'24) | baseline | **+7.71** |
| Qwen3-14B (AIME'25) | baseline | **+4.79** |

> "Training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance."

#### 为什么有效？

> "RL tends to preserve or increase the entropy of forking tokens, maintaining flexible reasoning paths. In contrast, SFT reduces token entropy, leading to memorization and poor generalization."

#### 实践配置

在 GRPO 中设置 `top_entropy_quantile = 0.2`。

---

### 2025年6月：The Surprising Effectiveness of Negative Reinforcement — 负样本的惊人效果

**论文**: [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347)
**发表时间**: 2025-06-02
**会议**: NeurIPS 2025

#### 核心概念：PSR vs NSR

将 RLVR 的学习信号分解为：

| 术语 | 含义 |
|------|------|
| **PSR** (Positive Sample Reinforcement) | 强化正确回答 |
| **NSR** (Negative Sample Reinforcement) | 惩罚错误回答 |

#### 惊人发现

> "Training with **only negative samples** — without reinforcing correct responses — can be highly effective: it consistently improves performance over the base model across the entire Pass@k spectrum."

#### NSR 如何工作？

> "NSR works by suppressing incorrect generations and **redistributing probability mass toward other plausible candidates**, guided by the model's prior beliefs. This effectively refines its existing knowledge without aggressively teaching new behaviors."

#### 核心公式：W-REINFORCE

$$\mathcal{L}_{\text{W-REINFORCE}}(\theta) = \lambda \cdot \mathcal{L}_{\text{PSR}}(\theta) + \mathcal{L}_{\text{NSR}}(\theta)$$

推荐 $\lambda = 0.1$（大幅下调正样本权重）。

#### 实验结论

| 方法 | Pass@1 | Pass@256 |
|------|--------|----------|
| PSR only | 最好 | 较差 |
| NSR only | 较差 | **接近最好** |
| **W-REINFORCE** | 好 | **最好** |

NSR 对熵的保持至关重要 → 大 k 时的多样性。

---

### 2025年6月：Rewarding the Unlikely — 修正 GRPO 的 Rank Bias

**论文**: [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355)
**机构**: CMU
**发表时间**: 2025-06-03
**会议**: EMNLP 2025

#### 核心问题：Rank Bias

> "A degenerate rank bias in GRPO in which **highly probable trajectories are reinforced and rare ones are neglected**. This results in distribution sharpening."

**后果**: 模型只学会用更少样本解决已经会的问题，但在 Pass@N（大 N）上反而不如直接从原模型多采样。

#### 解决方案：Unlikeliness Reward

> "Explicitly up-weighting rare but correct solutions."

对低概率但正确的解给予额外奖励。

#### 另一发现：PPO Epochs

> "Increasing the number of optimization steps per sample also mitigates rank bias. When taking multiple gradient steps, the initial steps may push high-rank solutions beyond the clipping threshold, so subsequent steps are forced to focus on low-rank samples."

#### 实验结果

在 miniF2F-test 定理证明 benchmark 上达到与 DeepSeek-Prover-V1.5-RL 相当的性能。

---

### 2025年10月：Rethinking Entropy Interventions — 从熵变化视角重新思考

**论文**: [Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective](https://arxiv.org/abs/2510.10150)
**机构**: 浙江大学
**发表时间**: 2025-10-11

#### 核心批评

> "Existing methods attempt to control entropy **indirectly** by only adjusting related factors such as the advantage signal and generation probability. Their effectiveness is inherently limited and prone to failure."

现有方法（如 Clip-Higher、advantage 重加权）都是间接控制熵，效果有限。

#### 核心方法：STEER

**STEER** = Stabilizing Token-level Entropy-changE via Reweighting

核心思路：分析**每个 token 的熵变化**，直接对熵变化进行 token 级别的细粒度调控。

> "The overall entropy dynamics during training arises from the accumulation of per-token entropy changes."

#### 与其他方法的区别

| 方法类型 | 问题 |
|----------|------|
| Clip 阈值解耦 | 对熵变化产生不对称、不可控的效果 |
| 正负样本重加权 | 间接影响，效果有限 |
| **STEER** | 直接在 token 级别稳定熵动态 |

---

### 2025年11月：Revisiting Entropy in RLVR — 正样本才是熵崩溃主因

**论文**: [Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2511.05993)
**发表时间**: 2025-11-08

#### 核心发现

> "Entropy collapse in RLVR **primarily arises from tokens with positive advantages**, and regulating their relative loss weights provides an effective means to control entropy."

**这与直觉相反**：不是负样本导致熵崩溃，而是正样本！

#### 理论解释

结合 Entropy Mechanism 论文的公式：

- 高概率 + 正 advantage → 熵大幅下降
- 低概率 + 正 advantage → 熵上升（但这种情况较少）

由于正确回答往往已经是高概率的，强化它们会进一步锐化分布。

#### 关键因素

实验识别出影响熵的三个关键因素：

1. **Off-policy 更新次数**: 越多越容易崩溃
2. **训练数据多样性**: 越低越容易崩溃
3. **Clip 阈值**: 设置不当会加速崩溃

#### 解决方案

> "Clipping settings and reweighting tokens with positive advantages, such as Progressive Advantage Reweighting."

---

### 2025年11月：EntroPIC — 用控制论稳定熵

**论文**: [EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control](https://arxiv.org/abs/2511.15248)
**机构**: 腾讯 AI Lab
**发表时间**: 2025-11-20

#### 核心思路

用 **PID 控制器**（工业控制理论）来稳定熵！

#### 核心公式

设目标熵为 $H_{\text{target}}$，当前熵为 $H(n)$，误差为：

$$e(n) = H_{\text{target}} - H(n)$$

**PI 控制信号**:

$$\alpha(n) = K_p \cdot e(n) + K_i \cdot \sum_{t=0}^{n} e(t)$$

其中：
- $K_p$: 比例系数（响应当前误差）
- $K_i$: 积分系数（响应累积误差）

根据 $\alpha(n)$ 动态调整正负样本的 loss 权重。

#### 代码实现

```python
# 积分项：累积熵误差
control_alpha_i = accumulate_entropy_error * K_i
# 比例项：当前熵误差
control_alpha_p = (entropy_loss - target_entropy) * K_p
# 总控制信号
control_alpha = control_alpha_i + control_alpha_p
```

#### 实验结果

> "Successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs. Validated through successful training on over 1M prompts."

适用于 on-policy 和 off-policy 训练。

---

### 2025年12月：SENT — 语义熵 + Token 熵双层控制

**论文**: [Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning](https://arxiv.org/abs/2512.04359)
**发表时间**: 2025-12-04

#### 核心框架

SENT = Semantic ENtropy with Token-level entropy optimization

**双层设计**:

| 层级 | 方法 | 作用 |
|------|------|------|
| **数据层** | 语义熵引导的课程学习 | 从易到难组织训练数据 |
| **算法层** | Token 级熵优化 | 对低熵 token 施加 KL 正则 |

#### 语义熵课程学习

> "Organizing training data from low to high semantic entropy guides progressive optimization from easier to more challenging tasks."

**原理**: 先在简单问题上建立推理能力，避免过早接触难题导致激进更新和熵崩溃。

#### Token 级优化

对低熵 token（关键决策点）施加更强约束：

$$\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \lambda \cdot \text{KL}_{\text{low-entropy tokens}} + \gamma \cdot \text{Cov-penalty}_{\text{high-cov tokens}}$$

#### 实验结果

在 6 个 benchmark、3 种规模模型（1.5B, 7B, 14B）上均优于其他熵控制方法。

---

## 统一分析

### 1. 熵崩溃的根本原因

从数学上，熵变化由**协方差**驱动：

$$\Delta H \propto -\text{Cov}[\log \pi(a|s), \pi(a|s) \cdot A(s,a)]$$

| 情况 | 熵变化 | 频率 |
|------|--------|------|
| 高概率 + 高 advantage | 大幅下降 | 高（正确回答通常高概率）|
| 低概率 + 高 advantage | 上升 | 低 |
| 任何 + 负 advantage | 相反效果 | - |

**结论**: 正样本是熵崩溃的主因，因为正确回答往往已是高概率。

### 2. Entropy-Performance Trade-off

$$R = -a \cdot e^H + b$$

这意味着：

- 熵是一种**可消耗资源**
- 性能提升以熵消耗为代价
- 熵耗尽时性能触顶（$H=0 \Rightarrow R_{\max} = -a + b$）

**实践意义**: 可以用这个公式**预测**训练的性能上限。

### 3. 缓解方法分类

| 类别 | 方法 | 代表论文 |
|------|------|----------|
| **裁剪策略** | Clip-Higher, 解耦 $\epsilon$ | DAPO |
| **协方差控制** | Clip-Cov, KL-Cov | Entropy Mechanism |
| **Token 筛选** | 只用高熵 token 梯度 | Beyond 80/20 |
| **样本重加权** | W-REINFORCE, Unlikeliness Reward | NSR, Rewarding Unlikely |
| **直接熵控制** | PID 控制器 | EntroPIC |
| **熵变化感知** | Token 级熵变化重加权 | STEER |
| **课程学习** | 语义熵排序数据 | SENT, SEED-GRPO |
| **负样本挖掘** | 从错误回答中提取正确步骤 | Unearthing Gems |

### 4. 正样本 vs 负样本的作用

| 样本类型 | 对熵的影响 | 对性能的影响 |
|----------|------------|--------------|
| **正样本** | 降低熵（锐化分布）| 提升 Pass@1 |
| **负样本** | 保持/提升熵（保持多样性）| 提升 Pass@k（大 k）|

**最佳实践**: W-REINFORCE 建议 $\lambda = 0.1$，即大幅下调正样本权重。

### 5. 高熵 Token 的特殊地位

只有 ~20% 的 token 是高熵的，但它们是：

- **推理分叉点**（如 "however", "because"）
- **决定推理路径多样性的关键**
- **RL 应该重点优化的对象**

> "RL preserves entropy of forking tokens → flexible reasoning. SFT reduces all entropy → memorization."

### 6. 数据领域的影响

| 数据类型 | 预训练曝光度 | 初始熵 | 熵下降速度 |
|----------|--------------|--------|------------|
| Math/Code | 高 | 较低 | 快 |
| 合成 Logic-game | 低 | 较高 | 慢 |

**建议**: 使用预训练阶段未见过的合成数据（如 [SynLogic](https://github.com/MiniMax-AI/SynLogic)）可以缓解熵崩溃。

---

## 实践建议

### 入门配置

1. 使用 **DAPO** 的 Clip-Higher（$\epsilon_{\text{high}} = 0.28$）
2. 设置 `top_entropy_quantile = 0.2` 只用高熵 token 梯度
3. 使用 **W-REINFORCE** 下调正样本权重（$\lambda = 0.1$）

### 进阶配置

1. 实现 **Clip-Cov** 或 **KL-Cov** 基于协方差控制更新
2. 使用 **EntroPIC** 的 PI 控制器动态调节
3. 采用 **SENT** 的语义熵课程学习

### 监控指标

- **策略熵**: 核心健康指标
- **R vs H 曲线**: 验证是否符合 $R = -a \cdot e^H + b$
- **Token 熵分布**: 高熵 token 的比例和位置

---

## 开放问题

1. **能否找到不导致熵下降的优化目标？** 当前所有方法都是缓解而非根治。

2. **熵高 vs 探索效率的权衡**: 熵高利于探索，但探索效率可能下降（需要更多 step 才见效）。

3. **跨领域泛化**: 大部分结论基于 Qwen2.5 + Math/Code，其他模型和领域是否适用？

---

## 参考文献

1. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (2025.03)
2. [SEED-GRPO: Semantic Entropy Enhanced GRPO](https://arxiv.org/abs/2505.12346) (2025.05)
3. [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation](https://arxiv.org/abs/2505.14403) (2025.05)
4. [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) (2025.05)
5. [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL](https://arxiv.org/abs/2506.01939) (2025.06, NeurIPS 2025)
6. [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347) (2025.06, NeurIPS 2025)
7. [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355) (2025.06, EMNLP 2025)
8. [Rethinking Entropy Interventions in RLVR](https://arxiv.org/abs/2510.10150) (2025.10)
9. [Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2511.05993) (2025.11)
10. [EntroPIC: Entropy Stabilization with Proportional-Integral Control](https://arxiv.org/abs/2511.15248) (2025.11)
11. [SENT: Semantic and Token Entropy for LLM Reasoning](https://arxiv.org/abs/2512.04359) (2025.12)

### 开源实现

- [verl (Clip-Cov, KL-Cov)](https://verl.readthedocs.io/en/latest/algo/entropy.html)
- [DAPO](https://dapo-sia.github.io/)
- [PRIME-RL/Entropy-Mechanism-of-RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)
- [TianHongZXY/RLVR-Decomposed (W-REINFORCE)](https://github.com/TianHongZXY/RLVR-Decomposed)
- [EntroPIC](https://github.com/yk7333/EntroPIC)
- [STEER](https://github.com/zz-haooo/STEER)
