---
layout: post
title: "LLM 与 RL (下)：GRPO 与 Long CoT RL"
date: 2025-12-19
author: Phonism
tags: [Reinforcement Learning, LLM, GRPO, PRM, Long CoT, Alignment]
---

本文是强化学习系列的第六篇，也是最后一篇。本篇介绍 GRPO（无 Critic 的在线 RL）、KL 散度估计器、On-Policy Distillation、过程奖励模型 PRM，以及 Long CoT RL 的挑战与方法。

## GRPO：Group Relative Policy Optimization

GRPO 是 DeepSeek 提出的方法，介于 PPO 和 DPO 之间：保留在线探索能力，但不需要 Critic 网络。

### GRPO 的动机

- **PPO 的问题**：需要 Critic 网络，显存开销大（额外一个大模型）
- **DPO 的问题**：完全离线，缺乏探索，难任务提升有限

GRPO 的思路：**用组内相对奖励代替 Critic**，实现"无 Critic 的在线 RL"。

### 组内标准化 Advantage

> **GRPO Advantage**
>
> 对于 prompt $x$，采样一组回复 $\\{y_1, \ldots, y_G\\}$，计算各自的奖励 $\\{R_1, \ldots, R_G\\}$，然后：
>
> $$\hat{A}_i = \frac{R_i - \bar{R}}{\text{Std}(R) + \epsilon}$$
>
> 其中 $\bar{R} = \frac{1}{G}\sum_i R_i$ 是组内均值，$\text{Std}(R)$ 是组内标准差。

```
                              ┌─────┐
          ┌───────────────────│ y₁  │───▶ R₁ = 0.8 ───▶ Â₁ > 0 ✓
          │                   └─────┘
          │                   ┌─────┐
┌─────────┴─────────┐         │ y₂  │───▶ R₂ = 0.6 ───▶ Â₂ > 0 ✓
│    Prompt x       │─────────└─────┘
│                   │         ┌─────┐
└─────────┬─────────┘         │ y₃  │───▶ R₃ = 0.3 ───▶ Â₃ < 0 ✗
          │                   └─────┘
          │                   ┌─────┐
          └───────────────────│ y₄  │───▶ R₄ = 0.1 ───▶ Â₄ < 0 ✗
                              └─────┘
                                         R̄ = 0.45

         组内相对比较：高于均值的增强，低于均值的抑制
```

组内标准化的优势：
1. **无需 Critic**：用组内均值代替价值函数估计
2. **Baseline 效果**：均值减法降低方差
3. **尺度归一化**：标准差归一化使 advantage 尺度稳定
4. **相对比较**：focus 在同一 prompt 下的相对好坏

### GRPO 目标函数

> **GRPO 目标**
>
> $$L_{\text{GRPO}} = \mathbb{E}_{x} \left[ \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min \left( \rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$
>
> 其中 $\rho_{i,t} = \frac{\pi_\theta(y_{i,t}\|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\|x, y_{i,<t})}$ 是 importance sampling 比率。

GRPO 与 PPO 的关键区别：
- PPO：$\hat{A}_t$ 由 GAE 计算，需要 Critic
- GRPO：$\hat{A}_i$ 对整个序列恒定，由组内标准化得到

### GRPO 实用技巧

1. **Clip-Higher**：上界可以更宽松（如 $1+0.28$ 而非 $1+0.2$），允许好回复更大幅度提升

2. **Dynamic Sampling**：过滤全对或全错的 prompt（方差为 0 时 advantage 无定义）

3. **长度惩罚**：防止生成过长的回复
   $$R_i = r_\phi(x, y_i) - \lambda \cdot |y_i|$$

4. **KL as Loss**：KL 惩罚作为独立 loss 项，而非放入 reward
   $$L = -L_{\text{GRPO}} + \lambda_{\text{KL}} \cdot \mathbb{E}[\text{KL}(\pi_\theta \| \pi_{\text{ref}})]$$

## KL 散度估计：k1, k2, k3

KL 散度 $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ 无法精确计算（需要遍历所有可能的序列），需要蒙特卡洛估计。

### KL 散度的定义

对于两个分布 $p$ 和 $q$：

$$\text{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[ \log \frac{p(x)}{q(x)} \right]$$

在 LLM 场景中，$p = \pi_\theta$，$q = \pi_{\text{ref}}$，从 $\pi_\theta$ 采样。

### k1 估计器：直接估计

> **k1 估计器**
>
> 定义比率 $r = \frac{\pi_{\text{ref}}(y\|x)}{\pi_\theta(y\|x)}$，则：
>
> $$k_1 = -\log r = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

性质：
- **无偏**：$\mathbb{E}_{y \sim \pi_\theta}[k_1] = \text{KL}(\pi_\theta \| \pi_{\text{ref}})$
- **高方差**：当 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 差异大时，方差很大

用法：通常放入 reward
$$r_t^{\text{RL}} = r_t^{\text{RM}} - \beta \cdot k_1^{(t)}$$

### k2 估计器：平方形式

> **k2 估计器**
>
> $$k_2 = \frac{1}{2}(\log r)^2 = \frac{1}{2} \left( \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)} \right)^2$$

性质：
- **有偏**：$\mathbb{E}[k_2] \neq \text{KL}$
- **梯度等价**：$\nabla_\theta \mathbb{E}[k_2] = \nabla_\theta \text{KL}$
- **更平滑**：平方形式在 $r=1$ 附近更平滑

用法：适合作为 loss 项（因为梯度正确）

### k3 估计器：Control Variate

> **k3 估计器**
>
> $$k_3 = (r - 1) - \log r = \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)} - 1 - \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)}$$

> **k3 的性质定理**：k3 是无偏且低方差的 KL 估计器。

**证明**：

无偏性：
$$\begin{align}
\mathbb{E}_{y \sim \pi_\theta}[k_3] &= \mathbb{E}[(r-1)] - \mathbb{E}[\log r] \\
&= \underbrace{\mathbb{E}\left[\frac{\pi_{\text{ref}}}{\pi_\theta}\right] - 1}_{= 0} + \mathbb{E}\left[\log \frac{\pi_\theta}{\pi_{\text{ref}}}\right] \\
&= \text{KL}(\pi_\theta \| \pi_{\text{ref}})
\end{align}$$

其中 $\mathbb{E}_{y \sim \pi_\theta}\left[\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}\right] = \sum_y \pi_\theta(y) \cdot \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} = \sum_y \pi_{\text{ref}}(y) = 1$。

低方差：$(r-1)$ 项是一个 control variate，期望为 0，与 $\log r$ 负相关，减少方差。

### 三种估计器对比

| 估计器 | 公式 | 偏差 | 方差 | 推荐用法 |
|--------|------|------|------|----------|
| k1 | $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | 无偏 | 高 | KL in reward |
| k2 | $\frac{1}{2}(\log \frac{\pi_{\text{ref}}}{\pi_\theta})^2$ | 有偏 | 低 | KL as loss |
| k3 | $\frac{\pi_{\text{ref}}}{\pi_\theta} - 1 - \log \frac{\pi_{\text{ref}}}{\pi_\theta}$ | 无偏 | 低 | KL as loss |

**KL 的两种用法**：

1. **KL in Reward**（经典 RLHF）：
   - Token reward 减去 $\beta \cdot k_1$
   - 然后用 PPO-Clip 更新
   - 优点：直接影响每步决策

2. **KL as Loss**（GRPO 等）：
   - 总 loss = $-L_{\text{RL}} + \lambda \cdot \mathbb{E}[k_3]$
   - 优点：更稳定，方差更低

## On-Policy Distillation

On-Policy Distillation 是近年来 LLM 后训练的重要进展，结合了 RL 与知识蒸馏的优势。

### 动机：Off-policy 蒸馏的分布偏移问题

传统的知识蒸馏（SFT on 教师数据）是 **off-policy** 的：

> **学生从教师生成的数据学习，但推理时走到的状态可能与训练时完全不同。**

这导致 **compounding errors**——学生在训练时没见过自己犯的错误，一旦偏离教师轨迹就会持续恶化。

### 三种后训练范式对比

| 方法 | 采样来源 | 奖励密度 | 特点 |
|------|----------|----------|------|
| SFT（监督微调） | 教师数据（Off-policy） | 密集 | 分布受限 |
| RL（强化学习） | 自身采样（On-policy） | 稀疏 | 搜索成本高 |
| On-Policy Distillation | **自身采样** | **密集** | 两者优势结合 |

On-Policy Distillation 的核心思想：
- **学生自己采样**（On-policy）：避免分布偏移
- **教师逐 token 打分**（Dense reward）：提供密集监督信号

### Reverse KL 损失

On-Policy Distillation 使用**反向 KL 散度**作为损失函数：

$$L_{\text{OPD}} = D_{\text{KL}}(\pi_\theta \| \pi_{\text{teacher}}) = \mathbb{E}_{y \sim \pi_\theta}\left[ \sum_t \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{teacher}}(y_t | x, y_{<t})} \right]$$

**Forward KL vs Reverse KL**：
- Forward KL $D_{\text{KL}}(\pi_{\text{teacher}} \| \pi_\theta)$：mode covering，学生试图覆盖教师的所有模式
- Reverse KL $D_{\text{KL}}(\pi_\theta \| \pi_{\text{teacher}})$：mode seeking，学生专注于教师的高概率区域

Reverse KL 更适合蒸馏场景——让学生在自己访问的状态上模仿教师，而非试图覆盖教师的全部行为。

### KDRL：结合知识蒸馏与强化学习

KDRL（Knowledge Distillation + Reinforcement Learning）将教师监督和奖励驱动的自我探索统一为联合优化：

$$\mathcal{J}_{\text{KDRL}}(\theta) = \underbrace{\mathcal{J}_{\text{GRPO}}(\theta)}_{\text{奖励驱动}} - \beta \cdot \underbrace{D_{\text{KL}}^{k_2}(\pi_\theta \| \pi_{\text{teacher}})}_{\text{教师监督}}$$

关键设计选择：
1. **KL 估计器选择**：$k_2$ 优于 $k_3$，因为 $k_2$ 提供无偏梯度估计
2. **系数退火**：从强模仿（$\beta$ 大）逐渐过渡到奖励驱动（$\beta$ 小）
3. **奖励引导 Masking**：只在低奖励样本上应用 KD，减少输出长度同时保持准确率

### 效率优势

On-Policy Distillation 的效率提升（以数学推理任务为例）：
- 相比 RL：训练速度快 7-10 倍，总计算效率提升 50-100 倍
- 相比离线蒸馏：计算成本降低 9-30 倍
- Qwen3 实验：蒸馏显著优于 RL，GPU 小时仅需 RL 的约 1/10

效率提升的根本原因：

> **教师提供的密集 token 级信号，比 RL 的稀疏序列级奖励包含更多信息。**
>
> RL 的大部分计算花在搜索（探索策略空间），蒸馏则直接利用教师的知识指导学生在关键"分叉点"做出正确选择。

## Process Reward Model (PRM)

PRM 提供过程级监督，将稀疏的终局奖励变成密集的步级奖励。

### ORM vs PRM

> **ORM 与 PRM**
> - **ORM (Outcome Reward Model)**：只看最终结果
>   - 输入：$(x, y)$
>   - 输出：最终答案的正确性分数
>
> - **PRM (Process Reward Model)**：评估每个中间步骤
>   - 输入：$(x, y_{\leq t})$
>   - 输出：到第 $t$ 步为止的正确性分数

```
ORM：只评估最终答案
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Step 1 │───▶│ Step 2 │───▶│ Step 3 │───▶│ Answer │
└────────┘    └────────┘    └────────┘    └────┬───┘
                                               │
                                            r = 1

PRM：评估每个步骤
┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
│ Step 1 │───▶│ Step 2 │───▶│ Step 3 │───▶│ Answer │
└────┬───┘    └────┬───┘    └────┬───┘    └────┬───┘
     │             │             │             │
  r₁ = 1 ✓     r₂ = 1 ✓     r₃ = 0 ✗      r₄ = 0 ✗
```

### PRM 的优势

1. **信用分配更清晰**：每步都有反馈，知道哪一步出错
2. **长链推理收敛更快**：dense reward 比 sparse reward 更容易学习
3. **可以做早停**：如果中间步骤得分持续下降，可以截断重采样
4. **支持 Best-of-N**：用累积 PRM 分数选择最佳推理路径

### PRM 训练

PRM 的训练数据来源：
- **人工标注**：专家标注每个推理步骤的正确性
- **自动标注**：用最终答案正确性反推中间步骤
- **MCTS 探索**：搜索发现正确/错误的分支点

PRM 用于 RL 的奖励：

$$r_t = \text{PRM}(x, y_{\leq t}) - \text{PRM}(x, y_{\leq t-1})$$

即每步的"边际贡献"。

## Long CoT RL

长序列 Chain-of-Thought（Long CoT）的 RL 训练面临独特挑战。随着 o1、DeepSeek-R1 等推理模型的出现，这成为重要的研究方向。

### 长序列 RL 的挑战

1. **方差爆炸**：Token 级 importance sampling 权重累积后方差指数增长
   $$\prod_{t=1}^{T} \frac{\pi_\theta(y_t|s_t)}{\pi_{\text{old}}(y_t|s_t)} \approx e^{\sum_t \delta_t}$$
   当 $T$ 很大时（数千 token），这个乘积的方差会爆炸。

2. **策略偏移**：长序列使 $\pi_\theta$ 和 $\pi_{\text{old}}$ 偏离更大

3. **稀疏奖励更难**：只有最终答案有反馈，信号传播数千步

```
        IS 权重方差
        (对数尺度)
           ▲
           │                   ╱
           │                 ╱   Token-level IS
           │               ╱     （指数增长）
           │             ╱
           │           ╱
           │         ╱
           │       ╱
           │     ╱ ─ ─ ─ ─ ─ ─ ─  Sequence-level IS
           │   ╱                  （线性增长）
           │ ╱
           └─────────────────────────────▶ 序列长度 T
```

### GSPO：序列级 IS

GSPO（Group Sequence Policy Optimization）使用序列级重要性采样：

> **GSPO 序列级 IS**
>
> 定义长度归一化的序列级 IS 权重：
>
> $$w_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \right)^{1/|y_i|} = \exp\left( \frac{1}{|y_i|} \sum_t \log \frac{\pi_\theta(y_{i,t}|s_{i,t})}{\pi_{\text{old}}(y_{i,t}|s_{i,t})} \right)$$
>
> 长度归一化后再 clip，所有 token 共用同一个权重。

GSPO 的优势：
- 避免 token 级权重累乘的方差爆炸
- 长度归一化使不同长度序列可比
- 保持 PPO-Clip 的稳定性

### CISPO：Clipped IS-weight

CISPO（Clipped Importance Sampling Policy Optimization）采用另一种策略：

- 保持 GRPO 的组内标准化
- 回到 token 级 REINFORCE
- Clip IS 权重（而非 loss）：

$$\hat{\rho}_{i,t} = \text{clip}\left(\frac{\pi_\theta(y_{i,t}|s_{i,t})}{\pi_{\text{old}}(y_{i,t}|s_{i,t})}, 1-\epsilon, 1+\epsilon\right)$$

### Kimi k1.5 技术要点

Kimi k1.5 的长 CoT RL 配方：
- **超长上下文**：128k context 直接 RL
- **Mirror Descent 更新**：更保守的策略更新
- **部分 Rollout**：不需要完整生成，截断后用价值估计
- **异步 Train/Infer**：分离训练和推理以提高效率
- **重复检测 + 早停**：检测到循环生成就截断
- **长度惩罚**：鼓励简洁的推理

**Long2Short RL**（长到短蒸馏）：
- 长 CoT 模型作 teacher
- 训练短 CoT student
- 奖励 = 正确性 + token 数惩罚
- 目标：又对又短

### DeepSeek-V3.2 改进

DeepSeek-V3.2 对 GRPO 的改进（核心思想：让 off-policy 训练尽量接近 on-policy 的行为）：

1. **Unbiased KL Estimate**：用 IS 比率修正 k3 的 off-policy 偏差
   $$\hat{\text{KL}} = \mathbb{E}_{\pi_{\text{old}}}\left[ \frac{\pi_\theta}{\pi_{\text{old}}} \cdot k_3 \right]$$

2. **Off-Policy Sequence Masking**：负 advantage 且 KL 偏离过大的序列被 mask

3. **Keep Routing**（MoE 专用）：保持采样时的 expert routing 路径

4. **Keep Sampling Mask**：保持 top-p/top-k 的 truncation mask

## Token-level vs Sequence-level 目标

### 一阶近似理论

Token-level 目标（如 REINFORCE、GRPO）是 sequence-level 目标的一阶近似。

设 $\frac{\pi_\theta(y_t\|s_t)}{\pi_{\theta_{\text{old}}}(y_t\|s_t)} = 1 + \delta_t$，其中 $\delta_t$ 是小量，则：

$$\prod_t (1 + \delta_t) \approx 1 + \sum_t \delta_t \quad \text{（一阶 Taylor 展开）}$$

因此，当策略变化较小时：

$$\nabla \mathcal{J}^{\text{seq}} \approx \nabla \mathcal{J}^{\text{token}}$$

成立条件：$\pi_\theta \approx \pi_{\theta_{\text{old}}}$，即每次更新步长足够小。

### Training-Inference Discrepancy

实际中，采样分布 $\mu$ 可能与训练时的 $\pi_{\theta_{\text{old}}}$ 不同：

$$\frac{\pi_\theta(y)}{\mu(y)} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y)}{\mu(y)}}_{\text{train-infer gap}} \times \underbrace{\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)}}_{\text{policy staleness}}$$

两个因素都会导致一阶近似失效：
- **Train-infer gap**：训练和推理的采样策略不同（如 temperature、top-p）
- **Policy staleness**：异步训练中，数据来自旧策略

## 本章总结

1. **GRPO**：用组内相对奖励代替 Critic，实现无 Critic 的在线 RL
   - 组内标准化：$\hat{A}_i = \frac{R_i - \bar{R}}{\text{Std}(R)}$
   - 保留探索能力，显存开销低于 PPO

2. **KL 估计器**：
   - k1（无偏高方差）：适合 KL in reward
   - k2（有偏低方差）：适合 KL as loss
   - k3（无偏低方差）：推荐用于 KL as loss

3. **On-Policy Distillation**：
   - 结合 RL 探索和蒸馏效率
   - Reverse KL 让学生在自己的分布上学习
   - 效率比纯 RL 高 10-100 倍

4. **PRM**：过程奖励模型，提供 dense reward 信号
   - 每步评估，解决稀疏奖励的信用分配问题
   - 支持 Best-of-N 选择和早停

5. **Long CoT RL**：
   - GSPO/CISPO 解决长序列方差爆炸问题
   - 序列级 IS 代替 token 级 IS
   - Kimi、DeepSeek 等实践技巧

```
           LLM 对齐 RL 方法演进

┌─────────────┐   简化   ┌─────────────┐   在线探索  ┌─────────────┐  长序列  ┌─────────────┐
│    RLHF     │────────▶│     DPO     │──────────▶│    GRPO     │────────▶│ Long CoT RL │
│ (2020-2022) │         │   (2023)    │           │   (2024)    │         │ (2024-2025) │
├─────────────┤         ├─────────────┤           ├─────────────┤         ├─────────────┤
│需要 RM+Critic│         │  离线训练   │           │  无 Critic   │         │  序列级 IS   │
│  实现复杂    │         │   无探索    │           │  组内标准化  │         │  方差控制   │
└─────────────┘         └─────────────┘           └─────────────┘         └─────────────┘
```

## 系列总结

本系列共六篇文章，系统介绍了强化学习从基础到 LLM 应用的完整知识体系：

1. **RL 基础**：MDP、轨迹、价值函数、RL 目标
2. **Value-Based RL**：Bellman 方程、Q-Learning、DQN
3. **Policy-Based RL**：Policy Gradient、REINFORCE、Actor-Critic、PPO
4. **Model-Based RL & MARL**：World Model、MCTS、AlphaZero、Self-Play
5. **LLM 对齐 (上)**：RLHF、Bradley-Terry、DPO
6. **LLM 对齐 (下)**：GRPO、KL 估计器、PRM、Long CoT RL

希望这个系列能帮助读者建立起 RL 的完整知识框架，并理解其在现代 LLM 对齐中的核心作用。
