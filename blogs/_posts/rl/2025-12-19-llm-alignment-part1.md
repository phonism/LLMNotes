---
layout: post
title: "RL 学习笔记（五）：RLHF 与 DPO"
date: 2025-12-19 07:00:00
author: Qi Lu
tags: [RL, LLM, RLHF, DPO, Alignment]
lang: zh
translation: /en/llm-alignment-part1/
series: rl
series_order: 5
---

本文是强化学习系列的第五篇，开始进入 LLM 与 RL 结合的领域。本篇介绍 LLM 对齐的 RL 建模、经典的 RLHF 三阶段方法，以及更简洁的 DPO 方法。

## 引言：从预训练到对齐

### 核心问题

大语言模型（LLM）通过海量文本预训练，获得了强大的语言理解和生成能力。但预训练目标（预测下一个 token）与人类期望的行为之间存在鸿沟：

> **预训练的 LLM 只学会了"像人类一样说话"，但没有学会"按人类期望行事"。**
>
> 如何让 LLM 不仅流利，还能有帮助、诚实、无害？

这就是 **LLM 对齐**（Alignment）问题。而强化学习正是解决这一问题的核心技术。

### 为什么需要 RL？

监督学习（SFT）可以让模型模仿高质量回复，但存在局限：

1. **分布受限**：只能学习训练集中出现的回复方式
2. **无法表达偏好**：难以区分"好"和"更好"
3. **无法探索**：不会尝试新的回答策略

强化学习提供了不同的视角：
- 将 LLM 生成过程建模为 MDP
- 用人类偏好定义奖励函数
- 通过最大化奖励来优化策略

<!-- tikz-source: rl-llm-training-pipeline
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % 预训练
    \node[box, fill=blue!20] (pt) at (0, 0) {预训练\\（Next Token Prediction）};

    % SFT
    \node[box, fill=green!20] (sft) at (5, 0) {监督微调 SFT\\（模仿高质量回复）};

    % RLHF
    \node[box, fill=orange!20] (rlhf) at (10, 0) {RL 对齐\\（优化人类偏好）};

    % 箭头
    \draw[arrow] (pt) -- node[above, font=\small, yshift=5pt] {语言能力} (sft);
    \draw[arrow] (sft) -- node[above, font=\small, yshift=5pt] {指令遵循} (rlhf);

    % 标注
    \node[font=\scriptsize, gray] at (0, -1) {会说话};
    \node[font=\scriptsize, gray] at (5, -1) {能回答问题};
    \node[font=\scriptsize, gray] at (10, -1) {按人类期望行事};
\end{tikzpicture}
-->
![LLM Training Pipeline]({{ site.baseurl }}/assets/figures/rl-llm-training-pipeline.svg)

## LLM 对齐的 RL 建模

### State/Action/Reward 定义

将 LLM 对齐问题建模为 RL 问题：

> **LLM 的 RL 建模**
> - **State** $s_t$：prompt $x$ + 已生成的 token 序列 $y_{<t} = (y_1, \ldots, y_{t-1})$
> - **Action** $a_t$：下一个 token $y_t$（词表大小 $\|\mathcal{V}\| \sim$ 100k）
> - **Policy** $\pi_\theta(a\|s)$：LLM 本身，$\pi_\theta(y_t \| x, y_{<t})$
> - **Trajectory** $\tau$：完整的生成序列 $y = (y_1, y_2, \ldots, y_T)$
> - **Reward** $r$：通常只在序列结束时给出

<!-- tikz-source: rl-llm-as-mdp
\begin{tikzpicture}[
    state/.style={draw, rounded corners, fill=blue!15, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    action/.style={circle, draw, fill=orange!20, minimum size=0.6cm, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % 状态序列
    \node[state] (s0) at (0, 0) {$x$ (prompt)};
    \node[state] (s1) at (3.5, 0) {$x, y_1$};
    \node[state] (s2) at (7, 0) {$x, y_1, y_2$};
    \node[font=\small] at (9.5, 0) {$\cdots$};
    \node[state] (sT) at (12, 0) {$x, y_{1:T}$};

    % 动作
    \node[action] (a1) at (1.75, 1) {$y_1$};
    \node[action] (a2) at (5.25, 1) {$y_2$};
    \node[action] (aT) at (10.5, 1) {$y_T$};

    % 奖励
    \node[font=\small, red] at (13.5, 0) {$r(x, y)$};

    % 连接
    \draw[arrow] (s0) -- (a1);
    \draw[arrow] (a1) -- (s1);
    \draw[arrow] (s1) -- (a2);
    \draw[arrow] (a2) -- (s2);
    \draw[arrow, dashed] (s2) -- (9, 0);
    \draw[arrow] (10, 0) -- (aT);
    \draw[arrow] (aT) -- (sT);
    \draw[arrow, red] (sT) -- (13.2, 0);

    % 标注
    \node[font=\scriptsize, gray] at (1.75, 1.6) {$\pi_\theta(y_1|x)$};
    \node[font=\scriptsize, gray] at (5.25, 1.6) {$\pi_\theta(y_2|x,y_1)$};
\end{tikzpicture}
-->
![LLM as MDP]({{ site.baseurl }}/assets/figures/rl-llm-as-mdp.svg)

LLM RL 的特点：
- **动作空间巨大**：词表通常有 10 万+ token
- **确定性状态转移**：下一状态 = 当前状态 + 新 token
- **Episode = 一次完整生成**：从 prompt 到 EOS
- **稀疏奖励**：只有序列结束时才有奖励信号

### 稀疏奖励问题

LLM 对齐的典型奖励结构：

$$r_t = \begin{cases} 0 & t < T \\ r_\phi(x, y) & t = T \text{（序列结束）} \end{cases}$$

稀疏奖励带来的挑战：
- **信用分配困难**：最终奖励如何归因到每个 token？
- **梯度信号弱**：大部分时刻没有学习信号
- **长序列尤其困难**：信号需要传播很远（数千 token）

解决稀疏奖励的两种思路：
1. **序列级方法**：把整个序列当作一个 bandit，用序列奖励直接更新（如 REINFORCE）
2. **过程奖励**：训练 PRM 提供中间步骤的奖励信号

## RLHF 三阶段

RLHF（Reinforcement Learning from Human Feedback）是 LLM 对齐的经典方法，由 OpenAI 在 InstructGPT 中系统化。

### RLHF 整体架构

<!-- tikz-source: rl-rlhf-architecture
\begin{tikzpicture}[scale=0.9, every node/.style={scale=0.9},
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1cm, align=center},
    data/.style={draw, rounded corners, fill=gray!15, minimum width=2cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Stage 1
    \begin{scope}[shift={(-5, 0)}]
        \node[box, fill=blue!20] (pt) at (0, 2) {预训练模型};
        \node[data] (sft_data) at (0, 0) {高质量对话\\数据};
        \node[box, fill=green!20] (sft) at (0, -2) {SFT 模型\\$\pi_{\text{ref}}$};

        \draw[arrow] (pt) -- (sft);
        \draw[arrow] (sft_data) -- (sft);

        \node[font=\bfseries] at (0, 3.5) {Stage 1: SFT};
    \end{scope}

    % Stage 2
    \begin{scope}[shift={(0, 0)}]
        \node[box, fill=green!15] (sft2) at (0, 2) {SFT 模型};
        \node[data] (pref_data) at (0, 0) {人类偏好数据\\$(x, y_w, y_l)$};
        \node[box, fill=orange!20] (rm) at (0, -2) {Reward Model\\$r_\phi(x, y)$};

        \draw[arrow] (sft2) -- (rm);
        \draw[arrow] (pref_data) -- (rm);

        \node[font=\bfseries] at (0, 3.5) {Stage 2: RM};
    \end{scope}

    % Stage 3
    \begin{scope}[shift={(5.5, 0)}]
        \node[box, fill=green!15] (ref) at (-1.8, 2) {$\pi_{\text{ref}}$};
        \node[box, fill=orange!15] (rm2) at (1.8, 2) {$r_\phi$};
        \node[box, fill=purple!20] (ppo) at (0, 0) {PPO 训练};
        \node[box, fill=red!20] (final) at (0, -2) {对齐模型\\$\pi_\theta$};

        \draw[arrow] (ref) -- (ppo);
        \draw[arrow] (rm2) -- (ppo);
        \draw[arrow] (ppo) -- (final);

        \node[font=\bfseries] at (0, 3.5) {Stage 3: PPO};
    \end{scope}

    % 连接箭头
    \draw[arrow, dashed, gray] (-3, -2) -- (-2, 2);
    \draw[arrow, dashed, gray] (2, -2) -- (4, 2);
\end{tikzpicture}
-->
![RLHF Architecture]({{ site.baseurl }}/assets/figures/rl-rlhf-architecture.svg)

### Stage 1: Supervised Fine-Tuning (SFT)

用高质量对话数据微调预训练模型：

$$L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[ \log \pi_\theta(y|x) \right] = -\mathbb{E} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t}) \right]$$

SFT 的作用：
- 让模型学会"指令遵循"的基本格式
- 提供 RL 的起点（参考模型 $\pi_{\text{ref}}$）
- 过滤预训练中的低质量模式

### Stage 2: Reward Model 训练

从人类偏好数据中学习 Reward Model。

> **偏好数据**：对于 prompt $x$，人类标注者比较两个回复，给出偏好：$y_w \succ y_l$（$y_w$ 优于 $y_l$）。

#### Bradley-Terry 模型

> **Bradley-Terry 模型**
>
> 假设人类偏好遵循 Bradley-Terry 模型——偏好概率由"能力差"决定：
>
> $$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + e^{-(r(x, y_w) - r(x, y_l))}}$$
>
> 其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数，$r(x, y)$ 是回复的"得分"。

Bradley-Terry 模型的直觉：
- 奖励差 = 0 时，偏好概率 = 0.5（无法区分）
- 奖励差越大，偏好概率越接近 1（更确定）
- 模型假设偏好是基于"内在质量分数"的概率比较

#### Reward Model 训练

Reward Model 的训练目标是最大化偏好数据的似然：

$$L_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

这是一个**二分类问题**：给定 $(y_w, y_l)$，预测哪个更好。

Reward Model 的架构选择：
- 通常用 SFT 模型初始化
- 去掉语言模型头，加上标量输出头
- 输入 $(x, y)$，输出标量 $r_\phi(x, y) \in \mathbb{R}$

### Stage 3: PPO 微调

使用 Reward Model 提供奖励信号，用 PPO 优化策略。

> **RLHF 优化目标**
>
> $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$
>
> 其中 $\beta > 0$ 是 KL 正则系数。

#### KL 正则的作用

KL 正则项 $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ 至关重要：

1. **防止 Reward Hacking**：
   - Reward Model 是不完美的代理
   - 无约束优化会找到"欺骗" RM 的方式
   - 例如：生成特定模式获得高分，但实际质量差

2. **保持生成质量**：
   - SFT 模型已经有较好的语言能力
   - KL 约束防止偏离太远导致流利度下降

3. **稳定训练**：
   - 约束优化空间，避免策略崩溃
   - 提供正则化效果

<!-- tikz-source: rl-kl-reward-tradeoff
\begin{tikzpicture}[
    arrow/.style={->, thick, >=stealth}
]
    % 坐标轴
    \draw[arrow] (-0.5, 0) -- (8, 0) node[right] {$\text{KL}(\pi_\theta \| \pi_{\text{ref}})$};
    \draw[arrow] (0, -0.5) -- (0, 5) node[above] {$\mathbb{E}[r_\phi]$};

    % 曲线
    \draw[thick, blue, domain=0.2:7, samples=100] plot (\x, {4 - 0.8*(\x-3)^2/9 + 0.5*ln(\x)});

    % 最优点
    \fill[red] (2.5, 3.8) circle (3pt);
    \node[font=\small, red] at (2.5, 4.3) {最优权衡};

    % 区域标注
    \node[font=\scriptsize, align=center] at (1, 2) {KL 太小\\改进有限};
    \node[font=\scriptsize, align=center] at (6, 2) {KL 太大\\Reward Hacking};

    % beta 的作用
    \draw[dashed, gray] (0, 3.8) -- (2.5, 3.8) -- (2.5, 0);
\end{tikzpicture}
-->
![KL-Reward Tradeoff]({{ site.baseurl }}/assets/figures/rl-kl-reward-tradeoff.svg)

#### PPO 更新流程

RLHF 中 PPO 的具体步骤：

```
输入: SFT 模型 π_ref，Reward Model r_φ，KL 系数 β
初始化 π_θ ← π_ref，Critic V_ψ

for each iteration:
    // 采样
    从 prompt 分布采样 x ∼ D
    用当前策略生成回复 y ∼ π_θ(·|x)

    // 计算奖励
    计算 RM 奖励：r^RM = r_φ(x, y)
    计算 KL 惩罚：r^KL_t = -β log [π_θ(y_t|x, y_{<t}) / π_ref(y_t|x, y_{<t})]
    总奖励：r_t = r^KL_t + 𝟙_{t=T} · r^RM

    // GAE 计算
    用 Critic V_ψ 计算 advantage Â_t

    // PPO 更新
    用 PPO-Clip 目标更新 π_θ
    用 TD 目标更新 V_ψ
```

> **重要提示**：RLHF 需要维护的模型：
> 1. $\pi_\theta$：正在训练的策略（Active Model）
> 2. $\pi_{\text{ref}}$：参考模型（冻结）
> 3. $r_\phi$：Reward Model（冻结）
> 4. $V_\psi$：Critic 网络
>
> 共 4 个大模型，显存开销巨大！这是 DPO、GRPO 等方法试图解决的问题。

## Direct Preference Optimization (DPO)

DPO 是一种绕过 Reward Model 和 PPO 的简化方法，由 Rafailov et al. 2023 提出。

### DPO 的动机

RLHF + PPO 的问题：
- **模型开销大**：需要维护 4 个模型
- **采样成本高**：大模型在线生成很贵
- **实现复杂**：PPO 超参敏感，需要精细调参
- **训练不稳定**：RL 训练容易崩溃

> **DPO 的核心问题**：能否直接在偏好数据 $(x, y_w, y_l)$ 上优化，像监督学习一样简单？

答案是可以的！关键洞察：KL 正则的 RL 问题有**闭式解**。

### DPO Loss 公式

> **DPO Loss**
>
> $$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right) \right]$$

### DPO 完整推导

> **DPO 等价性定理**：DPO Loss 与 RLHF 目标在最优解处等价。

**证明**：推导分为 5 个关键步骤。

**Step 1：RLHF 目标展开**

RLHF 优化目标：

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x, y) \right] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

展开 KL 散度：

$$= \mathbb{E}_{y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

**Step 2：引入配分函数 $Z(x)$**

为了让最优策略是合法的概率分布，定义配分函数：

$$Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$$

$Z(x)$ 是归一化常数，只依赖于 $x$（不依赖于被优化的策略）。

**Step 3：最优策略的闭式解**

KL 正则 RL 问题有闭式解：

> **KL 正则 RL 的最优策略引理**
>
> 目标 $\max_\pi \mathbb{E}_{y \sim \pi}[r(y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$ 的最优解为：
>
> $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$$

这是一个有约束优化问题（$\pi$ 需要是概率分布）。直觉：最优策略是参考策略按 $\exp(r/\beta)$ 重新加权。奖励越高，概率提升越多。

**Step 4：从最优策略反解 reward**

关键步骤：从最优策略反解 reward。

取对数：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) - \log Z(x) + \frac{r(x,y)}{\beta}$$

整理得：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

> **核心洞察**：reward 可以用策略的 log-ratio 表示！虽然有 $\log Z(x)$ 项，但它只依赖于 $x$，在 pairwise 比较中会消除。

**Step 5：代入 Bradley-Terry 模型，$Z(x)$ 消除**

将 reward 表达式代入 Bradley-Terry 模型：

$$\begin{align}
P(y_w \succ y_l) &= \sigma(r(x, y_w) - r(x, y_l)) \\
&= \sigma\left( \beta \left[ \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right)
\end{align}$$

$\beta \log Z(x)$ 项相消了！

最大化偏好数据的 log-likelihood，用 $\pi_\theta$ 代替 $\pi^*$，得到 DPO Loss。

<!-- tikz-source: rl-dpo-derivation
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=3.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    \node[box] (rlhf) at (0, 3) {RLHF 目标\\$\max \mathbb{E}[r] - \beta \cdot \text{KL}$};
    \node[box] (opt) at (0, 1) {最优策略闭式解\\$\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$};
    \node[box] (reward) at (0, -1) {反解 reward\\$r = \beta \log \frac{\pi^*}{\pi_{\text{ref}}} + \beta \log Z$};
    \node[box, fill=green!20] (dpo) at (0, -3) {DPO Loss\\$Z(x)$ 消除};

    \draw[arrow] (rlhf) -- node[right, font=\small] {KL-RL 闭式解} (opt);
    \draw[arrow] (opt) -- node[right, font=\small] {取对数} (reward);
    \draw[arrow] (reward) -- node[right, font=\small] {代入 BT 模型} (dpo);
\end{tikzpicture}
-->
![DPO Derivation]({{ site.baseurl }}/assets/figures/rl-dpo-derivation.svg)

> **DPO 的核心洞察**：
> 1. KL 正则 RL 问题有闭式解，最优策略是参考策略的指数重加权
> 2. 可以从最优策略反解隐式 reward
> 3. 配分函数 $Z(x)$ 在 pairwise 比较中消除——这是 DPO 能 work 的关键
> 4. 最终形式只需要计算 log-probability，像监督学习一样简单

### DPO 的直观理解

定义**隐式奖励**：

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

DPO Loss 可以写成：

$$L_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) \right]$$

直觉：
- $\hat{r}_\theta(x, y_w) > \hat{r}_\theta(x, y_l)$：$y_w$ 的隐式奖励更高，loss 变小
- 训练过程提高 $y_w$ 相对于 $\pi_{\text{ref}}$ 的概率，压低 $y_l$ 的概率
- $\beta$ 控制"相对于参考策略偏离多少"的尺度

### DPO vs RLHF 对比

| 特性 | RLHF + PPO | DPO |
|------|------------|-----|
| 需要 Reward Model | 是 | 否 |
| 需要 Critic 网络 | 是 | 否 |
| 训练方式 | 在线采样 | 离线训练 |
| 模型数量 | 4 个 | 2 个 |
| 实现复杂度 | 高 | 低 |
| 超参敏感性 | 高 | 低 |
| 探索能力 | 有 | 无 |
| 适用场景 | 复杂任务 | 简单对齐 |

DPO 的局限：
- **无探索**：完全离线，只能在已有偏好数据的分布内优化
- **Pairwise 信号粗糙**：只知道谁更好，不知道好多少
- **难任务提升有限**：在数学、代码等需要探索的任务上效果不如 RL

## 本章小结

1. **LLM 对齐的 RL 建模**：State = prompt + 已生成 tokens，Action = 下一个 token，稀疏奖励只在序列结束时给出

2. **RLHF 三阶段**：
   - Stage 1 (SFT)：监督微调，学习指令遵循
   - Stage 2 (RM)：从偏好数据训练 Reward Model（Bradley-Terry 模型）
   - Stage 3 (PPO)：用 RM 提供奖励，PPO 优化，KL 正则防止 reward hacking

3. **DPO**：
   - 利用 KL-RL 闭式解，绕过 RM 和 PPO
   - 直接在偏好数据上优化，像监督学习一样简单
   - 只需 2 个模型（$\pi_\theta$ 和 $\pi_{\text{ref}}$）
   - 局限：无探索能力，难任务提升有限

<!-- tikz-source: rl-rlhf-vs-dpo
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3.5cm, minimum height=2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (rlhf) at (0, 0) {
        \textbf{RLHF}\\[3pt]
        (2020-2022)\\[3pt]
        \scriptsize 需要 RM + Critic\\[-1pt]
        \scriptsize 实现复杂\\[-1pt]
        \scriptsize 4 个模型
    };

    \node[box, fill=green!20] (dpo) at (6, 0) {
        \textbf{DPO}\\[3pt]
        (2023)\\[3pt]
        \scriptsize 离线训练\\[-1pt]
        \scriptsize 无探索能力\\[-1pt]
        \scriptsize 2 个模型
    };

    \draw[arrow] (rlhf) -- node[above, font=\small] {简化} (dpo);
\end{tikzpicture}
-->
![RLHF vs DPO 对比]({{ site.baseurl }}/assets/figures/rl-rlhf-vs-dpo.svg)

下一篇将介绍 GRPO、KL 估计器、PRM 以及 Long CoT RL 等更先进的方法，这些方法试图在保持 DPO 简洁性的同时恢复在线探索能力。
