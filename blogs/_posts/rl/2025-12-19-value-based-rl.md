---
layout: post
title: "RL 学习笔记（二）：Bellman 方程与 DQN"
date: 2025-12-19 04:00:00
author: Phonism
tags: [RL, Bellman, Q-Learning, DQN, TD, MC]
lang: zh
translation: /en/value-based-rl/
---

在上一篇文章中，我们定义了价值函数 $V^\pi(s)$ 和 $Q^\pi(s,a)$，它们衡量了给定策略 $\pi$ 下状态或状态-动作对的"好坏"。现在自然的问题是：

> **如何计算价值函数？如何从价值函数导出最优策略？**

这两个问题的答案构成了 Value-Based RL 的核心。本文将介绍三类方法：
1. **动态规划**（Dynamic Programming）：当环境模型已知时，精确求解
2. **无模型估计**（MC 与 TD）：当环境模型未知时，从采样中学习
3. **深度 Q 网络**（DQN）：处理大规模或连续状态空间

## 1. 为什么要学习价值函数？

一个核心观察是：**如果我们知道最优动作价值函数 $Q^*(s,a)$，就能直接导出最优策略**：

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

这是 Value-Based 方法的理论基础——不直接学习策略，而是通过学习价值函数间接得到策略。

<div class="mermaid">
graph TB
    VBM["Value-Based Methods<br/>DP, MC, TD<br/>Q-Learning, DQN"] --> VF["Value Function<br/>V* or Q*"]
    VF -->|"argmax"| OP["Optimal Policy<br/>π*"]
</div>

## 2. Bellman 方程

Bellman 方程是 RL 的核心方程，它揭示了价值函数的递推结构。理解 Bellman 方程是掌握所有 Value-Based 方法的基础。

### 2.1 直觉：一步分解

Bellman 方程的核心形式是：

$$V^\pi(s) = \mathbb{E} \left[ r + \gamma V^\pi(s') \right]$$

用文字说：**当前状态的价值 = 即时奖励 + 折扣后的下一状态价值**。

这个递推关系使得我们可以通过迭代来计算价值函数，而不需要枚举所有可能的轨迹。

### 2.2 Bellman Expectation Equation

**定理 (Bellman Expectation Equation for $V^\pi$)**：对于任意策略 $\pi$，状态价值函数满足：

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^\pi(s') \right]$$

**证明**：从状态价值函数的定义出发：

$$\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] \\
&= \mathbb{E}_\pi \left[ r_t + \gamma G_{t+1} \mid S_t = s \right] \\
&= \mathbb{E}_\pi \left[ r_t + \gamma V^\pi(S_{t+1}) \mid S_t = s \right]
\end{aligned}$$

展开期望，由于给定 $S_t = s$：动作 $A_t = a$ 的概率为 $\pi(a\|s)$，给定 $(s, a)$，下一状态 $S_{t+1} = s'$ 的概率为 $P(s'\|s,a)$。

类似地，动作价值函数的 Bellman Expectation Equation：

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s',a')$$

> **$V^\pi$ 与 $Q^\pi$ 的相互表示是推导 Bellman 方程的关键**：
> $$V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s,a)$$
> $$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

### 2.3 Bellman Optimality Equation

最优价值函数满足 Bellman Optimality Equation。与 Expectation 版本不同，这里对动作求**最大值**而非期望。

**定理 (Bellman Optimality Equation for $V^*$)**：

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^*(s') \right]$$

**定理 (Bellman Optimality Equation for $Q^*$)**：

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \max_{a' \in \mathcal{A}} Q^*(s',a')$$

> **Bellman Expectation vs. Bellman Optimality 的关键区别**：
> - **Bellman Expectation**：对动作求加权平均（按 $\pi(a\|s)$），描述**给定策略**的价值
> - **Bellman Optimality**：对动作求最大值（$\max_a$），描述**最优策略**的价值
>
> **Q-Learning 直接逼近 $Q^*$，因此使用 Bellman Optimality Equation 作为更新目标**。

### 2.4 Bellman 算子与收缩性质

**定义 (Bellman 算子)**：

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

$$(\mathcal{T}^* V)(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

**定理 (Bellman 算子的收缩性)**：Bellman 算子是 $\gamma$-收缩映射：

$$\| \mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2 \|_\infty \leq \gamma \| V_1 - V_2 \|_\infty$$

> **收缩性质的重要推论**：
> 1. **唯一不动点**：Bellman 算子有唯一的不动点 $V^\pi$（或 $V^*$）
> 2. **迭代收敛**：从任意初始 $V_0$ 出发，$V_{k+1} = \mathcal{T}V_k$ 收敛到不动点
> 3. **收敛速度**：误差以 $\gamma^k$ 的速度指数衰减

## 3. 动态规划方法

当环境模型 $P(s'\|s,a)$ 和 $R(s,a)$ 完全已知时，可以使用动态规划（Dynamic Programming, DP）方法精确求解最优策略。

### 3.1 Policy Evaluation（策略评估）

给定策略 $\pi$，通过迭代 Bellman Expectation 方程来计算 $V^\pi$：

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

从任意初始 $V_0$ 开始，迭代直到收敛 $V_k \to V^\pi$。

### 3.2 Policy Improvement（策略改进）

**定理 (Policy Improvement Theorem)**：设 $\pi$ 和 $\pi'$ 是两个策略。如果对于所有 $s \in \mathcal{S}$：

$$Q^\pi(s, \pi'(s)) \geq V^\pi(s)$$

则 $\pi'$ 至少与 $\pi$ 一样好：对于所有 $s$，$V^{\pi'}(s) \geq V^\pi(s)$。

**贪心策略改进**：

$$\pi'(s) = \arg\max_a Q^\pi(s,a) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

### 3.3 Policy Iteration

交替进行 Policy Evaluation 和 Policy Improvement：

<div class="mermaid">
graph LR
    P0["π₀"] -->|"Eval"| V0["V^π₀"]
    V0 -->|"Improve"| P1["π₁"]
    P1 -->|"Eval"| V1["V^π₁"]
    V1 -->|"Improve"| P2["π₂"]
    P2 -->|"..."| END["π*"]
</div>

对于有限 MDP，Policy Iteration 在有限步内收敛到最优策略 $\pi^*$。

### 3.4 Value Iteration

将 Policy Evaluation 和 Policy Improvement 合并为一步，直接迭代 Bellman Optimality 方程：

$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

收敛后，$\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V(s') \right]$。

| | Policy Iteration | Value Iteration |
|---|------------------|-----------------|
| Policy Evaluation | 完整求解 $V^\pi$ | 仅一步 Bellman 更新 |
| 理论依据 | Bellman Expectation | Bellman Optimality |
| 每轮复杂度 | 高（需多次内循环） | 低（单次遍历） |
| 收敛轮数 | 少 | 多 |

### 3.5 DP 方法的局限性

1. **需要完整模型**：必须知道 $P(s'\|s,a)$ 和 $R(s,a)$
2. **状态空间遍历**：每轮需要遍历所有状态，复杂度 $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$
3. **无法处理连续空间**：表格方法无法直接应用

这些局限促使了无模型方法（MC、TD）和函数逼近方法（DQN）的发展。

## 4. 无模型方法：Monte Carlo vs Temporal Difference

当环境模型未知时，需要通过与环境交互的样本来估计价值函数。

### 4.1 Monte Carlo 估计

MC 方法的核心思想：**用实际轨迹的回报 $G_t$ 来估计期望**。

**Monte Carlo 更新**：

$$V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

其中 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$ 是从 $t$ 时刻开始到 episode 结束的实际回报。

MC 方法的特点：
- **必须等到 episode 结束**才能计算 $G_t$，仅适用于 episodic 任务
- **无偏估计**：$\mathbb{E}[G_t \| S_t = s] = V^\pi(s)$
- **高方差**：$G_t$ 累积了整条轨迹的随机性
- **不使用 Bootstrap**：目标完全来自真实采样

### 4.2 Temporal Difference 估计

TD 方法的核心思想：**用"一步奖励 + 下一状态的估计价值"来代替完整回报**。

**TD(0) 更新**：

$$V(S_t) \leftarrow V(S_t) + \alpha \left( \underbrace{r_t + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t) \right)$$

TD 误差：$\delta_t = r_t + \gamma V(S_{t+1}) - V(S_t)$

TD 方法的特点：
- **每步都可更新**，适用于 continuing 任务和在线学习
- **有偏估计**：使用了 $V(S_{t+1})$ 的估计值
- **低方差**：只使用单步奖励
- **使用 Bootstrap**：用当前估计 $V(S_{t+1})$ 来更新 $V(S_t)$

### 4.3 偏差-方差权衡

| | Monte Carlo | TD(0) |
|---|-------------|-------|
| 目标 | $G_t$（真实回报） | $r_t + \gamma V(S_{t+1})$ |
| 偏差 | 无偏 | 有偏（依赖 $V$ 估计） |
| 方差 | 高（累积轨迹随机性） | 低（仅单步随机性） |
| Bootstrap | 否 | 是 |
| 适用任务 | 仅 Episodic | Episodic 和 Continuing |

> **为什么 TD 方差更小？**
>
> MC 目标 $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ 包含了从 $t$ 到终止的所有奖励，方差叠加。TD 目标只有 $r_t$ 和 $S_{t+1}$ 是随机的，虽然 $V(S_{t+1})$ 可能不准确（有偏），但不会累积随机性。

### 4.4 n-step TD 与 TD(λ)

n-step TD 是 MC 和 TD(0) 的中间形式：

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(S_{t+n})$$

- $n=1$：TD(0)
- $n=\infty$：Monte Carlo

TD(λ) 方法将所有 n-step return 加权平均：

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

- $\lambda = 0$：等价于 TD(0)
- $\lambda = 1$：等价于 Monte Carlo

## 5. Q-Learning 与 SARSA

为了找到最优策略，我们需要估计动作价值函数 $Q$，这样才能通过 $\arg\max$ 导出策略。

### 5.1 On-policy vs Off-policy

- **On-policy**：评估和改进的是同一个策略（行为策略 = 目标策略）
- **Off-policy**：用行为策略收集数据，评估/改进不同的目标策略

Off-policy 的优势：
- 可以从历史数据（Experience Replay）中学习
- 可以从人类演示或其他策略的数据中学习

### 5.2 SARSA（On-policy）

SARSA 是一种 On-policy TD 控制方法，名字来自更新所需的五元组 $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$。

**SARSA 更新**：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( r_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right)$$

其中 $A_{t+1}$ 是按当前策略（如 $\epsilon$-greedy）实际采样的动作。

### 5.3 Q-Learning（Off-policy）

Q-Learning 直接逼近最优 $Q^*$，而非当前策略的 $Q^\pi$。

**Q-Learning 更新**：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right)$$

关键区别：SARSA 使用 $Q(S_{t+1}, A_{t+1})$（实际采样的动作），Q-Learning 使用 $\max_{a'} Q(S_{t+1}, a')$（最优动作）。

**定理 (Q-Learning 收敛性)**：在满足以下条件时，Q-Learning 收敛到 $Q^*$：
1. 所有状态-动作对被无限次访问
2. 学习率满足 Robbins-Monro 条件：$\sum_t \alpha_t = \infty$，$\sum_t \alpha_t^2 < \infty$

### 5.4 Cliff Walking 示例

Cliff Walking 是一个经典的 Grid World 环境，清晰展示了 Q-Learning 和 SARSA 的行为差异：

- **Q-Learning**：学习最优路径（沿悬崖边走），但训练时因探索经常掉下悬崖
- **SARSA**：学习考虑探索的保守路径，远离悬崖边缘

根本原因：SARSA 的目标包含实际执行的探索动作，Q-Learning 的目标总是选 $\max$。

| | Q-Learning | SARSA |
|---|------------|-------|
| 类型 | Off-policy | On-policy |
| TD target | $r + \gamma \max_{a'} Q(s',a')$ | $r + \gamma Q(s', a')$ |
| 学习目标 | $Q^*$（最优策略） | $Q^\pi$（当前策略） |
| 行为 | 更激进/乐观 | 更保守 |

## 6. Deep Q-Network (DQN)

表格 Q-Learning 无法处理大状态空间（如图像输入）或连续状态空间。DQN 使用神经网络逼近 $Q^*$，是深度强化学习的开创性工作。

### 6.1 函数逼近的动机

表格方法的局限：
- **状态空间爆炸**：Atari 游戏的像素空间约 $256^{210 \times 160 \times 3}$
- **无法泛化**：没见过的状态无法处理
- **连续状态**：无法用表格表示

解决方案：用参数化函数 $Q(s,a;\theta)$（如神经网络）逼近 $Q^*$。

### 6.2 DQN 损失函数

将 Q-Learning 更新转化为回归问题：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s',a';\theta^-)}_{\text{TD target } y} - Q(s,a;\theta) \right)^2 \right]$$

其中 $\mathcal{D}$ 是 Replay Buffer，$\theta^-$ 是 Target Network 的参数。

### 6.3 Experience Replay

将转移 $(s_t, a_t, r_t, s_{t+1})$ 存入 Replay Buffer $\mathcal{D}$，训练时从 $\mathcal{D}$ 中均匀随机采样 mini-batch。

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[scale=0.8]
    % Buffer box
    \draw[thick, rounded corners, fill=blue!10] (-5,-0.8) rectangle (5,0.8);
    \node at (0,1.3) {\textbf{Replay Buffer } $\mathcal{D}$};

    % Buffer contents
    \node at (-4,0) {[old]};
    \node at (-2,0) {$\cdots$};
    \node at (0,0) {$(s,a,r,s')$};
    \node at (2,0) {$\cdots$};
    \node at (4,0) {[new]};

    % Write arrow
    \draw[->, thick, green!60!black] (6,0) -- (5.2,0);
    \node[right] at (6,0) {Write};

    % Sample arrows
    \draw[->, thick, red!70] (-3,-0.8) -- (-3,-2);
    \draw[->, thick, red!70] (0,-0.8) -- (0,-2);
    \draw[->, thick, red!70] (3,-0.8) -- (3,-2);

    % Mini-batch
    \node at (0,-2.5) {Random sample mini-batch};
\end{tikzpicture}
</script>
</div>

Experience Replay 的好处：
1. **打破样本相关性**：随机采样提供更独立的样本
2. **提高数据效率**：每个样本可被多次使用
3. **稳定数据分布**：Buffer 中的数据分布变化缓慢

### 6.4 Target Network

使用一组旧参数 $\theta^-$ 计算 TD target，定期更新 $\theta^- \leftarrow \theta$（如每 $C$ 步）。

Target Network 的作用：
- 避免目标"追着当前估计跑"导致的震荡或发散
- TD target 在一段时间内保持固定，类似于监督学习的固定标签

另一种变体是**软更新**（Soft Update）：$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$，其中 $\tau \ll 1$。

### 6.5 DQN 算法

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[scale=0.75]
    % Title
    \node[font=\bfseries] at (0,6.5) {Deep Q-Network (DQN)};
    \draw[thick] (-6,6.2) -- (6,6.2);

    % Initialization
    \node[anchor=west] at (-5.5,5.5) {Initialize Replay Buffer $\mathcal{D}$};
    \node[anchor=west] at (-5.5,4.8) {Initialize Q-network $\theta$ randomly};
    \node[anchor=west] at (-5.5,4.1) {Initialize Target network $\theta^- \leftarrow \theta$};

    % Outer loop
    \draw[rounded corners, thick, blue!50] (-5.8,3.5) rectangle (5.8,-4.2);
    \node[anchor=west, blue!70] at (-5.5,3.2) {\textbf{For each episode:}};
    \node[anchor=west] at (-5,2.5) {Initialize state $s_1$};

    % Inner loop
    \draw[rounded corners, thick, green!50!black] (-4.8,2) rectangle (5.5,-3.8);
    \node[anchor=west, green!60!black] at (-4.5,1.7) {\textbf{For} $t = 1, 2, \ldots, T$\textbf{:}};

    \node[anchor=west, font=\small] at (-4.2,1.0) {$\epsilon$-greedy: $a = \arg\max_a Q(s,a;\theta)$};
    \node[anchor=west, font=\small] at (-4.2,0.3) {Execute $a$, observe $r, s'$};
    \node[anchor=west, font=\small] at (-4.2,-0.4) {Store $(s,a,r,s')$ in $\mathcal{D}$};
    \node[anchor=west, font=\small] at (-4.2,-1.1) {Sample mini-batch from $\mathcal{D}$};
    \node[anchor=west, font=\small] at (-4.2,-1.8) {TD target: $y = r + \gamma \max_{a'} Q(s',a';\theta^-)$};
    \node[anchor=west, font=\small] at (-4.2,-2.5) {Gradient descent on $(y - Q(s,a;\theta))^2$};
    \node[anchor=west, font=\small] at (-4.2,-3.2) {Every $C$ steps: $\theta^- \leftarrow \theta$};
\end{tikzpicture}
</script>
</div>

> **DQN 的两个关键技巧解决了深度 RL 的稳定性问题**：
> 1. **Experience Replay**：解决样本相关性问题，提高数据效率
> 2. **Target Network**：解决目标不稳定问题，避免震荡发散

### 6.6 DQN 变体

**Double DQN**：解决过估计问题，解耦动作选择和价值评估：

$$y = r + \gamma Q\left(s', \arg\max_{a'} Q(s',a';\theta); \theta^-\right)$$

**Dueling DQN**：将 Q 值分解为状态价值和动作优势：

$$Q(s,a;\theta) = V(s;\theta_v) + \left( A(s,a;\theta_a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a';\theta_a) \right)$$

**其他改进**：
- Prioritized Experience Replay：优先采样 TD 误差大的样本
- Multi-step Learning：使用 n-step return 作为 target
- Distributional RL：学习回报的分布而非期望
- Noisy Networks：用网络参数噪声替代 $\epsilon$-greedy 探索

Rainbow 将上述所有改进组合，在 Atari 游戏上取得了 SOTA 性能。

## 本章小结

**核心内容**：

1. **Bellman 方程**是 Value-Based RL 的理论基础
   - Bellman Expectation：描述给定策略的价值
   - Bellman Optimality：描述最优策略的价值
   - Bellman 算子是 $\gamma$-收缩映射，保证迭代收敛

2. **动态规划**（模型已知时）
   - Policy Evaluation：迭代求解 $V^\pi$
   - Policy Iteration：评估 → 改进 → 评估 → ...
   - Value Iteration：直接迭代 Bellman Optimality 方程

3. **MC vs TD**（模型未知时）
   - MC：无偏高方差，需完整轨迹
   - TD：有偏低方差，每步更新
   - n-step TD 和 TD(λ) 在两者之间权衡

4. **Q-Learning vs SARSA**
   - Q-Learning：Off-policy，学习 $Q^*$，更激进
   - SARSA：On-policy，学习 $Q^\pi$，更保守

5. **DQN**：深度 Value-Based RL
   - Experience Replay：打破样本相关性
   - Target Network：稳定 TD target
   - Double DQN、Dueling DQN 等改进

下一篇文章将介绍 Policy-Based 方法——直接优化参数化策略，以及 Actor-Critic 架构。
