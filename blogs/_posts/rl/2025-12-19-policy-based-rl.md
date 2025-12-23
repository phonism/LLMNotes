---
layout: post
title: "RL 学习笔记（三）：基于策略的强化学习"
date: 2025-12-19 05:00:00
author: Qi Lu
tags: [RL, PPO]
lang: zh
translation: /en/policy-based-rl/
series: rl
series_order: 3
---

在上一篇文章中，我们介绍了 Value-Based 方法：先学习 $Q^*$，再通过 $\arg\max$ 导出策略。这种方法在离散动作空间中效果很好，但遇到以下问题时会遇到困难：

> **如果动作空间是连续的（如机器人关节角度），如何计算 $\arg\max_a Q(s,a)$？**
>
> **如果最优策略是随机的（如石头剪刀布），如何用确定性策略表示？**

Policy-Based 方法提供了一种更直接的思路：**直接参数化策略 $\pi_\theta(a\|s)$，通过梯度上升最大化期望回报**。

## 1. 为什么需要 Policy-Based 方法？

### 1.1 Value-Based 方法的局限性

1. **连续动作空间困难**：$\max_a Q(s,a)$ 需要枚举或优化所有动作
2. **函数逼近不稳定**（Deadly Triad）：函数逼近 + Bootstrapping + Off-policy 可能发散
3. **优化目标间接**：最小化 TD 误差，而非直接优化期望回报 $J(\pi)$
4. **只能学习确定性策略**：$\arg\max$ 输出确定动作，但某些环境中随机策略更优

### 1.2 参数化策略

Policy-Based 方法直接参数化策略 $\pi_\theta(a\|s)$：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G_0 \right] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

策略参数化的常见形式：

- **离散动作空间**：Softmax 输出 Categorical 分布
  $$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a))}{\sum_{a'} \exp(f_\theta(s,a'))}$$

- **连续动作空间**：输出高斯分布的参数 $(\mu_\theta(s), \sigma_\theta(s))$
  $$\pi_\theta(a|s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta^2(s))$$

> **Policy-Based 方法的优势**：
> 1. **处理连续动作**：直接输出动作分布，无需 $\arg\max$
> 2. **学习随机策略**：可以输出动作的概率分布
> 3. **直接优化目标**：梯度上升直接最大化 $J(\theta)$
> 4. **更好的收敛性质**：策略参数的小变化导致策略的小变化（光滑）

## 2. Policy Gradient 定理

Policy Gradient 定理是 Policy-Based RL 的理论基础，它给出了目标函数 $J(\theta)$ 关于参数 $\theta$ 的梯度表达式。

### 2.1 Log-Derivative Trick

计算 $\nabla_\theta J(\theta)$ 的关键技巧是 **Log-Derivative Trick**：

$$\nabla_\theta p(x|\theta) = p(x|\theta) \nabla_\theta \log p(x|\theta)$$

**证明**：由对数的求导法则，$\nabla_\theta \log p(x\|\theta) = \frac{\nabla_\theta p(x\|\theta)}{p(x\|\theta)}$，两边同乘 $p(x\|\theta)$ 即得。

> Log-Derivative Trick 的妙处：将对 $p(x\|\theta)$ 的求导转化为对 $\log p(x\|\theta)$ 的求导，后者往往更容易计算，特别是当 $p$ 是乘积形式时。

### 2.2 Policy Gradient 定理

**定理 (Policy Gradient Theorem)**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 是从时刻 $t$ 开始的 reward-to-go。

**证明思路**：

**Step 1**：应用 Log-Derivative Trick

$$\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int p(\tau|\theta) R(\tau) d\tau \\
&= \int p(\tau|\theta) \nabla_\theta \log p(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p(\tau|\theta) \cdot R(\tau) \right]
\end{aligned}$$

**Step 2**：展开 $\nabla_\theta \log p(\tau\|\theta)$

回顾轨迹概率分解：$p(\tau\|\theta) = p(s_0) \prod_{t} \pi_\theta(a_t\|s_t) P(s_{t+1}\|s_t,a_t)$

取对数并求梯度：

$$\nabla_\theta \log p(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

> **关键观察**：$p(s_0)$ 是环境的初始状态分布，$P(s_{t+1}\|s_t,a_t)$ 是环境的动力学模型，它们都与策略参数 $\theta$ 无关，因此梯度为零！
>
> 这意味着：**即使不知道环境动力学 $P$，我们也能计算 Policy Gradient**——这是 Policy Gradient 方法能够 Model-Free 的根本原因。

**Step 3**：引入 Reward-to-go（因果性）

动作 $a_t$ 只影响未来奖励，不影响过去。因此可以用 reward-to-go $G_t$ 替代完整回报 $R(\tau)$。

> **Policy Gradient 定理的直观理解**：
> - $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ 是"增加动作 $a_t$ 概率"的方向
> - $G_t$ 是该动作之后获得的累积奖励
> - 如果 $G_t > 0$：沿梯度方向更新，增加 $a_t$ 的概率
> - 如果 $G_t < 0$：反向更新，减少 $a_t$ 的概率
>
> 简言之：**好的动作更可能被选择，坏的动作更少被选择**。

## 3. REINFORCE 算法

REINFORCE 是最简单的 Policy Gradient 算法，直接使用蒙特卡洛采样来估计梯度。

<!-- tikz-source: rl-reinforce
\begin{algorithm}[H]
\caption{REINFORCE}
\ForEach{episode}{
    从 $\pi_\theta$ 采样轨迹 $\tau = (s_0, a_0, r_0, \ldots, s_T)$\;
    \For{$t = 0, 1, \ldots, T$}{
        计算回报 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$\;
    }
    更新：$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$\;
}
\tcp{Monte Carlo: 使用实际回报 $G_t$}
\end{algorithm}
-->
![REINFORCE Algorithm]({{ site.baseurl }}/assets/figures/rl-reinforce.svg)

**REINFORCE 是无偏估计**：$\mathbb{E}[\hat{g}] = \nabla_\theta J(\theta)$

**但方差很大**：
- $G_t$ 累积了从 $t$ 到终止的所有随机性
- 轨迹越长，方差越大
- 奖励稀疏时，大部分轨迹的 $G_t \approx 0$

## 4. Baseline 与方差降低

### 4.1 Baseline 技巧

一个巧妙的技巧是：从 $G_t$ 中减去一个 baseline $b(s_t)$，可以降低方差而不引入偏差。

**定理 (Baseline 不改变期望)**：对于任意只依赖于状态 $s$（不依赖于动作 $a$）的函数 $b(s)$：

$$\mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \right] = 0$$

**证明**：由于 $b(s)$ 不依赖于 $a$，可以提出期望外：

$$b(s) \cdot \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$

因此，Policy Gradient 可以写成：

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]$$

### 4.2 为什么 Baseline 能降低方差？

- $G_t$ 可能总是正的（如奖励都是正数），导致所有动作概率都被增加
- 减去 $b(s)$（如平均回报），使得 $G_t - b(s_t)$ 有正有负
- **好于平均的动作被增强，差于平均的动作被削弱**

### 4.3 最优 Baseline

**定理**：在不改变期望的前提下，使方差最小的 baseline 是状态价值函数：$b^*(s) = V^\pi(s)$

当 $b(s) = V^\pi(s)$ 时，$G_t - V^\pi(s_t)$ 的期望正是 **advantage 函数**！

## 5. Advantage Function 与 Actor-Critic

### 5.1 Advantage 的定义与直觉

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

当使用 $V^\pi(s)$ 作为 baseline 时：

$$\mathbb{E}_\pi \left[ G_t - V^\pi(s_t) \mid s_t, a_t \right] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

因此，Policy Gradient with Advantage：

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]$$

> **Advantage $A(s,a)$ 的直觉**：
> - $V(s)$：在状态 $s$ 的"平均"表现
> - $Q(s,a)$：在状态 $s$ 选择动作 $a$ 的表现
> - $A(s,a) = Q(s,a) - V(s)$：动作 $a$ 比平均好多少
>
> $A > 0$：这个动作好于平均，应该增加其概率
> $A < 0$：这个动作差于平均，应该减少其概率

### 5.2 Advantage 的估计方法

1. **Monte Carlo 估计**：$\hat{A}\_t^{\text{MC}} = G\_t - \hat{V}(s\_t)$（无偏但高方差）

2. **TD 估计**（1-step）：$\hat{A}\_t^{\text{TD}} = r\_t + \gamma \hat{V}(s\_{t+1}) - \hat{V}(s\_t) = \delta\_t$（低方差但有偏）

3. **n-step 估计**：介于两者之间

4. **GAE**：通过 $\lambda$ 参数灵活权衡偏差和方差

### 5.3 Actor-Critic 架构

为了估计 $\hat{V}(s)$，我们引入一个 **Critic** 网络。Actor-Critic 方法同时学习：
- **Actor**：策略网络 $\pi_\theta(a\|s)$，输出动作分布
- **Critic**：价值网络 $\hat{V}_\phi(s)$，估计状态价值

<!-- tikz-source: rl-actor-critic
\begin{tikzpicture}[scale=0.85]
    % Actor box
    \draw[rounded corners, fill=blue!15, thick] (-2,2) rectangle (2,3.5);
    \node at (0,3.1) {\textbf{Actor}};
    \node at (0,2.5) {$\pi_\theta(a|s)$};

    % Critic box
    \draw[rounded corners, fill=green!15, thick] (-2,-0.5) rectangle (2,1);
    \node at (0,0.6) {\textbf{Critic}};
    \node at (0,0) {$\hat{V}_\phi(s)$};

    % Environment box
    \draw[rounded corners, fill=orange!20, thick] (5,0.5) rectangle (8,2.5);
    \node at (6.5,1.5) {\textbf{Environment}};

    % State input
    \node[left] at (-3,2.75) {$s$};
    \draw[->, thick] (-3,2.75) -- (-2,2.75);
    \draw[->, thick] (-3,0.25) -- (-2,0.25);
    \node[left] at (-3,0.25) {$s$};

    % Actor to Environment
    \draw[->, thick, blue!70] (2,2.75) -- (5,2.75) -- (5,2);
    \node[above] at (3.5,2.75) {$a \sim \pi_\theta$};

    % Environment feedback
    \draw[->, thick, red!70] (8,1.5) -- (9,1.5) -- (9,-1) -- (0,-1) -- (0,-0.5);
    \node[right] at (9,0.25) {$s', r$};

    % Advantage calculation
    \draw[->, thick, green!60!black] (0,1) -- (0,2);
    \node[right] at (0.2,1.5) {$\hat{A}_t$};

    % Update arrow
    \node[below] at (0,-1.5) {\small Update Actor with $\nabla_\theta \log \pi_\theta \cdot \hat{A}_t$};
\end{tikzpicture}
-->
![Actor-Critic Architecture]({{ site.baseurl }}/assets/figures/rl-actor-critic.svg)

**A2C (Advantage Actor-Critic)** 的核心更新规则：

**Actor 更新**（Policy Gradient with Advantage）：
$$\theta \leftarrow \theta + \alpha_\theta \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$$

**Critic 更新**（Value Function Regression）：
$$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi \sum_t \left( \hat{V}_\phi(s_t) - \text{target} \right)^2$$

> **为什么需要 Critic？**
> - 提供 $\hat{V}(s)$ 来计算 advantage $\hat{A}_t$
> - 比纯 MC（使用 $G_t$）方差更小
> - 可每步更新，不用等 episode 结束

## 6. Generalized Advantage Estimation (GAE)

GAE 提供了一种在偏差和方差之间灵活权衡的 advantage 估计方法，是现代 Policy Gradient 算法（如 PPO）的核心组件。

### 6.1 GAE 的定义

**定义 (Generalized Advantage Estimation)**：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ 是 TD 残差，$\lambda \in [0,1]$ 是衰减参数。

**定理**：GAE 等价于 n-step Advantage 的加权和：

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}$$

### 6.2 $\lambda$ 参数的偏差-方差权衡

| $\lambda$ 值 | 等价形式 | 偏差 | 方差 |
|-------------|---------|------|------|
| $\lambda = 0$ | $\delta_t$（TD） | 高（依赖 $\hat{V}$） | 低 |
| $\lambda = 1$ | $G_t - \hat{V}(s_t)$（MC） | 低 | 高 |
| $\lambda \in (0,1)$ | 加权平均 | 中等 | 中等 |

实践中，$\lambda = 0.95$ 或 $\lambda = 0.97$ 是常用的选择。

> **GAE 的直觉理解**：
> - $\delta_t$ 是"一步后用 Critic 估计剩余价值"的 advantage
> - GAE 把多步 $\delta$ 加权求和，$(\gamma\lambda)^l$ 让远处的 $\delta$ 权重指数衰减
> - $\lambda$ 越小，越依赖 Critic 估计（偏差大但方差小）
> - $\lambda$ 越大，越依赖实际回报（偏差小但方差大）

### 6.3 GAE 的实际计算

GAE 可以通过递推高效计算：

$$\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \hat{A}_{t+1}^{\text{GAE}}$$

边界条件：$\hat{A}_T^{\text{GAE}} = 0$。从后往前计算，复杂度为 $O(T)$。

## 7. 重要性采样与 Off-Policy Policy Gradient

### 7.1 On-Policy 的问题

Policy Gradient 是 on-policy 的：每次更新 $\theta$ 后，旧数据的分布就与新策略不同了。这导致：
- 数据只能用一次，样本效率低
- 每次更新都需要重新采样

重要性采样（Importance Sampling, IS）允许我们复用旧数据。

### 7.2 重要性采样原理

用分布 $q(x)$ 的样本估计 $p(x)$ 下的期望：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$

其中 $\rho(x) = \frac{p(x)}{q(x)}$ 称为**重要性权重**。

应用到 Policy Gradient，单步的重要性权重：

$$\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

**Off-policy Policy Gradient**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \rho_t(\theta) \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$$

### 7.3 方差问题

当 $\rho_t$ 偏离 1 太多时，方差会急剧增大。需要限制策略更新幅度，保持 $\rho_t \approx 1$。

## 8. Trust Region 方法：TRPO 与 PPO

### 8.1 TRPO：KL 约束优化

TRPO 通过 KL 散度约束限制策略更新：

$$\begin{aligned}
\max_\theta \quad & L(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \rho_t(\theta) \hat{A}_t \right] \\
\text{s.t.} \quad & \bar{D}_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) \leq \delta
\end{aligned}$$

TRPO 理论上保证单调改进，但需要计算 KL 散度的 Hessian，实现复杂。

### 8.2 PPO：简化的 Trust Region

PPO 通过更简单的方式近似 TRPO 的效果。

**PPO-Clip 目标**：

$$L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( \rho_t \hat{A}_t, \, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $\text{clip}(x, a, b) = \max(a, \min(x, b))$，$\epsilon$ 通常取 0.1 或 0.2。

**PPO-Clip 的直觉**：
- 当 $\hat{A}_t > 0$（好动作）：目标是增加概率，但当 $\rho_t > 1+\epsilon$ 时截断，防止过度增加
- 当 $\hat{A}_t < 0$（坏动作）：目标是减少概率，但当 $\rho_t < 1-\epsilon$ 时截断，防止过度减少

### 8.3 Entropy Bonus

为了鼓励探索，PPO 通常还会加入 entropy bonus：

$$L^{\text{total}}(\theta) = L^{\text{CLIP}}(\theta) + c_1 \cdot H(\pi_\theta)$$

其中 $H(\pi_\theta) = -\mathbb{E}[\log \pi_\theta(a\|s)]$ 是策略的熵。

### 8.4 PPO 完整算法

<!-- tikz-source: rl-ppo-algorithm
\begin{algorithm}[H]
\caption{Proximal Policy Optimization (PPO)}
\For{iteration $= 1, 2, \ldots$}{
    \tcp{数据收集}
    用 $\pi_\theta$ 收集 $N$ 条轨迹\;
    用 $\hat{V}_\phi$ 计算 GAE：$\hat{A}_t$\;
    计算回报：$\hat{R}_t = \hat{A}_t + \hat{V}_\phi(s_t)$\;
    保存 $\pi_{\text{old}} = \pi_\theta$（固定）\;
    \tcp{策略更新}
    \For{$k = 1, \ldots, K$}{
        计算重要性比率：$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$\;
        $L^{\text{CLIP}} = \mathbb{E}[\min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$\;
        $L^{\text{Value}} = \mathbb{E}[(\hat{V}_\phi(s_t) - \hat{R}_t)^2]$\;
        $L^{\text{Entropy}} = -\mathbb{E}[\log \pi_\theta(a_t|s_t)]$\;
        总目标：$L = L^{\text{CLIP}} - c_1 L^{\text{Value}} + c_2 L^{\text{Entropy}}$\;
        对 $\theta$ 梯度上升，对 $\phi$ 梯度下降\;
    }
}
\end{algorithm}
-->
![PPO Algorithm]({{ site.baseurl }}/assets/figures/rl-ppo-algorithm.svg)

> **PPO 的成功原因**：
> 1. **简单高效**：只需一阶优化，不需要计算 Hessian
> 2. **样本效率**：可多次复用同一批数据（$K$ 次更新）
> 3. **稳定性**：clip 机制防止策略剧烈变化
> 4. **鲁棒性**：对超参数不敏感，适用于多种任务
>
> PPO 是目前最常用的 Policy Gradient 算法，也是 RLHF 中的标准选择。

## 本章小结

**核心内容**：

1. **Policy Gradient 定理**
   - 给出了目标函数梯度的解析形式：$\nabla_\theta J = \mathbb{E}[\sum_t \nabla \log \pi \cdot G_t]$
   - Log-Derivative Trick 是推导的关键
   - 环境动力学与 $\theta$ 无关，实现了 Model-Free

2. **方差降低技术**
   - Baseline 技巧：减去 $b(s)$ 不改变期望但降低方差
   - 最优 baseline 是 $V^\pi(s)$
   - 使用 Advantage $A = Q - V$ 替代 $G_t$

3. **Actor-Critic 架构**
   - Actor（策略网络）+ Critic（价值网络）
   - Critic 提供 $\hat{V}(s)$ 来估计 advantage

4. **GAE**
   - $\hat{A}^{\text{GAE}} = \sum_l (\gamma\lambda)^l \delta_{t+l}$
   - $\lambda$ 控制偏差-方差权衡

5. **Trust Region 方法**
   - 重要性采样允许复用旧数据，但需要限制策略变化
   - TRPO：KL 约束优化，实现复杂
   - PPO：clip 机制，简单高效，是实践中的首选

<!-- tikz-source: rl-pg-evolution
\begin{tikzpicture}[
    node/.style={draw, rounded corners, fill=blue!15, minimum width=2cm, minimum height=1.2cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[node, fill=red!20] (r) at (0, 0) {REINFORCE\\{\footnotesize 无偏高方差}};
    \node[node, fill=orange!20] (b) at (3.5, 0) {+ Baseline\\{\footnotesize 降低方差}};
    \node[node, fill=yellow!30] (ac) at (7, 0) {Actor-Critic\\{\footnotesize 学习 Critic}};
    \node[node, fill=green!20] (gae) at (10.5, 0) {+ GAE\\{\footnotesize $\lambda$ 权衡}};
    \node[node, fill=purple!20] (ppo) at (14, 0) {PPO\\{\footnotesize 稳定高效}};

    \draw[arrow] (r) -- node[above, font=\footnotesize] {+Baseline} (b);
    \draw[arrow] (b) -- node[above, font=\footnotesize] {+Critic} (ac);
    \draw[arrow] (ac) -- node[above, font=\footnotesize] {+GAE} (gae);
    \draw[arrow] (gae) -- node[above, font=\footnotesize] {+Clip} (ppo);
\end{tikzpicture}
-->
![Policy Gradient 演进]({{ site.baseurl }}/assets/figures/rl-pg-evolution.svg)

下一篇文章将介绍 Model-Based RL 与多智能体学习，包括 MCTS 和 AlphaGo/Zero。
