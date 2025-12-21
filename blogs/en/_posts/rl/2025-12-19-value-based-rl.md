---
layout: post
title: "RL Notes (2): Bellman Equations and DQN"
date: 2025-12-19 04:00:00
author: Phonism
tags: [RL, Bellman, Q-Learning, DQN, TD, MC]
lang: en
translation: /value-based-rl/
---

In the previous article, we defined value functions $V^\pi(s)$ and $Q^\pi(s,a)$, which measure how "good" a state or state-action pair is under policy $\pi$. The natural questions now are:

> **How do we compute value functions? How do we derive the optimal policy from value functions?**

The answers to these questions form the core of Value-Based RL. This article introduces three categories of methods:
1. **Dynamic Programming (DP)**: Exact solutions when the environment model is known
2. **Model-free Estimation (MC and TD)**: Learning from samples when the model is unknown
3. **Deep Q-Networks (DQN)**: Handling large or continuous state spaces

## 1. Why Learn Value Functions?

A core observation is: **if we know the optimal action value function $Q^*(s,a)$, we can directly derive the optimal policy**:

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

This is the theoretical foundation of Value-Based methods—instead of learning the policy directly, we obtain it indirectly by learning the value function.

<div class="mermaid">
graph TB
    VBM["Value-Based Methods<br/>DP, MC, TD<br/>Q-Learning, DQN"] --> VF["Value Function<br/>V* or Q*"]
    VF -->|"argmax"| OP["Optimal Policy<br/>π*"]
</div>

## 2. Bellman Equations

The Bellman equation is the core equation of RL, revealing the recursive structure of value functions. Understanding Bellman equations is the foundation for mastering all Value-Based methods.

### 2.1 Intuition: One-Step Decomposition

The core form of the Bellman equation is:

$$V^\pi(s) = \mathbb{E} \left[ r + \gamma V^\pi(s') \right]$$

In words: **value of current state = immediate reward + discounted value of next state**.

This recursive relationship allows us to compute value functions through iteration, without enumerating all possible trajectories.

### 2.2 Bellman Expectation Equation

**Theorem (Bellman Expectation Equation for $V^\pi$)**: For any policy $\pi$, the state value function satisfies:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^\pi(s') \right]$$

**Proof**: Starting from the definition of state value function:

$$\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] \\
&= \mathbb{E}_\pi \left[ r_t + \gamma G_{t+1} \mid S_t = s \right] \\
&= \mathbb{E}_\pi \left[ r_t + \gamma V^\pi(S_{t+1}) \mid S_t = s \right]
\end{aligned}$$

Expanding the expectation: given $S_t = s$, action $A_t = a$ has probability $\pi(a\|s)$; given $(s, a)$, next state $S_{t+1} = s'$ has probability $P(s'\|s,a)$.

Similarly, the Bellman Expectation Equation for action value function:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s',a')$$

> **The mutual representation of $V^\pi$ and $Q^\pi$ is key to deriving Bellman equations**:
> $$V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s,a)$$
> $$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

### 2.3 Bellman Optimality Equation

Optimal value functions satisfy the Bellman Optimality Equation. Unlike the Expectation version, here we take the **maximum** over actions instead of expectation.

**Theorem (Bellman Optimality Equation for $V^*$)**:

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^*(s') \right]$$

**Theorem (Bellman Optimality Equation for $Q^*$)**:

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \max_{a' \in \mathcal{A}} Q^*(s',a')$$

> **Key difference between Bellman Expectation and Bellman Optimality**:
> - **Bellman Expectation**: Weighted average over actions (by $\pi(a\|s)$), describes value of a **given policy**
> - **Bellman Optimality**: Maximum over actions ($\max_a$), describes value of the **optimal policy**
>
> **Q-Learning directly approximates $Q^*$, hence uses Bellman Optimality Equation as the update target**.

### 2.4 Bellman Operator and Contraction Property

**Definition (Bellman Operator)**:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

$$(\mathcal{T}^* V)(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

**Theorem (Contraction Property of Bellman Operator)**: The Bellman operator is a $\gamma$-contraction mapping:

$$\| \mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2 \|_\infty \leq \gamma \| V_1 - V_2 \|_\infty$$

> **Important corollaries of contraction property**:
> 1. **Unique fixed point**: Bellman operator has a unique fixed point $V^\pi$ (or $V^*$)
> 2. **Iterative convergence**: Starting from any $V_0$, $V_{k+1} = \mathcal{T}V_k$ converges to the fixed point
> 3. **Convergence rate**: Error decays exponentially at rate $\gamma^k$

## 3. Dynamic Programming Methods

When environment model $P(s'\|s,a)$ and $R(s,a)$ are fully known, Dynamic Programming (DP) methods can exactly solve for the optimal policy.

### 3.1 Policy Evaluation

Given policy $\pi$, compute $V^\pi$ by iterating the Bellman Expectation equation:

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

Starting from any $V_0$, iterate until convergence $V_k \to V^\pi$.

### 3.2 Policy Improvement

**Theorem (Policy Improvement Theorem)**: Let $\pi$ and $\pi'$ be two policies. If for all $s \in \mathcal{S}$:

$$Q^\pi(s, \pi'(s)) \geq V^\pi(s)$$

then $\pi'$ is at least as good as $\pi$: for all $s$, $V^{\pi'}(s) \geq V^\pi(s)$.

**Greedy Policy Improvement**:

$$\pi'(s) = \arg\max_a Q^\pi(s,a) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

### 3.3 Policy Iteration

Alternate between Policy Evaluation and Policy Improvement:

<div class="mermaid">
graph LR
    P0["π₀"] -->|"Eval"| V0["V^π₀"]
    V0 -->|"Improve"| P1["π₁"]
    P1 -->|"Eval"| V1["V^π₁"]
    V1 -->|"Improve"| P2["π₂"]
    P2 -->|"..."| END["π*"]
</div>

For finite MDPs, Policy Iteration converges to optimal policy $\pi^*$ in finite steps.

### 3.4 Value Iteration

Combine Policy Evaluation and Policy Improvement into one step, directly iterating the Bellman Optimality equation:

$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

After convergence, $\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'\|s,a) V(s') \right]$.

| | Policy Iteration | Value Iteration |
|---|------------------|-----------------|
| Policy Evaluation | Full solve for $V^\pi$ | Only one Bellman update |
| Theoretical basis | Bellman Expectation | Bellman Optimality |
| Per-iteration complexity | High (multiple inner loops) | Low (single sweep) |
| Iterations to converge | Few | Many |

### 3.5 Limitations of DP Methods

1. **Requires complete model**: Must know $P(s'\|s,a)$ and $R(s,a)$
2. **State space enumeration**: Each iteration requires sweeping all states, complexity $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$
3. **Cannot handle continuous spaces**: Tabular methods don't apply directly

These limitations motivated the development of model-free methods (MC, TD) and function approximation (DQN).

## 4. Model-free Methods: Monte Carlo vs Temporal Difference

When the environment model is unknown, we need to estimate value functions from samples of interaction.

### 4.1 Monte Carlo Estimation

The core idea of MC: **use actual trajectory returns $G_t$ to estimate the expectation**.

**Monte Carlo Update**:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

where $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$ is the actual return from time $t$ to episode end.

MC method characteristics:
- **Must wait for episode to end** to compute $G_t$, only for episodic tasks
- **Unbiased estimate**: $\mathbb{E}[G_t \| S_t = s] = V^\pi(s)$
- **High variance**: $G_t$ accumulates randomness from the entire trajectory
- **No bootstrapping**: Target comes entirely from real samples

### 4.2 Temporal Difference Estimation

The core idea of TD: **use "one-step reward + estimated value of next state" to replace full return**.

**TD(0) Update**:

$$V(S_t) \leftarrow V(S_t) + \alpha \left( \underbrace{r_t + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t) \right)$$

TD error: $\delta_t = r_t + \gamma V(S_{t+1}) - V(S_t)$

TD method characteristics:
- **Can update every step**, suitable for continuing tasks and online learning
- **Biased estimate**: Uses estimated $V(S_{t+1})$
- **Low variance**: Only uses single-step reward
- **Uses bootstrapping**: Uses current estimate $V(S_{t+1})$ to update $V(S_t)$

### 4.3 Bias-Variance Trade-off

| | Monte Carlo | TD(0) |
|---|-------------|-------|
| Target | $G_t$ (actual return) | $r_t + \gamma V(S_{t+1})$ |
| Bias | Unbiased | Biased (depends on $V$ estimate) |
| Variance | High (accumulates trajectory randomness) | Low (only single-step randomness) |
| Bootstrap | No | Yes |
| Task type | Episodic only | Episodic and Continuing |

> **Why does TD have lower variance?**
>
> MC target $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ contains all rewards from $t$ to termination, variances add up. TD target only has $r_t$ and $S_{t+1}$ as random variables—although $V(S_{t+1})$ may be inaccurate (biased), it doesn't accumulate randomness.

### 4.4 n-step TD and TD(λ)

n-step TD is intermediate between MC and TD(0):

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(S_{t+n})$$

- $n=1$: TD(0)
- $n=\infty$: Monte Carlo

TD(λ) method computes weighted average of all n-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

- $\lambda = 0$: Equivalent to TD(0)
- $\lambda = 1$: Equivalent to Monte Carlo

## 5. Q-Learning and SARSA

To find the optimal policy, we need to estimate action value function $Q$, so we can derive policy via $\arg\max$.

### 5.1 On-policy vs Off-policy

- **On-policy**: Evaluate and improve the same policy (behavior policy = target policy)
- **Off-policy**: Use behavior policy to collect data, evaluate/improve a different target policy

Off-policy advantages:
- Can learn from historical data (Experience Replay)
- Can learn from human demonstrations or other policies

### 5.2 SARSA (On-policy)

SARSA is an on-policy TD control method, named after the quintuple $(S_t, A_t, R_t, S_{t+1}, A_{t+1})$ needed for updates.

**SARSA Update**:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( r_t + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right)$$

where $A_{t+1}$ is the action actually sampled according to current policy (e.g., $\epsilon$-greedy).

### 5.3 Q-Learning (Off-policy)

Q-Learning directly approximates optimal $Q^*$, not the current policy's $Q^\pi$.

**Q-Learning Update**:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( r_t + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right)$$

Key difference: SARSA uses $Q(S_{t+1}, A_{t+1})$ (actually sampled action), Q-Learning uses $\max_{a'} Q(S_{t+1}, a')$ (optimal action).

**Theorem (Q-Learning Convergence)**: Q-Learning converges to $Q^*$ under the following conditions:
1. All state-action pairs are visited infinitely often
2. Learning rate satisfies Robbins-Monro conditions: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$

### 5.4 Cliff Walking Example

Cliff Walking is a classic Grid World environment that clearly demonstrates the behavioral difference between Q-Learning and SARSA:

- **Q-Learning**: Learns optimal path (along cliff edge), but often falls during training due to exploration
- **SARSA**: Learns conservative path accounting for exploration, staying away from cliff edge

Root cause: SARSA's target includes actually executed exploration actions; Q-Learning's target always selects $\max$.

| | Q-Learning | SARSA |
|---|------------|-------|
| Type | Off-policy | On-policy |
| TD target | $r + \gamma \max_{a'} Q(s',a')$ | $r + \gamma Q(s', a')$ |
| Learning target | $Q^*$ (optimal policy) | $Q^\pi$ (current policy) |
| Behavior | More aggressive/optimistic | More conservative |

## 6. Deep Q-Network (DQN)

Tabular Q-Learning cannot handle large state spaces (like image inputs) or continuous state spaces. DQN uses neural networks to approximate $Q^*$, pioneering deep reinforcement learning.

### 6.1 Motivation for Function Approximation

Tabular method limitations:
- **State space explosion**: Atari game pixel space is about $256^{210 \times 160 \times 3}$
- **No generalization**: Cannot handle unseen states
- **Continuous states**: Cannot be represented by tables

Solution: Use parameterized function $Q(s,a;\theta)$ (e.g., neural network) to approximate $Q^*$.

### 6.2 DQN Loss Function

Transform Q-Learning update into a regression problem:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s',a';\theta^-)}_{\text{TD target } y} - Q(s,a;\theta) \right)^2 \right]$$

where $\mathcal{D}$ is the Replay Buffer, $\theta^-$ is the Target Network parameters.

### 6.3 Experience Replay

Store transitions $(s_t, a_t, r_t, s_{t+1})$ in Replay Buffer $\mathcal{D}$, randomly sample mini-batches from $\mathcal{D}$ for training.

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

Benefits of Experience Replay:
1. **Break sample correlation**: Random sampling provides more independent samples
2. **Improve data efficiency**: Each sample can be reused multiple times
3. **Stabilize data distribution**: Buffer distribution changes slowly

### 6.4 Target Network

Use old parameters $\theta^-$ to compute TD target, periodically update $\theta^- \leftarrow \theta$ (e.g., every $C$ steps).

Target Network purpose:
- Avoid oscillation or divergence from target "chasing" current estimate
- TD target stays fixed for a period, similar to fixed labels in supervised learning

A variant is **Soft Update**: $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$, where $\tau \ll 1$.

### 6.5 DQN Algorithm

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

> **DQN's two key techniques solve deep RL stability problems**:
> 1. **Experience Replay**: Addresses sample correlation, improves data efficiency
> 2. **Target Network**: Addresses target instability, prevents oscillation and divergence

### 6.6 DQN Variants

**Double DQN**: Addresses overestimation by decoupling action selection and value evaluation:

$$y = r + \gamma Q\left(s', \arg\max_{a'} Q(s',a';\theta); \theta^-\right)$$

**Dueling DQN**: Decomposes Q value into state value and action advantage:

$$Q(s,a;\theta) = V(s;\theta_v) + \left( A(s,a;\theta_a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a';\theta_a) \right)$$

**Other improvements**:
- Prioritized Experience Replay: Prioritize sampling transitions with large TD error
- Multi-step Learning: Use n-step return as target
- Distributional RL: Learn distribution of return rather than expectation
- Noisy Networks: Replace $\epsilon$-greedy exploration with network parameter noise

Rainbow combines all these improvements, achieving SOTA performance on Atari games.

## Summary

**Core content**:

1. **Bellman Equations** are the theoretical foundation of Value-Based RL
   - Bellman Expectation: Describes value of a given policy
   - Bellman Optimality: Describes value of optimal policy
   - Bellman operator is a $\gamma$-contraction, guaranteeing iterative convergence

2. **Dynamic Programming** (when model is known)
   - Policy Evaluation: Iteratively solve for $V^\pi$
   - Policy Iteration: Evaluate → Improve → Evaluate → ...
   - Value Iteration: Directly iterate Bellman Optimality equation

3. **MC vs TD** (when model is unknown)
   - MC: Unbiased, high variance, requires complete trajectory
   - TD: Biased, low variance, updates every step
   - n-step TD and TD(λ) trade off between the two

4. **Q-Learning vs SARSA**
   - Q-Learning: Off-policy, learns $Q^*$, more aggressive
   - SARSA: On-policy, learns $Q^\pi$, more conservative

5. **DQN**: Deep Value-Based RL
   - Experience Replay: Breaks sample correlation
   - Target Network: Stabilizes TD target
   - Double DQN, Dueling DQN, and other improvements

The next article will introduce Policy-Based methods—directly optimizing parameterized policies—and the Actor-Critic architecture.
