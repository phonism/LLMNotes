---
layout: post
title: "RL Notes (3): Policy-Based RL"
date: 2025-12-19 05:00:00
author: Qi Lu
tags: [RL, PPO]
lang: en
translation: /policy-based-rl/
---

In the previous article, we introduced Value-Based methods: first learn $Q^*$, then derive the policy through $\arg\max$. This approach works well in discrete action spaces, but encounters difficulties when facing the following problems:

> **If the action space is continuous (such as robot joint angles), how do we compute $\arg\max_a Q(s,a)$?**
>
> **If the optimal policy is stochastic (such as rock-paper-scissors), how can we represent it with a deterministic policy?**

Policy-Based methods provide a more direct approach: **directly parameterize the policy $\pi_\theta(a\|s)$ and maximize expected return through gradient ascent**.

## 1. Why Do We Need Policy-Based Methods?

### 1.1 Limitations of Value-Based Methods

1. **Continuous action spaces are difficult**: $\max_a Q(s,a)$ requires enumerating or optimizing over all actions
2. **Function approximation instability** (Deadly Triad): Function approximation + Bootstrapping + Off-policy can diverge
3. **Indirect optimization objective**: Minimizing TD error rather than directly optimizing expected return $J(\pi)$
4. **Can only learn deterministic policies**: $\arg\max$ outputs deterministic actions, but stochastic policies are better in some environments

### 1.2 Parameterized Policies

Policy-Based methods directly parameterize the policy $\pi_\theta(a\|s)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G_0 \right] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

Common forms of policy parameterization:

- **Discrete action space**: Softmax outputs Categorical distribution
  $$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a))}{\sum_{a'} \exp(f_\theta(s,a'))}$$

- **Continuous action space**: Output parameters of Gaussian distribution $(\mu_\theta(s), \sigma_\theta(s))$
  $$\pi_\theta(a|s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta^2(s))$$

> **Advantages of Policy-Based Methods**:
> 1. **Handle continuous actions**: Directly output action distributions, no need for $\arg\max$
> 2. **Learn stochastic policies**: Can output probability distributions over actions
> 3. **Direct optimization objective**: Gradient ascent directly maximizes $J(\theta)$
> 4. **Better convergence properties**: Small changes in policy parameters lead to small changes in policy (smoothness)

## 2. Policy Gradient Theorem

The Policy Gradient theorem is the theoretical foundation of Policy-Based RL, providing the gradient expression of the objective function $J(\theta)$ with respect to parameters $\theta$.

### 2.1 Log-Derivative Trick

The key trick for computing $\nabla_\theta J(\theta)$ is the **Log-Derivative Trick**:

$$\nabla_\theta p(x|\theta) = p(x|\theta) \nabla_\theta \log p(x|\theta)$$

**Proof**: By the derivative rule for logarithms, $\nabla_\theta \log p(x\|\theta) = \frac{\nabla_\theta p(x\|\theta)}{p(x\|\theta)}$, multiplying both sides by $p(x\|\theta)$ gives the result.

> The elegance of the Log-Derivative Trick: It converts the derivative of $p(x\|\theta)$ into the derivative of $\log p(x\|\theta)$, which is often easier to compute, especially when $p$ is in product form.

### 2.2 Policy Gradient Theorem

**Theorem (Policy Gradient Theorem)**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

where $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ is the reward-to-go from time $t$.

**Proof Sketch**:

**Step 1**: Apply the Log-Derivative Trick

$$\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int p(\tau|\theta) R(\tau) d\tau \\
&= \int p(\tau|\theta) \nabla_\theta \log p(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p(\tau|\theta) \cdot R(\tau) \right]
\end{aligned}$$

**Step 2**: Expand $\nabla_\theta \log p(\tau\|\theta)$

Recall the trajectory probability decomposition: $p(\tau\|\theta) = p(s_0) \prod_{t} \pi_\theta(a_t\|s_t) P(s_{t+1}\|s_t,a_t)$

Taking the logarithm and computing the gradient:

$$\nabla_\theta \log p(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

> **Key Observation**: $p(s_0)$ is the environment's initial state distribution, and $P(s_{t+1}\|s_t,a_t)$ is the environment's dynamics model, both of which are independent of the policy parameters $\theta$, so their gradients are zero!
>
> This means: **Even without knowing the environment dynamics $P$, we can still compute the Policy Gradient**—this is the fundamental reason why Policy Gradient methods can be Model-Free.

**Step 3**: Introduce Reward-to-go (Causality)

Action $a_t$ only affects future rewards, not past rewards. Therefore, we can replace the full return $R(\tau)$ with reward-to-go $G_t$.

> **Intuitive Understanding of the Policy Gradient Theorem**:
> - $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ is the direction that "increases the probability of action $a_t$"
> - $G_t$ is the cumulative reward obtained after that action
> - If $G_t > 0$: Update in the gradient direction, increase the probability of $a_t$
> - If $G_t < 0$: Update in the opposite direction, decrease the probability of $a_t$
>
> In short: **Good actions become more likely to be selected, bad actions become less likely to be selected**.

## 3. REINFORCE Algorithm

REINFORCE is the simplest Policy Gradient algorithm, using Monte Carlo sampling directly to estimate the gradient.

<!-- tikz-source: rl-reinforce-en
\begin{algorithm}[H]
\caption{REINFORCE}
\ForEach{episode}{
    Sample trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T)$ from $\pi_\theta$\;
    \For{$t = 0, 1, \ldots, T$}{
        Compute return $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$\;
    }
    Update: $\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot G_t$\;
}
\tcp{Monte Carlo: use actual returns $G_t$}
\end{algorithm}
-->
![REINFORCE Algorithm]({{ site.baseurl }}/assets/figures/rl-reinforce-en.svg)

**REINFORCE is an unbiased estimator**: $\mathbb{E}[\hat{g}] = \nabla_\theta J(\theta)$

**But has high variance**:
- $G_t$ accumulates all randomness from $t$ to termination
- Longer trajectories lead to larger variance
- When rewards are sparse, most trajectories have $G_t \approx 0$

## 4. Baseline and Variance Reduction

### 4.1 Baseline Trick

A clever trick is: subtracting a baseline $b(s_t)$ from $G_t$ can reduce variance without introducing bias.

**Theorem (Baseline doesn't change expectation)**: For any function $b(s)$ that depends only on state $s$ (not on action $a$):

$$\mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \right] = 0$$

**Proof**: Since $b(s)$ doesn't depend on $a$, it can be pulled outside the expectation:

$$b(s) \cdot \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$

Therefore, the Policy Gradient can be written as:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]$$

### 4.2 Why Does Baseline Reduce Variance?

- $G_t$ might always be positive (e.g., when all rewards are positive), causing all action probabilities to increase
- Subtracting $b(s)$ (such as the average return) makes $G_t - b(s_t)$ both positive and negative
- **Actions better than average are reinforced, actions worse than average are weakened**

### 4.3 Optimal Baseline

**Theorem**: Under the constraint of not changing expectation, the baseline that minimizes variance is the state value function: $b^*(s) = V^\pi(s)$

When $b(s) = V^\pi(s)$, the expectation of $G_t - V^\pi(s_t)$ is exactly the **advantage function**!

## 5. Advantage Function and Actor-Critic

### 5.1 Definition and Intuition of Advantage

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

When using $V^\pi(s)$ as the baseline:

$$\mathbb{E}_\pi \left[ G_t - V^\pi(s_t) \mid s_t, a_t \right] = Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$

Therefore, Policy Gradient with Advantage:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]$$

> **Intuition of Advantage $A(s,a)$**:
> - $V(s)$: "Average" performance in state $s$
> - $Q(s,a)$: Performance when choosing action $a$ in state $s$
> - $A(s,a) = Q(s,a) - V(s)$: How much better is action $a$ than average
>
> $A > 0$: This action is better than average, should increase its probability
> $A < 0$: This action is worse than average, should decrease its probability

### 5.2 Methods for Estimating Advantage

1. **Monte Carlo estimate**: $\hat{A}\_t^{\text{MC}} = G\_t - \hat{V}(s\_t)$ (unbiased but high variance)

2. **TD estimate** (1-step): $\hat{A}\_t^{\text{TD}} = r\_t + \gamma \hat{V}(s\_{t+1}) - \hat{V}(s\_t) = \delta\_t$ (low variance but biased)

3. **n-step estimate**: Between the two

4. **GAE**: Flexibly trades off bias and variance through the $\lambda$ parameter

### 5.3 Actor-Critic Architecture

To estimate $\hat{V}(s)$, we introduce a **Critic** network. Actor-Critic methods simultaneously learn:
- **Actor**: Policy network $\pi_\theta(a\|s)$, outputs action distribution
- **Critic**: Value network $\hat{V}_\phi(s)$, estimates state value

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

**A2C (Advantage Actor-Critic)** core update rules:

**Actor update** (Policy Gradient with Advantage):
$$\theta \leftarrow \theta + \alpha_\theta \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t$$

**Critic update** (Value Function Regression):
$$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi \sum_t \left( \hat{V}_\phi(s_t) - \text{target} \right)^2$$

> **Why Do We Need a Critic?**
> - Provides $\hat{V}(s)$ to compute advantage $\hat{A}_t$
> - Lower variance compared to pure MC (using $G_t$)
> - Can update at every step, no need to wait for episode termination

## 6. Generalized Advantage Estimation (GAE)

GAE provides a method for flexibly trading off between bias and variance in advantage estimation, and is a core component of modern Policy Gradient algorithms (such as PPO).

### 6.1 Definition of GAE

**Definition (Generalized Advantage Estimation)**:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$ is the TD residual, and $\lambda \in [0,1]$ is the decay parameter.

**Theorem**: GAE is equivalent to a weighted sum of n-step Advantages:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}$$

### 6.2 Bias-Variance Tradeoff of $\lambda$ Parameter

| $\lambda$ value | Equivalent form | Bias | Variance |
|-------------|---------|------|------|
| $\lambda = 0$ | $\delta_t$ (TD) | High (depends on $\hat{V}$) | Low |
| $\lambda = 1$ | $G_t - \hat{V}(s_t)$ (MC) | Low | High |
| $\lambda \in (0,1)$ | Weighted average | Medium | Medium |

In practice, $\lambda = 0.95$ or $\lambda = 0.97$ are commonly used choices.

> **Intuitive Understanding of GAE**:
> - $\delta_t$ is the advantage of "estimating remaining value with Critic after one step"
> - GAE is a weighted sum of multi-step $\delta$, with $(\gamma\lambda)^l$ making distant $\delta$ decay exponentially
> - Smaller $\lambda$ relies more on Critic estimation (higher bias but lower variance)
> - Larger $\lambda$ relies more on actual returns (lower bias but higher variance)

### 6.3 Practical Computation of GAE

GAE can be efficiently computed through recursion:

$$\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \hat{A}_{t+1}^{\text{GAE}}$$

Boundary condition: $\hat{A}_T^{\text{GAE}} = 0$. Compute from back to front with complexity $O(T)$.

## 7. Importance Sampling and Off-Policy Policy Gradient

### 7.1 The Problem with On-Policy

Policy Gradient is on-policy: after each update of $\theta$, the distribution of old data differs from the new policy. This leads to:
- Data can only be used once, low sample efficiency
- Need to resample after each update

Importance Sampling (IS) allows us to reuse old data.

### 7.2 Importance Sampling Principle

Using samples from distribution $q(x)$ to estimate expectation under $p(x)$:

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$

where $\rho(x) = \frac{p(x)}{q(x)}$ is called the **importance weight**.

Applying to Policy Gradient, the single-step importance weight:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

**Off-policy Policy Gradient**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \rho_t(\theta) \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$$

### 7.3 Variance Problem

When $\rho_t$ deviates too much from 1, variance increases dramatically. Need to limit the magnitude of policy updates to maintain $\rho_t \approx 1$.

## 8. Trust Region Methods: TRPO and PPO

### 8.1 TRPO: KL-Constrained Optimization

TRPO limits policy updates through KL divergence constraints:

$$\begin{aligned}
\max_\theta \quad & L(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \rho_t(\theta) \hat{A}_t \right] \\
\text{s.t.} \quad & \bar{D}_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) \leq \delta
\end{aligned}$$

TRPO theoretically guarantees monotonic improvement, but requires computing the Hessian of KL divergence, making implementation complex.

### 8.2 PPO: Simplified Trust Region

PPO approximates TRPO's effect through simpler means.

**PPO-Clip objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( \rho_t \hat{A}_t, \, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $\text{clip}(x, a, b) = \max(a, \min(x, b))$, and $\epsilon$ is typically 0.1 or 0.2.

**Intuition of PPO-Clip**:
- When $\hat{A}_t > 0$ (good action): The objective is to increase probability, but clips when $\rho_t > 1+\epsilon$ to prevent excessive increase
- When $\hat{A}_t < 0$ (bad action): The objective is to decrease probability, but clips when $\rho_t < 1-\epsilon$ to prevent excessive decrease

### 8.3 Entropy Bonus

To encourage exploration, PPO typically also adds an entropy bonus:

$$L^{\text{total}}(\theta) = L^{\text{CLIP}}(\theta) + c_1 \cdot H(\pi_\theta)$$

where $H(\pi_\theta) = -\mathbb{E}[\log \pi_\theta(a\|s)]$ is the entropy of the policy.

### 8.4 Complete PPO Algorithm

<!-- tikz-source: rl-ppo-algorithm-en
\begin{algorithm}[H]
\caption{Proximal Policy Optimization (PPO)}
\For{iteration $= 1, 2, \ldots$}{
    \tcp{Data Collection}
    Collect $N$ trajectories using $\pi_\theta$\;
    Compute GAE: $\hat{A}_t$ using $\hat{V}_\phi$\;
    Compute returns: $\hat{R}_t = \hat{A}_t + \hat{V}_\phi(s_t)$\;
    Store $\pi_{\text{old}} = \pi_\theta$ (fixed)\;
    \tcp{Policy Update}
    \For{$k = 1, \ldots, K$}{
        Compute importance ratio: $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$\;
        $L^{\text{CLIP}} = \mathbb{E}[\min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$\;
        $L^{\text{Value}} = \mathbb{E}[(\hat{V}_\phi(s_t) - \hat{R}_t)^2]$\;
        $L^{\text{Entropy}} = -\mathbb{E}[\log \pi_\theta(a_t|s_t)]$\;
        Total objective: $L = L^{\text{CLIP}} - c_1 L^{\text{Value}} + c_2 L^{\text{Entropy}}$\;
        Gradient ascent on $\theta$, gradient descent on $\phi$\;
    }
}
\end{algorithm}
-->
![PPO Algorithm]({{ site.baseurl }}/assets/figures/rl-ppo-algorithm-en.svg)

> **Reasons for PPO's Success**:
> 1. **Simple and efficient**: Only needs first-order optimization, no Hessian computation required
> 2. **Sample efficiency**: Can reuse the same batch of data multiple times ($K$ updates)
> 3. **Stability**: Clip mechanism prevents drastic policy changes
> 4. **Robustness**: Not sensitive to hyperparameters, applicable to various tasks
>
> PPO is currently the most commonly used Policy Gradient algorithm and is the standard choice in RLHF.

## Chapter Summary

**Core Content**:

1. **Policy Gradient Theorem**
   - Provides the analytical form of the objective function gradient: $\nabla_\theta J = \mathbb{E}[\sum_t \nabla \log \pi \cdot G_t]$
   - Log-Derivative Trick is the key to the derivation
   - Environment dynamics are independent of $\theta$, enabling Model-Free learning

2. **Variance Reduction Techniques**
   - Baseline trick: Subtracting $b(s)$ doesn't change expectation but reduces variance
   - Optimal baseline is $V^\pi(s)$
   - Use Advantage $A = Q - V$ instead of $G_t$

3. **Actor-Critic Architecture**
   - Actor (policy network) + Critic (value network)
   - Critic provides $\hat{V}(s)$ to estimate advantage

4. **GAE**
   - $\hat{A}^{\text{GAE}} = \sum_l (\gamma\lambda)^l \delta_{t+l}$
   - $\lambda$ controls bias-variance tradeoff

5. **Trust Region Methods**
   - Importance sampling allows reusing old data, but requires limiting policy changes
   - TRPO: KL-constrained optimization, complex implementation
   - PPO: Clip mechanism, simple and efficient, the preferred choice in practice

<!-- tikz-source: rl-pg-evolution-en
\begin{tikzpicture}[
    node/.style={draw, rounded corners, fill=blue!15, minimum width=2.2cm, minimum height=1.2cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[node, fill=red!20] (r) at (0, 0) {REINFORCE\\{\footnotesize unbiased, high var}};
    \node[node, fill=orange!20] (b) at (3.8, 0) {+ Baseline\\{\footnotesize reduce variance}};
    \node[node, fill=yellow!30] (ac) at (7.6, 0) {Actor-Critic\\{\footnotesize learn Critic}};
    \node[node, fill=green!20] (gae) at (11.4, 0) {+ GAE\\{\footnotesize $\lambda$ tradeoff}};
    \node[node, fill=purple!20] (ppo) at (15.2, 0) {PPO\\{\footnotesize stable, efficient}};

    \draw[arrow] (r) -- node[above, font=\footnotesize] {+Baseline} (b);
    \draw[arrow] (b) -- node[above, font=\footnotesize] {+Critic} (ac);
    \draw[arrow] (ac) -- node[above, font=\footnotesize] {+GAE} (gae);
    \draw[arrow] (gae) -- node[above, font=\footnotesize] {+Clip} (ppo);
\end{tikzpicture}
-->
![Policy Gradient Evolution]({{ site.baseurl }}/assets/figures/rl-pg-evolution-en.svg)

The next article will introduce Model-Based RL and multi-agent learning, including MCTS and AlphaGo/Zero.
