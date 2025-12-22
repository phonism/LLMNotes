---
layout: post
title: "RL Notes (6): GRPO and Long CoT RL"
date: 2025-12-19 08:00:00
author: Qi Lu
tags: [RL, LLM, GRPO, PRM, Long-CoT, Alignment]
lang: en
translation: /llm-alignment-part2/
---

This is the sixth and final article in the Reinforcement Learning series. This article introduces GRPO (online RL without a Critic), KL divergence estimators, On-Policy Distillation, Process Reward Models (PRM), and the challenges and methods of Long CoT RL.

## GRPO: Group Relative Policy Optimization

GRPO is a method proposed by DeepSeek that sits between PPO and DPO: it retains online exploration capabilities but doesn't require a Critic network.

### Motivation for GRPO

- **PPO's problem**: Requires a Critic network, large memory overhead (an additional large model)
- **DPO's problem**: Completely offline, lacks exploration, limited improvement on difficult tasks

GRPO's approach: **Use group-relative rewards to replace the Critic**, achieving "online RL without a Critic".

### Group-Normalized Advantage

> **GRPO Advantage**
>
> For prompt $x$, sample a group of responses $\\{y_1, \ldots, y_G\\}$, compute their respective rewards $\\{R_1, \ldots, R_G\\}$, then:
>
> $$\hat{A}_i = \frac{R_i - \bar{R}}{\text{Std}(R) + \epsilon}$$
>
> where $\bar{R} = \frac{1}{G}\sum_i R_i$ is the group mean, $\text{Std}(R)$ is the group standard deviation.

<!-- tikz-source: rl-grpo-sampling-en
\begin{tikzpicture}[
    sample/.style={circle, draw, minimum size=0.6cm, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % Prompt
    \node[draw, rounded corners, fill=blue!20, minimum width=2cm] (prompt) at (0, 0) {Prompt $x$};

    % Generate multiple responses
    \node[sample, fill=green!30] (y1) at (3, 1.5) {$y_1$};
    \node[sample, fill=green!20] (y2) at (3, 0.5) {$y_2$};
    \node[sample, fill=red!20] (y3) at (3, -0.5) {$y_3$};
    \node[sample, fill=red!30] (y4) at (3, -1.5) {$y_4$};

    \draw[arrow] (prompt) -- (y1);
    \draw[arrow] (prompt) -- (y2);
    \draw[arrow] (prompt) -- (y3);
    \draw[arrow] (prompt) -- (y4);

    % Rewards
    \node[font=\scriptsize] at (4.5, 1.5) {$R_1 = 0.8$};
    \node[font=\scriptsize] at (4.5, 0.5) {$R_2 = 0.6$};
    \node[font=\scriptsize] at (4.5, -0.5) {$R_3 = 0.3$};
    \node[font=\scriptsize] at (4.5, -1.5) {$R_4 = 0.1$};

    % Normalization
    \node[font=\small] at (7, 0) {$\bar{R} = 0.45$};

    % Advantage
    \node[font=\scriptsize, green!60!black] at (9, 1.5) {$\hat{A}_1 > 0$ \checkmark};
    \node[font=\scriptsize, green!60!black] at (9, 0.5) {$\hat{A}_2 > 0$ \checkmark};
    \node[font=\scriptsize, red] at (9, -0.5) {$\hat{A}_3 < 0$ $\times$};
    \node[font=\scriptsize, red] at (9, -1.5) {$\hat{A}_4 < 0$ $\times$};

    % Explanation
    \node[font=\small, align=center] at (5, -3) {Group-relative comparison:\\boost above mean, suppress below mean};
\end{tikzpicture}
-->
![GRPO Group Sampling]({{ site.baseurl }}/assets/figures/rl-grpo-sampling-en.svg)

Advantages of group normalization:
1. **No Critic needed**: Use group mean to replace value function estimation
2. **Baseline effect**: Mean subtraction reduces variance
3. **Scale normalization**: Standard deviation normalization makes advantage scale stable
4. **Relative comparison**: Focus on relative quality under the same prompt

### GRPO Objective Function

> **GRPO Objective**
>
> $$L_{\text{GRPO}} = \mathbb{E}_{x} \left[ \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min \left( \rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$
>
> where $\rho_{i,t} = \frac{\pi_\theta(y_{i,t}\|x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\|x, y_{i,<t})}$ is the importance sampling ratio.

Key differences between GRPO and PPO:
- PPO: $\hat{A}_t$ is computed by GAE, requires a Critic
- GRPO: $\hat{A}_i$ is constant for the entire sequence, obtained from group normalization

### GRPO Practical Tips

1. **Clip-Higher**: Upper bound can be more lenient (e.g., $1+0.28$ instead of $1+0.2$), allowing good responses to improve more significantly

2. **Dynamic Sampling**: Filter prompts where all responses are correct or all wrong (advantage is undefined when variance is 0)

3. **Length penalty**: Prevent generating overly long responses
   $$R_i = r_\phi(x, y_i) - \lambda \cdot |y_i|$$

4. **KL as Loss**: KL penalty as a separate loss term, rather than putting it in the reward
   $$L = -L_{\text{GRPO}} + \lambda_{\text{KL}} \cdot \mathbb{E}[\text{KL}(\pi_\theta \| \pi_{\text{ref}})]$$

## KL Divergence Estimation: k1, k2, k3

KL divergence $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ cannot be computed exactly (requires enumerating all possible sequences) and needs Monte Carlo estimation.

### Definition of KL Divergence

For two distributions $p$ and $q$:

$$\text{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[ \log \frac{p(x)}{q(x)} \right]$$

In the LLM scenario, $p = \pi_\theta$, $q = \pi_{\text{ref}}$, sampling from $\pi_\theta$.

### k1 Estimator: Direct Estimation

> **k1 Estimator**
>
> Define ratio $r = \frac{\pi_{\text{ref}}(y\|x)}{\pi_\theta(y\|x)}$, then:
>
> $$k_1 = -\log r = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

Properties:
- **Unbiased**: $\mathbb{E}_{y \sim \pi_\theta}[k_1] = \text{KL}(\pi_\theta \| \pi_{\text{ref}})$
- **High variance**: When $\pi_\theta$ and $\pi_{\text{ref}}$ differ significantly, variance is very large

Usage: Usually put in reward
$$r_t^{\text{RL}} = r_t^{\text{RM}} - \beta \cdot k_1^{(t)}$$

### k2 Estimator: Squared Form

> **k2 Estimator**
>
> $$k_2 = \frac{1}{2}(\log r)^2 = \frac{1}{2} \left( \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)} \right)^2$$

Properties:
- **Biased**: $\mathbb{E}[k_2] \neq \text{KL}$
- **Gradient equivalent**: $\nabla_\theta \mathbb{E}[k_2] = \nabla_\theta \text{KL}$
- **Smoother**: Squared form is smoother around $r=1$

Usage: Suitable as a loss term (because the gradient is correct)

### k3 Estimator: Control Variate

> **k3 Estimator**
>
> $$k_3 = (r - 1) - \log r = \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)} - 1 - \log \frac{\pi_{\text{ref}}(y|x)}{\pi_\theta(y|x)}$$

> **k3 Properties Theorem**: k3 is an unbiased and low-variance KL estimator.

**Proof**:

Unbiasedness:
$$\begin{align}
\mathbb{E}_{y \sim \pi_\theta}[k_3] &= \mathbb{E}[(r-1)] - \mathbb{E}[\log r] \\
&= \underbrace{\mathbb{E}\left[\frac{\pi_{\text{ref}}}{\pi_\theta}\right] - 1}_{= 0} + \mathbb{E}\left[\log \frac{\pi_\theta}{\pi_{\text{ref}}}\right] \\
&= \text{KL}(\pi_\theta \| \pi_{\text{ref}})
\end{align}$$

where $\mathbb{E}_{y \sim \pi_\theta}\left[\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}\right] = \sum_y \pi_\theta(y) \cdot \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} = \sum_y \pi_{\text{ref}}(y) = 1$.

Low variance: The $(r-1)$ term is a control variate with zero expectation, negatively correlated with $\log r$, reducing variance.

### Comparison of Three Estimators

| Estimator | Formula | Bias | Variance | Recommended Usage |
|--------|------|------|------|----------|
| k1 | $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | Unbiased | High | KL in reward |
| k2 | $\frac{1}{2}(\log \frac{\pi_{\text{ref}}}{\pi_\theta})^2$ | Biased | Low | KL as loss |
| k3 | $\frac{\pi_{\text{ref}}}{\pi_\theta} - 1 - \log \frac{\pi_{\text{ref}}}{\pi_\theta}$ | Unbiased | Low | KL as loss |

**Two Usage Patterns for KL**:

1. **KL in Reward** (classical RLHF):
   - Token reward minus $\beta \cdot k_1$
   - Then update with PPO-Clip
   - Advantage: Directly affects each step's decision

2. **KL as Loss** (GRPO, etc.):
   - Total loss = $-L_{\text{RL}} + \lambda \cdot \mathbb{E}[k_3]$
   - Advantage: More stable, lower variance

## On-Policy Distillation

On-Policy Distillation is an important advancement in LLM post-training in recent years, combining the advantages of RL and knowledge distillation.

### Motivation: Distribution Shift Problem in Off-policy Distillation

Traditional knowledge distillation (SFT on teacher data) is **off-policy**:

> **The student learns from teacher-generated data, but the states visited during inference may be completely different from those during training.**

This leads to **compounding errors** — the student hasn't seen its own mistakes during training, so once it deviates from the teacher's trajectory, it continues to deteriorate.

### Comparison of Three Post-Training Paradigms

| Method | Sampling Source | Reward Density | Characteristics |
|------|----------|----------|------|
| SFT (Supervised Fine-Tuning) | Teacher data (Off-policy) | Dense | Distribution constrained |
| RL (Reinforcement Learning) | Self-sampling (On-policy) | Sparse | High search cost |
| On-Policy Distillation | **Self-sampling** | **Dense** | Combines both advantages |

Core idea of On-Policy Distillation:
- **Student samples itself** (On-policy): Avoids distribution shift
- **Teacher scores token-by-token** (Dense reward): Provides dense supervision signal

### Reverse KL Loss

On-Policy Distillation uses **reverse KL divergence** as the loss function:

$$L_{\text{OPD}} = D_{\text{KL}}(\pi_\theta \| \pi_{\text{teacher}}) = \mathbb{E}_{y \sim \pi_\theta}\left[ \sum_t \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{teacher}}(y_t | x, y_{<t})} \right]$$

**Forward KL vs Reverse KL**:
- Forward KL $D_{\text{KL}}(\pi_{\text{teacher}} \| \pi_\theta)$: mode covering, student tries to cover all modes of the teacher
- Reverse KL $D_{\text{KL}}(\pi_\theta \| \pi_{\text{teacher}})$: mode seeking, student focuses on teacher's high-probability regions

Reverse KL is more suitable for distillation scenarios — let the student imitate the teacher in states it visits, rather than trying to cover all teacher behaviors.

### KDRL: Combining Knowledge Distillation and Reinforcement Learning

KDRL (Knowledge Distillation + Reinforcement Learning) unifies teacher supervision and reward-driven self-exploration into joint optimization:

$$\mathcal{J}_{\text{KDRL}}(\theta) = \underbrace{\mathcal{J}_{\text{GRPO}}(\theta)}_{\text{reward-driven}} - \beta \cdot \underbrace{D_{\text{KL}}^{k_2}(\pi_\theta \| \pi_{\text{teacher}})}_{\text{teacher supervision}}$$

Key design choices:
1. **KL estimator choice**: $k_2$ is superior to $k_3$, as $k_2$ provides unbiased gradient estimation
2. **Coefficient annealing**: Gradually transition from strong imitation (large $\beta$) to reward-driven (small $\beta$)
3. **Reward-guided masking**: Apply KD only on low-reward samples, reducing output length while maintaining accuracy

### Efficiency Advantages

Efficiency improvements of On-Policy Distillation (using math reasoning tasks as example):
- Compared to RL: 7-10x faster training, 50-100x overall computational efficiency improvement
- Compared to offline distillation: 9-30x reduction in computational cost
- Qwen3 experiments: Distillation significantly outperforms RL, GPU hours only ~1/10 of RL

Fundamental reason for efficiency improvement:

> **The dense token-level signal provided by the teacher contains much more information than RL's sparse sequence-level reward.**
>
> Most of RL's computation is spent on searching (exploring policy space), while distillation directly uses teacher knowledge to guide the student to make correct choices at critical "fork points".

## Process Reward Model (PRM)

PRM provides process-level supervision, transforming sparse terminal rewards into dense step-level rewards.

### ORM vs PRM

> **ORM and PRM**
> - **ORM (Outcome Reward Model)**: Only looks at final result
>   - Input: $(x, y)$
>   - Output: Correctness score of final answer
>
> - **PRM (Process Reward Model)**: Evaluates each intermediate step
>   - Input: $(x, y_{\leq t})$
>   - Output: Correctness score up to step $t$

<!-- tikz-source: rl-orm-vs-prm-en
\begin{tikzpicture}[
    stepbox/.style={draw, rounded corners, minimum width=1.5cm, minimum height=0.6cm, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % ORM
    \begin{scope}[shift={(-4, 0)}]
        \node[stepbox, fill=blue!20] (s1) at (0, 0) {Step 1};
        \node[stepbox, fill=blue!20] (s2) at (2, 0) {Step 2};
        \node[stepbox, fill=blue!20] (s3) at (4, 0) {Step 3};
        \node[stepbox, fill=green!30] (ans) at (6, 0) {Answer};

        \draw[arrow] (s1) -- (s2);
        \draw[arrow] (s2) -- (s3);
        \draw[arrow] (s3) -- (ans);

        \node[font=\small, red] at (6, -0.8) {$r = 1$};
        \node[font=\bfseries] at (3, 1.2) {ORM: Only evaluates final answer};
    \end{scope}

    % PRM
    \begin{scope}[shift={(-4, -3)}]
        \node[stepbox, fill=green!30] (s1) at (0, 0) {Step 1};
        \node[stepbox, fill=green!30] (s2) at (2, 0) {Step 2};
        \node[stepbox, fill=red!30] (s3) at (4, 0) {Step 3};
        \node[stepbox, fill=red!30] (ans) at (6, 0) {Answer};

        \draw[arrow] (s1) -- (s2);
        \draw[arrow] (s2) -- (s3);
        \draw[arrow] (s3) -- (ans);

        \node[font=\scriptsize, green!60!black] at (0, -0.7) {$r_1 = 1$ \checkmark};
        \node[font=\scriptsize, green!60!black] at (2, -0.7) {$r_2 = 1$ \checkmark};
        \node[font=\scriptsize, red] at (4, -0.7) {$r_3 = 0$ $\times$};
        \node[font=\scriptsize, red] at (6, -0.7) {$r_4 = 0$ $\times$};
        \node[font=\bfseries] at (3, 1.2) {PRM: Evaluates each step};
    \end{scope}
\end{tikzpicture}
-->
![ORM vs PRM Comparison]({{ site.baseurl }}/assets/figures/rl-orm-vs-prm-en.svg)

### Advantages of PRM

1. **Clearer credit assignment**: Feedback at every step, know which step went wrong
2. **Faster convergence for long-chain reasoning**: Dense reward is easier to learn than sparse reward
3. **Supports early stopping**: If intermediate step scores keep dropping, can truncate and resample
4. **Supports Best-of-N**: Use cumulative PRM score to select the best reasoning path

### PRM Training

PRM training data sources:
- **Human annotation**: Experts annotate correctness of each reasoning step
- **Automatic annotation**: Infer intermediate steps from final answer correctness
- **MCTS exploration**: Search to discover correct/incorrect branch points

PRM reward for RL:

$$r_t = \text{PRM}(x, y_{\leq t}) - \text{PRM}(x, y_{\leq t-1})$$

i.e., the "marginal contribution" of each step.

## Long CoT RL

RL training for long Chain-of-Thought sequences (Long CoT) faces unique challenges. With the emergence of reasoning models like o1 and DeepSeek-R1, this has become an important research direction.

### Challenges of Long Sequence RL

1. **Variance explosion**: Token-level importance sampling weights accumulate, causing exponential variance growth
   $$\prod_{t=1}^{T} \frac{\pi_\theta(y_t|s_t)}{\pi_{\text{old}}(y_t|s_t)} \approx e^{\sum_t \delta_t}$$
   When $T$ is large (thousands of tokens), the variance of this product explodes.

2. **Policy shift**: Long sequences make $\pi_\theta$ and $\pi_{\text{old}}$ diverge more

3. **Sparse reward harder**: Only the final answer has feedback, signal must propagate thousands of steps

<!-- tikz-source: rl-is-variance-en
\begin{tikzpicture}
    \begin{axis}[
        width=10cm, height=5cm,
        xlabel={Sequence length $T$},
        ylabel={IS weight variance (log scale)},
        domain=1:100,
        samples=50,
        ymode=log,
        grid=major,
        legend pos=north west
    ]
        \addplot[thick, blue] {exp(0.1*x)};
        \addlegendentry{Token-level IS (exponential)}
        \addplot[thick, red, dashed] {1 + 0.1*x};
        \addlegendentry{Sequence-level IS (linear)}
    \end{axis}
\end{tikzpicture}
-->
![IS Weight Variance]({{ site.baseurl }}/assets/figures/rl-is-variance-en.svg)

### GSPO: Sequence-level IS

GSPO (Group Sequence Policy Optimization) uses sequence-level importance sampling:

> **GSPO Sequence-level IS**
>
> Define length-normalized sequence-level IS weight:
>
> $$w_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \right)^{1/|y_i|} = \exp\left( \frac{1}{|y_i|} \sum_t \log \frac{\pi_\theta(y_{i,t}|s_{i,t})}{\pi_{\text{old}}(y_{i,t}|s_{i,t})} \right)$$
>
> After length normalization then clip, all tokens share the same weight.

GSPO advantages:
- Avoids variance explosion from token-level weight multiplication
- Length normalization makes sequences of different lengths comparable
- Maintains PPO-Clip's stability

### CISPO: Clipped IS-weight

CISPO (Clipped Importance Sampling Policy Optimization) adopts a different strategy:

- Maintains GRPO's group normalization
- Returns to token-level REINFORCE
- Clip IS weights (not loss):

$$\hat{\rho}_{i,t} = \text{clip}\left(\frac{\pi_\theta(y_{i,t}|s_{i,t})}{\pi_{\text{old}}(y_{i,t}|s_{i,t})}, 1-\epsilon, 1+\epsilon\right)$$

### Kimi k1.5 Technical Points

Kimi k1.5's long CoT RL recipe:
- **Ultra-long context**: 128k context direct RL
- **Mirror Descent update**: More conservative policy updates
- **Partial rollout**: Don't need complete generation, truncate and use value estimation
- **Asynchronous train/infer**: Separate training and inference to improve efficiency
- **Repetition detection + early stopping**: Truncate when loop generation is detected
- **Length penalty**: Encourage concise reasoning

**Long2Short RL** (long to short distillation):
- Long CoT model as teacher
- Train short CoT student
- Reward = correctness + token count penalty
- Goal: both correct and concise

### DeepSeek-V3.2 Improvements

DeepSeek-V3.2's improvements to GRPO (core idea: make off-policy training behave as close to on-policy as possible):

1. **Unbiased KL Estimate**: Use IS ratio to correct k3's off-policy bias
   $$\hat{\text{KL}} = \mathbb{E}_{\pi_{\text{old}}}\left[ \frac{\pi_\theta}{\pi_{\text{old}}} \cdot k_3 \right]$$

2. **Off-Policy Sequence Masking**: Sequences with negative advantage and excessive KL divergence are masked

3. **Keep Routing** (MoE-specific): Maintain expert routing paths from sampling time

4. **Keep Sampling Mask**: Maintain top-p/top-k truncation mask

## Token-level vs Sequence-level Objectives

### First-Order Approximation Theory

Token-level objectives (like REINFORCE, GRPO) are first-order approximations of sequence-level objectives.

Let $\frac{\pi_\theta(y_t\|s_t)}{\pi_{\theta_{\text{old}}}(y_t\|s_t)} = 1 + \delta_t$, where $\delta_t$ is a small quantity, then:

$$\prod_t (1 + \delta_t) \approx 1 + \sum_t \delta_t \quad \text{(first-order Taylor expansion)}$$

Therefore, when policy changes are small:

$$\nabla \mathcal{J}^{\text{seq}} \approx \nabla \mathcal{J}^{\text{token}}$$

Condition for validity: $\pi_\theta \approx \pi_{\theta_{\text{old}}}$, i.e., each update step is sufficiently small.

### Training-Inference Discrepancy

In practice, sampling distribution $\mu$ may differ from training-time $\pi_{\theta_{\text{old}}}$:

$$\frac{\pi_\theta(y)}{\mu(y)} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y)}{\mu(y)}}_{\text{train-infer gap}} \times \underbrace{\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)}}_{\text{policy staleness}}$$

Both factors can cause first-order approximation to fail:
- **Train-infer gap**: Different sampling strategies between training and inference (e.g., temperature, top-p)
- **Policy staleness**: In asynchronous training, data comes from old policy

## Chapter Summary

1. **GRPO**: Uses group-relative rewards to replace Critic, achieving online RL without a Critic
   - Group normalization: $\hat{A}_i = \frac{R_i - \bar{R}}{\text{Std}(R)}$
   - Retains exploration capability, lower memory overhead than PPO

2. **KL estimators**:
   - k1 (unbiased high variance): suitable for KL in reward
   - k2 (biased low variance): suitable for KL as loss
   - k3 (unbiased low variance): recommended for KL as loss

3. **On-Policy Distillation**:
   - Combines RL exploration and distillation efficiency
   - Reverse KL lets student learn on its own distribution
   - 10-100x more efficient than pure RL

4. **PRM**: Process reward model, provides dense reward signal
   - Evaluates each step, solves credit assignment problem of sparse rewards
   - Supports Best-of-N selection and early stopping

5. **Long CoT RL**:
   - GSPO/CISPO solve long sequence variance explosion
   - Sequence-level IS replaces token-level IS
   - Kimi, DeepSeek practical techniques

<!-- tikz-source: rl-alignment-evolution-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Method evolution
    \node[box, fill=blue!20] (rlhf) at (0, 0) {RLHF\\(2020-2022)};
    \node[box, fill=green!20] (dpo) at (4, 0) {DPO\\(2023)};
    \node[box, fill=orange!20] (grpo) at (8, 0) {GRPO\\(2024)};
    \node[box, fill=purple!20] (longcot) at (12, 0) {Long CoT RL\\(2024-2025)};

    \draw[arrow] (rlhf) -- node[above, font=\scriptsize] {Simplify} (dpo);
    \draw[arrow] (dpo) -- node[above, font=\scriptsize] {Online} (grpo);
    \draw[arrow] (grpo) -- node[above, font=\scriptsize] {Long seq} (longcot);

    % Feature labels
    \node[font=\scriptsize, gray, align=center] at (0, -1) {Need RM + Critic\\Complex impl};
    \node[font=\scriptsize, gray, align=center] at (4, -1) {Offline training\\No exploration};
    \node[font=\scriptsize, gray, align=center] at (8, -1) {No Critic\\Group normalization};
    \node[font=\scriptsize, gray, align=center] at (12, -1) {Sequence-level IS\\Variance control};
\end{tikzpicture}
-->
![LLM Alignment Evolution]({{ site.baseurl }}/assets/figures/rl-alignment-evolution-en.svg)

## Series Summary

This series consists of six articles, systematically introducing the complete knowledge system from RL fundamentals to LLM applications:

1. **RL Fundamentals**: MDP, trajectories, value functions, RL objectives
2. **Value-Based RL**: Bellman equations, Q-Learning, DQN
3. **Policy-Based RL**: Policy Gradient, REINFORCE, Actor-Critic, PPO
4. **Model-Based RL & MARL**: World Model, MCTS, AlphaZero, Self-Play
5. **LLM Alignment (Part 1)**: RLHF, Bradley-Terry, DPO
6. **LLM Alignment (Part 2)**: GRPO, KL estimators, PRM, Long CoT RL

I hope this series helps readers establish a complete RL knowledge framework and understand its core role in modern LLM alignment.
