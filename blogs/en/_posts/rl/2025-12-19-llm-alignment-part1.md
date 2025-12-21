---
layout: post
title: "RL Notes (5): RLHF and DPO"
date: 2025-12-19 04:00:00
author: Phonism
tags: [RL, LLM, RLHF, DPO, Alignment]
lang: en
translation: /llm-alignment-part1/
---

This is the fifth article in the reinforcement learning series, entering the domain where LLM and RL combine. This article introduces the RL modeling of LLM alignment, the classic three-stage RLHF approach, and the more concise DPO method.

## Introduction: From Pre-training to Alignment

### Core Problem

Large Language Models (LLMs) acquire powerful language understanding and generation capabilities through massive text pre-training. However, there is a gap between the pre-training objective (predicting the next token) and human-expected behavior:

> **Pre-trained LLMs only learn to "speak like humans," but not to "act according to human expectations."**
>
> How can we make LLMs not only fluent, but also helpful, honest, and harmless?

This is the **LLM Alignment** problem. And reinforcement learning is the core technology to solve this problem.

### Why Do We Need RL?

Supervised Learning (SFT) can make the model imitate high-quality responses, but has limitations:

1. **Limited distribution**: Can only learn response patterns that appear in the training set
2. **Cannot express preferences**: Difficult to distinguish between "good" and "better"
3. **Cannot explore**: Won't try new answering strategies

Reinforcement learning provides a different perspective:
- Model the LLM generation process as an MDP
- Define the reward function using human preferences
- Optimize the policy by maximizing rewards

<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Pre-training
    \node[box, fill=blue!20] (pt) at (0, 0) {Pre-training\\(Next Token Prediction)};

    % SFT
    \node[box, fill=green!20] (sft) at (5, 0) {SFT\\(Imitate Quality)};

    % RLHF
    \node[box, fill=orange!20] (rlhf) at (10, 0) {RL Alignment\\(Optimize Preferences)};

    % Arrows
    \draw[arrow] (pt) -- node[above, font=\small, yshift=5pt] {Language ability} (sft);
    \draw[arrow] (sft) -- node[above, font=\small, yshift=5pt] {Instruction following} (rlhf);

    % Labels
    \node[font=\scriptsize, gray] at (0, -1) {Can speak};
    \node[font=\scriptsize, gray] at (5, -1) {Can answer Qs};
    \node[font=\scriptsize, gray] at (10, -1) {Act as humans expect};
\end{tikzpicture}
</script>

## RL Modeling of LLM Alignment

### State/Action/Reward Definition

Modeling the LLM alignment problem as an RL problem:

> **RL Modeling of LLM**
> - **State** $s_t$: prompt $x$ + generated token sequence $y_{<t} = (y_1, \ldots, y_{t-1})$
> - **Action** $a_t$: next token $y_t$ (vocabulary size $\|\mathcal{V}\| \sim$ 100k)
> - **Policy** $\pi_\theta(a\|s)$: the LLM itself, $\pi_\theta(y_t \| x, y_{<t})$
> - **Trajectory** $\tau$: complete generation sequence $y = (y_1, y_2, \ldots, y_T)$
> - **Reward** $r$: usually only given at the end of the sequence

<script type="text/tikz">
\begin{tikzpicture}[
    state/.style={draw, rounded corners, fill=blue!15, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    action/.style={circle, draw, fill=orange!20, minimum size=0.6cm, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % State sequence
    \node[state] (s0) at (0, 0) {$x$ (prompt)};
    \node[state] (s1) at (3.5, 0) {$x, y_1$};
    \node[state] (s2) at (7, 0) {$x, y_1, y_2$};
    \node[font=\small] at (9.5, 0) {$\cdots$};
    \node[state] (sT) at (12, 0) {$x, y_{1:T}$};

    % Actions
    \node[action] (a1) at (1.75, 1) {$y_1$};
    \node[action] (a2) at (5.25, 1) {$y_2$};
    \node[action] (aT) at (10.5, 1) {$y_T$};

    % Reward
    \node[font=\small, red] at (13.5, 0) {$r(x, y)$};

    % Connections
    \draw[arrow] (s0) -- (a1);
    \draw[arrow] (a1) -- (s1);
    \draw[arrow] (s1) -- (a2);
    \draw[arrow] (a2) -- (s2);
    \draw[arrow, dashed] (s2) -- (9, 0);
    \draw[arrow] (10, 0) -- (aT);
    \draw[arrow] (aT) -- (sT);
    \draw[arrow, red] (sT) -- (13.2, 0);

    % Labels
    \node[font=\scriptsize, gray] at (1.75, 1.6) {$\pi_\theta(y_1|x)$};
    \node[font=\scriptsize, gray] at (5.25, 1.6) {$\pi_\theta(y_2|x,y_1)$};
\end{tikzpicture}
</script>

Characteristics of LLM RL:
- **Huge action space**: Vocabulary typically has 100k+ tokens
- **Deterministic state transitions**: Next state = current state + new token
- **Episode = one complete generation**: From prompt to EOS
- **Sparse rewards**: Reward signal only at the end of the sequence

### Sparse Reward Problem

Typical reward structure for LLM alignment:

$$r_t = \begin{cases} 0 & t < T \\ r_\phi(x, y) & t = T \text{ (end of sequence)} \end{cases}$$

Challenges brought by sparse rewards:
- **Credit assignment difficulty**: How to attribute the final reward to each token?
- **Weak gradient signal**: No learning signal at most time steps
- **Especially difficult for long sequences**: Signal needs to propagate very far (thousands of tokens)

Two approaches to solving sparse rewards:
1. **Sequence-level methods**: Treat the entire sequence as a bandit, update directly with sequence reward (e.g., REINFORCE)
2. **Process rewards**: Train PRM to provide reward signals for intermediate steps

## Three-Stage RLHF

RLHF (Reinforcement Learning from Human Feedback) is the classic approach to LLM alignment, systematized by OpenAI in InstructGPT.

### RLHF Overall Architecture

<script type="text/tikz">
\begin{tikzpicture}[scale=0.9, every node/.style={scale=0.9},
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1cm, align=center},
    data/.style={draw, rounded corners, fill=gray!15, minimum width=2cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Stage 1
    \begin{scope}[shift={(-5, 0)}]
        \node[box, fill=blue!20] (pt) at (0, 2) {Pretrained LLM};
        \node[data] (sft_data) at (0, 0) {High-quality\\dialogue data};
        \node[box, fill=green!20] (sft) at (0, -2) {SFT Model\\$\pi_{\text{ref}}$};

        \draw[arrow] (pt) -- (sft);
        \draw[arrow] (sft_data) -- (sft);

        \node[font=\bfseries] at (0, 3.5) {Stage 1: SFT};
    \end{scope}

    % Stage 2
    \begin{scope}[shift={(0, 0)}]
        \node[box, fill=green!15] (sft2) at (0, 2) {SFT Model};
        \node[data] (pref_data) at (0, 0) {Preference data\\$(x, y_w, y_l)$};
        \node[box, fill=orange!20] (rm) at (0, -2) {Reward Model\\$r_\phi(x, y)$};

        \draw[arrow] (sft2) -- (rm);
        \draw[arrow] (pref_data) -- (rm);

        \node[font=\bfseries] at (0, 3.5) {Stage 2: RM};
    \end{scope}

    % Stage 3
    \begin{scope}[shift={(5.5, 0)}]
        \node[box, fill=green!15] (ref) at (-1.8, 2) {$\pi_{\text{ref}}$};
        \node[box, fill=orange!15] (rm2) at (1.8, 2) {$r_\phi$};
        \node[box, fill=purple!20] (ppo) at (0, 0) {PPO Training};
        \node[box, fill=red!20] (final) at (0, -2) {Aligned Model\\$\pi_\theta$};

        \draw[arrow] (ref) -- (ppo);
        \draw[arrow] (rm2) -- (ppo);
        \draw[arrow] (ppo) -- (final);

        \node[font=\bfseries] at (0, 3.5) {Stage 3: PPO};
    \end{scope}

    % Connection arrows
    \draw[arrow, dashed, gray] (-3, -2) -- (-2, 2);
    \draw[arrow, dashed, gray] (2, -2) -- (4, 2);
\end{tikzpicture}
</script>

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune the pre-trained model with high-quality dialogue data:

$$L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[ \log \pi_\theta(y|x) \right] = -\mathbb{E} \left[ \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t}) \right]$$

Role of SFT:
- Make the model learn the basic format of "instruction following"
- Provide the starting point for RL (reference model $\pi_{\text{ref}}$)
- Filter out low-quality patterns from pre-training

### Stage 2: Reward Model Training

Learn the Reward Model from human preference data.

> **Preference data**: For prompt $x$, human annotators compare two responses and give a preference: $y_w \succ y_l$ ($y_w$ is preferred over $y_l$).

#### Bradley-Terry Model

> **Bradley-Terry Model**
>
> Assumes human preferences follow the Bradley-Terry model—preference probability is determined by "ability difference":
>
> $$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + e^{-(r(x, y_w) - r(x, y_l))}}$$
>
> where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function, and $r(x, y)$ is the "score" of the response.

Intuition of the Bradley-Terry model:
- When reward difference = 0, preference probability = 0.5 (cannot distinguish)
- The larger the reward difference, the closer the preference probability to 1 (more certain)
- The model assumes preferences are probabilistic comparisons based on "intrinsic quality scores"

#### Reward Model Training

The training objective of the Reward Model is to maximize the likelihood of preference data:

$$L_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

This is a **binary classification problem**: given $(y_w, y_l)$, predict which is better.

Reward Model architecture choices:
- Usually initialized from the SFT model
- Remove the language model head, add a scalar output head
- Input $(x, y)$, output scalar $r_\phi(x, y) \in \mathbb{R}$

### Stage 3: PPO Fine-tuning

Use the Reward Model to provide reward signals, optimize the policy with PPO.

> **RLHF Optimization Objective**
>
> $$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$
>
> where $\beta > 0$ is the KL regularization coefficient.

#### Role of KL Regularization

The KL regularization term $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ is crucial:

1. **Prevent Reward Hacking**:
   - The Reward Model is an imperfect proxy
   - Unconstrained optimization will find ways to "fool" the RM
   - For example: generating specific patterns to get high scores, but actual quality is poor

2. **Maintain generation quality**:
   - The SFT model already has good language capabilities
   - KL constraint prevents drifting too far and causing fluency degradation

3. **Stabilize training**:
   - Constrain the optimization space, avoid policy collapse
   - Provide regularization effect

<script type="text/tikz">
\begin{tikzpicture}[
    arrow/.style={->, thick, >=stealth}
]
    % Axes
    \draw[arrow] (-0.5, 0) -- (8, 0) node[right] {$\text{KL}(\pi_\theta \| \pi_{\text{ref}})$};
    \draw[arrow] (0, -0.5) -- (0, 5) node[above] {$\mathbb{E}[r_\phi]$};

    % Curve
    \draw[thick, blue, domain=0.2:7, samples=100] plot (\x, {4 - 0.8*(\x-3)^2/9 + 0.5*ln(\x)});

    % Optimal point
    \fill[red] (2.5, 3.8) circle (3pt);
    \node[font=\small, red] at (2.5, 4.3) {Optimal tradeoff};

    % Region labels
    \node[font=\scriptsize, align=center] at (1, 2) {KL too small\\Limited improvement};
    \node[font=\scriptsize, align=center] at (6, 2) {KL too large\\Reward Hacking};

    % Beta's effect
    \draw[dashed, gray] (0, 3.8) -- (2.5, 3.8) -- (2.5, 0);
\end{tikzpicture}
</script>

#### PPO Update Process

Specific steps of PPO in RLHF:

```
Input: SFT model π_ref, Reward Model r_φ, KL coefficient β
Initialize π_θ ← π_ref, Critic V_ψ

for each iteration:
    // Sampling
    Sample x ∼ D from prompt distribution
    Generate response with current policy y ∼ π_θ(·|x)

    // Compute reward
    Compute RM reward: r^RM = r_φ(x, y)
    Compute KL penalty: r^KL_t = -β log [π_θ(y_t|x, y_{<t}) / π_ref(y_t|x, y_{<t})]
    Total reward: r_t = r^KL_t + 𝟙_{t=T} · r^RM

    // GAE computation
    Compute advantage Â_t using Critic V_ψ

    // PPO update
    Update π_θ with PPO-Clip objective
    Update V_ψ with TD target
```

> **Important Note**: Models needed for RLHF:
> 1. $\pi_\theta$: policy being trained (Active Model)
> 2. $\pi_{\text{ref}}$: reference model (frozen)
> 3. $r_\phi$: Reward Model (frozen)
> 4. $V_\psi$: Critic network
>
> Total 4 large models, huge memory overhead! This is the problem that methods like DPO and GRPO try to solve.

## Direct Preference Optimization (DPO)

DPO is a simplified method that bypasses the Reward Model and PPO, proposed by Rafailov et al. 2023.

### DPO Motivation

Problems with RLHF + PPO:
- **Large model overhead**: Need to maintain 4 models
- **High sampling cost**: Online generation with large models is expensive
- **Complex implementation**: PPO is hyperparameter-sensitive, needs fine-tuning
- **Training instability**: RL training is prone to collapse

> **DPO's core question**: Can we optimize directly on preference data $(x, y_w, y_l)$, as simple as supervised learning?

The answer is yes! Key insight: The RL problem with KL regularization has a **closed-form solution**.

### DPO Loss Formula

> **DPO Loss**
>
> $$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right) \right]$$

### Complete DPO Derivation

> **DPO Equivalence Theorem**: DPO Loss is equivalent to the RLHF objective at the optimal solution.

**Proof**: The derivation has 5 key steps.

**Step 1: RLHF Objective Expansion**

RLHF optimization objective:

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x, y) \right] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

Expand the KL divergence:

$$= \mathbb{E}_{y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

**Step 2: Introduce Partition Function $Z(x)$**

To make the optimal policy a valid probability distribution, define the partition function:

$$Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$$

$Z(x)$ is a normalization constant that only depends on $x$ (not on the policy being optimized).

**Step 3: Closed-form Solution for Optimal Policy**

The KL-regularized RL problem has a closed-form solution:

> **Optimal Policy Lemma for KL-regularized RL**
>
> The optimal solution to the objective $\max_\pi \mathbb{E}_{y \sim \pi}[r(y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$ is:
>
> $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$$

This is a constrained optimization problem ($\pi$ needs to be a probability distribution). Intuition: The optimal policy is the reference policy reweighted by $\exp(r/\beta)$. Higher reward, higher probability boost.

**Step 4: Solve for Reward from Optimal Policy**

Key step: Solve for reward from the optimal policy.

Take logarithm:

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) - \log Z(x) + \frac{r(x,y)}{\beta}$$

Rearrange to get:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

> **Core Insight**: Reward can be expressed using the log-ratio of policies! Although there's a $\log Z(x)$ term, it only depends on $x$ and will cancel out in pairwise comparisons.

**Step 5: Substitute into Bradley-Terry Model, $Z(x)$ Cancels**

Substitute the reward expression into the Bradley-Terry model:

$$\begin{align}
P(y_w \succ y_l) &= \sigma(r(x, y_w) - r(x, y_l)) \\
&= \sigma\left( \beta \left[ \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right)
\end{align}$$

The $\beta \log Z(x)$ terms cancel out!

Maximize the log-likelihood of preference data, replace $\pi^*$ with $\pi_\theta$, and we get the DPO Loss.

<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=3.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    \node[box] (rlhf) at (0, 3) {RLHF Objective\\$\max \mathbb{E}[r] - \beta \cdot \text{KL}$};
    \node[box] (opt) at (0, 1) {Optimal Policy Closed Form\\$\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$};
    \node[box] (reward) at (0, -1) {Solve for reward\\$r = \beta \log \frac{\pi^*}{\pi_{\text{ref}}} + \beta \log Z$};
    \node[box, fill=green!20] (dpo) at (0, -3) {DPO Loss\\$Z(x)$ cancels};

    \draw[arrow] (rlhf) -- node[right, font=\small] {KL-RL closed form} (opt);
    \draw[arrow] (opt) -- node[right, font=\small] {Take log} (reward);
    \draw[arrow] (reward) -- node[right, font=\small] {Substitute into BT} (dpo);
\end{tikzpicture}
</script>

> **DPO's Core Insights**:
> 1. The KL-regularized RL problem has a closed-form solution, the optimal policy is exponential reweighting of the reference policy
> 2. We can solve for the implicit reward from the optimal policy
> 3. The partition function $Z(x)$ cancels out in pairwise comparisons—this is the key to why DPO works
> 4. The final form only needs to compute log-probability, as simple as supervised learning

### Intuitive Understanding of DPO

Define **implicit reward**:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

DPO Loss can be written as:

$$L_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) \right]$$

Intuition:
- $\hat{r}_\theta(x, y_w) > \hat{r}_\theta(x, y_l)$: $y_w$ has higher implicit reward, loss decreases
- Training process increases $y_w$'s probability relative to $\pi_{\text{ref}}$, decreases $y_l$'s probability
- $\beta$ controls the scale of "how much to deviate from the reference policy"

### DPO vs RLHF Comparison

| Feature | RLHF + PPO | DPO |
|------|------------|-----|
| Need Reward Model | Yes | No |
| Need Critic network | Yes | No |
| Training method | Online sampling | Offline training |
| Number of models | 4 | 2 |
| Implementation complexity | High | Low |
| Hyperparameter sensitivity | High | Low |
| Exploration ability | Yes | No |
| Applicable scenarios | Complex tasks | Simple alignment |

DPO limitations:
- **No exploration**: Completely offline, can only optimize within the distribution of existing preference data
- **Coarse pairwise signal**: Only knows which is better, not how much better
- **Limited improvement on difficult tasks**: Not as effective as RL on tasks like math and code that require exploration

## Chapter Summary

1. **RL Modeling of LLM Alignment**: State = prompt + generated tokens, Action = next token, sparse reward only given at end of sequence

2. **Three-Stage RLHF**:
   - Stage 1 (SFT): Supervised fine-tuning, learn instruction following
   - Stage 2 (RM): Train Reward Model from preference data (Bradley-Terry model)
   - Stage 3 (PPO): Use RM to provide rewards, PPO optimization, KL regularization prevents reward hacking

3. **DPO**:
   - Leverages KL-RL closed-form solution, bypasses RM and PPO
   - Optimize directly on preference data, as simple as supervised learning
   - Only needs 2 models ($\pi_\theta$ and $\pi_{\text{ref}}$)
   - Limitations: No exploration ability, limited improvement on difficult tasks

<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3.5cm, minimum height=2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (rlhf) at (0, 0) {
        \textbf{RLHF}\\[3pt]
        (2020-2022)\\[3pt]
        \scriptsize Need RM + Critic\\[-1pt]
        \scriptsize Complex impl\\[-1pt]
        \scriptsize 4 models
    };

    \node[box, fill=green!20] (dpo) at (6, 0) {
        \textbf{DPO}\\[3pt]
        (2023)\\[3pt]
        \scriptsize Offline training\\[-1pt]
        \scriptsize No exploration\\[-1pt]
        \scriptsize 2 models
    };

    \draw[arrow] (rlhf) -- node[above, font=\small] {Simplify} (dpo);
\end{tikzpicture}
</script>

The next article will introduce more advanced methods such as GRPO, KL estimators, PRM, and Long CoT RL, which attempt to restore online exploration capabilities while maintaining DPO's simplicity.
