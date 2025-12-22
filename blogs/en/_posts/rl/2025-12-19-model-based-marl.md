---
layout: post
title: "RL Notes (4): MCTS and AlphaZero"
date: 2025-12-19 06:00:00
author: Qi Lu
tags: [RL, Model-Based, MARL, MCTS, AlphaZero]
lang: en
translation: /model-based-marl/
---

This is the fourth article in the reinforcement learning series, introducing core concepts of Model-Based RL and Multi-Agent RL. Starting from the pursuit of sample efficiency, it explains World Models, Dyna architecture, and MCTS, ultimately demonstrating the powerful combination of these techniques through AlphaGo/AlphaZero as examples, and introduces game theory fundamentals and Self-Play methods.

## Introduction: The Pursuit of Sample Efficiency

### Core Problem

The Model-Free methods introduced in the previous three chapters (Q-Learning, Policy Gradient, PPO) are powerful but share a common flaw:

> **Extremely low sample efficiency**—training an Atari game agent requires hundreds of millions of frames, equivalent to hundreds of hours of human play. Yet humans typically need only a few minutes to learn basic operations. Why is there such a huge gap?

The key difference is: **humans have a world model in their minds**. When we imagine "what would happen if I do this," we are using this model for **mental simulation**, without needing to actually try it.

The core idea of Model-Based RL is precisely: **learn or utilize an environment model to improve sample efficiency through planning**.

### Model-Free vs Model-Based

RL methods are divided into two major categories based on whether they use an environment model:

> **Model-Free vs Model-Based**
> - **Model-Free**: Does not learn or use an environment model, directly learns value functions or policies from real experience
> - **Model-Based**: Learns or utilizes environment model $\hat{P}(s'|s,a)$, $\hat{R}(s,a)$, performs planning within the model

<!-- tikz-source: rl-mf-vs-mb-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.4cm, minimum height=0.9cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Model-Free
    \draw[rounded corners, thick, red!30, fill=red!5] (-5.5, -0.5) rectangle (-0.5, 2);
    \node[font=\bfseries\small, red!70] at (-3, 1.7) {Model-Free (Low efficiency)};
    \node[box, fill=blue!20] (env1) at (-4.5, 0.5) {Real Environment};
    \node[box, fill=green!20] (policy1) at (-1.5, 0.5) {Policy/Value};
    \draw[arrow] (env1) -- node[above, font=\footnotesize] {Many samples} (policy1);

    % Model-Based
    \draw[rounded corners, thick, green!40, fill=green!5] (0.5, -1.5) rectangle (7.5, 2);
    \node[font=\bfseries\small, green!60!black] at (4, 1.7) {Model-Based (High efficiency)};
    \node[box, fill=blue!20] (env2) at (1.5, 0.5) {Real Environment};
    \node[box, fill=orange!20] (model) at (4, 0.5) {World Model};
    \node[box, fill=green!20] (policy2) at (6.5, 0.5) {Policy/Value};
    \draw[arrow] (env2) -- node[above, font=\footnotesize] {Few samples} (model);
    \draw[arrow, very thick] (model) -- node[above, font=\footnotesize] {Simulation} (policy2);
    \draw[arrow, dashed] (env2) to[bend right=40] node[below, font=\footnotesize] {Correction} (policy2);
\end{tikzpicture}
-->
![Model-Free vs Model-Based]({{ site.baseurl }}/assets/figures/rl-mf-vs-mb-en.svg)

| Property | Model-Free | Model-Based |
|------|------------|-------------|
| Environment Model | Not needed | Needed (known or learned) |
| Sample Efficiency | Low | High |
| Computational Cost | Low | High (planning) |
| Model Error | None | May accumulate |
| Use Cases | Model hard to obtain | Model known or easy to learn |
| Typical Algorithms | Q-Learning, PPO | Dyna, MCTS, MuZero |

## Model-Based RL Overview

### Definition of World Model

> **World Model** is an estimate of environment dynamics, including:
> - **State transition model**: $\hat{P}(s'|s,a) \approx P(s'|s,a)$
> - **Reward model**: $\hat{R}(s,a) \approx R(s,a)$
>
> With a World Model, the agent can simulate action consequences "in its mind" without actually executing them.

There are two sources for World Models:

1. **Known rules**: Such as rules for board games, equations for physics engines
   - Advantage: Model is precise, no error
   - Disadvantage: Only applicable to domains with fully known rules

2. **Learning from data**: Use neural networks to learn from interaction experience
   - Advantage: Applicable to complex environments
   - Disadvantage: Model has errors

### Learning World Models

Learning a World Model is essentially a supervised learning problem. Given experience data $\{(s_t, a_t, r_t, s_{t+1})\}$:

1. **Deterministic model**: Directly predict next state

$$\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad L = \|s_{t+1} - \hat{s}_{t+1}\|^2$$

2. **Probabilistic model**: Predict state distribution

$$\hat{P}_\theta(s'|s, a), \quad L = -\log \hat{P}_\theta(s_{t+1}|s_t, a_t)$$

3. **Latent space model**: Predict in low-dimensional latent space

$$z_{t+1} = f_\theta(z_t, a_t), \quad z_t = \text{Encoder}(s_t)$$

> **Note**: Modern World Model methods (such as Dreamer, MuZero) typically perform predictions in latent space, avoiding direct prediction of high-dimensional raw observations (like images), greatly reducing learning difficulty.

### Model Bias Problem

> **Model Bias / Model Error**: When the learned model $\hat{P}, \hat{R}$ differs from the real environment $P, R$, strategies obtained from planning in the model may perform poorly in the real environment.

The key issue with Model Bias is **error compounding**:

<!-- tikz-source: rl-model-error-en
\begin{tikzpicture}[
    state/.style={draw, circle, minimum size=0.8cm, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Real trajectory
    \node[state, fill=blue!20] (s0) at (0, 0) {$s_0$};
    \node[state, fill=blue!20] (s1) at (2, 0) {$s_1$};
    \node[state, fill=blue!20] (s2) at (4, 0) {$s_2$};
    \node[state, fill=blue!20] (s3) at (6, 0) {$s_3$};
    \node[state, fill=blue!20] (s4) at (8, 0) {$s_4$};

    \draw[arrow] (s0) -- node[above, font=\footnotesize] {$a_0$} (s1);
    \draw[arrow] (s1) -- node[above, font=\footnotesize] {$a_1$} (s2);
    \draw[arrow] (s2) -- node[above, font=\footnotesize] {$a_2$} (s3);
    \draw[arrow] (s3) -- node[above, font=\footnotesize] {$a_3$} (s4);
    \node[above, font=\small] at (4, 0.7) {Real trajectory (solid)};

    % Predicted trajectory
    \node[state, fill=red!20] (h1) at (2, -2) {$\hat{s}_1$};
    \node[state, fill=red!20] (h2) at (4, -2.3) {$\hat{s}_2$};
    \node[state, fill=red!20] (h3) at (6, -2.7) {$\hat{s}_3$};
    \node[state, fill=red!20] (h4) at (8, -3.2) {$\hat{s}_4$};

    \draw[arrow, dashed, red!70] (s0) -- node[left, font=\footnotesize] {$a_0$} (h1);
    \draw[arrow, dashed, red!70] (h1) -- node[above, font=\footnotesize] {$a_1$} (h2);
    \draw[arrow, dashed, red!70] (h2) -- node[above, font=\footnotesize] {$a_2$} (h3);
    \draw[arrow, dashed, red!70] (h3) -- node[above, font=\footnotesize] {$a_3$} (h4);
    \node[below, font=\small, red!70] at (5, -3.6) {Predicted trajectory (dashed) --- error accumulates};
\end{tikzpicture}
-->
![Model Bias Error Compounding]({{ site.baseurl }}/assets/figures/rl-model-error-en.svg)

Error $\epsilon_t$ increases with steps: $\epsilon_1 < \epsilon_2 < \epsilon_3 < \epsilon_4$

> **Error Compounding Upper Bound Theorem**: Let the single-step model error be $\epsilon = \max_{s,a} \|\hat{P}(\cdot|s,a) - P(\cdot|s,a)\|_1$, then the upper bound of total variation distance for $H$-step planning is:
>
> $$\text{TV}(\hat{P}^H, P^H) \leq H \cdot \epsilon$$
>
> That is, error **accumulates linearly** with planning steps.

Strategies to mitigate Model Bias:
1. **Short-term planning**: Use model only for short-term predictions (such as 1-step in Dyna)
2. **Ensemble models**: Train multiple models, use uncertainty to guide exploration
3. **Continuous correction**: Continuously update model with real data
4. **Latent space planning**: Plan in abstract space (such as MuZero)

## Planning Methods

With an environment model, the next step is to utilize the model for **planning**. Based on planning timing, there are two categories:

> **Background Planning vs Decision-time Planning**
> - **Background Planning**: Outside of interaction with the real environment, use the model to generate simulated experience to train the policy
> - **Decision-time Planning**: When a decision needs to be made, use the model for forward search to select the optimal action

<!-- tikz-source: rl-planning-types-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=4cm, minimum height=2cm, align=center, font=\small}
]
    \node[box, fill=orange!15] (bg) at (0, 0) {Offline simulation\\Train policy network\\Example: Dyna};
    \node[font=\bfseries\small, orange!70] at (0, 1.5) {Background Planning};

    \node[box, fill=green!15] (dt) at (7, 0) {Online search\\No training\\Example: MCTS};
    \node[font=\bfseries\small, green!60!black] at (7, 1.5) {Decision-time Planning};
\end{tikzpicture}
-->
![Planning Types]({{ site.baseurl }}/assets/figures/rl-planning-types-en.svg)

### Dyna Architecture

Dyna is a classic framework for Background Planning, proposed by Sutton in 1991. Its core idea is: **after each real interaction, use the model to generate multiple simulated experiences to accelerate learning**.

<!-- tikz-source: rl-dyna-architecture-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=0.9cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (env) at (0, 2) {Real Environment};
    \node[box, fill=orange!20] (model) at (5.5, 2) {World Model $\hat{P}, \hat{R}$};
    \node[box, fill=yellow!20] (exp) at (0, 0) {Experience Buffer};
    \node[box, fill=green!20] (policy) at (5.5, 0) {Policy/Value $Q(s,a)$};

    \draw[arrow] (env) -- node[above, font=\footnotesize] {Learn Model} (model);
    \draw[arrow] (env) -- node[left, font=\footnotesize] {Real exp} (exp);
    \draw[arrow] (exp) -- node[above, font=\footnotesize] {Direct RL} (policy);
    \draw[arrow, very thick, green!60!black] (model) -- node[right, font=\footnotesize] {Simulated (n times)} (policy);
    \draw[arrow, dashed] (policy) to[bend left=30] node[right, font=\footnotesize] {Action} (env);
\end{tikzpicture}
-->
![Dyna Architecture]({{ site.baseurl }}/assets/figures/rl-dyna-architecture-en.svg)

> Each real interaction generates $n$ simulated steps

**Dyna-Q Algorithm**:

```
Input: Planning steps n, learning rate α, exploration rate ε
Initialize Q(s,a) ← 0, tabular model Model(s,a) ← ∅

For each episode:
    Initialize state s
    while s is not terminal:
        a ← ε-greedy(Q(s, ·))
        Execute a, observe r, s'

        // Direct RL learning
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s', a') - Q(s,a)]

        // Update model
        Model(s,a) ← (r, s')  // Deterministic model

        // Planning: learn from model
        for i = 1 to n:
            Randomly select previously visited state-action pair (s̃, ã)
            (r̃, s̃') ← Model(s̃, ã)
            Q(s̃,ã) ← Q(s̃,ã) + α[r̃ + γ max_a' Q(s̃', a') - Q(s̃,ã)]

        s ← s'
```

**Dyna's Core Advantages**:
1. **Improved sample efficiency**: Each real interaction can produce n simulated learning steps
2. **Flexible computation-sample tradeoff**: Increasing n can trade more computation for less real interaction
3. **Gradual convergence**: When the model is accurate, theoretically converges to the same policy as direct learning

### Decision-time Planning

Unlike Background Planning, Decision-time Planning performs planning at each decision point:

1. Starting from current state, use the model to simulate multiple possible trajectories
2. Evaluate the return of each trajectory
3. Select the optimal first-step action
4. Re-plan after execution (do not save intermediate results)

Characteristics of Decision-time Planning:
- **Computation focused**: All computation serves the current decision
- **Dynamic precision**: Can adjust search depth and breadth as needed
- **No training needed**: Can be directly used at test time

The most famous Decision-time Planning method is **Monte Carlo Tree Search (MCTS)**.

## Monte Carlo Tree Search (MCTS)

MCTS is a tree search-based Decision-time Planning method widely used in board games and combinatorial optimization problems.

### Core Idea of MCTS

The goal of MCTS is to estimate the value of each action in the current state within a limited computational budget. Its core idea is: **selectively expand the search tree, concentrating computational resources on the most promising branches**.

> *How to decide which branch is "most promising"? This requires balancing **exploitation** (selecting known good branches) and **exploration** (trying uncertain branches).*

### MCTS Four-Step Process

Each iteration of MCTS includes four steps:

<!-- tikz-source: rl-mcts-steps-en
\begin{tikzpicture}[
    node/.style={draw, circle, minimum size=0.6cm, font=\footnotesize},
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=3cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Step 1: Selection
    \draw[rounded corners, thick, blue!30, fill=blue!5] (-0.5, -2.5) rectangle (2.5, 1.5);
    \node[font=\bfseries\small, blue!70] at (1, 1.2) {1. Selection};
    \node[node, fill=orange!30] (r1) at (1, 0.5) {root};
    \node[node, fill=green!30] (a1) at (0.3, -0.5) {};
    \node[node] (b1) at (1.7, -0.5) {};
    \node[node, fill=green!30] (c1) at (0.3, -1.5) {leaf};
    \draw[arrow, thick, green!60!black] (r1) -- node[left, font=\tiny] {UCB} (a1);
    \draw[arrow] (r1) -- (b1);
    \draw[arrow, thick, green!60!black] (a1) -- node[left, font=\tiny] {UCB} (c1);

    % Step 2: Expansion
    \draw[rounded corners, thick, orange!30, fill=orange!5] (3, -2.5) rectangle (6, 1.5);
    \node[font=\bfseries\small, orange!70] at (4.5, 1.2) {2. Expansion};
    \node[node] (r2) at (4.5, 0.5) {root};
    \node[node] (a2) at (4.5, -0.5) {};
    \node[node] (c2) at (4, -1.5) {};
    \node[node, fill=yellow!50, dashed] (new) at (5, -1.5) {new};
    \draw[arrow] (r2) -- (a2);
    \draw[arrow] (a2) -- (c2);
    \draw[arrow, dashed, orange!70] (a2) -- (new);

    % Step 3: Evaluation
    \draw[rounded corners, thick, green!30, fill=green!5] (6.5, -2.5) rectangle (9.5, 1.5);
    \node[font=\bfseries\small, green!60!black] at (8, 1.2) {3. Evaluation};
    \node[node] (r3) at (8, 0.5) {root};
    \node[node] (a3) at (8, -0.5) {};
    \node[node, fill=purple!30] (c3) at (8, -1.5) {eval};
    \draw[arrow] (r3) -- (a3);
    \draw[arrow] (a3) -- (c3);
    \node[font=\footnotesize, purple!70] at (8, -2.1) {rollout / $v(s)$};

    % Step 4: Backup
    \draw[rounded corners, thick, purple!30, fill=purple!5] (10, -2.5) rectangle (13, 1.5);
    \node[font=\bfseries\small, purple!70] at (11.5, 1.2) {4. Backup};
    \node[node, fill=purple!20] (r4) at (11.5, 0.5) {$\uparrow$};
    \node[node, fill=purple!20] (a4) at (11.5, -0.5) {$\uparrow$};
    \node[node, fill=purple!30] (c4) at (11.5, -1.5) {$v$};
    \draw[arrow, purple!70] (c4) -- (a4);
    \draw[arrow, purple!70] (a4) -- (r4);

    % Arrows between steps
    \draw[arrow, very thick] (2.5, -0.5) -- (3, -0.5);
    \draw[arrow, very thick] (6, -0.5) -- (6.5, -0.5);
    \draw[arrow, very thick] (9.5, -0.5) -- (10, -0.5);
\end{tikzpicture}
-->
![MCTS Four Steps]({{ site.baseurl }}/assets/figures/rl-mcts-steps-en.svg)

- **Selection**: Select by UCB along tree
- **Expansion**: Expand new child node
- **Evaluation**: Rollout or Value Network
- **Backup**: Backpropagate statistics

1. **Selection**: Starting from the root node, recursively select child nodes using a **tree policy** (such as UCB) until reaching a leaf node (an incompletely expanded node).

2. **Expansion**: If the leaf node is not a terminal state, expand one or more new child nodes based on feasible actions.

3. **Evaluation**: Evaluate the value of the newly expanded node. Traditional methods use **rollout** (random simulation to game end); modern methods use a **value network** for direct estimation.

4. **Backup**: Backpropagate the evaluation value along the selection path, updating the visit count $N$ and value estimate $Q$ of all nodes on the path.

### UCB Formula

The core of the Selection phase is the **UCB (Upper Confidence Bound)** formula, which elegantly balances exploitation and exploration:

> **UCB for Trees (UCT)**
>
> In the Selection phase, select the action that maximizes the following value:
>
> $$\text{UCB}(s, a) = \underbrace{Q(s, a)}_{\text{exploitation}} + \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{exploration}}$$
>
> Where:
> - $Q(s, a)$: Average value estimate of action $a$ (statistics from historical simulations)
> - $N(s)$: Total visit count of state $s$
> - $N(s, a)$: Number of times action $a$ was executed in state $s$
> - $c$: Exploration coefficient, controls exploration-exploitation tradeoff

Intuitive understanding of UCB:
- **Exploitation term** $Q(s,a)$: Select actions with good historical performance
- **Exploration term**: Select actions with low visit counts (high uncertainty)
- As visit count increases, exploration bonus gradually decreases, eventually dominated by exploitation
- Larger $c$ means more exploration; smaller $c$ means more exploitation

### MCTS Algorithm

```
Input: Current state s₀, search budget B (iteration count), exploration coefficient c
Output: Optimal action a*

Initialize root node = s₀, N(root) = 0

for i = 1 to B:
    // Selection
    node ← root
    while node is fully expanded and not terminal:
        a ← argmax_a UCB(node, a)
        node ← child(node, a)

    // Expansion
    if node is not terminal:
        Select an unexpanded action a
        node ← expand child node child(node, a)

    // Evaluation
    v ← Evaluate(node)  // Rollout or value network

    // Backup
    while node ≠ null:
        N(node) ← N(node) + 1
        Q(node) ← Q(node) + (v - Q(node)) / N(node)
        node ← parent(node)

return argmax_a N(root, a)  // Select action with most visits
```

> **Important**: MCTS final action selection criterion:
> - During training/search: Use UCB (balance exploration-exploitation)
> - Final decision: Select action with **most visits** (more robust)
>
> Choose visit count rather than average value because high visit count means high confidence.

## AlphaGo and AlphaZero

AlphaGo and AlphaZero are milestone achievements of MCTS + deep learning + Self-Play.

### The Challenge of Go

Go is considered the most difficult board game for AI to master:

- **Enormous search space**: Average of $\sim 200$ legal moves per step, about $200$ steps per game, total states $\sim 10^{170}$
- **Difficult position evaluation**: Unlike chess with clear piece values, the merit of a Go position is hard to quantify
- **Long-term planning**: Need to consider strategic impact dozens of moves ahead

Traditional Go AI used exhaustive search + hand-crafted evaluation functions, reaching only amateur dan level.

### AlphaGo Architecture (2016)

AlphaGo defeated world champion Lee Sedol 4:1 in 2016. Its architecture includes:

<!-- tikz-source: rl-alphago-architecture-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3.5cm, minimum height=1.2cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (input) at (0, 0) {Board State $19 \times 19$};
    \node[box, fill=orange!20] (pn) at (-3, -2.5) {Policy Network $p(a|s)$\\{\footnotesize Supervised+RL tuning}};
    \node[box, fill=green!20] (vn) at (3, -2.5) {Value Network $v(s)$\\{\footnotesize Self-play prediction}};
    \node[box, fill=purple!20] (mcts) at (0, -5) {MCTS Search};
    \node[box, fill=yellow!30] (output) at (0, -7.5) {Final Action};

    \draw[arrow] (input) -- (pn);
    \draw[arrow] (input) -- (vn);
    \draw[arrow] (pn) -- node[left, font=\footnotesize] {Guide selection} (mcts);
    \draw[arrow] (vn) -- node[right, font=\footnotesize] {Evaluate leaf} (mcts);
    \draw[arrow] (mcts) -- (output);
\end{tikzpicture}
-->
![AlphaGo Architecture]({{ site.baseurl }}/assets/figures/rl-alphago-architecture-en.svg)

1. **Policy Network** $p_\theta(a|s)$:
   - Input: Board state (multi-channel features)
   - Output: Probability of placing a stone at each position
   - Training: First supervised learning on human game records, then reinforcement with Policy Gradient through self-play

2. **Value Network** $v_\phi(s)$:
   - Input: Board state
   - Output: Win rate estimate for current position $v_\phi(s) \approx \mathbb{E}[z|s]$, where $z \in \{-1, +1\}$
   - Training: Supervised learning on $(s, z)$ data generated from self-play

3. **Improved MCTS**:
   - Selection: Guided by Policy Network (PUCT formula)

   $$\text{UCB}(s,a) = Q(s,a) + c \cdot p_\theta(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

   - Evaluation: Mix Value Network and rollout

   $$v = (1-\lambda) v_\phi(s) + \lambda z_{\text{rollout}}$$

### AlphaZero's Simplification and Transcendence (2017)

AlphaZero in 2017 significantly simplified AlphaGo's design while achieving even stronger performance:

| Property | AlphaGo | AlphaZero |
|------|---------|-----------|
| Human game records | Needed (supervised pretraining) | **Not needed** |
| Network structure | Separate Policy + Value | **Unified network** |
| Rollout | Needed | **Not needed** |
| Feature engineering | Hand-crafted features | **Raw board input** |
| Training time | Months | **Hours** |
| Applicable games | Only Go | **Go, Chess, Shogi** |

<!-- tikz-source: rl-alphazero-network-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.5cm, minimum height=0.9cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (input) at (0, 0) {Board State $s$};
    \node[box, fill=orange!30] (net) at (0, -1.5) {ResNet (Unified Network)};
    \node[box, fill=green!20] (policy) at (-2.5, -3) {$p(a|s)$ Policy Head};
    \node[box, fill=purple!20] (value) at (2.5, -3) {$v(s)$ Value Head};

    \draw[arrow] (input) -- (net);
    \draw[arrow] (net) -- (policy);
    \draw[arrow] (net) -- (value);
\end{tikzpicture}
-->
![AlphaZero Network]({{ site.baseurl }}/assets/figures/rl-alphazero-network-en.svg)

> Single network outputs both policy and value. Shared representation, fewer params, more efficient.

### AlphaZero Training Loop

AlphaZero's training is a **self-reinforcing** loop:

<!-- tikz-source: rl-alphazero-loop-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1.2cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (selfplay) at (0, 0) {Self-Play\\Generate game data};
    \node[box, fill=orange!20] (train) at (5, 0) {Network Training\\Learn from search};
    \node[box, fill=green!20] (network) at (2.5, -2.5) {Neural Network\\$(p, v)$};

    \draw[arrow] (selfplay) -- node[above, font=\footnotesize] {$(s, \pi_{\text{MCTS}}, z)$} (train);
    \draw[arrow] (train) -- node[right, font=\footnotesize] {Update $\theta$} (network);
    \draw[arrow] (network) -- node[left, font=\footnotesize] {Guide search} (selfplay);
\end{tikzpicture}
-->
![AlphaZero Training Loop]({{ site.baseurl }}/assets/figures/rl-alphazero-loop-en.svg)

> **Positive loop**: Better network → better search → better training data → better network

**AlphaZero Training Algorithm**:

```
Initialize network parameters θ (random initialization)

repeat until convergence:
    // Self-Play to generate data
    for each game:
        for each step t:
            Use current network + MCTS search to get π_MCTS(a|s_t)
            Sample action a_t from π_MCTS
            Record (s_t, π_MCTS)

        Game ends, get win/loss z ∈ {-1, +1}
        Add (s_t, π_MCTS, z) to training data

    // Network training
    Sample batch from training data
    Minimize loss: L(θ) = (z - v_θ(s))² - π_MCTS^T log p_θ(s) + c‖θ‖²
```

Where each loss term means:
- **$(z - v_\theta(s))^2$**: Value loss, making value prediction approach game outcome
- **$-\pi_{\text{MCTS}}^\top \log p_\theta(s)$**: Policy loss, making policy approach MCTS search result
- **$c\|\theta\|^2$**: L2 regularization term

**AlphaZero's Core Insights**:
1. **MCTS as policy improvement**: Search-generated $\pi_{\text{MCTS}}$ is better than raw network $p_\theta$
2. **Network learns search**: Network is trained to imitate MCTS output
3. **Positive loop**: Better network → better search → better training data → better network

This loop requires no human knowledge, learning completely from scratch (tabula rasa).

## Multi-Agent RL Basics

When multiple agents exist in the environment, the problem becomes more complex.

### From Single-Agent to Multi-Agent

> **Core problem**: When other agents are also learning and changing their strategies, the environment is **non-stationary** for a single agent. This breaks the basic assumption of MDPs.

In multi-agent environments, state transitions and rewards depend not only on one's own actions but also on other agents' actions:

$$P(s'|s, a_1, a_2, \ldots, a_n), \quad R_i(s, a_1, a_2, \ldots, a_n)$$

When other agents' strategies $\pi_{-i}$ change during the learning process, from agent $i$'s perspective, the environment is non-stationary.

Impact of non-stationarity:
- Convergence guarantees of single-agent RL no longer apply
- Optimal response strategy varies with opponent's strategy
- May exhibit policy oscillation, unable to converge

### Game Theory Basics

Multi-agent problems can be described using game theory language.

> **Normal-form Game**
>
> An $n$-player game consists of the following elements:
> - **Player set**: $\mathcal{N} = \{1, 2, \ldots, n\}$
> - **Strategy space**: Each player $i$'s strategy set $\mathcal{A}_i$
> - **Utility function**: $u_i: \mathcal{A}_1 \times \cdots \times \mathcal{A}_n \to \mathbb{R}$, representing player $i$'s payoff under each strategy combination

**Prisoner's Dilemma Example**:

Two suspects are interrogated separately, each choosing "cooperate" (stay silent) or "defect" (confess):

|  | Player 2: Cooperate | Player 2: Defect |
|---|---|---|
| Player 1: Cooperate | (-1, -1) | (-3, 0) |
| Player 1: Defect | (0, -3) | (-2, -2) |

Analysis:
- Regardless of opponent's choice, "defect" is always more advantageous (dominant strategy)
- Result: Both defect, each gets -2
- But if both cooperate, each gets -1 (Pareto superior)

### Cooperative vs Competitive Settings

Multi-agent scenarios mainly fall into two categories:

1. **Cooperative**: All agents share rewards, maximize team total return
   - Examples: Multi-robot collaborative carrying, multi-agent coordinated navigation
   - Challenge: Credit Assignment—how much did each agent contribute?
   - Method: Centralized Training Decentralized Execution (CTDE)

2. **Competitive/Zero-sum**: One party's gain equals another's loss
   - Examples: Board games, adversarial games
   - Property: $u_1 + u_2 = 0$
   - Goal: Find Nash equilibrium

### Nash Equilibrium

> **Nash Equilibrium**
>
> A strategy combination $(\pi_1^*, \pi_2^*, \ldots, \pi_n^*)$ is a Nash equilibrium if and only if no player has an incentive to unilaterally change their strategy:
>
> $$\forall i, \forall \pi_i: \quad u_i(\pi_i^*, \pi_{-i}^*) \geq u_i(\pi_i, \pi_{-i}^*)$$
>
> Where $\pi_{-i}^*$ represents the strategies of all players except player $i$.

Meaning of Nash equilibrium:
- Each player is making a **best response** to other players' strategies
- It is a **stable state**: no one has an incentive to unilaterally deviate
- Not necessarily globally optimal (as in Prisoner's Dilemma where mutual defection is Nash equilibrium but not Pareto optimal)

**Nash Equilibrium in Rock-Paper-Scissors**:

|  | Rock | Scissors | Paper |
|---|---|---|---|
| Rock | (0, 0) | (1, -1) | (-1, 1) |
| Scissors | (-1, 1) | (0, 0) | (1, -1) |
| Paper | (1, -1) | (-1, 1) | (0, 0) |

Nash equilibrium: Both players randomly choose with probability $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.

Any deterministic strategy can be exploited by the opponent; only a **mixed strategy** (randomization) can achieve equilibrium.

> **Nash Equilibrium Existence Theorem**: Every finite game (finite players, finite strategies) has at least one Nash equilibrium (possibly a mixed strategy equilibrium).

## Self-Play Methods

Self-Play is a powerful method for training game AI, and is one of the keys to AlphaGo/AlphaZero's success.

### Definition of Self-Play

> **Self-Play**: The agent plays against itself (or its historical versions), learning to improve its strategy from game experience.

<!-- tikz-source: rl-self-play-en
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.5cm, minimum height=0.9cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    \node[box, fill=blue!20] (current) at (0, 2) {Current Policy $\pi$};
    \node[box, fill=orange!20] (opponent) at (5, 2) {Opponent $\pi$ or $\pi'$};
    \node[box, fill=yellow!20] (pool) at (5, 4) {Opponent Pool};
    \node[box, fill=green!30] (game) at (2.5, 0) {Game Play};
    \node[box, fill=purple!20] (exp) at (2.5, -2) {Experience $(s,a,r,s')$};

    \draw[arrow] (current) -- (game);
    \draw[arrow] (opponent) -- (game);
    \draw[arrow, dashed] (pool) -- (opponent);
    \draw[arrow] (game) -- (exp);
    \draw[arrow] (exp) to[bend right=40] node[left, font=\footnotesize] {Update} (current);
\end{tikzpicture}
-->
![Self-Play]({{ site.baseurl }}/assets/figures/rl-self-play-en.svg)

### Advantages of Self-Play

1. **Unlimited data**: Can generate arbitrarily many game data, not limited by human game count

2. **Adaptive difficulty**: Opponent grows stronger with oneself, always providing appropriate challenge
   - Early stage: Weak opponent, easy to learn basic strategies
   - Late stage: Strong opponent, drives learning of advanced strategies

3. **Discover new strategies**: Not limited by human prior knowledge, may discover innovative strategies unknown to humans
   - AlphaGo's "shoulder hit" and other new moves shocked professional players

4. **Approach Nash equilibrium**: In zero-sum games, Self-Play theoretically converges to Nash equilibrium

### Challenges of Self-Play

1. **Strategy forgetting**:
   - When strategy updates, may "forget" how to deal with old strategies
   - Solution: Maintain historical opponent pool, randomly sample opponents

2. **Local optima**:
   - May fall into local optimum of "only good at playing against itself"
   - Example: Two versions counter each other, forming a cycle
   - Solution: Add diversity rewards, or sample from opponent pool

3. **Difficulty in evaluation**:
   - No fixed baseline to measure progress
   - Solution: Use Elo rating system or play against fixed opponents

**Relationship between Self-Play and Nash Equilibrium**:
- In two-player zero-sum games, if Self-Play converges, it converges to Nash equilibrium
- Intuition: Nash equilibrium is a "fixed point of best responses," Self-Play iteratively seeks best responses
- But convergence is not guaranteed—may have strategy cycles

## Chapter Summary

1. **Model-Based RL** uses environment models to improve sample efficiency
   - World Model = state transition + reward model
   - Model Bias: Model errors accumulate, requiring short-term planning or continuous correction

2. **Dyna architecture** combines direct learning with planning
   - After each real interaction, use model to generate n simulated experiences
   - Provides flexible computation-sample tradeoff

3. **MCTS** is the representative of Decision-time Planning
   - Four-step process: Selection, Expansion, Evaluation, Backup
   - UCB formula balances exploration and exploitation

4. **AlphaGo/AlphaZero** demonstrates the powerful combination of MCTS + deep learning + Self-Play
   - AlphaZero starts from scratch, requires no human knowledge
   - Core loop: MCTS improves policy → network learns search → positive reinforcement

5. **Multi-Agent RL** faces non-stationarity challenges
   - Nash equilibrium: Stable strategy combination
   - Self-Play: Effective method for training game AI

<!-- tikz-source: rl-methods-taxonomy-en
\begin{tikzpicture}[
    node/.style={draw, rounded corners, minimum width=2.2cm, minimum height=0.9cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Root
    \node[node, fill=orange!30] (rl) at (0, 0) {RL Methods};

    % Level 1
    \node[node, fill=red!20] (mf) at (-4, -1.5) {Model-Free};
    \node[node, fill=green!20] (mb) at (4, -1.5) {Model-Based};
    \draw[arrow] (rl) -- (mf);
    \draw[arrow] (rl) -- (mb);

    % Model-Free branches
    \node[node, fill=blue!15] (vb) at (-6, -3.5) {Value-Based\\{\footnotesize DQN}};
    \node[node, fill=blue!15] (pb) at (-4, -3.5) {Policy-Based\\{\footnotesize REINFORCE}};
    \node[node, fill=blue!15] (ac) at (-2, -3.5) {Actor-Critic\\{\footnotesize PPO, SAC}};
    \draw[arrow] (mf) -- (vb);
    \draw[arrow] (mf) -- (pb);
    \draw[arrow] (mf) -- (ac);

    % Model-Based branches
    \node[node, fill=yellow!20] (bg) at (3, -3.5) {Background Planning\\{\footnotesize Dyna}};
    \node[node, fill=yellow!20] (dt) at (6, -3.5) {Decision-time Planning\\{\footnotesize MCTS}};
    \draw[arrow] (mb) -- (bg);
    \draw[arrow] (mb) -- (dt);

    % AlphaZero
    \node[node, fill=purple!30, font=\bfseries\small] (az) at (2, -5.5) {AlphaZero};
    \draw[arrow, dashed, purple!70] (ac) to[bend right=15] (az);
    \draw[arrow, dashed, purple!70] (dt) to[bend left=15] (az);
\end{tikzpicture}
-->
![RL Methods Taxonomy]({{ site.baseurl }}/assets/figures/rl-methods-taxonomy-en.svg)

> **AlphaZero = MCTS + Policy Network + Value Network + Self-Play**

The next blog post will enter the field of combining LLMs with RL, introducing methods like RLHF and DPO used for language model alignment.
