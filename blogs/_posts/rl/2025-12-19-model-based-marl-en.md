---
layout: post
title: "RL Notes (4): MCTS and AlphaZero"
date: 2025-12-19 05:00:00
author: Phonism
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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Model-Free
    \begin{scope}[shift={(-4,0)}]
        \node[box, fill=blue!20] (env1) at (0, 1.5) {Real Environment};
        \node[box, fill=orange!20] (policy1) at (0, -1) {Policy/Value Function};
        \draw[arrow, red, very thick] (env1) -- node[right, font=\small] {Many samples} (policy1);
        \node[font=\bfseries] at (0, 3) {Model-Free};
        \node[font=\scriptsize, text=red] at (0, -2.3) {Low sample efficiency};
    \end{scope}

    % Model-Based
    \begin{scope}[shift={(4,0)}]
        \node[box, fill=blue!20] (env2) at (0, 1.5) {Real Environment};
        \node[box, fill=green!20] (model) at (0, 0) {World Model};
        \node[box, fill=orange!20] (policy2) at (0, -1.5) {Policy/Value Function};
        \draw[arrow] (env2) -- node[right, font=\small] {Few samples} (model);
        \draw[arrow, green!60!black, very thick] (model) -- node[right, font=\small] {Simulation} (policy2);
        \draw[arrow, dashed] (env2.west) to[out=180, in=180] node[left, font=\small] {Correction} (policy2.west);
        \node[font=\bfseries] at (0, 3) {Model-Based};
        \node[font=\scriptsize, text=green!60!black] at (0, -2.8) {High sample efficiency};
    \end{scope}
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    state/.style={circle, draw, fill=blue!20, minimum size=0.8cm},
    pred/.style={circle, draw, dashed, fill=red!20, minimum size=0.8cm},
    arrow/.style={->, thick, >=stealth}
]
    % Real trajectory
    \node[state] (s0) at (0, 0) {$s_0$};
    \node[state] (s1) at (2, 0) {$s_1$};
    \node[state] (s2) at (4, 0) {$s_2$};
    \node[state] (s3) at (6, 0) {$s_3$};
    \node[state] (s4) at (8, 0) {$s_4$};

    \draw[arrow] (s0) -- node[above, font=\scriptsize] {$a_0$} (s1);
    \draw[arrow] (s1) -- node[above, font=\scriptsize] {$a_1$} (s2);
    \draw[arrow] (s2) -- node[above, font=\scriptsize] {$a_2$} (s3);
    \draw[arrow] (s3) -- node[above, font=\scriptsize] {$a_3$} (s4);

    % Predicted trajectory
    \node[pred] (h1) at (2, -1) {$\hat{s}_1$};
    \node[pred] (h2) at (4, -1.5) {$\hat{s}_2$};
    \node[pred] (h3) at (6, -2.2) {$\hat{s}_3$};
    \node[pred] (h4) at (8, -3) {$\hat{s}_4$};

    \draw[arrow, dashed, red] (s0) -- (h1);
    \draw[arrow, dashed, red] (h1) -- (h2);
    \draw[arrow, dashed, red] (h2) -- (h3);
    \draw[arrow, dashed, red] (h3) -- (h4);

    % Error labels
    \draw[<->, gray] (s1) -- node[right, font=\scriptsize] {$\epsilon_1$} (h1);
    \draw[<->, gray] (s2) -- node[right, font=\scriptsize] {$\epsilon_2$} (h2);
    \draw[<->, gray] (s3) -- node[right, font=\scriptsize] {$\epsilon_3$} (h3);
    \draw[<->, gray] (s4) -- node[right, font=\scriptsize] {$\epsilon_4$} (h4);

    % Legend
    \node[font=\small] at (4, 1) {Real trajectory (solid)};
    \node[font=\small, red] at (4, -4) {Predicted trajectory (dashed) — error accumulates};
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=3.5cm, minimum height=1.2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Background Planning
    \begin{scope}[shift={(-4.5, 0)}]
        \node[box, fill=green!20] (bg) at (0, 0) {Background\\Planning};
        \node[font=\small, align=center] at (0, -2) {Offline simulation\\Train policy network\\Example: Dyna};
        \node[font=\bfseries] at (0, 1.5) {Training-time Planning};
    \end{scope}

    % Decision-time Planning
    \begin{scope}[shift={(4.5, 0)}]
        \node[box, fill=orange!20] (dt) at (0, 0) {Decision-time\\Planning};
        \node[font=\small, align=center] at (0, -2) {Online search\\No training\\Example: MCTS};
        \node[font=\bfseries] at (0, 1.5) {Decision-time Planning};
    \end{scope}
\end{tikzpicture}
</script>
</div>

### Dyna Architecture

Dyna is a classic framework for Background Planning, proposed by Sutton in 1991. Its core idea is: **after each real interaction, use the model to generate multiple simulated experiences to accelerate learning**.

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[scale=0.95,
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Components
    \node[box, fill=blue!20] (env) at (0, 2) {Real Environment};
    \node[box, fill=green!20] (model) at (5, 2) {World Model\\$\hat{P}, \hat{R}$};
    \node[box, fill=orange!20] (policy) at (2.5, -1) {Policy/Value\\$Q(s,a)$};
    \node[box, fill=purple!15] (exp) at (-2.5, -1) {Experience Buffer\\$(s,a,r,s')$};

    % Connections
    \draw[arrow] (env) -- node[above, font=\small] {Learn Model} (model);
    \draw[arrow] (env) -- node[left, font=\small, pos=0.3] {Real exp} (exp);
    \draw[arrow] (exp) -- node[below, font=\small] {Direct RL} (policy);
    \draw[arrow, green!60!black, very thick] (model) -- node[right, font=\small, pos=0.3] {Simulated exp\\($n$ times)} (policy);
    \draw[arrow, dashed] (policy.north) to[out=120, in=240] node[left, font=\small] {Action} (env.south);

    % Annotations
    \node[font=\scriptsize, red] at (5, 0.3) {Each real interaction};
    \node[font=\scriptsize, red] at (5, -0.1) {generates $n$ simulated steps};
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[scale=0.8, every node/.style={scale=0.8},
    treenode/.style={circle, draw, minimum size=0.7cm},
    selected/.style={treenode, fill=blue!30, very thick},
    expanded/.style={treenode, fill=green!30, dashed},
    evaluated/.style={treenode, fill=orange!30},
    backed/.style={treenode, fill=red!20}
]
    % Step 1: Selection
    \begin{scope}[shift={(0, 0)}]
        \node[selected] (r1) at (0, 0) {};
        \node[selected] (a1) at (-0.8, -1) {};
        \node[treenode] (b1) at (0.8, -1) {};
        \node[selected] (c1) at (-1.2, -2) {};
        \node[treenode] (d1) at (-0.4, -2) {};

        \draw[very thick, blue, ->] (r1) -- (a1);
        \draw (r1) -- (b1);
        \draw[very thick, blue, ->] (a1) -- (c1);
        \draw (a1) -- (d1);

        \node[font=\small\bfseries] at (0, 0.8) {1. Selection};
        \node[font=\scriptsize, align=center] at (0, -3) {Select by UCB\\along tree};
    \end{scope}

    % Step 2: Expansion
    \begin{scope}[shift={(4, 0)}]
        \node[treenode] (r2) at (0, 0) {};
        \node[treenode] (a2) at (-0.8, -1) {};
        \node[treenode] (b2) at (0.8, -1) {};
        \node[treenode] (c2) at (-1.2, -2) {};
        \node[expanded] (new) at (-0.4, -2) {};

        \draw (r2) -- (a2);
        \draw (r2) -- (b2);
        \draw (a2) -- (c2);
        \draw[thick, green!60!black, dashed] (a2) -- (new);

        \node[font=\small\bfseries] at (0, 0.8) {2. Expansion};
        \node[font=\scriptsize, align=center] at (0, -3) {Expand new\\child node};
    \end{scope}

    % Step 3: Evaluation
    \begin{scope}[shift={(8, 0)}]
        \node[treenode] (r3) at (0, 0) {};
        \node[treenode] (a3) at (-0.8, -1) {};
        \node[treenode] (b3) at (0.8, -1) {};
        \node[evaluated] (c3) at (-1.2, -2) {};

        \draw (r3) -- (a3);
        \draw (r3) -- (b3);
        \draw (a3) -- (c3);

        % Rollout
        \draw[thick, orange, ->] (c3) -- ++(0.3, -0.8) -- ++(0.2, -0.6) -- ++(-0.1, -0.5);
        \node[font=\scriptsize] at (-0.3, -3.5) {$v = ?$};

        \node[font=\small\bfseries] at (0, 0.8) {3. Evaluation};
        \node[font=\scriptsize, align=center] at (0, -4.5) {Rollout or\\Value Network};
    \end{scope}

    % Step 4: Backup
    \begin{scope}[shift={(12, 0)}]
        \node[backed] (r4) at (0, 0) {$\uparrow$};
        \node[backed] (a4) at (-0.8, -1) {$\uparrow$};
        \node[treenode] (b4) at (0.8, -1) {};
        \node[backed] (c4) at (-1.2, -2) {$v$};

        \draw[thick, red, <-] (r4) -- (a4);
        \draw (r4) -- (b4);
        \draw[thick, red, <-] (a4) -- (c4);

        \node[font=\small\bfseries] at (0, 0.8) {4. Backup};
        \node[font=\scriptsize, align=center] at (0, -3) {Backpropagate\\statistics};
    \end{scope}
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1.2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Input
    \node[box, fill=blue!20] (input) at (0, 0) {Board State\\$19 \times 19$};

    % Policy Network
    \node[box, fill=green!20] (pn) at (-3.5, -2.5) {Policy Network\\$p_\theta(a|s)$};

    % Value Network
    \node[box, fill=orange!20] (vn) at (3.5, -2.5) {Value Network\\$v_\phi(s)$};

    % MCTS
    \node[box, fill=purple!20, minimum width=4cm] (mcts) at (0, -5) {MCTS Search};

    % Output
    \node[box, fill=red!15] (output) at (0, -7.5) {Final Action};

    % Connections
    \draw[arrow] (input) -- (pn);
    \draw[arrow] (input) -- (vn);
    \draw[arrow] (pn) -- node[left, font=\small] {Guide selection} (mcts);
    \draw[arrow] (vn) -- node[right, font=\small] {Evaluate leaf} (mcts);
    \draw[arrow] (mcts) -- (output);

    % Training annotations
    \node[font=\scriptsize, align=left] at (-6.5, -2.5) {Supervised\\(human games)\\+ RL tuning};
    \node[font=\scriptsize, align=right] at (6.5, -2.5) {Supervised\\(self-play\\result prediction)};
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Unified network
    \node[box, fill=blue!20] (input) at (0, 0) {Board State $s$};
    \node[box, fill=purple!25, minimum height=2cm] (net) at (0, -2.5) {ResNet\\(Unified Network)};

    % Dual head outputs
    \node[box, fill=green!20] (policy) at (-2.5, -5) {$p_\theta(a|s)$\\Policy Head};
    \node[box, fill=orange!20] (value) at (2.5, -5) {$v_\theta(s)$\\Value Head};

    \draw[arrow] (input) -- (net);
    \draw[arrow] (net) -- (policy);
    \draw[arrow] (net) -- (value);

    \node[font=\small, align=center] at (0, -6.5) {Single network outputs both policy and value\\Shared representation, fewer params, more efficient};
\end{tikzpicture}
</script>
</div>

### AlphaZero Training Loop

AlphaZero's training is a **self-reinforcing** loop:

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1.2cm, align=center},
    arrow/.style={->, very thick, >=stealth}
]
    % Three components
    \node[box, fill=green!20] (selfplay) at (0, 0) {Self-Play\\Generate game data};
    \node[box, fill=orange!20] (train) at (5, -3) {Network Training\\Learn from search};
    \node[box, fill=blue!20] (network) at (-5, -3) {Neural Network\\$(p_\theta, v_\theta)$};

    % Cycle arrows
    \draw[arrow, green!60!black] (selfplay) -- node[right, font=\small, pos=0.5] {$(s, \pi_{\text{MCTS}}, z)$} (train);
    \draw[arrow, orange] (train) -- node[below, font=\small, yshift=-3pt] {Update $\theta$} (network);
    \draw[arrow, blue] (network) -- node[left, font=\small, pos=0.5] {Guide search} (selfplay);

    % Center annotation
    \node[font=\small, align=center, text=gray] at (0, -1.8) {Positive loop\\Keep improving};
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Current policy
    \node[box, fill=blue!20] (current) at (0, 0) {Current Policy\\$\pi_\theta$};

    % Opponent (copy of self)
    \node[box, fill=blue!10] (opponent) at (5, 0) {Opponent\\$\pi_\theta$ or $\pi_{\theta'}$};

    % Game play
    \node[box, fill=green!20] (game) at (2.5, -2.5) {Game Play};

    % Experience
    \node[box, fill=orange!20] (exp) at (2.5, -5) {Experience\\$(s, a, r, s')$};

    % Update
    \draw[arrow] (current) -- (game);
    \draw[arrow] (opponent) -- (game);
    \draw[arrow] (game) -- (exp);
    \draw[arrow] (exp) to[out=180, in=270] node[left, font=\small] {Update} (current);

    % Historical opponent pool
    \node[box, fill=gray!20, dashed] (pool) at (8, -2.5) {Opponent Pool\\$\{\pi_{\theta_1}, \ldots\}$};
    \draw[arrow, dashed] (pool) -- (opponent);
\end{tikzpicture}
</script>
</div>

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

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Hierarchy
    \node[box, fill=blue!25] (rl) at (0, 0) {RL Methods};

    \node[box, fill=green!20] (mf) at (-4, -1.5) {Model-Free};
    \node[box, fill=orange!20] (mb) at (4, -1.5) {Model-Based};

    \node[box, minimum width=2.2cm] (vb) at (-6.5, -3) {Value-Based};
    \node[box, minimum width=2.2cm] (pb) at (-4, -3) {Policy-Based};
    \node[box, minimum width=2.2cm] (ac) at (-1.5, -3) {Actor-Critic};

    \node[box] (bg) at (2.5, -3) {Background\\Planning};
    \node[box] (dt) at (5.5, -3) {Decision-time\\Planning};

    \node[font=\scriptsize, gray] at (-6.5, -3.8) {DQN};
    \node[font=\scriptsize, gray] at (-4, -3.8) {REINFORCE};
    \node[font=\scriptsize, gray] at (-1.5, -3.8) {PPO, SAC};
    \node[font=\scriptsize, gray] at (2.5, -3.8) {Dyna};
    \node[font=\scriptsize, gray] at (5.5, -3.8) {MCTS};

    % AlphaZero spanning
    \node[box, fill=purple!20, minimum width=3cm] (az) at (4, -5) {AlphaZero};

    \draw[arrow] (rl) -- (mf);
    \draw[arrow] (rl) -- (mb);
    \draw[arrow] (mf) -- (vb);
    \draw[arrow] (mf) -- (pb);
    \draw[arrow] (mf) -- (ac);
    \draw[arrow] (mb) -- (bg);
    \draw[arrow] (mb) -- (dt);
    \draw[arrow, dashed] (dt) -- (az);
    \draw[arrow, dashed] (ac.south) to[out=-45, in=180] (az.west);

    \node[font=\scriptsize, align=center] at (0, -5.5) {AlphaZero = MCTS + Policy Network + Value Network + Self-Play};
\end{tikzpicture}
</script>
</div>

The next blog post will enter the field of combining LLMs with RL, introducing methods like RLHF and DPO used for language model alignment.
