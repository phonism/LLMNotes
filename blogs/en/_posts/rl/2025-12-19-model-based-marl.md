---
layout: post
title: "RL Notes (4): MCTS and AlphaZero"
date: 2025-12-19 06:00:00
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

<div class="mermaid">
flowchart TB
    subgraph MF["**Model-Free** (Low sample efficiency)"]
        ENV1["Real Environment"] -->|"Many samples"| POLICY1["Policy/Value Function"]
    end
    subgraph MB["**Model-Based** (High sample efficiency)"]
        ENV2["Real Environment"] -->|"Few samples"| MODEL["World Model"]
        MODEL -->|"Simulation"| POLICY2["Policy/Value Function"]
        ENV2 -.->|"Correction"| POLICY2
    end
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

<div class="mermaid">
flowchart LR
    subgraph Real["Real trajectory (solid)"]
        S0((s₀)) -->|a₀| S1((s₁))
        S1 -->|a₁| S2((s₂))
        S2 -->|a₂| S3((s₃))
        S3 -->|a₃| S4((s₄))
    end
    subgraph Pred["Predicted trajectory (dashed) — error accumulates"]
        S0 -.->|a₀| H1((ŝ₁))
        H1 -.->|a₁| H2((ŝ₂))
        H2 -.->|a₂| H3((ŝ₃))
        H3 -.->|a₃| H4((ŝ₄))
    end
</div>

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

<div class="mermaid">
flowchart LR
    subgraph BG["**Training-time Planning** Background Planning"]
        BG1["Offline simulation<br/>Train policy network<br/>Example: Dyna"]
    end
    subgraph DT["**Decision-time Planning**"]
        DT1["Online search<br/>No training<br/>Example: MCTS"]
    end
</div>

### Dyna Architecture

Dyna is a classic framework for Background Planning, proposed by Sutton in 1991. Its core idea is: **after each real interaction, use the model to generate multiple simulated experiences to accelerate learning**.

<div class="mermaid">
flowchart TB
    ENV["Real Environment"] -->|"Learn Model"| MODEL["World Model P̂, R̂"]
    ENV -->|"Real exp"| EXP["Experience Buffer (s,a,r,s')"]
    EXP -->|"Direct RL"| POLICY["Policy/Value Q(s,a)"]
    MODEL ==>|"Simulated exp (n times)"| POLICY
    POLICY -.->|"Action"| ENV
</div>

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

<div class="mermaid">
flowchart LR
    subgraph S1["1. Selection"]
        direction TB
        R1((root)) ==>|UCB| A1((node))
        R1 --> B1((node))
        A1 ==>|UCB| C1((leaf))
    end
    subgraph S2["2. Expansion"]
        direction TB
        R2((root)) --> A2((node))
        A2 --> C2((node))
        A2 -.->|new| NEW((new))
    end
    subgraph S3["3. Evaluation"]
        direction TB
        R3((root)) --> A3((node))
        A3 --> C3((eval))
        C3 -.->|"rollout/v(s)"| V["v=?"]
    end
    subgraph S4["4. Backup"]
        direction TB
        R4((↑)) --> A4((↑))
        A4 --> C4((v))
    end
    S1 --> S2 --> S3 --> S4
</div>

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

<div class="mermaid">
flowchart TB
    INPUT["Board State 19×19"] --> PN["Policy Network p(a|s)<br/><small>Supervised(human games)+RL tuning</small>"]
    INPUT --> VN["Value Network v(s)<br/><small>Supervised(self-play result prediction)</small>"]
    PN -->|"Guide selection"| MCTS["MCTS Search"]
    VN -->|"Evaluate leaf"| MCTS
    MCTS --> OUTPUT["Final Action"]
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

<div class="mermaid">
flowchart TB
    INPUT["Board State s"] --> NET["ResNet (Unified Network)"]
    NET --> POLICY["p(a|s) Policy Head"]
    NET --> VALUE["v(s) Value Head"]
</div>

> Single network outputs both policy and value. Shared representation, fewer params, more efficient.

### AlphaZero Training Loop

AlphaZero's training is a **self-reinforcing** loop:

<div class="mermaid">
flowchart LR
    SELFPLAY["Self-Play<br/>Generate game data"] -->|"(s, π_MCTS, z)"| TRAIN["Network Training<br/>Learn from search"]
    TRAIN -->|"Update θ"| NETWORK["Neural Network<br/>(p, v)"]
    NETWORK -->|"Guide search"| SELFPLAY
</div>

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

<div class="mermaid">
flowchart TB
    CURRENT["Current Policy π"] --> GAME["Game Play"]
    OPPONENT["Opponent π or π'"] --> GAME
    POOL["Opponent Pool"] -.-> OPPONENT
    GAME --> EXP["Experience (s,a,r,s')"]
    EXP -->|"Update"| CURRENT
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

<div class="mermaid">
flowchart TB
    RL["RL Methods"] --> MF["Model-Free"]
    RL --> MB["Model-Based"]

    MF --> VB["Value-Based<br/><small>DQN</small>"]
    MF --> PB["Policy-Based<br/><small>REINFORCE</small>"]
    MF --> AC["Actor-Critic<br/><small>PPO, SAC</small>"]

    MB --> BG["Background Planning<br/><small>Dyna</small>"]
    MB --> DT["Decision-time Planning<br/><small>MCTS</small>"]

    AC -.-> AZ["**AlphaZero**"]
    DT -.-> AZ
</div>

> **AlphaZero = MCTS + Policy Network + Value Network + Self-Play**

The next blog post will enter the field of combining LLMs with RL, introducing methods like RLHF and DPO used for language model alignment.
