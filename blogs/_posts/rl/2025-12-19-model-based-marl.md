---
layout: post
title: "RL 学习笔记（四）：MCTS 与 AlphaZero"
date: 2025-12-19 06:00:00
author: Phonism
tags: [RL, Model-Based, MARL, MCTS, AlphaZero]
lang: zh
translation: /en/model-based-marl/
---

本文是强化学习系列的第四篇，介绍 Model-Based RL 和 Multi-Agent RL 的核心概念，从样本效率的追求出发，讲解 World Model、Dyna 架构、MCTS，最后以 AlphaGo/AlphaZero 为例展示这些技术的强大组合，并介绍博弈论基础和 Self-Play 方法。

## 引言：样本效率的追求

### 核心问题

前三章介绍的 Model-Free 方法（Q-Learning、Policy Gradient、PPO）虽然强大，但有一个共同的缺陷：

> **样本效率极低**——训练一个 Atari 游戏 agent 需要数亿帧画面，相当于人类玩数百小时。而人类通常只需几分钟就能学会基本操作。为什么会有如此巨大的差距？

关键区别在于：**人类在脑中有一个世界模型**。当我们想象"如果我这样做会发生什么"时，我们在用这个模型进行**心智模拟**（mental simulation），而不需要真正尝试。

Model-Based RL 的核心思想正是：**学习或利用环境模型，通过规划（Planning）来提高样本效率**。

### Model-Free vs Model-Based

根据是否使用环境模型，RL 方法分为两大类：

> **Model-Free 与 Model-Based**
> - **Model-Free**：不学习或使用环境模型，直接从真实经验中学习价值函数或策略
> - **Model-Based**：学习或利用环境模型 $\hat{P}(s'\|s,a)$, $\hat{R}(s,a)$，在模型中进行规划

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Model-Free
    \begin{scope}[shift={(-4,0)}]
        \node[box, fill=blue!20] (env1) at (0, 1.5) {真实环境};
        \node[box, fill=orange!20] (policy1) at (0, -1) {策略/价值函数};
        \draw[arrow, red, very thick] (env1) -- node[right, font=\small] {大量真实经验} (policy1);
        \node[font=\bfseries] at (0, 3) {Model-Free};
        \node[font=\scriptsize, text=red] at (0, -2.3) {样本效率低};
    \end{scope}

    % Model-Based
    \begin{scope}[shift={(4,0)}]
        \node[box, fill=blue!20] (env2) at (0, 1.5) {真实环境};
        \node[box, fill=green!20] (model) at (0, 0) {环境模型};
        \node[box, fill=orange!20] (policy2) at (0, -1.5) {策略/价值函数};
        \draw[arrow] (env2) -- node[right, font=\small] {少量经验} (model);
        \draw[arrow, green!60!black, very thick] (model) -- node[right, font=\small] {大量模拟} (policy2);
        \draw[arrow, dashed] (env2.west) to[out=180, in=180] node[left, font=\small] {校正} (policy2.west);
        \node[font=\bfseries] at (0, 3) {Model-Based};
        \node[font=\scriptsize, text=green!60!black] at (0, -2.8) {样本效率高};
    \end{scope}
\end{tikzpicture}
</script>
</div>

| 特性 | Model-Free | Model-Based |
|------|------------|-------------|
| 环境模型 | 不需要 | 需要（已知或学习） |
| 样本效率 | 低 | 高 |
| 计算开销 | 低 | 高（规划） |
| 模型误差 | 无 | 可能累积 |
| 适用场景 | 模型难以获取 | 模型已知或易学 |
| 典型算法 | Q-Learning, PPO | Dyna, MCTS, MuZero |

## Model-Based RL 概述

### World Model 的定义

> **World Model** 是对环境动力学的估计，包括：
> - **状态转移模型**：$\hat{P}(s'\|s,a) \approx P(s'\|s,a)$
> - **奖励模型**：$\hat{R}(s,a) \approx R(s,a)$
>
> 有了 World Model，agent 可以在"脑中"模拟动作的后果，而不需要真正执行。

World Model 的来源有两种：

1. **已知规则**：如棋类游戏的规则、物理引擎的方程
   - 优点：模型精确，无误差
   - 缺点：仅适用于规则完全已知的领域

2. **从数据学习**：用神经网络从交互经验中学习
   - 优点：适用于复杂环境
   - 缺点：模型存在误差

### 学习 World Model

学习 World Model 本质上是一个监督学习问题。给定经验数据 $\\{(s_t, a_t, r_t, s_{t+1})\\}$：

1. **确定性模型**：直接预测下一状态

$$\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad L = \|s_{t+1} - \hat{s}_{t+1}\|^2$$

2. **概率模型**：预测状态分布

$$\hat{P}_\theta(s'|s, a), \quad L = -\log \hat{P}_\theta(s_{t+1}|s_t, a_t)$$

3. **隐空间模型**：在低维隐空间中预测

$$z_{t+1} = f_\theta(z_t, a_t), \quad z_t = \text{Encoder}(s_t)$$

> **注意**：现代 World Model 方法（如 Dreamer、MuZero）通常在隐空间中进行预测，避免直接预测高维原始观测（如图像），大大降低了学习难度。

### Model Bias 问题

> **Model Bias / Model Error**：当学习的模型 $\hat{P}, \hat{R}$ 与真实环境 $P, R$ 存在差异时，在模型中规划得到的策略在真实环境中可能表现不佳。

Model Bias 的关键问题是**误差累积**（Error Compounding）：

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    state/.style={circle, draw, fill=blue!20, minimum size=0.8cm},
    pred/.style={circle, draw, dashed, fill=red!20, minimum size=0.8cm},
    arrow/.style={->, thick, >=stealth}
]
    % 真实轨迹
    \node[state] (s0) at (0, 0) {$s_0$};
    \node[state] (s1) at (2, 0) {$s_1$};
    \node[state] (s2) at (4, 0) {$s_2$};
    \node[state] (s3) at (6, 0) {$s_3$};
    \node[state] (s4) at (8, 0) {$s_4$};

    \draw[arrow] (s0) -- node[above, font=\scriptsize] {$a_0$} (s1);
    \draw[arrow] (s1) -- node[above, font=\scriptsize] {$a_1$} (s2);
    \draw[arrow] (s2) -- node[above, font=\scriptsize] {$a_2$} (s3);
    \draw[arrow] (s3) -- node[above, font=\scriptsize] {$a_3$} (s4);

    % 预测轨迹
    \node[pred] (h1) at (2, -1) {$\hat{s}_1$};
    \node[pred] (h2) at (4, -1.5) {$\hat{s}_2$};
    \node[pred] (h3) at (6, -2.2) {$\hat{s}_3$};
    \node[pred] (h4) at (8, -3) {$\hat{s}_4$};

    \draw[arrow, dashed, red] (s0) -- (h1);
    \draw[arrow, dashed, red] (h1) -- (h2);
    \draw[arrow, dashed, red] (h2) -- (h3);
    \draw[arrow, dashed, red] (h3) -- (h4);

    % 误差标注
    \draw[<->, gray] (s1) -- node[right, font=\scriptsize] {$\epsilon_1$} (h1);
    \draw[<->, gray] (s2) -- node[right, font=\scriptsize] {$\epsilon_2$} (h2);
    \draw[<->, gray] (s3) -- node[right, font=\scriptsize] {$\epsilon_3$} (h3);
    \draw[<->, gray] (s4) -- node[right, font=\scriptsize] {$\epsilon_4$} (h4);

    % 图例
    \node[font=\small] at (4, 1) {真实轨迹（实线）};
    \node[font=\small, red] at (4, -4) {预测轨迹（虚线）——误差逐步累积};
\end{tikzpicture}
</script>
</div>

> **误差累积上界定理**：设单步模型误差为 $\epsilon = \max_{s,a} \|\hat{P}(\cdot\|s,a) - P(\cdot\|s,a)\|_1$，则 $H$ 步规划的总变差距离上界为：
>
> $$\text{TV}(\hat{P}^H, P^H) \leq H \cdot \epsilon$$
>
> 即误差随规划步数**线性累积**。

缓解 Model Bias 的策略：
1. **短期规划**：只用模型做短期预测（如 Dyna 中的 1-step）
2. **集成模型**：训练多个模型，用不确定性指导探索
3. **持续校正**：用真实数据不断更新模型
4. **隐空间规划**：在抽象空间中规划（如 MuZero）

## Planning 方法

有了环境模型，下一步是利用模型进行**规划**（Planning）。根据规划时机，分为两类：

> **Background Planning vs Decision-time Planning**
> - **Background Planning**：在与真实环境交互之外，利用模型生成模拟经验来训练策略
> - **Decision-time Planning**：在需要做决策时，利用模型进行前向搜索，选择最优动作

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=3.5cm, minimum height=1.2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % Background Planning
    \begin{scope}[shift={(-4.5, 0)}]
        \node[box, fill=green!20] (bg) at (0, 0) {Background\\Planning};
        \node[font=\small, align=center] at (0, -2) {离线生成经验\\训练策略网络\\代表：Dyna};
        \node[font=\bfseries] at (0, 1.5) {训练时规划};
    \end{scope}

    % Decision-time Planning
    \begin{scope}[shift={(4.5, 0)}]
        \node[box, fill=orange!20] (dt) at (0, 0) {Decision-time\\Planning};
        \node[font=\small, align=center] at (0, -2) {在线搜索决策\\不训练网络\\代表：MCTS};
        \node[font=\bfseries] at (0, 1.5) {决策时规划};
    \end{scope}
\end{tikzpicture}
</script>
</div>

### Dyna 架构

Dyna 是 Background Planning 的经典框架，由 Sutton 于 1991 年提出。其核心思想是：**每次真实交互后，用模型生成多次模拟经验来加速学习**。

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[scale=0.95,
    box/.style={draw, rounded corners, minimum width=2.8cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % 组件
    \node[box, fill=blue!20] (env) at (0, 2) {真实环境};
    \node[box, fill=green!20] (model) at (5, 2) {环境模型\\$\hat{P}, \hat{R}$};
    \node[box, fill=orange!20] (policy) at (2.5, -1) {策略/价值函数\\$Q(s,a)$};
    \node[box, fill=purple!15] (exp) at (-2.5, -1) {经验缓存\\$(s,a,r,s')$};

    % 连接
    \draw[arrow] (env) -- node[above, font=\small] {学习模型} (model);
    \draw[arrow] (env) -- node[left, font=\small, pos=0.3] {真实经验} (exp);
    \draw[arrow] (exp) -- node[below, font=\small] {直接学习} (policy);
    \draw[arrow, green!60!black, very thick] (model) -- node[right, font=\small, pos=0.3] {模拟经验\\（$n$ 次）} (policy);
    \draw[arrow, dashed] (policy.north) to[out=120, in=240] node[left, font=\small] {动作} (env.south);

    % 标注
    \node[font=\scriptsize, red] at (5, 0.3) {每步真实交互};
    \node[font=\scriptsize, red] at (5, -0.1) {可生成 $n$ 步模拟};
\end{tikzpicture}
</script>
</div>

**Dyna-Q 算法**：

```
输入: 规划步数 n，学习率 α，探索率 ε
初始化 Q(s,a) ← 0，表格模型 Model(s,a) ← ∅

对每个 episode:
    初始化状态 s
    while s 不是终止状态:
        a ← ε-greedy(Q(s, ·))
        执行 a，观察 r, s'

        // 直接 RL 学习
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s', a') - Q(s,a)]

        // 更新模型
        Model(s,a) ← (r, s')  // 确定性模型

        // 规划：从模型中学习
        for i = 1 to n:
            随机选择之前访问过的状态-动作对 (s̃, ã)
            (r̃, s̃') ← Model(s̃, ã)
            Q(s̃,ã) ← Q(s̃,ã) + α[r̃ + γ max_a' Q(s̃', a') - Q(s̃,ã)]

        s ← s'
```

**Dyna 的核心优势**：
1. **样本效率提升**：每次真实交互可产生 n 次模拟学习
2. **灵活的计算-样本权衡**：增大 n 可用更多计算换取更少真实交互
3. **渐进收敛**：当模型准确时，理论上与直接学习收敛到相同策略

### Decision-time Planning

与 Background Planning 不同，Decision-time Planning 在每次决策时进行规划：

1. 从当前状态出发，用模型模拟多条可能的轨迹
2. 评估每条轨迹的回报
3. 选择最优的第一步动作
4. 执行后重新规划（不保存中间结果）

Decision-time Planning 的特点：
- **计算集中**：所有计算都为当前决策服务
- **动态精度**：可根据需要调整搜索深度和广度
- **无需训练**：可直接用于测试时

最著名的 Decision-time Planning 方法是 **Monte Carlo Tree Search (MCTS)**。

## Monte Carlo Tree Search (MCTS)

MCTS 是一种基于树搜索的 Decision-time Planning 方法，广泛应用于棋类游戏和组合优化问题。

### MCTS 的核心思想

MCTS 的目标是在有限的计算预算内，估计当前状态下各动作的价值。其核心思想是：**有选择性地扩展搜索树，把计算资源集中在最有希望的分支上**。

> *如何决定哪个分支"最有希望"？这需要平衡**利用**（选择已知好的分支）和**探索**（尝试不确定的分支）。*

### MCTS 四步流程

MCTS 的每次迭代包含四个步骤：

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
        \node[font=\scriptsize, align=center] at (0, -3) {沿树用 UCB\\选择子节点};
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
        \node[font=\scriptsize, align=center] at (0, -3) {扩展一个\\新子节点};
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
        \node[font=\scriptsize, align=center] at (0, -4.5) {Rollout 或\\价值网络};
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
        \node[font=\scriptsize, align=center] at (0, -3) {沿路径\\更新统计};
    \end{scope}
\end{tikzpicture}
</script>
</div>

1. **Selection（选择）**：从根节点开始，使用**树策略**（如 UCB）递归选择子节点，直到到达叶节点（未完全扩展的节点）。

2. **Expansion（扩展）**：如果叶节点不是终止状态，根据可行动作扩展一个或多个新的子节点。

3. **Evaluation（评估）**：评估新扩展节点的价值。传统方法使用 **rollout**（随机模拟到终局）；现代方法使用**价值网络**直接估计。

4. **Backup（回溯）**：将评估值沿选择路径回传，更新路径上所有节点的访问次数 $N$ 和价值估计 $Q$。

### UCB 公式

Selection 阶段的核心是 **UCB（Upper Confidence Bound）**公式，它优雅地平衡了利用与探索：

> **UCB for Trees (UCT)**
>
> 在 Selection 阶段，选择最大化以下值的动作：
>
> $$\text{UCB}(s, a) = \underbrace{Q(s, a)}_{\text{利用}} + \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{探索}}$$
>
> 其中：
> - $Q(s, a)$：动作 $a$ 的平均价值估计（从历史模拟中统计）
> - $N(s)$：状态 $s$ 的总访问次数
> - $N(s, a)$：在状态 $s$ 执行动作 $a$ 的次数
> - $c$：探索系数，控制探索-利用权衡

UCB 的直觉理解：
- **利用项** $Q(s,a)$：选择历史表现好的动作
- **探索项**：选择访问次数少的动作（不确定性高）
- 随着访问次数增加，探索奖励逐渐减小，最终由利用主导
- $c$ 越大，越倾向探索；$c$ 越小，越倾向利用

### MCTS 算法

```
输入: 当前状态 s₀，搜索预算 B（迭代次数），探索系数 c
输出: 最优动作 a*

初始化根节点 root = s₀，N(root) = 0

for i = 1 to B:
    // Selection
    node ← root
    while node 已完全扩展 且 不是终止状态:
        a ← argmax_a UCB(node, a)
        node ← child(node, a)

    // Expansion
    if node 不是终止状态:
        选择一个未扩展的动作 a
        node ← 扩展子节点 child(node, a)

    // Evaluation
    v ← Evaluate(node)  // Rollout 或价值网络

    // Backup
    while node ≠ null:
        N(node) ← N(node) + 1
        Q(node) ← Q(node) + (v - Q(node)) / N(node)
        node ← parent(node)

return argmax_a N(root, a)  // 选择访问次数最多的动作
```

> **重要**：MCTS 最终选择动作的标准：
> - 训练/搜索时：用 UCB（平衡探索-利用）
> - 最终决策时：选择**访问次数最多**的动作（更稳健）
>
> 选择访问次数而非平均价值，因为高访问次数意味着高置信度。

## AlphaGo 与 AlphaZero

AlphaGo 和 AlphaZero 是 MCTS + 深度学习 + Self-Play 的里程碑式成果。

### 围棋的挑战

围棋被认为是 AI 最难攻克的棋类游戏：

- **搜索空间巨大**：平均每步有 $\sim 200$ 种合法走法，一局棋约 $200$ 步，总状态数 $\sim 10^{170}$
- **局面评估困难**：不像国际象棋有明确的子力价值，围棋的局面优劣难以量化
- **长期规划**：需要考虑数十步后的战略影响

传统围棋 AI 使用穷举搜索 + 手工评估函数，水平仅达业余段位。

### AlphaGo 架构（2016）

AlphaGo 在 2016 年以 4:1 击败世界冠军李世石，其架构包括：

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1.2cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % 输入
    \node[box, fill=blue!20] (input) at (0, 0) {棋盘状态\\$19 \times 19$};

    % Policy Network
    \node[box, fill=green!20] (pn) at (-3.5, -2.5) {Policy Network\\$p_\theta(a|s)$};

    % Value Network
    \node[box, fill=orange!20] (vn) at (3.5, -2.5) {Value Network\\$v_\phi(s)$};

    % MCTS
    \node[box, fill=purple!20, minimum width=4cm] (mcts) at (0, -5) {MCTS 搜索};

    % 输出
    \node[box, fill=red!15] (output) at (0, -7.5) {最终动作};

    % 连接
    \draw[arrow] (input) -- (pn);
    \draw[arrow] (input) -- (vn);
    \draw[arrow] (pn) -- node[left, font=\small] {指导选择} (mcts);
    \draw[arrow] (vn) -- node[right, font=\small] {评估叶节点} (mcts);
    \draw[arrow] (mcts) -- (output);

    % 训练方式标注
    \node[font=\scriptsize, align=left] at (-6.5, -2.5) {监督学习\\（人类棋谱）\\+ RL 微调};
    \node[font=\scriptsize, align=right] at (6.5, -2.5) {监督学习\\（自我对弈\\结果预测）};
\end{tikzpicture}
</script>
</div>

1. **Policy Network** $p_\theta(a\|s)$：
   - 输入：棋盘状态（多通道特征）
   - 输出：每个位置的落子概率
   - 训练：先用人类棋谱监督学习，再用 Policy Gradient 自我对弈强化

2. **Value Network** $v_\phi(s)$：
   - 输入：棋盘状态
   - 输出：当前局面的胜率估计 $v_\phi(s) \approx \mathbb{E}[z\|s]$，其中 $z \in \\{-1, +1\\}$
   - 训练：用自我对弈生成的 $(s, z)$ 数据监督学习

3. **改进的 MCTS**：
   - Selection：使用 Policy Network 指导（PUCT 公式）

   $$\text{UCB}(s,a) = Q(s,a) + c \cdot p_\theta(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

   - Evaluation：混合 Value Network 和 rollout

   $$v = (1-\lambda) v_\phi(s) + \lambda z_{\text{rollout}}$$

### AlphaZero 的简化与超越（2017）

AlphaZero 在 2017 年大幅简化了 AlphaGo 的设计，却取得了更强的性能：

| 特性 | AlphaGo | AlphaZero |
|------|---------|-----------|
| 人类棋谱 | 需要（监督预训练） | **不需要** |
| 网络结构 | Policy + Value 分离 | **统一网络** |
| Rollout | 需要 | **不需要** |
| 特征工程 | 手工设计特征 | **原始棋盘输入** |
| 训练时间 | 数月 | **数小时** |
| 适用游戏 | 仅围棋 | **围棋、国际象棋、将棋** |

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % 统一网络
    \node[box, fill=blue!20] (input) at (0, 0) {棋盘状态 $s$};
    \node[box, fill=purple!25, minimum height=2cm] (net) at (0, -2.5) {ResNet\\（统一网络）};

    % 双头输出
    \node[box, fill=green!20] (policy) at (-2.5, -5) {$p_\theta(a|s)$\\策略头};
    \node[box, fill=orange!20] (value) at (2.5, -5) {$v_\theta(s)$\\价值头};

    \draw[arrow] (input) -- (net);
    \draw[arrow] (net) -- (policy);
    \draw[arrow] (net) -- (value);

    \node[font=\small, align=center] at (0, -6.5) {单个网络同时输出策略分布和价值估计\\共享底层表示，参数更少，训练更高效};
\end{tikzpicture}
</script>
</div>

### AlphaZero 训练循环

AlphaZero 的训练是一个**自我增强**的循环：

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=3cm, minimum height=1.2cm, align=center},
    arrow/.style={->, very thick, >=stealth}
]
    % 三个组件
    \node[box, fill=green!20] (selfplay) at (0, 0) {Self-Play\\生成对弈数据};
    \node[box, fill=orange!20] (train) at (5, -3) {网络训练\\学习搜索结果};
    \node[box, fill=blue!20] (network) at (-5, -3) {神经网络\\$(p_\theta, v_\theta)$};

    % 循环箭头
    \draw[arrow, green!60!black] (selfplay) -- node[right, font=\small, pos=0.5] {$(s, \pi_{\text{MCTS}}, z)$} (train);
    \draw[arrow, orange] (train) -- node[below, font=\small, yshift=-3pt] {更新 $\theta$} (network);
    \draw[arrow, blue] (network) -- node[left, font=\small, pos=0.5] {指导搜索} (selfplay);

    % 中心说明
    \node[font=\small, align=center, text=gray] at (0, -1.8) {正向循环\\不断变强};
\end{tikzpicture}
</script>
</div>

**AlphaZero 训练算法**：

```
初始化网络参数 θ（随机初始化）

repeat 直到收敛:
    // Self-Play 生成数据
    for 每局对弈:
        for 每步 t:
            用当前网络 + MCTS 搜索，得到 π_MCTS(a|s_t)
            按 π_MCTS 采样动作 a_t
            记录 (s_t, π_MCTS)

        游戏结束，得到胜负 z ∈ {-1, +1}
        将 (s_t, π_MCTS, z) 加入训练数据

    // 网络训练
    从训练数据中采样 batch
    最小化损失：L(θ) = (z - v_θ(s))² - π_MCTS^T log p_θ(s) + c‖θ‖²
```

其中损失函数各项含义：
- **$(z - v_\theta(s))^2$**：价值损失，让价值预测接近游戏结果
- **$-\pi_{\text{MCTS}}^\top \log p_\theta(s)$**：策略损失，让策略接近 MCTS 搜索结果
- **$c\|\theta\|^2$**：L2 正则化项

**AlphaZero 的核心洞察**：
1. **MCTS 作为策略改进**：搜索产生的 $\pi_{\text{MCTS}}$ 比原始网络 $p_\theta$ 更好
2. **网络学习搜索**：网络被训练去模仿 MCTS 的输出
3. **正向循环**：更好的网络 → 更好的搜索 → 更好的训练数据 → 更好的网络

这个循环不需要任何人类知识，完全从零开始（tabula rasa）学习。

## Multi-Agent RL 基础

当环境中存在多个 agent 时，问题变得更加复杂。

### 从单 Agent 到多 Agent

> **核心问题**：当其他 agent 也在学习和改变策略时，环境对单个 agent 来说是**非稳态**的。这打破了 MDP 的基本假设。

在 Multi-Agent 环境中，状态转移和奖励不仅依赖于自己的动作，还依赖于其他 agent 的动作：

$$P(s'|s, a_1, a_2, \ldots, a_n), \quad R_i(s, a_1, a_2, \ldots, a_n)$$

当其他 agent 的策略 $\pi_{-i}$ 在学习过程中变化时，从 agent $i$ 的视角看，环境是非稳态的。

非稳态性的影响：
- 单 agent RL 的收敛性保证不再适用
- 最优响应策略随对手策略而变化
- 可能出现策略震荡，无法收敛

### 博弈论基础

Multi-Agent 问题可以用博弈论的语言来描述。

> **Normal-form Game**
>
> 一个 $n$ 人博弈由以下元素组成：
> - **玩家集合**：$\mathcal{N} = \\{1, 2, \ldots, n\\}$
> - **策略空间**：每个玩家 $i$ 的策略集合 $\mathcal{A}_i$
> - **效用函数**：$u_i: \mathcal{A}_1 \times \cdots \times \mathcal{A}_n \to \mathbb{R}$，表示每种策略组合下玩家 $i$ 的收益

**囚徒困境示例**：

两个嫌犯被分开审讯，各自选择"合作"（沉默）或"背叛"（坦白）：

|  | 玩家2: 合作 | 玩家2: 背叛 |
|---|---|---|
| 玩家1: 合作 | (-1, -1) | (-3, 0) |
| 玩家1: 背叛 | (0, -3) | (-2, -2) |

分析：
- 无论对方如何选择，"背叛"对自己都更有利（支配策略）
- 结果：双方都背叛，各获 -2
- 但如果双方都合作，各获 -1（帕累托更优）

### 合作与竞争设定

Multi-Agent 场景主要分为两类：

1. **合作（Cooperative）**：所有 agent 共享奖励，最大化团队总收益
   - 例：多机器人协作搬运、多 agent 协调导航
   - 挑战：信用分配（Credit Assignment）——哪个 agent 贡献了多少？
   - 方法：集中训练、分布式执行（CTDE）

2. **竞争/零和（Competitive/Zero-sum）**：一方获益等于另一方损失
   - 例：棋类游戏、对抗博弈
   - 特点：$u_1 + u_2 = 0$
   - 目标：找到 Nash 均衡

### Nash 均衡

> **Nash 均衡**
>
> 策略组合 $(\pi_1^\*, \pi_2^\*, \ldots, \pi_n^\*)$ 是 Nash 均衡，当且仅当没有任何玩家有动机单方面改变自己的策略：
>
> $$\forall i, \forall \pi_i: \quad u_i(\pi_i^*, \pi_{-i}^*) \geq u_i(\pi_i, \pi_{-i}^*)$$
>
> 其中 $\pi_{-i}^\*$ 表示除玩家 $i$ 外所有玩家的策略。

Nash 均衡的含义：
- 每个玩家都在对其他玩家的策略做**最优响应**
- 是一种**稳定状态**：没有人有动机单方面偏离
- 不一定是全局最优（如囚徒困境中双方背叛是 Nash 均衡，但不是帕累托最优）

**石头剪刀布的 Nash 均衡**：

|  | 石头 | 剪刀 | 布 |
|---|---|---|---|
| 石头 | (0, 0) | (1, -1) | (-1, 1) |
| 剪刀 | (-1, 1) | (0, 0) | (1, -1) |
| 布 | (1, -1) | (-1, 1) | (0, 0) |

Nash 均衡：双方都以 $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ 的概率随机选择。

任何确定性策略都可以被对手利用，只有**混合策略**（随机化）才能达到均衡。

> **Nash 均衡存在性定理**：每个有限博弈（有限玩家、有限策略）至少存在一个 Nash 均衡（可能是混合策略均衡）。

## Self-Play 方法

Self-Play 是训练博弈 AI 的强大方法，也是 AlphaGo/AlphaZero 成功的关键之一。

### Self-Play 的定义

> **Self-Play**：Agent 与自己（或自己的历史版本）进行对弈，从对弈经验中学习改进策略。

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.5cm, minimum height=1cm, align=center},
    arrow/.style={->, thick, >=stealth}
]
    % 当前策略
    \node[box, fill=blue!20] (current) at (0, 0) {当前策略\\$\pi_\theta$};

    % 对手（自己的副本）
    \node[box, fill=blue!10] (opponent) at (5, 0) {对手\\$\pi_\theta$ 或 $\pi_{\theta'}$};

    % 对弈
    \node[box, fill=green!20] (game) at (2.5, -2.5) {对弈};

    % 经验
    \node[box, fill=orange!20] (exp) at (2.5, -5) {对弈经验\\$(s, a, r, s')$};

    % 更新
    \draw[arrow] (current) -- (game);
    \draw[arrow] (opponent) -- (game);
    \draw[arrow] (game) -- (exp);
    \draw[arrow] (exp) to[out=180, in=270] node[left, font=\small] {更新} (current);

    % 可能的历史对手池
    \node[box, fill=gray!20, dashed] (pool) at (8, -2.5) {历史对手池\\$\{\pi_{\theta_1}, \ldots\}$};
    \draw[arrow, dashed] (pool) -- (opponent);
\end{tikzpicture}
</script>
</div>

### Self-Play 的优势

1. **无限数据**：可以生成任意多的对弈数据，不受人类对局数量限制

2. **自适应难度**：对手随自己一起变强，始终提供适当的挑战
   - 初期：对手弱，容易学习基本策略
   - 后期：对手强，推动学习高级策略

3. **发现新策略**：不受人类先验知识限制，可能发现人类未知的创新策略
   - AlphaGo 的"肩冲"等新招法震惊了职业棋手

4. **逼近 Nash 均衡**：在零和博弈中，Self-Play 在理论上收敛到 Nash 均衡

### Self-Play 的挑战

1. **策略遗忘**：
   - 当策略更新后，可能"忘记"如何对付旧策略
   - 解决：维护历史对手池（Opponent Pool），随机抽取对手

2. **局部最优**：
   - 可能陷入"只擅长对付自己"的局部最优
   - 例：两个版本互相克制，形成循环
   - 解决：添加多样性奖励，或从对手池采样

3. **评估困难**：
   - 没有固定基准来衡量进步
   - 解决：用 Elo 评分系统或与固定对手对战

**Self-Play 与 Nash 均衡的关系**：
- 在两人零和博弈中，如果 Self-Play 收敛，则收敛到 Nash 均衡
- 直觉：Nash 均衡是"最优响应的不动点"，Self-Play 就是迭代求最优响应
- 但收敛不保证——可能出现策略循环

## 本章总结

1. **Model-Based RL** 利用环境模型提高样本效率
   - World Model = 状态转移 + 奖励模型
   - Model Bias：模型误差会累积，需要短期规划或持续校正

2. **Dyna 架构**结合直接学习与规划
   - 每次真实交互后，用模型生成 n 次模拟经验
   - 提供计算-样本的灵活权衡

3. **MCTS** 是 Decision-time Planning 的代表
   - 四步流程：Selection, Expansion, Evaluation, Backup
   - UCB 公式平衡探索与利用

4. **AlphaGo/AlphaZero** 展示了 MCTS + 深度学习 + Self-Play 的强大组合
   - AlphaZero 从零开始，无需人类知识
   - 核心循环：MCTS 改进策略 → 网络学习搜索 → 正向增强

5. **Multi-Agent RL** 面临非稳态性挑战
   - Nash 均衡：稳定的策略组合
   - Self-Play：训练博弈 AI 的有效方法

<div class="tikz-container">
<script type="text/tikz">
\begin{tikzpicture}[
    box/.style={draw, rounded corners, fill=blue!10, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % 层次结构
    \node[box, fill=blue!25] (rl) at (0, 0) {RL 方法};

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

    % AlphaZero 横跨
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

下一篇博客将进入 LLM 与 RL 的结合领域，介绍 RLHF 和 DPO 等用于语言模型对齐的方法。
