---
layout: post
title: "Transformer Notes (IV): Mixture of Experts Architecture"
date: 2025-12-20 10:30:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [Transformer, MoE]
lang: en
translation: /transformer-part4-moe/
series: transformer
series_order: 4
---

This is the fourth article in the Transformer series, providing an in-depth analysis of the **Mixture of Experts (MoE)** sparse activation architecture. MoE achieves the goal of "large model capacity with small model compute" by activating only a subset of parameters for each token, and is the core architecture of frontier models like DeepSeek-V3 and Kimi K2.

## 1. Core Concepts of MoE

### 1.1 From Dense to Sparse

In traditional dense models, every token passes through all parameters. The core idea of MoE is to use a **Router** to select the most relevant **Experts** for each token, activating only a subset of parameters:

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

where $E_i$ is the $i$-th expert (typically an FFN), and $g_i(x)$ is the weight assigned by the router to expert $i$ for token $x$.

### 1.2 Top-K Routing Mechanism

Standard Top-K routing:

$$s_i = x \cdot W_r^{(i)} \quad \text{(routing score)}$$

$$g_i = \begin{cases}
\text{softmax}(s)_i & \text{if } i \in \text{Top-}K(s) \\
0 & \text{otherwise}
\end{cases}$$

where $W_r$ is the learnable parameter of the router. Each token is only sent to the $K$ experts with the highest scores.

### 1.3 Key Terminology

| Term | Meaning |
|------|------|
| Total Parameters | All model parameters (including all experts) |
| Activated Parameters | Parameters used in forward pass for a single token |
| Number of Experts $N$ | Total number of available experts |
| Activated Experts $K$ | Number of experts selected per token |
| Sparsity | $N/K$, higher values indicate greater sparsity |

## 2. DeepSeek MoE Architecture

DeepSeek has proposed the most influential MoE design to date, adopted by models like DeepSeek-V2, V3, and R1.

### 2.1 Fine-grained Expert Segmentation

Traditional MoE uses a small number of large experts (e.g., 8). DeepSeek proposes **Fine-grained Expert Segmentation**: increase the number of experts by $m$ times while reducing each expert's parameters by $m$ times:

$$N \to mN, \quad K \to mK, \quad \text{Expert Size} \to \frac{1}{m}$$

**Advantage**: More expert combinations provide more flexible knowledge representation.

- Selecting 2 from 8 experts: $\binom{8}{2} = 28$ combinations
- Selecting 16 from 64 experts: $\binom{64}{16} \approx 4.9 \times 10^{14}$ combinations

This combinatorial explosion brings exponential improvement in representation capacity.

### 2.2 Shared Expert Isolation

In addition to routed experts, DeepSeek introduces **Shared Experts**:

$$y = \underbrace{\sum_{i=1}^{K_s} E_i^{\text{shared}}(x)}_{\text{Shared Experts}} + \underbrace{\sum_{j=1}^{K_r} g_j(x) \cdot E_j^{\text{routed}}(x)}_{\text{Routed Experts}}$$

**Design Philosophy**:

- **Shared Experts**: Capture universal knowledge needed by all tokens (e.g., syntax, common sense)
- **Routed Experts**: Capture domain-specific knowledge (e.g., mathematics, code, medicine)

This separation reduces knowledge redundancy among routed experts and improves specialization.

### 2.3 DeepSeek Model Configurations

| Model | Total Params | Active Params | Expert Config |
|------|--------|----------|----------|
| DeepSeek-V2 | 236B | 21B | 160 routed + 2 shared |
| DeepSeek-V3 | 671B | 37B | 256 routed + 1 shared |
| DeepSeek-R1 | 671B | 37B | Same as V3 |

DeepSeek-V3 activates 8 routed experts + 1 shared expert per token, achieving a sparsity of $256/8 = 32$.

## 3. Load Balancing Strategies

The core challenge in MoE training is **load balancing**. If certain experts are over-selected, it leads to:

1. **Routing Collapse**: All tokens select the same few experts
2. **Decreased Computational Efficiency**: Unbalanced load during expert parallelism
3. **Wasted Knowledge**: Some experts are never trained

### 3.1 Traditional Method: Auxiliary Loss

Early methods (e.g., Switch Transformer) use auxiliary loss to force load balancing:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the proportion of tokens actually processed by expert $i$, and $P_i$ is the average routing score.

**Problem**: Auxiliary loss competes with the main task loss; if $\alpha$ is too large, it harms model performance.

### 3.2 DeepSeek-V2: Multi-level Auxiliary Loss

DeepSeek-V2 introduces three-level auxiliary loss:

1. **Expert-level**: Balance the load of individual experts
2. **Device-level**: Balance expert load across different devices
3. **Communication-level**: Reduce cross-device communication

### 3.3 DeepSeek-V3: Auxiliary-Loss-Free Load Balancing

DeepSeek-V3 proposes revolutionary **Auxiliary-Loss-Free load balancing**:

**Core Idea**: Introduce an adjustable bias term $b_i$ for each expert, used only for routing decisions and not participating in loss calculation:

$$s_i' = s_i + b_i$$

**Dynamic Adjustment**:

- When expert is overloaded, decrease $b_i$ to reduce selection probability
- When expert is idle, increase $b_i$ to improve selection probability

**Key Advantage**: Load balancing objectives are completely decoupled from quality optimization objectives, eliminating competition. Experiments show V3 maintains good load balancing throughout training without dropping any tokens.

## 4. Routing Constraints and Communication Optimization

### 4.1 Node-Limited Routing

In distributed training, experts are distributed across different nodes, and cross-node communication costs are high. DeepSeek introduces **node-limited routing**:

> Each token is sent to at most $M$ nodes.

This limits the scope of All-to-All communication, significantly reducing communication overhead.

### 4.2 Expert Tensor Parallelism

MiniMax proposes **Expert Tensor Parallel** (ETP): split the parameters of a single expert across multiple devices, rather than placing different experts on different devices. This approach is better suited for fine-grained expert architectures.

## 5. Industrial MoE Model Comparison

### 5.1 Overview of Mainstream Models

| Model | Total Params | Active Params | Expert Config | Features |
|------|--------|----------|----------|------|
| DeepSeek-V3 | 671B | 37B | 256+1 | Auxiliary-loss-free balancing |
| MiniMax-01 | 456B | 45.9B | 32 experts | Lightning Attention |
| Kimi K2 | 1T | 32B | 384 routed | MuonClip optimizer |
| Qwen2-57B-A14B | 57B | 14B | 60+4 shared | Upcycling |

### 5.2 DeepSeek-V3

**Key Innovations**:

- Auxiliary-Loss-Free load balancing
- Multi-Token Prediction (MTP)
- FP8 mixed precision training
- Completed training with only 2.788M H800 GPU hours

**Performance**: 671B parameters, activating 37B per token, achieving GPT-4-level performance on multiple benchmarks.

### 5.3 MiniMax-01

**Architecture Features**:

- 32 experts, activating approximately 45.9B parameters per token
- Hybrid architecture combining Lightning Attention
- 1 Softmax attention layer every 7 linear attention layers

**Long Context**: Trained on 1M tokens, can extend to 4M tokens during inference.

### 5.4 Kimi K2

**Scale**: 1T total parameters, 32B active parameters—one of the largest open-source MoE models.

**Architecture**:

- MLA + MoE architecture similar to DeepSeek-V3
- 384 routed experts, activating 8 per token
- Sparsity: $384/8 = 48$ (higher than DeepSeek-V3's 32)

**Training**: Uses Muon optimizer (MuonClip variant), trained on 15.5T tokens with zero training instability.

### 5.5 Qwen MoE

Qwen adopts the **Upcycling** strategy: initializing MoE experts from a dense model.

**Qwen2-57B-A14B**:

- Upcycled from Qwen2-7B
- 60 routed experts + 4 shared experts
- Activates 14B parameters, performance close to 34B dense model

## 6. Theoretical Understanding of MoE

### 6.1 Sparsity vs. Capacity Trade-off

The core trade-off in MoE is **sparsity** vs. **model capacity**:

- More experts → Greater capacity, but increased communication overhead
- Fewer activated experts → Higher efficiency, but potential underfitting

DeepSeek-V3's experience: 256 experts + 8 activated is a good balance.

### 6.2 Expert Specialization

Ideally, different experts should learn to handle different types of knowledge:

- Some experts handle mathematical reasoning
- Some experts handle code generation
- Some experts handle multilingual tasks

The introduction of shared experts helps routed experts specialize better, avoiding "every expert learns a bit of general knowledge."

### 6.3 Choosing Between MoE and Dense

MoE is not always superior to dense:

| Dimension | MoE Advantage | Dense Advantage |
|------|----------|------------|
| Capacity | Greater capacity under same compute budget | Simpler training and deployment |
| Inference | More efficient during inference | More stable on certain tasks |
| Scale | Preferred for ultra-large models | Still mainstream for small-to-medium scale |

The current trend is to use MoE for ultra-large models (100B+), while small-to-medium scale models remain predominantly dense.

## 7. MoE Training Tips

### 7.1 Load Balancing

- Prioritize DeepSeek-V3's auxiliary-loss-free method
- If using auxiliary loss, coefficient $\alpha$ needs careful tuning (typically $0.01 \sim 0.1$)

### 7.2 Expert Parallelism

- **Small scale**: All experts on a single device
- **Medium scale**: Expert Parallelism, different experts on different devices
- **Large scale**: Hybrid parallelism combining TP, EP, PP

### 7.3 Upcycling

Initializing MoE from a dense model can accelerate convergence:

1. Replicate the FFN of the dense model as initialization for each expert
2. Randomly initialize the router
3. Continue pretraining, experts gradually diverge

### 7.4 Capacity Factor

Traditional MoE sets a capacity factor to limit the maximum number of tokens each expert can process, with excess tokens being dropped. DeepSeek-V3 proves: a good load balancing strategy can **completely avoid token dropping**.

## 8. Summary

This article provides an in-depth analysis of the core design of MoE sparse architecture:

| Component | Key Techniques | Representative Work |
|------|----------|----------|
| Expert Design | Fine-grained segmentation + shared experts | DeepSeek MoE |
| Load Balancing | Auxiliary-loss-free dynamic adjustment | DeepSeek-V3 |
| Communication Optimization | Node-limited routing + ETP | DeepSeek/MiniMax |
| Initialization | Upcycling | Qwen MoE |

MoE architecture makes training and inference of hundred-billion parameter models feasible and represents an important direction in current large model development.

In the next article, we will discuss **Training Techniques**, including data processing, distributed training strategies, and novel optimizers.
