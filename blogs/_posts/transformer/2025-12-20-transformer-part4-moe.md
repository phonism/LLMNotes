---
layout: post
title: "Transformer 学习笔记（四）：Mixture of Experts 稀疏架构"
date: 2025-12-20 10:30:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [MoE, DeepSeek, Sparse Model, Expert Parallelism]
math: true
lang: zh
translation: /en/transformer-part4-moe/
series: transformer
series_order: 4
---

本文是 Transformer 系列的第四篇，深入解析 **Mixture of Experts (MoE)** 稀疏激活架构。MoE 通过让每个 token 只激活部分参数，实现了"大模型容量、小模型计算量"的目标，是 DeepSeek-V3、Kimi K2 等前沿模型的核心架构。

## 1. MoE 核心思想

### 1.1 从 Dense 到 Sparse

传统 Dense 模型中，每个 token 都要经过所有参数。MoE 的核心思想是：用 **路由器**（Router）为每个 token 选择最相关的 **专家**（Expert），只激活部分参数：

$$y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)$$

其中 $E_i$ 是第 $i$ 个专家（通常是 FFN），$g_i(x)$ 是路由器为 token $x$ 分配给专家 $i$ 的权重。

### 1.2 Top-K 路由机制

标准的 Top-K 路由：

$$s_i = x \cdot W_r^{(i)} \quad \text{(路由分数)}$$

$$g_i = \begin{cases}
\text{softmax}(s)_i & \text{if } i \in \text{Top-}K(s) \\
0 & \text{otherwise}
\end{cases}$$

其中 $W_r$ 是路由器的可学习参数。每个 token 只被发送到 $K$ 个得分最高的专家。

### 1.3 关键术语

| 术语 | 含义 |
|------|------|
| 总参数量 | 模型所有参数（包括所有专家） |
| 激活参数量 | 单个 token 前向传播使用的参数 |
| 专家数 $N$ | 可选专家的总数 |
| 激活专家数 $K$ | 每个 token 选择的专家数 |
| 稀疏度 | $N/K$，越大表示越稀疏 |

## 2. DeepSeek MoE 架构

DeepSeek 提出了目前最具影响力的 MoE 设计，被 DeepSeek-V2、V3、R1 等模型采用。

### 2.1 细粒度专家分割

传统 MoE 使用少量大专家（如 8 个），DeepSeek 提出 **Fine-grained Expert Segmentation**：将专家数量增加 $m$ 倍，同时将每个专家的参数减少 $m$ 倍：

$$N \to mN, \quad K \to mK, \quad \text{Expert Size} \to \frac{1}{m}$$

**优势**：更多专家组合提供更灵活的知识表示。

- 从 8 个专家选 2 个：$\binom{8}{2} = 28$ 种组合
- 从 64 个专家选 16 个：$\binom{64}{16} \approx 4.9 \times 10^{14}$ 种组合

组合爆炸带来了指数级的表示能力提升。

### 2.2 共享专家隔离

除了路由专家外，DeepSeek 引入 **Shared Experts**：

$$y = \underbrace{\sum_{i=1}^{K_s} E_i^{\text{shared}}(x)}_{\text{共享专家}} + \underbrace{\sum_{j=1}^{K_r} g_j(x) \cdot E_j^{\text{routed}}(x)}_{\text{路由专家}}$$

**设计哲学**：

- **共享专家**：捕获所有 token 都需要的通用知识（如语法、常识）
- **路由专家**：捕获领域特定知识（如数学、代码、医学）

这种分离减少了路由专家之间的知识冗余，提高了专业化程度。

### 2.3 DeepSeek 模型配置

| 模型 | 总参数 | 激活参数 | 专家配置 |
|------|--------|----------|----------|
| DeepSeek-V2 | 236B | 21B | 160 路由 + 2 共享 |
| DeepSeek-V3 | 671B | 37B | 256 路由 + 1 共享 |
| DeepSeek-R1 | 671B | 37B | 同 V3 |

DeepSeek-V3 每个 token 激活 8 个路由专家 + 1 个共享专家，稀疏度高达 $256/8 = 32$。

## 3. 负载均衡策略

MoE 训练的核心挑战是 **负载均衡**。如果某些专家被过度选择，会导致：

1. **路由崩塌**（Routing Collapse）：所有 token 都选同几个专家
2. **计算效率下降**：专家并行时负载不均
3. **知识浪费**：部分专家从未被训练

### 3.1 传统方法：辅助损失

早期方法（如 Switch Transformer）使用辅助损失强制负载均衡：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 $i$ 实际处理的 token 比例，$P_i$ 是路由分数的平均值。

**问题**：辅助损失与主任务损失相互竞争，$\alpha$ 过大会损害模型性能。

### 3.2 DeepSeek-V2：多级辅助损失

DeepSeek-V2 引入三级辅助损失：

1. **Expert-level**：平衡单个专家的负载
2. **Device-level**：平衡不同设备上的专家负载
3. **Communication-level**：减少跨设备通信

### 3.3 DeepSeek-V3：无辅助损失负载均衡

DeepSeek-V3 提出革命性的 **Auxiliary-Loss-Free 负载均衡**：

**核心思想**：为每个专家引入可调偏置项 $b_i$，仅用于路由决策，不参与 loss 计算：

$$s_i' = s_i + b_i$$

**动态调整**：

- 专家过载时，减小 $b_i$ 降低被选概率
- 专家空闲时，增大 $b_i$ 提高被选概率

**关键优势**：负载均衡目标与质量优化目标完全解耦，不再相互竞争。实验表明 V3 在整个训练过程中保持良好的负载均衡，无需丢弃任何 token。

## 4. 路由约束与通信优化

### 4.1 Node-Limited Routing

在分布式训练中，专家分布在不同节点上，跨节点通信成本高昂。DeepSeek 引入 **节点限制路由**：

> 每个 token 最多被发送到 $M$ 个节点。

这限制了 All-to-All 通信的范围，大幅减少通信开销。

### 4.2 Expert Tensor Parallelism

MiniMax 提出 **Expert Tensor Parallel**（ETP）：将单个专家的参数切分到多个设备，而非将不同专家放在不同设备。这种方式更适合细粒度专家架构。

## 5. 工业界 MoE 模型对比

### 5.1 主流模型概览

| 模型 | 总参数 | 激活参数 | 专家配置 | 特点 |
|------|--------|----------|----------|------|
| DeepSeek-V3 | 671B | 37B | 256+1 | 无辅助损失均衡 |
| MiniMax-01 | 456B | 45.9B | 32 专家 | Lightning Attention |
| Kimi K2 | 1T | 32B | 384 路由 | MuonClip 优化器 |
| Qwen2-57B-A14B | 57B | 14B | 60+4 共享 | Upcycling |

### 5.2 DeepSeek-V3

**关键创新**：

- Auxiliary-Loss-Free 负载均衡
- Multi-Token Prediction (MTP)
- FP8 混合精度训练
- 仅需 2.788M H800 GPU 小时完成训练

**性能**：671B 参数，每 token 激活 37B，在多项 benchmark 上达到 GPT-4 级别。

### 5.3 MiniMax-01

**架构特点**：

- 32 个专家，每 token 激活约 45.9B 参数
- 结合 Lightning Attention 的混合架构
- 每 7 层线性注意力后接 1 层 Softmax 注意力

**长上下文**：训练 1M tokens，推理可扩展至 4M tokens。

### 5.4 Kimi K2

**规模**：1T 总参数，32B 激活参数——目前最大的开源 MoE 模型之一。

**架构**：

- 类似 DeepSeek-V3 的 MLA + MoE 架构
- 384 个路由专家，每 token 激活 8 个
- 稀疏度：$384/8 = 48$（高于 DeepSeek-V3 的 32）

**训练**：使用 Muon 优化器（MuonClip 变体），15.5T tokens 训练，零训练不稳定性。

### 5.5 Qwen MoE

Qwen 采用 **Upcycling** 策略：从 Dense 模型初始化 MoE 专家。

**Qwen2-57B-A14B**：

- 从 Qwen2-7B upcycle 而来
- 60 个路由专家 + 4 个共享专家
- 激活 14B 参数，性能接近 34B Dense 模型

## 6. MoE 理论理解

### 6.1 稀疏性与容量的权衡

MoE 的核心 trade-off 是 **稀疏性** 与 **模型容量**：

- 更多专家 → 更大容量，但通信开销增加
- 更少激活专家 → 更高效率，但可能欠拟合

DeepSeek-V3 的经验：256 个专家 + 8 个激活是一个 good balance。

### 6.2 专家专业化

理想情况下，不同专家应学会处理不同类型的知识：

- 某些专家处理数学推理
- 某些专家处理代码生成
- 某些专家处理多语言

共享专家的引入帮助路由专家更好地专业化，避免"每个专家都学一点通用知识"。

### 6.3 MoE vs Dense 的选择

MoE 并非总是优于 Dense：

| 维度 | MoE 优势 | Dense 优势 |
|------|----------|------------|
| 容量 | 相同计算预算下更大容量 | 更简单的训练和部署 |
| 推理 | 推理时更高效 | 在某些任务上更稳定 |
| 规模 | 超大规模模型首选 | 中小规模仍是主流 |

当前趋势是在超大规模模型（100B+）中使用 MoE，中小规模仍以 Dense 为主。

## 7. MoE 训练技巧

### 7.1 负载均衡

- 优先使用 DeepSeek-V3 的无辅助损失方法
- 如使用辅助损失，系数 $\alpha$ 需仔细调优（通常 $0.01 \sim 0.1$）

### 7.2 专家并行

- **小规模**：所有专家放单卡
- **中规模**：Expert Parallelism，不同专家在不同卡
- **大规模**：结合 TP、EP、PP 的混合并行

### 7.3 Upcycling

从 Dense 模型初始化 MoE 可以加速收敛：

1. 复制 Dense 模型的 FFN 作为各专家的初始化
2. 随机初始化路由器
3. 继续预训练，专家逐渐分化

### 7.4 容量因子

传统 MoE 设置容量因子（Capacity Factor）限制每个专家处理的最大 token 数，超出的 token 被丢弃。DeepSeek-V3 证明：良好的负载均衡策略可以 **完全避免 token 丢弃**。

## 8. 总结

本文深入解析了 MoE 稀疏架构的核心设计：

| 组件 | 关键技术 | 代表工作 |
|------|----------|----------|
| 专家设计 | 细粒度分割 + 共享专家 | DeepSeek MoE |
| 负载均衡 | 无辅助损失动态调整 | DeepSeek-V3 |
| 通信优化 | 节点限制路由 + ETP | DeepSeek/MiniMax |
| 初始化 | Upcycling | Qwen MoE |

MoE 架构使得千亿参数模型的训练和推理成为可能，是当前大模型发展的重要方向。

下一篇我们将讨论 **训练技术**，包括数据处理、分布式训练策略和新型优化器。
