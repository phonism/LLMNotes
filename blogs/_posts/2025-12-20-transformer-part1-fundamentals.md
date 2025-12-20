---
layout: post
title: "Transformer 学习笔记（一）：基础理论"
date: 2025-12-20 10:00:00
author: Phonism
tags: [Transformer, LLM, Scaling Law, GPU]
lang: zh
---

从 RNN 到 LSTM 再到 Transformer，序列建模的范式经历了根本性转变。2017 年 Vaswani 等人提出的 Transformer 架构完全摒弃循环结构，仅使用注意力机制建模序列依赖，开启了大语言模型（LLM）时代。

本文是 Transformer 系列笔记的第一篇，涵盖基础理论：硬件性能模型、Transformer 计算分析，以及指导大规模训练的 Scaling Law。

## 1. 引言：为什么是 Transformer

### 1.1 RNN 的困境

传统的序列建模方法——RNN、LSTM、GRU——存在固有的局限性：

- **顺序计算**：必须按时间步依次处理序列，难以并行化
- **长距离依赖**：尽管 LSTM 引入了门控机制，信息在长序列中仍会衰减
- **计算效率**：训练和推理速度受限于序列长度

### 1.2 Transformer 的革命

Transformer 通过自注意力机制直接建模任意位置之间的依赖关系：

- **完全并行化**：大幅提升训练效率
- **直接长程依赖**：注意力可以直接连接任意两个位置
- **灵活可扩展**：易于堆叠和迁移

基于 Transformer 的 LLM 在随后几年取得了惊人进展。从 GPT 系列到 LLaMA、DeepSeek 等开源模型，参数规模从亿级跃升至万亿级。这些模型不仅在 NLP 任务上表现卓越，还展现出强大的涌现能力：上下文学习、链式推理、代码生成等。

### 1.3 新的挑战

随着模型规模扩展，新的挑战不断涌现：

| 挑战 | 描述 |
|------|------|
| 计算效率 | 标准注意力的 $O(n^2)$ 复杂度限制了长上下文建模 |
| 训练成本 | 千亿参数模型需要数千 GPU 训练数月 |
| 部署挑战 | KV Cache 的内存占用成为推理瓶颈 |
| 能力边界 | 复杂推理、多模态理解等任务仍具挑战性 |

理解这些挑战的根源，需要从硬件性能模型开始。

## 2. 硬件与性能基础

### 2.1 Roofline 模型

Roofline 模型是分析程序性能的经典框架，通过三个基本约束界定算法的性能上限：

- 计算速度（FLOPs/秒）
- 数据传输带宽（字节/秒）
- 总内存容量

**算术强度**（Arithmetic Intensity）是核心指标：

$$\text{Arithmetic Intensity} = \frac{\text{Total FLOPs}}{\text{Total Bytes Transferred}}$$

当算术强度高时，计算时间主导性能；当算术强度低时，内存带宽成为瓶颈。

### 2.2 GPU 内存层次结构

现代 GPU 存在明显的内存层次结构，不同层级的带宽和容量差异巨大（以 NVIDIA H100 为例）：

| Memory Type | Capacity | Bandwidth | Latency |
|-------------|----------|-----------|---------|
| Registers | ~256KB/SM | ~20 TB/s | 1 cycle |
| Shared Memory (SRAM) | 228KB/SM | ~19 TB/s | ~20 cycles |
| L2 Cache | 50MB | ~12 TB/s | ~200 cycles |
| HBM (Global Memory) | 80GB | 3.35 TB/s | ~400 cycles |

关键洞察：
- SRAM 访问比 HBM 快约 10 倍
- 算法设计应尽量减少 HBM 访问，最大化数据复用
- FlashAttention 等技术正是利用了这一特性

### 2.3 Compute-bound vs Memory-bound

计算时间与访存时间：

$$T_{\text{compute}} = \frac{\text{FLOPs}}{\text{Peak FLOPs/s}}, \quad T_{\text{memory}} = \frac{\text{Bytes}}{\text{Bandwidth}}$$

**临界算术强度**（Critical Arithmetic Intensity）：

$$I_{\text{critical}} = \frac{\text{Peak FLOPs/s}}{\text{Memory Bandwidth}}$$

- 当 $I < I_{\text{critical}}$ 时，程序是 **Memory-bound**
- 当 $I > I_{\text{critical}}$ 时，程序是 **Compute-bound**

```
Performance
    ^
    |                    ______ Compute-bound (峰值计算)
    |                   /
    |                  /
    |                 /  Memory-bound
    |                /   (带宽受限)
    |               /
    |              /
    +-------------/-----------------------> Arithmetic Intensity
                 I_critical
```

### 2.4 主流 AI 加速器规格

| Hardware | Peak FLOPs/s (BF16) | HBM Bandwidth | $I_{\text{critical}}$ |
|----------|---------------------|---------------|------------------------|
| NVIDIA A100 | 312 TFLOPs | 2.0 TB/s | ~156 |
| NVIDIA H100 | 990 TFLOPs | 3.35 TB/s | ~296 |
| Google TPU v5e | 197 TFLOPs | 820 GB/s | ~240 |
| AMD MI300X | 1,307 TFLOPs | 5.3 TB/s | ~247 |

### 2.5 矩阵乘法分析

矩阵乘法是 Transformer 的核心计算。对于 $C = AB$，其中 $A \in \mathbb{R}^{B \times D}$，$B \in \mathbb{R}^{D \times F}$：

$$\text{FLOPs} = 2BDF$$

$$\text{Bytes} = 2(BD + DF + BF) \quad \text{(fp16/bf16)}$$

算术强度：

$$I = \frac{BDF}{BD + DF + BF}$$

当 $B \ll D, F$ 时（小 batch 场景），$I \approx B$。这意味着：

- 小 batch 推理通常是 memory-bound
- 增大 batch size 可以提高计算效率
- 对于 H100，batch size 需要超过 ~300 才能充分利用计算能力

> **示例**：LLaMA-7B 在 H100 上的推理
> - 模型维度 $d = 4096$，FFN 维度 $d_{ff} = 11008$
> - 单 token 推理：$B = 1$，算术强度 $I \approx 1 \ll 296$，严重 memory-bound
> - Batch size = 512：$I \approx 512 > 296$，可以达到 compute-bound

### 2.6 对 Transformer 设计的启示

**注意力机制**：标准注意力显式构建 $n \times n$ 的注意力矩阵，HBM 访存量按 $O(n^2)$ 增长，是典型的 memory-bound kernel。FlashAttention 通过分块到 SRAM，避免显式构建完整注意力矩阵。

**KV Cache**：自回归生成时，KV cache 的加载是主要瓶颈。MQA 和 GQA 通过减少 KV 头数来降低内存访问量。

**混合精度**：使用 INT8 权重 + BF16 激活时，权重加载字节数减半，算术强度翻倍。

## 3. 分布式训练基础

当模型规模超过单个加速器的内存容量时，需要将参数和计算分布到多个设备上。

### 3.1 通信原语

分布式计算有四个核心通信原语（$N$ 个设备，每设备本地数据量 $V$）：

**AllGather**：收集分片，每设备获得完整副本
```
之前: D0:[A]    D1:[B]    D2:[C]    D3:[D]      每设备 V
之后: D0:[ABCD] D1:[ABCD] D2:[ABCD] D3:[ABCD]   每设备 4V
```

**ReduceScatter**：归约后分片
```
之前: D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  (每设备持有完整但未归约的梯度)
之后: D0:[ΣA] D1:[ΣB] D2:[ΣC] D3:[ΣD]        (每设备持有1/4的归约结果)
```

**AllReduce**：归约后广播（= ReduceScatter + AllGather）
```
之前: D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  每设备 V
之后: D0:[ΣA,ΣB,ΣC,ΣD] D1:[ΣA,ΣB,ΣC,ΣD] ...  每设备 V（完整归约结果）
```

**AllToAll**：重新分片（不归约，仅重分布）
```
之前: D0:[A0,A1,A2,A3] D1:[B0,B1,B2,B3] ...  (按"行"切分)
之后: D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  (按"列"切分)
```

| 原语 | 数据变化 | 是否归约 | 通信成本 | 典型用途 |
|------|----------|----------|----------|----------|
| AllGather | 分片→完整 | 否 | $V/W$ | TP 激活收集 |
| ReduceScatter | 完整→分片 | 是 | $V/W$ | ZeRO 梯度分片 |
| AllReduce | 完整→完整 | 是 | $2V/W$ | DDP 梯度同步 |
| AllToAll | 分片→分片 | 否 | $V/W$ | MoE 路由 |

### 3.2 并行策略

**Data Parallelism (DP)**：将 batch 维度切分，每设备持有完整模型副本
- 前向传播：各设备独立计算
- 反向传播：AllReduce 同步梯度
- 缺点：内存冗余

**Fully Sharded Data Parallelism (FSDP/ZeRO)**：参数、梯度、优化器状态都分片
$$\text{Memory/device} = \frac{\text{Model Size}}{N_{\text{devices}}} + \text{Activations}$$

**Tensor Parallelism (TP)**：将矩阵维度切分，每层内部并行
- Column Parallel：$W[D, F_X]$
- Row Parallel：$W[D_X, F]$
- 每层需要 2 次 AllReduce

**Pipeline Parallelism (PP)**：将模型层切分到不同设备
- 优点：通信量少
- 缺点：存在 pipeline bubble

**Expert Parallelism (EP)**：MoE 中不同专家分布在不同设备
- 需要 AllToAll 进行 token 路由和结果收集

## 4. Transformer 计算分析

### 4.1 符号定义

| 符号 | 含义 |
|------|------|
| $B$ | Batch size |
| $T$ | 序列长度 |
| $D$ | 模型维度（Hidden dimension） |
| $F$ | FFN 中间维度（通常 $F = 4D$ 或 $\frac{8}{3}D$） |
| $L$ | Transformer 层数 |
| $N$ | Query head 数量 |
| $K$ | KV head 数量（MHA: $K=N$，GQA: $K<N$，MQA: $K=1$） |
| $H$ | 每个 head 的维度（通常 $H = D/N$） |
| $V$ | 词表大小 |
| $P$ | 模型总参数量 |

### 4.2 基础运算

| 运算 | 表达式 | FLOPs |
|------|--------|-------|
| 向量点积 | $\mathbf{x} \cdot \mathbf{y}$, $\mathbf{x}, \mathbf{y} \in \mathbb{R}^k$ | $2k$ |
| 矩阵-向量乘 | $A\mathbf{x}$, $A \in \mathbb{R}^{m \times k}$ | $2mk$ |
| 矩阵-矩阵乘 | $AB$, $A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{k \times n}$ | $2mkn$ |

**前向与反向传播**：对于线性层 $Y = XW$（$X \in \mathbb{R}^{m \times k}$，$W \in \mathbb{R}^{k \times n}$）：

- 前向：$Y = XW$，FLOPs $= 2mkn$
- 反向：$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top$（$2mkn$）+ $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$（$2mkn$）
- **总计：$6mkn$ FLOPs**

这导出了训练 FLOPs 的核心公式：

$$\boxed{\text{Training FLOPs} \approx 6 \times P \times T_{\text{tokens}}}$$

其中 $P$ 是参数量，$T_{\text{tokens}}$ 是训练 token 总数。

### 4.3 MLP 层

MLP 层（也称 FFN）有两种常见形式：

**Standard FFN**：
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

**SwiGLU FFN**（现代模型普遍采用）：
$$\text{SwiGLU}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$$

其中 $\odot$ 是逐元素乘法，起**门控**作用。

| 类型 | 参数量 | 前向 FLOPs | 训练 FLOPs |
|------|--------|------------|------------|
| Standard FFN | $2DF$ | $4BTDF$ | $12BTDF$ |
| SwiGLU | $3DF$ | $6BTDF$ | $18BTDF$ |

> **参数量一致性**：为保持总参数量一致，不同结构调整 $F$ 的取值：
> - Standard FFN：$F = 4D$ → 参数量 $= 8D^2$
> - SwiGLU：$F = \frac{8}{3}D$ → 参数量 $= 8D^2$

### 4.4 Attention 层

Multi-Head Attention 包含四个投影和注意力计算：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V, \quad O = \text{Attn}(Q, K, V) W_O$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{H}}\right)V$$

| 组件 | 参数量 | 训练 FLOPs |
|------|--------|------------|
| Q projection | $D^2$ | $6BTD^2$ |
| K projection | $DKH$ | $6BTDKH$ |
| V projection | $DKH$ | $6BTDKH$ |
| O projection | $D^2$ | $6BTD^2$ |
| $QK^\top$ | — | $6BT^2NH$ |
| $\text{softmax} \cdot V$ | — | $6BT^2NH$ |
| **Total** | $2D^2 + 2DKH$ | $12BTD^2 + 12BTDKH + 12BT^2NH$ |

**MHA / GQA / MQA 对比**：

- **MHA**（Multi-Head Attention）：$K = N$，每个 head 独立 KV
- **GQA**（Grouped-Query Attention）：$1 < K < N$，多个 Q head 共享 KV
- **MQA**（Multi-Query Attention）：$K = 1$，所有 head 共享一个 KV

### 4.5 Attention vs MLP 计算量

简化假设（$F = 4D$，$K \ll N$，$NH = D$）下：

$$\frac{\text{Attention FLOPs}}{\text{MLP FLOPs}} \approx \frac{T}{8D}$$

当 $T < 8D$ 时，**MLP 计算量主导**。对于 $D = 8192$ 的模型，序列长度需超过 $65536$ 才能使注意力成为主要计算瓶颈。这解释了为什么只有长上下文场景才需要特别关注注意力效率。

### 4.6 完整模型参数量

$$\boxed{P_{\text{total}} = 2VD + L \cdot (2D^2 + 2DKH + 3DF + 4D)}$$

> **示例：LLaMA-7B 参数计算**
> $D = 4096$, $F = 11008$, $L = 32$, $V = 32000$, $N = K = 32$（MHA）, $H = 128$：
>
> - $P_{\text{embed}} = 32000 \times 4096 = 131\text{M}$
> - $P_{\text{attn/layer}} = 2 \times 4096^2 + 2 \times 4096 \times 32 \times 128 = 67\text{M}$
> - $P_{\text{mlp/layer}} = 3 \times 4096 \times 11008 = 135\text{M}$
> - $P_{\text{total}} \approx 2 \times 131\text{M} + 32 \times (67\text{M} + 135\text{M}) \approx \mathbf{6.7B}$

### 4.7 训练内存占用

| 组件 | 内存 | 说明 |
|------|------|------|
| 参数（bf16） | $2P$ | 模型权重 |
| 梯度（bf16） | $2P$ | 反向传播梯度 |
| 优化器状态（Adam, fp32） | $8P$ | momentum + variance |
| 激活值 | $O(BTD \cdot L)$ | 中间结果 |
| **总计（无优化）** | $\approx 12P + \text{activations}$ | |

**Activation Checkpointing**：通过只保存每层输入、重新计算中间激活来节省内存：
- 内存：从 $O(L \cdot BTD)$ 降至 $O(BTD)$
- 代价：额外约 33% 重计算

### 4.8 推理分析

**KV Cache**：自回归生成时需缓存历史的 K 和 V：

$$\boxed{\text{KV Cache Size} = 2 \times B \times S \times L \times K \times H \times \text{bytes}}$$

> **示例**：70B 模型 KV Cache
> $D = 8192$, $L = 80$, $K = 8$（GQA）, $H = 128$, $S = 8192$, bf16：
> $$\text{KV Cache} = 2 \times 1 \times 8192 \times 80 \times 8 \times 128 \times 2 = \mathbf{2.1\text{ GB/request}}$$

**Prefill vs Decode**：

|  | Prefill | Decode |
|--|---------|--------|
| 输入 | 整个 prompt（$T$ tokens） | 单个 token |
| 计算模式 | 并行处理所有 token | 逐 token 生成 |
| 瓶颈 | Compute-bound | Memory-bound |
| 主要开销 | 矩阵乘法计算 | 权重加载 + KV Cache 读写 |

## 5. Scaling Law

Scaling Law 揭示了 LLM 性能与计算量、数据量、模型规模之间的幂律关系，是指导大规模训练资源分配的核心理论。

### 5.1 基本形式

$$L = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

其中：
- $N$：模型参数量
- $D$：训练数据量（tokens）
- $L_\infty$：不可约损失（数据本身的熵）

### 5.2 Kaplan Scaling Law (2020)

OpenAI 的 Kaplan 等人首次系统研究了 LLM 的 Scaling Law：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

**核心结论**：
1. **模型规模主导**：在固定计算预算下，更大的模型（训练更少步数）比小模型（训练更多步数）更优
2. **最优分配**：计算量增加 10 倍时，模型参数应增加约 5.5 倍，数据量增加约 1.8 倍
3. **架构不敏感**：Scaling Law 对 Transformer 的具体超参数不敏感

### 5.3 Chinchilla Scaling Law (2022)

DeepMind 的 Hoffmann 等人挑战了 Kaplan 的结论：

**核心发现**：之前的模型普遍欠训练（undertrained）：
- GPT-3 (175B)：训练了 300B tokens，但最优应该是约 3.5T tokens
- Gopher (280B)：同样严重不足

**新的最优分配**：

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

更简洁的表述：

$$\boxed{D_{opt} \approx 20 \times N}$$

即**最优训练数据量约为模型参数的 20 倍**。

**物理直觉**：模型参数需要足够的"训练信号"才能收敛到最优值。每个参数平均需要看到约 20 个 tokens 的信息才能学到有意义的表示。

### 5.4 Chinchilla 的验证

DeepMind 训练了 70B 参数的 Chinchilla 模型，使用 1.4T tokens：

| 模型 | 参数量 | 训练 Tokens | MMLU |
|------|--------|-------------|------|
| Gopher | 280B | 300B | 60.0% |
| Chinchilla | 70B | 1.4T | 67.6% |

- 计算量与 Gopher 相同
- 性能全面超越 Gopher
- 推理成本降低 4 倍

### 5.5 超越 Chinchilla：推理最优

Chinchilla 解决的是"训练最优"问题。但在工业部署中，还需要考虑推理成本：

$$\text{Total Cost} = C_{train} + n_{infer} \times C_{infer}(N)$$

当 $n_{infer}$ 很大时，最优策略发生转变——**Over-training 更小的模型**：

| 策略 | 模型 | 数据 | 训练成本 | 推理成本 |
|------|------|------|----------|----------|
| Chinchilla | 70B | 1.4T | 基准 | 高 |
| Over-training | 8B | 15T | +22% | -88% |

**LLaMA 系列的策略**：
- LLaMA-7B：1T tokens（143× 参数）
- LLaMA 2-7B：2T tokens（286× 参数）
- LLaMA 3-8B：15T tokens（1875× 参数）

### 5.6 主流模型的参数-数据配比

| 模型 | 参数量 | 训练 Tokens | Tokens/参数 |
|------|--------|-------------|-------------|
| GPT-3 | 175B | 300B | 1.7× |
| Chinchilla | 70B | 1.4T | 20× |
| LLaMA | 65B | 1.4T | 21.5× |
| LLaMA 2 | 70B | 2T | 28.6× |
| LLaMA 3 | 70B | 15T | 214× |
| Qwen 2 | 72B | 7T+ | 97× |

### 5.7 Test-time Compute Scaling

OpenAI o1 等推理模型展示了新的 Scaling 维度：

$$\text{Performance} = f(\text{Pretraining Compute}, \text{Inference Compute})$$

通过更多推理时计算（更长的思考链）提升性能，开辟了"用推理换性能"的新范式。

### 5.8 Scaling Law 的局限

- **外推风险**：从小规模实验外推大规模可能失准
- **数据质量**：Scaling Law 假设数据质量恒定
- **数据墙**：高质量互联网文本有限（估计约 10-15T tokens）

**应对策略**：
- 合成数据：用模型生成训练数据
- 多模态数据：图像、视频、音频
- 代码数据：GitHub 等代码仓库

## 本章小结

本章建立了理解 Transformer 的基础框架：

1. **硬件性能模型**：Roofline 分析揭示了 memory-bound vs compute-bound 的本质区别
2. **内存层次结构**：SRAM 比 HBM 快 10 倍，这是 FlashAttention 等优化的基础
3. **Transformer 计算分析**：
   - 训练 FLOPs $\approx 6 \times P \times T_{\text{tokens}}$
   - 当 $T < 8D$ 时，MLP 计算量主导
   - KV Cache 是长上下文推理的主要瓶颈
4. **Scaling Law**：
   - Chinchilla：$D_{opt} \approx 20 \times N$
   - 工业实践：Over-training 更小的模型以降低推理成本
   - 新维度：Test-time Compute Scaling

下一篇将介绍 Transformer 的核心组件：分词器、位置编码与门控机制。
