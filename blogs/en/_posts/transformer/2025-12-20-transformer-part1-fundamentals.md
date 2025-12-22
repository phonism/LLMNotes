---
layout: post
title: "Transformer Notes (I): Fundamentals"
date: 2025-12-20 10:00:00
author: Qi Lu
tags: [Transformer, LLM, Scaling Law, GPU]
lang: en
translation: /transformer-part1-fundamentals/
---

From RNN to LSTM to Transformer, the paradigm of sequence modeling has undergone a fundamental shift. The Transformer architecture proposed by Vaswani et al. in 2017 completely abandoned recurrent structures, using only attention mechanisms to model sequence dependencies, ushering in the era of Large Language Models (LLMs).

This is the first post in the Transformer series, covering fundamental theory: hardware performance models, Transformer computational analysis, and the Scaling Laws that guide large-scale training.

## 1. Introduction: Why Transformer

### 1.1 The Dilemma of RNNs

Traditional sequence modeling methods—RNN, LSTM, GRU—have inherent limitations:

- **Sequential computation**: Must process sequences step by step, difficult to parallelize
- **Long-range dependencies**: Despite LSTM's gating mechanism, information still decays in long sequences
- **Computational efficiency**: Training and inference speed limited by sequence length

### 1.2 The Transformer Revolution

Transformer directly models dependencies between arbitrary positions through self-attention:

- **Full parallelization**: Dramatically improves training efficiency
- **Direct long-range dependencies**: Attention can directly connect any two positions
- **Flexible and scalable**: Easy to stack and transfer

Transformer-based LLMs have achieved remarkable progress in subsequent years. From the GPT series to open-source models like LLaMA and DeepSeek, parameter scales have jumped from billions to trillions. These models not only excel at NLP tasks but also demonstrate powerful emergent capabilities: in-context learning, chain-of-thought reasoning, code generation, and more.

### 1.3 New Challenges

As model scale expands, new challenges continue to emerge:

| Challenge | Description |
|-----------|-------------|
| Computational efficiency | Standard attention's $O(n^2)$ complexity limits long-context modeling |
| Training cost | Hundred-billion parameter models require thousands of GPUs training for months |
| Deployment challenges | KV Cache memory footprint becomes an inference bottleneck |
| Capability boundaries | Complex reasoning, multimodal understanding remain challenging |

Understanding the root causes of these challenges requires starting from hardware performance models.

## 2. Hardware and Performance Fundamentals

### 2.1 Roofline Model

The Roofline model is a classic framework for analyzing program performance, defining performance bounds through three basic constraints:

- Compute speed (FLOPs/s)
- Data transfer bandwidth (bytes/s)
- Total memory capacity

**Arithmetic Intensity** is the core metric:

$$\text{Arithmetic Intensity} = \frac{\text{Total FLOPs}}{\text{Total Bytes Transferred}}$$

When arithmetic intensity is high, compute time dominates performance; when arithmetic intensity is low, memory bandwidth becomes the bottleneck.

### 2.2 GPU Memory Hierarchy

Modern GPUs have a clear memory hierarchy, with significant differences in bandwidth and capacity between levels (using NVIDIA H100 as example):

| Memory Type | Capacity | Bandwidth | Latency |
|-------------|----------|-----------|---------|
| Registers | ~256KB/SM | ~20 TB/s | 1 cycle |
| Shared Memory (SRAM) | 228KB/SM | ~19 TB/s | ~20 cycles |
| L2 Cache | 50MB | ~12 TB/s | ~200 cycles |
| HBM (Global Memory) | 80GB | 3.35 TB/s | ~400 cycles |

Key insights:
- SRAM access is about 10× faster than HBM
- Algorithm design should minimize HBM access and maximize data reuse
- Techniques like FlashAttention exploit this property

### 2.3 Compute-bound vs Memory-bound

Compute time vs memory access time:

$$T_{\text{compute}} = \frac{\text{FLOPs}}{\text{Peak FLOPs/s}}, \quad T_{\text{memory}} = \frac{\text{Bytes}}{\text{Bandwidth}}$$

**Critical Arithmetic Intensity**:

$$I_{\text{critical}} = \frac{\text{Peak FLOPs/s}}{\text{Memory Bandwidth}}$$

- When $I < I_{\text{critical}}$, the program is **Memory-bound**
- When $I > I_{\text{critical}}$, the program is **Compute-bound**

<!-- tikz-source: transformer-roofline-en
\begin{tikzpicture}[scale=0.9]
    % Axes
    \draw[->] (0,0) -- (8,0) node[right] {Arithmetic Intensity (FLOPs/Byte)};
    \draw[->] (0,0) -- (0,5) node[above] {Performance (FLOPs/s)};
    % Roofline
    \draw[thick, blue] (0,0) -- (3,4) node[midway, above, sloped] {Memory-bound};
    \draw[thick, red] (3,4) -- (8,4) node[midway, above] {Compute-bound};
    % Critical point
    \fill (3,4) circle (2pt) node[above left] {$I_{\text{critical}}$};
    % Peak lines (dashed)
    \draw[dashed, gray] (0,4) -- (3,4);
    \draw[dashed, gray] (3,0) -- (3,4);
    % Labels
    \node[left] at (0,4) {Peak FLOPs/s};
\end{tikzpicture}
-->
![Roofline Model]({{ site.baseurl }}/assets/figures/transformer-roofline-en.svg)

### 2.4 Mainstream AI Accelerator Specs

| Hardware | Peak FLOPs/s (BF16) | HBM Bandwidth | $I_{\text{critical}}$ |
|----------|---------------------|---------------|------------------------|
| NVIDIA A100 | 312 TFLOPs | 2.0 TB/s | ~156 |
| NVIDIA H100 | 990 TFLOPs | 3.35 TB/s | ~296 |
| Google TPU v5e | 197 TFLOPs | 820 GB/s | ~240 |
| AMD MI300X | 1,307 TFLOPs | 5.3 TB/s | ~247 |

### 2.5 Matrix Multiplication Analysis

Matrix multiplication is the core computation of Transformers. For $C = AB$, where $A \in \mathbb{R}^{B \times D}$, $B \in \mathbb{R}^{D \times F}$:

$$\text{FLOPs} = 2BDF$$

$$\text{Bytes} = 2(BD + DF + BF) \quad \text{(fp16/bf16)}$$

Arithmetic intensity:

$$I = \frac{BDF}{BD + DF + BF}$$

When $B \ll D, F$ (small batch scenario), $I \approx B$. This means:

- Small batch inference is typically memory-bound
- Increasing batch size improves computational efficiency
- For H100, batch size needs to exceed ~300 to fully utilize compute capacity

> **Example**: LLaMA-7B inference on H100
> - Model dimension $d = 4096$, FFN dimension $d_{ff} = 11008$
> - Single token inference: $B = 1$, arithmetic intensity $I \approx 1 \ll 296$, severely memory-bound
> - Batch size = 512: $I \approx 512 > 296$, can achieve compute-bound

### 2.6 Implications for Transformer Design

**Attention mechanism**: Standard attention explicitly constructs an $n \times n$ attention matrix, with HBM access growing as $O(n^2)$, a typical memory-bound kernel. FlashAttention avoids explicitly constructing the full attention matrix by tiling to SRAM.

**KV Cache**: During autoregressive generation, loading KV cache is the main bottleneck. MQA and GQA reduce memory access by reducing the number of KV heads.

**Mixed precision**: When using INT8 weights + BF16 activations, weight loading bytes are halved, doubling arithmetic intensity.

## 3. Distributed Training Fundamentals

When model scale exceeds a single accelerator's memory capacity, parameters and computation must be distributed across multiple devices.

### 3.1 Communication Primitives

Distributed computing has four core communication primitives ($N$ devices, local data volume $V$ per device):

**AllGather**: Collect shards, each device gets complete copy
```
Before: D0:[A]    D1:[B]    D2:[C]    D3:[D]      V per device
After:  D0:[ABCD] D1:[ABCD] D2:[ABCD] D3:[ABCD]   4V per device
```

**ReduceScatter**: Reduce then shard
```
Before: D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  (each holds full unreduced gradients)
After:  D0:[ΣA] D1:[ΣB] D2:[ΣC] D3:[ΣD]        (each holds 1/4 of reduced result)
```

**AllReduce**: Reduce then broadcast (= ReduceScatter + AllGather)
```
Before: D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  V per device
After:  D0:[ΣA,ΣB,ΣC,ΣD] D1:[ΣA,ΣB,ΣC,ΣD] ...  V per device (full reduced result)
```

**AllToAll**: Reshard (no reduction, just redistribution)
```
Before: D0:[A0,A1,A2,A3] D1:[B0,B1,B2,B3] ...  (row-partitioned)
After:  D0:[A0,B0,C0,D0] D1:[A1,B1,C1,D1] ...  (column-partitioned)
```

| Primitive | Data Change | Reduces? | Comm Cost | Typical Use |
|-----------|-------------|----------|-----------|-------------|
| AllGather | Shard→Full | No | $V/W$ | TP activation gather |
| ReduceScatter | Full→Shard | Yes | $V/W$ | ZeRO gradient sharding |
| AllReduce | Full→Full | Yes | $2V/W$ | DDP gradient sync |
| AllToAll | Shard→Shard | No | $V/W$ | MoE routing |

### 3.2 Parallelism Strategies

**Data Parallelism (DP)**: Split batch dimension, each device holds complete model replica
- Forward pass: Each device computes independently
- Backward pass: AllReduce synchronizes gradients
- Drawback: Memory redundancy

**Fully Sharded Data Parallelism (FSDP/ZeRO)**: Parameters, gradients, optimizer states all sharded
$$\text{Memory/device} = \frac{\text{Model Size}}{N_{\text{devices}}} + \text{Activations}$$

**Tensor Parallelism (TP)**: Split matrix dimensions, parallelize within each layer
- Column Parallel: $W[D, F_X]$
- Row Parallel: $W[D_X, F]$
- 2 AllReduce ops per layer

**Pipeline Parallelism (PP)**: Split model layers across different devices
- Advantage: Low communication volume
- Drawback: Pipeline bubbles

**Expert Parallelism (EP)**: In MoE, different experts distributed across devices
- Requires AllToAll for token routing and result collection

## 4. Transformer Computational Analysis

### 4.1 Symbol Definitions

| Symbol | Meaning |
|--------|---------|
| $B$ | Batch size |
| $T$ | Sequence length |
| $D$ | Model dimension (Hidden dimension) |
| $F$ | FFN intermediate dimension (typically $F = 4D$ or $\frac{8}{3}D$) |
| $L$ | Number of Transformer layers |
| $N$ | Number of Query heads |
| $K$ | Number of KV heads (MHA: $K=N$, GQA: $K<N$, MQA: $K=1$) |
| $H$ | Dimension per head (typically $H = D/N$) |
| $V$ | Vocabulary size |
| $P$ | Total model parameters |

### 4.2 Basic Operations

| Operation | Expression | FLOPs |
|-----------|------------|-------|
| Vector dot product | $\mathbf{x} \cdot \mathbf{y}$, $\mathbf{x}, \mathbf{y} \in \mathbb{R}^k$ | $2k$ |
| Matrix-vector multiply | $A\mathbf{x}$, $A \in \mathbb{R}^{m \times k}$ | $2mk$ |
| Matrix-matrix multiply | $AB$, $A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{k \times n}$ | $2mkn$ |

**Forward and backward propagation**: For linear layer $Y = XW$ ($X \in \mathbb{R}^{m \times k}$, $W \in \mathbb{R}^{k \times n}$):

- Forward: $Y = XW$, FLOPs $= 2mkn$
- Backward: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top$ ($2mkn$) + $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$ ($2mkn$)
- **Total: $6mkn$ FLOPs**

This derives the core formula for training FLOPs:

$$\boxed{\text{Training FLOPs} \approx 6 \times P \times T_{\text{tokens}}}$$

where $P$ is parameter count and $T_{\text{tokens}}$ is total training tokens.

### 4.3 MLP Layer

MLP layers (also called FFN) have two common forms:

**Standard FFN**:
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

**SwiGLU FFN** (commonly used in modern models):
$$\text{SwiGLU}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$$

where $\odot$ is element-wise multiplication, acting as a **gate**.

| Type | Parameters | Forward FLOPs | Training FLOPs |
|------|------------|---------------|----------------|
| Standard FFN | $2DF$ | $4BTDF$ | $12BTDF$ |
| SwiGLU | $3DF$ | $6BTDF$ | $18BTDF$ |

> **Parameter consistency**: To maintain consistent total parameters, different structures adjust $F$:
> - Standard FFN: $F = 4D$ → Parameters $= 8D^2$
> - SwiGLU: $F = \frac{8}{3}D$ → Parameters $= 8D^2$

### 4.4 Attention Layer

Multi-Head Attention includes four projections and attention computation:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V, \quad O = \text{Attn}(Q, K, V) W_O$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{H}}\right)V$$

| Component | Parameters | Training FLOPs |
|-----------|------------|----------------|
| Q projection | $D^2$ | $6BTD^2$ |
| K projection | $DKH$ | $6BTDKH$ |
| V projection | $DKH$ | $6BTDKH$ |
| O projection | $D^2$ | $6BTD^2$ |
| $QK^\top$ | — | $6BT^2NH$ |
| $\text{softmax} \cdot V$ | — | $6BT^2NH$ |
| **Total** | $2D^2 + 2DKH$ | $12BTD^2 + 12BTDKH + 12BT^2NH$ |

**MHA / GQA / MQA Comparison**:

- **MHA** (Multi-Head Attention): $K = N$, independent KV per head
- **GQA** (Grouped-Query Attention): $1 < K < N$, multiple Q heads share KV
- **MQA** (Multi-Query Attention): $K = 1$, all heads share one KV

### 4.5 Attention vs MLP Compute Ratio

Under simplified assumptions ($F = 4D$, $K \ll N$, $NH = D$):

$$\frac{\text{Attention FLOPs}}{\text{MLP FLOPs}} \approx \frac{T}{8D}$$

When $T < 8D$, **MLP compute dominates**. For a model with $D = 8192$, sequence length needs to exceed $65536$ for attention to become the main computational bottleneck. This explains why only long-context scenarios need special attention efficiency considerations.

### 4.6 Complete Model Parameter Count

$$\boxed{P_{\text{total}} = 2VD + L \cdot (2D^2 + 2DKH + 3DF + 4D)}$$

> **Example: LLaMA-7B Parameter Calculation**
> $D = 4096$, $F = 11008$, $L = 32$, $V = 32000$, $N = K = 32$ (MHA), $H = 128$:
>
> - $P_{\text{embed}} = 32000 \times 4096 = 131\text{M}$
> - $P_{\text{attn/layer}} = 2 \times 4096^2 + 2 \times 4096 \times 32 \times 128 = 67\text{M}$
> - $P_{\text{mlp/layer}} = 3 \times 4096 \times 11008 = 135\text{M}$
> - $P_{\text{total}} \approx 2 \times 131\text{M} + 32 \times (67\text{M} + 135\text{M}) \approx \mathbf{6.7B}$

### 4.7 Training Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Parameters (bf16) | $2P$ | Model weights |
| Gradients (bf16) | $2P$ | Backprop gradients |
| Optimizer states (Adam, fp32) | $8P$ | momentum + variance |
| Activations | $O(BTD \cdot L)$ | Intermediate results |
| **Total (no optimization)** | $\approx 12P + \text{activations}$ | |

**Activation Checkpointing**: Save memory by only keeping each layer's input and recomputing intermediate activations:
- Memory: From $O(L \cdot BTD)$ to $O(BTD)$
- Cost: ~33% extra recomputation

### 4.8 Inference Analysis

**KV Cache**: Autoregressive generation requires caching historical K and V:

$$\boxed{\text{KV Cache Size} = 2 \times B \times S \times L \times K \times H \times \text{bytes}}$$

> **Example**: 70B model KV Cache
> $D = 8192$, $L = 80$, $K = 8$ (GQA), $H = 128$, $S = 8192$, bf16:
> $$\text{KV Cache} = 2 \times 1 \times 8192 \times 80 \times 8 \times 128 \times 2 = \mathbf{2.1\text{ GB/request}}$$

**Prefill vs Decode**:

|  | Prefill | Decode |
|--|---------|--------|
| Input | Entire prompt ($T$ tokens) | Single token |
| Compute mode | Process all tokens in parallel | Generate token by token |
| Bottleneck | Compute-bound | Memory-bound |
| Main overhead | Matrix multiplication | Weight loading + KV Cache read/write |

## 5. Scaling Law

Scaling Laws reveal power-law relationships between LLM performance and compute, data, and model scale—the core theory guiding large-scale training resource allocation.

### 5.1 Basic Form

$$L = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

where:
- $N$: Model parameters
- $D$: Training data (tokens)
- $L_\infty$: Irreducible loss (entropy of the data itself)

### 5.2 Kaplan Scaling Law (2020)

OpenAI's Kaplan et al. first systematically studied LLM Scaling Laws:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

**Core conclusions**:
1. **Model scale dominates**: Under fixed compute budget, larger models (trained fewer steps) outperform smaller models (trained more steps)
2. **Optimal allocation**: When compute increases 10×, model parameters should increase ~5.5× and data ~1.8×
3. **Architecture insensitivity**: Scaling Laws are insensitive to Transformer hyperparameters

### 5.3 Chinchilla Scaling Law (2022)

DeepMind's Hoffmann et al. challenged Kaplan's conclusions:

**Core finding**: Previous models were systematically undertrained:
- GPT-3 (175B): Trained on 300B tokens, but optimal would be ~3.5T tokens
- Gopher (280B): Similarly undertrained

**New optimal allocation**:

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

More concisely:

$$\boxed{D_{opt} \approx 20 \times N}$$

That is, **optimal training data should be about 20× model parameters**.

**Physical intuition**: Model parameters need sufficient "training signal" to converge to optimal values. Each parameter needs to see about 20 tokens on average to learn meaningful representations.

### 5.4 Chinchilla Validation

DeepMind trained the 70B parameter Chinchilla model using 1.4T tokens:

| Model | Parameters | Training Tokens | MMLU |
|-------|------------|-----------------|------|
| Gopher | 280B | 300B | 60.0% |
| Chinchilla | 70B | 1.4T | 67.6% |

- Same compute as Gopher
- Comprehensively outperforms Gopher
- 4× lower inference cost

### 5.5 Beyond Chinchilla: Inference Optimal

Chinchilla solved the "training optimal" problem. But in industrial deployment, inference cost must also be considered:

$$\text{Total Cost} = C_{train} + n_{infer} \times C_{infer}(N)$$

When $n_{infer}$ is large, the optimal strategy shifts—**Over-training smaller models**:

| Strategy | Model | Data | Training Cost | Inference Cost |
|----------|-------|------|---------------|----------------|
| Chinchilla | 70B | 1.4T | Baseline | High |
| Over-training | 8B | 15T | +22% | -88% |

**LLaMA series strategy**:
- LLaMA-7B: 1T tokens (143× parameters)
- LLaMA 2-7B: 2T tokens (286× parameters)
- LLaMA 3-8B: 15T tokens (1875× parameters)

### 5.6 Parameter-Data Ratios of Mainstream Models

| Model | Parameters | Training Tokens | Tokens/Param |
|-------|------------|-----------------|--------------|
| GPT-3 | 175B | 300B | 1.7× |
| Chinchilla | 70B | 1.4T | 20× |
| LLaMA | 65B | 1.4T | 21.5× |
| LLaMA 2 | 70B | 2T | 28.6× |
| LLaMA 3 | 70B | 15T | 214× |
| Qwen 2 | 72B | 7T+ | 97× |

### 5.7 Test-time Compute Scaling

Reasoning models like OpenAI o1 demonstrate a new Scaling dimension:

$$\text{Performance} = f(\text{Pretraining Compute}, \text{Inference Compute})$$

Improving performance through more inference-time compute (longer chains of thought) opens a new paradigm of "trading inference for performance."

### 5.8 Limitations of Scaling Laws

- **Extrapolation risk**: Extrapolating from small-scale experiments to large scale may be inaccurate
- **Data quality**: Scaling Laws assume constant data quality
- **Data wall**: High-quality internet text is limited (estimated ~10-15T tokens)

**Mitigation strategies**:
- Synthetic data: Generate training data with models
- Multimodal data: Images, video, audio
- Code data: GitHub and code repositories

## Summary

This chapter established the foundational framework for understanding Transformers:

1. **Hardware performance model**: Roofline analysis reveals the essential difference between memory-bound vs compute-bound
2. **Memory hierarchy**: SRAM is 10× faster than HBM, the foundation for optimizations like FlashAttention
3. **Transformer computational analysis**:
   - Training FLOPs $\approx 6 \times P \times T_{\text{tokens}}$
   - When $T < 8D$, MLP compute dominates
   - KV Cache is the main bottleneck for long-context inference
4. **Scaling Law**:
   - Chinchilla: $D_{opt} \approx 20 \times N$
   - Industrial practice: Over-training smaller models to reduce inference cost
   - New dimension: Test-time Compute Scaling

The next post will cover Transformer core components: tokenizers, positional encoding, and gating mechanisms.
