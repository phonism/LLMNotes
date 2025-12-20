---
layout: post
title: "Transformer Notes (III): Attention Mechanisms"
date: 2025-12-20 10:20:00
author: Phonism
tags: [Transformer, FlashAttention, MLA, Sparse Attention, Linear Attention]
lang: en
translation: /transformer-part3-attention/
---

The attention mechanism is the core of Transformers, but the standard $O(N^2)$ complexity becomes a bottleneck for long-context modeling. This post systematically introduces four optimization paths: FlashAttention accelerates computation through IO-aware optimization; MLA reduces KV Cache through low-rank compression; sparse attention only computes important token pairs; linear attention fundamentally changes the computation form.

## 1. FlashAttention: IO-Aware Efficient Attention

### 1.1 Problem: GPU Memory Hierarchy Bottleneck

Modern GPUs have computational capabilities far exceeding memory bandwidth. Taking NVIDIA A100 as an example:
- Compute capability: 312 TFLOPS (FP16 Tensor Core)
- HBM bandwidth: 2 TB/s
- SRAM bandwidth: approximately 19 TB/s

Standard attention implementations require multiple HBM reads and writes:
1. Read $Q, K$ from HBM, compute $S = QK^\top$, write back to HBM
2. Read $S$ from HBM, compute $P = \text{softmax}(S)$, write back to HBM
3. Read $P, V$ from HBM, compute $O = PV$, write back to HBM

Total HBM access: $O(N^2 + Nd)$, causing severe IO bottlenecks.

### 1.2 FlashAttention v1: Attention Matrix That Never Materializes

FlashAttention's core idea: **Never write the complete $N \times N$ attention matrix to HBM**.

**Online Softmax Algorithm**: Enables single-pass, incremental computation. For concatenation of two blocks $x^{(1)}, x^{(2)}$:

$$m^{new} = \max(m^{(1)}, m^{(2)})$$
$$\ell^{new} = e^{m^{(1)} - m^{new}} \ell^{(1)} + e^{m^{(2)} - m^{new}} \ell^{(2)}$$

**Tiling Algorithm**: Divide $Q, K, V$ into blocks of size $B_r \times d$ and $B_c \times d$, completing computation in SRAM.

**Recomputation Strategy**: During backpropagation, recompute $S, P$ from $Q, K, V$ blocks. While this increases FLOPs by approximately 30%, it dramatically reduces HBM access.

**IO Complexity**: Reduced from $O(N^2 + Nd)$ in standard attention to $O(N^2 d^2 / M)$. When $M = \Theta(Nd)$, HBM accesses are reduced by a factor of $O(N)$.

### 1.3 FlashAttention-2: Parallelism Strategy Optimization

FlashAttention-2 optimizes parallelism strategies, achieving 230 TFLOPS on A100 (approximately 73% peak utilization), about 2× faster than v1.

**Sliced-Q Scheme**: Partition $Q$ among 4 warps, with $K, V$ visible to all warps. Eliminates inter-warp communication and reduces shared memory reads/writes.

| Metric | v1 | v2 |
|------|----|----|
| A100 Peak (TFLOPS) | 124 | 230 |
| GPU Utilization | 25-40% | 50-73% |

### 1.4 FlashAttention-3: Hopper Architecture Optimization

Designed for NVIDIA H100, achieving 740 TFLOPS (75% peak utilization).

**Three Major Optimization Techniques**:
1. **Warp Specialization**: Producer warps handle TMA data transfer, Consumer warps handle WGMMA computation
2. **Ping-Pong Scheduling**: Two warpgroups alternately execute GEMM and Softmax
3. **Intra-warpgroup Overlapping**: Pipelining Softmax with GEMM

**FP8 Support**: Through Block Quantization and Incoherent Processing, FP8 throughput approaches 1.2 PFLOPS.

### 1.5 Flash Decoding: KV Parallelism During Inference

FlashAttention has issues during inference: each step only generates 1 token, severely underutilizing the GPU.

Flash Decoding parallelizes across the **KV sequence length** dimension:
1. Split KV Cache into $S$ blocks
2. Compute partial attention independently for each block
3. Merge results using Log-Sum-Exp

Performance: Long sequence decoding speedup up to **8×**, attention operations on CodeLLaMa-34B are **50× faster** than FlashAttention.

### 1.6 FlashAttention Version Evolution

| Version | GPU | Peak TFLOPS | Utilization | Speedup vs v1 |
|------|-----|-------------|--------|--------------|
| v1 | A100 | 124 | 40% | 1× |
| v2 | A100 | 230 | 73% | 1.85× |
| v3 | H100 | 740 | 75% | 6× |
| v3 (FP8) | H100 | 1200 | - | 9.7× |
| v4 | B200 | PFLOPS | - | 15× |

**Engineering Impact**: FlashAttention has become standard in modern LLMs, increasing practical context lengths from 2-4K to 128K+.

## 2. Multi-head Latent Attention (MLA)

### 2.1 KV Cache Challenge

During autoregressive generation, KV Cache becomes the primary memory bottleneck for long-context inference:

$$\text{KV Cache Size} = 2 \times B \times S \times L \times n_h \times d_h \times \text{bytes}$$

For a 70B model ($L=80$, $K=8$, $S=8192$), KV Cache reaches **2.1 GB/request**.

### 2.2 Limitations of Existing Solutions

| Method | KV Cache Size | Performance |
|--------|---------------|-------------|
| MHA | $2 n_h d_h$ | Optimal |
| GQA | $2 \frac{n_h}{g} d_h$ | Slight degradation |
| MQA | $2 d_h$ | Significant degradation |
| **MLA** | $d_c + d_h^R$ | **Close to MHA** |

GQA/MQA reduce cache by **forcing sharing** of KV heads, but this harms model performance. MLA's core insight: **KV can be recovered from a low-dimensional latent vector**.

### 2.3 MLA Core Principles

**Low-rank Compression of KV**:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

Recover K and V from latent vector:

$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$

where $d_c \ll n_h d_h$ is the compression dimension.

**Decoupled RoPE**: Split each attention head into two parts:
- **Content part** ($d_h^C$ dimensions): Recovered from compressed vector, RoPE **not** applied
- **Position part** ($d_h^R$ dimensions): Additional projection, RoPE applied

$$\mathbf{q}_t = [\mathbf{q}_t^C; \text{RoPE}(\mathbf{q}_t^R, t)]$$

**Weight Absorption**: Since there's no nonlinear activation between compression and recovery, matrices can be merged:

$$W^{QK} = (W^{UQ})^\top W^{UK}$$

### 2.4 KV Cache Compression Effectiveness

| Method | Cache Elements | DeepSeek-V2 |
|--------|----------------|-------------|
| MHA | $2 n_h d_h$ | 32768 |
| GQA (8 groups) | $2 \times 8 \times d_h$ | 2048 |
| **MLA** | $d_c + d_h^R$ | **576** |

DeepSeek-V2 configuration: $d_c = 512$, $d_h^R = 64$, achieving a compression ratio of **56.9×**.

### 2.5 Quality-Efficiency Tradeoff

MLA's 56.9× compression ratio comes from the assumption: **K and V can be losslessly recovered from a 512-dimensional latent space**.

**Compression Dimension Selection**:
- $d_c < 256$: Significant quality degradation
- $d_c \in [256, 768]$: Quality close to MHA
- $d_c > 768$: No further quality improvement

$d_c = 512$ is near the Pareto optimal.

**Application Models**: DeepSeek-V2 (236B), DeepSeek-V3 (671B), DeepSeek-R1.

## 3. Sparse Attention

### 3.1 Core Idea

Standard Softmax Attention has $O(N^2)$ complexity, but not all token pairs are equally important. Sparse attention only computes "important" token pairs, reducing complexity to $O(Nk)$.

| Feature | Sparse Attention | Linear Attention |
|------|------------|------------|
| Complexity | $O(Nk)$ | $O(Nd^2)$ |
| Attention Type | Exact Softmax | Approximate/Alternative |
| Long-range Exact Retrieval | Strong | Weak |

### 3.2 Sliding Window Attention

**Mistral's Sliding Window**: Each position $t$ only attends to tokens in the range $[t-w, t]$.

Key insight: Through stacked Transformer layers, information can propagate "across windows". A 32-layer model with window 4096 can theoretically have a receptive field of 128K.

**StreamingLLM**: Retains Sink Tokens + sliding window, maintaining stability in streaming scenarios with **4 million+ tokens**.

### 3.3 MoBA: Block Sparse Attention

MoBA (Mixture of Block Attention) applies the MoE idea to Attention:

$$\text{MoBA}(q, K, V) = \text{softmax}(qK_{[\mathcal{I}]}^\top)V_{[\mathcal{I}]}$$

**Routing Score Calculation** (parameter-free):

$$s_i = \langle q, \text{mean\_pool}(K_{[\mathcal{I}_i]}) \rangle$$

Select the top-$k$ blocks with highest scores for attention computation (typical setting $k = 12$, block size $L = 4096$).

| Metric | MoBA |
|------|------|
| LM Loss Difference | $< 10^{-3}$ |
| Sparsity @32K | 95.31% |
| Speedup @10M | 16× |

### 3.4 NSA: Native Sparse Attention

Three parallel paths proposed by DeepSeek:

1. **Compression Attention**: Learnable MLP compresses blocks into a single representation, capturing global coarse-grained information
2. **Selection Attention**: Lightning Indexer selects Top-k blocks, preserving fine-grained precise information
3. **Sliding Window**: Guarantees accurate modeling of local context

**End-to-end Trainable**: Uses sparse attention from the pre-training stage.

| Configuration | Value |
|------|-----|
| Compression Block Size | 32 |
| Selection Block Size | 64 |
| Number of Selection Blocks | 16 |
| Sliding Window | 512 |

**Training Speedup** (64K sequence, A100): Forward 9×, Backward 6×.

### 3.5 DSA: DeepSeek Sparse Attention

DSA is the next-generation sparse attention deployed in DeepSeek-V3.2, using **fine-grained token-level retrieval**.

**Core Differences from NSA**:

| Feature | NSA | DSA |
|------|-----|-----|
| Selection Granularity | Block-level | Token-level |
| Number of Branches | 3 | 1 |
| Attention Variant | GQA | MLA |
| Validation Model | 27B | 671B |

Each query only needs to attend to a fixed $k=2048$ most relevant tokens, achieving true **linear complexity** $O(Nk)$.

**Performance Gains**: Long-context API cost reduced by approximately **50%**.

### 3.6 Comparison of Three Methods

| Design Dimension | NSA | MoBA | DSA |
|----------|-----|------|-----|
| Release Date | 2025.02 | 2025.02 | 2025.09 |
| Selection Granularity | Block-level | Block-level | Token-level |
| Routing Mechanism | Learnable MLP | Parameter-free mean-pool | Learnable $w$ |
| Complexity | $O(N^2/L)$ | $O(N \cdot kL)$ | $O(Nk)$ |

**Design Philosophy Differences**:
- **NSA**: Comprehensive coverage, hierarchical fusion
- **MoBA**: Simple elegance, MoE thinking
- **DSA**: Aggressive sparsity, end-to-end optimization

### 3.7 Ring Attention

When sequence length exceeds single GPU memory, Ring Attention splits long sequences across multiple GPUs, implementing distributed attention computation through ring communication.

Key optimization: **Compute-communication overlap**—asynchronously transmit the next KV block while computing attention with the current KV block.

## 4. Linear Attention

### 4.1 Core Idea

Linear attention fundamentally changes the mathematical form of attention, reducing complexity to truly $O(N)$.

**From Softmax to Linearization**:

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot (\phi(K)^\top V)$$

First compute $\phi(K)^\top V$ (a $d \times d$ matrix), then left-multiply by $\phi(Q)$. Total complexity reduces to $O(nd^2)$.

**Recurrent Form** (Transformer as RNN):

$$S_t = S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

Hidden state $S_t$ accumulates historical information, supporting $O(1)$ incremental updates.

### 4.2 Classic Methods

**Linear Transformer**: $\phi(x) = \text{elu}(x) + 1$, achieving up to 4000× speedup on autoregressive tasks.

**Performer**: Uses random features to approximate the softmax kernel, can serve as a drop-in replacement.

**cosFormer**: Uses ReLU to ensure non-negativity, introduces cosine position reweighting.

### 4.3 Linear Attention with Forgetting Gates

Problem with original linear attention: State matrix $S_t$ can only accumulate, cannot forget.

**RetNet**: Introduces exponential decay factor $\gamma$:

$$S_t = \gamma S_{t-1} + v_t k_t^\top$$

**Multi-Scale Retention**: Different heads use different $\gamma$ values, achieving multi-scale memory.

Performance: 7B model on 8k sequences, inference speed **8.4× faster** than Transformer, memory reduced by **70%**.

**Lightning Attention** (MiniMax): Block computation strategy, first linear attention architecture scaled to commercial-grade.

| Parameter | MiniMax-01 |
|------|------------|
| Total Parameters | 456B |
| Active Parameters | 45.9B |
| Training Context Length | 1M tokens |
| Inference Extrapolation Length | 4M tokens |

### 4.4 DeltaNet: Based on Delta Rule

**Memory Overload Problem**: Original linear attention can only add new key-value associations, cannot erase existing information.

**Delta Rule Update**:

$$S_t = S_{t-1} - (S_{t-1} \cdot k_t - v_t) \cdot k_t^\top$$

Intuitive understanding: Calculate the difference (delta) between retrieved value and true value, correct memory based on delta.

**Gated DeltaNet**: Introduces gating mechanism to control forgetting and update strength. Adopted by **Qwen3-Next**.

### 4.5 Test-Time Training Perspective

Modern linear attention can be unified under the **TTT** framework:

| Method | Update Rule | Corresponding Optimizer |
|------|----------|------------|
| Linear Attention | $S_t = S_{t-1} + v_t k_t^\top$ | Accumulated gradient |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | Accumulated with decay |
| DeltaNet | Delta Rule | Precise update |

### 4.6 Industrial Deployment Status

| Company/Model | Architecture Type | Context Length |
|-----------|----------|------------|
| MiniMax-01 | Lightning Attention + MoE | 1M-4M |
| MiniMax-M1 | Lightning Attention | 1M+80k generation |
| Qwen3-Next | Gated DeltaNet | - |

**Key Observation**: MiniMax is currently the only vendor to scale linear attention to commercial-grade.

### 4.7 Limitations and Future Outlook

**Current Limitations**:
- Lags behind Softmax Attention on complex reasoning tasks
- Weaker in-context learning capability
- Unstable long-range exact retrieval

**Development Trends**:
- Hybrid architectures: Linear layers + sparse Softmax layers
- Gating mechanisms: More fine-grained memory management
- Knowledge distillation: Distilling from Softmax models

## Chapter Summary

This chapter introduced four optimization paths for attention mechanisms:

1. **FlashAttention**: IO-aware optimization, doesn't change computation but dramatically reduces memory access
   - v1→v4 continuous evolution, 15× peak performance improvement

2. **MLA**: Low-rank compression of KV Cache
   - DeepSeek-V2 achieves 56.9× compression ratio, approaching MHA performance

3. **Sparse Attention**: Only computes important token pairs
   - NSA/MoBA/DSA represent the evolution of "learned sparsity"
   - DSA validated on 671B model, API cost reduced by 50%

4. **Linear Attention**: Changes computation form, truly $O(N)$ complexity
   - Lightning Attention first scaled to commercial-grade
   - Gated DeltaNet adopted by Qwen3-Next

**Development Trend**: Sparse attention and linear attention are moving from academic research to industrial mainstream. In 2025, both approaches have been validated on 600B+ scale models.

The next post will introduce Mixture of Experts (MoE) architecture.
