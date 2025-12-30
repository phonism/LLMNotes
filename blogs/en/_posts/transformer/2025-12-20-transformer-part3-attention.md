---
layout: post
title: "Transformer Notes (III): Attention Mechanisms"
date: 2025-12-20 10:20:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [Transformer, Attention]
lang: en
translation: /transformer-part3-attention/
series: transformer
series_order: 3
---

The attention mechanism is the core of Transformers, but the standard $O(N^2)$ complexity becomes a bottleneck for long-context modeling. This post systematically introduces four optimization paths: FlashAttention accelerates computation through IO-aware optimization; MLA reduces KV Cache through low-rank compression; sparse attention only computes important token pairs; linear attention fundamentally changes the computation form.

## 1. FlashAttention: IO-Aware Efficient Attention

The Roofline model reveals a key insight: modern GPUs have computational power far exceeding memory bandwidth. What determines performance is often not "how fast can we compute" but "how fast can data arrive." Standard attention is a typical case of this bottleneck—it needs to repeatedly write $O(N^2)$ intermediate results to GPU memory, and these IO operations become the real performance killers.

FlashAttention's core idea is to reorganize the computation order so that the attention matrix **never leaves the GPU's high-speed cache**. This is not a mathematical approximation but an exactly equivalent rearrangement—we trade more computation for less memory access, which suits modern GPUs that have excess compute but scarce bandwidth.

### 1.1 Motivation: GPU Memory Hierarchy

Modern GPUs have computational capabilities far exceeding memory bandwidth. Taking NVIDIA A100 as an example:
- **Compute capability**: 312 TFLOPS (FP16 Tensor Core)
- **HBM bandwidth**: 2 TB/s
- **SRAM capacity**: 20 MB (shared memory + L1 cache)
- **SRAM bandwidth**: approximately 19 TB/s

**Arithmetic Intensity** is defined as FLOPs per byte of memory access:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Access Bytes}}$$

For A100, reaching peak compute requires arithmetic intensity $\geq 156$ FLOPs/Byte. Standard attention's arithmetic intensity is far below this, making it **memory-bound**.

**IO Problem of Standard Attention**:

Standard attention implementation requires multiple HBM reads and writes:
1. Read $Q, K$ from HBM, compute $S = QK^\top$, write back to HBM
2. Read $S$ from HBM, compute $P = \text{softmax}(S)$, write back to HBM
3. Read $P, V$ from HBM, compute $O = PV$, write back to HBM

Total HBM access: $O(N^2 + Nd)$, where $N$ is sequence length and $d$ is head dimension. For long sequences, the $N^2$ term dominates, causing severe IO bottlenecks.

### 1.2 FlashAttention v1: Attention Matrix That Never Materializes

FlashAttention's core idea: **Never write the complete $N \times N$ attention matrix to HBM**.

#### Online Softmax Algorithm

Standard Softmax computation requires two passes:

$$m = \max_j(x_j), \quad \ell = \sum_j e^{x_j - m}, \quad \text{softmax}(x)_i = \frac{e^{x_i - m}}{\ell}$$

**Online Softmax** enables single-pass, incremental computation. For concatenation of two blocks $x^{(1)}, x^{(2)}$:

$$m^{(1)} = \max(x^{(1)}), \quad \ell^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}}$$
$$m^{(2)} = \max(x^{(2)}), \quad \ell^{(2)} = \sum_j e^{x_j^{(2)} - m^{(2)}}$$
$$m^{new} = \max(m^{(1)}, m^{(2)})$$
$$\ell^{new} = e^{m^{(1)} - m^{new}} \ell^{(1)} + e^{m^{(2)} - m^{new}} \ell^{(2)}$$

Output can also be incrementally updated:

$$O^{new} = \frac{1}{\ell^{new}} \left[ e^{m^{old} - m^{new}} \ell^{old} \cdot O^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)} \cdot O^{(j)} \right]$$

#### Tiling Algorithm

FlashAttention divides $Q, K, V$ into blocks of size $B_r \times d$ and $B_c \times d$:
- $B_c = \lceil M / (4d) \rceil$ (constrained by SRAM size $M$)
- $B_r = \min(B_c, d)$

**FlashAttention Forward Pass Algorithm**:

<!-- tikz-source: transformer-flashattention-algorithm-en
\begin{algorithm}[H]
\caption{FlashAttention Forward Pass}
\KwIn{$Q, K, V \in \mathbb{R}^{N \times d}$, block sizes $B_r, B_c$}
\KwOut{$O \in \mathbb{R}^{N \times d}$}
Initialize $O = 0$, $\ell = 0$, $m = -\infty$ (all $N$-dimensional vectors)\;
\For{$j = 1$ \KwTo $\lceil N/B_c \rceil$}{
    Load $K_j, V_j \in \mathbb{R}^{B_c \times d}$ from HBM to SRAM\;
    \For{$i = 1$ \KwTo $\lceil N/B_r \rceil$}{
        Load $Q_i, O_i, \ell_i, m_i$ from HBM to SRAM\;
        Compute $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$ in SRAM\;
        Compute $m_{ij} = \text{rowmax}(S_{ij})$, $\tilde{P}_{ij} = \exp(S_{ij} - m_{ij})$\;
        Compute $\ell_{ij} = \text{rowsum}(\tilde{P}_{ij})$\;
        Update $m_i^{\text{new}}, \ell_i^{\text{new}}$ (Online Softmax update)\;
        Update $O_i = \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}O_i + e^{m_{ij} - m_i^{\text{new}}}\tilde{P}_{ij} V_j)$\;
        Write $O_i, \ell_i^{\text{new}}, m_i^{\text{new}}$ back to HBM\;
    }
}
\Return{$O$}
\end{algorithm}
-->
![FlashAttention Forward Pass Algorithm]({{ site.baseurl }}/assets/figures/transformer-flashattention-algorithm-en.svg)

#### Backward Pass and Recomputation

Standard backpropagation requires storing $S, P \in \mathbb{R}^{N \times N}$. FlashAttention uses a **Recomputation** strategy:
- Forward pass only stores $O$ and statistics $(m, \ell)$
- Backward pass recomputes $S, P$ from $Q, K, V$ blocks

While this increases FLOPs by approximately 30%, it dramatically reduces HBM access, resulting in faster overall speed.

#### IO Complexity Analysis

**Theorem (FlashAttention IO Complexity)**: Given SRAM size $M$, sequence length $N$, and head dimension $d$, FlashAttention's HBM access is:

$$O\left( \frac{N^2 d^2}{M} \right)$$

Standard attention's HBM access is $O(N^2 + Nd)$. When $M = \Theta(Nd)$, FlashAttention reduces HBM access by a factor of $O(N)$.

| Scenario | Speedup | Memory Savings |
|------|--------|----------|
| BERT-large (seq=512) | 1.15× | 5× |
| GPT-2 (seq=1K) | 3× | 10× |
| Long-range (seq=4K) | 2.4× | 20× |

### 1.3 FlashAttention-2: Parallelism Strategy Optimization

FlashAttention-2 optimizes parallelism strategies and work distribution, achieving 230 TFLOPS on A100 (approximately 73% peak utilization), about 2× faster than v1.

#### Parallelism Strategy Improvements

**FlashAttention v1**: Parallelizes across batch and head dimensions, with each thread block processing one attention head. When $\text{batch} \times \text{heads} < 108$ (number of A100 SMs), GPU utilization is low.

**FlashAttention-2**: Additionally parallelizes across the **sequence length dimension**. For long sequences (which typically mean small batch), this significantly improves GPU utilization.

#### Warp Work Distribution

GPU thread hierarchy: Thread → Warp (32 threads) → Thread Block → Grid.

**v1's Sliced-K Scheme**: Partitions $K, V$ among 4 warps, with $Q$ visible to all warps. Problem: requires inter-warp synchronization and shared memory for intermediate results.

**v2's Sliced-Q Scheme**: Partitions $Q$ among 4 warps, with $K, V$ visible to all warps. Advantage: eliminates inter-warp communication and reduces shared memory reads/writes.

#### Reducing Non-Matmul FLOPs

A100's matrix multiplication throughput is **16× higher** than non-matrix multiplication (312 vs 19.5 TFLOPS). v2 reduces non-matmul operations through:
- Optimizing Online Softmax rescaling operations
- Improving boundary checking and causal mask implementation

| Metric | v1 | v2 | Improvement |
|------|----|----|------|
| A100 Peak (TFLOPS) | 124 | 230 | 1.85× |
| GPU Utilization | 25-40% | 50-73% | ~2× |
| GPT-3 Training (TFLOPS) | 173 | 225 | 1.3× |

### 1.4 FlashAttention-3: Hopper Architecture Optimization

FlashAttention-3 is designed for NVIDIA Hopper architecture (H100), fully leveraging new hardware features to achieve 740 TFLOPS (75% peak utilization).

#### New Hopper Hardware Features

**WGMMA (Warpgroup Matrix Multiply-Accumulate)**: New Tensor Core instruction introduced in Hopper, with significantly higher throughput than Ampere's `mma.sync`. A warpgroup (4 warps, 128 threads) can execute large-scale matrix multiplications.

**TMA (Tensor Memory Accelerator)**: Dedicated hardware unit for data transfer between Global Memory and Shared Memory:
- Automatically handles index calculation and boundary checking
- Frees register resources, allowing larger tile sizes
- Supports asynchronous transfer, overlapping with computation

#### Three Major Optimization Techniques

**1. Warp Specialization**: Divides warps into **Producer** and **Consumer**:
- Producer warps: Handle TMA data transfer
- Consumer warps: Handle WGMMA computation

Data transfer and computation are completely asynchronously overlapped.

**2. Ping-Pong Scheduling**: Alternates execution between two warpgroups:
- While Warpgroup 1 executes GEMM, Warpgroup 2 executes Softmax
- Then roles switch

This scheduling improves FP16 forward pass from ~570 TFLOPS to 620 TFLOPS.

**3. Intra-warpgroup Overlapping**: Within a single warpgroup, Softmax computation is pipelined with GEMM:
- While GEMM computes the current block, simultaneously execute Softmax on the previous block
- Further improves to 640-660 TFLOPS

#### FP8 Support

FlashAttention-3 supports FP8 low precision through **Block Quantization** and **Incoherent Processing** (based on Hadamard transform) to reduce quantization error:
- FP8 throughput: approaches 1.2 PFLOPS
- Quantization error is 2.6× lower than baseline FP8 attention

### 1.5 FlashAttention-4: Blackwell Architecture Optimization

FlashAttention-4 is designed for NVIDIA Blackwell architecture (B200), being the first attention kernel to break the PFLOPS barrier.

#### Five-Stage Warp Pipeline

Expanded from v3's 2 stages to a **5-stage pipeline**, with each warp type highly specialized:
1. **Load Warp**: Loads $Q, K, V$ from Global Memory to Shared Memory via TMA
2. **MMA Warp**: Executes matrix multiplication, computes attention scores and output accumulation
3. **Softmax Warps** (8 total): Computes normalized attention scores, maintains running statistics
4. **Correction Warps** (4 total): Rescales output when scaling factors change
5. **Epilogue Warps**: Writes completed output blocks back to Global Memory

#### Software exp2 Emulation

Traditional implementations rely on Special Function Units (SFU) to compute exponential functions, but SFUs are scarce resources. FlashAttention-4 uses **cubic polynomial approximation**:

$$2^x \approx a_0 + a_1 x + a_2 x^2 + a_3 x^3, \quad x \in [0, 1)$$

Efficiently computed using Horner's method with vectorized FMA instructions on CUDA Cores, avoiding SFU bottlenecks.

#### Selective Rescaling

Traditional Online Softmax rescales every time a new maximum is encountered. FlashAttention-4 introduces **threshold checking**: rescaling is only triggered when the maximum value change is sufficient to affect numerical stability. Reportedly, this reduces rescaling operations by approximately 10×, while maintaining numerical precision.

**Performance**:
- ~20% faster than cuDNN attention
- ~2× faster than FlashAttention-3
- ~15× faster than original FlashAttention

### 1.6 Flash Decoding: KV Parallelism During Inference

FlashAttention is optimized for training, but has issues during **inference**. Flash Decoding specifically addresses inference bottlenecks.

#### Inference Problem

During autoregressive generation, each step only generates 1 token, meaning $Q$ has sequence length 1:
- FlashAttention parallelizes across batch and head dimensions
- When $\text{batch} \times \text{heads} < 108$, GPU is severely underutilized
- In long-context scenarios (small batch size), **FlashAttention may only use 1% of the GPU**

#### KV Sequence Length Parallelism

Flash Decoding's core idea: Parallelize across the **KV sequence length** dimension.

1. Split KV Cache into $S$ blocks: $K = [K_1, ..., K_S]$, $V = [V_1, ..., V_S]$
2. Compute partial attention independently for each block:
   $$O_s = \text{softmax}(Q K_s^\top) V_s, \quad (m_s, \ell_s) = \text{statistics}$$
3. Merge results using Log-Sum-Exp:
   $$O = \frac{\sum_s e^{m_s - m_{global}} \ell_s \cdot O_s}{\sum_s e^{m_s - m_{global}} \ell_s}$$

**Performance Improvements**:
- Long sequence decoding speedup up to **8×**
- On CodeLLaMa-34B, attention operations are **50× faster** than FlashAttention
- Sequence length increase from 512 to 64K, generation speed remains nearly unchanged

### 1.7 FlashDecoding++

FlashDecoding++ further optimizes, published at MLSys 2024.

**Asynchronous Softmax**: Flash Decoding's reduction step requires synchronously waiting for all partial results. FlashDecoding++ introduces **Unified Max Value**:
- Pre-estimate a global maximum $m_{unified}$ (based on statistics or heuristics)
- All blocks use the same $m_{unified}$, no synchronization needed
- Fine-grained pipelining, Prefill speedup 1.05×, Decoding speedup 1.14×

**Flat GEMM Optimization**: Inference GEMM shapes are "flat" ($1 \times N$), standard implementations are inefficient:
- cuBLAS/CUTLASS has up to 50% performance loss for this shape
- FlashDecoding++ uses Double Buffering and targeted optimizations
- Flat GEMM speedup up to 52%

### 1.8 FlashAttention Version Evolution

| Version | GPU | Peak TFLOPS | Utilization | Speedup vs v1 |
|------|-----|-------------|--------|--------------|
| v1 | A100 | 124 | 40% | 1× |
| v2 | A100 | 230 | 73% | 1.85× |
| v2 | H100 | 335 | 35% | 2.7× |
| v3 | H100 | 740 | 75% | 6× |
| v3 (FP8) | H100 | 1200 | - | 9.7× |
| v4 | B200 | PFLOPS | - | 15× |

### 1.9 Engineering Impact of FlashAttention

FlashAttention has become the **standard** for modern LLM training and inference:
- **PyTorch 2.0+**: Built-in `scaled_dot_product_attention` uses FlashAttention
- **vLLM, TensorRT-LLM**: Inference engines use it by default
- **All major LLMs**: GPT-4, Claude, LLaMA, DeepSeek all use FlashAttention

**Context Length Revolution**: FlashAttention increased practical context lengths from 2-4K to 128K+:
- Memory reduced from $O(N^2)$ to $O(N)$
- 64K sequences require 16GB memory with standard attention, FlashAttention only needs ~1GB

> **When to Use FlashAttention**: FlashAttention provides the most benefit in these scenarios:
> - **Long sequences**: Sequence length > 512
> - **Large batch**: Fully utilize GPU parallelism
> - **Training**: Memory savings allow larger batches
>
> For short sequences and small models, standard attention may be faster (reduced kernel launch overhead).

### 1.10 FlexAttention: Programmable FlashAttention

FlexAttention is a new API introduced in PyTorch 2.5, providing a flexible programming interface for FlashAttention.

**Motivation**: While FlashAttention is efficient, each attention variant (Causal, ALiBi, Sliding Window, etc.) requires specialized implementation. When researchers want to experiment with new variants, they often need to hand-write Triton kernels. FlexAttention automatically generates efficient kernels through `torch.compile`, reducing development time from weeks to minutes.

**Core API**: FlexAttention provides two functional interfaces:
- **score_mod**: Modifies the score matrix after $QK^\top$ (e.g., adding position bias)
- **mask_mod**: Defines mask pattern (positions returning True participate in computation)

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Causal mask
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# ALiBi position encoding
def alibi(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx) * slope[h]

# Usage
block_mask = create_block_mask(causal, B, H, Q_LEN, KV_LEN)
out = flex_attention(q, k, v, score_mod=alibi, block_mask=block_mask)
```

**Performance**: FlexAttention achieves approximately **85-90%** of FlashAttention-2's performance, but improves development efficiency by 100×. For variants not natively supported by FlashAttention (such as Document Masking), FlexAttention is 5-8× faster than standard SDPA.

## 2. Multi-head Latent Attention (MLA)

FlashAttention solves the IO bottleneck of attention matrices during training, but during inference there's another memory challenge: **KV Cache**. Autoregressive generation requires caching all historical tokens' Key and Value vectors. As context length increases, this memory usage can exceed the model parameters themselves.

Multi-head Latent Attention (MLA) is DeepSeek-V2's solution to this problem. Its core insight is: although each attention head needs independent K and V, they may have a **low-rank structure**—i.e., they can be recovered from a shared low-dimensional "latent vector". This compression differs from GQA/MQA's forced sharing; instead, it lets the network learn the optimal compression method.

### 2.1 KV Cache Challenge

During autoregressive generation, KV Cache becomes the primary memory bottleneck for long-context inference:

$$\text{KV Cache Size} = 2 \times B \times S \times L \times n_h \times d_h \times \text{bytes}$$

For large models (e.g., $n_h = 128$, $d_h = 128$), KV Cache becomes the main memory bottleneck for long-context inference. For a 70B model ($L=80$, $K=8$, $S=8192$), KV Cache reaches **2.1 GB/request**.

### 2.2 Limitations of Existing Solutions

| Method | KV Cache Size | Performance | Principle |
|--------|---------------|-------------|------|
| MHA | $2 n_h d_h$ | Optimal | Independent KV per head |
| GQA | $2 \frac{n_h}{g} d_h$ | Slight degradation | $g$ Q heads share KV |
| MQA | $2 d_h$ | Significant degradation | All heads share KV |
| **MLA** | $d_c + d_h^R$ | **Close to MHA** | Low-rank compression |

GQA/MQA reduce cache by **forcing sharing** of KV heads, but this forced sharing often harms model performance. MLA's core insight: **KV can be recovered from a low-dimensional latent vector**, without explicit sharing.

### 2.3 MLA Core Principles

MLA's core idea is to compress high-dimensional Key and Value into a shared low-dimensional latent vector, then recover K and V from this vector during inference.

#### Low-rank Compression of KV

For input $\mathbf{h}_t \in \mathbb{R}^d$, first compress to latent vector:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

where $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the down-projection matrix, $d_c \ll n_h d_h$ is the compression dimension.

Recover K and V from latent vector:

$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$

where $W^{UK}, W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}$ are up-projection matrices.

#### Low-rank Compression of Query

Similarly, Query can also be low-rank compressed (mainly to reduce activation memory during training):

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t, \quad \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q$$

where $W^{DQ} \in \mathbb{R}^{d_c' \times d}$, $W^{UQ} \in \mathbb{R}^{n_h d_h \times d_c'}$.

#### Decoupled RoPE

RoPE needs to apply rotation at each position, but if RoPE is directly applied to the compressed $\mathbf{c}_t^{KV}$, it breaks subsequent weight absorption optimization. MLA adopts a **Decoupled RoPE** strategy:

1. Split each attention head into two parts:
   - **Content part** ($d_h^C$ dimensions): Recovered from compressed vector, RoPE **not** applied
   - **Position part** ($d_h^R$ dimensions): Additional projection, RoPE applied

2. Final Q and K are concatenations of both parts:

$$\mathbf{q}_t = [\mathbf{q}_t^C; \text{RoPE}(\mathbf{q}_t^R, t)]$$
$$\mathbf{k}_t = [\mathbf{k}_t^C; \text{RoPE}(\mathbf{k}_t^R, t)]$$

Where position parts are computed as:

$$\mathbf{q}_t^R = W^{QR} \mathbf{c}_t^Q, \quad \mathbf{k}_t^R = W^{KR} \mathbf{h}_t$$

> **Why Decoupled RoPE?** RoPE is position-dependent: $\text{RoPE}(\mathbf{x}, t)$ depends on position $t$. If RoPE is applied to $\mathbf{c}_t^{KV}$ then K is recovered, then $\mathbf{k}_t = W^{UK} \cdot \text{RoPE}(\mathbf{c}_t^{KV}, t)$, and $W^{UK}$ cannot be absorbed into $W^Q$ (because RoPE is in between). The Decoupled strategy isolates RoPE to separate dimensions, preserving the possibility of weight absorption.

#### Weight Absorption

A key optimization of MLA is **Weight Absorption**. Since there's no nonlinear activation between compression and recovery, matrices can be merged:

**Query-Key Absorption**: Computing attention scores:

$$\mathbf{q}_t^{C\top} \mathbf{k}_s^C = (\mathbf{c}_t^Q)^\top (W^{UQ})^\top W^{UK} \mathbf{c}_s^{KV} = (\mathbf{c}_t^Q)^\top \underbrace{W^{QK}}_{\text{absorbed}} \mathbf{c}_s^{KV}$$

where $W^{QK} = (W^{UQ})^\top W^{UK} \in \mathbb{R}^{d_c' \times d_c}$.

**Output-Value Absorption**: Computing output projection:

$$W^O \mathbf{v}_t^C = W^O W^{UV} \mathbf{c}_t^{KV} = \underbrace{W^{OV}}_{\text{absorbed}} \mathbf{c}_t^{KV}$$

where $W^{OV} = W^O W^{UV} \in \mathbb{R}^{d \times d_c}$.

**Inference Flow**: After weight absorption, during inference:
1. Cache $\mathbf{c}_t^{KV}$ and $\mathbf{k}_t^R$ (position part)
2. Use $W^{QK}$ to directly compute content part attention scores
3. Use absorbed $W^{OV}$ to compute output

### 2.4 KV Cache Compression Effectiveness

| Method | Cache Elements | DeepSeek-V2 (Specific Values) |
|--------|----------------|---------------------|
| MHA | $2 n_h d_h$ | $2 \times 128 \times 128 = 32768$ |
| GQA (8 groups) | $2 \times 8 \times d_h$ | $2 \times 8 \times 128 = 2048$ |
| **MLA** | $d_c + d_h^R$ | $512 + 64 = 576$ |

DeepSeek-V2 configuration: $d_c = 512$, $d_h^R = 64$, achieving compression ratio of $\frac{32768}{576} \approx \mathbf{56.9×}$.

### 2.5 PyTorch Implementation

Below is a simplified PyTorch implementation of MLA:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,      # Model dimension
        n_heads: int,      # Number of attention heads
        d_c: int,          # KV compression dimension
        d_c_q: int,        # Q compression dimension
        d_head_r: int,     # RoPE dimension per head
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_c = d_c
        self.d_head_r = d_head_r

        # KV compression
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, d_model, bias=False)
        self.W_uv = nn.Linear(d_c, d_model, bias=False)

        # Q compression
        self.W_dq = nn.Linear(d_model, d_c_q, bias=False)
        self.W_uq = nn.Linear(d_c_q, d_model, bias=False)

        # Decoupled RoPE projections
        self.W_qr = nn.Linear(d_c_q, n_heads * d_head_r, bias=False)
        self.W_kr = nn.Linear(d_model, n_heads * d_head_r, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope_fn, kv_cache=None):
        B, T, D = x.shape

        # KV compression: only c_kv needs caching
        c_kv = self.W_dkv(x)  # (B, T, d_c)

        # Decoupled RoPE keys
        k_r = self.W_kr(x)    # (B, T, n_heads * d_head_r)
        k_r = rope_fn(k_r)    # Apply RoPE

        # Handle KV cache
        if kv_cache is not None:
            c_kv_cached, k_r_cached = kv_cache
            c_kv = torch.cat([c_kv_cached, c_kv], dim=1)
            k_r = torch.cat([k_r_cached, k_r], dim=1)
        new_cache = (c_kv, k_r)

        # Q compression
        c_q = self.W_dq(x)    # (B, T, d_c_q)
        q_c = self.W_uq(c_q)  # (B, T, D) - content part
        q_r = self.W_qr(c_q)  # (B, T, n_heads * d_head_r) - RoPE part
        q_r = rope_fn(q_r)

        # Reconstruct K, V from compressed cache
        k_c = self.W_uk(c_kv)  # (B, S, D) - content part
        v = self.W_uv(c_kv)    # (B, S, D)

        # Reshape for multi-head attention
        q_c = q_c.view(B, T, self.n_heads, self.d_head)
        k_c = k_c.view(B, -1, self.n_heads, self.d_head)
        v = v.view(B, -1, self.n_heads, self.d_head)
        q_r = q_r.view(B, T, self.n_heads, self.d_head_r)
        k_r = k_r.view(B, -1, self.n_heads, self.d_head_r)

        # Concatenate content and RoPE parts
        q = torch.cat([q_c, q_r], dim=-1)  # (B, T, H, d_head + d_head_r)
        k = torch.cat([k_c, k_r], dim=-1)  # (B, S, H, d_head + d_head_r)

        # Scaled dot-product attention
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        scale = (self.d_head + self.d_head_r) ** -0.5
        attn = F.softmax(q @ k.transpose(-2,-1) * scale, dim=-1)
        out = attn @ v  # (B, H, T, d_head)

        # Output projection
        out = out.transpose(1,2).reshape(B, T, D)
        out = self.W_o(out)

        return out, new_cache
```

> **Implementation Optimization**: The code above is for educational purposes. In actual deployment:
> - **Weight Absorption**: Pre-compute $W^{QK} = (W^{UQ})^\top W^{UK}$ and $W^{OV} = W^O W^{UV}$
> - **FlashAttention**: Use FlashAttention to accelerate attention computation
> - **Fused Operators**: Fuse multiple small matrix multiplications

### 2.6 MLA vs Other Methods

| Feature | MHA | GQA | MQA | MLA |
|---------|-----|-----|-----|-----|
| KV Cache | $2n_h d_h$ | $2\frac{n_h}{g} d_h$ | $2 d_h$ | $d_c + d_h^R$ |
| Parameters | Baseline | Reduced | Minimum | Slightly increased |
| Expressiveness | Strongest | Strong | Weaker | Close to MHA |
| Inference Latency | High | Medium | Low | Medium |
| Long Context | Limited | Good | Good | **Best** |

**MLA Advantages**:
- **Extreme compression**: KV Cache reduced by over 93%
- **Performance preservation**: Unlike GQA/MQA forced sharing, learns optimal compression
- **Long context friendly**: Makes 128K context possible

**MLA Costs**:
- **Compute overhead**: Requires additional compression/recovery computation
- **Implementation complexity**: Decoupled RoPE and weight absorption increase implementation difficulty
- **Training cost**: Low-rank constraint may require more training

### 2.7 Quality-Efficiency Tradeoff

MLA's 56.9× compression ratio comes from a strong assumption: **K and V can be losslessly recovered from a 512-dimensional latent space**. Under what conditions does this assumption hold?

#### Validity of Low-rank Assumption

Consider the diversity requirements of attention patterns. If a task requires $r$ essentially different attention patterns (e.g., local dependencies, long-range dependencies, syntactic structure, semantic associations, etc.), then KV representation needs at least $r$ degrees of freedom.

**Lower Bound on Compression Dimension**: If $d_c < r$, compression causes information loss. DeepSeek chose $d_c = 512$, implicitly assuming that common NLP task attention patterns can be expressed in a space of no more than 512 dimensions.

**Task Dependency**: Different tasks have different attention diversity requirements:
- **Text generation**: Relatively fixed patterns, low-rank assumption holds, MLA performs well
- **Code understanding**: Needs to track complex variable dependencies and scopes, may require higher-rank representations
- **Mathematical reasoning**: Multi-step reasoning needs to maintain multiple reasoning chains, high attention diversity requirements

#### Selection Guide

| Scenario | Recommended Method | Rationale |
|------|----------|------|
| Long-context inference (>32K) | MLA | KV Cache is main bottleneck |
| Short context, high throughput | GQA | Simple implementation, low overhead |
| Quality priority (small batch) | MHA | No compression loss |
| Code/Math tasks | GQA or MHA | High attention diversity requirements |
| Edge device deployment | MLA | Extreme memory compression |

**Rule of Thumb**: When $\text{context length} \times \text{batch size} > 10^6$, KV Cache becomes the main bottleneck, and MLA's benefits become apparent.

#### Compression Ratio vs Quality Pareto Frontier

DeepSeek's ablation experiments show that $d_c$ selection has a "sweet spot":
- $d_c < 256$: Significant quality degradation, over-compressed
- $d_c \in [256, 768]$: Quality close to MHA, compression effective
- $d_c > 768$: No further quality improvement, diminishing compression returns

$d_c = 512$ is near Pareto optimal—further compression harms quality, further expansion wastes cache space.

### 2.8 Applications and Extensions

MLA has been applied in the following models:
- **DeepSeek-V2**: First proposed, 236B parameter MoE model
- **DeepSeek-V3**: Further optimized, 671B parameters
- **DeepSeek-R1**: Reasoning model, inherits MLA architecture

> **Synergy between MLA and MoE**: MLA is particularly suitable for MoE architecture: MoE's sparse activation already reduces computation, while MLA further addresses memory bottleneck. The combination allows DeepSeek-V2 to maintain high performance while reducing training cost by 42.5%.

## 3. Sparse Attention

FlashAttention and MLA optimize attention from computation efficiency and memory usage perspectives respectively, but they both preserve the full $O(N^2)$ attention computation—they just make this computation faster and more memory-efficient. This section explores another path: if most attention weights are close to zero anyway, can we **skip these meaningless computations**?

### 3.1 Core Idea

Standard Softmax Attention has $O(N^2)$ complexity, but in practice not all token pairs are equally important:
- Attention distributions are usually sparse (a few tokens get most of the weight)
- Distant tokens typically have weak attention
- Semantically related tokens tend to cluster at specific positions

The core idea of sparse attention: **Only compute important token pairs**, reducing complexity from $O(N^2)$ to $O(Nk)$, where $k \ll N$.

| Feature | Sparse Attention | Linear Attention |
|------|------------|------------|
| Complexity | $O(Nk)$ | $O(Nd^2)$ |
| Attention Type | Exact Softmax | Approximate/Alternative |
| Long-range Exact Retrieval | Strong | Weak |
| KV Cache | Requires full | Compressible |
| Compatibility with Original Transformer | High | Medium |

### 3.2 Sliding Window Attention

Sliding Window Attention (SWA) is the most intuitive form of sparse attention: each token only attends to tokens within a fixed window around it.

#### Mistral's Sliding Window

Mistral 7B is the first open-source model to deploy sliding window attention at scale, with window size 4096.

**Core Mechanism**: Each position $t$ token only attends to tokens in range $[t-w, t]$:

$$\text{Attention}_t = \text{softmax}\left(\frac{q_t K_{[t-w:t]}^\top}{\sqrt{d}}\right) V_{[t-w:t]}$$

**Cross-layer Information Propagation**: The key insight of sliding window is that through stacked Transformer layers, information can propagate "across windows". At layer $k$, the token at position $t$ can actually access information from range $[t - k \cdot w, t]$. For a 32-layer model with window size 4096, the theoretical receptive field can reach 128K.

**Inference Optimization**:
- **Rolling Buffer**: KV Cache only needs to retain the most recent $w$ tokens
- **Memory Savings**: 50% cache savings at 8K sequence length
- **Speed Improvement**: Combined with FlashAttention, 2× speedup at 16K sequences

#### StreamingLLM: Infinite Length Input

StreamingLLM solves an important problem: how to let LLMs process "infinitely long" streaming input.

**Attention Sink Phenomenon**: Research found that regardless of input length, models always assign abnormally high attention weight to **the first few tokens**—even if these tokens are not semantically important. This is called "Attention Sink".

**SinkAttention**: Retains two parts of KV Cache:
- **Sink Tokens**: First 4 tokens of sequence (fixed)
- **Sliding Window**: Most recent $w$ tokens

$$\text{KV Cache} = \text{Sink}_{[1:4]} \cup \text{Window}_{[t-w:t]}$$

**Effect**: Maintains stable performance in streaming scenarios with **4 million+ tokens**, while standard sliding window collapses beyond pre-training length.

### 3.3 KV Cache Sparsification

KV Cache sparsification is sparse attention during inference: dynamically discarding "unimportant" KV entries.

**H2O (Heavy-Hitter Oracle)**: Based on an observation—a few "heavy-hitter" tokens accumulate most attention weight. Algorithm: maintain cumulative attention score for each token, retain Top-k tokens with highest scores + most recent sliding window tokens.

**SnapKV**: Attention patterns are mostly determined during prefill phase, can prune "once and for all". Analyze attention distribution at the end of prefill, identify important positions, permanently retain KV at these positions.

**PyramidKV**: Different layers have different attention sparsity—lower layers have more dispersed attention needing more KV, higher layers have concentrated attention allowing aggressive compression. Uses **pyramid-shaped** KV allocation: more at bottom, less at top.

| Method | Pruning Timing | Dynamicity | Integration Framework |
|------|----------|--------|----------|
| H2O | Every step | Dynamic | vLLM |
| SnapKV | After Prefill | Static | vLLM |
| StreamingLLM | Continuous | Static | -- |
| PyramidKV | Layer-level | Static | -- |

### 3.4 MoBA: Block Sparse Attention

MoBA (Mixture of Block Attention) is a representative block-level sparse attention work, deployed in Kimi's long-context service.

#### Core Idea: Applying MoE to Attention

MoBA's core insight: **Not all context is equally important to the current token**. Rather than computing attention over the entire sequence, let the model learn "which blocks to focus on".

$$\text{MoBA}(q, K, V) = \text{softmax}(qK_{[\mathcal{I}]}^\top)V_{[\mathcal{I}]}$$

where $\mathcal{I} \subseteq [N]$ is the selected KV subset, determined by the routing mechanism.

#### Block Partitioning and Routing

**Block Partitioning**: Evenly divide the context of length $N$ into $n$ blocks, each of size $B = N/n$:

$$\mathcal{I}_i = [(i-1) \cdot B + 1, \, i \cdot B], \quad i = 1, \ldots, n$$

**Routing Score Calculation** (parameter-free): For each query $q$, compute affinity score with each block:

$$s_i = \langle q, \text{mean\_pool}(K_{[\mathcal{I}_i]}) \rangle$$

i.e., inner product of query with the mean vector of all keys in the block. This is a **parameter-free** routing mechanism.

**Top-k Selection**: Select the $k$ blocks with highest scores for attention computation (typical setting $k = 12$, block size $L = 4096$).

#### Causality Guarantee

In autoregressive scenarios:
1. **Future block masking**: For query at position $t$, all blocks with $i > \lceil t/B \rceil$ have $s_i = -\infty$
2. **Current block forced selection**: The block containing the query is always routed

| Metric | MoBA | Full Attention |
|------|------|----------------|
| LM Loss Difference | \multicolumn{2}{c}{$< 10^{-3}$} |
| Sparsity @32K | \multicolumn{2}{c}{95.31%} |
| Speedup @1M | \multicolumn{2}{c}{6.5×} |
| Speedup @10M | \multicolumn{2}{c}{16×} |

### 3.5 NSA: Native Sparse Attention

Native Sparse Attention (NSA) is a hierarchical sparse attention mechanism proposed by DeepSeek.

#### Three Attention Paths

NSA decomposes attention computation into three parallel paths:

**1. Compression Attention**: Uses learnable MLP to compress consecutive tokens into block-level representations:

$$\tilde{K}_i = \text{MLP}(K_{[(i-1)l+1:il]}), \quad \tilde{V}_i = \text{MLP}(V_{[(i-1)l+1:il]})$$

where $l$ is compression block size (NSA uses $l=32$). This captures **global coarse-grained** information.

**2. Selection Attention**: Selects most relevant blocks through Lightning Indexer to maintain original precision:
- Compute query relevance score with all blocks
- Select Top-$n$ blocks (NSA uses $n=16$ blocks, block size $l'=64$)
- Perform exact Softmax attention on selected blocks

This preserves **fine-grained precise** information.

**3. Sliding Window Attention**: Full attention on the most recent $w$ tokens (NSA uses $w=512$). This guarantees accurate modeling of **local context**.

#### Lightning Indexer

Lightning Indexer is NSA's core innovation for efficient block selection:
- Maintains separate **FP8 quantized** Key cache (not MLA's KV Cache)
- Each query computes relevance score with all blocks
- Selects Top-k blocks (default 2048 tokens)
- Hardware optimized: Efficient CUDA kernel implemented with DeepGEMM

**Key Design**: Index computation is separated from attention computation; indexing uses low precision for speed.

#### End-to-end Trainable

Unlike ClusterKV, MagicPIG, and other methods relying on non-differentiable operations, NSA is **natively trainable**—using sparse attention from the pre-training stage.

| Configuration | Value |
|------|-----|
| Compression Block Size | 32 |
| Selection Block Size | 64 |
| Number of Selection Blocks | 16 |
| Sliding Window | 512 |

**Training Speedup** (64K sequence, A100): Forward 9×, Backward 6×. Speedup increases with sequence length: 4× at 8K, 6.4× at 16K, 9.1× at 32K, 11.6× at 64K.

### 3.6 DSA: DeepSeek Sparse Attention

DSA (DeepSeek Sparse Attention, September 2025) is the next-generation sparse attention deployed in DeepSeek-V3.2, fundamentally different from NSA. DSA abandons NSA's complex three-branch design, adopting simpler **fine-grained token-level retrieval**.

#### Algorithm Design

DSA's core idea: Each query only needs to attend to a fixed number $k$ of most relevant tokens ($k=2048$).

**Importance Score Calculation**: DSA introduces learnable weight $w$ to compute token importance:

$$\text{score}_i = w \cdot f(q, k_i)$$

This is a compromise—simpler than NSA's MLP, but more expressive than MoBA's parameter-free mean-pooling.

**Top-k Retrieval**: Select Top-$k$ tokens based on importance scores for exact attention computation:

$$\mathcal{I} = \text{Top-}k(\{\text{score}_i\}_{i=1}^N), \quad |\mathcal{I}| = 2048$$

**Complexity**: Single query accesses fixed $k$ tokens, so overall complexity is $O(Nk)$, true **linear complexity**.

#### Core Differences from NSA

| Feature | NSA | DSA |
|------|-----|-----|
| Selection Granularity | Block-level | Token-level |
| Number of Branches | 3 (compress+select+window) | 1 (direct selection) |
| Importance Calculation | Learnable MLP | Learnable $w$ weight |
| Attention Variant | GQA | MLA |
| Validation Model | 27B | 671B |

#### Engineering Implementation

- **TileLang Kernel**: Fine-grained sparsity + MLA requires custom kernels, TileLang outperforms Triton
- **vLLM/SGLang Integration**: Day-0 support, using DeepGEMM and FlashMLA
- **Blackwell Optimization**: Collaboration with NVIDIA for B200 optimization

**Performance Gains**: Long-context API cost reduced by approximately **50%**, significant speedup on 64K sequences, quality nearly lossless on 671B model.

### 3.7 Comparison of Three Methods

| Design Dimension | NSA | MoBA | DSA |
|----------|-----|------|-----|
| Release Date | 2025.02 | 2025.02 | 2025.09 |
| Proposer | DeepSeek | Moonshot (Kimi) | DeepSeek |
| Selection Granularity | Block-level | Block-level | Token-level |
| Routing Mechanism | Learnable MLP | Parameter-free mean-pool | Learnable $w$ |
| Local Window | Yes ($w$=512) | Current block forced | No |
| Complexity | $O(N^2/L)$ | $O(N \cdot kL)$ | $O(Nk)$ |

#### Hyperparameter Comparison

| Parameter | NSA | MoBA | DSA |
|------|-----|------|-----|
| Block Size | $l$=32, $l'$=64 | $L$=4096 | -- |
| Selection Count | $n$=16 blocks | $k$=12 blocks | $k$=2048 tokens |
| Sliding Window | $w$=512 | -- | -- |
| @32K Accessed Tokens | ~2560 | 49152 | 2048 |
| @32K Sparsity | 92% | 0% (sees all) | 94% |

**Key Observation**: At 32K length, MoBA has almost no sparsity (selected 12 blocks × 4096 = 49152 > 32K)! MoBA's sparsity advantage only appears at longer sequences (e.g., 128K+).

#### Design Philosophy Differences

- **NSA**: Comprehensive coverage, hierarchical fusion—three branches fused through learnable gating, misses no important information
- **MoBA**: Simple elegance, MoE thinking—treats KV Cache as "expert pool", parameter-free routing lets attention scores naturally decide selection
- **DSA**: Aggressive sparsity, end-to-end optimization—Token-level selection, each query only sees 2048 tokens (approximately 3%@64K)

#### Applicable Scenarios

| Method | Applicable Scenarios |
|------|----------|
| NSA | Need precise preservation of multi-scale information; can accept complex hyperparameter tuning; using GQA architecture |
| MoBA | Pursuing simple design; want seamless replacement of existing Attention; sequence length 128K+ |
| DSA | Using MLA architecture; pursuing extreme sparsity; ultra-large models (100B+) |

### 3.8 Ring Attention

When sequence length exceeds single GPU memory, Ring Attention splits long sequences across multiple GPUs, implementing distributed attention computation through ring communication.

**Algorithm Flow**:
1. Split Query, Key, Value along sequence dimension across $P$ GPUs
2. Each GPU holds local Query block, computes attention with local KV
3. KV blocks are passed in ring fashion between GPUs, accumulating global attention
4. Uses Online Softmax to avoid numerical overflow

**Communication Hiding**: Key optimization is **compute-communication overlap**—asynchronously transmit the next KV block while computing attention with the current block.

**LLaMA 3's Context Parallelism**: Uses All-Gather style Context Parallelism—first All-Gather to collect all KV, then compute local Query's attention. For load balancing, split sequence into $2 \times \text{CP}$ blocks and shuffle, supporting efficient training at 128K context.

### 3.9 Sparse Attention Method Overview

| Method | Sparsity Strategy | Complexity | Global Information | Deployment |
|------|----------|--------|----------|------|
| Sliding Window | Fixed window | $O(Nw)$ | None | Mistral |
| StreamingLLM | Sink+window | $O(N(s+w))$ | Sink tokens | -- |
| MoBA | Block routing | $O(N \cdot kL)$ | Top-k blocks | Kimi |
| NSA | Compress+select+window | $O(N^2/L)$ | Compress+select | -- |
| DSA | Token-level retrieval | $O(Nk)$ | Top-k tokens | DeepSeek-V3.2 |
| Ring Attention | Distributed | $O(N^2/P)$ | Complete | LLaMA 3 |

> **Evolution of Sparse Attention**: From Longformer/BigBird's "hand-designed patterns" to MoBA/NSA/DSA's "learned sparsity", sparse attention is undergoing a paradigm shift. In 2025, sparse attention was first validated on 600B+ scale models (DeepSeek-V3.2), marking the technology's transition from academic research to industrial mainstream.

## 4. Linear Attention

Sparse attention reduces $O(N^2)$ to $O(Nk)$ by "only computing important token pairs", but it still preserves the softmax computation form. This section introduces a more radical path: **fundamentally changing the mathematical form of attention**, reducing complexity to truly $O(N)$.

### 4.1 Core Idea

The core insight of linear attention: softmax attention requires $O(N^2)$ because it must first compute the complete $N \times N$ attention matrix before normalization. If we replace softmax with other functions that allow computation to be "rearranged", we can avoid explicitly constructing this matrix.

#### From Softmax Attention to Linearization

Standard self-attention (single head) computation:

$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)}_{n \times n} V$$

This requires explicitly constructing the $n \times n$ attention matrix, with both time and space complexity $O(n^2)$.

The core idea of linear attention: rewrite softmax or $QK^\top$ in a decomposable form, using associativity of multiplication to change the computation order:

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot \underbrace{(\phi(K)^\top V)}_{d \times d}$$

where $\phi(\cdot)$ is some feature mapping function. The key is: first compute $\phi(K)^\top V$ (a $d \times d$ matrix linear in length $n$), then left-multiply by $\phi(Q)$. Total complexity reduces to $O(nd^2)$, approximately $O(n)$ when $d \ll n$.

#### Recurrent Form: Transformer as RNN

In autoregressive (causal) scenarios, linear attention can be written in RNN-like recurrent form. Let $q_t, k_t, v_t$ be the query, key, value vectors at step $t$, define state matrix $S_t \in \mathbb{R}^{d \times d}$:

$$S_t = S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

This reveals a deep connection: **Linear Attention is essentially an RNN**, with hidden state $S_t$ accumulating historical information.

> **Dual-Mode of Linear Attention**: Linear attention supports two equivalent computation modes:
> - **Parallel mode**: Uses matrix multiplication during training, fully utilizing GPU parallelism
> - **Recurrent mode**: Uses RNN form during inference, enabling $O(1)$ incremental updates
>
> This dual-mode property allows linear attention to achieve optimal efficiency in both training and inference phases.

### 4.2 Classic Methods

#### Linear Transformer

The first work to systematically propose "Transformer as RNN". Core idea is to rewrite softmax attention in kernel form:

$$\text{Attention}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$

where $\phi(x) = \text{elu}(x) + 1$ ensures non-negativity. Experiments show up to **4000× speedup** on autoregressive tasks.

**Limitation**: Simple feature mappings struggle to precisely approximate softmax behavior, with performance gaps on complex language tasks.

#### Performer

Proposes FAVOR+ (Fast Attention Via positive Orthogonal Random features), using random features to approximate the softmax kernel:

$$\text{softmax}(q^\top k) \approx \phi(q)^\top \phi(k)$$

where $\phi$ is constructed using orthogonal random features, with unbiased or approximately unbiased theoretical guarantees.

**Advantage**: Fully compatible with original Transformer, can serve as drop-in replacement.
**Limitation**: Random approximation still has precision loss in practice.

#### cosFormer

Rather than hard-approximating softmax, designs a linear alternative based on two key properties of softmax:
1. **Non-negativity**: Attention weights should be non-negative
2. **Concentration**: Attention should concentrate at relevant positions

cosFormer uses ReLU to ensure non-negativity, and introduces cosine position reweighting:

$$\text{Attention}_{ij} = \text{ReLU}(q_i)^\top \text{ReLU}(k_j) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i - j}{n}\right)$$

Achieved state-of-the-art performance on Long-Range Arena and other long-sequence benchmarks, representing "usable linear Attention".

### 4.3 Linear Attention with Forgetting Gates

A fundamental problem with original linear attention: state matrix $S_t$ can only accumulate, cannot forget. As sequence grows, historical information "crowds together", degrading retrieval capability.

#### RetNet

Microsoft's Retentive Network introduces exponential decay factor $\gamma \in (0, 1)$:

$$S_t = \gamma S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

This applies exponential decay to historical information, emphasizing the importance of recent tokens.

**Multi-Scale Retention**: Different attention heads use different $\gamma$ values, achieving multi-scale memory retention:
- Small $\gamma$: Focus on recent information (short-range dependencies)
- Large $\gamma$: Retain longer history (long-range dependencies)

**Three Computation Modes**:
1. **Parallel mode**: Matrix computation during training
2. **Recurrent mode**: $O(1)$ inference
3. **Chunk-wise recurrent mode**: Efficient processing of long sequences

**Performance**: 7B model on 8k sequences, inference speed **8.4× faster** than Transformer, memory reduced by **70%**.

#### Lightning Attention

Lightning Attention proposed by MiniMax is currently the **first linear attention architecture scaled to commercial-grade**. Core innovation:

**Block Computation Strategy**: Divides attention computation into intra-block and inter-block:
- **Intra-block**: Uses "left-multiply" form within blocks, parallelizable
- **Inter-block**: Uses "right-multiply" form between blocks, accumulating state

This decomposition avoids the slow cumsum operation in traditional linear attention.

**Hybrid Architecture**: Every 8 layers, 7 use Lightning Attention, 1 uses standard Softmax Attention, balancing efficiency and precision.

| Parameter | MiniMax-01 |
|------|------------|
| Total Parameters | 456B |
| Active Parameters (MoE) | 45.9B |
| Number of Experts | 32 |
| Training Context Length | 1M tokens |
| Inference Extrapolation Length | 4M tokens |

### 4.4 DeltaNet: Based on Delta Rule

#### Motivation: Solving Memory Overload

The core defect of original linear attention is **memory overload**: can only add new key-value associations, cannot erase existing information. This causes retrieval errors to accumulate as sequence grows.

#### Delta Rule Update

DeltaNet introduces the "out with the old, in with the new" Delta Rule:

$$S_t = S_{t-1} - \underbrace{(S_{t-1} \cdot k_t - v_t)}_{\text{delta}} \cdot k_t^\top$$

Intuitive understanding:
1. $S_{t-1} \cdot k_t$: Use current key to retrieve value from memory
2. $S_{t-1} \cdot k_t - v_t$: Compute difference (delta) between retrieved and true value
3. Correct memory based on delta, achieving "precise update"

#### Gated DeltaNet

ICLR 2025 work further introduces gating mechanism:

$$S_t = \alpha_t \odot S_{t-1} + \beta_t \odot (v_t - S_{t-1} \cdot k_t) \cdot k_t^\top$$

where $\alpha_t$ controls forgetting, $\beta_t$ controls update strength.

**Complementarity**: Gating enables fast memory erasure, Delta Rule enables precise memory update. The combination outperforms Mamba2 and original DeltaNet on multiple benchmarks.

**Industrial Adoption**: Gated DeltaNet has been adopted by **Qwen3-Next** as the linear attention layer.

### 4.5 Connection to State Space Models

Mamba is another important efficient sequence modeling approach, based on Selective State Space Models (Selective SSM). The Mamba-2 paper reveals the **Structured State Space Duality (SSD)**:

> "Compared to standard self-attention, SSD has only two differences: removing softmax normalization, and applying a separate element-wise mask matrix."

This indicates linear attention and SSM can be viewed as different instances of the same framework:
- **Linear Attention**: Decomposes attention matrix through feature mapping
- **SSM**: Models sequences through state space equations
- Both have linear complexity and recurrent form

**Hybrid Architectures**: In practice, pure linear models still have gaps on some tasks, leading to hybrid architectures:
- **Jamba** (AI21): Mamba + Attention
- **MiniMax-01**: Lightning Attention + Sparse Softmax Attention
- **Qwen3-Next**: Gated DeltaNet + SwiGLU

### 4.6 Test-Time Training Perspective

Su Jianlin pointed out in "A Brief History of Linear Attention" that modern linear attention can be unified under the **Test-Time Training** (TTT) framework:

> "View sequence modeling as an online learning problem, construct RNN using optimizers. Different loss functions correspond to different RNN models."

| Method | Update Rule | Corresponding Optimizer |
|------|----------|------------|
| Linear Attention | $S_t = S_{t-1} + v_t k_t^\top$ | Accumulated gradient |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | Accumulated with decay |
| DeltaNet | $S_t = S_{t-1} - (S_{t-1}k_t - v_t)k_t^\top$ | Delta Rule |
| Gated DeltaNet | Gated Delta Rule | Adaptive learning rate |

This perspective provides principled guidance for designing new linear attention: choose appropriate "optimizers" to update memory state.

### 4.7 Industrial Deployment Status

| Company/Model | Architecture Type | Context Length | Features |
|-----------|----------|------------|------|
| MiniMax-01 | Lightning Attention + MoE | 1M-4M | First commercial-grade linear Attention LLM |
| MiniMax-M1 | Lightning Attention | 1M+80k generation | Open-source reasoning model |
| Qwen3-Next | Gated DeltaNet | -- | Linear layer + gated Attention |

**Key Observation**: MiniMax is currently the only vendor to scale linear attention to commercial-grade. Other vendors (like Kimi, DeepSeek) prefer the sparse attention route.

### 4.8 Limitations and Future Outlook

**Current Limitations**:
1. **Precision gap**: Pure linear attention still lags behind Softmax Attention on complex reasoning tasks
2. **In-context learning capability**: Linear models typically have weaker few-shot capability than Transformers
3. **Long-range exact retrieval**: Unstable performance on tasks like passkey retrieval

**Development Trends**:
1. **Hybrid architectures**: Combining linear layers with sparse Softmax layers
2. **Gating mechanisms**: More fine-grained memory management (like Gated DeltaNet)
3. **Knowledge distillation**: Distilling from Softmax models to linear models (like LAWCAT)
4. **TTT principles**: Designing new architectures based on optimizer perspective

> **Historical Assessment**: Su Jianlin's evaluation—"Linear attention has evolved from simply imitating Softmax Attention to 'feeding back' to it—applying DeltaNet improvements to Softmax Attention through kernel tricks. This shows the field is thriving with broad exploration space."

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
