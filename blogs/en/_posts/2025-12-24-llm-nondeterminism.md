---
layout: post
title: "Nondeterminism in LLM Inference: Root Cause Analysis and Batch Invariance Solutions"
date: 2025-12-24 12:00:00
author: Qi Lu
tags: [LLM, Inference, Determinism, Batch Invariance, Reproducibility, CUDA]
lang: en
translation: /llm-nondeterminism/
---

## Problem Statement

In LLM inference services, identical inputs should produce identical outputs. However, empirical observations show that even with greedy decoding (temperature=0), outputs remain nondeterministic. The following experimental data comes from Thinking Machines Lab[1](#ref-1):

| Model | Samples | Distinct Outputs | Most Frequent Output Count |
|-------|---------|------------------|---------------------------|
| Qwen3-235B-A22B | 1000 | 80 | 78 |

This phenomenon contradicts the mathematical definition of greedy decoding: $\hat{y}\_t = \arg\max\_v p(v \mid y\_{\<t}, x)$ should be deterministic.

The objectives of this article are: (1) identify the root cause of nondeterminism; (2) design engineering-deployable solutions.

---

## Hypothesis Testing: Concurrent Floating-Point vs. Batch Size Variation

### Hypothesis 1: Floating-Point Non-Associativity from GPU Concurrency

The mainstream hypothesis suggests that GPU parallel computation introduces nondeterminism. The reasoning chain is:

1. Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$
2. GPU parallel reduction execution order depends on thread scheduling
3. Different execution orders produce different accumulation results
4. Small differences are amplified by argmax, changing the output token

The typical supporting evidence for this hypothesis is the nondeterminism of CUDA's `atomicAdd` operation.

### Problems with Hypothesis 1

Thinking Machines Lab's[1](#ref-1) analysis points out that core operations in modern Transformer inference do not rely on atomic operations:

- **GEMM**: Uses cuBLAS tiled matrix multiplication with fixed reduction tree structure
- **LayerNorm/RMSNorm**: Standard implementations use deterministic warp-level reduction
- **Attention**: FlashAttention's tiled implementation also uses fixed reduction order

Experimental verification: With a single GPU and fixed batch size, multiple inference runs on the same input produce completely identical outputs. This rules out thread-level nondeterminism as the primary source.

### Hypothesis 2: Batch Size Variation Causes Numerical Path Divergence

The real problem is: **kernel numerical output is a function of batch size**.

<!-- tikz-source: nondeterminism-batch-path-divergence
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=1.4cm, minimum height=0.5cm, align=center, font=\scriptsize},
    kernel/.style={draw, rounded corners, fill=blue!10, minimum width=1.2cm, minimum height=0.45cm, align=center, font=\scriptsize},
    label/.style={font=\tiny, gray},
    arrow/.style={->, >=stealth}
]
    % Request
    \node[box, fill=gray!20] (req) at (0, 0) {Request $x$};

    % Split point
    \coordinate (split) at (1.2, 0);

    % Path labels
    \node[font=\tiny, anchor=east] at (1.1, 1.2) {BS=1};
    \node[font=\tiny, anchor=east] at (1.1, -1.2) {BS=4};

    % Path 1 (upper)
    \node[kernel] (k1a) at (2.5, 1.2) {RMSNorm};
    \node[kernel] (k1b) at (4, 1.2) {MatMul};
    \node[kernel] (k1c) at (5.5, 1.2) {Attn};
    \node[box, fill=green!20] (out1) at (7, 1.2) {$\ell_1$};

    % Path 2 (lower)
    \node[kernel] (k2a) at (2.5, -1.2) {RMSNorm};
    \node[kernel] (k2b) at (4, -1.2) {MatMul};
    \node[kernel] (k2c) at (5.5, -1.2) {Attn};
    \node[box, fill=red!20] (out2) at (7, -1.2) {$\ell_2$};

    % Connections
    \draw[arrow] (req) -- (split);
    \draw[arrow] (split) |- (k1a);
    \draw[arrow] (split) |- (k2a);
    \draw[arrow] (k1a) -- (k1b);
    \draw[arrow] (k1b) -- (k1c);
    \draw[arrow] (k1c) -- (out1);
    \draw[arrow] (k2a) -- (k2b);
    \draw[arrow] (k2b) -- (k2c);
    \draw[arrow] (k2c) -- (out2);

    % Middle annotations
    \node[label] at (2.5, 0) {tiling};
    \node[label] at (4, 0) {reduction};
    \node[label] at (5.5, 0) {split-K};

    % Results differ
    \node[font=\scriptsize, red] at (7, 0) {$\neq$};
\end{tikzpicture}
-->
![Batch Size Variation Causes Numerical Path Divergence]({{ site.baseurl }}/assets/figures/nondeterminism-batch-path-divergence.svg)

When inference services use dynamic batching, the same request may be assigned to batches of different sizes at different times. Each kernel's tiling strategy and reduction partitioning may change with batch size, leading to different numerical results.

---

## Mathematical Definition of Batch Invariance

Let kernel $K$ operate on input tensor $X \in \mathbb{R}^{B \times N \times D}$, producing output $Y = K(X)$.

**Batch Invariance** requires: for any sample $x_i$ in the batch, its output $y_i$ depends only on $x_i$ itself, independent of other samples in the batch:

$$K(X)[i] = K'(x_i), \quad \forall i \in [1, B]$$

where $K'$ is an equivalent single-sample kernel.

In practice, this property does not hold in standard kernel implementations. The reason lies in the **batch dependency of tiling strategies**.

---

## Batch Variance Analysis of Key Kernels

### RMSNorm

RMSNorm is computed as:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}$$

The key step is reduction: $\sum_{i=1}^{D} x_i^2$.

Standard CUDA implementations typically use **tree reduction**:

<!-- tikz-source: nondeterminism-tree-reduction
\begin{tikzpicture}[
    node/.style={circle, draw, minimum size=0.6cm, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % Level 0
    \foreach \i in {0,...,7} {
        \node[node, fill=blue!20] (l0-\i) at (\i*1.2, 0) {$x_\i$};
    }

    % Level 1
    \foreach \i in {0,...,3} {
        \pgfmathtruncatemacro{\left}{2*\i}
        \pgfmathtruncatemacro{\right}{2*\i+1}
        \node[node, fill=green!20] (l1-\i) at (\left*1.2 + 0.6, 1.2) {$+$};
        \draw[arrow] (l0-\left) -- (l1-\i);
        \draw[arrow] (l0-\right) -- (l1-\i);
    }

    % Level 2
    \foreach \i in {0,1} {
        \pgfmathtruncatemacro{\left}{2*\i}
        \pgfmathtruncatemacro{\right}{2*\i+1}
        \node[node, fill=orange!20] (l2-\i) at (\left*2.4 + 1.2, 2.4) {$+$};
        \draw[arrow] (l1-\left) -- (l2-\i);
        \draw[arrow] (l1-\right) -- (l2-\i);
    }

    % Level 3
    \node[node, fill=red!20] (l3) at (3.6, 3.6) {$\sum$};
    \draw[arrow] (l2-0) -- (l3);
    \draw[arrow] (l2-1) -- (l3);

    % Annotations
    \node[font=\scriptsize, gray, right] at (9, 0) {Elements};
    \node[font=\scriptsize, gray, right] at (9, 1.8) {Fixed reduction order};
    \node[font=\scriptsize, gray, right] at (9, 3.6) {Deterministic result};
\end{tikzpicture}
-->
![Tree Reduction Diagram]({{ site.baseurl }}/assets/figures/nondeterminism-tree-reduction.svg)

**The problem**: When batch size changes, the CUDA kernel may choose different block size and grid configurations. Different configurations result in different reduction partitioning:

- Batch size = 1: May use 256 threads/block, reduction in 4 rounds
- Batch size = 8: May use 128 threads/block, reduction in 8 rounds

Different partitioning produces different intermediate rounding errors, with final results differing at the ULP (Unit in the Last Place) level.

### Matrix Multiplication

Standard tiled GEMM implementation: $C = AB$, where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$.

cuBLAS selects optimal tiling configuration based on matrix dimensions and GPU architecture. For example:

| Configuration | Tile Size | Reduction Partitioning |
|---------------|-----------|------------------------|
| Small matrix | 64×64 | K dimension split into 2 blocks |
| Large matrix | 128×128 | K dimension split into 4 blocks |

When batch size changes, the effective matrix dimension $M' = B \times M$ changes, triggering different tiling choices, resulting in different K-dimension reduction partitioning.

**Mathematical expression**: Let the K dimension be split into $P$ blocks, each of size $k_p$:

$$C_{ij} = \sum_{p=1}^{P} \sum_{l \in \text{block}_p} A_{il} B_{lj}$$

The order of floating-point additions depends on $P$ and block boundaries. Different $P$ values produce different results.

### Attention

The core of FlashAttention is **tiled computation + Online Softmax**. The incremental update formulas for Softmax:

$$m^{new} = \max(m^{old}, m^{(j)})$$
$$\ell^{new} = e^{m^{old} - m^{new}} \ell^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)}$$
$$O^{new} = \frac{e^{m^{old} - m^{new}} \ell^{old} \cdot O^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)} \cdot O^{(j)}}{\ell^{new}}$$

The key parameter is **KV split size**: how many blocks to split the KV cache into for incremental computation.

<!-- tikz-source: nondeterminism-attention-split
\begin{tikzpicture}[
    block/.style={draw, minimum width=1.2cm, minimum height=0.6cm, align=center, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % Split = 2
    \node[font=\small\bfseries] at (-1, 2) {Split = 2};
    \node[block, fill=blue!20] (kv1a) at (1, 2) {KV$_1$};
    \node[block, fill=blue!20] (kv1b) at (2.5, 2) {KV$_2$};
    \node[block, fill=green!20] (o1a) at (4.5, 2) {$O^{(1)}$};
    \node[block, fill=green!30] (o1b) at (6, 2) {$O^{(2)}$};
    \node[block, fill=orange!20] (out1) at (8, 2) {Output};

    \draw[arrow] (kv1a) -- (o1a);
    \draw[arrow] (kv1b) -- (o1b);
    \draw[arrow] (o1a) -- (out1);
    \draw[arrow] (o1b) -- (out1);

    % Split = 4
    \node[font=\small\bfseries] at (-1, 0) {Split = 4};
    \node[block, fill=blue!15] (kv2a) at (0.5, 0) {KV$_1$};
    \node[block, fill=blue!15] (kv2b) at (1.5, 0) {KV$_2$};
    \node[block, fill=blue!15] (kv2c) at (2.5, 0) {KV$_3$};
    \node[block, fill=blue!15] (kv2d) at (3.5, 0) {KV$_4$};
    \node[block, fill=green!15] (o2a) at (5, 0) {$O^{(1)}$};
    \node[block, fill=green!20] (o2b) at (6, 0) {$O^{(2)}$};
    \node[block, fill=green!25] (o2c) at (7, 0) {$O^{(3)}$};
    \node[block, fill=green!30] (o2d) at (8, 0) {$O^{(4)}$};
    \node[block, fill=red!20] (out2) at (10, 0) {Output'};

    \draw[arrow] (kv2a) -- (o2a);
    \draw[arrow] (kv2b) -- (o2b);
    \draw[arrow] (kv2c) -- (o2c);
    \draw[arrow] (kv2d) -- (o2d);
    \draw[arrow] (o2a) -- (out2);
    \draw[arrow] (o2b) -- (out2);
    \draw[arrow] (o2c) -- (out2);
    \draw[arrow] (o2d) -- (out2);

    % Not equal
    \node[font=\small, red] at (9, 1) {$\neq$};

    % Annotation
    \node[font=\scriptsize, gray] at (4.5, -1) {More rescaling steps $\Rightarrow$ More rounding error accumulation};
\end{tikzpicture}
-->
![Attention Split Size Impact]({{ site.baseurl }}/assets/figures/nondeterminism-attention-split.svg)

During decoding, FlashAttention dynamically selects split size based on KV cache length and GPU configuration. Different split sizes lead to different numbers of rescaling operations, accumulating different rounding errors.

---

## Error Propagation: From ULP to Token Divergence

Individual kernel numerical differences are typically on the order of $10^{-6}$ to $10^{-8}$. How does this lead to token-level divergence?

### Error Accumulation Model

Let a Transformer have $L$ layers, with relative error $\epsilon$ per layer. The final logits relative error is approximately:

$$\epsilon_{total} \approx L \cdot \epsilon$$

For $L = 80$ (like Llama-70B) and $\epsilon = 10^{-7}$:

$$\epsilon_{total} \approx 8 \times 10^{-6}$$

### Argmax Fragility

When two tokens have close probabilities, small logits differences can flip the argmax:

$$\Delta \ell = \ell_1 - \ell_2$$

If $\|\Delta \ell\| < \epsilon\_{total}$, the result is unstable.

Empirical observations show that in greedy decoding, approximately 1-5% of token positions are in this "fragile" state. Once divergence occurs, subsequent generation is completely different, exhibiting a **butterfly effect**.

<!-- tikz-source: nondeterminism-butterfly-effect
\begin{tikzpicture}[
    token/.style={draw, rounded corners, minimum width=0.8cm, minimum height=0.5cm, align=center, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % Common prefix
    \node[token, fill=gray!20] (t1) at (0, 0) {The};
    \node[token, fill=gray!20] (t2) at (1.2, 0) {quick};
    \node[token, fill=gray!20] (t3) at (2.4, 0) {brown};

    % Divergence point
    \node[token, fill=red!30] (t4a) at (4, 0.8) {fox};
    \node[token, fill=blue!30] (t4b) at (4, -0.8) {dog};

    % Branch A
    \node[token, fill=red!20] (t5a) at (5.2, 0.8) {jumps};
    \node[token, fill=red!20] (t6a) at (6.4, 0.8) {over};
    \node[token, fill=red!20] (t7a) at (7.6, 0.8) {...};

    % Branch B
    \node[token, fill=blue!20] (t5b) at (5.2, -0.8) {runs};
    \node[token, fill=blue!20] (t6b) at (6.4, -0.8) {fast};
    \node[token, fill=blue!20] (t7b) at (7.6, -0.8) {...};

    % Connections
    \draw[arrow] (t1) -- (t2);
    \draw[arrow] (t2) -- (t3);
    \draw[arrow] (t3) -- (3.5, 0) -- (3.5, 0.8) -- (t4a);
    \draw[arrow] (t3) -- (3.5, 0) -- (3.5, -0.8) -- (t4b);
    \draw[arrow] (t4a) -- (t5a);
    \draw[arrow] (t5a) -- (t6a);
    \draw[arrow] (t6a) -- (t7a);
    \draw[arrow] (t4b) -- (t5b);
    \draw[arrow] (t5b) -- (t6b);
    \draw[arrow] (t6b) -- (t7b);

    % Annotation
    \node[font=\scriptsize] at (3.5, 0.3) {$\Delta \ell < 10^{-5}$};
    \node[font=\scriptsize, red] at (4, 0) {Divergence};
\end{tikzpicture}
-->
![Butterfly Effect of Token Divergence]({{ site.baseurl }}/assets/figures/nondeterminism-butterfly-effect.svg)

---

## Solution: Batch-Invariant Kernel Design

### Design Principles

Core principles for achieving batch invariance:

1. **Fixed tiling configuration**: Use fixed block size and grid configuration regardless of input dimensions
2. **Fixed reduction order**: Ensure reduction tree structure is independent of batch size
3. **Fixed split size**: Use fixed KV split in Attention, not dynamically adjusted by sequence length

### Batch-Invariant RMSNorm Implementation

Key modification: **Enforce fixed reduction partitioning**.

```python
# Standard implementation (batch-variant)
def rmsnorm_standard(x, gamma, eps):
    # Reduction partitioning decided by CUDA runtime
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma

# Batch-invariant implementation
def rmsnorm_batch_invariant(x, gamma, eps, fixed_block_size=256):
    # Enforce fixed partitioning
    D = x.shape[-1]
    num_blocks = (D + fixed_block_size - 1) // fixed_block_size

    # Block accumulation in fixed order
    sq_sum = 0.0
    for i in range(num_blocks):
        start = i * fixed_block_size
        end = min(start + fixed_block_size, D)
        sq_sum = sq_sum + torch.sum(x[..., start:end] ** 2, dim=-1, keepdim=True)

    rms = torch.sqrt(sq_sum / D + eps)
    return x / rms * gamma
```

Actual CUDA implementations need to ensure fixed warp-level reduction order, typically through explicit `__shfl_down_sync` sequences.

### Batch-Invariant MatMul Implementation

cuBLAS auto-tuning selects different kernels based on matrix dimensions. Solutions:

1. **Disable auto-tuning**: Force use of fixed GEMM kernel
2. **Pad to fixed dimensions**: Pad matrices to $2^n$ sizes to ensure consistent tiling

```python
# Use torch.Library to replace default kernel
import batch_invariant_ops

# Automatically replaces all matmul with batch-invariant versions
# Internally uses fixed tile size = 128x128
```

### Batch-Invariant Attention Implementation

Key: **Fixed KV split size**.

```python
# FlashAttention batch-invariant configuration
flash_attn_func(
    q, k, v,
    deterministic=True,      # Enable deterministic mode
    fixed_split_kv=64        # Fix KV split to 64
)
```

SGLang's implementation supports batch-invariant mode for multiple attention backends:

| Backend | Fixed Split Size | Performance Overhead |
|---------|------------------|---------------------|
| FlashInfer | 64 | ~30% |
| FlashAttention 3 | 128 | ~25% |
| Triton | 64 | ~40% |

---

## Multi-GPU Scenarios: Deterministic AllReduce

In Tensor Parallelism, AllReduce operations also introduce nondeterminism. NCCL's default implementation uses ring-based or tree-based algorithms, where reduction order depends on inter-GPU communication latency.

### Deterministic AllReduce

Solution: Use fixed-order reduce-scatter + all-gather:

<!-- tikz-source: nondeterminism-allreduce
\begin{tikzpicture}[
    gpu/.style={draw, rounded corners, minimum width=1.5cm, minimum height=0.8cm, align=center, font=\small},
    data/.style={draw, fill=blue!20, minimum width=0.4cm, minimum height=0.4cm},
    arrow/.style={->, thick, >=stealth}
]
    % GPUs
    \node[gpu, fill=green!20] (g0) at (0, 0) {GPU 0};
    \node[gpu, fill=green!20] (g1) at (3, 0) {GPU 1};
    \node[gpu, fill=green!20] (g2) at (6, 0) {GPU 2};
    \node[gpu, fill=green!20] (g3) at (9, 0) {GPU 3};

    % Data
    \node[data] (d0) at (0, -1) {};
    \node[data] (d1) at (3, -1) {};
    \node[data] (d2) at (6, -1) {};
    \node[data] (d3) at (9, -1) {};

    % Fixed order reduce
    \node[font=\scriptsize] at (4.5, -2) {Fixed order: GPU 0 $\rightarrow$ 1 $\rightarrow$ 2 $\rightarrow$ 3};

    \draw[arrow, red] (d0) -- (1.5, -1) -- (1.5, -1.5) -- (d1);
    \draw[arrow, red] (d1) -- (4.5, -1) -- (4.5, -1.5) -- (d2);
    \draw[arrow, red] (d2) -- (7.5, -1) -- (7.5, -1.5) -- (d3);

    % Annotation
    \node[font=\scriptsize, gray] at (4.5, -3) {Deterministic reduction: $((d_0 + d_1) + d_2) + d_3$};
\end{tikzpicture}
-->
![Deterministic AllReduce]({{ site.baseurl }}/assets/figures/nondeterminism-allreduce.svg)

SGLang implements deterministic tensor parallelism, ensuring complete reproducibility in multi-GPU scenarios.

---

## Additional Challenges for MoE Models

Mixture-of-Experts (MoE) architectures introduce sources of nondeterminism that don't exist in dense models. MoE architectures like Qwen3-235B-A22B and DeepSeek-V3, with their sparse activation characteristics, face additional challenges for deterministic inference.

### Nondeterminism in Token Routing

The core of MoE is the **gating network**, which decides which experts each token is routed to:

$$G(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$$

The problem: When multiple experts have close gating scores, small numerical perturbations can change the TopK selection result.

<!-- tikz-source: nondeterminism-moe-routing
\begin{tikzpicture}[
    token/.style={draw, circle, minimum size=0.6cm, font=\scriptsize},
    expert/.style={draw, rounded corners, minimum width=1.2cm, minimum height=0.8cm, align=center, font=\small},
    gate/.style={draw, rounded corners, fill=yellow!20, minimum width=2cm, minimum height=0.6cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth}
]
    % Tokens
    \node[token, fill=blue!20] (t1) at (0, 2) {$t_1$};
    \node[token, fill=blue!20] (t2) at (0, 0) {$t_2$};
    \node[token, fill=blue!20] (t3) at (0, -2) {$t_3$};

    % Gate
    \node[gate] (gate) at (3, 0) {Gating\\Network};

    % Experts
    \node[expert, fill=green!20] (e1) at (7, 2) {Expert 1};
    \node[expert, fill=green!20] (e2) at (7, 0) {Expert 2};
    \node[expert, fill=green!20] (e3) at (7, -2) {Expert 3};

    % Connections
    \draw[arrow] (t1) -- (gate);
    \draw[arrow] (t2) -- (gate);
    \draw[arrow] (t3) -- (gate);

    % Routing (solid = stable, dashed = unstable)
    \draw[arrow, thick, green!60!black] (gate) -- node[above, font=\scriptsize] {0.45} (e1);
    \draw[arrow, thick, green!60!black] (gate) -- node[above, font=\scriptsize] {0.35} (e2);
    \draw[arrow, dashed, red] (gate) -- node[below, font=\scriptsize] {0.20} (e3);

    % Annotation
    \node[font=\scriptsize, red] at (5, -3) {When $|s_2 - s_3| < \epsilon$, routing is unstable};
\end{tikzpicture}
-->
![MoE Token Routing Diagram]({{ site.baseurl }}/assets/figures/nondeterminism-moe-routing.svg)

### Expert Capacity and Token Dropping

To ensure computational efficiency, MoE typically sets **Expert Capacity**: the maximum number of tokens each expert can process. When an expert receives more tokens than its capacity, excess tokens are **dropped**, passing directly to the next layer via residual connections.

$$\text{Expert Capacity} = \frac{\text{Batch Tokens} \times \text{Capacity Factor}}{\text{Num Experts}}$$

Sources of nondeterminism from token dropping:

1. **Batch composition changes**: Different batches have different token distributions; the same token may be processed in some batches and dropped in others
2. **Routing order dependency**: When capacity is near saturation, tokens arriving first are processed, later ones are dropped
3. **Load imbalance**: Popular experts are more likely to trigger dropping

### Training-Inference Inconsistency

Research shows significant differences in routing behavior between training and inference phases of MoE models:

> "Even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes."

This inconsistency is particularly severe in RL training:

- **On-policy requirement**: RL training requires rollouts to use the same policy as training
- **Routing drift**: Inference routing decisions may deviate from training distribution
- **Reward noise**: Nondeterministic routing introduces hard-to-track reward fluctuations

### Solutions for Deterministic MoE Inference

#### 1. Fixed Routing Threshold

Avoid unstable decisions when gating scores are close:

```python
def deterministic_topk(scores, k, margin=1e-5):
    # Use fixed tie-breaking rule when score difference < margin
    sorted_scores, indices = torch.sort(scores, descending=True)

    # Detect tie situations
    for i in range(k-1, len(sorted_scores)-1):
        if sorted_scores[i] - sorted_scores[i+1] < margin:
            # Use expert index as tie-breaker
            pass

    return indices[:k]
```

#### 2. Disable Token Dropping

Trade computational efficiency for determinism:

```python
# Set large enough capacity factor to prevent dropping
config.moe_capacity_factor = 2.0  # or higher
config.moe_drop_tokens = False
```

#### 3. Soft MoE

Use soft routing instead of hard routing, where each token is assigned to all experts with weights:

$$y = \sum_{i=1}^{E} g_i(x) \cdot \text{Expert}_i(x)$$

Soft MoE eliminates discrete routing decisions, but has higher computational overhead.

### Special Impact on RL Training

MoE models face dual challenges in RL training:

1. **Numerical nondeterminism**: The batch variance problem described above
2. **Structural nondeterminism**: Routing decision instability

Research shows that MoE RL training instability partly stems from routing distribution drift:

> "The routing distribution has been identified as a pivotal factor contributing to the instability of MoE RL."

Experimental data shows that approximately 10% of routers select different experts between training and inference, and 94% of tokens select a different expert in at least one layer.

### Routing Replay: R2 and R3

To address training-inference routing inconsistency, researchers proposed the **Routing Replay** mechanism, with the core idea of replaying inference-phase routing decisions during training.

#### R2: Vanilla Routing Replay

**Definition**: Reuse experts selected by the **training system** during rollout.

```
Rollout (Training System) → Record routing decisions → Replay during training
```

- **Pros**: Simple implementation, low overhead
- **Cons**: Effectiveness decreases when off-policy degree is large

#### R3: Rollout Routing Replay

**Definition**: Reuse experts selected by the **inference system** during rollout.

```
Rollout (Inference System) → Record routing decisions → Replay during training
```

R3 enforces the constraint: specific experts activated during rollout must be strictly reused during training backpropagation.

<!-- tikz-source: nondeterminism-r2-r3
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.5cm, minimum height=0.8cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth},
    dasharrow/.style={->, thick, >=stealth, dashed}
]
    % R2
    \node[font=\bfseries] at (0, 3) {R2: Vanilla Routing Replay};
    \node[box, fill=blue!20] (r2-train) at (-2, 2) {Training\\System};
    \node[box, fill=green!20] (r2-rollout) at (2, 2) {Rollout};
    \node[box, fill=orange!20] (r2-bp) at (2, 0.5) {Backprop};

    \draw[arrow] (r2-train) -- node[above, font=\scriptsize] {Generate} (r2-rollout);
    \draw[arrow] (r2-rollout) -- node[right, font=\scriptsize] {Replay routing} (r2-bp);
    \draw[dasharrow, gray] (r2-train) -- (r2-bp);

    % R3
    \node[font=\bfseries] at (8, 3) {R3: Rollout Routing Replay};
    \node[box, fill=purple!20] (r3-infer) at (6, 2) {Inference\\System};
    \node[box, fill=green!20] (r3-rollout) at (10, 2) {Rollout};
    \node[box, fill=orange!20] (r3-bp) at (10, 0.5) {Backprop};

    \draw[arrow] (r3-infer) -- node[above, font=\scriptsize] {Generate} (r3-rollout);
    \draw[arrow] (r3-rollout) -- node[right, font=\scriptsize] {Replay routing} (r3-bp);

    % Annotations
    \node[font=\scriptsize, gray] at (2, -0.5) {Training system routing};
    \node[font=\scriptsize, gray] at (10, -0.5) {Inference system routing};
\end{tikzpicture}
-->
![R2 vs R3 Comparison]({{ site.baseurl }}/assets/figures/nondeterminism-r2-r3.svg)

#### Choosing Between R2 and R3

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Small off-policy degree | R2 | R3's cost of modifying target policy outweighs benefits |
| Large off-policy degree | R3 | Maintaining validity of first-order approximation is more important |

#### Implementation Details

R3 can be integrated with KV Cache:

```python
# Store routing mask for each layer and token prefix
routing_cache = {}

def forward_with_replay(x, layer_idx, prefix_hash):
    if prefix_hash in routing_cache:
        # Cache hit, reuse routing decision
        mask = routing_cache[prefix_hash]
    else:
        # Compute new routing
        mask = compute_routing(x)
        routing_cache[prefix_hash] = mask

    # Use mask for softmax gating, but don't block gradient flow
    return apply_routing(x, mask, allow_grad=True)
```

Key point: The replayed mask is used for softmax gating computation but does not block gradient flow through standard router weights, ensuring the router remains trainable.

### Other Solutions

#### RSPO (Router-Shift Policy Optimization)

Unlike R2/R3's hard constraints, RSPO uses a soft adjustment mechanism:

$$w_{rspo} = w_{is} \cdot \text{clip}(\text{router\_shift\_ratio}, 1-\epsilon, 1+\epsilon)$$

- Compute **router shift ratio** between current and old policies
- Quantify routing deviation for each token
- Adaptively reweight updates, limiting excessive updates while maintaining router flexibility

#### Freeze Router

The simplest approach is freezing router parameters, but this harms model adaptability and is generally not recommended.

---

## Performance Analysis

Main overhead sources for batch-invariant kernels:

1. **Abandoning dynamic optimization**: Cannot use optimal kernels for specific sizes
2. **Fixed tiling inefficiency**: Small matrices using large tiles waste resources
3. **Additional synchronization overhead**: Ensuring fixed execution order requires more synchronization points

### Benchmarks

| Approach | Throughput (tokens/s) | Relative Overhead |
|----------|----------------------|-------------------|
| Standard vLLM | 1000 | baseline |
| TML batch_invariant_ops | 385 | 61.5% |
| SGLang + CUDA Graphs | 657 | 34.3% |

CUDA Graphs eliminates kernel launch overhead by precompiling execution graphs, significantly reducing performance loss in deterministic mode.

### Latency Distribution

Another advantage of deterministic mode is **reduced latency variance**:

```
Standard mode:      P50=45ms, P99=120ms, stddev=25ms
Deterministic mode: P50=52ms, P99=58ms,  stddev=3ms
```

For latency-sensitive applications, lower variance may be more important than lower mean.

---

## Use Cases

### Reproducibility in RL Training

In RLHF/RLVR training, policy rollouts need to correspond precisely with training steps. Nondeterministic inference causes:

1. **Debugging difficulty**: Cannot reproduce specific failure cases
2. **Training instability**: Same checkpoint behaves inconsistently across different machines
3. **Incomparable experiments**: Cannot distinguish algorithmic improvements from random fluctuations

SGLang's collaboration with slime achieved 100% reproducible RL training:

> "Taking this deterministic inference capability further, SGLang collaborated with the slime team to unlock 100% reproducible RL training."

### Safety and Compliance

For AI systems requiring auditing, deterministic behavior is a fundamental requirement:

- Medical diagnosis
- Financial decisions
- Legal analysis

Nondeterminism means inability to trace and explain the origin of specific outputs.

### Model Debugging

When models produce anomalous outputs, deterministic inference allows:

1. Precisely reproduce the problem
2. Layer-by-layer inspection of intermediate states
3. Binary search to locate the source of issues

---

## Related Work

### LayerCast (NeurIPS 2025)

Yuan et al. proposed an alternative approach: reduce error accumulation by increasing numerical precision.

- **Method**: Store weights in FP16/BF16, compute in FP32
- **Advantage**: No kernel modification required
- **Limitation**: Cannot completely eliminate batch variance, only reduces its impact

Experiments show LayerCast reduced accuracy fluctuation of DeepSeek-R1-Distill-Qwen-7B from 9% to 2%, but still not fully deterministic.

### OpenAI seed Parameter

OpenAI API provides a `seed` parameter to improve reproducibility:

> "If specified, our system will make a best effort to sample deterministically..."

But the official documentation explicitly states **determinism is not guaranteed**, due to:

1. Backend model updates
2. Load balancing to different hardware
3. System configuration changes

The `system_fingerprint` field is used to track configuration changes, but cannot guarantee complete determinism with the same fingerprint.

---

## Conclusion

Nondeterminism in LLM inference stems from **kernel implementations' sensitivity to batch size**, not GPU concurrency characteristics. By designing batch-invariant kernels, fully deterministic inference can be achieved at the engineering level.

**Key techniques for dense models**:

1. Fix RMSNorm reduction partitioning
2. Fix MatMul tiling configuration
3. Fix Attention KV split size
4. Use deterministic AllReduce in multi-GPU scenarios

**Additional challenges for MoE models**:

1. Discrete routing decisions in token routing are unstable when gating scores are close
2. Expert capacity and token dropping introduce batch-dependent nondeterminism
3. Training-inference routing inconsistency requires Routing Replay (R2/R3) mechanisms

The performance cost is approximately 30-60%, but is a necessary investment for RL training, model debugging, and safety auditing scenarios.

---

## References

<a id="ref-1"></a>[1] Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). 2025.

<a id="ref-2"></a>[2] Yuan, J. et al. [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501). NeurIPS 2025 (Oral).

<a id="ref-3"></a>[3] LMSYS Org. [Towards Deterministic Inference in SGLang and Reproducible RL Training](https://lmsys.org/blog/2025-09-22-sglang-deterministic/). 2025.

<a id="ref-4"></a>[4] Thinking Machines Lab. [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops). GitHub.

<a id="ref-5"></a>[5] vLLM Documentation. [Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/).

<a id="ref-6"></a>[6] Dao, T. et al. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). 2023.

<a id="ref-7"></a>[7] Ma, W. et al. [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370). 2025.

<a id="ref-8"></a>[8] [Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts](https://arxiv.org/abs/2510.23027). arXiv 2025.
