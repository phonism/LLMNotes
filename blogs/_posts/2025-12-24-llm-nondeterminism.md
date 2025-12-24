---
layout: post
title: "LLM 推理的非确定性：根因分析与 Batch Invariance 解决方案"
date: 2025-12-24 12:00:00
author: Qi Lu
tags: [LLM, Inference, Determinism, Batch Invariance, Reproducibility, CUDA]
lang: zh
translation: /en/llm-nondeterminism/
---

## 问题定义

在 LLM 推理服务中，相同的输入理应产生相同的输出。然而实际观测表明，即使采用贪婪解码（temperature=0），输出仍存在不确定性。以下实验数据来自 Thinking Machines Lab：

| 模型 | 采样次数 | 不同输出数 | 最频繁输出出现次数 |
|------|----------|-----------|------------------|
| Qwen3-235B-A22B | 1000 | 80 | 78 |

这一现象与贪婪解码的数学定义矛盾：$\hat{y}_t = \arg\max_v p(v \mid y_{<t}, x)$ 应当是确定性的。

本文的目标是：(1) 定位非确定性的根本来源；(2) 设计可工程化部署的解决方案。

---

## 假说检验：并发浮点运算 vs. Batch Size 变化

### 假说 1：GPU 并发导致的浮点非关联性

主流假说认为，GPU 的并行计算引入了不确定性。其逻辑链条如下：

1. 浮点加法不满足结合律：$(a + b) + c \neq a + (b + c)$
2. GPU 并行 reduction 的执行顺序取决于线程调度
3. 不同的执行顺序产生不同的累加结果
4. 微小差异经 argmax 放大后改变输出 token

这一假说的典型支持证据是 CUDA 的 `atomicAdd` 操作的非确定性。

### 假说 1 的问题

Thinking Machines Lab 的分析指出，现代 Transformer 推理中的核心操作并不依赖原子操作：

- **GEMM**：使用 cuBLAS 的分块矩阵乘法，reduction 树结构固定
- **LayerNorm/RMSNorm**：标准实现使用确定性的 warp-level reduction
- **Attention**：FlashAttention 的 tiled 实现同样采用固定 reduction 顺序

实验验证：在单 GPU、固定 batch size 条件下，同一输入的多次推理输出完全一致。这排除了线程级非确定性作为主要来源。

### 假说 2：Batch Size 变化导致数值路径分歧

真正的问题在于：**kernel 的数值输出是 batch size 的函数**。

<!-- tikz-source: nondeterminism-batch-path-divergence
\begin{tikzpicture}[
    box/.style={draw, rounded corners, minimum width=2.2cm, minimum height=0.8cm, align=center, font=\small},
    kernel/.style={draw, rounded corners, fill=blue!15, minimum width=1.8cm, minimum height=0.7cm, align=center, font=\footnotesize},
    arrow/.style={->, thick, >=stealth}
]
    % 请求
    \node[box, fill=gray!20] (req) at (0, 0) {Request $x$};

    % 分支
    \node[font=\small] at (3, 1.5) {Batch size = 1};
    \node[font=\small] at (3, -1.5) {Batch size = 4};

    % 路径 1
    \node[kernel] (k1a) at (5, 1.5) {RMSNorm\\$f_1(x)$};
    \node[kernel] (k1b) at (7.5, 1.5) {MatMul\\$g_1(x)$};
    \node[kernel] (k1c) at (10, 1.5) {Attn\\$h_1(x)$};
    \node[box, fill=green!20] (out1) at (12.5, 1.5) {logits $\ell_1$};

    % 路径 2
    \node[kernel] (k2a) at (5, -1.5) {RMSNorm\\$f_4(x)$};
    \node[kernel] (k2b) at (7.5, -1.5) {MatMul\\$g_4(x)$};
    \node[kernel] (k2c) at (10, -1.5) {Attn\\$h_4(x)$};
    \node[box, fill=red!20] (out2) at (12.5, -1.5) {logits $\ell_2$};

    % 连接
    \draw[arrow] (req) -- (2, 0) -- (2, 1.5) -- (k1a);
    \draw[arrow] (req) -- (2, 0) -- (2, -1.5) -- (k2a);
    \draw[arrow] (k1a) -- (k1b);
    \draw[arrow] (k1b) -- (k1c);
    \draw[arrow] (k1c) -- (out1);
    \draw[arrow] (k2a) -- (k2b);
    \draw[arrow] (k2b) -- (k2c);
    \draw[arrow] (k2c) -- (out2);

    % 标注
    \node[font=\footnotesize, red] at (12.5, 0) {$\ell_1 \neq \ell_2$};
    \node[font=\scriptsize, gray] at (5, 0) {不同 tiling};
    \node[font=\scriptsize, gray] at (7.5, 0) {不同 reduction};
    \node[font=\scriptsize, gray] at (10, 0) {不同 split};
\end{tikzpicture}
-->
![Batch Size 变化导致数值路径分歧]({{ site.baseurl }}/assets/figures/nondeterminism-batch-path-divergence.svg)

当推理服务采用动态 batching 时，同一请求在不同时刻可能被分配到不同大小的 batch 中。每个 kernel 的 tiling 策略、reduction 分块方式都可能随 batch size 变化，导致数值结果不同。

---

## Batch Invariance 的数学定义

设 kernel $K$ 作用于输入张量 $X \in \mathbb{R}^{B \times N \times D}$，输出 $Y = K(X)$。

**Batch Invariance** 要求：对于任意 batch 中的样本 $x_i$，其输出 $y_i$ 仅取决于 $x_i$ 本身，而与 batch 中其他样本无关：

$$K(X)[i] = K'(x_i), \quad \forall i \in [1, B]$$

其中 $K'$ 是等价的单样本 kernel。

实际上，这一性质在标准 kernel 实现中并不成立。原因在于 **tiling 策略的 batch 依赖性**。

---

## 关键 Kernel 的 Batch Variance 分析

### RMSNorm

RMSNorm 计算如下：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}$$

关键步骤是 reduction：$\sum_{i=1}^{D} x_i^2$。

标准 CUDA 实现通常采用 **tree reduction**：

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

    % 标注
    \node[font=\scriptsize, gray, right] at (9, 0) {元素};
    \node[font=\scriptsize, gray, right] at (9, 1.8) {固定 reduction 顺序};
    \node[font=\scriptsize, gray, right] at (9, 3.6) {确定性结果};
\end{tikzpicture}
-->
![Tree Reduction 示意]({{ site.baseurl }}/assets/figures/nondeterminism-tree-reduction.svg)

**问题所在**：当 batch size 改变时，CUDA kernel 可能选择不同的 block size 和 grid 配置。不同配置下的 reduction 分块方式不同：

- Batch size = 1：可能使用 256 threads/block，reduction 分 4 轮完成
- Batch size = 8：可能使用 128 threads/block，reduction 分 8 轮完成

不同的分块方式产生不同的中间舍入误差，最终结果在 ULP（Unit in the Last Place）级别存在差异。

### 矩阵乘法

GEMM 的标准分块实现：$C = AB$，其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$。

cuBLAS 根据矩阵尺寸和 GPU 架构选择最优 tiling 配置。例如：

| 配置 | Tile Size | Reduction 分块 |
|------|-----------|---------------|
| 小矩阵 | 64×64 | K 维分 2 块 |
| 大矩阵 | 128×128 | K 维分 4 块 |

当 batch size 改变时，等效的矩阵尺寸 $M' = B \times M$ 发生变化，触发不同的 tiling 选择，导致 K 维 reduction 的分块方式不同。

**数学表达**：设 K 维被分为 $P$ 个块，每块大小 $k_p$：

$$C_{ij} = \sum_{p=1}^{P} \sum_{l \in \text{block}_p} A_{il} B_{lj}$$

浮点加法的顺序取决于 $P$ 和各块的边界划分。不同的 $P$ 产生不同的结果。

### Attention

FlashAttention 的核心是 **分块计算 + Online Softmax**。Softmax 的增量更新公式：

$$m^{new} = \max(m^{old}, m^{(j)})$$
$$\ell^{new} = e^{m^{old} - m^{new}} \ell^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)}$$
$$O^{new} = \frac{e^{m^{old} - m^{new}} \ell^{old} \cdot O^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)} \cdot O^{(j)}}{\ell^{new}}$$

关键参数是 **KV split size**：将 KV cache 分成多少块进行增量计算。

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

    % 不等
    \node[font=\small, red] at (9, 1) {$\neq$};

    % 标注
    \node[font=\scriptsize, gray] at (4.5, -1) {更多 rescaling 步骤 $\Rightarrow$ 更多舍入误差累积};
\end{tikzpicture}
-->
![Attention Split Size 影响]({{ site.baseurl }}/assets/figures/nondeterminism-attention-split.svg)

在 decoding 阶段，FlashAttention 根据 KV cache 长度和 GPU 配置动态选择 split size。不同的 split size 导致不同数量的 rescaling 操作，累积不同的舍入误差。

---

## 误差传播：从 ULP 到 Token 分歧

单个 kernel 的数值差异通常在 $10^{-6}$ 到 $10^{-8}$ 量级。如何导致 token 级别的分歧？

### 误差累积模型

设 Transformer 有 $L$ 层，每层的相对误差为 $\epsilon$。最终 logits 的相对误差约为：

$$\epsilon_{total} \approx L \cdot \epsilon$$

对于 $L = 80$（如 Llama-70B）、$\epsilon = 10^{-7}$：

$$\epsilon_{total} \approx 8 \times 10^{-6}$$

### Argmax 的脆弱性

当两个 token 的概率接近时，微小的 logits 差异足以翻转 argmax：

$$\Delta \ell = \ell_1 - \ell_2$$

若 $|\Delta \ell| < \epsilon_{total}$，则结果不稳定。

实际观测表明，在 greedy decoding 中，约 1-5% 的 token 位置存在这种"脆弱"状态。一旦发生分歧，后续生成完全不同，呈现 **蝴蝶效应**。

<!-- tikz-source: nondeterminism-butterfly-effect
\begin{tikzpicture}[
    token/.style={draw, rounded corners, minimum width=0.8cm, minimum height=0.5cm, align=center, font=\scriptsize},
    arrow/.style={->, thick, >=stealth}
]
    % 共同前缀
    \node[token, fill=gray!20] (t1) at (0, 0) {The};
    \node[token, fill=gray!20] (t2) at (1.2, 0) {quick};
    \node[token, fill=gray!20] (t3) at (2.4, 0) {brown};

    % 分歧点
    \node[token, fill=red!30] (t4a) at (4, 0.8) {fox};
    \node[token, fill=blue!30] (t4b) at (4, -0.8) {dog};

    % 分支 A
    \node[token, fill=red!20] (t5a) at (5.2, 0.8) {jumps};
    \node[token, fill=red!20] (t6a) at (6.4, 0.8) {over};
    \node[token, fill=red!20] (t7a) at (7.6, 0.8) {...};

    % 分支 B
    \node[token, fill=blue!20] (t5b) at (5.2, -0.8) {runs};
    \node[token, fill=blue!20] (t6b) at (6.4, -0.8) {fast};
    \node[token, fill=blue!20] (t7b) at (7.6, -0.8) {...};

    % 连接
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

    % 标注
    \node[font=\scriptsize] at (3.5, 0.3) {$\Delta \ell < 10^{-5}$};
    \node[font=\scriptsize, red] at (4, 0) {分歧点};
\end{tikzpicture}
-->
![Token 分歧的蝴蝶效应]({{ site.baseurl }}/assets/figures/nondeterminism-butterfly-effect.svg)

---

## 解决方案：Batch-Invariant Kernel 设计

### 设计原则

实现 batch invariance 的核心原则：

1. **固定 tiling 配置**：无论输入尺寸如何，使用固定的 block size 和 grid 配置
2. **固定 reduction 顺序**：确保 reduction 树结构与 batch size 无关
3. **固定 split size**：Attention 中使用固定的 KV split，不随序列长度动态调整

### RMSNorm 的 Batch-Invariant 实现

关键修改：**强制使用固定的 reduction 分块**。

```python
# 标准实现（batch-variant）
def rmsnorm_standard(x, gamma, eps):
    # reduction 分块由 CUDA runtime 决定
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma

# Batch-invariant 实现
def rmsnorm_batch_invariant(x, gamma, eps, fixed_block_size=256):
    # 强制固定分块
    D = x.shape[-1]
    num_blocks = (D + fixed_block_size - 1) // fixed_block_size

    # 分块累加，固定顺序
    sq_sum = 0.0
    for i in range(num_blocks):
        start = i * fixed_block_size
        end = min(start + fixed_block_size, D)
        sq_sum = sq_sum + torch.sum(x[..., start:end] ** 2, dim=-1, keepdim=True)

    rms = torch.sqrt(sq_sum / D + eps)
    return x / rms * gamma
```

实际的 CUDA 实现需要确保 warp-level reduction 的顺序固定，通常通过显式的 `__shfl_down_sync` 序列实现。

### MatMul 的 Batch-Invariant 实现

cuBLAS 的自动 tuning 会根据矩阵尺寸选择不同 kernel。解决方案：

1. **禁用自动 tuning**：强制使用固定的 GEMM kernel
2. **Padding 到固定尺寸**：将矩阵 pad 到 $2^n$ 大小，确保相同的 tiling

```python
# 使用 torch.Library 替换默认 kernel
import batch_invariant_ops

# 自动替换所有 matmul 为 batch-invariant 版本
# 内部实现固定 tile size = 128x128
```

### Attention 的 Batch-Invariant 实现

关键：**固定 KV split size**。

```python
# FlashAttention batch-invariant 配置
flash_attn_func(
    q, k, v,
    deterministic=True,      # 启用确定性模式
    fixed_split_kv=64        # 固定 KV split 为 64
)
```

SGLang 的实现支持多种 attention 后端的 batch-invariant 模式：

| 后端 | 固定 Split Size | 性能开销 |
|------|----------------|----------|
| FlashInfer | 64 | ~30% |
| FlashAttention 3 | 128 | ~25% |
| Triton | 64 | ~40% |

---

## 多 GPU 场景：AllReduce 的确定性

在 Tensor Parallelism 中，AllReduce 操作同样引入非确定性。NCCL 的默认实现使用 ring-based 或 tree-based 算法，reduction 顺序取决于 GPU 间通信延迟。

### 确定性 AllReduce

解决方案：使用固定顺序的 reduce-scatter + all-gather：

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

    % 数据
    \node[data] (d0) at (0, -1) {};
    \node[data] (d1) at (3, -1) {};
    \node[data] (d2) at (6, -1) {};
    \node[data] (d3) at (9, -1) {};

    % 固定顺序 reduce
    \node[font=\scriptsize] at (4.5, -2) {固定顺序: GPU 0 $\rightarrow$ 1 $\rightarrow$ 2 $\rightarrow$ 3};

    \draw[arrow, red] (d0) -- (1.5, -1) -- (1.5, -1.5) -- (d1);
    \draw[arrow, red] (d1) -- (4.5, -1) -- (4.5, -1.5) -- (d2);
    \draw[arrow, red] (d2) -- (7.5, -1) -- (7.5, -1.5) -- (d3);

    % 标注
    \node[font=\scriptsize, gray] at (4.5, -3) {确定性 reduction: $((d_0 + d_1) + d_2) + d_3$};
\end{tikzpicture}
-->
![确定性 AllReduce]({{ site.baseurl }}/assets/figures/nondeterminism-allreduce.svg)

SGLang 实现了 deterministic tensor parallelism，确保多 GPU 场景下的完全可复现。

---

## MoE 模型的额外挑战

Mixture-of-Experts（MoE）架构引入了 Dense 模型不存在的非确定性来源。以 Qwen3-235B-A22B、DeepSeek-V3 等模型为代表的 MoE 架构，其稀疏激活特性使得确定性推理面临额外挑战。

### Token Routing 的非确定性

MoE 的核心是 **门控网络**（Gating Network），决定每个 token 被路由到哪些 expert：

$$G(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$$

问题在于：当多个 expert 的门控分数接近时，微小的数值扰动可能改变 TopK 的选择结果。

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

    % 连接
    \draw[arrow] (t1) -- (gate);
    \draw[arrow] (t2) -- (gate);
    \draw[arrow] (t3) -- (gate);

    % 路由（实线 = 稳定，虚线 = 不稳定）
    \draw[arrow, thick, green!60!black] (gate) -- node[above, font=\scriptsize] {0.45} (e1);
    \draw[arrow, thick, green!60!black] (gate) -- node[above, font=\scriptsize] {0.35} (e2);
    \draw[arrow, dashed, red] (gate) -- node[below, font=\scriptsize] {0.20} (e3);

    % 标注
    \node[font=\scriptsize, red] at (5, -3) {当 $|s_2 - s_3| < \epsilon$，路由决策不稳定};
\end{tikzpicture}
-->
![MoE Token Routing 示意]({{ site.baseurl }}/assets/figures/nondeterminism-moe-routing.svg)

### Expert Capacity 与 Token Dropping

为保证计算效率，MoE 通常设置 **Expert Capacity**：每个 expert 能处理的最大 token 数。当某个 expert 接收的 token 超过容量时，多余的 token 被 **丢弃**（dropped），直接通过残差连接传递到下一层。

$$\text{Expert Capacity} = \frac{\text{Batch Tokens} \times \text{Capacity Factor}}{\text{Num Experts}}$$

Token dropping 的非确定性来源：

1. **Batch 组成变化**：不同 batch 中的 token 分布不同，同一 token 可能在某些 batch 中被处理，在另一些中被丢弃
2. **路由顺序依赖**：当 capacity 接近饱和时，先到达的 token 被处理，后到达的被丢弃
3. **负载不均衡**：热门 expert 更容易触发 dropping

### 训练-推理不一致

研究发现，MoE 模型在训练和推理阶段的路由行为存在显著差异：

> "Even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes."

这种不一致在 RL 训练中尤为严重：

- **On-policy 要求**：RL 训练要求 rollout 与训练使用相同的策略
- **路由漂移**：推理时的路由决策可能偏离训练时的分布
- **奖励噪声**：非确定性路由引入难以追踪的奖励波动

### MoE 确定性推理的解决方案

#### 1. 固定路由阈值

避免在门控分数接近时产生不稳定决策：

```python
def deterministic_topk(scores, k, margin=1e-5):
    # 当分数差异小于 margin 时，使用固定的 tie-breaking 规则
    sorted_scores, indices = torch.sort(scores, descending=True)

    # 检测 tie 情况
    for i in range(k-1, len(sorted_scores)-1):
        if sorted_scores[i] - sorted_scores[i+1] < margin:
            # 使用 expert index 作为 tie-breaker
            pass

    return indices[:k]
```

#### 2. 取消 Token Dropping

以计算效率换取确定性：

```python
# 设置足够大的 capacity factor，确保不发生 dropping
config.moe_capacity_factor = 2.0  # 或更高
config.moe_drop_tokens = False
```

#### 3. Soft MoE

使用软路由替代硬路由，每个 token 以加权方式分配给所有 expert：

$$y = \sum_{i=1}^{E} g_i(x) \cdot \text{Expert}_i(x)$$

Soft MoE 消除了离散的路由决策，但计算开销更高。

### 对 RL 训练的特殊影响

MoE 模型的 RL 训练面临双重挑战：

1. **数值非确定性**：前述的 batch variance 问题
2. **结构非确定性**：路由决策的不稳定性

研究表明，MoE RL 训练的不稳定性部分源于路由分布的漂移：

> "The routing distribution has been identified as a pivotal factor contributing to the instability of MoE RL."

实验数据显示，约 10% 的 router 在训练和推理间选择不同的 expert，94% 的 token 至少在一层选择了不同的 expert。

### Routing Replay：R2 与 R3

为解决训练-推理路由不一致问题，研究者提出了 **Routing Replay** 机制，核心思想是在训练时重放推理阶段的路由决策。

#### R2：Vanilla Routing Replay

**定义**：重用 **训练系统** 在 rollout 阶段选择的 expert。

```
Rollout (Training System) → 记录路由决策 → Training 时重放
```

- **优点**：实现简单，开销较小
- **缺点**：当 off-policy 程度较大时效果下降

#### R3：Rollout Routing Replay

**定义**：重用 **推理系统** 在 rollout 阶段选择的 expert。

```
Rollout (Inference System) → 记录路由决策 → Training 时重放
```

R3 强制约束：在 rollout 阶段激活的特定 expert 必须在训练反向传播时严格重用。

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

    \draw[arrow] (r2-train) -- node[above, font=\scriptsize] {生成} (r2-rollout);
    \draw[arrow] (r2-rollout) -- node[right, font=\scriptsize] {重放路由} (r2-bp);
    \draw[dasharrow, gray] (r2-train) -- (r2-bp);

    % R3
    \node[font=\bfseries] at (8, 3) {R3: Rollout Routing Replay};
    \node[box, fill=purple!20] (r3-infer) at (6, 2) {Inference\\System};
    \node[box, fill=green!20] (r3-rollout) at (10, 2) {Rollout};
    \node[box, fill=orange!20] (r3-bp) at (10, 0.5) {Backprop};

    \draw[arrow] (r3-infer) -- node[above, font=\scriptsize] {生成} (r3-rollout);
    \draw[arrow] (r3-rollout) -- node[right, font=\scriptsize] {重放路由} (r3-bp);

    % 标注
    \node[font=\scriptsize, gray] at (2, -0.5) {训练系统的路由决策};
    \node[font=\scriptsize, gray] at (10, -0.5) {推理系统的路由决策};
\end{tikzpicture}
-->
![R2 与 R3 对比]({{ site.baseurl }}/assets/figures/nondeterminism-r2-r3.svg)

#### 选择 R2 还是 R3？

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| Off-policy 程度小 | R2 | R3 对目标策略的改变代价大于收益 |
| Off-policy 程度大 | R3 | 保持一阶近似的有效性更重要 |

#### 实现细节

R3 可与 KV Cache 集成：

```python
# 对于每层每个 token prefix，存储对应的路由 mask
routing_cache = {}

def forward_with_replay(x, layer_idx, prefix_hash):
    if prefix_hash in routing_cache:
        # 命中缓存，重用路由决策
        mask = routing_cache[prefix_hash]
    else:
        # 计算新路由
        mask = compute_routing(x)
        routing_cache[prefix_hash] = mask

    # 使用 mask 进行 softmax gating，但不阻断梯度流
    return apply_routing(x, mask, allow_grad=True)
```

关键点：重放的 mask 用于 softmax gating 计算，但不阻止梯度流经标准 router 权重，确保 router 仍可训练。

### 其他解决方案

#### RSPO（Router-Shift Policy Optimization）

不同于 R2/R3 的硬约束，RSPO 采用软调整机制：

$$w_{rspo} = w_{is} \cdot \text{clip}(\text{router\_shift\_ratio}, 1-\epsilon, 1+\epsilon)$$

- 计算当前策略与旧策略间的 **router shift ratio**
- 量化每个 token 的路由偏移程度
- 自适应重加权更新，限制过大更新同时保持 router 灵活性

#### 冻结 Router

最简单的方案是冻结 router 参数，但会损害模型适应性，通常不推荐。

---

## 性能分析

Batch-invariant kernels 的主要开销来源：

1. **放弃动态优化**：无法使用针对特定尺寸的最优 kernel
2. **固定 tiling 的低效**：小矩阵使用大 tile 造成资源浪费
3. **额外的同步开销**：确保固定执行顺序需要更多同步点

### 基准测试

| 方案 | 吞吐量 (tokens/s) | 相对开销 |
|------|------------------|---------|
| 标准 vLLM | 1000 | baseline |
| TML batch_invariant_ops | 385 | 61.5% |
| SGLang + CUDA Graphs | 657 | 34.3% |

CUDA Graphs 通过预编译执行图，消除了 kernel launch 开销，显著减少了确定性模式的性能损失。

### 延迟分布

确定性模式的另一个优势是 **延迟方差降低**：

```
标准模式:   P50=45ms, P99=120ms, stddev=25ms
确定性模式: P50=52ms, P99=58ms,  stddev=3ms
```

对于延迟敏感的应用，更低的方差可能比更低的均值更重要。

---

## 应用场景

### RL 训练的可复现性

在 RLHF/RLVR 训练中，策略的 rollout 需要与训练步骤精确对应。非确定性推理导致：

1. **调试困难**：无法复现特定的失败案例
2. **训练不稳定**：相同 checkpoint 在不同机器上表现不一致
3. **实验不可比较**：无法区分算法改进和随机波动

SGLang 与 slime 的合作实现了 100% 可复现的 RL 训练：

> "Taking this deterministic inference capability further, SGLang collaborated with the slime team to unlock 100% reproducible RL training."

### 安全与合规

对于需要审计的 AI 系统，确定性行为是基本要求：

- 医疗诊断
- 金融决策
- 法律分析

非确定性意味着无法追溯和解释特定输出的来源。

### 模型调试

当模型出现异常输出时，确定性推理允许：

1. 精确复现问题
2. 逐层检查中间状态
3. 二分定位问题来源

---

## 相关工作

### LayerCast（NeurIPS 2025）

Yuan et al. 提出了另一种思路：通过提高数值精度减少误差累积。

- **方法**：权重以 FP16/BF16 存储，计算时转换为 FP32
- **优势**：不修改 kernel 实现
- **局限**：无法完全消除 batch variance，只是降低其影响

实验显示，LayerCast 将 DeepSeek-R1-Distill-Qwen-7B 的准确率波动从 9% 降低到 2%，但仍非完全确定。

### OpenAI seed 参数

OpenAI API 提供 `seed` 参数以提高可复现性：

> "If specified, our system will make a best effort to sample deterministically..."

但官方明确表示 **不保证确定性**，原因包括：

1. 后端模型更新
2. 负载均衡到不同硬件
3. 系统配置变化

`system_fingerprint` 字段用于追踪配置变化，但无法保证相同 fingerprint 下的完全确定性。

---

## 总结

LLM 推理的非确定性源于 **kernel 实现对 batch size 的敏感性**，而非 GPU 的并发特性。通过设计 batch-invariant kernels，可以在工程层面实现完全确定的推理。

**Dense 模型的关键技术点**：

1. 固定 RMSNorm 的 reduction 分块
2. 固定 MatMul 的 tiling 配置
3. 固定 Attention 的 KV split size
4. 多 GPU 场景使用确定性 AllReduce

**MoE 模型的额外挑战**：

1. Token routing 的离散决策在门控分数接近时不稳定
2. Expert capacity 和 token dropping 引入 batch 依赖的非确定性
3. 训练-推理路由不一致需要 Routing Replay（R2/R3）机制

性能代价约 30-60%，但在 RL 训练、模型调试、安全审计等场景中是必要的投入。

---

## 参考文献

1. Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). 2025.

2. Yuan, J. et al. [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501). NeurIPS 2025 (Oral).

3. LMSYS Org. [Towards Deterministic Inference in SGLang and Reproducible RL Training](https://lmsys.org/blog/2025-09-22-sglang-deterministic/). 2025.

4. Thinking Machines Lab. [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops). GitHub.

5. vLLM Documentation. [Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/).

6. Dao, T. et al. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). 2023.

7. Ma, W. et al. [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370). 2025.

8. [Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts](https://arxiv.org/abs/2510.23027). arXiv 2025.
