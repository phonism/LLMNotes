---
layout: post
title: "Transformer 学习笔记（三）：注意力机制"
date: 2025-12-20 10:20:00
author: Qi Lu
tags: [Transformer, Attention]
lang: zh
translation: /en/transformer-part3-attention/
series: transformer
series_order: 3
---

注意力机制是 Transformer 的核心，但标准的 $O(N^2)$ 复杂度成为长上下文建模的瓶颈。本文系统介绍四种优化路径：FlashAttention 通过 IO 感知优化加速计算；MLA 通过低秩压缩减少 KV Cache；稀疏注意力只计算重要的 token 对；线性注意力彻底改变计算形式。

## 1. FlashAttention：IO 感知的高效注意力

Roofline 模型揭示了一个关键洞察：现代 GPU 的算力远超内存带宽，决定性能的往往不是"能算多快"，而是"数据能多快送达"。标准注意力机制正是这一瓶颈的典型案例——它需要将 $O(N^2)$ 规模的中间结果反复写入显存，而这些 IO 操作成为真正的性能杀手。

FlashAttention 的核心思想是重新组织计算顺序，使注意力矩阵**永远不离开 GPU 的高速缓存**。这不是数学上的近似，而是精确等价的重新排列——我们用更多的计算换取更少的内存访问，而现代 GPU 恰好计算过剩、带宽稀缺。

### 1.1 动机：GPU 内存层次结构

现代 GPU 的计算能力远超内存带宽。以 NVIDIA A100 为例：
- **计算能力**：312 TFLOPS（FP16 Tensor Core）
- **HBM 带宽**：2 TB/s
- **SRAM 容量**：20 MB（共享内存 + L1 缓存）
- **SRAM 带宽**：约 19 TB/s

**算术强度**（Arithmetic Intensity）定义为每字节内存访问的 FLOPs：

$$\text{算术强度} = \frac{\text{FLOPs}}{\text{内存访问字节数}}$$

对于 A100，达到峰值计算需要算术强度 $\geq 156$ FLOPs/Byte。标准注意力的算术强度远低于此，因此是**内存受限**（Memory-Bound）的。

**标准注意力的 IO 问题**：

标准注意力实现需要多次 HBM 读写：
1. 从 HBM 读取 $Q, K$，计算 $S = QK^\top$，写回 HBM
2. 从 HBM 读取 $S$，计算 $P = \text{softmax}(S)$，写回 HBM
3. 从 HBM 读取 $P, V$，计算 $O = PV$，写回 HBM

总 HBM 访问量：$O(N^2 + Nd)$，其中 $N$ 是序列长度，$d$ 是头维度。对于长序列，$N^2$ 项主导，造成严重的 IO 瓶颈。

### 1.2 FlashAttention v1：永不落地的注意力矩阵

FlashAttention 的核心思想：**永远不将完整的 $N \times N$ 注意力矩阵写入 HBM**。

#### Online Softmax 算法

Softmax 的标准计算需要两次遍历：

$$m = \max_j(x_j), \quad \ell = \sum_j e^{x_j - m}, \quad \text{softmax}(x)_i = \frac{e^{x_i - m}}{\ell}$$

**Online Softmax** 允许单次遍历、增量计算。对于两个块 $x^{(1)}, x^{(2)}$ 的拼接：

$$m^{(1)} = \max(x^{(1)}), \quad \ell^{(1)} = \sum_j e^{x_j^{(1)} - m^{(1)}}$$
$$m^{(2)} = \max(x^{(2)}), \quad \ell^{(2)} = \sum_j e^{x_j^{(2)} - m^{(2)}}$$
$$m^{new} = \max(m^{(1)}, m^{(2)})$$
$$\ell^{new} = e^{m^{(1)} - m^{new}} \ell^{(1)} + e^{m^{(2)} - m^{new}} \ell^{(2)}$$

输出也可以增量更新：

$$O^{new} = \frac{1}{\ell^{new}} \left[ e^{m^{old} - m^{new}} \ell^{old} \cdot O^{old} + e^{m^{(j)} - m^{new}} \ell^{(j)} \cdot O^{(j)} \right]$$

#### Tiling 算法

FlashAttention 将 $Q, K, V$ 分成大小为 $B_r \times d$ 和 $B_c \times d$ 的块：
- $B_c = \lceil M / (4d) \rceil$（SRAM 大小 $M$ 约束）
- $B_r = \min(B_c, d)$

**FlashAttention 前向传播算法**：

<!-- tikz-source: transformer-flashattention-algorithm
\begin{algorithm}[H]
\caption{FlashAttention 前向传播}
\KwInput{$Q, K, V \in \mathbb{R}^{N \times d}$，块大小 $B_r, B_c$}
\KwOutput{$O \in \mathbb{R}^{N \times d}$}
初始化 $O = 0$, $\ell = 0$, $m = -\infty$（均为 $N$ 维向量）\;
\For{$j = 1$ \KwTo $\lceil N/B_c \rceil$}{
    从 HBM 加载 $K_j, V_j \in \mathbb{R}^{B_c \times d}$ 到 SRAM\;
    \For{$i = 1$ \KwTo $\lceil N/B_r \rceil$}{
        从 HBM 加载 $Q_i, O_i, \ell_i, m_i$ 到 SRAM\;
        在 SRAM 中计算 $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$\;
        计算 $m_{ij} = \text{rowmax}(S_{ij})$, $\tilde{P}_{ij} = \exp(S_{ij} - m_{ij})$\;
        计算 $\ell_{ij} = \text{rowsum}(\tilde{P}_{ij})$\;
        更新 $m_i^{\text{new}}, \ell_i^{\text{new}}$（Online Softmax 更新）\;
        更新 $O_i = \text{diag}(\ell_i^{\text{new}})^{-1}(\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}O_i + e^{m_{ij} - m_i^{\text{new}}}\tilde{P}_{ij} V_j)$\;
        将 $O_i, \ell_i^{\text{new}}, m_i^{\text{new}}$ 写回 HBM\;
    }
}
\Return{$O$}
\end{algorithm}
-->
![FlashAttention 前向传播算法]({{ site.baseurl }}/assets/figures/transformer-flashattention-algorithm.svg)

#### 反向传播与重计算

标准反向传播需要存储 $S, P \in \mathbb{R}^{N \times N}$。FlashAttention 采用**重计算**（Recomputation）策略：
- 前向传播只存储 $O$ 和统计量 $(m, \ell)$
- 反向传播时从 $Q, K, V$ 块重新计算 $S, P$

虽然增加了 FLOPs（约多 30%），但大幅减少 HBM 访问，总体速度更快。

#### IO 复杂度分析

**定理（FlashAttention IO 复杂度）**：设 SRAM 大小为 $M$，序列长度为 $N$，头维度为 $d$。FlashAttention 的 HBM 访问量为：

$$O\left( \frac{N^2 d^2}{M} \right)$$

而标准注意力的 HBM 访问量为 $O(N^2 + Nd)$。当 $M = \Theta(Nd)$ 时，FlashAttention 减少 $O(N)$ 倍 HBM 访问。

| 场景 | 加速比 | 内存节省 |
|------|--------|----------|
| BERT-large (seq=512) | 1.15× | 5× |
| GPT-2 (seq=1K) | 3× | 10× |
| Long-range (seq=4K) | 2.4× | 20× |

### 1.3 FlashAttention-2：并行策略优化

FlashAttention-2 通过优化并行策略和工作分配，在 A100 上达到 230 TFLOPS（约 73% 峰值利用率），比 v1 快约 2 倍。

#### 并行策略改进

**FlashAttention v1**：在 batch 和 head 维度并行，每个 thread block 处理一个 attention head。当 $\text{batch} \times \text{heads} < 108$（A100 SM 数量）时，GPU 利用率低。

**FlashAttention-2**：额外在**序列长度维度**并行。对于长序列（通常意味着小 batch），这显著提高 GPU 利用率。

#### Warp 工作分配

GPU 的线程层次：Thread → Warp（32 线程）→ Thread Block → Grid。

**v1 的 Sliced-K 方案**：将 $K, V$ 在 4 个 warp 间分割，$Q$ 对所有 warp 可见。问题：需要 warp 间同步和共享内存中间结果。

**v2 的 Sliced-Q 方案**：将 $Q$ 在 4 个 warp 间分割，$K, V$ 对所有 warp 可见。优势：消除 warp 间通信，减少共享内存读写。

#### 减少非矩阵乘法 FLOPs

A100 的矩阵乘法吞吐量是非矩阵乘法的 **16 倍**（312 vs 19.5 TFLOPS）。v2 通过以下方式减少非 matmul 操作：
- 优化 Online Softmax 的 rescaling 操作
- 改进边界检查和因果 mask 的实现

| 指标 | v1 | v2 | 提升 |
|------|----|----|------|
| A100 峰值 (TFLOPS) | 124 | 230 | 1.85× |
| GPU 利用率 | 25-40% | 50-73% | 约 2× |
| GPT-3 训练 (TFLOPS) | 173 | 225 | 1.3× |

### 1.4 FlashAttention-3：Hopper 架构优化

FlashAttention-3 针对 NVIDIA Hopper 架构（H100）设计，充分利用新硬件特性，达到 740 TFLOPS（75% 峰值利用率）。

#### Hopper 新硬件特性

**WGMMA（Warpgroup Matrix Multiply-Accumulate）**：Hopper 引入的新 Tensor Core 指令，吞吐量显著高于 Ampere 的 `mma.sync`。一个 warpgroup（4 个 warp，128 线程）可以执行大规模矩阵乘法。

**TMA（Tensor Memory Accelerator）**：专用硬件单元，负责 Global Memory 和 Shared Memory 之间的数据传输：
- 自动处理索引计算和边界检查
- 释放寄存器资源，允许更大的 tile size
- 支持异步传输，与计算重叠

#### 三大优化技术

**1. Warp Specialization（Warp 专门化）**：将 warp 分为 **Producer** 和 **Consumer**：
- Producer warp：负责 TMA 数据传输
- Consumer warp：负责 WGMMA 计算

数据传输和计算完全异步重叠。

**2. Ping-Pong Scheduling**：在两个 warpgroup 之间交替执行：
- Warpgroup 1 执行 GEMM 时，Warpgroup 2 执行 Softmax
- 然后角色互换

这种调度将 FP16 前向传播从约 570 TFLOPS 提升到 620 TFLOPS。

**3. Intra-warpgroup Overlapping**：在单个 warpgroup 内，Softmax 计算与 GEMM 流水线化：
- 当 GEMM 计算当前块时，同时对上一块执行 Softmax
- 进一步提升到 640-660 TFLOPS

#### FP8 支持

FlashAttention-3 支持 FP8 低精度，通过 **Block Quantization** 和 **Incoherent Processing**（基于 Hadamard 变换）减少量化误差：
- FP8 吞吐量：接近 1.2 PFLOPS
- 量化误差比基线 FP8 注意力低 2.6 倍

### 1.5 FlashAttention-4：Blackwell 架构优化

FlashAttention-4 针对 NVIDIA Blackwell 架构（B200）设计，是首个突破 PFLOPS 屏障的注意力内核。

#### 五阶段 Warp 流水线

从 v3 的 2 阶段扩展到 **5 阶段流水线**，每种 warp 高度专门化：
1. **Load Warp**：通过 TMA 从 Global Memory 加载 $Q, K, V$ 到 Shared Memory
2. **MMA Warp**：执行矩阵乘法，计算注意力分数和输出累加
3. **Softmax Warps**（8 个）：计算归一化注意力分数，维护 running statistics
4. **Correction Warps**（4 个）：当 scaling factor 变化时重新缩放输出
5. **Epilogue Warps**：将完成的输出块写回 Global Memory

#### 软件 exp2 模拟

传统实现依赖 Special Function Units（SFU）计算指数函数，但 SFU 是稀缺资源。FlashAttention-4 使用**三次多项式近似**：

$$2^x \approx a_0 + a_1 x + a_2 x^2 + a_3 x^3, \quad x \in [0, 1)$$

通过 Horner 方法高效计算，在 CUDA Core 上使用向量化 FMA 指令，避免 SFU 瓶颈。

#### 选择性重缩放

传统 Online Softmax 在每次遇到新最大值时都重新缩放。FlashAttention-4 引入**阈值判断**：只有当最大值变化足以影响数值稳定性时才触发重缩放。据报告，这将重缩放次数减少约 10 倍，同时保持数值精度。

**性能**：
- 比 cuDNN 注意力快约 20%
- 比 FlashAttention-3 快约 2 倍
- 比原始 FlashAttention 快约 15 倍

### 1.6 Flash Decoding：推理时的 KV 并行

FlashAttention 针对训练优化，但在**推理**时存在问题。Flash Decoding 专门解决推理瓶颈。

#### 推理时的问题

自回归生成时，每步只生成 1 个 token，即 $Q$ 的序列长度为 1：
- FlashAttention 在 batch 和 head 维度并行
- 当 $\text{batch} \times \text{heads} < 108$ 时，GPU 严重 underutilized
- 长上下文场景下（batch size 小），**FlashAttention 可能只用到 GPU 的 1%**

#### KV 序列长度并行

Flash Decoding 的核心思想：在 **KV 序列长度**维度并行。

1. 将 KV Cache 分成 $S$ 个块：$K = [K_1, ..., K_S]$, $V = [V_1, ..., V_S]$
2. 对每个块独立计算部分注意力：
   $$O_s = \text{softmax}(Q K_s^\top) V_s, \quad (m_s, \ell_s) = \text{统计量}$$
3. 使用 Log-Sum-Exp 合并结果：
   $$O = \frac{\sum_s e^{m_s - m_{global}} \ell_s \cdot O_s}{\sum_s e^{m_s - m_{global}} \ell_s}$$

**性能提升**：
- 长序列解码加速高达 **8 倍**
- 在 CodeLLaMa-34B 上，注意力操作比 FlashAttention 快 **50 倍**
- 序列长度从 512 增加到 64K，生成速度几乎不变

### 1.7 FlashDecoding++

FlashDecoding++ 进一步优化，在 MLSys 2024 发表。

**异步 Softmax**：Flash Decoding 的 Reduction 步骤需要同步等待所有部分结果。FlashDecoding++ 引入 **Unified Max Value**：
- 预估一个全局最大值 $m_{unified}$（基于统计或启发式）
- 所有块使用相同的 $m_{unified}$，无需同步
- 细粒度流水线，Prefill 加速 1.05×，Decoding 加速 1.14×

**Flat GEMM 优化**：推理时的 GEMM 形状是"扁平"的（$1 \times N$），标准实现效率低：
- cuBLAS/CUTLASS 对这种形状有高达 50% 的性能损失
- FlashDecoding++ 使用 Double Buffering 和针对性优化
- Flat GEMM 加速高达 52%

### 1.8 FlashAttention 版本演进

| 版本 | GPU | 峰值 TFLOPS | 利用率 | 相对 v1 加速 |
|------|-----|-------------|--------|--------------|
| v1 | A100 | 124 | 40% | 1× |
| v2 | A100 | 230 | 73% | 1.85× |
| v2 | H100 | 335 | 35% | 2.7× |
| v3 | H100 | 740 | 75% | 6× |
| v3 (FP8) | H100 | 1200 | - | 9.7× |
| v4 | B200 | PFLOPS | - | 15× |

### 1.9 FlashAttention 的工程影响

FlashAttention 已成为现代 LLM 训练和推理的**标配**：
- **PyTorch 2.0+**：内置 `scaled_dot_product_attention` 使用 FlashAttention
- **vLLM、TensorRT-LLM**：推理引擎默认使用
- **所有主流 LLM**：GPT-4、Claude、LLaMA、DeepSeek 等都使用 FlashAttention

**上下文长度革命**：FlashAttention 将实用上下文长度从 2-4K 提升到 128K+：
- 内存从 $O(N^2)$ 降到 $O(N)$
- 64K 序列在标准注意力下需要 16GB 显存，FlashAttention 只需约 1GB

> **何时使用 FlashAttention**：FlashAttention 在以下场景收益最大：
> - **长序列**：序列长度 > 512
> - **大 batch**：充分利用 GPU 并行
> - **训练**：内存节省允许更大 batch
>
> 短序列、小模型场景下，标准注意力可能更快（减少 kernel launch 开销）。

### 1.10 FlexAttention：可编程的 FlashAttention

FlexAttention 是 PyTorch 2.5 引入的新 API，提供 FlashAttention 的灵活编程接口。

**动机**：FlashAttention 虽然高效，但每种注意力变体（Causal、ALiBi、Sliding Window 等）都需要专门实现。研究者想试验新变体时，往往需要手写 Triton kernel。FlexAttention 通过 `torch.compile` 自动生成高效 kernel，将开发时间从数周缩短到数分钟。

**核心 API**：FlexAttention 提供两个函数式接口：
- **score_mod**：修改 $QK^\top$ 后的分数矩阵（如添加位置偏置）
- **mask_mod**：定义 mask 模式（返回 True 的位置参与计算）

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Causal mask
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# ALiBi 位置编码
def alibi(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx) * slope[h]

# 使用
block_mask = create_block_mask(causal, B, H, Q_LEN, KV_LEN)
out = flex_attention(q, k, v, score_mod=alibi, block_mask=block_mask)
```

**性能**：FlexAttention 达到 FlashAttention-2 约 **85-90%** 的性能，但开发效率提升 100 倍。对于 FlashAttention 不原生支持的变体（如 Document Masking），FlexAttention 比标准 SDPA 快 5-8 倍。

## 2. Multi-head Latent Attention (MLA)

FlashAttention 解决了训练时注意力矩阵的 IO 瓶颈，但推理时还存在另一个内存挑战：**KV Cache**。自回归生成需要缓存所有历史 token 的 Key 和 Value 向量，随着上下文长度增加，这部分内存占用可能超过模型参数本身。

Multi-head Latent Attention（MLA）是 DeepSeek-V2 针对这一问题提出的解决方案。其核心洞察是：虽然每个注意力头需要独立的 K 和 V，但它们可能存在**低秩结构**——即可以从一个共享的低维"潜在向量"中恢复。这种压缩不同于 GQA/MQA 的强制共享，而是让网络学习最优的压缩方式。

### 2.1 KV Cache 挑战

自回归生成时，KV Cache 成为长上下文推理的主要内存瓶颈：

$$\text{KV Cache Size} = 2 \times B \times S \times L \times n_h \times d_h \times \text{bytes}$$

对于大模型（如 $n_h = 128$，$d_h = 128$），KV Cache 成为长上下文推理的主要内存瓶颈。对于 70B 模型（$L=80$, $K=8$, $S=8192$），KV Cache 达到 **2.1 GB/request**。

### 2.2 现有方案的局限

| Method | KV Cache Size | Performance | 原理 |
|--------|---------------|-------------|------|
| MHA | $2 n_h d_h$ | 最优 | 每头独立 KV |
| GQA | $2 \frac{n_h}{g} d_h$ | 轻微下降 | $g$ 个 Q 头共享 KV |
| MQA | $2 d_h$ | 明显下降 | 所有头共享 KV |
| **MLA** | $d_c + d_h^R$ | **接近 MHA** | 低秩压缩 |

GQA/MQA 通过**强制共享** KV 头来减少缓存，但这种强制共享往往损害模型性能。MLA 的核心洞察：**KV 可以从一个低维潜在向量中恢复**，而不必显式共享。

### 2.3 MLA 核心原理

MLA 的核心思想是将高维的 Key 和 Value 压缩到一个共享的低维潜在向量（latent vector），推理时从该向量恢复 K 和 V。

#### KV 的低秩压缩

对于输入 $\mathbf{h}_t \in \mathbb{R}^d$，首先压缩到潜在向量：

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

其中 $W^{DKV} \in \mathbb{R}^{d_c \times d}$ 是下投影矩阵，$d_c \ll n_h d_h$ 是压缩维度。

从潜在向量恢复 K 和 V：

$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$

其中 $W^{UK}, W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}$ 是上投影矩阵。

#### Query 的低秩压缩

类似地，Query 也可以进行低秩压缩（主要用于减少训练时的激活内存）：

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t, \quad \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q$$

其中 $W^{DQ} \in \mathbb{R}^{d_c' \times d}$，$W^{UQ} \in \mathbb{R}^{n_h d_h \times d_c'}$。

#### Decoupled RoPE

RoPE 需要在每个位置应用旋转，但如果直接对压缩后的 $\mathbf{c}_t^{KV}$ 应用 RoPE，会破坏后续的权重吸收优化。MLA 采用**解耦 RoPE**（Decoupled RoPE）策略：

1. 将每个注意力头分为两部分：
   - **内容部分**（$d_h^C$ 维）：从压缩向量恢复，**不**应用 RoPE
   - **位置部分**（$d_h^R$ 维）：额外投影，应用 RoPE

2. 最终的 Q 和 K 为两部分的拼接：

$$\mathbf{q}_t = [\mathbf{q}_t^C; \text{RoPE}(\mathbf{q}_t^R, t)]$$
$$\mathbf{k}_t = [\mathbf{k}_t^C; \text{RoPE}(\mathbf{k}_t^R, t)]$$

其中位置部分的计算为：

$$\mathbf{q}_t^R = W^{QR} \mathbf{c}_t^Q, \quad \mathbf{k}_t^R = W^{KR} \mathbf{h}_t$$

> **为什么需要 Decoupled RoPE？** RoPE 是位置相关的：$\text{RoPE}(\mathbf{x}, t)$ 依赖于位置 $t$。如果对 $\mathbf{c}_t^{KV}$ 应用 RoPE 再恢复 K，则 $\mathbf{k}_t = W^{UK} \cdot \text{RoPE}(\mathbf{c}_t^{KV}, t)$，此时 $W^{UK}$ 无法被吸收到 $W^Q$ 中（因为 RoPE 在中间）。Decoupled 策略将 RoPE 隔离到单独的维度，保留了权重吸收的可能性。

#### 权重吸收

MLA 的一个关键优化是**权重吸收**（Weight Absorption）。由于压缩和恢复之间没有非线性激活，矩阵可以合并：

**Query-Key 吸收**：注意力分数的计算：

$$\mathbf{q}_t^{C\top} \mathbf{k}_s^C = (\mathbf{c}_t^Q)^\top (W^{UQ})^\top W^{UK} \mathbf{c}_s^{KV} = (\mathbf{c}_t^Q)^\top \underbrace{W^{QK}}_{\text{absorbed}} \mathbf{c}_s^{KV}$$

其中 $W^{QK} = (W^{UQ})^\top W^{UK} \in \mathbb{R}^{d_c' \times d_c}$。

**Output-Value 吸收**：输出投影的计算：

$$W^O \mathbf{v}_t^C = W^O W^{UV} \mathbf{c}_t^{KV} = \underbrace{W^{OV}}_{\text{absorbed}} \mathbf{c}_t^{KV}$$

其中 $W^{OV} = W^O W^{UV} \in \mathbb{R}^{d \times d_c}$。

**推理流程**：权重吸收后，推理时：
1. 缓存 $\mathbf{c}_t^{KV}$ 和 $\mathbf{k}_t^R$（位置部分）
2. 用 $W^{QK}$ 直接计算内容部分的注意力分数
3. 用吸收后的 $W^{OV}$ 计算输出

### 2.4 KV Cache 压缩效果

| Method | Cache Elements | DeepSeek-V2 (具体值) |
|--------|----------------|---------------------|
| MHA | $2 n_h d_h$ | $2 \times 128 \times 128 = 32768$ |
| GQA (8 组) | $2 \times 8 \times d_h$ | $2 \times 8 \times 128 = 2048$ |
| **MLA** | $d_c + d_h^R$ | $512 + 64 = 576$ |

DeepSeek-V2 配置：$d_c = 512$，$d_h^R = 64$，压缩比达到 $\frac{32768}{576} \approx \mathbf{56.9×}$。

### 2.5 PyTorch 实现

以下是 MLA 的简化 PyTorch 实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,      # 模型维度
        n_heads: int,      # 注意力头数
        d_c: int,          # KV压缩维度
        d_c_q: int,        # Q压缩维度
        d_head_r: int,     # RoPE维度/头
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

> **实现优化**：上述代码为教学版本。实际部署时：
> - **权重吸收**：预计算 $W^{QK} = (W^{UQ})^\top W^{UK}$ 和 $W^{OV} = W^O W^{UV}$
> - **FlashAttention**：使用 FlashAttention 加速注意力计算
> - **融合算子**：将多个小矩阵乘法融合

### 2.6 MLA vs 其他方法

| Feature | MHA | GQA | MQA | MLA |
|---------|-----|-----|-----|-----|
| KV Cache | $2n_h d_h$ | $2\frac{n_h}{g} d_h$ | $2 d_h$ | $d_c + d_h^R$ |
| 参数量 | 基准 | 减少 | 最少 | 略增 |
| 表达能力 | 最强 | 较强 | 较弱 | 接近 MHA |
| 推理延迟 | 高 | 中 | 低 | 中 |
| 长上下文 | 受限 | 较好 | 好 | **最好** |

**MLA 的优势**：
- **极致压缩**：KV Cache 减少 93% 以上
- **性能保持**：不像 GQA/MQA 强制共享，而是学习最优压缩
- **长上下文友好**：128K 上下文成为可能

**MLA 的代价**：
- **计算开销**：需要额外的压缩/恢复计算
- **实现复杂度**：Decoupled RoPE 和权重吸收增加实现难度
- **训练成本**：低秩约束可能需要更多训练

### 2.7 质量-效率权衡

MLA 的 56.9 倍压缩比来自于一个强假设：**K 和 V 可以从 512 维潜在空间无损恢复**。这个假设在什么条件下成立？

#### 低秩假设的有效性

考虑注意力模式的多样性需求。设任务需要 $r$ 种本质不同的注意力模式（如：局部依赖、长程依赖、语法结构、语义关联等），则 KV 表示至少需要 $r$ 个自由度。

**压缩维度的下界**：若 $d_c < r$，则压缩会造成信息丢失。DeepSeek 选择 $d_c = 512$，隐含假设是：常见 NLP 任务的注意力模式可以用不超过 512 维的空间表达。

**任务依赖性**：不同任务对注意力多样性的需求不同：
- **文本生成**：模式相对固定，低秩假设成立，MLA 表现良好
- **代码理解**：需要追踪复杂的变量依赖和作用域，可能需要更高秩的表示
- **数学推理**：多步推理需要维持多条推理链，对注意力多样性要求高

#### 选择指南

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 长上下文推理 (>32K) | MLA | KV Cache 是主要瓶颈 |
| 短上下文、高吞吐 | GQA | 实现简单，开销低 |
| 质量优先（小 batch） | MHA | 无压缩损失 |
| 代码/数学任务 | GQA 或 MHA | 注意力多样性需求高 |
| 边缘设备部署 | MLA | 极致内存压缩 |

**经验法则**：当 $\text{上下文长度} \times \text{batch size} > 10^6$ 时，KV Cache 成为主要瓶颈，MLA 的收益开始显现。

#### 压缩比与质量的帕累托边界

DeepSeek 的消融实验表明，$d_c$ 的选择存在"甜点区"：
- $d_c < 256$：质量显著下降，压缩过度
- $d_c \in [256, 768]$：质量接近 MHA，压缩有效
- $d_c > 768$：质量无进一步提升，压缩收益递减

$d_c = 512$ 处于帕累托最优附近——进一步压缩会损害质量，进一步扩大则浪费缓存空间。

### 2.8 应用与扩展

MLA 已在以下模型中应用：
- **DeepSeek-V2**：首次提出，236B 参数 MoE 模型
- **DeepSeek-V3**：进一步优化，671B 参数
- **DeepSeek-R1**：推理模型，继承 MLA 架构

> **MLA 与 MoE 的协同**：MLA 特别适合 MoE 架构：MoE 的稀疏激活已经减少了计算量，而 MLA 进一步解决了内存瓶颈。两者结合使得 DeepSeek-V2 在保持高性能的同时，训练成本降低 42.5%。

## 3. 稀疏注意力

FlashAttention 和 MLA 分别从计算效率和内存占用角度优化了注意力机制，但它们都保留了完整的 $O(N^2)$ 注意力计算——只是让这个计算更快、更省内存。本节探索另一条路径：如果大多数注意力权重本就接近零，我们能否**跳过这些无意义的计算**？

### 3.1 核心思想

标准 Softmax Attention 的复杂度为 $O(N^2)$，但实际上并非所有 token 对都同等重要：
- 注意力分布通常是稀疏的（少数 token 获得大部分权重）
- 远距离 token 的注意力通常较弱
- 语义相关的 token 往往聚集在特定位置

稀疏注意力的核心思想：**只计算重要的 token 对**，将复杂度从 $O(N^2)$ 降至 $O(Nk)$，其中 $k \ll N$。

| 特性 | 稀疏注意力 | 线性注意力 |
|------|------------|------------|
| 复杂度 | $O(Nk)$ | $O(Nd^2)$ |
| 注意力类型 | 精确 Softmax | 近似/替代 |
| 长程精确检索 | 强 | 弱 |
| KV Cache | 需要完整 | 可压缩 |
| 与原始 Transformer 兼容 | 高 | 中 |

### 3.2 滑动窗口注意力

滑动窗口注意力（Sliding Window Attention, SWA）是最直观的稀疏注意力形式：每个 token 只关注其周围固定窗口内的 token。

#### Mistral 的滑动窗口

Mistral 7B 是首个将滑动窗口注意力规模化部署的开源模型，窗口大小为 4096。

**核心机制**：每个位置 $t$ 的 token 只关注 $[t-w, t]$ 范围内的 token：

$$\text{Attention}_t = \text{softmax}\left(\frac{q_t K_{[t-w:t]}^\top}{\sqrt{d}}\right) V_{[t-w:t]}$$

**层间信息传递**：滑动窗口的关键洞察是——通过 Transformer 的堆叠层，信息可以"跨窗口"传播。在第 $k$ 层，位置 $t$ 的 token 实际上可以访问到 $[t - k \cdot w, t]$ 范围的信息。对于 32 层模型、窗口大小 4096，理论感受野可达 128K。

**推理优化**：
- **滚动缓存**（Rolling Buffer）：KV Cache 只需保留最近 $w$ 个 token
- **内存节省**：8K 序列长度下节省 50% 缓存
- **速度提升**：配合 FlashAttention，16K 序列上获得 2× 加速

#### StreamingLLM：无限长输入

StreamingLLM 解决了一个重要问题：如何让 LLM 处理"无限长"的流式输入。

**Attention Sink 现象**：研究发现，无论输入多长，模型总是对**最开头的几个 token** 分配异常高的注意力权重——即使这些 token 在语义上并不重要。这被称为"注意力汇聚"（Attention Sink）。

**SinkAttention**：保留两部分 KV Cache：
- **Sink Tokens**：序列开头的 4 个 token（固定）
- **滑动窗口**：最近的 $w$ 个 token

$$\text{KV Cache} = \text{Sink}_{[1:4]} \cup \text{Window}_{[t-w:t]}$$

**效果**：在 **400 万+ token** 的流式场景下保持稳定性能，而普通滑动窗口在超过预训练长度后崩溃。

### 3.3 KV Cache 稀疏化

KV Cache 稀疏化是推理时的稀疏注意力：动态丢弃"不重要"的 KV 条目。

**H2O（Heavy-Hitter Oracle）**：基于一个观察——少数"重击手" token 累积了大部分注意力权重。算法：维护每个 token 的累积注意力分数，保留分数最高的 Top-k token + 最近的滑动窗口 token。

**SnapKV**：注意力模式在 prefill 阶段基本确定，可以"一次性"剪枝。在 prefill 阶段末尾分析注意力分布，识别重要位置，永久保留这些位置的 KV。

**PyramidKV**：不同层的注意力稀疏度不同——低层注意力较分散需要更多 KV，高层注意力集中可大幅压缩。采用**金字塔形**的 KV 分配：底层多、高层少。

| 方法 | 剪枝时机 | 动态性 | 集成框架 |
|------|----------|--------|----------|
| H2O | 每步 | 动态 | vLLM |
| SnapKV | Prefill 后 | 静态 | vLLM |
| StreamingLLM | 持续 | 静态 | -- |
| PyramidKV | 层级 | 静态 | -- |

### 3.4 MoBA：块稀疏注意力

MoBA（Mixture of Block Attention）是块级稀疏注意力的代表性工作，已部署于 Kimi 的长上下文服务。

#### 核心思想：将 MoE 应用于 Attention

MoBA 的核心洞察是：**并非所有上下文对当前 token 都同等重要**。与其对整个序列计算注意力，不如让模型自主学习"关注哪些块"。

$$\text{MoBA}(q, K, V) = \text{softmax}(qK_{[\mathcal{I}]}^\top)V_{[\mathcal{I}]}$$

其中 $\mathcal{I} \subseteq [N]$ 是被选中的 KV 子集，由路由机制决定。

#### 块划分与路由

**块划分**：将长度为 $N$ 的上下文均匀划分为 $n$ 个块，每块大小 $B = N/n$：

$$\mathcal{I}_i = [(i-1) \cdot B + 1, \, i \cdot B], \quad i = 1, \ldots, n$$

**路由分数计算**（无参数）：对每个 query $q$，计算其与各块的亲和度分数：

$$s_i = \langle q, \text{mean\_pool}(K_{[\mathcal{I}_i]}) \rangle$$

即 query 与块内所有 key 的平均向量的内积。这是一个**无参数**的路由机制。

**Top-k 选择**：选择分数最高的 $k$ 个块进行注意力计算（典型设置 $k = 12$，块大小 $L = 4096$）。

#### 因果性保证

在自回归场景下：
1. **未来块屏蔽**：对于位置 $t$ 的 query，所有 $i > \lceil t/B \rceil$ 的块设 $s_i = -\infty$
2. **当前块强制选中**：query 所在的块始终被路由

| 指标 | MoBA | Full Attention |
|------|------|----------------|
| LM Loss 差异 | \multicolumn{2}{c}{$< 10^{-3}$} |
| 稀疏度 @32K | \multicolumn{2}{c}{95.31%} |
| 加速比 @1M | \multicolumn{2}{c}{6.5×} |
| 加速比 @10M | \multicolumn{2}{c}{16×} |

### 3.5 NSA：原生稀疏注意力

Native Sparse Attention（NSA）是 DeepSeek 提出的层级稀疏注意力机制。

#### 三条注意力路径

NSA 将注意力计算分解为三条并行路径：

**1. Compression Attention（压缩注意力）**：使用可学习的 MLP 将连续 token 压缩为块级表示：

$$\tilde{K}_i = \text{MLP}(K_{[(i-1)l+1:il]}), \quad \tilde{V}_i = \text{MLP}(V_{[(i-1)l+1:il]})$$

其中 $l$ 是压缩块大小（NSA 中 $l=32$）。这捕获**全局粗粒度**信息。

**2. Selection Attention（选择注意力）**：通过 Lightning Indexer 选择最相关的块保持原始精度：
- 计算 query 与所有块的相关性分数
- 选择 Top-$n$ 个块（NSA 中 $n=16$ 个块，块大小 $l'=64$）
- 对选中块进行精确 Softmax 注意力

这保留**细粒度精确**信息。

**3. Sliding Window Attention（滑动窗口注意力）**：对最近的 $w$ 个 token 进行完整注意力（NSA 中 $w=512$）。这保证**局部上下文**的精确建模。

#### Lightning Indexer

Lightning Indexer 是 NSA 的核心创新，用于高效选择相关块：
- 维护独立的 **FP8 量化** Key 缓存（非 MLA 的 KV Cache）
- 每个 query 计算与所有块的相关性分数
- 选择 Top-k 块（默认 2048 个 token）
- 硬件优化：DeepGEMM 实现的高效 CUDA kernel

**关键设计**：索引计算与注意力计算分离，索引使用低精度快速完成。

#### 端到端可训练

与 ClusterKV、MagicPIG 等依赖不可微操作的方法不同，NSA 是**原生可训练**的——从预训练阶段就使用稀疏注意力。

| 配置 | 值 |
|------|-----|
| 压缩块大小 | 32 |
| 选择块大小 | 64 |
| 选择块数量 | 16 |
| 滑动窗口 | 512 |

**训练加速**（64K 序列，A100）：前向 9×，反向 6×。加速比随序列长度增加：8K 时 4×，16K 时 6.4×，32K 时 9.1×，64K 时 11.6×。

### 3.6 DSA：DeepSeek 稀疏注意力

DSA（DeepSeek Sparse Attention，2025 年 9 月）是 DeepSeek 在 V3.2 中部署的新一代稀疏注意力，与 NSA 有本质区别。DSA 摒弃了 NSA 复杂的三分支设计，采用更简洁的**细粒度 token 级检索**。

#### 算法设计

DSA 的核心思想：每个 query 只需关注固定数量 $k$ 个最相关的 token（$k=2048$）。

**重要度分数计算**：DSA 引入可学习权重 $w$ 计算 token 重要度：

$$\text{score}_i = w \cdot f(q, k_i)$$

这是一个折中方案——比 NSA 的 MLP 简单，但比 MoBA 的无参数 mean-pooling 更有表达力。

**Top-k 检索**：根据重要度分数选择 Top-$k$ 个 token 进行精确注意力计算：

$$\mathcal{I} = \text{Top-}k(\{\text{score}_i\}_{i=1}^N), \quad |\mathcal{I}| = 2048$$

**复杂度**：单 query 需访问固定 $k$ 个 token，因此整体复杂度为 $O(Nk)$，是真正的**线性复杂度**。

#### 与 NSA 的核心区别

| 特性 | NSA | DSA |
|------|-----|-----|
| 选择粒度 | 块级 | Token 级 |
| 分支数量 | 3（压缩+选择+窗口） | 1（直接选择） |
| 重要度计算 | 可学习 MLP | 可学习 $w$ 权重 |
| Attention 变种 | GQA | MLA |
| 验证模型 | 27B | 671B |

#### 工程实现

- **TileLang Kernel**：细粒度稀疏 + MLA 需要定制 kernel，TileLang 比 Triton 性能更优
- **vLLM/SGLang 集成**：Day-0 支持，使用 DeepGEMM 和 FlashMLA
- **Blackwell 优化**：与 NVIDIA 合作优化 B200

**性能收益**：长上下文 API 成本降低约 **50%**，64K 序列上实现显著加速，671B 模型上验证质量几乎无损。

### 3.7 三种方法对比

| 设计维度 | NSA | MoBA | DSA |
|----------|-----|------|-----|
| 发布时间 | 2025.02 | 2025.02 | 2025.09 |
| 提出者 | DeepSeek | Moonshot (Kimi) | DeepSeek |
| 选择粒度 | 块级 | 块级 | Token 级 |
| 路由机制 | 可学习 MLP | 无参数 mean-pool | 可学习 $w$ |
| 局部窗口 | 有（$w$=512） | 当前块强制选中 | 无 |
| 复杂度 | $O(N^2/L)$ | $O(N \cdot kL)$ | $O(Nk)$ |

#### 超参数对比

| 参数 | NSA | MoBA | DSA |
|------|-----|------|-----|
| 块大小 | $l$=32, $l'$=64 | $L$=4096 | -- |
| 选择数量 | $n$=16 块 | $k$=12 块 | $k$=2048 tokens |
| 滑动窗口 | $w$=512 | -- | -- |
| @32K 访问 token 数 | ~2560 | 49152 | 2048 |
| @32K 稀疏度 | 92% | 0%（看全部） | 94% |

**关键观察**：在 32K 这个长度上，MoBA 实际上几乎没有稀疏（选中 12 块 × 4096 = 49152 > 32K）！MoBA 的稀疏优势在更长序列（如 128K+）才能体现。

#### 设计哲学差异

- **NSA**：全面覆盖，层级融合——三分支通过可学习门控融合，不遗漏任何重要信息
- **MoBA**：简洁优雅，MoE 思想——把 KV Cache 视为"专家池"，无参数路由让注意力分数自然决定选择
- **DSA**：激进稀疏，端到端优化——Token 级选择，每个 query 只看 2048 个 token（约 3%@64K）

#### 适用场景

| 方法 | 适用场景 |
|------|----------|
| NSA | 需要精确保留多尺度信息；可接受复杂超参调优；使用 GQA 架构 |
| MoBA | 追求简洁设计；希望无缝替换现有 Attention；序列长度 128K+ |
| DSA | 使用 MLA 架构；追求极致稀疏度；超大模型（100B+） |

### 3.8 Ring Attention

当序列长度超过单 GPU 显存时，Ring Attention 将长序列分割到多个 GPU，通过环形通信实现分布式注意力计算。

**算法流程**：
1. 将 Query、Key、Value 按序列维度分割到 $P$ 个 GPU
2. 每个 GPU 持有本地 Query 块，计算与本地 KV 的注意力
3. KV 块在 GPU 之间环形传递，累积计算全局注意力
4. 使用 Online Softmax 避免数值溢出

**通信隐藏**：关键优化是**计算-通信重叠**——在计算当前 KV 块注意力的同时，异步传递下一个 KV 块。

**LLaMA 3 的上下文并行**：采用 All-Gather 方式的 Context Parallelism——先 All-Gather 收集所有 KV，再计算本地 Query 的注意力。为负载均衡，将序列分为 $2 \times \text{CP}$ 个块并 shuffle，支持 128K 上下文的高效训练。

### 3.9 稀疏注意力方法全景

| 方法 | 稀疏策略 | 复杂度 | 全局信息 | 部署 |
|------|----------|--------|----------|------|
| Sliding Window | 固定窗口 | $O(Nw)$ | 无 | Mistral |
| StreamingLLM | Sink+窗口 | $O(N(s+w))$ | Sink tokens | -- |
| MoBA | 块路由 | $O(N \cdot kL)$ | Top-k 块 | Kimi |
| NSA | 压缩+选择+窗口 | $O(N^2/L)$ | 压缩+选择 | -- |
| DSA | Token 级检索 | $O(Nk)$ | Top-k token | DeepSeek-V3.2 |
| Ring Attention | 分布式 | $O(N^2/P)$ | 完整 | LLaMA 3 |

> **稀疏注意力的演进**：从 Longformer/BigBird 的"手工设计模式"到 MoBA/NSA/DSA 的"学习式稀疏"，稀疏注意力正经历范式转变。2025 年，稀疏注意力首次在 600B+ 规模模型上得到验证（DeepSeek-V3.2），标志着该技术从学术研究走向工业主流。

## 4. 线性注意力

稀疏注意力通过"只计算重要的 token 对"将 $O(N^2)$ 降至 $O(Nk)$，但它仍然保留了 softmax 的计算形式。本节介绍一条更激进的路径：**彻底改变注意力的数学形式**，使复杂度降至真正的 $O(N)$。

### 4.1 核心思想

线性注意力的核心洞察是：softmax 注意力之所以需要 $O(N^2)$，是因为必须先算出完整的 $N \times N$ 注意力矩阵再做归一化。如果我们用其他函数替代 softmax，使得计算可以"重新排列"，就能避免显式构造这个矩阵。

#### 从 Softmax Attention 到线性化

标准自注意力（单头）的计算为：

$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)}_{n \times n} V$$

其中需要显式构造 $n \times n$ 的注意力矩阵，时间和空间复杂度均为 $O(n^2)$。

线性注意力的核心思想是：将 softmax 或 $QK^\top$ 改写为可分解的形式，利用乘法结合律改变计算顺序：

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot \underbrace{(\phi(K)^\top V)}_{d \times d}$$

其中 $\phi(\cdot)$ 是某种特征映射函数。关键在于：先计算 $\phi(K)^\top V$（与长度 $n$ 线性相关的 $d \times d$ 矩阵），再左乘 $\phi(Q)$，总复杂度降为 $O(nd^2)$，当 $d \ll n$ 时近似 $O(n)$。

#### 递推形式：Transformer 即 RNN

在自回归（causal）场景下，线性注意力可以写成类似 RNN 的递推式。设 $q_t, k_t, v_t$ 分别为第 $t$ 步的 query、key、value 向量，定义状态矩阵 $S_t \in \mathbb{R}^{d \times d}$：

$$S_t = S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

这揭示了一个深刻联系：**线性 Attention 本质上是一种 RNN**，其隐状态 $S_t$ 累积了历史信息。

> **线性注意力的双模式**：线性注意力支持两种等价的计算模式：
> - **并行模式**：训练时使用矩阵乘法，充分利用 GPU 并行性
> - **递推模式**：推理时使用 RNN 形式，实现 $O(1)$ 的增量更新
>
> 这种双模式特性使线性注意力在训练和推理阶段都能获得最优效率。

### 4.2 经典方法

#### Linear Transformer

最早系统化提出"Transformer 即 RNN"的工作。核心思想是将 softmax attention 重写为核函数形式：

$$\text{Attention}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$

其中 $\phi(x) = \text{elu}(x) + 1$ 保证非负性。实验表明在自回归任务上可获得高达 **4000 倍**的加速。

**局限**：简单的特征映射难以精确近似 softmax 的行为，在复杂语言任务上存在性能差距。

#### Performer

提出 FAVOR+（Fast Attention Via positive Orthogonal Random features）方法，用随机特征近似 softmax 核：

$$\text{softmax}(q^\top k) \approx \phi(q)^\top \phi(k)$$

其中 $\phi$ 通过正交随机特征构造，具有无偏或近似无偏的理论保证。

**优势**：与原始 Transformer 完全兼容，可作为 drop-in replacement。
**局限**：随机近似在实际任务中仍有精度损失。

#### cosFormer

不再硬近似 softmax，而是基于 softmax 的两个关键性质设计线性替代：
1. **非负性**：注意力权重应为非负
2. **分布集中性**：注意力应集中在相关位置

cosFormer 使用 ReLU 保证非负性，并引入基于余弦的位置再加权机制：

$$\text{Attention}_{ij} = \text{ReLU}(q_i)^\top \text{ReLU}(k_j) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i - j}{n}\right)$$

在 Long-Range Arena 等长序列 benchmark 上取得当时最优性能，是"好用的线性 Attention"的代表。

### 4.3 带遗忘门的线性注意力

原始线性注意力的一个根本问题是：状态矩阵 $S_t$ 只能累加，无法遗忘。随着序列增长，历史信息"挤在一起"，导致检索能力下降。

#### RetNet

微软提出的 Retentive Network 引入指数衰减因子 $\gamma \in (0, 1)$：

$$S_t = \gamma S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

这相当于对历史信息施加指数衰减，强调近期 token 的重要性。

**Multi-Scale Retention**：不同 attention head 使用不同的 $\gamma$ 值，实现多尺度的记忆保持：
- 小 $\gamma$：关注近期信息（短程依赖）
- 大 $\gamma$：保留更长历史（长程依赖）

**三种计算模式**：
1. **并行模式**：训练时的矩阵计算
2. **递推模式**：$O(1)$ 推理
3. **分块递推模式**：长序列的高效处理

**性能**：7B 模型在 8k 序列下，推理速度比 Transformer 快 **8.4 倍**，内存减少 **70%**。

#### Lightning Attention

MiniMax 提出的 Lightning Attention 是目前**首个规模化到商业级的线性注意力架构**。核心创新：

**分块计算策略**：将注意力计算分为 intra-block 和 inter-block 两部分：
- **Intra-block**：块内使用"左乘"形式，可并行计算
- **Inter-block**：块间使用"右乘"形式，累积状态

这种分解避免了传统线性注意力中缓慢的 cumsum 操作。

**混合架构**：每 8 层中，7 层使用 Lightning Attention，1 层使用标准 Softmax Attention，平衡效率与精度。

| 参数 | MiniMax-01 |
|------|------------|
| 总参数量 | 456B |
| 激活参数量（MoE） | 45.9B |
| 专家数量 | 32 |
| 训练上下文长度 | 1M tokens |
| 推理外推长度 | 4M tokens |

### 4.4 DeltaNet：基于 Delta Rule

#### 动机：解决记忆过载问题

原始线性注意力的核心缺陷是**记忆过载**（memory overload）：只能添加新的 key-value 关联，无法擦除已有信息。这导致随着序列增长，检索错误累积。

#### Delta Rule 更新

DeltaNet 引入"除旧迎新"的 Delta Rule：

$$S_t = S_{t-1} - \underbrace{(S_{t-1} \cdot k_t - v_t)}_{\text{delta}} \cdot k_t^\top$$

直观理解：
1. $S_{t-1} \cdot k_t$：用当前 key 检索记忆中的 value
2. $S_{t-1} \cdot k_t - v_t$：计算检索值与真实值的差异（delta）
3. 根据 delta 修正记忆，实现"精准更新"

#### Gated DeltaNet

ICLR 2025 工作进一步引入门控机制：

$$S_t = \alpha_t \odot S_{t-1} + \beta_t \odot (v_t - S_{t-1} \cdot k_t) \cdot k_t^\top$$

其中 $\alpha_t$ 控制遗忘，$\beta_t$ 控制更新强度。

**互补性**：门控实现快速记忆擦除，Delta Rule 实现精准记忆更新，两者结合在多个 benchmark 上超越 Mamba2 和原始 DeltaNet。

**工业采用**：Gated DeltaNet 已被 **Qwen3-Next** 采用作为线性注意力层。

### 4.5 与状态空间模型的联系

Mamba 是另一条重要的高效序列建模路线，基于选择性状态空间模型（Selective SSM）。Mamba-2 论文揭示了**结构化状态空间对偶**（Structured State Space Duality, SSD）：

> "与标准自注意力相比，SSD 只有两个区别：去掉 softmax 归一化，并应用一个独立的逐元素掩码矩阵。"

这表明线性注意力和 SSM 可以视为同一框架的不同实例：
- **线性注意力**：通过特征映射分解 attention 矩阵
- **SSM**：通过状态空间方程建模序列
- 两者都有线性复杂度和递推形式

**混合架构**：实践中，纯线性模型在某些任务上仍有差距，因此出现了混合架构：
- **Jamba**（AI21）：Mamba + Attention
- **MiniMax-01**：Lightning Attention + 稀疏 Softmax Attention
- **Qwen3-Next**：Gated DeltaNet + SwiGLU

### 4.6 Test-Time Training 视角

苏剑林在"线性注意力简史"中指出，现代线性注意力的核心思想可以统一到 **Test-Time Training**（TTT）框架：

> "将序列建模视为在线学习问题，用优化器构建 RNN。不同的损失函数对应不同的 RNN 模型。"

| 方法 | 更新规则 | 对应优化器 |
|------|----------|------------|
| Linear Attention | $S_t = S_{t-1} + v_t k_t^\top$ | 累积梯度 |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | 带衰减的累积 |
| DeltaNet | $S_t = S_{t-1} - (S_{t-1}k_t - v_t)k_t^\top$ | Delta Rule |
| Gated DeltaNet | 门控 Delta Rule | 自适应学习率 |

这个视角为设计新型线性注意力提供了原则性指导：选择合适的"优化器"来更新记忆状态。

### 4.7 工业部署现状

| 公司/模型 | 架构类型 | 上下文长度 | 特点 |
|-----------|----------|------------|------|
| MiniMax-01 | Lightning Attention + MoE | 1M-4M | 首个商业级线性 Attention LLM |
| MiniMax-M1 | Lightning Attention | 1M+80k 生成 | 开源 reasoning 模型 |
| Qwen3-Next | Gated DeltaNet | -- | 线性层 + 门控 Attention |

**关键观察**：MiniMax 是目前唯一将线性注意力规模化到商业级的厂商。其他厂商（如 Kimi、DeepSeek）更倾向于稀疏注意力路线。

### 4.8 局限与展望

**当前局限**：
1. **精度差距**：在复杂推理任务上，纯线性注意力仍落后于 Softmax Attention
2. **In-context learning 能力**：线性模型的 few-shot 能力通常弱于 Transformer
3. **长程精确检索**：passkey retrieval 等任务上表现不稳定

**发展趋势**：
1. **混合架构**：结合线性层和稀疏 Softmax 层
2. **门控机制**：更精细的记忆管理（如 Gated DeltaNet）
3. **知识蒸馏**：从 Softmax 模型蒸馏到线性模型（如 LAWCAT）
4. **TTT 原则**：基于优化器视角设计新架构

> **历史评价**：苏剑林的评价——"线性注意力已从单纯模仿 Softmax Attention 发展到'反哺'它——通过核技巧将 DeltaNet 改进应用于 Softmax Attention。这表明该领域方兴未艾，仍有广阔探索空间。"

## 本章小结

本章介绍了四种注意力机制的优化路径：

1. **FlashAttention**：IO 感知优化，不改变计算量但大幅减少内存访问
   - v1→v4 持续演进，峰值性能提升 15 倍

2. **MLA**：低秩压缩 KV Cache
   - DeepSeek-V2 实现 56.9× 压缩比，接近 MHA 性能

3. **稀疏注意力**：只计算重要的 token 对
   - NSA/MoBA/DSA 代表"学习式稀疏"的演进
   - DSA 在 671B 模型上验证，API 成本降低 50%

4. **线性注意力**：改变计算形式，真正的 $O(N)$ 复杂度
   - Lightning Attention 首次规模化到商业级
   - Gated DeltaNet 被 Qwen3-Next 采用

**发展趋势**：稀疏注意力与线性注意力正从学术研究走向工业主流，2025 年两条路线都在 600B+ 规模模型上得到验证。

下一篇将介绍 Mixture of Experts（MoE）架构。
