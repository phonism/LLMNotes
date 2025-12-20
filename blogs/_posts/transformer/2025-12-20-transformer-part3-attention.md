---
layout: post
title: "Transformer 学习笔记（三）：注意力机制"
date: 2025-12-20 10:20:00
author: Phonism
tags: [Transformer, FlashAttention, MLA, Sparse Attention, Linear Attention]
lang: zh
translation: /transformer-part3-attention-en/
---

注意力机制是 Transformer 的核心，但标准的 $O(N^2)$ 复杂度成为长上下文建模的瓶颈。本文系统介绍四种优化路径：FlashAttention 通过 IO 感知优化加速计算；MLA 通过低秩压缩减少 KV Cache；稀疏注意力只计算重要的 token 对；线性注意力彻底改变计算形式。

## 1. FlashAttention：IO 感知的高效注意力

### 1.1 问题：GPU 内存层次瓶颈

现代 GPU 的计算能力远超内存带宽。以 NVIDIA A100 为例：
- 计算能力：312 TFLOPS（FP16 Tensor Core）
- HBM 带宽：2 TB/s
- SRAM 带宽：约 19 TB/s

标准注意力实现需要多次 HBM 读写：
1. 从 HBM 读取 $Q, K$，计算 $S = QK^\top$，写回 HBM
2. 从 HBM 读取 $S$，计算 $P = \text{softmax}(S)$，写回 HBM
3. 从 HBM 读取 $P, V$，计算 $O = PV$，写回 HBM

总 HBM 访问量：$O(N^2 + Nd)$，造成严重的 IO 瓶颈。

### 1.2 FlashAttention v1：永不落地的注意力矩阵

FlashAttention 的核心思想：**永远不将完整的 $N \times N$ 注意力矩阵写入 HBM**。

**Online Softmax 算法**：允许单次遍历、增量计算。对于两个块 $x^{(1)}, x^{(2)}$ 的拼接：

$$m^{new} = \max(m^{(1)}, m^{(2)})$$
$$\ell^{new} = e^{m^{(1)} - m^{new}} \ell^{(1)} + e^{m^{(2)} - m^{new}} \ell^{(2)}$$

**Tiling 算法**：将 $Q, K, V$ 分成大小为 $B_r \times d$ 和 $B_c \times d$ 的块，在 SRAM 中完成计算。

**重计算策略**：反向传播时从 $Q, K, V$ 块重新计算 $S, P$，虽然增加约 30% FLOPs，但大幅减少 HBM 访问。

**IO 复杂度**：从标准注意力的 $O(N^2 + Nd)$ 降至 $O(N^2 d^2 / M)$，当 $M = \Theta(Nd)$ 时减少 $O(N)$ 倍 HBM 访问。

### 1.3 FlashAttention-2：并行策略优化

FlashAttention-2 通过优化并行策略，在 A100 上达到 230 TFLOPS（约 73% 峰值利用率），比 v1 快约 2 倍。

**Sliced-Q 方案**：将 $Q$ 在 4 个 warp 间分割，$K, V$ 对所有 warp 可见。消除 warp 间通信，减少共享内存读写。

| 指标 | v1 | v2 |
|------|----|----|
| A100 峰值 (TFLOPS) | 124 | 230 |
| GPU 利用率 | 25-40% | 50-73% |

### 1.4 FlashAttention-3：Hopper 架构优化

针对 NVIDIA H100 设计，达到 740 TFLOPS（75% 峰值利用率）。

**三大优化技术**：
1. **Warp Specialization**：Producer warp 负责 TMA 数据传输，Consumer warp 负责 WGMMA 计算
2. **Ping-Pong Scheduling**：两个 warpgroup 交替执行 GEMM 和 Softmax
3. **Intra-warpgroup Overlapping**：Softmax 与 GEMM 流水线化

**FP8 支持**：通过 Block Quantization 和 Incoherent Processing，FP8 吞吐量接近 1.2 PFLOPS。

### 1.5 Flash Decoding：推理时的 KV 并行

FlashAttention 在推理时存在问题：每步只生成 1 个 token，GPU 严重 underutilized。

Flash Decoding 在 **KV 序列长度**维度并行：
1. 将 KV Cache 分成 $S$ 个块
2. 对每个块独立计算部分注意力
3. 使用 Log-Sum-Exp 合并结果

性能：长序列解码加速高达 **8 倍**，在 CodeLLaMa-34B 上注意力操作比 FlashAttention 快 **50 倍**。

### 1.6 FlashAttention 版本演进

| 版本 | GPU | 峰值 TFLOPS | 利用率 | 相对 v1 加速 |
|------|-----|-------------|--------|--------------|
| v1 | A100 | 124 | 40% | 1× |
| v2 | A100 | 230 | 73% | 1.85× |
| v3 | H100 | 740 | 75% | 6× |
| v3 (FP8) | H100 | 1200 | - | 9.7× |
| v4 | B200 | PFLOPS | - | 15× |

**工程影响**：FlashAttention 已成为现代 LLM 的标配，将实用上下文长度从 2-4K 提升到 128K+。

## 2. Multi-head Latent Attention (MLA)

### 2.1 KV Cache 挑战

自回归生成时，KV Cache 成为长上下文推理的主要内存瓶颈：

$$\text{KV Cache Size} = 2 \times B \times S \times L \times n_h \times d_h \times \text{bytes}$$

对于 70B 模型（$L=80$, $K=8$, $S=8192$），KV Cache 达到 **2.1 GB/request**。

### 2.2 现有方案的局限

| Method | KV Cache Size | Performance |
|--------|---------------|-------------|
| MHA | $2 n_h d_h$ | 最优 |
| GQA | $2 \frac{n_h}{g} d_h$ | 轻微下降 |
| MQA | $2 d_h$ | 明显下降 |
| **MLA** | $d_c + d_h^R$ | **接近 MHA** |

GQA/MQA 通过**强制共享** KV 头来减少缓存，但损害模型性能。MLA 的核心洞察：**KV 可以从一个低维潜在向量中恢复**。

### 2.3 MLA 核心原理

**KV 的低秩压缩**：

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$$

从潜在向量恢复 K 和 V：

$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}$$

其中 $d_c \ll n_h d_h$ 是压缩维度。

**Decoupled RoPE**：将每个注意力头分为两部分：
- **内容部分**（$d_h^C$ 维）：从压缩向量恢复，**不**应用 RoPE
- **位置部分**（$d_h^R$ 维）：额外投影，应用 RoPE

$$\mathbf{q}_t = [\mathbf{q}_t^C; \text{RoPE}(\mathbf{q}_t^R, t)]$$

**权重吸收**：由于压缩和恢复之间没有非线性激活，矩阵可以合并：

$$W^{QK} = (W^{UQ})^\top W^{UK}$$

### 2.4 KV Cache 压缩效果

| Method | Cache Elements | DeepSeek-V2 |
|--------|----------------|-------------|
| MHA | $2 n_h d_h$ | 32768 |
| GQA (8 组) | $2 \times 8 \times d_h$ | 2048 |
| **MLA** | $d_c + d_h^R$ | **576** |

DeepSeek-V2 配置：$d_c = 512$，$d_h^R = 64$，压缩比达到 **56.9×**。

### 2.5 质量-效率权衡

MLA 的 56.9 倍压缩比来自假设：**K 和 V 可以从 512 维潜在空间无损恢复**。

**压缩维度的选择**：
- $d_c < 256$：质量显著下降
- $d_c \in [256, 768]$：质量接近 MHA
- $d_c > 768$：质量无进一步提升

$d_c = 512$ 处于帕累托最优附近。

**应用模型**：DeepSeek-V2（236B）、DeepSeek-V3（671B）、DeepSeek-R1。

## 3. 稀疏注意力

### 3.1 核心思想

标准 Softmax Attention 的复杂度为 $O(N^2)$，但并非所有 token 对都同等重要。稀疏注意力只计算"重要"的 token 对，将复杂度降至 $O(Nk)$。

| 特性 | 稀疏注意力 | 线性注意力 |
|------|------------|------------|
| 复杂度 | $O(Nk)$ | $O(Nd^2)$ |
| 注意力类型 | 精确 Softmax | 近似/替代 |
| 长程精确检索 | 强 | 弱 |

### 3.2 滑动窗口注意力

**Mistral 的滑动窗口**：每个位置 $t$ 只关注 $[t-w, t]$ 范围内的 token。

关键洞察：通过 Transformer 的堆叠层，信息可以"跨窗口"传播。32 层模型、窗口 4096，理论感受野可达 128K。

**StreamingLLM**：保留 Sink Tokens + 滑动窗口，在 **400 万+ token** 的流式场景下保持稳定。

### 3.3 MoBA：块稀疏注意力

MoBA（Mixture of Block Attention）将 MoE 思想应用于 Attention：

$$\text{MoBA}(q, K, V) = \text{softmax}(qK_{[\mathcal{I}]}^\top)V_{[\mathcal{I}]}$$

**路由分数计算**（无参数）：

$$s_i = \langle q, \text{mean\_pool}(K_{[\mathcal{I}_i]}) \rangle$$

选择分数最高的 $k$ 个块进行注意力计算（典型设置 $k = 12$，块大小 $L = 4096$）。

| 指标 | MoBA |
|------|------|
| LM Loss 差异 | $< 10^{-3}$ |
| 稀疏度 @32K | 95.31% |
| 加速比 @10M | 16× |

### 3.4 NSA：原生稀疏注意力

DeepSeek 提出的三条并行路径：

1. **Compression Attention**：可学习 MLP 将块压缩为单个表示，捕获全局粗粒度信息
2. **Selection Attention**：Lightning Indexer 选择 Top-k 块，保留细粒度精确信息
3. **Sliding Window**：保证局部上下文精确建模

**端到端可训练**：从预训练阶段就使用稀疏注意力。

| 配置 | 值 |
|------|-----|
| 压缩块大小 | 32 |
| 选择块大小 | 64 |
| 选择块数量 | 16 |
| 滑动窗口 | 512 |

**训练加速**（64K 序列，A100）：前向 9×，反向 6×。

### 3.5 DSA：DeepSeek 稀疏注意力

DSA 是 DeepSeek-V3.2 部署的新一代稀疏注意力，采用**细粒度 token 级检索**。

**与 NSA 的核心区别**：

| 特性 | NSA | DSA |
|------|-----|-----|
| 选择粒度 | 块级 | Token 级 |
| 分支数量 | 3 | 1 |
| Attention 变种 | GQA | MLA |
| 验证模型 | 27B | 671B |

每个 query 只需关注固定 $k=2048$ 个最相关的 token，实现真正的**线性复杂度** $O(Nk)$。

**性能收益**：长上下文 API 成本降低约 **50%**。

### 3.6 三种方法对比

| 设计维度 | NSA | MoBA | DSA |
|----------|-----|------|-----|
| 发布时间 | 2025.02 | 2025.02 | 2025.09 |
| 选择粒度 | 块级 | 块级 | Token 级 |
| 路由机制 | 可学习 MLP | 无参数 mean-pool | 可学习 $w$ |
| 复杂度 | $O(N^2/L)$ | $O(N \cdot kL)$ | $O(Nk)$ |

**设计哲学差异**：
- **NSA**：全面覆盖，层级融合
- **MoBA**：简洁优雅，MoE 思想
- **DSA**：激进稀疏，端到端优化

### 3.7 Ring Attention

当序列长度超过单 GPU 显存时，Ring Attention 将长序列分割到多个 GPU，通过环形通信实现分布式注意力计算。

关键优化：**计算-通信重叠**——在计算当前 KV 块注意力的同时，异步传递下一个 KV 块。

## 4. 线性注意力

### 4.1 核心思想

线性注意力彻底改变注意力的数学形式，使复杂度降至真正的 $O(N)$。

**从 Softmax 到线性化**：

$$\text{Attention}(Q, K, V) \approx \phi(Q) \cdot (\phi(K)^\top V)$$

先计算 $\phi(K)^\top V$（$d \times d$ 矩阵），再左乘 $\phi(Q)$，总复杂度降为 $O(nd^2)$。

**递推形式**（Transformer 即 RNN）：

$$S_t = S_{t-1} + v_t k_t^\top, \quad o_t = S_t \cdot q_t$$

隐状态 $S_t$ 累积了历史信息，支持 $O(1)$ 的增量更新。

### 4.2 经典方法

**Linear Transformer**：$\phi(x) = \text{elu}(x) + 1$，自回归任务上可获得高达 4000 倍加速。

**Performer**：用随机特征近似 softmax 核，可作为 drop-in replacement。

**cosFormer**：使用 ReLU 保证非负性，引入余弦位置再加权。

### 4.3 带遗忘门的线性注意力

原始线性注意力的问题：状态矩阵 $S_t$ 只能累加，无法遗忘。

**RetNet**：引入指数衰减因子 $\gamma$：

$$S_t = \gamma S_{t-1} + v_t k_t^\top$$

**Multi-Scale Retention**：不同 head 使用不同的 $\gamma$ 值，实现多尺度记忆。

性能：7B 模型在 8k 序列下，推理速度比 Transformer 快 **8.4 倍**，内存减少 **70%**。

**Lightning Attention**（MiniMax）：分块计算策略，首个规模化到商业级的线性注意力架构。

| 参数 | MiniMax-01 |
|------|------------|
| 总参数量 | 456B |
| 激活参数量 | 45.9B |
| 训练上下文长度 | 1M tokens |
| 推理外推长度 | 4M tokens |

### 4.4 DeltaNet：基于 Delta Rule

**记忆过载问题**：原始线性注意力只能添加新的 key-value 关联，无法擦除已有信息。

**Delta Rule 更新**：

$$S_t = S_{t-1} - (S_{t-1} \cdot k_t - v_t) \cdot k_t^\top$$

直观理解：计算检索值与真实值的差异（delta），根据 delta 修正记忆。

**Gated DeltaNet**：引入门控机制，控制遗忘和更新强度。已被 **Qwen3-Next** 采用。

### 4.5 Test-Time Training 视角

现代线性注意力可以统一到 **TTT** 框架：

| 方法 | 更新规则 | 对应优化器 |
|------|----------|------------|
| Linear Attention | $S_t = S_{t-1} + v_t k_t^\top$ | 累积梯度 |
| RetNet | $S_t = \gamma S_{t-1} + v_t k_t^\top$ | 带衰减的累积 |
| DeltaNet | Delta Rule | 精准更新 |

### 4.6 工业部署现状

| 公司/模型 | 架构类型 | 上下文长度 |
|-----------|----------|------------|
| MiniMax-01 | Lightning Attention + MoE | 1M-4M |
| MiniMax-M1 | Lightning Attention | 1M+80k 生成 |
| Qwen3-Next | Gated DeltaNet | - |

**关键观察**：MiniMax 是目前唯一将线性注意力规模化到商业级的厂商。

### 4.7 局限与展望

**当前局限**：
- 在复杂推理任务上落后于 Softmax Attention
- In-context learning 能力较弱
- 长程精确检索不稳定

**发展趋势**：
- 混合架构：线性层 + 稀疏 Softmax 层
- 门控机制：更精细的记忆管理
- 知识蒸馏：从 Softmax 模型蒸馏

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
