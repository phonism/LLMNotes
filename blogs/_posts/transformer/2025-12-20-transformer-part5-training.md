---
layout: post
title: "Transformer 学习笔记（五）：训练技术全景"
date: 2025-12-20 10:40:00
categories: [Deep Learning, Transformer]
tags: [Data Engineering, Distributed Training, ZeRO, FSDP, Muon Optimizer]
math: true
lang: zh
translation: /transformer-part5-training-en/
---

本文是 Transformer 系列的第五篇，全面解析大模型的 **训练技术**，包括数据工程、分布式训练框架和新型优化器。这些技术共同支撑了千亿参数模型的高效训练。

## 1. 数据工程

数据是大语言模型的基石。本节系统介绍预训练数据、后训练数据的构建方法。

### 1.1 预训练数据来源

现代 LLM 的预训练数据通常来自以下来源：

| 数据源 | 描述 | 规模 | 质量 |
|--------|------|------|------|
| **网页数据** | | | |
| Common Crawl | 最大的网页爬虫数据 | PB 级 | 低 |
| RefinedWeb | 过滤后的高质量网页 | 5T tokens | 中 |
| C4 | Colossal Clean Crawled Corpus | 800B tokens | 中 |
| **高质量文本** | | | |
| Wikipedia | 百科全书 | 数十 B tokens | 高 |
| Books | 书籍（Books3, Pile-Books） | 数十 B tokens | 高 |
| arXiv | 学术论文 | 数十 B tokens | 高 |
| **代码** | | | |
| GitHub | 开源代码仓库 | 数百 B tokens | 中-高 |
| The Stack | 去重的开源代码 | 3T tokens | 高 |

### 1.2 数据处理流程

从原始数据到训练数据需要经过多个处理阶段：

**1. 文本提取**
- HTML 解析：使用 trafilatura、jusText 等工具提取正文
- PDF 解析：使用 PyMuPDF、pdfplumber 处理学术文档
- 代码处理：保留注释和文档字符串

**2. 质量过滤**
- **启发式规则**：文档长度、特殊字符比例、重复行比例、语言检测
- **模型打分**：困惑度过滤、质量分类器、教育价值评分（FineWeb-Edu）

**3. 去重**

去重是预训练数据处理的关键步骤，重复数据会导致训练效率下降、模型记忆而非泛化。

主要去重方法：
- **精确去重**：基于哈希的完全匹配
- **模糊去重**：MinHash + LSH（Locality-Sensitive Hashing）

```python
# MinHash 去重示例
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

# 创建 LSH 索引
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc_id, text in documents:
    minhash = get_minhash(text)
    lsh.insert(doc_id, minhash)
```

**4. 敏感内容过滤**
- PII（个人身份信息）移除
- 有害内容过滤
- 版权内容处理

### 1.3 数据配比

数据配比（Data Mix）对模型能力有重要影响：

| 模型 | 网页 | 代码 | 书籍 | 学术 |
|------|------|------|------|------|
| GPT-3 | 60% | 3% | 16% | - |
| LLaMA | 67% | 4.5% | 4.5% | 2.5% |
| DeepSeek | 56% | 18% | - | 5% |

**配比原则**：
- **代码数据**：提升推理能力，通常占 5-20%
- **数学数据**：提升数学能力，但占比过高可能损害通用能力
- **多语言数据**：中文模型通常 30-50% 中文
- **高质量数据**：虽然量少，但应该多次采样（upsampling）

### 1.4 数据规模与 Over-training

Chinchilla Scaling Law 指出，计算最优的数据量与模型参数成正比：

$$D_{opt} \approx 20 \times N$$

其中 $D$ 是 token 数，$N$ 是参数量。即 7B 模型需要约 140B tokens。

然而，实践中常常使用更多数据（over-training）：

| 模型 | 参数量 | 训练 tokens | 倍数 |
|------|--------|-------------|------|
| LLaMA-7B | 7B | 1T | 143x |
| LLaMA 2-7B | 7B | 2T | 286x |
| LLaMA 3-8B | 8B | 15T | 1875x |

**为什么 Over-training？**
- 推理成本固定，训练成本可摊销
- 更小的模型更容易部署
- 数据重复使用在一定程度内是有益的

### 1.5 数据课程与退火

**数据课程（Data Curriculum）**：按特定顺序呈现数据

1. **阶段 1**：通用网页数据，建立基础语言能力
2. **阶段 2**：增加高质量数据比例（书籍、维基百科）
3. **阶段 3**：增加代码和数学数据
4. **阶段 4**：Annealing 阶段，使用最高质量数据，降低学习率

**退火阶段（Annealing）**：训练末期使用高质量数据
- 学习率从正常值退火到接近 0
- 数据切换为最高质量子集
- 通常占总训练的 1-5%
- LLaMA 3 报告此阶段显著提升 benchmark 性能

### 1.6 后训练数据

后训练包括监督微调（SFT）和人类偏好对齐（RLHF/DPO）。

**SFT 数据格式**：
```json
{
  "instruction": "将以下句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
```

**偏好数据格式**：
```json
{
  "prompt": "解释量子计算",
  "chosen": "量子计算是一种利用量子力学原理...",
  "rejected": "量子计算就是很快的计算机..."
}
```

**LIMA 的启示**：1000 条精心策划的 SFT 数据可以产生强大的对话模型。数据多样性比数量更重要，响应风格的一致性很关键。

## 2. 分布式训练框架

训练大规模语言模型需要跨多 GPU、多节点的分布式系统。

### 2.1 为什么需要分布式训练

现代 LLM 的参数量已达千亿级别：
- GPT-3：175B 参数，需要约 700GB 显存（FP32）
- DeepSeek-V3：671B 参数

单 GPU 显存有限（A100/H100: 80GB），必须将模型分布到多个设备上。

**训练显存占用分解**（以 Adam + FP16 混合精度为例）：

| 组件 | 精度 | 显存 |
|------|------|------|
| 模型参数 | FP16 | $2\Phi$ |
| 梯度 | FP16 | $2\Phi$ |
| 优化器状态（Adam） | | |
| - FP32 参数副本 | FP32 | $4\Phi$ |
| - 一阶矩 $m$ | FP32 | $4\Phi$ |
| - 二阶矩 $v$ | FP32 | $4\Phi$ |
| **总计（不含激活）** | | $16\Phi$ |

对于 7B 模型：$16 \times 7 \times 10^9 = 112$GB，超过单卡容量。

### 2.2 数据并行 (DP)

数据并行是最简单的分布式策略：

1. 每个 GPU 持有 **完整的模型副本**
2. 数据集被分割，每个 GPU 处理不同的 mini-batch
3. 前向传播独立进行
4. 反向传播后，**All-Reduce** 同步梯度
5. 每个 GPU 独立更新参数（结果相同）

**局限**：每个 GPU 必须能容纳完整模型，无法训练超大模型。

### 2.3 ZeRO：零冗余优化器

ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 的核心技术，通过分片消除数据并行中的内存冗余。

**三个阶段**：

| 阶段 | 分片内容 | 单 GPU 显存 | 通信量 |
|------|----------|-------------|--------|
| DDP | 无 | $16\Phi$ | $2\Phi$ |
| ZeRO-1 | 优化器状态 | $4\Phi + 12\Phi/N$ | $2\Phi$ |
| ZeRO-2 | + 梯度 | $2\Phi + 14\Phi/N$ | $2\Phi$ |
| ZeRO-3 | + 参数 | $16\Phi/N$ | $3\Phi$ |

**ZeRO-1**：将 Adam 的 $m, v$ 和 FP32 参数副本分片到 $N$ 个 GPU

$$\text{优化器显存}: 12\Phi \to \frac{12\Phi}{N}$$

**ZeRO-2**：梯度也按 $1/N$ 分片

**ZeRO-3**：模型参数也分片，前向/反向传播时按需 All-Gather

**ZeRO-Offload**：将优化器状态和计算卸载到 CPU，单 GPU 可训练 10B+ 模型

**ZeRO-Infinity**：进一步卸载到 NVMe SSD，512 GPU 可训练万亿参数模型

### 2.4 张量并行 (TP)

张量并行将单个层的参数矩阵切分到多个 GPU。

**MLP 层的张量并行**：

对于 FFN 层 $Y = \text{GeLU}(XW_1)W_2$：

- **列切分** $W_1$（沿输出维度）：每个 GPU 独立计算部分输出，无需通信
- **行切分** $W_2$（沿输入维度）：需要 All-Reduce 求和

**注意力层的张量并行**：多头注意力天然适合张量并行，将 $h$ 个头分配到 $N$ 个 GPU，每个 GPU 处理 $h/N$ 个头。

**通信开销**：每个 Transformer 层前向需要 2 次 All-Reduce（注意力 + MLP）

张量并行适合 **节点内** 高带宽互连（NVLink: 600GB/s）。

### 2.5 流水线并行 (PP)

流水线并行将模型按层切分到不同 GPU：
- GPU 0：Layer 0-7
- GPU 1：Layer 8-15
- ...

**朴素流水线的问题**：顺序执行导致严重的 **流水线气泡**（Pipeline Bubble）

$$\text{气泡比例} = \frac{p - 1}{m + p - 1}$$

其中 $p$ 是流水线阶段数，$m$ 是 micro-batch 数。

**1F1B 调度**：一个前向、一个反向交替执行
- 稳态时每个 GPU 同时有 1 个 micro-batch 在前向、1 个在反向
- 激活显存只需存储 $p$ 个 micro-batch

### 2.6 3D 并行

Megatron-LM 将 DP、TP、PP 组合成 **3D 并行**：

$$\text{总 GPU 数} = N_{DP} \times N_{TP} \times N_{PP}$$

**并行度选择原则**：
1. **TP 优先用于节点内**：NVLink 带宽高，TP 通信频繁
2. **PP 用于节点间**：通信量小（只传激活），可跨节点
3. **DP 扩展吞吐**：通信可与计算重叠

**配置示例**：

| 模型 | GPU 数 | TP | PP | DP |
|------|--------|----|----|----|
| GPT-3 175B | 1024 | 8 | 8 | 16 |
| LLaMA-70B | 64 | 8 | 2 | 4 |

### 2.7 PyTorch FSDP

Fully Sharded Data Parallel（FSDP）是 PyTorch 原生的 ZeRO-3 实现。

**核心特性**：
- **参数分片**：模型参数、梯度、优化器状态全部分片
- **按需 All-Gather**：前向/反向时临时恢复完整参数
- **与 torch.compile 兼容**：可获得额外加速

**分片策略**：
- `FULL_SHARD`：完全分片（类似 ZeRO-3）
- `SHARD_GRAD_OP`：只分片梯度和优化器（类似 ZeRO-2）
- `HYBRID_SHARD`：节点内分片，节点间复制

### 2.8 混合精度训练

混合精度训练使用低精度（FP16/BF16）加速计算，同时保持训练稳定性。

**数值格式对比**：

| 格式 | 位数 | 指数位 | 尾数位 | 动态范围 |
|------|------|--------|--------|----------|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ |
| FP16 | 16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ |
| FP8 | 8 | 4-5 | 2-3 | $\pm 448$ ~ $\pm 5.7 \times 10^{4}$ |

**BF16 的优势**：与 FP32 有相同的指数位，无需 Loss Scaling，训练更稳定。

**FP8 训练**：需要每张量缩放（Per-tensor Scaling），H100 上可比 BF16 快约 10%。

### 2.9 集合通信原语

| 原语 | 功能 | 应用场景 |
|------|------|----------|
| Broadcast | 一对多广播 | 参数初始化 |
| All-Reduce | 归约后广播 | DDP 梯度同步 |
| All-Gather | 收集所有分片 | ZeRO 参数恢复 |
| Reduce-Scatter | 归约后分片 | ZeRO 梯度分片 |
| All-to-All | 全交换 | MoE 专家通信 |

**Ring All-Reduce**：通信量 $2(N-1)/N \cdot D \approx 2D$，与 GPU 数 $N$ 无关，是带宽最优的算法。

## 3. Muon 优化器

Adam 优化器自 2014 年提出以来，一直是深度学习的标准选择。然而，Adam 本质上是 **逐元素**（element-wise）的优化器，没有利用神经网络参数的 **矩阵结构**。Muon 通过对动量进行矩阵正交化，实现了更高效的参数更新。

### 3.1 Adam 的局限

Adam 的更新规则：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

所有操作都是 **逐元素** 进行的。对于矩阵参数 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，Adam 将其展平为向量处理，完全忽略了矩阵的行列结构。

一个关键观察是：SGD 和 Adam 产生的梯度更新通常具有 **极高的条件数**，即接近低秩矩阵。更新主要沿少数"主方向"进行，而"稀有方向"被严重抑制。

### 3.2 Muon 的核心思想：正交化动量

Muon 的核心思想是将动量矩阵 $M$ 替换为其最近的半正交矩阵，即进行 **极分解**（Polar Decomposition）。

设 $M = U\Sigma V^\top$ 是 $M$ 的奇异值分解，则：

$$\text{msign}(M) = UV^\top$$

这称为矩阵的 **符号函数**（matrix sign function），类似于标量的 sign 函数将所有奇异值映射为 1。

**Muon 算法**：
1. 计算梯度 $G_t = \nabla_W \mathcal{L}(W_{t-1})$
2. 更新动量 $M_t = \beta M_{t-1} + G_t$
3. 计算正交化更新 $\Delta_t = \text{msign}(M_t)$
4. 更新参数 $W_t = W_{t-1} - \eta \cdot \Delta_t$

**为什么正交化有效？**

$\text{msign}(G) = UV^\top$ 将所有奇异值映射为 1：

$$G = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top \xrightarrow{\text{msign}} U \cdot \text{diag}(1, \ldots, 1) \cdot V^\top$$

效果：
- **主方向被抑制**：$\sigma_1 \to 1$，更新幅度降低
- **稀有方向被放大**：$\sigma_r \to 1$，更新幅度提升
- **方向信息保留**：$U, V$ 不变，只改变"步长"

**几何直觉**：想象损失函数的等高线是一个狭长的椭圆（高条件数）。Adam 沿梯度方向走，容易在窄谷中震荡。Muon 将椭圆"压成圆"——沿各方向走相同步长，这正是牛顿法的效果，但无需计算 Hessian。

### 3.3 Newton-Schulz 迭代

直接计算 SVD 的复杂度为 $O(\min(d_{out}, d_{in})^3)$，对于大矩阵不可接受。Muon 使用 **Newton-Schulz 迭代** 高效近似 $\text{msign}(M)$：

$$X_{k+1} = aX_k + b(X_k X_k^\top)X_k + c(X_k X_k^\top)^2 X_k$$

其中 $a = 3.4445$，$b = -4.7750$，$c = 2.0315$ 是优化过的系数。

```python
def newton_schulz5(G, steps=5, eps=1e-7):
    """近似计算 msign(G)"""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # 归一化

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T
    return X
```

实践中 **5 次迭代** 即可达到足够精度，计算开销通常 < 1%。

### 3.4 Muon 的四个版本

| 版本 | 缩放因子 | 特点 |
|------|----------|------|
| 朴素版 | $1$ | 最简单，但学习率不可迁移 |
| KellerJordan 版 | $\sqrt{\max(1, d_{out}/d_{in})}$ | 默认版本 |
| MuP 版 | $\sqrt{d_{out}/d_{in}}$ | 学习率可迁移 |
| Moonlight 版 | $0.2 \times \sqrt{\max(d_{out}, d_{in})}$ | 可直接沿用 Adam 学习率 |

### 3.5 哪些参数使用 Muon？

| 参数类型 | 优化器 | 原因 |
|----------|--------|------|
| 隐藏层 Linear 权重 | Muon | 核心矩阵参数 |
| Attention 的 $W_Q, W_K, W_V, W_O$ | Muon | 矩阵参数 |
| MLP 的 $W_{gate}, W_{up}, W_{down}$ | Muon | 矩阵参数 |
| Embedding 层 | AdamW | 本质是查找表 |
| LayerNorm 参数 | AdamW | 1D 向量 |
| Bias | AdamW | 1D 向量 |

### 3.6 大规模训练：Moonlight

月之暗面（Moonshot AI）在 Moonlight 模型中验证了 Muon 的大规模可扩展性。

**性能对比**：

| 指标 | Muon | AdamW |
|------|------|-------|
| 计算效率 | ~2× | 基准 |
| 达到相同性能所需 FLOPs | 52% | 100% |
| 样本效率 | 1.92× | 基准 |

**Moonlight 模型规格**：
- 总参数：15.29B（MoE 架构）
- 激活参数：2.24B
- 训练数据：5.7T tokens

### 3.7 实践指南

**超参数设置**：
- **动量系数**：$\beta = 0.95$（比 Adam 的 0.9 略大）
- **Newton-Schulz 步数**：5 步
- **学习率**：Moonlight 版直接用 Adam 学习率；其他版本乘以 $0.2\sqrt{d_{hidden}}$

**注意事项**：
1. Muon 只用于 2D 矩阵参数，其他参数用 AdamW
2. 正确识别框架中的 $d_{in}$ 和 $d_{out}$（PyTorch vs Keras 不同）
3. bfloat16 精度足够，无需 float32

## 4. 实践建议

### 4.1 分布式策略选择

**小模型（< 10B）**：
- 单卡能放下：使用 DDP
- 单卡放不下：使用 FSDP 或 ZeRO-2

**中等模型（10B - 100B）**：
- 单节点：FSDP/ZeRO-3 + 激活检查点
- 多节点：3D 并行（TP=8 节点内，PP 跨节点）

**超大模型（100B+）**：
- 必须使用 3D 并行
- 结合专家并行（MoE）
- 考虑 FP8 混合精度

### 4.2 数据工程建议

- **预训练**：投资于数据处理流水线，去重和过滤至关重要
- **SFT**：质量优先，人工审核每一条数据
- **偏好对齐**：确保标注一致性，避免噪声标签
- **持续改进**：建立数据飞轮，不断收集和迭代

## 5. 总结

本文全面解析了大模型训练的三大支柱：

| 领域 | 关键技术 | 代表工作 |
|------|----------|----------|
| 数据工程 | 去重、过滤、配比、课程学习 | FineWeb, LIMA |
| 分布式训练 | ZeRO、3D 并行、FSDP | DeepSpeed, Megatron |
| 优化器 | 矩阵正交化、Newton-Schulz | Muon, Moonlight |

这些技术共同支撑了千亿参数模型的高效训练，是大模型时代的基础设施。

下一篇我们将讨论 **评测与 Benchmark**，介绍如何全面评估模型能力。
