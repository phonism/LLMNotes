---
layout: post
title: "Speculative Decoding 全面解析：原理、方法与加速本质"
date: 2026-01-05 12:00:00
author: Qi Lu
tags: [LLM, Inference, Speculative Decoding, Efficiency]
lang: zh
translation: /en/speculative-decoding/
---

Speculative Decoding（推测解码）是近年来 LLM 推理加速领域最重要的技术之一。它通过"先猜测、后验证"的范式，在**不改变输出分布**的前提下实现 2-3 倍的推理加速。

本文将全面介绍 Speculative Decoding 的原理、各类实现方法、Draft Model 的获取方式，以及深入分析其加速本质。

---

## 1. 动机：自回归解码的瓶颈

### 1.1 为什么 LLM 推理慢？

LLM 采用自回归（Autoregressive）方式生成文本：每个 token 依赖前面所有 token，必须串行生成。

**两种解码模式**：

| 模式 | 公式 | 特点 |
|------|------|------|
| **Greedy** | $y_t = \arg\max_v P(v \mid y_{<t}, x)$ | 确定性，每步选概率最大的 token |
| **Sampling** | $y_t \sim P(\cdot \mid y_{<t}, x)$ | 随机性，按概率分布采样 |

无论哪种模式，**生成 K 个 token 都需要 K 次串行的前向传播**。

### 1.2 Memory-Bound 问题

现代 GPU 算力充沛，但 LLM 推理是 **memory-bound** 而非 compute-bound：

| 瓶颈 | 说明 |
|------|------|
| **权重读取** | 每步前向传播需要从 HBM 读取几乎全部模型权重 |
| **KV Cache** | 需要读取所有历史 token 的 KV Cache |
| **低并行度** | 每步只生成 1 个 token，GPU 利用率低 |

> 注：权重常驻 GPU 显存（HBM），但每次前向传播都需要将权重从 HBM 读入计算单元，这个读取带宽是瓶颈。

**核心矛盾**：GPU 有海量算力，但每步只做一个 token 的计算，大部分时间在等待显存读取。

### 1.3 关键洞察

> "Hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models."

大模型生成的很多 token 是"简单"的（如常见词、语法结构），小模型也能预测对。只有少数"困难" token 真正需要大模型的能力。

---

## 2. 核心机制

### 2.1 Draft-Then-Verify 范式

Speculative Decoding 的核心思想：

1. **Draft（草稿）**：用快速的 **Draft Model** 串行生成 $\gamma$ 个候选 token
2. **Verify（验证）**：用 **Target Model** 并行验证这些 token
3. **Accept/Reject（接受/拒绝）**：通过 rejection sampling 决定接受哪些 token

```
Draft Model:  [x] → t1 → t2 → t3 → t4 → t5  (γ=5, 串行)
                ↓    ↓    ↓    ↓    ↓
Target Model: [x, t1, t2, t3, t4, t5]        (并行验证)
                ↓    ↓    ↓    ↓    ↓
Result:       [✓]  [✓]  [✓]  [✗]  [—]       (接受3个 + 重采样1个)
```

### 2.2 Rejection Sampling 算法

设 Draft Model 分布为 $q(x)$，Target Model 分布为 $p(x)$：

**接受概率**：

$$P(\text{accept}) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

**两种情况**：

| 情况 | 条件 | 处理 |
|------|------|------|
| Draft 保守 | $q(x) \leq p(x)$ | 100% 接受 |
| Draft 过度自信 | $q(x) > p(x)$ | 以 $p(x)/q(x)$ 概率接受 |

**拒绝时的重采样**：

$$x \sim \text{norm}\left(\max(0, p(x) - q(x))\right)$$

### 2.3 输出不变性保证

Speculative Decoding 在两种解码模式下都保证输出不变：

**Greedy 模式**：
- Draft 和 Target 对同一位置给出相同的 argmax token → 直接接受
- 不同 → 拒绝，使用 Target 的结果
- **保证**：输出序列与纯 Target 解码**完全一致**（确定性）

**Sampling 模式**：
- 通过 rejection sampling 保证采样分布不变
- **定理**：通过 Speculative Sampling 从 $p(x)$ 和 $q(x)$ 采样得到的 token，与直接从 $p(x)$ 采样的分布**完全相同**

这意味着：
- 输出质量与原始 Target Model **完全一致**
- 加速是 **lossless** 的
- 不需要额外的质量-速度权衡

---

## 3. 加速本质分析

### 3.1 Memory-Bound：LLM 推理的本质瓶颈

理解 Speculative Decoding 的加速原理，首先要理解 LLM 推理为什么慢。

**推理时间的决定因素**：

$$T_{inference} = \max(T_{compute}, T_{memory})$$

- $T_{compute}$：GPU 计算所需时间
- $T_{memory}$：从 HBM 读取数据所需时间

**自回归解码的问题**：

每生成一个 token，需要：
1. 从 HBM 读取几乎全部模型权重
2. 读取 KV Cache
3. 执行矩阵乘法

**粗略估算**（Back-of-envelope，仅看数量级）：

```
假设：70B 模型，FP16，A100 80GB (2 TB/s 带宽, 312 TFLOPS)
忽略：KV Cache、activation、通信开销、kernel 调度

单 token 生成：
├── 显存读取：~140GB 权重 → O(100ms) 量级
└── 计算量：~140 GFLOPs → O(1ms) 量级

带宽 vs 算力：差两个数量级！
```

> ⚠️ 实际耗时受精度（FP16/INT8/INT4）、并行策略（TP/PP）、序列长度、kernel 融合等影响，上述仅为说明数量级差异。

**算术强度（Arithmetic Intensity）**：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

| 场景 | 算术强度 | 瓶颈 |
|------|----------|------|
| 自回归（batch=1） | ~1 FLOP/Byte | **Bandwidth** |
| 批处理（batch=128） | ~128 FLOP/Byte | Compute |
| A100 平衡点 | ~156 FLOP/Byte | - |

**关键洞察**：自回归解码的算术强度远低于 GPU 的平衡点，**绝大部分算力被浪费在等待显存读取**。

### 3.2 Speculative Decoding 的加速本质

**核心思想：用一次显存读取，完成多个 token 的计算**

```
传统自回归（生成 5 个 token）：
├── Step 1: 读取权重(140GB) + 计算(t1) → 70ms
├── Step 2: 读取权重(140GB) + 计算(t2) → 70ms
├── Step 3: 读取权重(140GB) + 计算(t3) → 70ms
├── Step 4: 读取权重(140GB) + 计算(t4) → 70ms
└── Step 5: 读取权重(140GB) + 计算(t5) → 70ms
Total: 5 × 70ms = 350ms，显存读取 700GB

Speculative Decoding（验证 5 个 token）：
├── Draft: 5 × 7ms = 35ms（小模型，可忽略）
└── Verify: 读取权重(140GB) + 计算(t1,t2,t3,t4,t5) → ~75ms
Total: 110ms，显存读取 154GB（假设 Draft 10% 大小）
```

**为什么验证 5 个 token 只比 1 个 token 慢一点？**

| 操作 | 1 token | 5 tokens | 增长 |
|------|---------|----------|------|
| 读取权重 | 140GB | 140GB | **1x**（不变！）|
| KV Cache 读取 | K bytes | K bytes | ~1x |
| 计算量 | 140 GFLOPs | 700 GFLOPs | 5x |
| **总时间** | ~70ms | **~75ms** | **1.07x** |

**本质**：
- 权重只读取一次，这是大头
- 计算量虽然增加 5 倍，但计算时间本来就可以忽略
- 显存带宽是瓶颈，计算是"免费"的

### 3.3 加速比分析

**理论加速比**：

假设：
- Target Model 单 token 时间 $T_t$
- Draft Model 单 token 时间 $T_d$（通常 $T_d \ll T_t$）
- 每轮生成 $\gamma$ 个草稿 token
- 平均接受 $k$ 个 token

$$\text{Speedup} = \frac{k \cdot T_t}{\gamma \cdot T_d + T_t} \approx \frac{k \cdot T_t}{T_t} = k$$

当 $T_d \ll T_t$ 时，**加速比约等于平均接受的 token 数**。

**期望接受 token 数**（假设每个位置独立接受概率为 $\alpha$）：

$$\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

| 接受率 $\alpha$ | $\gamma=5$ 时期望接受数 | 实际加速比 |
|----------------|------------------------|-----------|
| 0.5 | 1.97 | ~2x |
| 0.7 | 3.16 | ~3x |
| 0.9 | 4.69 | ~4x |

### 3.4 什么情况下效果好？

从 Memory-Bound 本质出发，可以推导出：

**效果好的条件**：

| 条件 | 原因 |
|------|------|
| Target Model 足够大（≥30B） | 越大越 memory-bound，计算"免费"的部分越多 |
| Draft 与 Target 分布接近 | 高接受率，每轮能确认更多 token |
| 输出可预测性强 | 代码、格式化文本、翻译等任务接受率高 |

**效果差的条件**：

| 条件 | 原因 |
|------|------|
| Target Model 较小（<7B） | 接近 compute-bound，验证多 token 开销变大 |
| Draft 与 Target 差异大 | 低接受率，频繁重采样，加速比下降 |
| 高创造性任务 | 输出不可预测，接受率低 |

**直观理解**：Speculative Decoding 本质是"用小模型的多次显存读取，换 Target 的一次显存读取"。如果 Target 不够大、不够 memory-bound，这个交换就不划算。

---

## 4. Draft Model 获取方式

### 4.1 独立小模型

最直接的方式：使用同系列的小模型作为 Draft。

| Target Model | Draft Model | 参数比 | 来源 |
|--------------|-------------|--------|------|
| Llama-70B | Llama-7B | 10:1 | 同系列小模型 |
| Chinchilla-70B | Chinchilla-1B | 70:1 | DeepMind 原始实验 |
| T5-XXL (11B) | T5-small (60M) | 183:1 | Google 原始实验 |

**选择原则**：
- Draft 应该比 Target 快 10-100 倍
- 同系列模型分布更接近，接受率更高
- 太小的 Draft 接受率低，太大的 Draft 开销高

**优点**：无需额外训练，开箱即用
**缺点**：分布差异可能较大，接受率受限

---

### 4.2 知识蒸馏

通过知识蒸馏让 Draft Model 更好地拟合 Target Model 的输出分布。

#### 4.2.1 DistillSpec（ICLR 2024）

**论文**: [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461)

**核心问题**：现成的小模型与 Target 分布差异大，导致接受率低。

**两个关键设计**：

1. **On-Policy 数据生成**：
   - 使用 Draft Model 自己生成的数据进行训练
   - 而非使用固定数据集
   - 原因：Draft 需要在自己可能生成的 token 上对齐

2. **任务定制的散度函数**：
   - 不同任务/解码策略使用不同的 KL 散度变体
   - Greedy decoding：Forward KL
   - Sampling：Reverse KL 或 JSD

**训练流程**：
```
1. Draft Model 生成候选序列
2. Target Model 计算这些位置的概率分布
3. 最小化 Draft 与 Target 分布的散度
4. 重复直到收敛
```

**效果**：
- 相比标准 SD 提升 **10-45%** 加速
- XSum 任务：6.4x 延迟降低
- GSM8K 任务：10.7x 延迟降低

#### 4.2.2 AdaSPEC（2025）

**核心改进**：选择性 token 过滤

**观察**：有些 token 天然难以预测（如专有名词、低频词），强行对齐反而损害简单 token 的预测。

**方法**：
1. 使用参考模型识别"困难" token
2. 在蒸馏时过滤掉这些 token
3. 让 Draft 专注于对齐"简单" token

**效果**：接受率比 DistillSpec 提升最高 **15%**

---

### 4.3 Self-Speculative Decoding

不使用独立 Draft Model，而是从 Target Model 自身派生，实现"自己给自己打草稿"。

#### 4.3.1 LayerSkip（ACL 2024）

**论文**: [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
**代码**: [GitHub](https://github.com/facebookresearch/LayerSkip)

**核心思想**：跳过后面的层，用前 E 层的输出直接预测 token。

**三阶段方案**：

**阶段 1：训练时 Layer Dropout**
```python
# 训练时对不同层使用不同 dropout 率
# 浅层：低 dropout（保持稳定）
# 深层：高 dropout（增强早退能力）
for layer_idx, layer in enumerate(layers):
    dropout_rate = layer_idx / num_layers * max_dropout
    x = layer(x, dropout=dropout_rate)
```

**阶段 2：Early Exit Loss**
- 所有层共享同一个 LM Head
- 训练时在每一层都计算 loss
- 让浅层也具备预测能力

**阶段 3：Self-Speculative Decoding**
```
Self-Draft:   前 E 层 → LM Head → draft tokens
Self-Verify:  剩余层验证 + 完整前向传播
关键优化:     验证时复用 draft 阶段的 KV Cache
```

**使用示例**：
```bash
torchrun generate.py --model facebook/layerskip-llama2-7B \
    --generation_strategy self_speculative \
    --exit_layer 8 \          # 第8层退出做draft
    --num_speculations 6      # 每轮生成6个草稿token
```

**效果**：
- CNN/DM 摘要：2.16x 加速
- 代码生成：1.82x 加速
- 语义解析：2.0x 加速

**优势**：
- 只需一个模型，内存占用不增加
- Draft 与 Target 天然对齐（同一个模型）
- 部分计算可复用

**集成状态**：已集成到 HuggingFace Transformers 和 PyTorch TorchTune。

---

### 4.4 Additional Heads（额外预测头）

在 Target Model 上添加轻量级预测头，不改变原模型，只添加新组件。

#### 4.4.1 Medusa（ICML 2024）

**论文**: [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
**代码**: [GitHub](https://github.com/FasterDecoding/Medusa)

**核心思想**：添加多个"Medusa Head"，每个 head 预测不同位置的未来 token。

**架构**：
```
                    ┌─→ Head 1 → 预测 t+1
Hidden State (t) ───┼─→ Head 2 → 预测 t+2
    from LLM        ├─→ Head 3 → 预测 t+3
                    └─→ Head 4 → 预测 t+4
```

**Tree Attention 机制**：

由于每个 head 可能有多个候选（top-k），组合起来形成候选树：

```
假设 Head 1 取 top-2，Head 2 取 top-3：
候选树有 2 × 3 = 6 条路径

        t1
       / \
      t1a t1b
     /|\  /|\
    ...  ...
```

**Tree Attention 实现**：
- 使用特殊的 attention mask，每个 token 只能看到其祖先
- 单次前向传播同时处理所有候选路径
- 预处理 attention mask 提高效率

**两种训练模式**：

| 模式 | 训练方式 | 效果 |
|------|----------|------|
| **Medusa-1** | 冻结 LLM，只训练 heads | 2.2x 加速，无损 |
| **Medusa-2** | 联合微调 LLM + heads | 2.3-3.6x 加速 |

**Medusa-2 特殊训练 Recipe**：
- 需要平衡原始能力保持和推测能力获取
- 使用渐进式训练策略

**实测数据**：
- Medusa heads 预测下一个 token 的 top-1 准确率约 **60%**
- 但 top-5 准确率超过 **80%**
- 因此使用 tree structure 可以显著提升接受率

#### 4.4.2 EAGLE / EAGLE-3

**EAGLE 论文**: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) (ICML 2024)
**EAGLE-3 论文**: [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) (NeurIPS 2025)
**代码**: [GitHub](https://github.com/SafeAILab/EAGLE)

**核心洞察**：

1. **Feature-level 自回归比 Token-level 更容易**
   - Token 空间离散、稀疏
   - Feature 空间连续、平滑
   - 在 feature 空间预测更稳定

2. **Feature 不确定性是瓶颈**
   - Token 采样结果对下一步 feature 有影响
   - 但 draft 阶段看不到真实采样结果

**EAGLE 架构**：
```
Target LLM:
Input → [...Layers...] → Top Layer Feature → LM Head → Token
                              ↓
EAGLE Draft Head:        Feature + Token(t-1)
                              ↓
                         Predict Feature(t+1)
                              ↓
                         LM Head → Draft Token
```

**关键设计**：
- 复用 Target Model 的 top-layer feature
- Draft Head 输入：当前 feature + 上一步采样的 token
- Draft Head 输出：下一步的 feature 预测
- 使用 Target 的 LM Head 将 feature 转为 token

**训练细节**：
- 数据：ShareGPT，68,000 对话
- 学习率：3e-5
- 优化器：AdamW (β1=0.9, β2=0.95)
- 梯度裁剪：0.5
- Loss：$L = L_{reg} + w_{cls} L_{cls}$

**EAGLE 参数量**（相对 Target）：

| Target Size | EAGLE Params | 比例 |
|-------------|--------------|------|
| 7B | 0.24B | 3.4% |
| 13B | 0.37B | 2.8% |
| 33B | 0.56B | 1.7% |
| 70B | 0.99B | 1.4% |

**EAGLE-3 改进**（[arXiv:2503.01840](https://arxiv.org/abs/2503.01840), NeurIPS 2025）：
- 放弃 feature 预测，改为直接 token 预测
- 多层 feature 融合（Training-Time Test, TTT）
- 支持更大规模训练数据
- 性能显著提升

**EAGLE-3 架构改进**：
```
EAGLE-2: Feature(t) + Token(t-1) → Predict Feature(t+1)
EAGLE-3: Multi-layer Features + Token(t-1) → Direct Token Prediction
```

关键改进点：
1. **TTT（Training-Time Test）**：融合多层 feature，而非只用 top-layer
2. **简化预测目标**：直接预测 token 而非 feature，降低学习难度
3. **更好的泛化**：在分布外数据上表现更稳定

**效果**：LLaMA2-Chat 70B 上实现 **2.7-3.5x** 延迟加速。

**SpecForge：EAGLE-3 训练框架**

[SpecForge](https://github.com/lmsys-ai/specforge) 是 LMSYS 开源的 EAGLE-3 训练框架，支持高效训练各种规模的 draft model。

**两种训练模式**：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Online** | Target 和 Draft 同时运行，实时生成特征 | GPU 充足，追求最佳效果 |
| **Offline** | 预先生成 Target 特征，Draft 离线训练 | GPU 受限，大规模训练 |

**Online 模式**：
```bash
# 使用 FSDP 训练
python -m specforge.train \
    --target_model meta-llama/Llama-3.1-70B-Instruct \
    --mode online \
    --backend fsdp \
    --data_path train_data.jsonl
```

**Offline 模式**：
```bash
# Step 1: 预生成特征
python -m specforge.generate_features \
    --target_model meta-llama/Llama-3.1-70B-Instruct \
    --output_path features/

# Step 2: 离线训练
python -m specforge.train \
    --mode offline \
    --feature_path features/
```

**支持的后端**：
- **FSDP**：PyTorch 原生分布式，适合多 GPU
- **Tensor Parallel**：模型并行，适合超大模型
- **vLLM**：利用 vLLM 高效推理生成特征

---

### 4.5 Draft-Free 方法

完全不使用 Draft Model，通过算法创新实现并行解码。

#### 4.5.1 Lookahead Decoding（ICML 2024）

**论文**: [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)
**代码**: [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding)

**核心思想**：将自回归解码视为求解非线性方程组，用 Jacobi 迭代并行求解。

**Jacobi 解码的问题**：

传统 Jacobi 解码对 LLM 几乎没有加速（约 1.05x），因为：
- LLM 训练时是自回归的
- 给定错误的前缀，几乎不可能预测对后续 token
- 每轮 Jacobi 迭代通常只修正 1 个 token

**Lookahead 的解决方案**：

虽然单次 Jacobi 迭代只能确定 1 个 token，但过程中会产生有价值的 **n-gram 副产品**。

**2D 窗口设计**：

```
维度 1: Window Size W (向前看多少个位置)
维度 2: N-gram Size N (向后看多少步历史)

     时间轴 (Jacobi 迭代步)
        t-3  t-2  t-1   t
位置 1   a    b    c    d  ← 可提取 4-gram: abcd
位置 2   e    f    g    h
位置 3   i    j    k    l
位置 4   m    n    o    p
  ↑
序列轴
```

**两个并行分支**：

1. **Lookahead Branch**：
   - 维护 2D 窗口
   - 每步更新窗口中所有位置的预测
   - 从轨迹中收集 n-gram 存入候选池

2. **Verification Branch**：
   - 从候选池中选择首 token 匹配的 n-gram
   - 并行验证这些候选
   - 接受最长的有效前缀

**算法流程**：
```python
# 配置参数
lade.config_lade(
    LEVEL=5,           # N-gram size N
    WINDOW_SIZE=7,     # 窗口大小 W
    GUESS_SET_SIZE=7,  # 候选池大小 G
)

# 每步解码：
for step in decoding:
    # 1. Lookahead branch: 并行生成 W 个位置的预测
    # 2. 收集新产生的 n-gram 到候选池
    # 3. Verification branch: 验证匹配的 n-gram
    # 4. 接受最长有效前缀，更新窗口
```

**效果**：
- MT-bench：1.8x 加速
- 代码生成（多 GPU）：4x 加速
- 无需任何额外模型或数据

#### 4.5.2 Prompt Lookup Decoding

**核心思想**：从 prompt 中查找与当前生成匹配的 n-gram。

**适用场景**：
- 摘要任务：输出常包含输入片段
- 编辑任务：大部分内容保持不变
- 代码补全：变量名、函数名重复出现

**实现**：
```python
# vLLM 配置
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[ngram]",        # 启用 n-gram lookup
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,          # 最大 4-gram
)
```

**效果**：vLLM 实测摘要任务 **2.8x** 加速。

**优点**：零额外开销，无需训练
**局限**：仅在 prompt 与输出有重叠时有效

---

## 5. 验证策略

### 5.1 Sequential Verification（顺序验证）

原始方法：从左到右逐个验证，遇到第一个拒绝就停止。

```
Draft:  t1  t2  t3  t4  t5
        ✓   ✓   ✗   -   -
Accept: t1, t2 + resample t3'
```

**问题**：一个拒绝会浪费后续所有草稿。

### 5.2 Tree-based Verification（树形验证）

**SpecInfer**（ASPLOS 2024）：

将候选组织为 **Token Tree**，而非线性序列：

```
        t1
       / | \
      t2 t2' t2''
     /|   |
    t3 t3' t3
```

**优势**：
- 显著提高验证成功率（52-57% → 96-97%）
- 单次前向传播验证整棵树
- 1.5-2.8x（分布式）/ 2.6-3.5x（offloading）加速

**实现**：使用 Tree Attention 并行处理所有路径。

### 5.3 Block Verification（块验证）

**观察**：逐 token 独立验证并非最优。

**Block Verification**：
- 联合验证整个 block
- 利用 token 间的统计依赖
- 比独立验证接受更多 token

---

## 6. 主要方法总结

### 6.1 开创性工作

| 工作 | 时间 | 机构 | 贡献 |
|------|------|------|------|
| **Fast Inference via Speculative Decoding** | 2022-11 | Google | 首次提出 Speculative Decoding |
| **Speculative Sampling** | 2023-02 | DeepMind | 独立提出，Chinchilla 2-2.5x |

### 6.2 Draft Model 改进

| 工作 | 核心思想 | 效果 |
|------|----------|------|
| **DistillSpec** | 知识蒸馏对齐 Draft | +10-45% |
| **Online Speculative Decoding** | 在线更新 Draft | 适应分布偏移 |
| **Draft & Verify** | Self-speculative | 无需额外模型 |
| **LayerSkip** | 层跳过 | 复用计算 |

### 6.3 Additional Heads

| 工作 | 核心思想 | 加速 |
|------|----------|------|
| **Medusa** | 多解码头 + 树注意力 | 2.2-3.6x |
| **EAGLE** | Feature-level 预测头 | 2-3x |
| **EAGLE-3** | 训练时测试优化 | SOTA |
| **Hydra** | 多头变体 | - |

### 6.4 验证优化

| 工作 | 核心思想 | 效果 |
|------|----------|------|
| **SpecInfer** | Token Tree + Tree Attention | 2.6-3.5x |
| **Block Verification** | 联合块验证 | 更高接受率 |
| **Staged Speculative Decoding** | 多级验证 | - |

### 6.5 Draft-Free

| 工作 | 核心思想 | 加速 |
|------|----------|------|
| **Lookahead Decoding** | Jacobi 迭代 + n-gram 缓存 | 1.5-2.3x |
| **Prompt Lookup** | 从 prompt 查找 n-gram | 2.8x（摘要） |
| **REST** | 检索增强 | - |

---

## 7. 实际部署

### 7.1 框架支持

| 框架 | 支持的方法 |
|------|-----------|
| **vLLM** | Draft model, Prompt lookup, Medusa, EAGLE |
| **TensorRT-LLM** | Draft model, Medusa |
| **SGLang** | Draft model, EAGLE |
| **HuggingFace** | Assisted generation |

### 7.2 vLLM 使用示例

```python
from vllm import LLM, SamplingParams

# Draft model-based
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
)

# Prompt lookup (无需 draft model)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)
```

### 7.3 何时使用？

**推荐使用**：
- Target Model ≥ 30B 参数
- 输出可预测性较高（代码、格式化、翻译）
- 延迟敏感场景
- 有合适的 Draft Model

**不推荐使用**：
- Target Model 较小（7B 以下）
- 高度创造性任务
- 没有合适的 Draft Model 且无法使用 draft-free 方法

### 7.4 SpecBundle：生产级 EAGLE-3 模型

[SpecBundle](https://github.com/lmsys/spec-bundle) 是 LMSYS 使用 SpecForge 训练并开源的生产级 EAGLE-3 draft model 集合。

**Phase 1 发布（2025-12）**：

| Target Model | Draft Model | 加速比 | 模型链接 |
|--------------|-------------|--------|----------|
| Llama-3.1-8B-Instruct | EAGLE-3 | 2.5-3.0x | [HuggingFace](https://huggingface.co/lmsys/Llama-3.1-8B-Instruct-EAGLE3) |
| Llama-3.1-70B-Instruct | EAGLE-3 | 3.0-4.0x | [HuggingFace](https://huggingface.co/lmsys/Llama-3.1-70B-Instruct-EAGLE3) |
| Llama-3.3-70B-Instruct | EAGLE-3 | 3.0-4.0x | [HuggingFace](https://huggingface.co/lmsys/Llama-3.3-70B-Instruct-EAGLE3) |
| Qwen2.5-7B-Instruct | EAGLE-3 | 2.5-3.0x | [HuggingFace](https://huggingface.co/lmsys/Qwen2.5-7B-Instruct-EAGLE3) |
| Qwen2.5-32B-Instruct | EAGLE-3 | 2.8-3.5x | [HuggingFace](https://huggingface.co/lmsys/Qwen2.5-32B-Instruct-EAGLE3) |
| Qwen2.5-72B-Instruct | EAGLE-3 | 3.0-4.0x | [HuggingFace](https://huggingface.co/lmsys/Qwen2.5-72B-Instruct-EAGLE3) |
| DeepSeek-V3 | EAGLE-3 | 3.0-3.5x | [HuggingFace](https://huggingface.co/lmsys/DeepSeek-V3-EAGLE3) |
| Gemma-2-9B-it | EAGLE-3 | 2.5-3.0x | [HuggingFace](https://huggingface.co/lmsys/Gemma-2-9B-it-EAGLE3) |
| Gemma-2-27B-it | EAGLE-3 | 2.8-3.5x | [HuggingFace](https://huggingface.co/lmsys/Gemma-2-27B-it-EAGLE3) |
| Mistral-7B-Instruct-v0.3 | EAGLE-3 | 2.5-3.0x | [HuggingFace](https://huggingface.co/lmsys/Mistral-7B-Instruct-v0.3-EAGLE3) |
| Mistral-Large-2 | EAGLE-3 | 3.0-3.5x | [HuggingFace](https://huggingface.co/lmsys/Mistral-Large-2-EAGLE3) |

**特点**：
- **生产就绪**：经过充分测试，可直接用于生产环境
- **广泛覆盖**：支持主流开源模型系列
- **显著加速**：大模型（70B+）可达 **4x** 加速
- **即插即用**：与 vLLM、SGLang 无缝集成

**使用示例（vLLM）**：

```python
from vllm import LLM, SamplingParams

# 使用 SpecBundle 的 EAGLE-3 模型
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="lmsys/Llama-3.1-70B-Instruct-EAGLE3",
    num_speculative_tokens=5,
)

# 正常推理
output = llm.generate(
    ["Explain quantum computing in simple terms."],
    SamplingParams(temperature=0.7, max_tokens=512)
)
```

**vLLM Speculators 训练支持**：

vLLM v0.3.0 起通过 [vllm-speculators](https://github.com/vllm-project/vllm-speculators) 项目提供端到端 EAGLE-3 训练支持：

```bash
# 安装
pip install vllm-speculators

# 训练 EAGLE-3
python -m vllm_speculators.train_eagle3 \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --output_path ./my-eagle3-model \
    --data_path train_data.jsonl
```

---

## 8. 方法分类总结

| 类别 | 核心思想 | 代表工作 |
|------|----------|----------|
| **独立 Draft** | 使用小模型生成草稿 | Google/DeepMind 原始方法, DistillSpec |
| **Self-Speculative** | 从 Target 自身派生 Draft | LayerSkip, Draft&Verify, SPEED |
| **Additional Heads** | 添加预测头到 Target | Medusa, EAGLE, Hydra |
| **Tree Verification** | 树形候选 + 并行验证 | SpecInfer |
| **Draft-Free** | 不使用 Draft Model | Lookahead, Prompt Lookup, REST |

---

## 参考文献

### 开创性工作
- Fast Inference from Transformers via Speculative Decoding: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Accelerating LLM Decoding with Speculative Sampling: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

### Draft Model 改进
- DistillSpec: [arXiv:2310.08461](https://arxiv.org/abs/2310.08461)
- Online Speculative Decoding: [arXiv:2310.07177](https://arxiv.org/abs/2310.07177)
- Draft & Verify: [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)

### Additional Heads
- Medusa: [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- EAGLE: [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) (ICML 2024)
- EAGLE-3: [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) (NeurIPS 2025)

### 验证优化
- SpecInfer: [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
- Block Verification: [OpenReview](https://openreview.net/forum?id=frsg32u0rO)

### Draft-Free
- Lookahead Decoding: [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)

### 综述
- Comprehensive Survey of Speculative Decoding: [arXiv:2401.07851](https://arxiv.org/abs/2401.07851)
- Decoding Speculative Decoding: [arXiv:2402.01528](https://arxiv.org/abs/2402.01528)

### 训练生态
- SpecForge: [GitHub](https://github.com/lmsys-ai/specforge)
- SpecBundle: [LMSYS Blog](https://lmsys.org/blog/2025-12-23-spec-bundle-phase-1/)
- vLLM Speculators: [GitHub](https://github.com/vllm-project/vllm-speculators)

### 资源
- Speculative Decoding Papers: [GitHub](https://github.com/hemingkx/SpeculativeDecodingPapers)
- vLLM Speculative Decoding: [Blog](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- Google Research Blog: [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
