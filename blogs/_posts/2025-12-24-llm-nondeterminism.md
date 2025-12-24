---
layout: post
title: "LLM 推理的非确定性之谜：从浮点幻觉到 Batch Invariance"
date: 2025-12-24 12:00:00
author: Qi Lu
tags: [LLM, Inference, Determinism, Batch Invariance, Reproducibility]
lang: zh
---

## 引言

可复现性是科学研究的基石。然而，当我们试图从大型语言模型中获取可复现的结果时，却面临着一个令人困惑的现象：即使将采样温度设置为 0（理论上应当进行确定性的贪婪解码），模型的输出仍然可能发生变化。

这不是偶发现象。使用 Qwen3-235B 模型在 temperature=0 条件下对同一 prompt 采样 1000 次，竟然产生了 80 种不同的输出，其中最频繁的输出仅出现 78 次。这一结果与"贪婪解码必然确定"的直觉形成了鲜明对比。

2025 年 9 月，由前 OpenAI CTO Mira Murati 创立的 Thinking Machines Lab 发布了一篇研究论文，系统性地分析了这一问题的根源，并提出了工程化的解决方案。本文将深入解析这一研究，探讨 LLM 推理非确定性的真正来源及其解决路径。

---

## 常见误解：浮点非关联性假说

### 浮点运算的数学特性

在讨论非确定性来源之前，有必要回顾浮点运算的一个基本特性：**非关联性**（non-associativity）。

在理想的实数运算中，加法满足结合律：

$$(a + b) + c = a + (b + c)$$

然而，由于浮点数的有限精度和舍入误差，这一等式在计算机中并不总是成立。考虑以下示例：

```python
a = 1e-10
b = 1.0
c = -1.0
print((a + b) + c)  # 输出: 1.0000000000000002e-10
print(a + (b + c))  # 输出: 1e-10
```

当 GPU 执行大规模并行计算时，不同线程完成的顺序可能不同，导致累加操作的顺序发生变化，进而产生微小的数值差异。

### 主流假说的局限性

基于上述观察，业界普遍持有一种假说：LLM 推理的非确定性源于 **"并发执行 + 浮点非关联性"** 的组合效应。具体而言：

1. GPU 的并行线程以不确定的顺序完成计算
2. 原子加法（atomic add）操作的执行顺序不可预测
3. 不同的执行顺序导致不同的舍入误差累积
4. 最终 logits 产生微小差异，在贪婪解码时可能选择不同的 token

这一假说看似合理，却忽略了现代 Transformer 实现的一个关键事实：**主流 LLM 推理引擎中的大多数核心操作实际上使用的是确定性的 reduction trees，而非原子操作**。

Thinking Machines Lab 的研究明确指出：

> "Although this can lead to nondeterministic kernels, concurrency (and atomic adds) end up being completely uninvolved in LLM inference nondeterminism."

---

## 真正的元凶：Batch Size 的隐形影响

### 动态批处理的数值后果

现代 LLM 推理服务为了最大化 GPU 利用率，普遍采用**动态批处理**（dynamic batching）策略：将多个用户请求合并为一个批次进行并行计算。批次大小取决于当前系统负载，因此具有不可预测性。

问题在于：**许多核心计算 kernel 的数值输出会随批次大小变化而变化**。即使单个样本的计算逻辑相同，其数值路径也会因批次中其他样本的存在而发生改变。

这一现象被称为 **Batch Invariance 的缺失**，它构成了 LLM 推理非确定性的主要来源。

```
┌─────────────────────────────────────────────────────────────┐
│                    Dynamic Batching Server                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Request A ──┐                                             │
│               │     ┌───────────┐                           │
│   Request B ──┼────▶│ Batch = 3 │────▶ Different numerical  │
│               │     └───────────┘      path for A           │
│   Request C ──┘                                             │
│                                                             │
│   ─────────────────────────────────────────────────────     │
│                                                             │
│   Request A ──┐     ┌───────────┐                           │
│               ├────▶│ Batch = 2 │────▶ Different numerical  │
│   Request D ──┘     └───────────┘      path for A           │
│                                                             │
│   Same request A, different batch contexts,                 │
│   potentially different outputs                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

*图 1：动态批处理导致相同请求在不同批次上下文中产生不同数值路径*

值得强调的是，这一问题并非 GPU 特有。**CPU 和 TPU 上的推理服务同样存在因批次大小变化导致的非确定性**，因为问题根源在于 kernel 的数值计算方式，而非特定硬件的并发特性。

### 三个关键操作的深入分析

Thinking Machines Lab 的研究表明，要实现 batch-invariant 的 Transformer 推理，需要关注三个核心操作：

#### 1. RMSNorm（Root Mean Square Normalization）

RMSNorm 是 LLM 中广泛使用的归一化层。其计算涉及对隐藏维度的求和操作：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

当批次大小变化时，GPU kernel 的 tiling 策略可能发生改变，导致求和的分块方式不同，进而产生不同的舍入误差。

#### 2. 矩阵乘法（Matrix Multiplication）

矩阵乘法是 Transformer 中计算量最大的操作，涉及大规模的点积累加。不同的 tiling 和 reduction 策略会导致不同的数值结果：

```
Standard MatMul: C = A × B

Tiling affects reduction order:
┌─────────────────────────────────────┐
│  Tile 1    Tile 2    Tile 3         │
│  ┌───┐    ┌───┐    ┌───┐           │
│  │   │    │   │    │   │           │
│  │ A │ ×  │ B │ =  │ C │           │
│  │   │    │   │    │   │           │
│  └───┘    └───┘    └───┘           │
│                                     │
│  Different batch sizes may trigger  │
│  different tiling configurations    │
└─────────────────────────────────────┘
```

#### 3. Attention 机制

Attention 的计算包含 softmax 归一化和加权求和，两者都涉及 reduction 操作：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

FlashAttention 等高效实现采用了复杂的 tiling 策略来优化内存访问，但这也引入了对批次大小的敏感性。

### 微小差异的雪崩效应

一个自然的问题是：如此微小的数值差异（通常在 $10^{-6}$ 到 $10^{-8}$ 量级）如何导致完全不同的输出？

答案在于贪婪解码的**离散性**和**自回归特性**：

1. **离散选择的脆弱性**：当两个 token 的概率非常接近时（例如 0.1500001 vs 0.1499999），微小的 logits 变化可能翻转 argmax 的结果
2. **误差的级联传播**：一旦选择了不同的 token，后续的整个生成序列都会发生改变
3. **长序列的累积效应**：序列越长，发生分歧的概率越高

```
Token probability example:

Logits (Run 1):  [2.3456789, 2.3456788, 1.2, 0.8, ...]
                      ↑ winner
Logits (Run 2):  [2.3456787, 2.3456790, 1.2, 0.8, ...]
                            ↑ winner (different!)

A difference of ~3e-7 in logits flips the argmax,
leading to entirely different generation paths.
```

*图 2：微小的 logits 差异导致贪婪解码选择不同的 token*

---

## 解决方案：Batch-Invariant Kernels

### 设计原则

实现 batch-invariant 推理的核心原则是：**对于每个涉及 reduction 的操作，选择一个固定的 tiling 和指令组合，使其 reduction 顺序与批次大小和序列分块无关**。

具体而言：

1. **固定 reduction 树结构**：无论批次大小如何变化，始终使用相同的求和顺序
2. **固定 split 大小**：在 attention 的 decoding 阶段，使用固定的 KV split 大小
3. **避免批次间的数据依赖**：确保每个样本的计算路径独立于同批次的其他样本

### 实现架构

Thinking Machines Lab 开源了 [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) 库，提供了以下 batch-invariant 实现：

| 操作 | 标准实现 | Batch-Invariant 实现 |
|------|----------|---------------------|
| RMSNorm | 动态 tiling | 固定 reduction 顺序 |
| MatMul | cuBLAS（动态优化）| 固定 tiling 配置 |
| Softmax | 动态 block size | 固定 reduction 树 |
| Attention | FlashAttention（动态 split）| 固定 split-KV 大小 |

该库通过 `torch.Library` 机制替换 PyTorch 的默认 kernel，对现有模型代码的侵入性极低：

```python
import batch_invariant_ops  # 导入即生效

# 现有模型代码无需修改
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
output = model.generate(input_ids, temperature=0)
# 现在输出是确定性的
```

### 与推理引擎的集成

#### vLLM 集成

vLLM 已原生支持 batch-invariant 模式，可通过环境变量启用：

```bash
VLLM_BATCH_INVARIANT=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B
```

vLLM 的实现基于 FlexAttention 后端，结合 `torch.Library` 实现了对大多数相关操作的透明替换。

#### SGLang 集成

SGLang 在 Thinking Machines Lab 工作的基础上进行了进一步优化，实现了：

- 多后端支持：FlashInfer、FlashAttention 3、Triton
- 与 chunked prefill、radix cache 的兼容
- 支持 temperature > 0 的确定性采样（通过 per-request seed）

```python
from sglang import RuntimeEndpoint

runtime = RuntimeEndpoint(
    model_path="Qwen/Qwen3-8B",
    deterministic=True  # 启用确定性模式
)
```

---

## 工程实践与性能权衡

### 性能开销分析

确定性推理不可避免地带来性能开销，因为它放弃了部分动态优化策略：

| 实现方案 | 性能开销 | 备注 |
|----------|----------|------|
| Thinking Machines Lab 原始实现 | ~61.5% | 基准实现 |
| SGLang + CUDA Graphs | ~34.35% | 2.8x 加速 |
| vLLM batch-invariant 模式 | ~40-50% | 视硬件和模型而定 |

CUDA Graphs 通过预编译和缓存 GPU 执行图，显著减少了 kernel launch 开销，是优化确定性推理性能的关键技术。

```
Performance comparison (relative to non-deterministic baseline):

Non-deterministic:  ████████████████████████████████████████ 100%
TML Original:       ████████████████                         38.5%
SGLang + CUDA:      ██████████████████████████               65.7%
```

*图 3：不同实现方案的相对性能对比*

### 硬件要求

当前的 batch-invariant 实现对硬件有一定要求：

- **vLLM batch-invariant 模式**：需要 NVIDIA GPU，compute capability ≥ 9.0（H100/H800）
- **SGLang 确定性模式**：支持更广泛的硬件，但最佳性能仍需 Hopper 架构

### 适用场景

确定性推理并非所有场景都需要。以下场景值得付出性能代价：

1. **强化学习训练**：RL 训练需要可复现的 rollout 来保证训练稳定性
2. **模型调试与测试**：确定性输出使得 bug 可复现，便于定位问题
3. **安全审计**：可审计的 AI 系统需要确定性行为
4. **科学研究**：实验的可复现性是发表和验证的基础
5. **合规要求**：某些行业对 AI 系统的可预测性有明确要求

相反，以下场景可能不需要确定性推理：

- 面向终端用户的聊天应用
- 创意生成任务
- 对延迟高度敏感的在线服务

---

## 更广泛的影响

### 对 RL 训练的意义

确定性推理对强化学习训练具有特殊意义。SGLang 团队与 slime 合作，实现了 **100% 可复现的 RL 训练**：

> "Taking this deterministic inference capability further, SGLang collaborated with the slime team to unlock 100% reproducible RL training."

在 RLHF 和 RLVR 等范式中，策略模型的 rollout 需要与训练步骤精确对应。非确定性推理会引入难以追踪的噪声，影响训练稳定性和调试效率。

### 与其他研究的关联

2025 年的 NeurIPS 收录了另一篇相关研究（Oral）：[Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501)。该工作从数值精度的角度进行了系统性分析，发现：

- 系统配置（batch size、GPU 数量、GPU 型号）对 LLM 可复现性有显著影响
- 推理模型（reasoning models）对数值差异尤为敏感，准确率波动可达 9%
- 提出了 **LayerCast** 方案：16-bit 存储 + FP32 计算，平衡内存效率和数值稳定性

两项研究相互补充：Thinking Machines Lab 聚焦于 batch invariance 这一主要来源，而 NeurIPS 论文则提供了更全面的数值精度分析。

### 重新定义问题

Thinking Machines Lab 的工作最重要的贡献或许在于**重新定义了问题本身**：

> "The paper reframes LLM nondeterminism as an engineering bug (batch-sensitive kernels) rather than an inevitable hardware limitation."

这一视角的转变意味着：LLM 推理的非确定性不是必须接受的物理限制，而是可以通过工程手段解决的实现缺陷。这为构建更可靠、可预测的 AI 系统提供了理论和实践基础。

---

## 总结

LLM 推理的非确定性问题长期困扰着研究者和工程师。Thinking Machines Lab 的研究揭示了一个被广泛忽视的事实：**主要问题不在于 GPU 的并发特性，而在于 kernel 实现对 batch size 的敏感性**。

通过 batch-invariant kernels 的设计和实现，我们现在可以在工程层面实现真正的确定性推理。尽管存在一定的性能开销，但在 RL 训练、模型调试、安全审计等场景中，这一代价是值得的。

随着 vLLM、SGLang 等主流推理引擎的支持，确定性推理正在从研究原型走向工程实践。这标志着 LLM 系统在可靠性和可控性方面迈出了重要一步。

---

## 参考资料

1. Thinking Machines Lab. [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). September 2025.

2. Yuan, J. et al. [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501). NeurIPS 2025 (Oral).

3. LMSYS Org. [Towards Deterministic Inference in SGLang and Reproducible RL Training](https://lmsys.org/blog/2025-09-22-sglang-deterministic/). September 2025.

4. vLLM Documentation. [Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/).

5. Thinking Machines Lab. [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops). GitHub Repository.

6. OpenAI. [How to make your completions outputs consistent with the seed parameter](https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter). OpenAI Cookbook.
