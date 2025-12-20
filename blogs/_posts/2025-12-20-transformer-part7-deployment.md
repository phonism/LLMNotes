---
layout: post
title: "Transformer 学习笔记（七）：部署优化"
date: 2025-12-20 11:00:00
categories: [Deep Learning, Transformer]
tags: [Quantization, Inference, vLLM, Speculative Decoding, KV Cache]
math: true
lang: zh
---

本文是 Transformer 系列的第七篇，全面解析大语言模型的 **部署优化** 技术，包括模型量化和推理加速。这些技术是将千亿参数模型高效部署到实际应用的关键。

## 1. 模型量化

模型量化是将神经网络中的浮点数表示转换为低精度表示的技术，是大语言模型高效部署的核心技术之一。

### 1.1 为什么需要量化

现代 LLM 的参数规模带来严峻的部署挑战：

- **内存需求**：70B 参数模型以 FP16 存储需要 140GB 显存
- **带宽瓶颈**：推理时主要受限于内存带宽而非计算
- **能耗成本**：数据移动消耗的能量远超计算本身

**不同精度的存储特性**：

| 精度 | 位宽 | 相对内存 | 典型用途 |
|------|------|----------|----------|
| FP32 | 32 | 1× | 训练梯度累积 |
| FP16/BF16 | 16 | 0.5× | 标准训练与推理 |
| FP8 | 8 | 0.25× | 高效训练（Hopper+） |
| INT8 | 8 | 0.25× | 量化推理 |
| INT4 | 4 | 0.125× | 激进量化推理 |

### 1.2 量化的数学定义

**均匀量化**（Uniform Quantization）是最常用的量化方式。给定浮点数 $x$，量化过程为：

$$Q(x) = \text{clamp}\left( \left\lfloor \frac{x}{s} \right\rceil + z, 0, 2^b - 1 \right)$$

其中 $s$ 是缩放因子（scale），$z$ 是零点（zero-point），$b$ 是目标位宽。

**反量化**恢复近似值：

$$\hat{x} = s \cdot (Q(x) - z)$$

**对称量化** vs **非对称量化**：
- 对称量化：$z = 0$，实现更简单
- 非对称量化：允许 $z \neq 0$，对偏斜分布更有效

### 1.3 量化粒度

量化参数 $s, z$ 的计算粒度影响精度与开销的权衡：

- **Per-Tensor**：整个张量共享一组参数，开销最小但精度损失大
- **Per-Channel**：每个输出通道独立量化，常用于权重
- **Per-Token**：每个 token 独立量化，常用于激活值
- **Per-Group**：将通道分组，每组独立量化，精度与开销的折中

**Group Quantization 示例**：设 group size 为 128，在 INT4 量化时有效位宽约为 $4 + 32/128 = 4.25$ 位。

### 1.4 训练后量化 (PTQ)

训练后量化在模型训练完成后进行，无需重新训练，是 LLM 量化的主流方法。

**基本 PTQ 流程**：
1. **校准**：使用少量代表性数据统计激活值分布
2. **确定量化参数**：根据统计信息计算 $s, z$
3. **量化权重**：将浮点权重转换为低精度表示
4. **（可选）校正**：通过额外优化减少量化误差

**校准策略**：
- **MinMax 校准**：使用观测到的最大最小值，对离群值敏感
- **百分位校准**：使用第 p 和 100-p 百分位数
- **MSE 校准**：最小化量化误差
- **KL 散度校准**：最小化原始分布与量化分布的 KL 散度

### 1.5 LLM 量化的挑战：激活值离群值

LLM 的一个关键特性是激活值中存在 **离群值**（outliers）：极少数通道包含数值远大于其他通道的激活值。这些离群值：

- 出现在特定通道，跨 token 一致
- 数值可达正常值的 100 倍以上
- 移除这些通道会导致模型性能崩溃

标准量化方法被迫扩大量化范围以覆盖离群值，导致正常值的量化精度严重下降。

### 1.6 SmoothQuant

SmoothQuant 是解决激活离群值问题的突破性方法，实现了 LLM 的 W8A8 量化。

**核心思想**：将量化难度从激活"迁移"到权重：

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$$

其中 $s$ 是迁移因子。$\hat{X}$ 的分布更均匀，更易量化；$\hat{W}$ 吸收了部分难度，但权重本身分布良好，影响有限。

**迁移因子选择**：

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

其中 $\alpha = 0.5$ 对大多数模型效果良好。

### 1.7 GPTQ

GPTQ 是基于二阶信息的权重量化方法，可将 LLM 压缩至 4 位精度。

**问题形式化**：逐层优化，最小化量化后的输出误差：

$$\arg\min_{\hat{W}} \| WX - \hat{W}X \|_2^2$$

**性能**：
- 175B 参数模型可在单 GPU 上 4 小时内完成量化
- 3-4 位量化精度损失极小（困惑度增加 < 0.5）
- 广泛应用于开源 LLM 的量化分发

### 1.8 AWQ

AWQ（Activation-aware Weight Quantization）基于一个关键观察：不同权重通道的重要性差异巨大。

**核心观察**：如果激活值 $X$ 的某些通道数值较大，则对应权重通道的量化误差影响更大。

**方法**：通过 per-channel 缩放放大重要通道，量化后再恢复。在 4 位量化下，AWQ 的困惑度通常优于 GPTQ。

### 1.9 GGUF 格式

GGUF 是 llama.cpp 项目定义的模型存储格式，广泛用于本地 LLM 部署。

| 格式 | 有效位宽 | 精度损失 |
|------|----------|----------|
| Q8_0 | 8.5 bits | 极小 |
| Q5_K_M | 5.5 bits | 小 |
| Q4_K_M | 4.8 bits | 中等 |
| Q4_0 | 4.5 bits | 较大 |
| Q2_K | 3.4 bits | 大 |

"K" 表示使用 k-quant 方法，对重要层使用更高精度。

### 1.10 FP8 量化

FP8 是一种 8 位浮点格式，相比 INT8 保留了动态范围。

**两种格式**：
- **E4M3**：4 位指数、3 位尾数，动态范围更大，适合前向传播
- **E5M2**：5 位指数、2 位尾数，精度更高，适合梯度

DeepSeek-V3 展示了 FP8 训练的工业应用：
- 内存带宽需求减半
- H100 上 FP8 吞吐量是 FP16 的 2 倍
- 训练仅需 2.788M H800 GPU 小时

### 1.11 KV Cache 量化

长上下文推理的内存瓶颈主要来自 KV Cache 而非模型权重。

**KV Cache 大小**：以 LLaMA-70B（80 层、64 头、128 维）为例：
- 100K 上下文、batch=1、FP16：约 40GB
- 同等设置使用 INT4：约 10GB

**KIVI 方法**：
- Key：per-channel 量化
- Value：per-token 量化
- 可压缩至 2-bit，内存减少 8 倍

## 2. 推理优化

LLM 推理面临独特的挑战：模型参数量巨大、自回归生成逐 token 进行、KV Cache 随序列长度增长。

### 2.1 两阶段推理

**Prefill 阶段**：处理输入 prompt 的所有 token
- 计算特性：并行处理，计算密集型（Compute-bound）
- 瓶颈：矩阵乘法的计算量
- 指标：Time To First Token（TTFT）

**Decode 阶段**：逐个生成后续 token
- 计算特性：自回归生成，内存带宽受限（Memory-bound）
- 瓶颈：加载模型参数和 KV Cache 的内存带宽
- 指标：Tokens Per Second（TPS）

| 特性 | Prefill | Decode |
|------|---------|--------|
| Token 数 | N（输入长度） | 1 |
| 计算模式 | 并行 | 串行 |
| 瓶颈 | 计算 | 内存带宽 |
| GPU 利用率 | 高 | 低 |

### 2.2 Continuous Batching

传统静态批处理的问题：请求长度不一，短请求需等待长请求完成，填充浪费计算资源。

**Continuous Batching** 动态管理请求：
- 请求完成后立即释放资源，新请求立即加入
- 无需等待整批完成
- 迭代级别调度，而非请求级别

### 2.3 PagedAttention 与 vLLM

vLLM 引入 PagedAttention，借鉴操作系统虚拟内存的思想管理 KV Cache。

**传统 KV Cache 的问题**：
- 按最大序列长度预分配，造成内存浪费
- 不同请求长度不一，产生碎片化
- 无法动态扩展，限制并发请求数

**PagedAttention 原理**：
- 将 KV Cache 分成固定大小的 Page（块）
- Page 可以非连续存储（类似虚拟内存）
- 按需分配，用完即释放

**vLLM 性能**：
- 相比 HuggingFace Transformers，吞吐量提升 **24 倍**
- 显存利用率接近 100%（无碎片）

### 2.4 Prefix Caching 与 SGLang

许多应用场景存在共享前缀（System Prompt、Few-shot 示例等）。

**RadixAttention**：SGLang 使用基数树（Radix Tree）管理 KV Cache：
- 树的每条边对应一段 token 序列
- 共享前缀的请求共享 KV Cache
- LRU 策略管理缓存淘汰

**SGLang 特性**：
- RadixAttention：自动 Prefix Caching
- 结构化输出：压缩有限状态机加速 JSON 生成
- 相比 vLLM，吞吐量提升可达 **5-6 倍**

### 2.5 投机解码

投机解码（Speculative Decoding）是加速自回归生成的重要技术，核心思想是"先用小模型快速猜测，再用大模型批量验收"。

**工作流程**：
1. **Draft 阶段**：用小模型自回归生成 K 个候选 token
2. **Verify 阶段**：将 K 个 token 并行输入大模型验证
3. **Accept/Reject**：连续一致的 token 直接接受；分歧点由大模型给出正确 token

**关键保证**：即使 draft 全部猜错，也能从大模型获得至少 1 个正确 token。

**采样场景下的验证机制**：

设 Draft 模型在位置 $t$ 生成 token $x$ 的分布为 $q(x)$，Target 模型的分布为 $p(x)$。接受概率为：

$$a(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

拒绝时从残差分布重采样：

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

这保证了最终输出分布严格等于 $p(x)$（无偏性）。

**EAGLE 系列**：

| 方法 | 额外模型 | 训练需求 | 加速比 |
|------|----------|----------|--------|
| 独立 Draft | 是 | 无 | 2-3× |
| EAGLE-1 | 否 | 训练 Head | 2.5-3× |
| EAGLE-2 | 否 | 训练 Head | 3-4× |
| EAGLE-3 | 否 | 训练 Head | 4-5× |

EAGLE 的核心创新是利用 Target 模型的隐状态来指导 Draft 生成，显著提高猜测准确率。

### 2.6 KV Cache 压缩

**KV Cache 量化**：
- INT8/FP8：显存减少 50%，精度损失很小
- 2-4 bit（KVQuant、KIVI）：显存减少 4-8 倍

**KV Cache 稀疏化**：
- **H2O**：动态识别重要 token，保留"Heavy Hitter"
- **SnapKV**：基于观察窗口选择重要 KV，16K 输入可达 3.6 倍加速

### 2.7 推理引擎对比

| 引擎 | 核心技术 | Prefix Cache | 投机解码 | 特点 |
|------|----------|--------------|----------|------|
| vLLM | PagedAttention | 支持 | 支持 | 最广泛使用 |
| SGLang | RadixAttention | 原生 | EAGLE 等 | 结构化输出快 |
| TensorRT-LLM | 深度优化 | 支持 | 多种 | NVIDIA 官方 |
| llama.cpp | CPU 优化 | 有限 | 支持 | 本地部署 |

## 3. 最佳实践

### 3.1 量化策略选择

| 场景 | 推荐方案 |
|------|----------|
| 内存充足 | FP16/BF16，无精度损失 |
| 一般部署 | INT8 或 FP8，精度损失极小 |
| 边缘设备 | INT4（GPTQ/AWQ），精度损失可接受 |
| 极限压缩 | 2-3 bit，需仔细评估任务影响 |

### 3.2 推理优化策略

**延迟优先场景**：
- 使用 Prefix Caching（SGLang）
- 投机解码（EAGLE）
- 小批处理 + 高并行度

**吞吐优先场景**：
- Continuous Batching（vLLM）
- 大批处理
- KV Cache 量化

**长上下文场景**：
- KV Cache 量化（KVQuant、KIVI）
- KV Cache 稀疏化（SnapKV、H2O）
- Prefill-Decode 分离

## 4. 总结

本文全面解析了大模型部署优化的两大核心技术：

| 领域 | 关键技术 | 效果 |
|------|----------|------|
| 模型量化 | GPTQ/AWQ（4-bit） | 模型大小减少 4 倍 |
| 模型量化 | SmoothQuant（W8A8） | 推理速度提升 1.5 倍 |
| KV Cache | KIVI（2-bit） | 显存减少 8 倍 |
| 批处理 | PagedAttention | 吞吐提升 24 倍 |
| 解码加速 | EAGLE-3 | 加速 4-5 倍 |
| Prefix Cache | RadixAttention | 吞吐提升 5-6 倍 |

这些技术使得千亿参数模型能够在有限硬件资源上高效运行，是大模型落地的关键基础设施。

下一篇也是本系列的最后一篇，我们将讨论 **前沿应用**，包括多模态和推理增强技术。
