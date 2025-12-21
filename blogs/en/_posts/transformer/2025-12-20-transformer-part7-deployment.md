---
layout: post
title: "Transformer Notes (VII): Deployment Optimization"
date: 2025-12-20 11:00:00
author: Phonism
tags: [Quantization, Inference, vLLM, Speculative Decoding, KV Cache]
math: true
lang: en
translation: /transformer-part7-deployment/
---

This is the seventh article in the Transformer series, providing a comprehensive analysis of **deployment optimization** techniques for large language models, including model quantization and inference acceleration. These techniques are key to efficiently deploying hundred-billion-parameter models in practical applications.

## 1. Model Quantization

Model quantization is a technique that converts floating-point representations in neural networks to low-precision representations, and is one of the core technologies for efficient deployment of large language models.

### 1.1 Why Quantization is Needed

The parameter scale of modern LLMs brings severe deployment challenges:

- **Memory Requirements**: A 70B parameter model stored in FP16 requires 140GB of GPU memory
- **Bandwidth Bottleneck**: Inference is primarily limited by memory bandwidth rather than computation
- **Energy Costs**: Energy consumed by data movement far exceeds computation itself

**Storage characteristics of different precisions**:

| Precision | Bit Width | Relative Memory | Typical Use |
|------|------|----------|----------|
| FP32 | 32 | 1× | Training gradient accumulation |
| FP16/BF16 | 16 | 0.5× | Standard training and inference |
| FP8 | 8 | 0.25× | Efficient training (Hopper+) |
| INT8 | 8 | 0.25× | Quantized inference |
| INT4 | 4 | 0.125× | Aggressive quantized inference |

### 1.2 Mathematical Definition of Quantization

**Uniform Quantization** is the most commonly used quantization method. Given a floating-point number $x$, the quantization process is:

$$Q(x) = \text{clamp}\left( \left\lfloor \frac{x}{s} \right\rceil + z, 0, 2^b - 1 \right)$$

where $s$ is the scale factor, $z$ is the zero-point, and $b$ is the target bit width.

**Dequantization** recovers the approximate value:

$$\hat{x} = s \cdot (Q(x) - z)$$

**Symmetric Quantization** vs **Asymmetric Quantization**:
- Symmetric quantization: $z = 0$, simpler implementation
- Asymmetric quantization: allows $z \neq 0$, more effective for skewed distributions

### 1.3 Quantization Granularity

The computation granularity of quantization parameters $s, z$ affects the trade-off between accuracy and overhead:

- **Per-Tensor**: The entire tensor shares one set of parameters, minimal overhead but large accuracy loss
- **Per-Channel**: Each output channel is quantized independently, commonly used for weights
- **Per-Token**: Each token is quantized independently, commonly used for activations
- **Per-Group**: Channels are divided into groups, each group quantized independently, a compromise between accuracy and overhead

**Group Quantization Example**: With a group size of 128, the effective bit width for INT4 quantization is approximately $4 + 32/128 = 4.25$ bits.

### 1.4 Post-Training Quantization (PTQ)

Post-training quantization is performed after model training is complete, requires no retraining, and is the mainstream method for LLM quantization.

**Basic PTQ Process**:
1. **Calibration**: Use a small amount of representative data to collect activation distribution statistics
2. **Determine quantization parameters**: Calculate $s, z$ based on statistical information
3. **Quantize weights**: Convert floating-point weights to low-precision representation
4. **(Optional) Correction**: Reduce quantization error through additional optimization

**Calibration Strategies**:
- **MinMax Calibration**: Uses observed maximum and minimum values, sensitive to outliers
- **Percentile Calibration**: Uses p-th and (100-p)-th percentiles
- **MSE Calibration**: Minimizes quantization error
- **KL Divergence Calibration**: Minimizes KL divergence between original and quantized distributions

### 1.5 Challenges in LLM Quantization: Activation Outliers

A key characteristic of LLMs is the presence of **outliers** in activations: a very small number of channels contain activation values far larger than other channels. These outliers:

- Appear in specific channels, consistent across tokens
- Can be 100 times larger than normal values
- Removing these channels causes model performance to collapse

Standard quantization methods are forced to expand the quantization range to cover outliers, resulting in severe degradation of quantization accuracy for normal values.

### 1.6 SmoothQuant

SmoothQuant is a breakthrough method for solving the activation outlier problem, achieving W8A8 quantization for LLMs.

**Core Idea**: Migrate the quantization difficulty from activations to weights:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \hat{W}$$

where $s$ is the migration factor. $\hat{X}$ has a more uniform distribution and is easier to quantize; $\hat{W}$ absorbs part of the difficulty, but weights themselves have good distribution, so the impact is limited.

**Migration Factor Selection**:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

where $\alpha = 0.5$ works well for most models.

### 1.7 GPTQ

GPTQ is a weight quantization method based on second-order information that can compress LLMs to 4-bit precision.

**Problem Formulation**: Optimize layer by layer to minimize output error after quantization:

$$\arg\min_{\hat{W}} \| WX - \hat{W}X \|_2^2$$

**Performance**:
- 175B parameter models can be quantized in 4 hours on a single GPU
- 3-4 bit quantization with minimal accuracy loss (perplexity increase < 0.5)
- Widely used for quantized distribution of open-source LLMs

### 1.8 AWQ

AWQ (Activation-aware Weight Quantization) is based on a key observation: the importance of different weight channels varies greatly.

**Core Observation**: If certain channels of activation values $X$ have larger numerical values, the quantization error of corresponding weight channels has greater impact.

**Method**: Amplify important channels through per-channel scaling, then restore after quantization. At 4-bit quantization, AWQ's perplexity is typically better than GPTQ.

### 1.9 GGUF Format

GGUF is a model storage format defined by the llama.cpp project, widely used for local LLM deployment.

| Format | Effective Bit Width | Accuracy Loss |
|------|----------|----------|
| Q8_0 | 8.5 bits | Minimal |
| Q5_K_M | 5.5 bits | Small |
| Q4_K_M | 4.8 bits | Medium |
| Q4_0 | 4.5 bits | Larger |
| Q2_K | 3.4 bits | Large |

"K" indicates the use of k-quant method, using higher precision for important layers.

### 1.10 FP8 Quantization

FP8 is an 8-bit floating-point format that preserves dynamic range compared to INT8.

**Two Formats**:
- **E4M3**: 4-bit exponent, 3-bit mantissa, larger dynamic range, suitable for forward propagation
- **E5M2**: 5-bit exponent, 2-bit mantissa, higher precision, suitable for gradients

DeepSeek-V3 demonstrates industrial application of FP8 training:
- Memory bandwidth requirement reduced by half
- FP8 throughput on H100 is 2× that of FP16
- Training requires only 2.788M H800 GPU hours

### 1.11 KV Cache Quantization

The memory bottleneck for long-context inference comes mainly from KV Cache rather than model weights.

**KV Cache Size**: Taking LLaMA-70B (80 layers, 64 heads, 128 dimensions) as an example:
- 100K context, batch=1, FP16: approximately 40GB
- Same settings using INT4: approximately 10GB

**KIVI Method**:
- Key: per-channel quantization
- Value: per-token quantization
- Can compress to 2-bit, reducing memory by 8×

## 2. Inference Optimization

LLM inference faces unique challenges: huge number of model parameters, autoregressive generation token by token, KV Cache grows with sequence length.

### 2.1 Two-Phase Inference

**Prefill Phase**: Process all tokens of the input prompt
- Computation characteristics: Parallel processing, compute-bound
- Bottleneck: Computational load of matrix multiplication
- Metric: Time To First Token (TTFT)

**Decode Phase**: Generate subsequent tokens one by one
- Computation characteristics: Autoregressive generation, memory-bound
- Bottleneck: Memory bandwidth for loading model parameters and KV Cache
- Metric: Tokens Per Second (TPS)

| Characteristic | Prefill | Decode |
|------|---------|--------|
| Token Count | N (input length) | 1 |
| Computation Mode | Parallel | Serial |
| Bottleneck | Computation | Memory Bandwidth |
| GPU Utilization | High | Low |

### 2.2 Continuous Batching

Problem with traditional static batching: requests have varying lengths, short requests must wait for long requests to complete, padding wastes computational resources.

**Continuous Batching** dynamically manages requests:
- Resources are released immediately after request completion, new requests join immediately
- No need to wait for entire batch to complete
- Iteration-level scheduling, rather than request-level

### 2.3 PagedAttention and vLLM

vLLM introduces PagedAttention, borrowing the idea of virtual memory from operating systems to manage KV Cache.

**Problems with Traditional KV Cache**:
- Pre-allocate according to maximum sequence length, causing memory waste
- Different requests have varying lengths, creating fragmentation
- Cannot dynamically expand, limiting concurrent request count

**PagedAttention Principle**:
- Divide KV Cache into fixed-size Pages (blocks)
- Pages can be stored non-contiguously (similar to virtual memory)
- Allocate on demand, release when finished

**vLLM Performance**:
- Compared to HuggingFace Transformers, throughput improved **24×**
- GPU memory utilization close to 100% (no fragmentation)

### 2.4 Prefix Caching and SGLang

Many application scenarios have shared prefixes (System Prompt, Few-shot examples, etc.).

**RadixAttention**: SGLang uses a Radix Tree to manage KV Cache:
- Each edge of the tree corresponds to a token sequence segment
- Requests sharing prefixes share KV Cache
- LRU strategy manages cache eviction

**SGLang Features**:
- RadixAttention: Automatic Prefix Caching
- Structured output: Compressed finite state machine accelerates JSON generation
- Compared to vLLM, throughput improvement can reach **5-6×**

### 2.5 Speculative Decoding

Speculative Decoding is an important technique for accelerating autoregressive generation. The core idea is "first use a small model to quickly guess, then use a large model to batch verify".

**Workflow**:
1. **Draft Phase**: Use small model to autoregressively generate K candidate tokens
2. **Verify Phase**: Input K tokens in parallel to large model for verification
3. **Accept/Reject**: Consecutively consistent tokens are directly accepted; divergence point gets correct token from large model

**Key Guarantee**: Even if all drafts are wrong, at least 1 correct token can be obtained from the large model.

**Verification Mechanism in Sampling Scenarios**:

Let the draft model's distribution for generating token $x$ at position $t$ be $q(x)$, and the target model's distribution be $p(x)$. The acceptance probability is:

$$a(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

When rejected, resample from the residual distribution:

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

This ensures that the final output distribution is strictly equal to $p(x)$ (unbiased).

**EAGLE Series**:

| Method | Additional Model | Training Requirement | Speedup |
|------|----------|----------|--------|
| Independent Draft | Yes | None | 2-3× |
| EAGLE-1 | No | Train Head | 2.5-3× |
| EAGLE-2 | No | Train Head | 3-4× |
| EAGLE-3 | No | Train Head | 4-5× |

EAGLE's core innovation is using the target model's hidden states to guide draft generation, significantly improving guess accuracy.

### 2.6 KV Cache Compression

**KV Cache Quantization**:
- INT8/FP8: 50% memory reduction, minimal accuracy loss
- 2-4 bit (KVQuant, KIVI): 4-8× memory reduction

**KV Cache Sparsification**:
- **H2O**: Dynamically identify important tokens, retain "Heavy Hitters"
- **SnapKV**: Select important KV based on observation window, 16K input can achieve 3.6× speedup

### 2.7 Inference Engine Comparison

| Engine | Core Technology | Prefix Cache | Speculative Decoding | Features |
|------|----------|--------------|----------|------|
| vLLM | PagedAttention | Supported | Supported | Most widely used |
| SGLang | RadixAttention | Native | EAGLE etc. | Fast structured output |
| TensorRT-LLM | Deep optimization | Supported | Multiple | NVIDIA official |
| llama.cpp | CPU optimization | Limited | Supported | Local deployment |

## 3. Best Practices

### 3.1 Quantization Strategy Selection

| Scenario | Recommended Solution |
|------|----------|
| Sufficient memory | FP16/BF16, no accuracy loss |
| General deployment | INT8 or FP8, minimal accuracy loss |
| Edge devices | INT4 (GPTQ/AWQ), acceptable accuracy loss |
| Extreme compression | 2-3 bit, need careful task impact evaluation |

### 3.2 Inference Optimization Strategies

**Latency-Priority Scenarios**:
- Use Prefix Caching (SGLang)
- Speculative decoding (EAGLE)
- Small batch size + high parallelism

**Throughput-Priority Scenarios**:
- Continuous Batching (vLLM)
- Large batch size
- KV Cache quantization

**Long-Context Scenarios**:
- KV Cache quantization (KVQuant, KIVI)
- KV Cache sparsification (SnapKV, H2O)
- Prefill-Decode separation

## 4. Summary

This article provides a comprehensive analysis of the two core technologies for large model deployment optimization:

| Domain | Key Technology | Effect |
|------|----------|------|
| Model Quantization | GPTQ/AWQ (4-bit) | 4× model size reduction |
| Model Quantization | SmoothQuant (W8A8) | 1.5× inference speedup |
| KV Cache | KIVI (2-bit) | 8× memory reduction |
| Batching | PagedAttention | 24× throughput improvement |
| Decoding Acceleration | EAGLE-3 | 4-5× speedup |
| Prefix Cache | RadixAttention | 5-6× throughput improvement |

These techniques enable hundred-billion-parameter models to run efficiently on limited hardware resources, and are key infrastructure for large model deployment.

The next article, also the last in this series, will discuss **Advanced Applications**, including multimodal and reasoning-enhanced technologies.
