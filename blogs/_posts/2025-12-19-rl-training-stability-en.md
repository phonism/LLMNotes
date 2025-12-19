---
layout: post
title: "LLM-RL Training Stability: Root Cause Analysis and Solutions"
date: 2025-12-19
author: Phonism
tags: [RL, RLHF, PPO, GRPO, GSPO, Training Stability]
lang: en
translation: /rl-training-stability/
---

## Introduction

In large language model reinforcement learning (LLM-RL) training, it's common to observe training curves that rise steadily for a period before suddenly collapsing. Whether it's complex reasoning RL or multi-turn tool-calling Agentic RL, many practitioners have encountered this mysterious training collapse.

This blog post synthesizes multiple important works including ByteDance's [When Speed Kills Stability](https://richardli.xyz/rl-collapse), Qwen team's [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374), and vLLM's [Bitwise Consistent Training](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) to systematically analyze the root causes of LLM-RL training instability and summarize practical solutions.

**Notation Conventions**:

| Symbol | Meaning |
|------|------|
| $\pi_\theta$ | Current policy being optimized (computed by training engine) |
| $\pi_{\text{old}}$ | Policy at sampling time (computed by training engine, but with old parameters) |
| $\pi_\text{vllm}$ | Policy computed by inference engine (vLLM/SGLang, numerically different from $\pi$) |
| $\pi_{\text{ref}}$ | Reference policy (anchor for KL regularization, typically the SFT model) |

Core distinction: $\pi$ vs $\pi_\text{vllm}$ represents **numerical differences of the same parameters across different engines**, while $\pi_\theta$ vs $\pi_{\text{old}}$ represents **parameter differences at different time steps within the same engine**.

## Problem Manifestation: Sudden Collapse

Typical collapse patterns include:

1. **Training Reward**: Steady increase → sudden drop or severe oscillations
2. **Gradient Norm**: Normal range → sudden explosion
3. **PPL (Perplexity)**: Stable → sharp spike
4. **Entropy**: Gradual decline → abnormal fluctuations

Most perplexingly, these collapses are often **unpredictable**—the same code and hyperparameters may behave completely differently across different GPUs.

## Root Cause Analysis

LLM-RL training instability has **two levels of root causes**:

1. **System Level**: Training-Inference Mismatch (numerical differences between inference engine $\pi_\text{vllm}$ and training engine $\pi$)
2. **Algorithm Level**: Token-Sequence Mismatch (token-level optimization objective vs sequence-level reward)

These two issues are independent but compound each other. Let's analyze them separately.

### Root Cause One: Training-Inference Mismatch

**The Trade-off Between Speed and Consistency**

Modern LLM-RL systems typically use **high-speed inference engines** (such as vLLM, SGLang) for rollout sampling, while using **training frameworks** (such as FSDP, Megatron-LM) for parameter updates. These two types of systems have fundamentally different optimization objectives:

| System | Optimization Goal | Typical Techniques |
|------|----------|----------|
| Inference Engine | Throughput maximization | Speculative Decoding, INT8/FP8, batch-variant CUDA kernels |
| Training Framework | Numerical stability | FP32 Master Weights, deterministic operators |

This divergence in optimization objectives leads to **inevitable numerical inconsistencies**. Even with identical parameters, the inference engine's computed $\pi_\text{vllm}(y\mid x)$ and the training engine's computed $\pi(y\mid x)$ will differ.

**Actual Gradient vs Theoretical Gradient**

In theory, the on-policy policy gradient should be:

$$\mathbb{E}_{y \sim \pi_\theta} \left[ R(x,y) \nabla_\theta \log \pi_\theta(y|x) \right]$$

But in practice, since samples come from the inference engine $\pi_\text{vllm}$:

$$\mathbb{E}_{y \sim \pi_\text{vllm}} \left[ R(x,y) \nabla_\theta \log \pi_\theta(y|x) \right]$$

This means: **You think you're doing on-policy training, but you're actually doing off-policy training.**

### Root Cause Two: Token-Sequence Mismatch

Mainstream RL algorithms (PPO, GRPO) use **token-level optimization objectives**, but rewards are **sequence-level**.

**First-order Approximation Theoretical Foundation**: The sequence-level IS weight can be expanded as:

$$\frac{\pi_\theta(y|x)}{\pi_\text{vllm}(y|x)} = \prod_{t=1}^{|y|}(1+\delta_t) \approx 1 + \sum_{t=1}^{|y|}\delta_t$$

where $\delta_t = \frac{\pi_\theta(y_t \mid s_t)}{\pi_\text{vllm}(y_t \mid s_t)} - 1$. This shows that the token-level objective is a **first-order approximation** of the sequence-level objective, ignoring higher-order terms of $O(\delta^2)$.

**IS Weight Decomposition**: The token-level IS weight can be decomposed into two factors:

$$\frac{\pi_\theta(y_t|s_t)}{\pi_\text{vllm}(y_t|s_t)} = \underbrace{\frac{\pi_{\text{old}}(y_t|s_t)}{\pi_\text{vllm}(y_t|s_t)}}_{\text{Training-Inference Discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|s_t)}{\pi_{\text{old}}(y_t|s_t)}}_{\text{Policy Staleness}}$$

- **Training-Inference Discrepancy**: Numerical differences between inference engine $\pi_\text{vllm}$ and training engine $\pi$ (Root Cause One)
- **Policy Staleness**: Policy drift during mini-batch processing

This decomposition shows: **The two root causes compound multiplicatively**—if either deviates significantly, the IS weight will deviate from 1, undermining the validity of the first-order approximation.

## Specific Manifestations and Challenges

### Challenge One: Low Probability Token Trap

**Mismatch is most severe at low probability tokens**. When vLLM samples a token with near-zero probability, the probability computed by FSDP may be orders of magnitude lower than vLLM's, leading to:

- PPL explosion (denominator approaching 0)
- Gradient explosion
- Training collapse

This explains why multi-turn tool-calling scenarios are particularly prone to collapse—tool-returned OOD (Out-of-Distribution) text causes the model to generate more low probability tokens.

### Challenge Two: Hardware Differences Amplify the Problem

The degree of mismatch varies drastically across different GPU architectures:

$$\text{vllm-kl}: \quad \text{H20} < \text{L20} < \text{A100}$$

- H20: $5 \times 10^{-4}$ ~ $10^{-3}$
- L20: $10^{-3}$ ~ $10^{-2}$
- A100: $10^{-2}$ ~ $1$

The same code may collapse on L20, but resuming from checkpoint on H20 can immediately stabilize training!

### Challenge Three: High Variance and Entropy Collapse

**High Variance Problem**: In long CoT scenarios, the accumulation of token-level IS weights leads to variance explosion:

$$\prod_{t=1}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} = \exp\left(\sum_t \log \rho_t\right)$$

Logprob differences **accumulate linearly** over long sequences, then become **exponential differences** after exponentiation. The consequence is that gradients are dominated by very few samples and very few tokens, which is especially catastrophic for MoE models—a small number of extreme updates may break expert routing.

**Entropy Collapse**: RL optimization tends to enhance the probability of high-reward tokens while compressing low-probability tokens, leading to continuous policy entropy decline. When entropy approaches 0:
- Exploration capability is lost
- Diversity collapses
- Unable to discover new solutions

Research shows that policy performance comes at the cost of entropy consumption, with a theoretical upper bound.

## Solutions

### Solution One: Sequence-Level Importance Sampling

The correct unbiased estimator requires applying the importance ratio over the entire sequence:

$$g_{\text{seq}} = \mathbb{E}_{y \sim \pi_\text{vllm}} \left[ \frac{\pi_\theta(y|x)}{\pi_\text{vllm}(y|x)} \cdot R(x,y) \cdot \nabla \log \pi_\theta(y|x) \right]$$

In practice, there are two variants:

+ **Truncated IS (TIS)**: Truncate the ratio $$\rho(y) \gets \min(\rho(y), C)$$
+ **Masked IS (MIS)**: Directly mask sequences exceeding the threshold $$\rho(y) \gets \rho(y) \cdot \mathbb{I}\{\rho(y) \le C\}$$

Experiments show that **MIS works better than TIS**, not only stabilizing training but also exceeding the peak performance before collapse.

### Solution Two: Off-Policy Sequence Masking (DeepSeek-V3.2)

DeepSeek-V3.2 adopts a more refined masking strategy:

$$M_{i,t} = \begin{cases} 0 & \text{if } \hat{A}_{i,t} < 0 \text{ and } \frac{1}{\lvert o_i \rvert}\sum \log \frac{\pi_{\text{old}}}{\pi_\theta} > \delta \\ 1 & \text{otherwise} \end{cases}$$

Core idea: **Only mask sequences with negative advantage and off-policy degree exceeding threshold $\delta$**.

Here $\frac{1}{\lvert o_i \rvert}\sum \log \frac{\pi_{\text{old}}}{\pi_\theta}$ measures the off-policy degree, essentially **per-token average KL divergence** (equivalent to the geometric mean of log ratios). This length normalization avoids systematic discarding of long sequences.

**Why only mask negative advantage?** Samples with positive advantage, even if off-policy, still provide useful gradient directions; whereas off-policy samples with negative advantage may introduce harmful gradient noise.

DeepSeek-V3.2 also introduces complementary stabilization techniques:

**Keep Routing (MoE-specific)**: Expert routing in inference and training frameworks may be inconsistent. The solution is to save the routing path during inference and force the same path during training.

**Keep Sampling Mask**: Top-p/top-k sampling truncates low probability tokens, causing inconsistent action spaces between $\pi_{\text{old}}$ and $\pi_\theta$. The solution is to save the truncation mask during sampling and apply the same mask to $\pi_\theta$ during training.

**Unbiased KL Estimation**: The standard K3 estimator is biased in off-policy settings. Corrected formula:
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_\theta}{\pi_{\text{old}}} \left[ \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1 \right]$$

### Solution Three: Bitwise Consistent Training

Another path is to **make inference and training completely consistent**.

Core approach:
1. Audit every kernel call in the forward pass
2. Import vLLM's fused operations (SiLU MLPs, RMSNorm) into the training framework
3. Implement corresponding backward passes

Experiments show that enabling bitwise consistency results in:
- Faster convergence
- Higher final rewards
- More stable training

But the cost is approximately **2.4x performance degradation**.

### Solution Four: IcePop (Token-Level Discrepancy Masking)

IcePop, proposed by Ring-1T, handles mismatch at **token granularity**, complementing the previous sequence-level methods.

**Core idea**: Define token-level ratio $k_{i,t} = \frac{\pi(y_t \mid s_t)}{\pi_\text{vllm}(y_t \mid s_t)}$, and mask tokens exceeding reasonable ranges:

$$M(k) = \begin{cases} k & \text{if } k \in [\alpha, \beta] \\ 0 & \text{otherwise} \end{cases}$$

Typical parameters: $\alpha = 0.5$, $\beta = 5.0$.

**Bidirectional Truncation**: Unlike sequence-level MIS which only focuses on $k > C$, IcePop handles both directions:
- $k > \beta$: Training probability far exceeds inference probability (may cause gradient explosion)
- $k < \alpha$: Training probability far below inference probability (may cause PPL explosion)

**Why is token-level effective?** In MoE models, differences in expert routing cause uneven mismatch distribution across token positions. Token-level masking can precisely remove problematic tokens rather than discarding entire sequences.

**Comparison with sequence-level**:

| Method | Granularity | Advantages | Disadvantages |
|------|------|------|------|
| Seq-MIS | Sequence | Theoretically unbiased | May discard too much data |
| IcePop | Token | Fine-grained control | Doesn't correct state occupancy |

In practice, they can be combined: first use IcePop to handle extreme tokens, then use sequence-level methods to handle overall drift.

### Solution Five: GSPO (Sequence-Level IS)

GSPO (Group Sequence Policy Optimization) elevates the IS operation to the sequence level:

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right)^{1/\lvert y_i \rvert}$$

**Core improvements**:
- First apply **length normalization** to the sequence-level ratio, then clip
- All tokens in the same sequence share the same IS weight

**Differences from GRPO**:

| Dimension | GRPO | GSPO |
|------|------|------|
| IS Granularity | Token-level | Sequence-level |
| Clip Target | Each token's ratio | Normalized ratio of entire sequence |
| Long Sequence Stability | Poor (variance explosion) | Good (length normalization) |

**Advantages**:
- Avoids variance explosion from token-level weight multiplication
- More stable for MoE models
- Simplifies RL infrastructure design

### Solution Six: Multiple Sampling Estimation (MoE-specific)

The KAT-Coder team proposed a different perspective: for MoE models, **sampling noise itself is the dominant factor causing training instability**, not training-inference inconsistency.

**Noise Source Analysis**:

| Model Type | Train-Infer Gap | Inference Noise Variance | Training Noise Variance |
|----------|----------|--------------|--------------|
| Dense | ~0.002 | ~$10^{-5}$ | 0 (Megatron deterministic) |
| MoE | ~0.008 | ~$10^{-3}$ | ~$10^{-7}$ (scatter_add randomness) |

MoE's inference noise variance is two orders of magnitude higher than Dense—this is the primary cause of instability.

**Core Method**: When computing $\pi_{\text{old}}$, directly use the **inference engine** to recompute n times (n=8) and take the average:

$$\hat{\pi}_{\text{old}}(y \mid x) = \frac{1}{n} \sum_{i=1}^{n} \pi_{\text{inference}}^{(i)}(y \mid x)$$

**Key Advantages**:
- Obtains **unbiased estimate with variance reduced by factor of n**
- **No training engine recompute needed**, directly uses inference engine
- In asynchronous frameworks, multiple sampling time can overlap with rollout
- KV cache hit rate approaches 100%
- End-to-end **10-20% training time reduction** in practice

**Comparison with Other Solutions**:

| Solution | Issue |
|------|------|
| Routing Replay | Hard to guarantee prefix cache hits in large-scale agentic scenarios |
| Truncated IS (TIS) | Sensitive to truncation boundary, doesn't address root cause of estimation bias |
| Deterministic Inference | Requires deep inference engine modifications, 40-70% throughput drop |
| Multiple Sampling Estimation | No hyperparameters, engineering-friendly, best performance |

Experiments on Qwen3-235B-A22B show that recompute and rollout_logprob methods crash after 60-80 steps, while this method maintains stable growth and outperforms TIS.

### Solution Seven: Engineering Tuning

Some practical engineering approaches:

| Method | Effect | Applicable Scenarios |
|------|------|----------|
| Lower top-p | Reduce low probability tokens | Sacrifices exploration |
| Switch GPU | H20 most stable | When hardware is flexible |
| Disable Cascade Attention | Significantly reduces mismatch on A100 | A100 users |
| FP32 LM Head | Slight improvement | Limited effectiveness |

## Monitoring Metrics

**vllm-kl** is an important early warning indicator:

$$\text{vllm-kl} = \mathbb{E}_{s, a \sim \pi_\text{vllm}} \left[ \log \frac{\pi_\text{vllm}(a|s)}{\pi(a|s)} \right]$$

Recommended to monitor simultaneously:
- **vllm-kl**: Degree of mismatch
- **fsdp-ppl**: Training engine perplexity
- **Gradient norm**: Stability indicator
- **Entropy**: Policy distribution health

When vllm-kl shows a spike, it often foreshadows an impending collapse.

## Practical Recommendations

1. **Accept that mismatch is inevitable**: This is a fundamental trade-off between speed and consistency, not a temporary bug.

2. **Use sequence-level corrections**: Token-level IS is theoretically biased and will fail on complex tasks. MIS or Geo-RS recommended.

3. **Monitor vllm-kl**: This is the most direct health indicator.

4. **Verify hardware impact**: Test on target hardware; results may not be portable.

5. **For MoE models**: Consider Routing Replay to stabilize expert routing.

6. **Trade-off in solution selection**:
   - Pursue highest performance → Bitwise Consistent Training (sacrifice speed)
   - Pursue practical balance → Sequence-Level MIS + appropriate top-p

## Summary

The stability problem in LLM-RL training is essentially a side effect of **modern system architecture division of labor**. The optimizations made by inference engines and training frameworks for efficiency are amplified into systemic instability through RL's closed-loop feedback.

Key insights for understanding this problem:
- **Seemingly on-policy but actually off-policy**
- **Token-level correction insufficient, sequence-level needed**
- **Low probability tokens are the weakest link**
- **High variance and entropy collapse require special handling**

As Reasoning RL and Agentic RL continue to develop, this problem will only become more important. I hope this article helps you avoid some pitfalls in practice.

## References

1. [When Speed Kills Stability](https://richardli.xyz/rl-collapse)
2. [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374)
3. [Bitwise Consistent Train-Inference](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html)
4. [DeepSeek-V3.2 Technical Report](https://arxiv.org/abs/2512.02556)
5. [Ring-1T: Scaling RL to Trillion Parameters](https://arxiv.org/abs/2510.18855)
6. [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
7. [KAT-Coder: MoE 模型 RL 训练稳定性](https://kwaikat.github.io/kwaikat-blog/posts/katcoder_1201/)
