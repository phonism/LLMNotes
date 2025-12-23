---
layout: post
title: "Entropy Control in LLM-RL: A Systematic Survey from Entropy Collapse to Exploration"
date: 2025-12-23 12:00:00
author: Qi Lu
tags: [RL, Entropy, GRPO, RLVR, Negative Samples]
lang: en
translation: /rl/entropy-in-llm-rl/
---

## Introduction

In 2025, the LLM reinforcement learning community (especially RLVR: Reinforcement Learning with Verifiable Rewards) witnessed an explosion of research on **Entropy Collapse**. The core problem is: during RL training, the model's output diversity inevitably decreases, leading to loss of exploration capability and premature convergence to suboptimal solutions.

This article systematically reviews key works in this field **organized by publication timeline**, extracting core insights and formulas, followed by a unified analysis and critical reflection.

---

## Paper Timeline

### March 2025: DAPO — Industrial-Scale RL System

**Paper**: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
**Institution**: ByteDance Seed
**Date**: 2025-03-18

#### Core Problem

The DAPO team observed in large-scale RL training:
> "PPO and GRPO suffer from entropy collapse — entropy of policy decreases quickly with training, causing sampled responses to become identical."

#### Core Method: Clip-Higher (Decoupled Clipping)

Standard GRPO uses a single parameter $\epsilon$ for clipping:

$$\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$$

DAPO **decouples** it into $\epsilon_{\text{low}}$ and $\epsilon_{\text{high}}$:

$$\text{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})$$

**Key Configuration**: $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$

> "By increasing the value of $\epsilon_{\text{high}}$, we leave more room for the increase of low-probability tokens. This adjustment effectively enhances the policy's entropy."

#### Four Key Techniques

| Technique | Effect |
|-----------|--------|
| **Clip-Higher** | More room for low-probability tokens, mitigates entropy collapse |
| **Dynamic Sampling** | Improves training efficiency and stability |
| **Token-Level PG Loss** | Adapts to long CoT scenarios |
| **Overlong Reward Shaping** | Reduces reward noise from truncated samples |

#### Results

Based on Qwen2.5-32B, achieves 50 points on AIME 2024, surpassing DeepSeek-R1-Zero-Qwen-32B (47 points) with only 50% training steps.

---

### April 2025: VAPO — Value-Augmented Policy Optimization

**Paper**: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
**Institution**: ByteDance
**Date**: 2025-04-07

#### Core Contribution

VAPO introduces **7 innovative techniques** on top of PPO, significantly improving value learning and balancing exploration-exploitation.

#### Key Techniques

1. **Clip-Higher**: Adopts DAPO's asymmetric clipping ($\epsilon_{\text{high}} = 0.28$, $\epsilon_{\text{low}} = 0.2$)
2. **Value Learning Improvements**: More accurate value estimation reduces variance
3. **Exploration-Exploitation Balance**: Maintains stable entropy, neither collapsing nor exploding

#### Key Results

> "VAPO matches DAPO's performance using only **60% of DAPO's steps** and achieves a new SOTA score of 60.4 within just 5,000 steps."

> "VAPO maintains stable entropy — neither collapsing nor becoming excessively high."

---

### May 2025: SEED-GRPO — Semantic Entropy-Guided Optimization

**Paper**: [SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization](https://arxiv.org/abs/2505.12346)
**Date**: 2025-05-18

#### Core Problem

> "Vanilla GRPO treats all prompts equally during policy updates, ignoring important information about the model's knowledge boundaries."

#### Key Insight: Semantic Entropy vs Token Entropy

- **Token Entropy**: Measures uncertainty at individual token positions
- **Semantic Entropy**: Measures **semantic diversity** across multiple responses, clustering by meaning rather than form

> "Semantic entropy clusters responses based on meaning rather than form. This makes semantic entropy a more faithful and robust indicator of a model's uncertainty."

#### Core Formula

Given prompt $q$, sample $G$ responses $\{o_1, ..., o_G\}$, modulate advantage using semantic entropy:

$$\hat{A}_i = f(\text{SE}(q)) \cdot (r_i - \bar{r})$$

Where $\text{SE}(q)$ is semantic entropy, $f$ is the modulation function.

**Strategy**: High semantic entropy (model uncertain) → conservative update; Low semantic entropy (model confident) → aggressive update.

---

### May 2025: Unearthing Gems from Stones — Mining Correct Steps from Negative Samples

**Paper**: [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning](https://arxiv.org/abs/2505.14403)
**Institution**: CASIA, StepFun
**Date**: 2025-05-20

#### Core Problem

> "Negative responses contain valuable components such as self-reflection and error-correction steps, yet existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL)."

#### Core Method: BCPG-NSA

**Three-Stage Pipeline**:

1. **Sample Segmentation**: Use SAT model to segment long reasoning trajectories into independent steps
2. **Consensus Assessment**: LLM Judge + PRM dual judgment for step correctness
3. **NSA Optimization**: Give positive rewards to correct steps within negative samples

#### Core Idea

> "Mining positive steps within negative samples" — not simply penalizing entire negative samples, but **extracting** correct reasoning tokens.

---

### May 2025: The Entropy Mechanism — Mathematical Law of Entropy-Performance Trade-off

**Paper**: [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617)
**Institution**: PRIME-RL (Shanghai AI Lab)
**Date**: 2025-05-28

#### Core Discovery: R = -a·exp(H) + b

This is one of the most important theoretical discoveries in this field:

$$R = -a \cdot e^H + b$$

Where $R$ is downstream performance, $H$ is policy entropy, $a, b$ are fitting coefficients.

> "This empirical law strongly indicates that policy performance is traded from policy entropy, thus bottlenecked by its exhaustion. The ceiling is fully predictable: $H=0 \Rightarrow R = -a + b$."

**Implication**: Performance is gained at the cost of entropy; when entropy is exhausted, performance hits ceiling.

#### Core Theorem: Covariance Drives Entropy Change

For vanilla Policy Gradient, logit difference is:

$$\Delta z_{s,a} = \eta \cdot \pi_\theta(a|s) \cdot A(s,a)$$

**Entropy Change Formula (Theorem 1)**:

$$H(\pi^{k+1}_\theta|s) - H(\pi^k_\theta|s) \approx -\eta \cdot \text{Cov}_{a \sim \pi^k_\theta}[\log \pi^k_\theta(a|s), \pi^k_\theta(a|s) \cdot A(s,a)]$$

#### Key Insight

> "A high-probability action with high advantage would **reduce** policy entropy, while a rare action with high advantage would **increase** policy entropy."

| Scenario | Entropy Change |
|----------|----------------|
| High probability + High advantage | Decrease |
| Low probability + High advantage | Increase |

Empirically, the covariance term remains positive → entropy monotonically decreases.

#### Solutions: Clip-Cov and KL-Cov

- **Clip-Cov**: Randomly select a portion of positive-covariance tokens, detach their gradients
- **KL-Cov**: Apply KL penalty to highest-covariance tokens

$$\mathcal{L}_{\text{KL-Cov}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})_{\text{high-cov tokens}}$$

#### Results

| Model | Method | AIME24 | AIME25 |
|-------|--------|--------|--------|
| Qwen2.5-32B | GRPO | baseline | baseline |
| Qwen2.5-32B | **KL-Cov** | **+15.0%** | **+14.6%** |

Merged into [verl framework](https://verl.readthedocs.io/en/latest/algo/entropy.html).

---

### May 2025: OPO — Advantages of Exact On-Policy Training

**Paper**: [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2505.23585)
**Institution**: Microsoft Research
**Date**: 2025-05-29

#### Core View

OPO emphasizes the importance of **strict exact on-policy training**, contrasting with PPO/GRPO's multiple update strategy.

#### Two Innovations

1. **Exact On-Policy**: Single gradient update per batch (`ppo_mini_batch_size = train_batch_size`)
2. **Optimal Baseline**: Baseline depending on both policy and reward, minimizing gradient variance

#### Key Finding

> "Exact on-policy training demonstrates superior pass@1 performance and significantly **lower KL divergence and higher output entropy** throughout training compared to off-policy variants."

**Configuration**: `entropy_coeff: 0`, `use_kl_loss: False` — no explicit entropy regularization needed!

#### Results

| Benchmark | OPO | GRPO |
|-----------|-----|------|
| MATH-500 | 95.26% | 95.10% |
| AIME 2025 (Pass@16) | **85.33%** | 81.33% |

Merged into [verl framework](https://verl.readthedocs.io/en/latest/algo/opo.html).

---

### May 2025: Skywork-OR1 — MAGIC Pipeline with Adaptive Entropy Control

**Paper**: [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.22312)
**Institution**: Skywork AI
**Date**: 2025-05-29
**Open Source**: [GitHub](https://github.com/SkyworkAI/Skywork-OR1)

#### MAGIC Pipeline

**MAGIC** = Multi-stage Adaptive entropy scheduling for GRPO In Convergence

Core components:
- Rigorous data collection (offline + online filtering)
- Multi-stage training (progressively increasing context length)
- High-temperature sampling to enhance exploration

#### Adaptive Entropy Control

> "By leveraging **adaptive entropy control**, we maintain the model's entropy at a reasonable level throughout training and effectively prevent premature collapse."

Uses adaptive coefficient $\alpha_k$ to dynamically adjust entropy term weight.

#### Key Finding

> "Training approaches that **accelerate entropy collapse lead to worse test performance**."

---

### May 2025: ProRL — Stability for Prolonged RL Training

**Paper**: [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries](https://arxiv.org/abs/2505.24864)
**Date**: 2025-05-30

#### Core Problem

How to maintain stability in **prolonged RL training**?

#### Solutions

1. **KL Divergence Penalty**: Stronger stability than Clip-Higher
2. **Periodic Reset of Reference Policy**: Periodically reset reference policy

> "While DAPO and temperature adjustment help slow entropy collapse, **explicit regularization via KL divergence penalty provides a stronger and more stable solution**."

---

### June 2025: Beyond the 80/20 Rule — High-Entropy Minority Tokens Drive Effective RL

**Paper**: [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)
**Institution**: Qwen/Alibaba, Tsinghua University
**Date**: 2025-06-02
**Venue**: NeurIPS 2025

#### Core Discovery: 20% High-Entropy Tokens Determine Everything

> "Only a small fraction of tokens exhibit high entropy, and these tokens act as **critical forks** that steer the model toward diverse reasoning pathways."

**Token Distribution**:
- ~80% tokens: Low entropy (sentence completion, deterministic elements)
- ~20% tokens: High entropy (logical connectors like "however", "because", "thus")

#### Core Experiment

Training with only top 20% high-entropy token gradients:

| Model | Full Gradient | **Top 20% Only** |
|-------|---------------|------------------|
| Qwen3-32B (AIME'25) | baseline | **+11.04** |
| Qwen3-32B (AIME'24) | baseline | **+7.71** |
| Qwen3-14B (AIME'25) | baseline | **+4.79** |

> "Training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance."

#### Why It Works

> "RL tends to preserve or increase the entropy of forking tokens, maintaining flexible reasoning paths. In contrast, SFT reduces token entropy, leading to memorization and poor generalization."

#### Practical Configuration

Set `top_entropy_quantile = 0.2` in GRPO.

---

### June 2025: The Surprising Effectiveness of Negative Reinforcement

**Paper**: [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347)
**Date**: 2025-06-02
**Venue**: NeurIPS 2025

#### Core Concepts: PSR vs NSR

Decomposing RLVR's learning signal:

| Term | Meaning |
|------|---------|
| **PSR** (Positive Sample Reinforcement) | Reinforce correct answers |
| **NSR** (Negative Sample Reinforcement) | Penalize incorrect answers |

#### Surprising Finding

> "Training with **only negative samples** — without reinforcing correct responses — can be highly effective: it consistently improves performance over the base model across the entire Pass@k spectrum."

#### How NSR Works

> "NSR works by suppressing incorrect generations and **redistributing probability mass toward other plausible candidates**, guided by the model's prior beliefs. This effectively refines its existing knowledge without aggressively teaching new behaviors."

#### Core Formula: W-REINFORCE

$$\mathcal{L}_{\text{W-REINFORCE}}(\theta) = \lambda \cdot \mathcal{L}_{\text{PSR}}(\theta) + \mathcal{L}_{\text{NSR}}(\theta)$$

Recommended $\lambda = 0.1$ (significantly down-weighting positive samples).

#### Experimental Conclusions

| Method | Pass@1 | Pass@256 |
|--------|--------|----------|
| PSR only | Best | Poor |
| NSR only | Poor | **Near best** |
| **W-REINFORCE** | Good | **Best** |

NSR is crucial for maintaining entropy → diversity at large k.

---

### June 2025: Rewarding the Unlikely — Fixing GRPO's Rank Bias

**Paper**: [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355)
**Institution**: CMU
**Date**: 2025-06-03
**Venue**: EMNLP 2025

#### Core Problem: Rank Bias

> "A degenerate rank bias in GRPO in which **highly probable trajectories are reinforced and rare ones are neglected**. This results in distribution sharpening."

**Consequence**: Model only learns to solve already-solvable problems with fewer samples, but underperforms on Pass@N (large N) compared to simply sampling more from the original model.

#### Solution: Unlikeliness Reward

> "Explicitly up-weighting rare but correct solutions."

Give extra reward to low-probability but correct solutions.

---

### June 2025: LUFFY — Off-Policy Guidance Maintains High Entropy

**Paper**: [LUFFY: Learning to reason Under oFF-policY guidance](https://arxiv.org/abs/2506.07527)
**Date**: 2025-06-11

#### Core Problem

Limitation of on-policy RL: model can only learn from its own generations, unable to access superior reasoning patterns.

#### Solution

Introduce **off-policy guidance from stronger policies** (e.g., DeepSeek-R1).

#### Key Finding

> "LUFFY consistently sustains **higher entropy** compared to On-Policy RL throughout training. The generation entropy of On-Policy RL rapidly converges to nearly zero after ~200 steps, while the elevated entropy in LUFFY allows continuous exploration."

| Method | Entropy after 200 steps |
|--------|------------------------|
| On-Policy RL | ~0 |
| **LUFFY** | Remains high |

---

### June 2025: Dr. GRPO — Fixing GRPO's Bias

**Paper**: Dr. GRPO (Getting GRPO Done Right)
**Date**: 2025-06

#### Core Problem

GRPO's length normalization and std normalization may cause **biased optimization**, leading models to generate longer incorrect answers.

#### Solution

> "Removing both the length and std normalization terms in GRPO."

Simple but effective fix.

---

### October 2025: Rethinking Entropy Interventions — From an Entropy Change Perspective

**Paper**: [Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective](https://arxiv.org/abs/2510.10150)
**Institution**: Zhejiang University
**Date**: 2025-10-11

#### Core Criticism

> "Existing methods attempt to control entropy **indirectly** by only adjusting related factors such as the advantage signal and generation probability. Their effectiveness is inherently limited and prone to failure."

Existing methods (like Clip-Higher, advantage reweighting) only control entropy indirectly, with limited effectiveness.

#### Core Method: STEER

**STEER** = Stabilizing Token-level Entropy-changE via Reweighting

Core idea: Analyze **each token's entropy change**, directly control entropy dynamics at token level.

> "The overall entropy dynamics during training arises from the accumulation of per-token entropy changes."

---

### November 2025: Revisiting Entropy in RLVR — Positive Samples Are the Main Cause

**Paper**: [Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2511.05993)
**Date**: 2025-11-08

#### Core Discovery

> "Entropy collapse in RLVR **primarily arises from tokens with positive advantages**, and regulating their relative loss weights provides an effective means to control entropy."

**Counter-intuitive**: It's not negative samples causing entropy collapse, but positive samples!

#### Theoretical Explanation

Combined with the Entropy Mechanism paper's formula:

- High probability + Positive advantage → Large entropy decrease
- Low probability + Positive advantage → Entropy increase (but rare)

Since correct answers are often already high-probability, reinforcing them further sharpens the distribution.

#### Key Factors

Experiments identify three key factors affecting entropy:

1. **Number of off-policy updates**: More updates → easier collapse
2. **Training data diversity**: Lower diversity → easier collapse
3. **Clip threshold**: Improper settings accelerate collapse

---

### November 2025: EntroPIC — Stabilizing Entropy with Control Theory

**Paper**: [EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control](https://arxiv.org/abs/2511.15248)
**Institution**: Tencent AI Lab
**Date**: 2025-11-20

#### Core Idea

Use **PID controller** (industrial control theory) to stabilize entropy!

#### Core Formula

Let target entropy be $H_{\text{target}}$, current entropy be $H(n)$, error is:

$$e(n) = H_{\text{target}} - H(n)$$

**PI Control Signal**:

$$\alpha(n) = K_p \cdot e(n) + K_i \cdot \sum_{t=0}^{n} e(t)$$

Where:
- $K_p$: Proportional coefficient (responds to current error)
- $K_i$: Integral coefficient (responds to accumulated error)

Dynamically adjust positive/negative sample loss weights based on $\alpha(n)$.

#### Code Implementation

```python
# Integral term: accumulated entropy error
control_alpha_i = accumulate_entropy_error * K_i
# Proportional term: current entropy error
control_alpha_p = (entropy_loss - target_entropy) * K_p
# Total control signal
control_alpha = control_alpha_i + control_alpha_p
```

#### Results

> "Successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs. Validated through successful training on over 1M prompts."

Works for both on-policy and off-policy training.

---

### December 2025: SENT — Dual-Layer Semantic and Token Entropy Control

**Paper**: [Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning](https://arxiv.org/abs/2512.04359)
**Date**: 2025-12-04

#### Core Framework

SENT = Semantic ENtropy with Token-level entropy optimization

**Dual-Layer Design**:

| Level | Method | Effect |
|-------|--------|--------|
| **Data Level** | Semantic entropy-guided curriculum learning | Organize training data from easy to hard |
| **Algorithm Level** | Token-level entropy optimization | Apply KL regularization to low-entropy tokens |

#### Semantic Entropy Curriculum Learning

> "Organizing training data from low to high semantic entropy guides progressive optimization from easier to more challenging tasks."

**Principle**: Build reasoning capabilities on easier problems first, avoid encountering hard problems too early which leads to aggressive updates and entropy collapse.

---

## Unified Analysis

### 1. Root Cause of Entropy Collapse

Mathematically, entropy change is driven by **covariance**:

$$\Delta H \propto -\text{Cov}[\log \pi(a|s), \pi(a|s) \cdot A(s,a)]$$

| Scenario | Entropy Change | Frequency |
|----------|----------------|-----------|
| High probability + High advantage | Large decrease | High (correct answers are usually high-probability) |
| Low probability + High advantage | Increase | Low |
| Any + Negative advantage | Opposite effect | - |

**Conclusion**: Positive samples are the main cause of entropy collapse, because correct answers are often already high-probability.

### 2. Entropy-Performance Trade-off

$$R = -a \cdot e^H + b$$

This means:

- Entropy is a **consumable resource**
- Performance gains come at the cost of entropy
- When entropy is exhausted, performance hits ceiling ($H=0 \Rightarrow R_{\max} = -a + b$)

**Practical Implication**: This formula can **predict** the training performance ceiling.

### 3. Mitigation Methods Classification

| Category | Method | Representative Paper |
|----------|--------|---------------------|
| **Clipping Strategy** | Clip-Higher, decoupled $\epsilon$ | DAPO, VAPO |
| **Covariance Control** | Clip-Cov, KL-Cov | Entropy Mechanism |
| **Token Filtering** | Only use high-entropy token gradients | Beyond 80/20 |
| **Sample Reweighting** | W-REINFORCE, Unlikeliness Reward | NSR, Rewarding Unlikely |
| **Direct Entropy Control** | PID controller, adaptive coefficient | EntroPIC, Skywork-OR1 |
| **Entropy Change Aware** | Token-level entropy change reweighting | STEER |
| **Curriculum Learning** | Semantic entropy-ordered data | SENT, SEED-GRPO |
| **Negative Sample Mining** | Extract correct steps from incorrect answers | Unearthing Gems |
| **On-Policy Optimization** | Exact on-policy, optimal baseline | OPO |
| **Off-Policy Guidance** | External strong policy guidance | LUFFY |
| **KL Regularization** | KL penalty + periodic reset | ProRL |
| **Normalization Fix** | Remove length/std normalization | Dr. GRPO |

### 4. Role of Positive vs Negative Samples

| Sample Type | Effect on Entropy | Effect on Performance |
|-------------|-------------------|----------------------|
| **Positive** | Decrease entropy (sharpen distribution) | Improve Pass@1 |
| **Negative** | Maintain/increase entropy (preserve diversity) | Improve Pass@k (large k) |

**Best Practice**: W-REINFORCE recommends $\lambda = 0.1$, i.e., significantly down-weight positive samples.

### 5. Special Status of High-Entropy Tokens

Only ~20% of tokens are high-entropy, but they are:

- **Reasoning fork points** (like "however", "because")
- **Key determinants of reasoning path diversity**
- **What RL should focus on optimizing**

> "RL preserves entropy of forking tokens → flexible reasoning. SFT reduces all entropy → memorization."

### 6. Impact of Data Domain

| Data Type | Pretraining Exposure | Initial Entropy | Entropy Decay Rate |
|-----------|---------------------|-----------------|-------------------|
| Math/Code | High | Lower | Fast |
| Synthetic Logic-game | Low | Higher | Slow |

**Recommendation**: Use synthetic data not seen during pretraining (e.g., [SynLogic](https://github.com/MiniMax-AI/SynLogic)) to mitigate entropy collapse.

---

## Practical Recommendations

### Starter Configuration

1. Use **DAPO**'s Clip-Higher ($\epsilon_{\text{high}} = 0.28$)
2. Set `top_entropy_quantile = 0.2` to only use high-entropy token gradients
3. Use **W-REINFORCE** to down-weight positive samples ($\lambda = 0.1$)

### Advanced Configuration

1. Implement **Clip-Cov** or **KL-Cov** for covariance-based update control
2. Use **EntroPIC**'s PI controller for dynamic adjustment
3. Adopt **SENT**'s semantic entropy curriculum learning

### Monitoring Metrics

- **Policy Entropy**: Core health indicator
- **R vs H Curve**: Verify if it follows $R = -a \cdot e^H + b$
- **Token Entropy Distribution**: Ratio and position of high-entropy tokens

---

## Critical Reflection: Does Entropy Control Really Matter?

After surveying these 16 papers, we must ask a pointed question: **Are these entropy discussions truly important in industrial practice?**

### What Industry Actually Does

#### DeepSeek V3.2

DeepSeek V3.2's core techniques are:

```
1. Off-policy sequence masking (mask samples with advantage<0 and high off-policy degree)
2. Keep Routing (MoE-specific)
3. Keep Sampling Mask
4. Unbiased KL Estimation
```

**No explicit entropy control.**

#### Qwen MiniRL / GRPO

Main focus:
- Data filtering (accuracy within certain range)
- Within-group advantage normalization
- Clipping

**Also no explicit entropy control.**

### Entropy May Be Effect, Not Cause

These papers treat entropy as the core problem, but the actual causal chain might be:

```
Poor data quality / Training instability / Reward hacking
        ↓
    Entropy collapse (symptom)
        ↓
   Performance stagnation
```

**Industry may solve upstream problems directly, and entropy naturally stabilizes.**

DeepSeek V3.2's masking logic:

```python
if advantage < 0 and off_policy_degree > threshold:
    mask_this_sample()
```

This simple rule might be more effective than all entropy papers combined, because it directly addresses:
- Harmful gradients from off-policy samples
- Over-penalization of negative samples

### The "Hammer Looking for Nails" Effect in Academic Papers

Entropy is:
- ✓ An easy-to-define mathematical quantity
- ✓ An easy-to-measure metric
- ✓ An easy-to-formulate object

So many papers revolve around it. **But this doesn't mean it's the most important factor.**

### Re-evaluation: Which Findings Are Actually Valuable?

| Finding | Value | Reason |
|---------|-------|--------|
| **R = -ae^H + b** | ⭐⭐⭐ | Diagnostic tool, can predict training ceiling |
| **Positive samples cause entropy collapse** | ⭐⭐⭐ | Explains why down-weighting positive samples works |
| **20% high-entropy tokens are forks** | ⭐⭐ | Can reduce computation, but industry may not care |
| **Exact on-policy is more stable** | ⭐⭐⭐ | Engineering guidance, but sacrifices sample efficiency |
| **Various entropy control methods** | ⭐ | May be over-engineering |

### An Awkward Reality

These papers' experimental settings:
- Mostly based on **Qwen2.5 + AIME/MATH**
- Relatively small training scale (thousands to tens of thousands of steps)
- No comparison with DeepSeek V3.2-level baselines

Yet DeepSeek V3.2 achieved SOTA with simple masking, possibly suggesting:

> **With good enough data and training setup, entropy control is a secondary concern.**

### Revised Perspective

1. **Entropy is a useful monitoring metric** (like loss curve), but not an optimization target
2. **Explicit entropy control may only be necessary in specific scenarios**: limited data, weak models, long training
3. **Industry focuses more on upstream problems**: data quality, training stability, reward design
4. **These papers' value is more in theoretical understanding**, helping us know "why" rather than "must do this"

### When Should You Care About Entropy?

| Scenario | Need Explicit Entropy Control? |
|----------|-------------------------------|
| Abundant high-quality data + short training | ❌ Probably not |
| Limited data + need long training | ✓ Probably yes |
| Smaller model + prone to overfitting | ✓ Probably yes |
| Already using good masking/filtering | ❌ Entropy will naturally stabilize |

---

## Open Questions

1. **Entropy control vs data/training optimization**: Should we prioritize solving upstream problems rather than directly controlling entropy?

2. **Can we find an optimization objective that doesn't cause entropy decrease?** All current methods are mitigation, not cure.

3. **High entropy vs exploration efficiency trade-off**: High entropy helps exploration, but exploration efficiency may decrease (needs more steps to see effect).

4. **Cross-domain generalization**: Most conclusions are based on Qwen2.5 + Math/Code. Do they apply to other models and domains?

5. **Why doesn't industry use these methods?** Is it because they don't work, or because there are simpler alternatives?

---

## References

### Chronologically Ordered

1. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (2025.03, ByteDance)
2. [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118) (2025.04, ByteDance)
3. [SEED-GRPO: Semantic Entropy Enhanced GRPO](https://arxiv.org/abs/2505.12346) (2025.05)
4. [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation](https://arxiv.org/abs/2505.14403) (2025.05, CASIA/StepFun)
5. [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) (2025.05, Shanghai AI Lab)
6. [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.22312) (2025.05, Skywork AI)
7. [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2505.23585) (2025.05, Microsoft Research)
8. [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries](https://arxiv.org/abs/2505.24864) (2025.05)
9. [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL](https://arxiv.org/abs/2506.01939) (2025.06, NeurIPS 2025, Qwen/Alibaba)
10. [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347) (2025.06, NeurIPS 2025)
11. [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2506.02355) (2025.06, EMNLP 2025, CMU)
12. [LUFFY: Learning to reason Under oFF-policY guidance](https://arxiv.org/abs/2506.07527) (2025.06)
13. [Rethinking Entropy Interventions in RLVR](https://arxiv.org/abs/2510.10150) (2025.10, Zhejiang University)
14. [Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2511.05993) (2025.11)
15. [EntroPIC: Entropy Stabilization with Proportional-Integral Control](https://arxiv.org/abs/2511.15248) (2025.11, Tencent AI Lab)
16. [SENT: Semantic and Token Entropy for LLM Reasoning](https://arxiv.org/abs/2512.04359) (2025.12)

### Open Source Implementations

- [verl (Clip-Cov, KL-Cov, OPO)](https://verl.readthedocs.io/en/latest/algo/entropy.html)
- [DAPO](https://dapo-sia.github.io/)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1)
- [PRIME-RL/Entropy-Mechanism-of-RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)
- [TianHongZXY/RLVR-Decomposed (W-REINFORCE)](https://github.com/TianHongZXY/RLVR-Decomposed)
- [EntroPIC](https://github.com/yk7333/EntroPIC)
- [STEER](https://github.com/zz-haooo/STEER)
