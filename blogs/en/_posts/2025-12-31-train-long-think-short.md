---
layout: post
title: "Train Long, Think Short: A Survey on LLM Reasoning Length Control"
date: 2025-12-31 12:00:00
author: Qi Lu
tags: [RL, GRPO, Reasoning, Efficiency]
lang: en
translation: /train-long-think-short/
---

When training RLVR, even for simple problems, the model's chain of thought often runs thousands or even tens of thousands of tokens, yet mainstream commercial models like ChatGPT and Claude keep their reasoning remarkably concise. What's the difference?

With this question in mind, I surveyed research on **reasoning length control** and found quite a bit of work in this area, roughly falling into two categories: training-time optimization and inference-time control.

---

## 1. Background

### 1.1 The Overthinking Phenomenon

In RLVR (Reinforcement Learning with Verifiable Rewards) settings, reasoning models commonly show these problems:

- **Redundant verification**: Answer is correct, but model continues "Wait, let me verify..."
- **Repeated hesitation**: Using "Hmm", "Alternatively" to repeatedly switch approaches
- **Length inflation**: Small models require thousands of tokens for medium-difficulty reasoning

### 1.2 Optimization Objective

Minimize reasoning tokens without sacrificing accuracy:

$$\min_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot\mid x)}[\text{len}(y)] \quad \text{s.t.} \quad \text{Acc}(\pi) \geq \text{Acc}(\pi_0)$$

Evaluation metrics include:
- **Accuracy-Length Pareto Front**: Shorter at same accuracy, or more accurate at same length
- **Length distribution of correct samples**: Focus on long tail, not just mean

---

## 2. Training-Time Methods

### 2.1 Hard Truncation: ThinkPrune

**Paper**: [ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296)
**Date**: 2025-04
**Institution**: UCSB
**Code**: [GitHub](https://github.com/UCSB-NLP-Chang/ThinkPrune)

**Approach**: Set token limits during training; incomplete reasoning exceeding the limit gets truncated, resulting in zero reward. Iteratively tighten the limit to force the model to learn more concise reasoning.

**Method**:
1. Set initial length limit $L_0$
2. Samples exceeding limit cannot produce valid answers → reward = 0
3. Iteratively tighten: $L_{t+1} = \alpha \cdot L_t$, where $\alpha < 1$

**Results**:
- DeepSeek-R1-Distill-Qwen-1.5B on AIME24: length halved, accuracy drops only 2%
- DeepScaleR-1.5B-Preview: 5,914 → 3,370 tokens
- QwQ-32B: 8,763 → 4,494 tokens

**Pros**: No complex reward engineering needed
**Risk**: Over-tight limits may truncate correct solutions

---

### 2.2 Length Reward: GRPO-LEAD

**Paper**: [GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning](https://arxiv.org/abs/2504.09696)
**Date**: 2025-04
**Code**: [GitHub](https://github.com/aeroplanepaper/GRPO-LEAD)

**LEAD = Length-dependent rewards + Explicit penalties + Advantage reweighting for Difficulty**

This method includes three modifications:

1. **Length-dependent accuracy reward**: Rank correct samples by length, encouraging shorter correct solutions

2. **Explicit error penalties**: Additional negative constraints for incorrect samples

3. **Difficulty-aware advantage reweighting**: Weight by empirical solve rate, amplifying learning signal for harder problems

Notably, length ranking only applies within correct samples; errors are handled separately with penalties.

**Results**: 14B model achieves SOTA, significantly improving accuracy, conciseness, and efficiency.

---

### 2.3 Step Reward Shaping: LASER

**Paper**: [Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](https://arxiv.org/abs/2505.15612)
**Date**: 2025-05
**Code**: [GitHub](https://github.com/hkust-nlp/Laser)

This work proposes a unified framework formalizing efficient reasoning methods as length-based reward shaping. Based on this framework, the authors introduce **LASER (Length-bAsed StEp Reward shaping)** using step functions:

$$r_{\text{shaped}}(y) = r_{\text{task}}(y) + f(\text{len}(y))$$

**LASER-D (Dynamic and Difficulty-aware) extension**:
1. Reward schedule should adapt as model behavior evolves during training
2. Length rewards should be difficulty-aware—penalize long CoT more on easy problems

**Results**: LASER-D improves AIME2024 by +6.1 points while reducing token usage by 63%.

---

### 2.4 Adaptive Constraint: LEASH

**Paper**: [Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model](https://arxiv.org/abs/2512.21540)
**Date**: 2025-12

LEASH formulates length control as constrained optimization, using **Lagrangian Primal-Dual** to dynamically adjust penalty coefficients:

$$\max_\pi \mathbb{E}[r_{\text{task}}] \quad \text{s.t.} \quad \mathbb{E}[\text{len}(y)] \leq L_{\text{target}}$$

**Dynamic adjustment**:
- Generation exceeds target length → penalty increases
- Generation below target length → penalty relaxes

**One-sided penalty**: Only penalize "too long", avoiding incentives to become infinitely short.

**Results**: On Deepseek-R1-Distill-Qwen-1.5B and Qwen3-4B-Thinking-2507, average reasoning length reduced by 60% across tasks (including in-distribution math and OOD code/instruction-following) while maintaining competitive performance.

---

### 2.5 Curriculum Learning: Train Long, Think Short

**Paper**: [Train Long, Think Short: Curriculum Learning for Efficient Reasoning](https://arxiv.org/abs/2508.08940)
**Date**: 2025-08
**Code**: [GitHub](https://github.com/hammoudhasan/curriculum_grpo)

Uses a curriculum approach—first let the model "learn to solve", then gradually compress budget:

1. **Phase 1**: Generous token budget for exploring effective solution strategies
2. **Phase 2**: Gradually tighten budget, encouraging distillation into concise reasoning chains
3. **Combined training signal**: Correctness (verifier feedback) + length efficiency + format adherence

**Results**: On GSM8K, MATH500, SVAMP, College Math, GSM+, curriculum training consistently outperforms fixed-budget baselines at the same final budget.

---

### 2.6 Prompt-Controllable: L1 / LCPO

**Paper**: [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://arxiv.org/abs/2503.04697)
**Date**: 2025-03
**Homepage**: [CMU L3 Lab](https://cmu-l3.github.io/l1)

**LCPO (Length Controlled Policy Optimization)** embeds target length in the prompt:

- **LCPO-Exact**: "Think for exactly N tokens"
- **LCPO-Max**: "Think for maximum N tokens"

RL objective includes length deviation term for controllable budget reasoning.

**Results**:
- 1.5B L1 model outperforms GPT-4o at same reasoning length
- Outperforms s1 (Budget Forcing) baseline
- Can export Short Reasoning Models (SRMs): CoT length comparable to non-reasoning models while retaining reasoning mode

---

### 2.7 Difficulty-Adaptive Length Penalty: Just Enough Thinking

**Paper**: [Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning](https://arxiv.org/abs/2506.05256)
**Date**: 2025-06

LRMs often "overthink" simple problems—for instance, DeepSeek-R1 and Qwen-QwQ32B generate over 10,000 tokens for "2+3=?".

This work proposes **Adaptive Length Penalty (ALP)**, adjusting penalty based on each prompt's **online solve rate**:
- High solve rate (easy) prompts → higher extra token cost
- Low solve rate (hard) prompts → penalty unchanged

Simply put, save tokens on easy problems, reallocate budget to hard problems.

**Results**:
- DeepScaleR-1.5B with ALP post-training: **50% average token reduction**, minimal performance drop
- Higher accuracy on hardest problems compared to fixed-budget and uniform penalty baselines

---

### 2.8 Long2Short: Kimi k1.5

**Paper**: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
**Date**: 2025-01
**Institution**: Moonshot AI
**Code**: [GitHub](https://github.com/MoonshotAI/Kimi-k1.5)

Long CoT reasoning achieves high accuracy but incurs heavy compute costs. Kimi k1.5 introduces **Long2Short** techniques to compress long CoT strategies into efficient short CoT representations.

**Three Long2Short methods**:

| Method | Description |
|--------|-------------|
| **Model Merging** | Weight averaging of long-CoT and short-CoT models |
| **Shortest Rejection Sampling** | Select shortest correct response for SFT |
| **Preference-based RL** | Train model to prefer brevity while maintaining correctness |

**Results** (Short CoT SOTA):
- AIME 2024: **60.8**
- MATH500: **94.6**
- LiveCodeBench: **47.3**
- Outperforms GPT-4o and Claude Sonnet 3.5 by up to **+550%**

---

### 2.9 Length-Harmonizing Fine-Tuning: O1-Pruner

**Paper**: [O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning](https://arxiv.org/abs/2501.12570)
**Date**: 2025-01
**Code**: [GitHub](https://github.com/StarDewXXX/O1-Pruner)

O1-like long-thinking models struggle to effectively allocate token budgets based on problem difficulty and reasoning redundancy. O1-Pruner proposes **Length-Harmonizing Fine-Tuning** to address this:
1. **Pre-sampling**: Estimate model's baseline performance across problems
2. **RL-style Fine-tuning**: Encourage shorter reasoning under accuracy constraints

**Results**:
- Inference overhead reduced by **50%**
- Accuracy improves rather than drops
- Applicable to various mathematical reasoning benchmarks

---

### 2.10 Conciseness-Guided RL: ConciseRL

**Paper**: [ConciseRL: Conciseness-Guided Reinforcement Learning for Efficient Reasoning Models](https://arxiv.org/abs/2505.17250)
**Date**: 2025-05

Reasoning traces often extend beyond reaching correct answers, causing wasted computation, reduced readability, and even hallucinations. ConciseRL introduces a **hyperparameter-free conciseness score** as RL reward signal:
- Use LLM-as-judge to evaluate conciseness
- Dynamic, context-aware feedback (not just token count)

**Results**:
- TheoremQA: accuracy **+2.2%** while using **12.5x fewer** tokens
- Dynamically adjusts reasoning length based on problem difficulty
- Stronger judge models yield greater gains

---

## 3. Inference-Time Methods

### 3.1 Answer Convergence

**Paper**: [Answer Convergence as a Signal for Early Stopping in Reasoning](https://arxiv.org/abs/2506.02536)
**Date**: 2025-06

An interesting finding: on MATH and similar tasks, models typically converge to the final answer after **60% of reasoning steps**; the remaining content is mostly redundant.

Based on this observation, the authors propose three inference-time strategies:
1. **Answer Consistency early stopping**: Stop when consecutive reasoning chunks produce same answer
2. **Think Token Adjustment**: Increase probability of generating end-of-thinking signal
3. **Learn-to-Stop**: Train classifier on internal activations for "when to stop"

**Results**:
- Learn-to-Stop on NQ + QwQ-32B: 48% token reduction, sometimes improving accuracy
- Answer Consistency on NaturalQuestions: 40%+ token reduction with accuracy improvement

---

### 3.2 Step Answer Monitoring: ES-CoT

**Paper**: [Early Stopping Chain-of-thoughts in Large Language Models](https://arxiv.org/abs/2509.14004)
**Date**: 2025-09

A few key concepts:

- **Step Answer**: Model's current answer guess at each reasoning step
- **Run**: Consecutive sequence of steps with same answer
- **Run-Jump Test**: Terminate when run length of same step answer shows statistically significant jump

The idea is straightforward: "stop thinking when the answer stabilizes"—no extra model or retraining needed.

**Results**: Across 5 reasoning datasets and 3 LLMs, ES-CoT reduces generated tokens by **41%** on average while maintaining accuracy comparable to original CoT.

---

### 3.3 Transition Point Monitoring: DEER

**Paper**: [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)
**Date**: 2025-04
**Code**: [GitHub](https://github.com/iie-ycx/DEER)

DEER's observation: long CoT contains "pearl reasoning"—critical positions that are sufficient but not redundant.

The approach:
1. Monitor **Action Transition Points (ATP)**: Phrases like "Wait", "Alternatively" indicating approach switches
2. Induce trial answers at ATP
3. Use confidence to decide early exit—incomplete reasoning yields low confidence; sufficient reasoning yields high confidence

**Advantage**: No extra training needed, seamlessly integrates with existing o1-like reasoning LLMs.

**Results**: Across 10 reasoning benchmarks (GSM8K, MATH-500, AMC, GPQA, AIME, LiveCodeBench) and 11 frontier reasoning LLMs:
- CoT length reduced by 19.1% - 80.1% on average
- Accuracy improved by 0.3% - 5.0%

---

### 3.4 Three-Stage Reasoning Theory: Stop Spinning Wheels

**Paper**: [Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit](https://arxiv.org/abs/2508.17627)
**Date**: 2025-08

This work divides the reasoning process into **three stages**:
1. **Insufficient Exploration Stage**: Exploring problem space
2. **Compensatory Reasoning Stage**: Where correct answer typically emerges
3. **Reasoning Convergence Stage**: Often triggers overthinking

The key is finding the **Reasoning Completion Point (RCP)**—end of compensatory reasoning stage, typically at the first complete reasoning cycle.

Detection methods include:
- Query LLM sentence by sentence
- Monitor probability of `</think>` end-of-thinking tokens
- Mine more sensitive and consistent RCP patterns + lightweight threshold strategy

**Results**: On AIME24, AIME25, GPQA-D, reduces token consumption while maintaining or improving reasoning accuracy.

---

### 3.5 Budget Forcing: s1

**Paper**: [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
**Date**: 2025-01
**Code**: [GitHub](https://github.com/simplescaling/s1)

s1's approach is straightforward:
- Curate small dataset **s1K** with 1,000 question + reasoning trace pairs
- SFT on Qwen2.5-32B-Instruct (only 26 minutes on 16×H100)
- **Budget Forcing**: Control reasoning length by forced termination or repeatedly appending "Wait"

**Results**:
- s1-32B outperforms o1-preview by 27% on competition math (MATH and AIME24)
- Budget forcing improves AIME24 from 50% to 57%

---

### 3.6 Suppressing Reflection Tokens: NoWait

**Paper**: [Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency](https://aclanthology.org/2025.findings-emnlp.394/)
**Date**: EMNLP 2025
**arXiv**: [2506.08343](https://arxiv.org/abs/2506.08343)

Budget forcing isn't always effective across reasoning models. This work observes that explicit self-reflection ("Wait", "Hmm", "Alternatively") may not be necessary.

The method is simple: **logit suppression** on specific reflection/hesitation tokens at inference:
1. Identify key reflection words (via 32 independent runs, selecting most frequent single-word tokens)
2. Suppress these tokens during inference

**Results**: Across 5 R1-style model families (QwQ, Phi4, Qwen3, Kimi-VL, QvQ):
- CoT length reduced by **27%-51%**
- Maintains model utility across text, vision, and video reasoning tasks
- Plug-and-play, no training required

---

### 3.7 Dynamic Budget: ABF

**Paper**: [Reasoning at the Right Length: Adaptive Budget Forcing for Efficient and Accurate LLM Inference](https://openreview.net/forum?id=ieBgxTG7Mt)
**Date**: 2025-09

**Adaptive Budget Forcing (ABF)** dynamically adjusts reasoning length by monitoring real-time certainty signals (token-level confidence, entropy, semantic consistency):
- Sufficient confidence → stop generation
- Insufficient confidence → continue reasoning

**Difference from traditional Budget Forcing**: Traditional methods use fixed length constraints or predetermined control tokens; ABF monitors the model's "thinking trajectory" in real-time for adaptive stopping decisions.

---

## 4. Method Taxonomy

### Training-Time Methods

| Category | Core Idea | Representative Works |
|----------|-----------|---------------------|
| **Reward Shaping** | Add length penalty to RL reward, encouraging shorter correct reasoning | ThinkPrune, GRPO-LEAD, LASER, LEASH, Just Enough Thinking, ConciseRL |
| **Curriculum/Distillation** | First let model learn to solve, then gradually compress reasoning or distill from long to short CoT | Train Long Think Short, Kimi k1.5, O1-Pruner |
| **Prompt-Controllable** | Train model to control reasoning length based on budget instructions in prompt | L1/LCPO |

### Inference-Time Methods

| Category | Core Idea | Representative Works |
|----------|-----------|---------------------|
| **Early Stop Detection** | Monitor answer convergence, confidence, or reasoning completion signals for early termination | Answer Convergence, ES-CoT, DEER, Stop Spinning Wheels |
| **Token Intervention** | Control generation length via forced budgets, reflection word suppression, or dynamic thresholds | s1, NoWait, ABF |

---

## 5. Open Problems

1. **Accuracy-Efficiency Trade-off**: How to ensure compression doesn't hurt correctness?
2. **Difficulty Awareness**: Compress more on easy problems, preserve long thinking on hard ones
3. **Generalization**: Can training-time methods generalize to OOD tasks?
4. **Inference vs Training**: Can both approaches be effectively combined?

---

## References

### Training-Time Methods
- ThinkPrune: [arXiv:2504.01296](https://arxiv.org/abs/2504.01296)
- GRPO-LEAD: [arXiv:2504.09696](https://arxiv.org/abs/2504.09696)
- LASER: [arXiv:2505.15612](https://arxiv.org/abs/2505.15612)
- LEASH: [arXiv:2512.21540](https://arxiv.org/abs/2512.21540)
- Train Long, Think Short: [arXiv:2508.08940](https://arxiv.org/abs/2508.08940)
- L1/LCPO: [arXiv:2503.04697](https://arxiv.org/abs/2503.04697)
- Just Enough Thinking: [arXiv:2506.05256](https://arxiv.org/abs/2506.05256)
- Kimi k1.5: [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
- O1-Pruner: [arXiv:2501.12570](https://arxiv.org/abs/2501.12570)
- ConciseRL: [arXiv:2505.17250](https://arxiv.org/abs/2505.17250)

### Inference-Time Methods
- Answer Convergence: [arXiv:2506.02536](https://arxiv.org/abs/2506.02536)
- ES-CoT: [arXiv:2509.14004](https://arxiv.org/abs/2509.14004)
- DEER: [arXiv:2504.15895](https://arxiv.org/abs/2504.15895)
- Stop Spinning Wheels: [arXiv:2508.17627](https://arxiv.org/abs/2508.17627)
- s1: [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)
- NoWait: [arXiv:2506.08343](https://arxiv.org/abs/2506.08343)
- ABF: [OpenReview](https://openreview.net/forum?id=ieBgxTG7Mt)
