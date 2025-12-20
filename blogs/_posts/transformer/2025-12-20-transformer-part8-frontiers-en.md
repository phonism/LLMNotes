---
layout: post
title: "Transformer Notes (VIII): Frontier Applications"
date: 2025-12-20 11:10:00
author: Phonism
tags: [Multimodal, Reasoning, GPT-4o, DeepSeek-R1, VLM]
lang: en
translation: /transformer-part8-frontiers/
---

This is the final article in the Transformer series, exploring **frontier applications** of large language models: multimodal large models and reasoning large models. These two directions represent the cutting edge of current AI research and are profoundly changing our understanding of intelligence.

## 1. Multimodal Large Models

As large language models achieve breakthroughs in the text domain, researchers have begun exploring how to combine visual, audio, and other multimodal information with language capabilities.

### 1.1 From Unimodal to Multimodal

Based on the depth and method of modality fusion, multimodal large models can be categorized as follows:

- **Cascaded**: Multiple independent models connected in series, such as using a vision model to extract descriptions first, then feeding them into a language model
- **Adapter-based**: Adding visual adapters on top of pre-trained LLMs, such as LLaVA, BLIP-2
- **Native**: Jointly trained on multimodal data from scratch, such as GPT-4o, Gemini

### 1.2 Core Challenges

**Modality Alignment**: Images are continuous pixel values, while text is a discrete sequence of tokens. How can we align them in the same semantic space?

**Information Compression**: A 224×224 image contains 50,176 pixels. How can we compress visual representations while retaining key information?

**Unifying Understanding and Generation**: Visual understanding requires high-level semantic abstraction, while image generation requires fine-grained pixel-level information.

### 1.3 Vision Encoders

**Vision Transformer (ViT)**: Divides images into fixed-size patches and processes them like text tokens:

$$\mathbf{z}_0 = [\mathbf{x}_\text{class}; \mathbf{E}\mathbf{x}_1; \mathbf{E}\mathbf{x}_2; ...; \mathbf{E}\mathbf{x}_N] + \mathbf{E}_\text{pos}$$

**CLIP**: Trained on 400 million image-text pairs through contrastive learning, aligning image representations with text descriptions in semantic space.

**SigLIP**: Uses sigmoid loss instead of softmax, allowing training with larger batch sizes. Widely adopted by new-generation models like InternVL and Qwen2-VL.

### 1.4 Modality Fusion Mechanisms

| Method | Representative Models | Additional Parameters | Visual Tokens | Characteristics |
|--------|----------------------|----------------------|---------------|-----------------|
| Linear Projection | LLaVA | ~2M | 576 | Simple and efficient |
| MLP Projection | LLaVA-1.5 | ~20M | 576 | More expressive |
| Q-Former | BLIP-2 | ~107M | 32 | Compresses visual information |
| Cross-Attention | LLaMA 3.2 | ~1B | Variable | Deep fusion |

**LLaVA**: Uses simple linear/MLP projection to connect CLIP ViT and LLM, proving that "simple can be effective."

**BLIP-2**: Q-Former uses 32 learnable query embeddings to extract information from visual features through cross-attention.

**LLaMA 3.2 Vision**: Inserts cross-attention layers inside the LLM to achieve deep fusion.

### 1.5 Native Multimodal Models

"Native multimodal" refers to models designed with multimodal processing capabilities from the start, rather than "grafting" them onto unimodal models.

**GPT-4o**: OpenAI's first native multimodal flagship model
- Single model processes text, audio, and visual inputs end-to-end
- Real-time voice conversation latency reduced to 232ms (approaching human reaction speed)
- Audio input preserves non-semantic information like tone and emotion

**Gemini**: Google's native multimodal model series
- Early fusion architecture, jointly trained on multimodal data from the pre-training stage
- Supports 32K to 1M token context

**Chameleon**: Meta's open-source native multimodal model
- Represents all modalities as discrete tokens with a unified vocabulary
- Uses standard Transformer architecture to process mixed-modality sequences
- Trained with 5 million A100 GPU hours

### 1.6 Unifying Understanding and Generation

Traditional multimodal models either focus on understanding or generation. Recent research has begun exploring unifying both capabilities in a single model.

**Show-o**: Uses Omni-Attention (causal attention for text, full attention for images), combining autoregressive and discrete diffusion modeling.

**Janus**: DeepSeek's "decoupled encoding, unified processing" strategy
- Understanding encoder: SigLIP, extracts high-level semantic features
- Generation encoder: VQ tokenizer, produces discrete visual representations
- Shared Transformer for unified processing

### 1.7 Multimodal Post-Training

**Visual Instruction Tuning**: Training models to follow vision-related instructions using high-quality multimodal instruction data. LLaVA first used GPT-4 to generate multimodal instruction data.

**Multimodal RLHF**: LLaVA-RLHF addresses multimodal hallucination issues, training a reward model using 10K human preference data and optimizing through PPO.

**mDPO**: Addresses the problem of standard DPO ignoring image conditions in multimodal scenarios by explicitly optimizing image preferences.

**Multimodal Hallucinations**: Model-generated content that doesn't match the input image
- Object hallucination: Describing objects not present in the image
- Attribute hallucination: Incorrectly describing object color, size, or position
- Relationship hallucination: Incorrectly describing relationships between objects

**LLaVA-Critic**: The first open-source multimodal general evaluation model, implementing a "self-rewarding" path for self-improvement.

## 2. Reasoning Large Models

A series of breakthroughs in 2024 revealed a new dimension: sometimes, **letting the model answer more slowly** can lead to better results.

### 2.1 From Fast Thinking to Slow Thinking

Traditional LLMs adopt "System 1"-style fast responses, which have limitations on tasks requiring complex reasoning:

- **Limited reasoning depth**: Lack the ability to "go back and check"
- **Error accumulation**: Early errors in reasoning chains propagate
- **Lack of planning**: Can only "walk and see"

**Test-Time Compute Scaling**: Investing more computational resources during inference to achieve better output quality.

Main approaches:
1. **Search**: Generate multiple candidate answers and use a verifier to select the best
2. **Thinking**: Let the model "think" longer, generating detailed reasoning processes
3. **Iteration**: Multiple rounds of self-correction and optimization

### 2.2 Chain-of-Thought and Self-Consistency

**Chain-of-Thought (CoT)**: Guides the model to generate intermediate reasoning steps. Zero-shot CoT only needs to add "Let's think step by step" to activate reasoning capabilities.

**Self-Consistency**: Generate multiple reasoning paths for the same problem and select the most consistent answer through majority voting.
- GSM8K: +17.9%
- MATH: +12.2%

### 2.3 Reward Models and Verifiers

**Outcome Reward Model (ORM)**: Only provides reward signals for the final answer. Low annotation cost but difficult credit assignment.

**Process Reward Model (PRM)**: Provides reward signals for each step of reasoning. OpenAI experiments show PRM achieves 78.2% on MATH, significantly outperforming ORM.

| Feature | ORM | PRM |
|---------|-----|-----|
| Feedback Granularity | Overall result | Each step |
| Annotation Cost | Low | High |
| Credit Assignment | Difficult | Precise |
| Search Efficiency | Lower | Higher |

### 2.4 OpenAI o1

OpenAI o1 (September 2024) is the first large-scale commercial reasoning large model.

**Key Features**:
- **Reasoning tokens**: Model generates internal reasoning process before answering
- **Hidden thinking**: Reasoning tokens are invisible to users (but are billed)
- **Reinforcement learning training**: Learns "how to think" through large-scale RL

**Performance**:

| Benchmark | GPT-4o | o1-preview | o1 |
|-----------|--------|------------|-----|
| AIME 2024 | 12% | 44% | 74% |
| MATH-500 | 60.3% | 85.5% | 94.8% |
| GPQA Diamond | 50.6% | 73.3% | 78.0% |

**Scaling Laws**: o1 demonstrates two dimensions of scaling—more RL training brings stronger reasoning capabilities, and longer thinking time brings better answer quality.

### 2.5 DeepSeek-R1

DeepSeek-R1 (January 2025) is the first open-source model to prove that **pure reinforcement learning can activate reasoning capabilities**.

**Key Findings from R1-Zero**:
- No need for SFT, strong reasoning capabilities can be obtained through RL alone
- Advanced reasoning patterns such as self-reflection, verification, and dynamic strategy adjustment emerge
- AIME 2024: improved from 15.6% to 71.0%, with majority voting reaching 86.7%

**GRPO Algorithm**: Group Relative Policy Optimization
- Eliminates the need for a Critic model of the same scale as the policy model
- Uses within-group relative scores as baseline estimates
- Significantly reduces training costs

$$\mathcal{L}_\text{GRPO} = -\mathbb{E}_{x, \{y_i\}}\left[\sum_i \frac{r(x, y_i) - \bar{r}}{\sigma_r} \log \pi_\theta(y_i|x)\right]$$

**Emergent Capabilities**:
- **Self-reflection**: "Wait, let me reconsider..."
- **Verification**: Checking the correctness of intermediate steps
- **Backtracking**: Going back and retrying after discovering errors
- **Strategy switching**: Trying another approach when one doesn't work

### 2.6 Knowledge Distillation

DeepSeek proved that reasoning capabilities can be transferred to smaller models through distillation.

| Model | Base | AIME 2024 | MATH-500 |
|-------|------|-----------|----------|
| R1-Distill-Qwen-1.5B | Qwen2.5-1.5B | 28.9% | 83.9% |
| R1-Distill-Qwen-7B | Qwen2.5-7B | 55.5% | 92.8% |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 72.6% | 94.3% |
| R1-Distill-Llama-70B | Llama3.3-70B | 70.0% | 94.5% |

**Key Finding**: R1-Distill-Qwen-32B surpasses o1-mini, proving distillation is an efficient path to reasoning capabilities.

### 2.7 Open-Source Reasoning Models

**QwQ (Qwen with Questions)**: Released by Alibaba's Qwen team (November 2024)
- 32B parameters, 32K context length
- GPQA: 65.2%, AIME 2024: 50.0%

**Mainstream Reasoning Model Comparison**:

| Model | Parameters | Open-source | AIME | MATH | Release |
|-------|-----------|-------------|------|------|---------|
| o1 | - | No | 74% | 94.8% | 2024.12 |
| DeepSeek-R1 | 671B | Yes | 79.8% | 97.3% | 2025.01 |
| QwQ-32B | 32B | Yes | 50% | 90.6% | 2024.11 |
| R1-Distill-32B | 32B | Yes | 72.6% | 94.3% | 2025.01 |

### 2.8 Applications and Limitations

**Applicable Scenarios**:
- Mathematical problems: Competition mathematics, theorem proving
- Code generation: Complex algorithms, debugging
- Scientific reasoning: Physics, chemistry problems
- Logical reasoning: Planning, constraint satisfaction

**Current Limitations**:
- High latency: Not suitable for real-time interaction
- High cost: Reasoning tokens consume significant computation
- Over-thinking: Even simple problems may produce lengthy reasoning
- Circular reasoning: May get stuck in meaningless thought loops

## 3. Future Directions

### 3.1 Multimodal Reasoning

Extending reasoning capabilities to multimodal:
- Visual reasoning: Logical relationships in images
- Video understanding: Temporal reasoning
- Embodied intelligence: Planning in the physical world

### 3.2 Unifying All Modalities

Most current models primarily handle images and text. The future will extend to:
- Native support for audio/speech
- Video understanding and generation
- 3D scene understanding

### 3.3 Reasoning and Agents

Reasoning large models provide stronger planning capabilities for AI Agents:
- Task decomposition and planning
- Tool usage decisions
- Long-term goal tracking

### 3.4 Efficiency Improvements

- Compute-optimal strategies: Dynamically adjust test-time compute based on task difficulty
- Early stopping strategies: Stop early when answer convergence is detected
- Lightweight models: Distill smaller reasoning models

## 4. Series Summary

This 8-article series provides a comprehensive analysis of the Transformer architecture and its applications in large language models:

| Article | Topic | Core Content |
|---------|-------|--------------|
| I | Fundamentals | Hardware background, Transformer computation, Scaling Law |
| II | Core Components | Tokenizer, positional encoding (RoPE), gating mechanisms |
| III | Attention Mechanisms | FlashAttention, MLA, sparse/linear attention |
| IV | Model Architecture | MoE sparse architecture, load balancing |
| V | Training Techniques | Data engineering, distributed training, Muon optimizer |
| VI | Evaluation Systems | MMLU, LiveCodeBench, Chatbot Arena |
| VII | Deployment Optimization | Quantization, inference engines, speculative decoding |
| VIII | Frontier Applications | Multimodal, reasoning large models |

Since the Transformer paper was published in 2017, this architecture has completely transformed the field of artificial intelligence. Looking ahead:

- **Larger scale**: Trillion-parameter models will become standard
- **Longer context**: Million-token-level processing capabilities
- **Stronger reasoning**: Paradigm shift from "fast thinking" to "slow thinking"
- **More modalities**: Truly "omnipotent" artificial intelligence

We are in the golden age of artificial intelligence development. I hope this series has helped you deeply understand the core of this technological revolution.

---

*Series completed. Thank you for reading!*
