---
layout: post
title: "Transformer Notes (VIII): Frontier Applications"
date: 2025-12-20 11:10:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [Transformer, Multimodal, Reasoning]
lang: en
translation: /transformer-part8-frontiers/
series: transformer
series_order: 8
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

Building multimodal large models faces several key challenges:

**Modality Alignment**: Images and text exist in different representation spaces, requiring effective cross-modal mapping. Images are continuous pixel values, while text is a discrete sequence of tokens. How to align them in the same semantic space is the core problem.

**Information Compression**: A 224×224 image contains 50,176 pixels, while typical vision encoders produce 196-576 visual tokens. How can we compress visual representations while retaining key information, avoiding excessive sequence length burden on the LLM?

**Unifying Understanding and Generation**: Visual understanding (like VQA) requires high-level semantic abstraction, while image generation requires fine-grained pixel-level information. How can we support both seemingly contradictory requirements in a single model?

### 1.3 Vision Encoders

Vision encoders are the "eyes" of multimodal large models, responsible for converting images into representations that language models can understand.

#### Vision Transformer (ViT)

Vision Transformer applies the Transformer architecture to image processing. Its core idea is to divide images into fixed-size patches, then process these patches like text tokens:

$$\mathbf{z}_0 = [\mathbf{x}_\text{class}; \mathbf{E}\mathbf{x}_1; \mathbf{E}\mathbf{x}_2; ...; \mathbf{E}\mathbf{x}_N] + \mathbf{E}_\text{pos}$$

where $\mathbf{x}\_{i} \in \mathbb{R}^{P^2 \cdot C}$ is the flattened vector of the $i$-th image patch, $\mathbf{E}$ is the patch embedding matrix, and $\mathbf{E}\_{\text{pos}}$ is the positional encoding.

#### CLIP and Contrastive Learning

CLIP (Contrastive Language-Image Pre-training) trains vision encoders on 400 million image-text pairs through contrastive learning, aligning image representations with corresponding text descriptions in semantic space:

$$\mathcal{L}_\text{CLIP} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j)/\tau)}\right]$$

CLIP's vision encoder (typically ViT-L/14) became the standard choice for early multimodal large models due to its powerful cross-modal alignment capability.

#### SigLIP and Improvements

SigLIP improves CLIP's training objective by using sigmoid loss instead of softmax:

$$\mathcal{L}_\text{SigLIP} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\sigma(y_{ij} \cdot \text{sim}(\mathbf{v}_i, \mathbf{t}_j) \cdot \tau)$$

where $y_{ij} = 1$ when $i=j$, otherwise $y_{ij} = -1$. This design allows training with larger batch sizes without requiring global negative sample synchronization, making training more efficient. SigLIP is widely used in new-generation models like InternVL and Qwen2-VL.

### 1.4 Modality Fusion Mechanisms

The way visual features are injected into language models determines the architectural design of multimodal models. Current mainstream fusion mechanisms include:

#### Linear/MLP Projection

The simplest approach is to use linear layers or MLP to map visual features to the language model's embedding space:

$$\mathbf{H}_v = \mathbf{W}_\text{proj} \cdot \mathbf{Z}_\text{vision} + \mathbf{b}$$

**LLaVA** initially adopted this approach, connecting CLIP ViT-L/14 and Vicuna through a simple linear projection matrix:
- Keeps vision encoder and LLM parameters frozen
- Only trains the projection matrix (~2M parameters)
- Two-stage training: pre-training alignment + instruction tuning

**LLaVA-1.5** upgraded linear projection to a two-layer MLP, significantly improving multimodal capabilities:

$$\mathbf{H}_v = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{Z}_\text{vision})$$

#### Q-Former (Querying Transformer)

BLIP-2 proposed the Q-Former architecture, using learnable query tokens to extract information from visual features through cross-attention:

$$\mathbf{Q}_\text{out} = \text{CrossAttn}(\mathbf{Q}_\text{learnable}, \mathbf{K}_\text{vision}, \mathbf{V}_\text{vision})$$

Q-Former's core design:
- 32 learnable query embeddings (dimension 768)
- Transformer blocks initialized from BERT
- Cross-attention and self-attention layers alternately stacked
- Outputs fixed number of visual tokens (32), regardless of input image resolution

**Two-stage Pre-training**:
1. **Vision-language representation learning**: Train Q-Former with frozen vision encoder using ITC, ITM, ITG losses
2. **Vision-language generative learning**: Connect Q-Former output to frozen LLM, train generation capability

BLIP-2 surpasses Flamingo-80B by 8.7% on VQAv2 zero-shot, with only 1/54 of trainable parameters.

#### Cross-Attention Adapter

Flamingo and LLaMA 3.2 Vision adopt the approach of inserting cross-attention layers inside the LLM:

$$\mathbf{h}_l' = \mathbf{h}_l + \text{CrossAttn}(\mathbf{h}_l, \mathbf{K}_\text{vision}, \mathbf{V}_\text{vision})$$

**LLaMA 3.2 Vision** is built on LLaMA 3.1:
- Adds visual adapters on frozen LLaMA 3.1 text model
- Adapters include multi-layer cross-attention to inject image encoder representations into LLM
- Updates vision encoder and adapters during training, but freezes LLM parameters
- Maintains text capabilities unchanged, achieving "plug-and-play" replacement of LLaMA 3.1

#### Fusion Mechanism Comparison

| Method | Representative Model | Additional Parameters | Visual Tokens | Characteristics |
|--------|----------------------|----------------------|---------------|-----------------|
| Linear Projection | LLaVA | ~2M | 576 | Simple and efficient |
| MLP Projection | LLaVA-1.5 | ~20M | 576 | More expressive |
| Q-Former | BLIP-2 | ~107M | 32 | Compresses visual information |
| Cross-Attention | LLaMA 3.2 | ~1B | Variable | Deep fusion |

### 1.5 Representative Multimodal Models

#### LLaVA Series

LLaVA (Large Language and Vision Assistant) is one of the most influential open-source multimodal large models.

**LLaVA-1.0**:
- Vision encoder: CLIP ViT-L/14 (frozen)
- Language model: Vicuna-7B/13B (frozen)
- Connection: Linear projection layer
- Training data: 595K image-text pairs (pre-training) + 158K visual instruction data (fine-tuning)

**LLaVA-1.5 Improvements**:
- MLP replaces linear projection
- Input resolution increased from 224 to 336
- Added academic VQA data
- Larger language model (Vicuna-13B)

**LLaVA-NeXT** further supports dynamic resolution, dividing images into multiple sub-images for separate encoding.

#### Qwen-VL Series

**Qwen-VL** uses a larger vision encoder and higher resolution:
- Vision encoder: OpenCLIP ViT-bigG (448×448)
- Language model: Qwen-7B
- Connection: Single-layer cross-attention

**Qwen2-VL Innovations**:
- **Dynamic resolution**: Removes ViT's absolute position encoding, introduces 2D-RoPE, supports arbitrary resolution input
- **M-RoPE**: Multimodal Rotary Position Embedding, decomposing rotary position encoding into temporal and spatial (height, width) components
- **Token compression**: MLP layer compresses adjacent 2×2 tokens into 1, a 224×224 image produces only 66 visual tokens

#### InternVL Series

InternVL's unique design lies in scaling up the vision encoder:
- Vision encoder expanded to 6 billion parameters (InternViT-6B)
- Introduces QLLaMA as a "glue layer" (8B parameters) connecting vision and language
- Three-stage training: contrastive learning → generative learning → instruction tuning

InternVL 2.5 is the first open-source model to break 70% on the MMMU benchmark, reaching GPT-4o level.

### 1.6 Native Multimodal Models

"Native multimodal" refers to models designed with multimodal processing capabilities from the start, rather than "grafting" them onto unimodal models.

**Non-native multimodal** (e.g., ChatGPT with GPT-4V):
- Text generation: GPT-4
- Image understanding: GPT-4V (separate vision module)
- Speech recognition: Whisper
- Image generation: DALL-E 3
- Modules are independent, connected via API or text intermediary

**Native multimodal** (e.g., GPT-4o, Gemini):
- Single neural network processes all modalities end-to-end
- Jointly trained on multimodal data from scratch
- Shared representation space across modalities for deep fusion
- No text intermediary between modalities, reducing information loss

#### GPT-4o

GPT-4o ("o" stands for "omni") was released in May 2024 as OpenAI's first native multimodal flagship model.

**Core Features**:
- Single model processes text, audio, and visual inputs end-to-end
- Can directly generate text, audio, and image outputs
- Real-time voice conversation latency reduced to 232ms (approaching human reaction speed)
- Audio input preserves non-semantic information like tone and emotion

**Difference from GPT-4V**:
- GPT-4V: Upload image → Vision model recognition → Convert to text description → GPT-4 processing → Generate response
- GPT-4o: Upload image → Direct understanding and response generation (no intermediate conversion)

#### Google Gemini

Gemini is Google's native multimodal model series.

**Technical Report Statement**:
> "Gemini models are natively multimodal, as they are trained jointly across text, image, audio, and video."

**Architecture Features**:
- Early Fusion architecture
- Jointly trained on multimodal data from the pre-training stage
- Supports 32K (Gemini 1.0) to 1M (Gemini 1.5/2.5) token context

**Model Series**:
- Gemini Ultra: Largest scale, first to surpass human expert level on MMLU
- Gemini Pro: Balanced performance and efficiency
- Gemini Nano: Optimized for on-device deployment
- Gemini 2.5 Pro: Released in 2025, adds "thinking model" capability

#### Meta Chameleon

Chameleon is Meta's open-source native multimodal model, adopting a thorough early fusion architecture.

**Core Design**:
- Represents all modalities (images, text, code) as discrete tokens
- Unified vocabulary includes text, code, and image tokens
- Uses standard Transformer architecture to process mixed-modality sequences
- End-to-end training from scratch, no separate image encoder/decoder needed

**Image Discretization**: Uses improved VQ-VAE to encode images as discrete tokens:
- Images encoded as 1024 discrete tokens (32×32 latent grid)
- Codebook size 8192
- Shares unified embedding space with text tokens

**Training Scale**:
- 7B and 34B parameter versions
- Approximately 4.4 trillion tokens training data (text, image-text pairs, interleaved sequences)
- Over 5 million A100 GPU hours

### 1.7 Unifying Understanding and Generation

Traditional multimodal models either focus on understanding (like VQA) or generation (like text-to-image). Recent research has begun exploring unifying both capabilities in a single model.

#### Challenges and Contradictions

Understanding and generation have different requirements for visual representations:
- **Understanding**: Needs high-level semantic abstraction, focusing on "what is it"
- **Generation**: Needs fine-grained details, focusing on "how to draw it"

Using the same visual encoder for both tasks creates conflicts—semantic encoders (like CLIP) excel at understanding but generate images lacking details; pixel encoders (like VQ-GAN) can reconstruct details but have weak semantic understanding.

#### Show-o

Show-o proposes using a single Transformer to unify understanding and generation:

**Core Design**:
- **Omni-Attention**: Causal attention for text tokens, full attention for image tokens
- **Mixed modeling**: Autoregressive generation for text, discrete diffusion model for images
- **Unified vocabulary**: Text tokens and image tokens (VQ-GAN encoded) share the vocabulary

**Task Capabilities**:
- Image Captioning
- Visual Question Answering (VQA)
- Text-to-Image Generation
- Image Editing (Inpainting/Outpainting)
- Mixed-modality generation

Show-o outperforms larger models like NExT-GPT and Chameleon on VQAv2, while achieving FID 9.24 (MSCOCO 30K) on image generation.

#### Janus

DeepSeek's Janus adopts a "decoupled encoding, unified processing" strategy:

**Core Insight**: Understanding and generation need different visual encodings, but can share language model processing.

**Dual Encoder Design**:
- **Understanding encoder**: SigLIP, extracts high-level semantic features
- **Generation encoder**: VQ tokenizer, produces discrete visual representations
- **Shared Transformer**: Unified processing of token sequences from both encoders

**Janus-Pro** (January 2025) further improves:
- Based on DeepSeek-LLM-7B
- MMBench reaches 79.2 (surpassing LLaVA-v1.5)
- Image generation FID 8.53 (MSCOCO 30K)

#### JanusFlow

JanusFlow changes the generation end from discrete tokens to continuous flow (Rectified Flow):
- Understanding end remains unchanged (SigLIP encoder)
- Generation end uses Rectified Flow instead of VQ tokenizer
- Image generation quality further improved

### 1.8 Visual Tokenizer

Visual tokenizers are key components of native multimodal and unified models, responsible for converting continuous images into discrete tokens.

#### VQ-VAE and VQ-GAN

**VQ-VAE** first proposed mapping continuous representations to a learnable discrete codebook:

$$z_q = \arg\min_{e_k \in \mathcal{C}} \|z_e - e_k\|_2$$

where $z_e$ is the encoder output, and $\mathcal{C}$ is the codebook.

**VQ-GAN** introduces adversarial loss on top of VQ-VAE:

$$\mathcal{L}_\text{VQ-GAN} = \mathcal{L}_\text{rec} + \mathcal{L}_\text{commit} + \mathcal{L}_\text{GAN} + \mathcal{L}_\text{perceptual}$$

VQ-GAN can encode a 256×256 image into 16×16=256 discrete tokens, each token from a codebook of size 1024-16384.

#### Tokenizer Type Comparison

| Type | Representative | Codebook | Characteristics |
|------|----------------|----------|-----------------|
| Pixel-level | VQ-GAN | 8K-16K | High reconstruction quality, weak semantics |
| Semantic-level | CLIP-ViT | - | Strong semantics, cannot reconstruct |
| Hybrid | SEED | 8K | Balances semantics and reconstruction |
| Unified | TokenFlow | 16K | Dual encoder + shared mapping |

### 1.9 Multimodal Post-Training

Multimodal post-training is crucial for aligning with human preferences and improving instruction-following capabilities.

#### Visual Instruction Tuning

Training models to follow vision-related instructions using high-quality multimodal instruction data. LLaVA pioneered multimodal instruction tuning by using GPT-4 to generate multimodal instruction data:
- Using COCO dataset image annotations (bounding boxes, captions)
- Inputting visual information as prompts to GPT-4
- Generating 158K high-quality multimodal conversations, complex reasoning, and detailed descriptions

#### Multimodal RLHF

LLaVA-RLHF addresses multimodal hallucination issues:
- Training a reward model using 10K human preference data
- Optimizing the policy model through PPO (Proximal Policy Optimization)
- Significantly reducing hallucination rate and improving factual accuracy

#### mDPO (Multimodal DPO)

Standard DPO has issues in multimodal scenarios: when images are the same condition in preferred and rejected samples, the image condition cancels out in DPO's optimization objective, causing the optimization process to ignore visual information.

mDPO introduces anchor samples to explicitly optimize image preferences:

$$\mathcal{L}_\text{mDPO} = \mathcal{L}_\text{DPO}(y_w, y_l | x, v) + \lambda \cdot \mathcal{L}_\text{anchor}(y_w | v, v')$$

where $v'$ is a reference image different from $v$, and $\mathcal{L}_\text{anchor}$ ensures the model attends to image differences.

#### Multimodal Hallucinations

Model-generated content that doesn't match the input image is a major problem for multimodal models:

| Hallucination Type | Description | Example |
|--------------------|-------------|---------|
| Object hallucination | Describing objects not in the image | Saying "there's a cat" when there isn't |
| Attribute hallucination | Incorrectly describing object attributes | Calling a red car blue |
| Relationship hallucination | Incorrectly describing relationships | "Person riding horse" but actually standing beside it |
| Quantity hallucination | Incorrectly counting objects | Saying 5 apples when there are 3 |

**Causes of Hallucination**:
- LLM's language prior: Tendency to generate descriptions following language statistics
- Insufficient visual information utilization: Model may over-rely on text context
- Training data bias: Certain object-attribute combinations are more common in training data

#### LLaVA-Critic

LLaVA-Critic is the first open-source multimodal general evaluation model, capable of evaluating output quality from other multimodal models.

**Core Capabilities**:
- **Reference-free evaluation**: Directly evaluates generation quality without ground truth
- **Pairwise comparison**: Judges which of two responses is better
- **Multi-dimensional scoring**: Accuracy, relevance, detail level, hallucination degree

**Self-improvement Path**: LLaVA-Critic implements a "Self-Reward" closed loop:
1. Generative model produces multiple candidate responses
2. LLaVA-Critic evaluates and ranks
3. Uses preference data for DPO training
4. Model capabilities continuously improve

## 2. Reasoning Large Models

A series of breakthroughs in 2024 revealed a new dimension: sometimes, **letting the model answer more slowly** can lead to better results.

### 2.1 From Fast Thinking to Slow Thinking

Traditional LLMs adopt autoregressive generation, directly predicting the next token given input. This "System 1"-style fast response excels on many tasks but has limitations on tasks requiring complex reasoning:

- **Limited reasoning depth**: Each token's generation only depends on previous context, lacking the ability to "go back and check"
- **Error accumulation**: Early errors in reasoning chains propagate to subsequent steps
- **Lack of planning**: Cannot plan solution paths in advance, can only "walk and see"

#### Test-Time Compute Scaling

The core idea of reasoning large models is **Test-Time Compute Scaling**: investing more computational resources during inference to achieve better output quality.

**Key Findings from Snell et al. (2024)**:
- Test-time compute scaling can be more effective than scaling model parameters
- Using "compute-optimal" strategies, test-time compute efficiency can improve by more than 4x
- In FLOPs-matched evaluation, small model + test-time compute can surpass models 14x larger

Main approaches for test-time compute:

| Approach | Description | Representative Methods |
|----------|-------------|----------------------|
| **Search** | Generate multiple candidate answers, use verifier to select best | Best-of-N, MCTS |
| **Thinking** | Let model "think" longer, generate detailed reasoning process | CoT, o1, R1 |
| **Iteration** | Multiple rounds of self-correction and optimization | Self-Refine, Reflexion |

### 2.2 Chain-of-Thought and Self-Consistency

#### Chain-of-Thought (CoT)

Chain-of-Thought prompting is a foundational technique for reasoning large models, improving complex task performance by guiding the model to generate intermediate reasoning steps.

**Basic Form**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls, each containing 3 balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans have 2*3=6 balls. 5+6=11. The answer is 11.
```

**Zero-shot CoT**: Simply adding "Let's think step by step" can activate the model's reasoning capabilities without providing examples.

#### Self-Consistency

Self-Consistency is an important improvement to Chain-of-Thought. The core idea is:
- Generate multiple reasoning paths for the same problem (by sampling different CoTs)
- Select the most consistent answer through majority voting
- Leverage the intuition that correct answers should be reachable through multiple approaches

**Performance Improvements**:

| Dataset | Improvement |
|---------|-------------|
| GSM8K | +17.9% |
| SVAMP | +11.0% |
| AQuA | +12.2% |
| StrategyQA | +6.4% |

**Self-Consistency Improvements**:
- **CISC** (Confidence-Informed SC): Confidence-weighted voting, reducing sampling requirements by over 40%
- **RASC** (Reasoning-Aware SC): Dynamically adjusts sampling count—fewer samples for easy problems, more for difficult ones
- **LSC** (Latent SC): Selection based on semantic consistency, suitable for long-form open-ended answers

### 2.3 Reward Models and Verifiers

Verifiers are used to evaluate the quality of model-generated reasoning processes and answers, serving as core components of search strategies.

#### Outcome Reward Model (ORM)

Outcome Reward Models only provide reward signals for the final answer:

$$r_\text{ORM}(x, y) = \begin{cases} 1 & \text{if } y \text{ is correct} \\ 0 & \text{otherwise} \end{cases}$$

**Advantages**: Low annotation cost, only need to judge final answer correctness

**Disadvantages**:
- Difficult credit assignment: Cannot identify which step went wrong
- Delayed feedback: Reward only available after completing entire reasoning

#### Process Reward Model (PRM)

Process Reward Models provide reward signals for each reasoning step:

$$r_\text{PRM}(x, y_{1:t}) = \text{score}(y_t | x, y_{1:t-1})$$

where $y_t$ is the $t$-th reasoning step, and score is typically $\{-1, 0, +1\}$ representing $\{$wrong, neutral, correct$\}$.

**OpenAI Experimental Results**: Using pre-RLHF GPT-4 as base model, PRM achieves 78.2% accuracy on MATH test set, significantly outperforming ORM.

**PRM vs ORM Comparison**:

| Feature | ORM | PRM |
|---------|-----|-----|
| Feedback Granularity | Overall result | Each step |
| Annotation Cost | Low | High |
| Credit Assignment | Difficult | Precise |
| Reward Hacking Risk | Low | Higher |
| Search Efficiency | Lower | Higher |

**Implicit PRM**: Recent research found that training an ORM and then using it as a PRM can obtain "free" process rewards without expensive step-level annotation.

#### Process Advantage Verifier (PAV)

PAV combines process supervision with advantage estimation:
- Search accuracy improved by over 8% compared to ORM
- Computational efficiency improved 1.5-5x
- Online RL sample efficiency improved 5-6x

### 2.4 Search and Planning

#### Best-of-N Sampling

The simplest search strategy is to generate N candidate answers and use a verifier to select the best:

$$y^* = \arg\max_{y \in \{y_1, ..., y_N\}} r(x, y)$$

OpenAI o1's performance on AIME 2024:
- Single sampling (pass@1): 74%
- 64 samples + consensus (consensus@64): 83%

#### Monte Carlo Tree Search (MCTS)

MCTS models the reasoning process as a tree search problem, where each node is a reasoning state and edges are reasoning steps.

**Basic Process**:
1. **Selection**: Use UCB formula to select promising nodes
2. **Expansion**: Generate new reasoning steps
3. **Simulation**: Complete reasoning and obtain results
4. **Backpropagation**: Update values of all nodes on the path

**UCB Formula**:

$$\text{UCB}(s, a) = Q(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

where $Q(s, a)$ is the action value estimate, $N(s)$ is the node visit count, and $c$ is the exploration coefficient.

**MCTSr (MCT Self-Refine)**: Combines LLM self-improvement with MCTS, achieving excellent results on Olympiad-level math problems.

**SC-MCTS\***: Uses contrastive decoding to design interpretable reward models, combined with speculative decoding for acceleration, improving average per-node speed by 51.9%. Surpasses o1-mini by 17.4% on Blocksworld dataset.

### 2.5 OpenAI o1

OpenAI o1 (released September 2024) is the first large-scale commercial reasoning large model. Its core innovation is internalizing chain-of-thought as model capability.

#### Core Design

**Key Features**:
- **Reasoning tokens**: Model generates internal reasoning process before answering
- **Hidden thinking**: Reasoning tokens are invisible to users (but are billed)
- **Reinforcement learning training**: Learns "how to think" through large-scale RL

**OpenAI Official Description**:
> "Similar to how a human may think for a long time before responding to a difficult question, o1 uses a chain of thought when attempting to solve a problem. Through reinforcement learning, o1 learns to hone its chain of thought and refine the strategies it uses."

#### Performance

| Benchmark | GPT-4o | o1-preview | o1 |
|-----------|--------|------------|-----|
| AIME 2024 | 12% | 44% | 74% |
| Codeforces Rating | 808 | 1673 | 1891 |
| MATH-500 | 60.3% | 85.5% | 94.8% |
| GPQA Diamond | 50.6% | 73.3% | 78.0% |

#### Scaling Laws

o1 demonstrates two dimensions of scaling:
1. **Training-time compute**: More RL training brings stronger reasoning capabilities
2. **Test-time compute**: Longer thinking time brings better answer quality

This opens a new scaling path: performance can be improved not only by increasing parameters and training data, but also by increasing computation during inference.

### 2.6 DeepSeek-R1

DeepSeek-R1 (January 2025) is the first open-source model to prove that **pure reinforcement learning can activate reasoning capabilities**.

#### Breakthrough of Pure RL Training

**Key Findings from DeepSeek-R1-Zero**:
- No need for SFT, strong reasoning capabilities can be obtained through RL alone
- Advanced reasoning patterns such as self-reflection, verification, and dynamic strategy adjustment emerge
- AIME 2024: improved from 15.6% to 71.0% (pass@1), with majority voting reaching 86.7%

#### GRPO Algorithm

DeepSeek uses Group Relative Policy Optimization (GRPO) for reinforcement learning training:

**Core Ideas**:
- Eliminates the need for a Critic model of the same scale as the policy model in traditional RLHF
- Uses within-group relative scores as baseline estimates
- Significantly reduces training costs

**GRPO Optimization Objective**:

$$\mathcal{L}_\text{GRPO} = -\mathbb{E}_{x, \{y_i\}}\left[\sum_i \frac{r(x, y_i) - \bar{r}}{\sigma_r} \log \pi_\theta(y_i|x)\right]$$

where $\bar{r}$ is the within-group average reward, and $\sigma_r$ is the within-group reward standard deviation.

#### Complete Training Pipeline

DeepSeek-R1's training includes four stages:

1. **Cold Start Data**: Small amount of high-quality reasoning data, solving R1-Zero's readability issues
2. **Reasoning RL**: Large-scale RL training, discovering better reasoning patterns
3. **Rejection Sampling SFT**: Collecting high-quality outputs from RL model for SFT
4. **Preference RL**: Alignment with human preferences

#### Emergent Capabilities

R1-Zero exhibited multiple advanced reasoning behaviors during training:

| Emergent Behavior | Description | Example Expression |
|-------------------|-------------|-------------------|
| **Self-reflection** | Re-examining reasoning process | "Wait, let me reconsider..." |
| **Verification** | Checking correctness of intermediate steps | "Let me verify this step..." |
| **Backtracking** | Going back and retrying after discovering errors | "That's wrong, going back..." |
| **Strategy switching** | Trying another approach when one doesn't work | "Let me try a different approach..." |

### 2.7 Knowledge Distillation

DeepSeek pioneered proving that reasoning capabilities can be transferred to smaller models through distillation.

#### Distillation Method

- Using DeepSeek-R1 to generate 800K reasoning samples
- Performing SFT on smaller models (no additional RL needed)
- Smaller models acquire similar reasoning capabilities

#### Distillation Model Performance

| Model | Base | AIME 2024 | MATH-500 |
|-------|------|-----------|----------|
| R1-Distill-Qwen-1.5B | Qwen2.5-1.5B | 28.9% | 83.9% |
| R1-Distill-Qwen-7B | Qwen2.5-7B | 55.5% | 92.8% |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7% | 93.9% |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 72.6% | 94.3% |
| R1-Distill-Llama-8B | Llama3.1-8B | 50.4% | 89.1% |
| R1-Distill-Llama-70B | Llama3.3-70B | 70.0% | 94.5% |

**Key Findings**:
- R1-Distill-Qwen-32B surpasses o1-mini
- Distillation outperforms direct RL training on same-size models
- Distillation is an efficient path to acquiring reasoning capabilities

### 2.8 Open-Source Reasoning Models

#### QwQ (Qwen with Questions)

QwQ is an open-source reasoning model released by Alibaba's Qwen team (November 2024).

**Design Philosophy**:
> "QwQ approaches every problem with genuine wonder and doubt. It knows that it knows nothing, and that's precisely what drives its curiosity."

**Technical Features**:
- 32B parameters, 32K context length
- Uses regularized reinforcement learning to embed reasoning capabilities
- Generates long thinking chains during inference

**Performance**:
- GPQA: 65.2% (graduate-level scientific reasoning)
- AIME 2024: 50.0%
- MATH-500: 90.6%
- LiveCodeBench: 50.0%

**Known Limitations**:
- May mix languages or unexpectedly switch languages
- May get stuck in circular reasoning, producing overly long outputs

#### Marco-o1

Another reasoning model from Alibaba, Marco-o1 uses MCTS algorithm to generate synthetic training data, combined with CoT samples for training.

#### Mainstream Reasoning Model Comparison

| Model | Parameters | Open-source | Training Method | AIME | MATH | Release |
|-------|-----------|-------------|-----------------|------|------|---------|
| GPT-4o | - | No | SFT | 12% | 60.3% | 2024.05 |
| o1-preview | - | No | RL | 44% | 85.5% | 2024.09 |
| o1 | - | No | RL | 74% | 94.8% | 2024.12 |
| QwQ-32B | 32B | Yes | RL | 50% | 90.6% | 2024.11 |
| DeepSeek-R1 | 671B | Yes | RL | 79.8% | 97.3% | 2025.01 |
| R1-Distill-32B | 32B | Yes | Distillation | 72.6% | 94.3% | 2025.01 |

#### Training Paradigm Comparison

| Paradigm | Representative Model | Characteristics |
|----------|---------------------|-----------------|
| **Large-scale RL + Hidden Reasoning** | o1 | Closed-source, reasoning process invisible |
| **GRPO + Multi-stage Training** | DeepSeek-R1 | Fully open-source, four-stage training |
| **Regularized RL** | QwQ | Open weights, long thinking chains |
| **SFT Distillation** | R1-Distill series | Efficient path to reasoning capabilities |

### 2.9 Applications and Limitations

#### Applicable Scenarios

Reasoning large models are particularly suitable for:
- **Mathematical problems**: Competition mathematics, theorem proving
- **Code generation**: Complex algorithms, debugging
- **Scientific reasoning**: Physics, chemistry problems
- **Logical reasoning**: Planning, constraint satisfaction

#### Current Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **High latency** | Long thinking time | Not suitable for real-time interaction |
| **High cost** | Reasoning tokens consume significant computation | Increased API call costs |
| **Over-thinking** | Even simple problems may produce lengthy reasoning | Resource waste |
| **Circular reasoning** | May get stuck in meaningless thought loops | Cannot converge |
| **Language mixing** | May mix multiple languages during thinking | Reduced readability |

#### Open Questions

- **Optimal thinking length**: How to determine when to stop thinking?
- **Thinking interpretability**: Can hidden reasoning processes be trusted?
- **General reasoning**: Currently mainly in math/code domains, how to extend to more domains?
- **Efficiency optimization**: How to reduce computational cost while maintaining reasoning quality?

## 3. Future Directions

### 3.1 Multimodal Reasoning

Extending reasoning capabilities to multimodal is an important research direction:

| Direction | Capability | Application Scenarios |
|-----------|------------|----------------------|
| **Visual reasoning** | Logical relationship inference in images | Math geometry problems, chart understanding |
| **Video understanding** | Temporal reasoning, event causal analysis | Video Q&A, action prediction |
| **Embodied intelligence** | Planning and interaction in physical world | Robot manipulation, autonomous driving |

### 3.2 Unifying All Modalities

Most current models primarily handle images and text. The future will extend to more modalities:

- **Audio/Speech**: Native speech understanding and generation (like GPT-4o)
- **Video**: Long video understanding and generation
- **3D**: 3D scene understanding, spatial reasoning
- **Tactile/Force feedback**: Perception capabilities for embodied AI

### 3.3 Reasoning and Agents

Reasoning large models provide stronger planning capabilities for AI Agents:

| Capability | Description | Value |
|------------|-------------|-------|
| **Task decomposition** | Breaking complex tasks into subtasks | Reduces execution difficulty |
| **Planning** | Pre-planning execution paths | Improves success rate |
| **Tool usage** | Deciding when to call which tools | Expands capability boundaries |
| **Long-term goals** | Tracking and progressing toward long-term goals | Complex task completion |

### 3.4 Efficiency Improvements

Research directions for improving reasoning efficiency:

- **Compute-optimal strategies**: Dynamically adjust test-time compute based on task difficulty
  - Simple problems: Fast response
  - Difficult problems: Deep thinking
  - Automatically predict difficulty and select strategy
- **Early stopping strategies**: Stop early when answer convergence is detected
- **Speculative decoding**: Accelerate reasoning token generation
- **Sparse activation**: Only activate parameters related to reasoning
- **Lightweight models**: Distill smaller reasoning models

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
