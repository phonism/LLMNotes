---
layout: post
title: "Transformer Notes (II): Core Components"
date: 2025-12-20 10:10:00
author: Phonism
tags: [Transformer, Tokenizer, RoPE, SwiGLU]
lang: en
translation: /transformer-part2-components/
---

The powerful capabilities of Transformers are built upon three carefully designed core components: the Tokenizer converts text into discrete symbols that the model can process; Position Encoding injects sequence order information into the self-attention mechanism; and Gating mechanisms enable the network to learn to selectively pass information.

This article explores in depth the design principles and engineering implementations of these three components.

## 1. Tokenizer

The tokenizer is the entry point for large language models, responsible for converting raw text into token sequences:

$$\text{"Hello world"} \xrightarrow{\text{Tokenizer}} [15496, 995] \xrightarrow{\text{Embedding}} \mathbb{R}^{2 \times d}$$

### 1.1 Trade-offs in Tokenization Granularity

| Granularity | Vocab Size | Sequence Length | Issues |
|------|----------|----------|------|
| Character-level | ~256 | Very long | Sequences too long, difficult to model long-range dependencies |
| Word-level | ~100K+ | Short | OOV problem, vocabulary too large |
| **Subword-level** | ~32K-128K | **Moderate** | **Balanced, mainstream choice** |

### 1.2 Byte Pair Encoding (BPE)

BPE is the most widely used subword tokenization algorithm, originally from the data compression field.

**Training Algorithm**:
1. Initialize vocabulary with all characters (or bytes)
2. Count frequencies of adjacent token pairs
3. Merge the most frequent token pair into a new token, add to vocabulary
4. Repeat steps 2-3 until reaching target vocabulary size

> **Example**: Suppose corpus is "low lower lowest"
> 1. Initial: `l, o, w, e, r, s, t, _` (_ represents word boundary)
> 2. Most frequent pair `(l, o)` → merge into `lo`
> 3. Most frequent pair `(lo, w)` → merge into `low`
> 4. Most frequent pair `(low, e)` → merge into `lowe`
> 5. ...

**Tokenization Algorithm**:

```python
def bpe_tokenize(text, merges):
    tokens = list(text)  # Initially characters
    for (a, b) in merges:  # In training order
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens = tokens[:i] + [a+b] + tokens[i+2:]
            else:
                i += 1
    return tokens
```

### 1.3 Byte-level BPE

An improvement introduced by GPT-2, operating directly at the byte level:
- Base vocabulary of 256 bytes, no pre-tokenization needed
- Can represent any UTF-8 text, no OOV problem
- Avoids special handling for different languages

### 1.4 WordPiece vs BPE

WordPiece was proposed by Google and used in BERT. The main difference from BPE lies in the merging strategy:

- **BPE**: Selects the most frequent token pair
- **WordPiece**: Selects the token pair that maximizes language model likelihood

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

WordPiece uses `##` to mark non-initial subwords:
```
"tokenization" -> ["token", "##ization"]
```

### 1.5 Unigram Language Model

Unigram adopts the opposite strategy—starting from a large vocabulary and gradually pruning:

1. Initialize a large candidate vocabulary
2. Use EM algorithm to estimate probability of each token
3. Calculate impact of removing each token on likelihood
4. Remove the token with minimal impact
5. Repeat until reaching target vocabulary size

**Advantage**: Same text can have multiple tokenizations, supports sampling (Subword Regularization).

### 1.6 SentencePiece and Tiktoken

**SentencePiece** (Google):
- Language-agnostic: treats spaces as regular characters (represented by ▁)
- Supports BPE and Unigram
- Reversible: tokenization results can be losslessly restored to original text

```python
import sentencepiece as spm

# Training
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',
)

# Usage
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode('Hello world', out_type=str)
# ['▁Hello', '▁world']
```

**Tiktoken** (OpenAI):
- Rust implementation: 3-6× faster than Python implementation
- Regex pre-tokenization: pre-split using regular expressions
- GPT-4 uses cl100k_base encoding

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello world")  # [9906, 1917]
print(enc.n_vocab)  # 100277
```

### 1.7 Vocabulary Configurations of Mainstream Models

| Model | Vocab Size | Tokenizer |
|------|----------|--------|
| GPT-2 | 50,257 | Byte-level BPE |
| GPT-4 | 100,277 | Byte-level BPE (cl100k) |
| BERT | 30,522 | WordPiece |
| LLaMA | 32,000 | SentencePiece BPE |
| LLaMA 3 | 128,256 | Tiktoken BPE |
| Qwen | 151,936 | Byte-level BPE |
| DeepSeek | 102,400 | Byte-level BPE |

**Impact of Vocabulary Size**:
- Larger vocabulary: Shorter sequence length, larger embedding parameters, better multilingual support
- Smaller vocabulary: Longer sequences, smaller model parameters, rare words may be over-segmented

### 1.8 Multilingual Tokenization: Fairness Issues

Token efficiency varies significantly across languages (GPT-4, same semantic content):

| Language | Token Count | Relative to English |
|------|----------|----------|
| English | 100 | 1.0× |
| Spanish | 120 | 1.2× |
| Chinese | 150 | 1.5× |
| Japanese | 180 | 1.8× |
| Burmese | 400 | 4.0× |

**Root Causes of Efficiency Differences**:

1. **Alphabetic vs Logographic Writing**: English combines 26 letters into words, making it easy for BPE to learn common subwords. Chinese has about 3500 commonly used characters covering 99.9% of text, each being an independent morpheme.

2. **Training Data Skew**: When English comprises 90% of training corpus, English subwords are fully merged, while Chinese vocabulary remains split due to low frequency.

3. **UTF-8 Encoding Overhead**: English characters occupy 1 byte, Chinese characters occupy 3 bytes. In Byte-level BPE, a Chinese character requires at least 3 basic tokens.

**Practical Impact**:
- Cost: Same semantic content consumes 1.5-4× tokens, proportionally increasing API fees
- Context: Effective context window shrinks (128K tokens for Chinese users equivalent to 85K for English users)
- Latency: Generating same content requires more decoding steps

**Improvement Strategies**:
- LLaMA 3 expanded vocabulary from 32K to 128K, improving Chinese token efficiency by ~3×
- Oversample low-resource languages during tokenizer training

## 2. Position Encoding

Self-attention mechanism itself is **permutation invariant**—pure attention modules cannot capture input order. The introduction of position encoding is essential.

### 2.1 Position Encoding Classification

| Type | Method | Application Position | Extrapolation | Representative Models |
|------|------|----------|--------|----------|
| Absolute | Sinusoidal | Embedding | Poor | Transformer |
| Absolute | Learned | Embedding | Poor | BERT, GPT |
| Relative | T5 Bias | Attention score | Good | T5 |
| Relative | ALiBi | Attention score | Good | BLOOM, MPT |
| **Relative** | **RoPE** | **Q/K vectors** | **Good** | **LLaMA, Qwen** |

### 2.2 Absolute Position Encoding

**Learned**:
The most naive approach, treating position encoding as trainable parameters. For example, with max length 512 and encoding dimension 768, initialize a $512 \times 768$ matrix.

Drawback: Lacks extrapolation—if pretrained max length is 512, cannot handle longer sequences.

**Sinusoidal**:

$$p_{k,2i} = \sin\left(k / 10000^{2i/d}\right), \quad p_{k,2i+1} = \cos\left(k / 10000^{2i/d}\right)$$

Design intuition: Different dimensions correspond to periodic functions of different frequencies—low dimensions change quickly (capture local positions), high dimensions change slowly (capture global positions).

### 2.3 Rotary Position Embedding (RoPE)

RoPE is currently the most mainstream position encoding method, widely adopted by LLaMA, Mistral, Qwen, and other models.

**Core Idea**: Fuse absolute and relative positions—by applying absolute position rotation operations on Q and K, the inner product naturally depends only on relative positions.

**Theoretical Origin**: RoPE's design inspiration comes from properties of complex numbers:

$$\langle q e^{im\theta}, k e^{in\theta} \rangle = \text{Re}[q \bar{k} e^{i(m-n)\theta}]$$

Depends only on relative position $m-n$.

**Problem Setting**: Add absolute position information to $\mathbf{q}, \mathbf{k}$:

$$\tilde{\mathbf{q}}_m = f(\mathbf{q}, m), \quad \tilde{\mathbf{k}}_n = f(\mathbf{k}, n)$$

Want the inner product result to contain **relative position information**:

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m-n)$$

**Solution for 2D Case**:

$$\boxed{f(\mathbf{q}, m) = \|\mathbf{q}\| e^{i(\Theta(\mathbf{q}) + m\theta)} = \mathbf{q} e^{im\theta}}$$

This is precisely multiplying the vector by rotation factor $e^{im\theta}$, corresponding to **rotating by angle $m\theta$**.

**Matrix Form**:

$$\text{RoPE}(\mathbf{x}, m) = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & & \\
\sin m\theta_1 & \cos m\theta_1 & & \\
& & \cos m\theta_2 & -\sin m\theta_2 \\
& & \sin m\theta_2 & \cos m\theta_2 \\
& & & & \ddots
\end{pmatrix} \mathbf{x}$$

**Frequency Parameters**:

$$\theta_i = \text{base}^{-2(i-1)/d}, \quad i = 1, 2, \ldots, d/2$$

where $\text{base} = 10000$ is the original setting.

**Key Property**:

$$(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_m^\top \mathbf{R}_n \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$$

The inner product automatically contains relative position information. $\mathbf{R}_m$ is an orthogonal matrix, **preserving vector magnitude**, maintaining model stability.

**Efficient Implementation**:

$$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})$$

Complexity is $O(d)$, no need to construct the full rotation matrix.

### 2.4 RoPE Base Selection

**Semantic Discriminability** is defined as the model's ability to distinguish between similar tokens and random tokens:

$$B_{m,\theta} = \sum_{i=1}^{d/2} \cos(m\theta_i)$$

As relative distance $m$ increases, $B_{m,\theta}$ gradually decreases (attention decay). When base is too small, it becomes negative—the model actually gives higher attention to random tokens.

**Base Lower Bound** vs Context Length Relationship:

| Context Length $L$ | 4K | 8K | 32K | 128K | 1M |
|--------------------|----|----|-----|------|-----|
| Base Lower Bound $b^*$ | $4.5 \times 10^4$ | $8.4 \times 10^4$ | $6.4 \times 10^5$ | $3.4 \times 10^6$ | $6.5 \times 10^7$ |

Asymptotic analysis shows $b^* \approx O(L)$, meaning base should grow **linearly** with context length.

**Base Selection in Actual Models**:
- LLaMA 3: Training length 8192, but base chosen as 500000, far exceeding lower bound
- Mixtral: base = 1000000, supports 128K context

### 2.5 Length Extrapolation Methods

**Position Interpolation (PI)**: Scale position indices

$$m' = \frac{m}{s}, \quad s = \frac{L'}{L}$$

"Compress" long sequences into original position range. Problem: Uniform scaling destroys high-frequency information.

**NTK-aware Interpolation**: Adjust base instead of uniform scaling

$$\text{base}' = \text{base} \cdot s^{d/(d-2)}$$

Distribute interpolation pressure across different dimensions: less interpolation for high-frequency dimensions, more for low-frequency dimensions.

**YaRN**: Combines NTK-by-parts and **attention temperature scaling**

$$\text{Attention}'_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d} \cdot t}$$

YaRN requires only ~400 finetuning steps to extend LLaMA 2 from 4K to 64K.

| Method | Extrapolation | Finetuning | Notes |
|--------|---------------|------------|-------|
| PI | 2× | Required | Uniform scaling |
| NTK-aware | 32× | Optional | Good without finetuning |
| YaRN | 16× | Minimal | Combined with temperature scaling |
| Dynamic | 64× | None | Dynamic adjustment at inference |

### 2.6 RoPE vs ALiBi

ALiBi directly adds distance penalty on attention score:

$$\text{Attention}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} - m \cdot |i - j|$$

| Feature | RoPE | ALiBi |
|---------|------|-------|
| Encoding Position | Q/K vectors | Attention score |
| Parameters | 0 | 0 |
| Extrapolation Ability | Medium (needs extension methods) | Good |
| KV Cache Friendly | Yes | Yes |
| Adopted Models | LLaMA, Mistral, Qwen | BLOOM, MPT |

### 2.7 Latest Developments

**iRoPE (LLaMA 4)**: Hybrid use of RoPE layers and no-position-encoding layers, combined with attention temperature scaling at inference, achieving extreme extrapolation from 256K training length to 10M context window.

**2D/3D RoPE**: Extends RoPE to two-dimensional (image) and three-dimensional (video) position encoding.

## 3. Gating Mechanisms

### 3.1 Why Do We Need Gating?

Standard linear transformation $y = Wx + b$ treats all inputs equally—regardless of input content, weights $W$ remain constant.

Gating mechanisms introduce **data-dependent dynamics**:

$$y = g(x) \odot f(x)$$

where $g(x) \in [0, 1]^d$ is the gating signal. Key insight: $g$ itself depends on input $x$, transforming from static $f$ to dynamic $g \odot f$.

**Information Bottleneck Perspective**: Gating implements **adaptive compression**—when $g \to 0$, information can be actively discarded. This "active forgetting" capability is crucial for filtering noise and focusing on key information.

**Sparse Activation Perspective**: Gating naturally induces sparsity. Experiments show activation sparsity in gated networks can reach 60-80%.

### 3.2 MLP Layer Gating: SwiGLU

As described in the first article, modern Transformers commonly adopt SwiGLU to replace standard FFN:

**Standard FFN**:
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

**SwiGLU FFN**:
$$\text{SwiGLU}(x) = W_2 \cdot \underbrace{(\text{SiLU}(W_1 x) \odot W_3 x)}_{\text{gating}}$$

Here $W_3 x$ acts as a gating signal, controlling information flow through $\text{SiLU}(W_1 x)$.

### 3.3 Attention Layer Gating: Gated Attention

Add sigmoid gating after Scaled Dot-Product Attention (SDPA) output:

**Standard SDPA**:
$$Y = \text{softmax}\left(\frac{QK^\top}{\sqrt{H}}\right)V$$

**Gated Attention**:
$$Y' = Y \odot \sigma(XW_g)$$

**Why is Gating at SDPA Output Optimal?**

| Configuration | Effectiveness |
|---------------|---------------|
| Gate on Values | Effective but not optimal |
| Gate on Keys | Effective but not optimal |
| **Gate on SDPA output** | **Optimal position** |

Reason: Gating on final output can **holistically suppress** contribution of entire attention head, breaking softmax's forced allocation constraint. Essentially implements **head-level dynamic pruning**.

### 3.4 Attention Sink Problem

**Observation**: In long sequence tasks, large amounts of attention weights concentrate on a few tokens at the beginning of the sequence (usually the first token), even when these tokens are not semantically important. More importantly, **this phenomenon is more pronounced in deeper layers**.

Key finding from StreamingLLM: When using sliding window attention, once initial tokens are moved out of the window, model output **completely collapses**. But simply retaining the first 4 tokens largely recovers performance.

<div class="mermaid">
graph LR
    subgraph shallow["Shallow Attention (diagonal)"]
        A1["■□□□□"]
        A2["□■□□□"]
        A3["□□■□□"]
        A4["□□□■□"]
        A5["□□□□■"]
    end
    subgraph deep["Deep Attention (Sink)"]
        B1["■□□□□"]
        B2["■□■□□"]
        B3["■□□■□"]
        B4["■□□□■"]
        B5["■□□□□"]
    end
    shallow --> |"vs"| deep
</div>

> **Figure Explanation**: Left shows shallow attention mainly concentrated on diagonal (each token attends to itself); right shows deep attention with Sink phenomenon in first column (all tokens attend to first token).

**Surface Cause: Softmax Probability Constraint**

$$\sum_{j=1}^{T} \text{softmax}(q_i^\top k_j / \sqrt{H}) = 1$$

Each query **must** allocate all attention, even when ideally it should "not attend to any token". The network's coping strategy is to learn a "garbage bin" position to absorb excess attention.

**Deep Cause: Context-Aware Identity Layer Hypothesis**

> *Attention Sink stems from Transformer's inherent need for "context-aware identity layers"—the model needs to be able to decide, based on context, that a certain Attention Block outputs no change.*

Evidence:
1. Sink Token's Value is close to zero—model actively learns to zero it out
2. Early Decoding correlates with layer depth—deeper layers need to maintain identity transformation
3. Sink Token's Key has independent subspace—model allocates dedicated space for it

### 3.5 Solutions to Attention Sink

**Solution 1: Retain Initial Tokens (StreamingLLM)**

$$\text{Attention Range} = \{1, 2, \ldots, k_{\text{sink}}\} \cup \{t - w + 1, \ldots, t\}$$

Retaining 4 initial tokens enables model to stably handle **4 million+ token** streaming input.

**Solution 2: Learnable Softmax Bias**

$$\text{Attention}_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_k \exp(q_i^\top k_k / \sqrt{d}) + \exp(b_h)}$$

where $b_h$ is a learnable scalar for each attention head. When $b_h$ is large, the denominator increases, diluting all attention weights.

Representative models: GPT-OSS, MiMo-V2-Flash

**Solution 3: Output Gating**

$$Y' = Y \odot \text{gate}(X)$$

Allows attention to output zero vectors without relying on sink tokens.

Representative models: Kimi Linear, Qwen

| Solution | Extra Parameters | Eliminates Sink | Representative Models |
|------|----------|-----------|----------|
| Retain Initial Tokens | 0 | No (bypass) | StreamingLLM |
| Softmax Bias | $n_h$ | Yes | GPT-OSS, MiMo-V2-Flash |
| Output Gating | $D^2$ | Yes | Kimi Linear, Qwen |

**Theoretical Unification**: These solutions all address the problem of **how Attention outputs zero**. Notably, output gating not only eliminates the sink but also **releases dimensions occupied by sink**—this capacity can be used for more meaningful representation learning.

### 3.6 Industrial Applications of Gating

Gated Attention has been integrated into the Qwen3-Next architecture, validating its effectiveness in large-scale industrial applications.

**Additional Benefits**:
- Training stability: Smoother loss curves, can use larger learning rates
- Long context extrapolation: Combined with YaRN, extrapolating from 32k training length to 128k shows significantly less performance degradation than baseline

> **MLP Gating vs Attention Gating**: The two types of gating serve complementary roles:
> - MLP gating (SwiGLU): Selectively activates neurons during feature transformation
> - Attention gating: Selectively passes attention output during information aggregation
>
> Modern models (like Qwen3-Next) employ both.

## Chapter Summary

This chapter explored in depth three core components of Transformers:

1. **Tokenizer**:
   - BPE is the most mainstream subword tokenization algorithm
   - Multilingual efficiency differences are an underestimated fairness issue
   - LLaMA 3 expanded vocabulary to 128K to improve multilingual support

2. **Position Encoding**:
   - RoPE fuses absolute and relative position information through rotation operations
   - Base should grow linearly with context length
   - Methods like YaRN can achieve effective length extrapolation

3. **Gating Mechanisms**:
   - SwiGLU implements selective activation in MLP layers
   - Gated Attention solves the Attention Sink problem
   - Gating releases model capacity occupied by sinks

The next article will explore in depth attention mechanism optimizations: FlashAttention, MLA, sparse attention, and linear attention.
