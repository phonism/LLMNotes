---
layout: post
title: "Transformer Notes (V): Training Techniques"
date: 2025-12-20 10:40:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [Transformer, Training]
lang: en
translation: /transformer-part5-training/
series: transformer
series_order: 5
---

This is the fifth article in the Transformer series, providing a comprehensive analysis of **training techniques** for large language models, including data engineering, distributed training frameworks, and novel optimizers. These techniques collectively support the efficient training of models with hundreds of billions of parameters.

## 1. Data Engineering

Data is the foundation of large language models. This section systematically introduces the construction methods for pretraining data and post-training data.

### 1.1 Pretraining Data Sources

Modern LLM pretraining data typically comes from the following sources:

| Data Source | Description | Scale | Quality |
|-------------|-------------|-------|---------|
| **Web Data** | | | |
| Common Crawl | Largest web crawl data | PB scale | Low |
| RefinedWeb | Filtered high-quality web pages | 5T tokens | Medium |
| C4 | Colossal Clean Crawled Corpus | 800B tokens | Medium |
| **High-Quality Text** | | | |
| Wikipedia | Encyclopedia | Tens of B tokens | High |
| Books | Books (Books3, Pile-Books) | Tens of B tokens | High |
| arXiv | Academic papers | Tens of B tokens | High |
| **Code** | | | |
| GitHub | Open-source code repositories | Hundreds of B tokens | Medium-High |
| The Stack | Deduplicated open-source code | 3T tokens | High |

### 1.2 Data Processing Pipeline

Going from raw data to training data requires multiple processing stages:

**1. Text Extraction**
- HTML parsing: Use tools like trafilatura, jusText to extract main content
- PDF parsing: Use PyMuPDF, pdfplumber to process academic documents
- Code processing: Preserve comments and docstrings

**2. Quality Filtering**
- **Heuristic rules**: Document length, special character ratio, duplicate line ratio, language detection
- **Model scoring**: Perplexity filtering, quality classifier, educational value scoring (FineWeb-Edu)

**3. Deduplication**

Deduplication is a crucial step in pretraining data processing. Duplicate data leads to decreased training efficiency and causes models to memorize rather than generalize.

Main deduplication methods:
- **Exact deduplication**: Hash-based exact matching
- **Fuzzy deduplication**: MinHash + LSH (Locality-Sensitive Hashing)

```python
# MinHash deduplication example
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

# Create LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc_id, text in documents:
    minhash = get_minhash(text)
    lsh.insert(doc_id, minhash)
```

**4. Sensitive Content Filtering**
- PII (Personally Identifiable Information) removal
- Harmful content filtering
- Copyright content handling

### 1.3 Data Mix

Data mix has a significant impact on model capabilities:

| Model | Web | Code | Books | Academic |
|-------|-----|------|-------|----------|
| GPT-3 | 60% | 3% | 16% | - |
| LLaMA | 67% | 4.5% | 4.5% | 2.5% |
| DeepSeek | 56% | 18% | - | 5% |

**Mix Principles**:
- **Code data**: Improves reasoning ability, typically 5-20%
- **Math data**: Improves mathematical ability, but excessive ratio may harm general capabilities
- **Multilingual data**: Chinese models typically use 30-50% Chinese
- **High-quality data**: Although limited in quantity, should be upsampled

### 1.4 Data Scale and Over-training

The Chinchilla Scaling Law indicates that the compute-optimal data amount is proportional to model parameters:

$$D_{opt} \approx 20 \times N$$

Where $D$ is the number of tokens and $N$ is the number of parameters. That is, a 7B model needs approximately 140B tokens.

However, in practice, much more data is often used (over-training):

| Model | Parameters | Training tokens | Multiplier |
|-------|------------|-----------------|------------|
| LLaMA-7B | 7B | 1T | 143x |
| LLaMA 2-7B | 7B | 2T | 286x |
| LLaMA 3-8B | 8B | 15T | 1875x |

**Why Over-training?**
- Inference cost is fixed, training cost can be amortized
- Smaller models are easier to deploy
- Data reuse is beneficial within certain limits

### 1.5 Data Curriculum and Annealing

**Data Curriculum**: Presenting data in a specific order

1. **Stage 1**: General web data, establishing basic language capabilities
2. **Stage 2**: Increase proportion of high-quality data (books, Wikipedia)
3. **Stage 3**: Add code and math data
4. **Stage 4**: Annealing phase, use highest quality data, reduce learning rate

**Annealing Phase**: Using high-quality data at the end of training
- Learning rate anneals from normal value to near 0
- Data switches to highest quality subset
- Typically accounts for 1-5% of total training
- LLaMA 3 reports significant benchmark performance improvement from this phase

### 1.6 Post-training Data

Post-training includes supervised fine-tuning (SFT) and human preference alignment (RLHF/DPO).

**SFT Data Format**:
```json
{
  "instruction": "Translate the following sentence to English",
  "input": "今天天气很好",
  "output": "The weather is nice today."
}
```

**Preference Data Format**:
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing is a type of computation that leverages quantum mechanics principles...",
  "rejected": "Quantum computing is just a very fast computer..."
}
```

**LIMA's Insight**: 1000 carefully curated SFT examples can produce a powerful conversational model. Data diversity is more important than quantity, and consistency in response style is crucial.

## 2. Distributed Training Frameworks

Training large-scale language models requires distributed systems across multiple GPUs and nodes.

### 2.1 Why Distributed Training is Needed

Modern LLMs have reached hundreds of billions of parameters:
- GPT-3: 175B parameters, requires approximately 700GB memory (FP32)
- DeepSeek-V3: 671B parameters

Single GPU memory is limited (A100/H100: 80GB), so the model must be distributed across multiple devices.

**Training Memory Breakdown** (using Adam + FP16 mixed precision):

| Component | Precision | Memory |
|-----------|-----------|---------|
| Model parameters | FP16 | $2\Phi$ |
| Gradients | FP16 | $2\Phi$ |
| Optimizer states (Adam) | | |
| - FP32 parameter copy | FP32 | $4\Phi$ |
| - First moment $m$ | FP32 | $4\Phi$ |
| - Second moment $v$ | FP32 | $4\Phi$ |
| **Total (excluding activations)** | | $16\Phi$ |

For a 7B model: $16 \times 7 \times 10^9 = 112$GB, exceeding single-card capacity.

### 2.2 Data Parallel (DP)

Data parallelism is the simplest distributed strategy:

1. Each GPU holds a **complete model copy**
2. The dataset is split, each GPU processes a different mini-batch
3. Forward propagation proceeds independently
4. After backward propagation, **All-Reduce** synchronizes gradients
5. Each GPU independently updates parameters (with identical results)

**Limitation**: Each GPU must be able to hold the complete model, cannot train super-large models.

### 2.3 ZeRO: Zero Redundancy Optimizer

ZeRO (Zero Redundancy Optimizer) is the core technology of DeepSpeed, eliminating memory redundancy in data parallelism through sharding.

**Three Stages**:

| Stage | Sharded Content | Single GPU Memory | Communication |
|-------|----------------|-------------------|---------------|
| DDP | None | $16\Phi$ | $2\Phi$ |
| ZeRO-1 | Optimizer states | $4\Phi + 12\Phi/N$ | $2\Phi$ |
| ZeRO-2 | + Gradients | $2\Phi + 14\Phi/N$ | $2\Phi$ |
| ZeRO-3 | + Parameters | $16\Phi/N$ | $3\Phi$ |

**ZeRO-1**: Shards Adam's $m, v$ and FP32 parameter copy across $N$ GPUs

$$\text{Optimizer memory}: 12\Phi \to \frac{12\Phi}{N}$$

**ZeRO-2**: Gradients are also sharded by $1/N$

**ZeRO-3**: Model parameters are also sharded, All-Gather on-demand during forward/backward

**ZeRO-Offload**: Offload optimizer states and computation to CPU, single GPU can train 10B+ models

**ZeRO-Infinity**: Further offload to NVMe SSD, 512 GPUs can train trillion-parameter models

### 2.4 Tensor Parallel (TP)

Tensor parallelism splits parameter matrices of a single layer across multiple GPUs.

**MLP Layer Tensor Parallelism**:

For FFN layer $Y = \text{GeLU}(XW_1)W_2$:

- **Column split** $W_1$ (along output dimension): Each GPU independently computes partial output, no communication needed
- **Row split** $W_2$ (along input dimension): Requires All-Reduce summation

**Attention Layer Tensor Parallelism**: Multi-head attention is naturally suited for tensor parallelism, distributing $h$ heads across $N$ GPUs, each GPU processes $h/N$ heads.

**Communication Overhead**: Each Transformer layer forward requires 2 All-Reduce operations (attention + MLP)

Tensor parallelism is suitable for **intra-node** high-bandwidth interconnects (NVLink: 600GB/s).

### 2.5 Pipeline Parallel (PP)

Pipeline parallelism splits the model by layers across different GPUs:
- GPU 0: Layer 0-7
- GPU 1: Layer 8-15
- ...

**Naive Pipeline Problem**: Sequential execution leads to severe **Pipeline Bubbles**

$$\text{Bubble ratio} = \frac{p - 1}{m + p - 1}$$

Where $p$ is the number of pipeline stages and $m$ is the number of micro-batches.

**1F1B Schedule**: Alternating one forward, one backward
- In steady state, each GPU simultaneously has 1 micro-batch in forward, 1 in backward
- Activation memory only needs to store $p$ micro-batches

### 2.6 3D Parallelism

Megatron-LM combines DP, TP, PP into **3D parallelism**:

$$\text{Total GPUs} = N_{DP} \times N_{TP} \times N_{PP}$$

**Parallelism Selection Principles**:
1. **TP prioritized for intra-node**: High NVLink bandwidth, TP communication is frequent
2. **PP for inter-node**: Small communication volume (only passes activations), can span nodes
3. **DP scales throughput**: Communication can overlap with computation

**Configuration Examples**:

| Model | GPUs | TP | PP | DP |
|-------|------|----|----|----|
| GPT-3 175B | 1024 | 8 | 8 | 16 |
| LLaMA-70B | 64 | 8 | 2 | 4 |

### 2.7 PyTorch FSDP

Fully Sharded Data Parallel (FSDP) is PyTorch's native ZeRO-3 implementation.

**Core Features**:
- **Parameter sharding**: Model parameters, gradients, optimizer states all sharded
- **On-demand All-Gather**: Temporarily restore complete parameters during forward/backward
- **Compatible with torch.compile**: Can achieve additional speedup

**Sharding Strategies**:
- `FULL_SHARD`: Full sharding (similar to ZeRO-3)
- `SHARD_GRAD_OP`: Only shard gradients and optimizer (similar to ZeRO-2)
- `HYBRID_SHARD`: Intra-node sharding, inter-node replication

### 2.8 Mixed Precision Training

Mixed precision training uses low precision (FP16/BF16) to accelerate computation while maintaining training stability.

**Numerical Format Comparison**:

| Format | Bits | Exponent | Mantissa | Dynamic Range |
|--------|------|----------|----------|---------------|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ |
| FP16 | 16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ |
| FP8 | 8 | 4-5 | 2-3 | $\pm 448$ ~ $\pm 5.7 \times 10^{4}$ |

**BF16 Advantage**: Same exponent bits as FP32, no need for Loss Scaling, more stable training.

**FP8 Training**: Requires per-tensor scaling, approximately 10% faster than BF16 on H100.

### 2.9 Collective Communication Primitives

| Primitive | Function | Use Case |
|-----------|----------|----------|
| Broadcast | One-to-many broadcast | Parameter initialization |
| All-Reduce | Reduce then broadcast | DDP gradient sync |
| All-Gather | Gather all shards | ZeRO parameter restoration |
| Reduce-Scatter | Reduce then shard | ZeRO gradient sharding |
| All-to-All | Full exchange | MoE expert communication |

**Ring All-Reduce**: Communication volume $2(N-1)/N \cdot D \approx 2D$, independent of GPU count $N$, is the bandwidth-optimal algorithm.

## 3. Muon Optimizer

Since its introduction in 2014, the Adam optimizer has been the standard choice for deep learning. However, Adam is essentially an **element-wise** optimizer that does not leverage the **matrix structure** of neural network parameters. Muon achieves more efficient parameter updates through matrix orthogonalization of momentum.

### 3.1 Limitations of Adam

Adam's update rules:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

All operations are performed **element-wise**. For matrix parameters $W \in \mathbb{R}^{d_{out} \times d_{in}}$, Adam flattens them into vectors, completely ignoring the matrix's row-column structure.

A key observation is: gradient updates from SGD and Adam typically have **extremely high condition numbers**, i.e., they are close to low-rank matrices. Updates mainly occur along a few "principal directions", while "rare directions" are severely suppressed.

### 3.2 Muon's Core Idea: Orthogonalizing Momentum

Muon's core idea is to replace the momentum matrix $M$ with its nearest semi-orthogonal matrix, i.e., performing **polar decomposition**.

Let $M = U\Sigma V^\top$ be the singular value decomposition of $M$, then:

$$\text{msign}(M) = UV^\top$$

This is called the **matrix sign function**, analogous to the scalar sign function mapping all singular values to 1.

**Muon Algorithm**:
1. Compute gradient $G_t = \nabla_W \mathcal{L}(W_{t-1})$
2. Update momentum $M_t = \beta M_{t-1} + G_t$
3. Compute orthogonalized update $\Delta_t = \text{msign}(M_t)$
4. Update parameters $W_t = W_{t-1} - \eta \cdot \Delta_t$

**Why does orthogonalization work?**

$\text{msign}(G) = UV^\top$ maps all singular values to 1:

$$G = U \cdot \text{diag}(\sigma_1, \ldots, \sigma_r) \cdot V^\top \xrightarrow{\text{msign}} U \cdot \text{diag}(1, \ldots, 1) \cdot V^\top$$

Effects:
- **Principal directions are suppressed**: $\sigma_1 \to 1$, update magnitude decreases
- **Rare directions are amplified**: $\sigma_r \to 1$, update magnitude increases
- **Directional information preserved**: $U, V$ unchanged, only "step size" changes

**Geometric Intuition**: Imagine the loss function's contours as a narrow ellipse (high condition number). Adam walks along the gradient direction, easily oscillating in the narrow valley. Muon "compresses the ellipse into a circle" — walking the same step size in all directions, which is exactly the effect of Newton's method, but without computing the Hessian.

### 3.3 Newton-Schulz Iteration

Directly computing SVD has complexity $O(\min(d_{out}, d_{in})^3)$, which is unacceptable for large matrices. Muon uses **Newton-Schulz iteration** to efficiently approximate $\text{msign}(M)$:

$$X_{k+1} = aX_k + b(X_k X_k^\top)X_k + c(X_k X_k^\top)^2 X_k$$

Where $a = 3.4445$, $b = -4.7750$, $c = 2.0315$ are optimized coefficients.

```python
def newton_schulz5(G, steps=5, eps=1e-7):
    """Approximate computation of msign(G)"""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # Normalize

    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T
    return X
```

In practice, **5 iterations** achieve sufficient precision, with computational overhead typically < 1%.

### 3.4 Four Versions of Muon

| Version | Scaling Factor | Characteristics |
|---------|----------------|-----------------|
| Naive | $1$ | Simplest, but learning rate not transferable |
| KellerJordan | $\sqrt{\max(1, d_{out}/d_{in})}$ | Default version |
| MuP | $\sqrt{d_{out}/d_{in}}$ | Learning rate transferable |
| Moonlight | $0.2 \times \sqrt{\max(d_{out}, d_{in})}$ | Can directly use Adam learning rate |

### 3.5 Which Parameters Use Muon?

| Parameter Type | Optimizer | Reason |
|----------------|-----------|--------|
| Hidden layer Linear weights | Muon | Core matrix parameters |
| Attention $W_Q, W_K, W_V, W_O$ | Muon | Matrix parameters |
| MLP $W_{gate}, W_{up}, W_{down}$ | Muon | Matrix parameters |
| Embedding layer | AdamW | Essentially a lookup table |
| LayerNorm parameters | AdamW | 1D vector |
| Bias | AdamW | 1D vector |

### 3.6 Large-scale Training: Moonlight

Moonshot AI validated Muon's large-scale scalability in the Moonlight model.

**Performance Comparison**:

| Metric | Muon | AdamW |
|--------|------|-------|
| Compute efficiency | ~2× | Baseline |
| FLOPs to reach same performance | 52% | 100% |
| Sample efficiency | 1.92× | Baseline |

**Moonlight Model Specs**:
- Total parameters: 15.29B (MoE architecture)
- Active parameters: 2.24B
- Training data: 5.7T tokens

### 3.7 Practical Guide

**Hyperparameter Settings**:
- **Momentum coefficient**: $\beta = 0.95$ (slightly larger than Adam's 0.9)
- **Newton-Schulz steps**: 5 steps
- **Learning rate**: Moonlight version directly uses Adam learning rate; other versions multiply by $0.2\sqrt{d_{hidden}}$

**Considerations**:
1. Muon only used for 2D matrix parameters, other parameters use AdamW
2. Correctly identify $d_{in}$ and $d_{out}$ in the framework (PyTorch vs Keras differ)
3. bfloat16 precision is sufficient, float32 not needed

## 4. Practical Recommendations

### 4.1 Distributed Strategy Selection

**Small Models (< 10B)**:
- Fits on single card: Use DDP
- Doesn't fit on single card: Use FSDP or ZeRO-2

**Medium Models (10B - 100B)**:
- Single node: FSDP/ZeRO-3 + activation checkpointing
- Multi-node: 3D parallelism (TP=8 intra-node, PP inter-node)

**Super Large Models (100B+)**:
- Must use 3D parallelism
- Combine expert parallelism (MoE)
- Consider FP8 mixed precision

### 4.2 Data Engineering Recommendations

- **Pretraining**: Invest in data processing pipeline, deduplication and filtering are crucial
- **SFT**: Quality first, manually review every data point
- **Preference alignment**: Ensure annotation consistency, avoid noisy labels
- **Continuous improvement**: Establish data flywheel, continuously collect and iterate

## 5. Summary

This article comprehensively analyzes the three pillars of large model training:

| Domain | Key Technologies | Representative Work |
|--------|------------------|---------------------|
| Data Engineering | Deduplication, filtering, mixing, curriculum learning | FineWeb, LIMA |
| Distributed Training | ZeRO, 3D parallelism, FSDP | DeepSpeed, Megatron |
| Optimizer | Matrix orthogonalization, Newton-Schulz | Muon, Moonlight |

These techniques collectively support the efficient training of models with hundreds of billions of parameters and are the infrastructure of the large model era.

In the next article, we will discuss **Evaluation and Benchmarks**, introducing how to comprehensively assess model capabilities.
