---
layout: post
title: "Transformer Notes (VI): Evaluation and Benchmarks"
date: 2025-12-20 10:50:00
author: Qi Lu
tags: [Transformer, Evaluation]
lang: en
translation: /transformer-part6-evaluation/
---

This is the sixth article in the Transformer series, systematically introducing **evaluation and benchmarks** for large language models. Evaluation is a complex and rapidly evolving field. This article focuses on evaluation benchmarks widely adopted by top-tier models since 2024.

## 1. Evaluation System Overview

### 1.1 Why Multi-Dimensional Evaluation is Needed

A single benchmark cannot comprehensively reflect model capabilities:

- **Capability Diversity**: Knowledge, reasoning, code, instruction following, and other dimensions are independent
- **Data Contamination**: Training data may include test sets, leading to inflated scores
- **Evaluation Saturation**: Old benchmarks get saturated through optimization, reducing discriminative power

### 1.2 Modern Evaluation Framework

Mainstream models typically report benchmarks in the following categories when released:

| Dimension | Core Benchmarks | Notes |
|-----------|----------------|-------|
| Knowledge & Understanding | MMLU, MMLU-Pro, C-Eval | Multidisciplinary knowledge |
| Reasoning | GPQA, ARC-C, BBH | Complex reasoning |
| Mathematics | GSM8K, MATH-500, AIME | Elementary to competition level |
| Code | HumanEval, LiveCodeBench | Code generation and execution |
| Instruction Following | IFEval, MT-Bench | Instruction understanding and execution |
| Long Context | RULER, LongBench | Long text processing |
| Multilingual | MGSM, C-Eval | Non-English capabilities |
| Safety & Alignment | TruthfulQA, BBQ | Truthfulness and bias |

## 2. Knowledge and Understanding

### 2.1 MMLU (Massive Multitask Language Understanding)

MMLU is the most widely used knowledge evaluation benchmark, covering 57 subjects:

- **Scale**: Approximately 14,000 multiple-choice questions (4 choices)
- **Subjects**: STEM, humanities, social sciences, other
- **Difficulty**: From high school to graduate level

**Evaluation Method**:
- Zero-shot or Few-shot (typically 5-shot)
- Calculate model probabilities for A/B/C/D options
- Report overall accuracy and per-subject accuracy

**Current Performance** (5-shot):

| Model | MMLU | Release Date |
|-------|------|--------------|
| GPT-4o | 88.7% | 2024.05 |
| Claude 3.5 Sonnet | 88.7% | 2024.06 |
| DeepSeek-V3 | 88.5% | 2024.12 |
| Qwen2.5-72B | 86.1% | 2024.09 |
| LLaMA 3.1-405B | 88.6% | 2024.07 |

### 2.2 MMLU-Pro

An upgraded version of MMLU that addresses issues with the original:

- **More Options**: Changed from 4 choices to 10, reducing guessing benefits
- **Harder Questions**: Filters easy questions, retaining those requiring reasoning
- **Reduced Noise**: Corrects incorrect labels in the original MMLU

**Better Discrimination**: The gap between GPT-4 and Claude on MMLU is about 1%, but on MMLU-Pro it expands to 5-10%, better reflecting true capability differences.

### 2.3 GPQA (Graduate-Level Google-Proof QA)

Targets graduate-level specialized questions:

- **Source**: Questions created by PhD students in physics, chemistry, and biology
- **Characteristics**: Questions designed to be "Google-proof" - difficult to find answers directly through search engines
- **Difficulty**: Domain experts achieve ~65% accuracy, non-experts ~30%

**GPQA-Diamond** is the most difficult subset, serving as a key benchmark for distinguishing top-tier models:

| Model | GPQA-Diamond |
|-------|--------------|
| DeepSeek-R1 | 71.5% |
| o1-preview | 73.3% |
| DeepSeek-V3 | 59.1% |
| Claude 3.5 Sonnet | 59.4% |
| GPT-4o | 53.6% |

## 3. Reasoning Capabilities

### 3.1 BBH (BIG-Bench Hard)

The 23 most challenging tasks from BIG-Bench:

- **Task Types**: Logical reasoning, causal judgment, algorithm execution, etc.
- **Characteristics**: Previous models performed close to random
- **Evaluation**: Typically uses Chain-of-Thought prompting

### 3.2 ARC (AI2 Reasoning Challenge)

Scientific reasoning questions:

- **ARC-Easy**: Simple science questions
- **ARC-Challenge**: Difficult questions requiring multi-step reasoning
- **Source**: U.S. 3rd-9th grade science exams

### 3.3 HellaSwag

Common sense reasoning and sentence completion:

- Given a scenario description, choose the most reasonable continuation
- Tests model's common sense understanding
- Current top models achieve >95% accuracy, reducing discriminative power

## 4. Mathematical Capabilities

### 4.1 GSM8K

Elementary school math word problems:

- **Scale**: 8,500 questions
- **Difficulty**: 2-8 step reasoning
- **Characteristics**: Requires understanding problem text and performing multi-step calculations

Current top models achieve >95% accuracy, approaching saturation.

### 4.2 MATH

Competition-level mathematical problems:

- **Source**: Mathematical competitions like AMC, AIME
- **Difficulty Levels**: Level 1-5, Level 5 being hardest
- **Domains**: Algebra, geometry, number theory, probability, etc.

**MATH-500**: A curated set of 500 high-difficulty problems from the MATH dataset, currently the mainstream evaluation standard.

### 4.3 AIME (American Invitational Mathematics Examination)

American Invitational Mathematics Examination:

- 15 fill-in-the-blank questions, each answer is an integer from 0-999
- Represents the highest level of high school competitions
- Key benchmark for distinguishing reasoning models (o1, R1) from regular models

**Math Benchmark Performance Comparison**:

| Model | GSM8K | MATH-500 | AIME 2024 |
|-------|-------|----------|-----------|
| o1 | 96.4% | 96.4% | 74% |
| DeepSeek-R1 | 97.3% | 97.3% | 79.8% |
| DeepSeek-V3 | 91.1% | 90.2% | 39.2% |
| Claude 3.5 Sonnet | 96.4% | 78.3% | - |
| GPT-4o | 95.8% | 76.6% | - |

## 5. Code Capabilities

### 5.1 HumanEval

Python function generation:

- **Scale**: 164 problems
- **Format**: Given function signature and docstring, generate implementation
- **Evaluation**: Pass@k (at least one pass in k samples)

**HumanEval+**: Adds more test cases, reducing false positives.

### 5.2 LiveCodeBench

**The most important code evaluation innovation in 2024**, solving data contamination problems:

- **Continuous Updates**: Continuously collects new problems from LeetCode, AtCoder, CodeForces
- **Time Stamps**: Each problem has a publication date, allowing verification of whether it's after training data cutoff
- **Multi-Dimensional**: Code generation, self-repair, test output prediction

**Why LiveCodeBench is Important**:
- HumanEval has been "gamed" - many models have seen it in training data
- LiveCodeBench's new problems ensure fair evaluation
- Currently the gold standard for evaluating code capabilities

### 5.3 SWE-bench

Real-world software engineering tasks:

- **Task**: Fix real issues from GitHub
- **Format**: Given code repository and issue description, generate patch
- **Difficulty**: Requires understanding large codebases, very challenging

**Code Benchmark Performance Comparison**:

| Model | HumanEval | LiveCodeBench | SWE-bench Verified |
|-------|-----------|---------------|-------------------|
| Claude 3.5 Sonnet | 92.0% | 41.4% | 50.8% |
| DeepSeek-V3 | 82.6% | 40.5% | 42.0% |
| GPT-4o | 90.2% | 34.2% | 38.4% |

## 6. Instruction Following

### 6.1 IFEval (Instruction Following Evaluation)

Tests the model's ability to strictly follow instructions:

- **Scale**: 500+ instructions with constraints
- **Constraint Types**:
  - Length constraints: "Write more than 400 words"
  - Format constraints: "Output in JSON format"
  - Content constraints: "Mention AI at least 3 times"
  - Structure constraints: "Divide into 5 paragraphs"
- **Evaluation**: Whether constraints are satisfied (programmatically verifiable)

**Two Metrics**:
- **Prompt-level**: All constraints in the entire prompt are satisfied
- **Instruction-level**: Satisfaction rate of individual constraints

IFEval is one of the core benchmarks in the Open LLM Leaderboard.

### 6.2 MT-Bench

Multi-turn conversation evaluation:

- **Format**: 80 two-turn conversations
- **Scoring**: GPT-4 as judge, scores 1-10
- **Categories**: 8 categories including writing, role-playing, reasoning, math, etc.

### 6.3 Arena-Hard

A difficult subset based on Chatbot Arena:

- 500 difficult questions selected from real user conversations
- GPT-4-Turbo as judge
- Highly correlated with Chatbot Arena rankings

## 7. Long Context Evaluation

### 7.1 RULER

Systematic evaluation of long context capabilities:

**Task Types**:
- **Needle-in-a-Haystack**: Find specific information in long text
- **Multi-hop QA**: Requires integrating information from multiple locations
- **Aggregation**: Statistical or summary information

**Length Range**: 4K to 128K+

**Evaluation**: Accuracy degradation curves at different lengths

### 7.2 LongBench

Multi-task long text evaluation:

- 21 tasks across 6 major categories
- Average length ~15K tokens
- Covers single-document/multi-document QA, summarization, code completion, etc.

### 7.3 Needle-in-a-Haystack

The simplest but most intuitive long context test:

- Insert a "needle" (key information) at a random position in long text
- Tests whether the model can accurately retrieve it
- Generates position-length heatmaps

## 8. Multilingual Evaluation

### 8.1 C-Eval / CMMLU

Chinese knowledge evaluation:

- **C-Eval**: 52 subjects, covering the Chinese education system
- **CMMLU**: Chinese version of MMLU
- Core benchmarks for evaluating Chinese capabilities

### 8.2 MGSM (Multilingual GSM)

Multilingual mathematical reasoning:

- GSM8K translated into 10 languages
- Tests non-English mathematical reasoning capabilities
- Reveals language biases in models

## 9. Safety and Alignment

### 9.1 TruthfulQA

Tests whether models generate false but common misinformation:

- 817 questions covering common misconceptions
- Questions that humans often answer incorrectly due to biases
- Tests whether models learn human erroneous beliefs

### 9.2 SimpleQA

Factual accuracy evaluation (released by OpenAI in 2024):

- Simple factual questions
- Tests whether models "hallucinate" false information
- Evaluates ability to refuse answering ("I don't know")

## 10. Comprehensive Evaluation Platforms

### 10.1 Open LLM Leaderboard

Open evaluation platform maintained by Hugging Face:

**Current version (v2) includes**:
- IFEval (instruction following)
- BBH (complex reasoning)
- MATH Level 5 (high-difficulty math)
- GPQA (graduate-level QA)
- MuSR (multi-step reasoning)
- MMLU-Pro (knowledge understanding)

Features: Anyone can submit models for evaluation, transparent and reproducible.

### 10.2 Chatbot Arena

Evaluation based on real user voting:

- Users blindly evaluate responses from two models
- Uses ELO ranking system
- Considered the best reflection of real user preferences
- But difficult to control variables, less "scientific"

### 10.3 LiveBench

Dynamic evaluation resistant to contamination:

- Updates questions monthly
- Strict temporal controls to prevent data contamination
- Covers multiple dimensions: math, code, reasoning, language, etc.

## 11. Evaluation Best Practices

### 11.1 Avoiding Data Contamination

- **Use New Benchmarks**: Continuously updated ones like LiveCodeBench, LiveBench
- **Temporal Separation**: Ensure evaluation data is later than training data cutoff date
- **Multi-Source Validation**: Cross-validate the same capability with multiple benchmarks

### 11.2 Standardized Evaluation Configuration

- Clearly report few-shot count
- Unified prompt templates
- Use same decoding parameters (temperature, top_p, etc.)

### 11.3 Choosing Appropriate Benchmarks

| Evaluation Goal | Recommended Benchmarks |
|-----------------|------------------------|
| Quick general capability assessment | MMLU-Pro, GPQA-Diamond |
| Mathematical reasoning | MATH-500, AIME |
| Code generation | LiveCodeBench, SWE-bench |
| Instruction following | IFEval |
| Long context | RULER, Needle-in-Haystack |
| Chinese capabilities | C-Eval, CMMLU |
| Real user preferences | Chatbot Arena, Arena-Hard |

## 12. Summary

This article systematically introduces the evaluation system for large language models:

| Dimension | Key Benchmarks | Current Trends |
|-----------|----------------|----------------|
| Knowledge | MMLU-Pro, GPQA | Toward harder, more specialized |
| Math | MATH-500, AIME | Competition-level problems becoming standard |
| Code | LiveCodeBench | Dynamic updates to prevent contamination |
| Instruction | IFEval | Programmatically verifiable constraints |
| Comprehensive | Chatbot Arena | Real user preferences |

**Limitations of Evaluation**:
- Benchmark ≠ Real Capability: High scores don't necessarily mean good real-world performance
- Misaligned Optimization Goals: Over-optimizing for benchmarks may harm general capabilities
- Evaluation Evolution: Benchmarks saturate and require continuous updates
- Human Evaluation: Some capabilities (creativity, empathy) are difficult to automatically evaluate

In the next article, we will discuss **deployment optimization**, including model quantization and inference acceleration techniques.
