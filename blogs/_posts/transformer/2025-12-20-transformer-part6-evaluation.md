---
layout: post
title: "Transformer 学习笔记（六）：评测与 Benchmark"
date: 2025-12-20 10:50:00
author: Qi Lu
categories: [Deep Learning, Transformer]
tags: [Transformer, Evaluation]
math: true
lang: zh
translation: /en/transformer-part6-evaluation/
series: transformer
series_order: 6
---

本文是 Transformer 系列的第六篇，系统介绍大语言模型的 **评测与 Benchmark**。评测是一个复杂且快速演进的领域，本文重点关注 2024 年以来顶级模型普遍采用的评测基准。

## 1. 评测体系概述

### 1.1 为什么需要多维度评测

单一 benchmark 无法全面反映模型能力：

- **能力多样性**：知识、推理、代码、指令遵循等维度独立
- **数据污染**：训练数据可能包含测试集，导致虚高
- **评测饱和**：旧 benchmark 被刷榜，区分度下降

### 1.2 现代评测框架

主流模型发布时通常报告以下类别的 benchmark：

| 维度 | 核心 Benchmark | 备注 |
|------|----------------|------|
| 知识与理解 | MMLU, MMLU-Pro, C-Eval | 多学科知识 |
| 推理能力 | GPQA, ARC-C, BBH | 复杂推理 |
| 数学能力 | GSM8K, MATH-500, AIME | 从小学到竞赛 |
| 代码能力 | HumanEval, LiveCodeBench | 代码生成与执行 |
| 指令遵循 | IFEval, MT-Bench | 指令理解与执行 |
| 长上下文 | RULER, LongBench | 长文本处理 |
| 多语言 | MGSM, C-Eval | 非英语能力 |
| 安全对齐 | TruthfulQA, BBQ | 真实性与偏见 |

## 2. 知识与理解

### 2.1 MMLU (Massive Multitask Language Understanding)

MMLU 是最广泛使用的知识评测基准，覆盖 57 个学科：

- **规模**：约 14,000 道四选一题目
- **学科**：STEM、人文、社科、其他
- **难度**：从高中到研究生水平

**评测方式**：
- Zero-shot 或 Few-shot（通常 5-shot）
- 计算模型对 A/B/C/D 选项的概率
- 报告整体准确率和各学科准确率

**当前水平**（5-shot）：

| 模型 | MMLU | 发布时间 |
|------|------|----------|
| GPT-4o | 88.7% | 2024.05 |
| Claude 3.5 Sonnet | 88.7% | 2024.06 |
| DeepSeek-V3 | 88.5% | 2024.12 |
| Qwen2.5-72B | 86.1% | 2024.09 |
| LLaMA 3.1-405B | 88.6% | 2024.07 |

### 2.2 MMLU-Pro

MMLU 的升级版，解决原版的问题：

- **更多选项**：从 4 选 1 变为 10 选 1，降低猜测收益
- **更难题目**：过滤简单题，保留需要推理的题目
- **减少噪声**：修正原 MMLU 中的错误标注

**区分度更强**：MMLU 上 GPT-4 与 Claude 差距约 1%，MMLU-Pro 上差距扩大到 5-10%，更能反映真实能力差异。

### 2.3 GPQA (Graduate-Level Google-Proof QA)

针对研究生水平的专业问题：

- **来源**：物理、化学、生物领域的博士生出题
- **特点**：问题设计为"Google-proof"，搜索引擎难以直接找到答案
- **难度**：领域专家准确率约 65%，非专家约 30%

**GPQA-Diamond** 是其中最难的子集，是区分顶级模型的关键 benchmark：

| 模型 | GPQA-Diamond |
|------|--------------|
| DeepSeek-R1 | 71.5% |
| o1-preview | 73.3% |
| DeepSeek-V3 | 59.1% |
| Claude 3.5 Sonnet | 59.4% |
| GPT-4o | 53.6% |

## 3. 推理能力

### 3.1 BBH (BIG-Bench Hard)

BIG-Bench 中最具挑战性的 23 个任务：

- **任务类型**：逻辑推理、因果判断、算法执行等
- **特点**：之前模型表现接近随机
- **评测**：通常使用 Chain-of-Thought prompting

### 3.2 ARC (AI2 Reasoning Challenge)

科学推理问题：

- **ARC-Easy**：简单科学问题
- **ARC-Challenge**：需要多步推理的难题
- **来源**：美国 3-9 年级科学考试

### 3.3 HellaSwag

常识推理与句子补全：

- 给定场景描述，选择最合理的后续
- 测试模型的常识理解能力
- 当前顶级模型准确率 > 95%，区分度下降

## 4. 数学能力

### 4.1 GSM8K

小学数学应用题：

- **规模**：8,500 道题目
- **难度**：2-8 步推理
- **特点**：需要理解题意并进行多步计算

当前顶级模型准确率 > 95%，已接近饱和。

### 4.2 MATH

竞赛级数学问题：

- **来源**：AMC、AIME 等数学竞赛
- **难度分级**：Level 1-5，Level 5 最难
- **领域**：代数、几何、数论、概率等

**MATH-500**：从 MATH 数据集中精选的 500 道高难度题目，是当前主流评测标准。

### 4.3 AIME (American Invitational Mathematics Examination)

美国数学邀请赛：

- 15 道填空题，每题答案为 0-999 的整数
- 代表高中竞赛最高水平
- 是区分推理模型（o1、R1）与普通模型的关键 benchmark

**数学 Benchmark 性能对比**：

| 模型 | GSM8K | MATH-500 | AIME 2024 |
|------|-------|----------|-----------|
| o1 | 96.4% | 96.4% | 74% |
| DeepSeek-R1 | 97.3% | 97.3% | 79.8% |
| DeepSeek-V3 | 91.1% | 90.2% | 39.2% |
| Claude 3.5 Sonnet | 96.4% | 78.3% | - |
| GPT-4o | 95.8% | 76.6% | - |

## 5. 代码能力

### 5.1 HumanEval

Python 函数生成：

- **规模**：164 道题目
- **形式**：给定函数签名和 docstring，生成实现
- **评测**：Pass@k（k 次采样至少一次通过）

**HumanEval+**：增加更多测试用例，减少假阳性。

### 5.2 LiveCodeBench

**2024 年最重要的代码评测创新**，解决数据污染问题：

- **持续更新**：从 LeetCode、AtCoder、CodeForces 持续收集新题
- **时间标记**：每道题有发布日期，可验证是否在训练数据截止日期之后
- **多维度**：代码生成、自我修复、测试输出预测

**为什么 LiveCodeBench 重要**：
- HumanEval 已被"刷榜"，很多模型在训练数据中见过
- LiveCodeBench 的新题确保公平评测
- 是当前评估代码能力的金标准

### 5.3 SWE-bench

软件工程真实任务：

- **任务**：修复 GitHub 上真实的 issue
- **形式**：给定代码仓库和 issue 描述，生成 patch
- **难度**：需要理解大型代码库，非常具有挑战性

**代码 Benchmark 性能对比**：

| 模型 | HumanEval | LiveCodeBench | SWE-bench Verified |
|------|-----------|---------------|-------------------|
| Claude 3.5 Sonnet | 92.0% | 41.4% | 50.8% |
| DeepSeek-V3 | 82.6% | 40.5% | 42.0% |
| GPT-4o | 90.2% | 34.2% | 38.4% |

## 6. 指令遵循

### 6.1 IFEval (Instruction Following Evaluation)

测试模型严格遵循指令的能力：

- **规模**：500+ 条带约束的指令
- **约束类型**：
  - 长度约束："写超过 400 字"
  - 格式约束："用 JSON 格式输出"
  - 内容约束："至少提到 3 次 AI"
  - 结构约束："分成 5 个段落"
- **评测**：约束是否被满足（可程序验证）

**两种指标**：
- **Prompt-level**：整个 prompt 的所有约束都满足
- **Instruction-level**：单个约束的满足率

IFEval 是 Open LLM Leaderboard 的核心 benchmark 之一。

### 6.2 MT-Bench

多轮对话评测：

- **形式**：80 个两轮对话
- **评分**：GPT-4 作为评判，1-10 分
- **类别**：写作、角色扮演、推理、数学等 8 类

### 6.3 Arena-Hard

基于 Chatbot Arena 的困难子集：

- 从真实用户对话中筛选的 500 道困难问题
- GPT-4-Turbo 作为评判
- 与 Chatbot Arena 排名高度相关

## 7. 长上下文评测

### 7.1 RULER

长上下文能力的系统评测：

**任务类型**：
- **Needle-in-a-Haystack**：在长文本中找到特定信息
- **Multi-hop QA**：需要整合多处信息
- **Aggregation**：统计或汇总信息

**长度范围**：4K 到 128K+

**评测**：不同长度下的准确率衰减曲线

### 7.2 LongBench

多任务长文本评测：

- 6 大类 21 个任务
- 平均长度约 15K tokens
- 涵盖单文档/多文档 QA、摘要、代码补全等

### 7.3 Needle-in-a-Haystack

最简单但直观的长上下文测试：

- 在长文本的随机位置插入一个"针"（关键信息）
- 测试模型能否准确检索
- 生成位置-长度的热力图

## 8. 多语言评测

### 8.1 C-Eval / CMMLU

中文知识评测：

- **C-Eval**：52 个学科，覆盖中国教育体系
- **CMMLU**：中文版 MMLU
- 是评估中文能力的核心 benchmark

### 8.2 MGSM (Multilingual GSM)

多语言数学推理：

- GSM8K 翻译成 10 种语言
- 测试非英语数学推理能力
- 揭示模型的语言偏差

## 9. 安全与对齐

### 9.1 TruthfulQA

测试模型是否会生成虚假但常见的错误信息：

- 817 个问题，涵盖常见误解
- 人类因为偏见经常答错的问题
- 测试模型是否学习了人类的错误信念

### 9.2 SimpleQA

事实准确性评测（OpenAI 2024 发布）：

- 简单的事实性问题
- 测试模型是否会"幻觉"错误信息
- 评估拒绝回答（"我不知道"）的能力

## 10. 综合评测平台

### 10.1 Open LLM Leaderboard

Hugging Face 维护的开放评测平台：

**当前版本（v2）包含**：
- IFEval（指令遵循）
- BBH（复杂推理）
- MATH Level 5（高难度数学）
- GPQA（研究生水平 QA）
- MuSR（多步推理）
- MMLU-Pro（知识理解）

特点：任何人可以提交模型评测，透明、可复现。

### 10.2 Chatbot Arena

基于真实用户投票的评测：

- 用户盲评两个模型的回复
- 使用 ELO 排名系统
- 被认为是最能反映真实用户偏好的评测
- 但难以控制变量，不够"科学"

### 10.3 LiveBench

抗污染的动态评测：

- 每月更新题目
- 严格的时间控制防止数据污染
- 涵盖数学、代码、推理、语言等多个维度

## 11. 评测最佳实践

### 11.1 避免数据污染

- **使用新 benchmark**：LiveCodeBench、LiveBench 等持续更新
- **时间切分**：确保评测数据晚于训练数据截止日期
- **多源验证**：同一能力用多个 benchmark 交叉验证

### 11.2 评测配置标准化

- 明确报告 Few-shot 数量
- 统一 prompt 模板
- 使用相同的解码参数（temperature、top_p 等）

### 11.3 选择合适的 Benchmark

| 评测目标 | 推荐 Benchmark |
|----------|----------------|
| 通用能力快速评估 | MMLU-Pro, GPQA-Diamond |
| 数学推理 | MATH-500, AIME |
| 代码生成 | LiveCodeBench, SWE-bench |
| 指令遵循 | IFEval |
| 长上下文 | RULER, Needle-in-Haystack |
| 中文能力 | C-Eval, CMMLU |
| 真实用户偏好 | Chatbot Arena, Arena-Hard |

## 12. 总结

本文系统介绍了大语言模型的评测体系：

| 维度 | 关键 Benchmark | 当前趋势 |
|------|----------------|----------|
| 知识 | MMLU-Pro, GPQA | 向更难、更专业发展 |
| 数学 | MATH-500, AIME | 竞赛级题目成为标准 |
| 代码 | LiveCodeBench | 动态更新防止污染 |
| 指令 | IFEval | 可程序验证的约束 |
| 综合 | Chatbot Arena | 真实用户偏好 |

**评测的局限性**：
- Benchmark ≠ 真实能力：高分不代表实际应用效果好
- 优化目标错位：过度优化 benchmark 可能损害通用能力
- 评测演进：benchmark 会饱和，需要持续更新
- 人类评估：某些能力（创意、共情）难以自动评测

下一篇我们将讨论 **部署优化**，包括模型量化和推理加速技术。
