---
layout: post
title: "LLM-RL 中的熵控制：负样本、多样性与探索效率"
date: 2025-12-23 12:00:00
author: 知乎转载
tags: [RL, Entropy, GRPO, Negative Samples]
lang: zh
source: https://zhuanlan.zhihu.com/p/1913295888731861490
---

## 引言

近期，社区开源了很多关于 entropy 在 RL 中的作用的研究 [1][2][3][9]，基本结论是：

**合理的熵控制，可以让 RL 训的时间更久、效果越好**（保持多样性，避免坍缩到某几个输出 pattern）。输出多样性更容易提升模型的 exploration 效果（熵是一种衡量输出多样性的方法），其他多样性衡量的方法如 SMI/DPP/Self-BLEU 等等，都需要在 group-level 计算（大部分通过 reward-shaping 控制熵的变化如 [6]）。

当然，[1][2] 均是在 off-policy 的 setting 下给出的解决方案，[3][9] 均验证了 exact-onpolicy 能够更好的保持 policy 的 entropy。

> 笔者日常实验都是 exact-onpolicy，agent 环境下会比较慢（加了 acc-filter 等等后，为了不浪费 rollout 样本，即使 exact-onpolicy-setting 也会变成 offpolicy-setting，全看 rollout 队列/exp 队列的大小设置）

## 优化目标与熵下降

从优化目标入手，其优化目标是 **reverse-KL**，优化过程会使得模型输出多样性下降，进而导致熵下降。

**熵下降几乎不可避免，但可以缓解**，常见方法包括：
- clip-higher
- entropy-loss
- low-entropy-token-mask
- 等等

## 负样本在 Policy Gradient 中的作用

[4][5] 则另辟蹊径探讨了 **negative-samples 在 policy-gradient 中的作用**。

### 负样本 SFT 的研究 [5]

[5] 研究了 instruct-model 的负样本 SFT 会带来怎样的影响：
- 指标依然可以涨，但不如 positive-samples-SFT

作者进一步对 step 做了 segmentation，并通过 LLM-as-Judge 对 negative-sample 的 step 做了更细致的打分，**避免过度惩罚负样本中的"正确 step"**（捞起来哪些思路正确的 token）。

### 负样本在 Zero-RL 中的作用 [4]

[4] 则进一步探讨了负样本在 zero-RL 中的作用：

**简而言之：负样本对于熵的保持有重要作用**

且训练 steps 相对较少的情况下，可以提升效果（训的过多肯定会崩）。

这里，**NSR** 代表 Negative Sample Reinforcement（GRPO/REINFORCE 等等 critic-free 的方法中，同一个 response 中每个 token 的 advantages 是一样的，对于答案错误的 trajectory 可以看成某种 "NSR"）。

从 [4] 中可以看到，zero-RL-setting 下，**NSR 在前 100 多个 step 保持着更好的 entropy**（几乎与 base-model 接近），而其他方法或多或少都会下降至某个区间（上下抖动）。

近期很多工作也会**离线筛选 RL-data**（保留某个解决率区间的数据），结合 [4] 来看，也是期望能够引入一定比例的负样本，保持基线模型的熵较高（更容易探索）。

## 思考

### 从 KL-Divergence 角度

从 KL-divergence 角度出发，**reverse-KL 不可避免降低输出多样性**。

### 从 PG-Loss 角度

从 PG-loss 出发，大概可以分出来两部分：
1. **positive-sample-reinforce**
2. **negative-sample-reinforce**

**Positive-sample-reinforce** 为 SFT，虽然是 MLE 优化（有 mode-coverage 的效果），但受限于 SFT 的 response 多样性，SFT 后的模型输出也会变窄（熵掉的比较多）。

而 [4][5] 则进一步探讨了 **negative-sample-reinforce** 的作用：
- 即使答案错误的 SFT/zero-RL，在合适的训练 setting 下，也能让下游 performance 提升
- [4] 进一步使用 negative-sample-reinforce 在 zero-RL-setting 下，可以更好的保持 base-model 的 entropy，并在合适的训练 steps 后效果提升显著

### Unlikelihood Training 视角

不从 RL 角度思考问题，以上工作大概都可以归结为某种 **unlikelihood-training** [6]，即更为有效地使用负样本提升优化效率和效果（或者 unlearning）。

### 数据领域的影响

由于目前都在 math/code 上面实验（math/code 也是各家 base-model 重点 pretrain/CT/数据退火的重要组成部分），使用公开的 math/code 数据集基本都会被训进去（使得熵更容易下降）。

而一些合成的 **logic-game** [7] 则较少地被用于 base-model 训练（体现为起始的熵/熵下降都更缓慢）。

### 缓解熵下降的方向

**从数据角度**：
- 合成预训练阶段没见过的数据
- 更为合理的 negative-sample 保留（比如只保留 acc 在 [0.1, 0.5] 之间的数据）
- 加入 diversity-reward [8]

**从算法角度**：
- 可能得找一个新的 divergence，从根本上避免熵下降的问题

> （说明：由于都是基于 Qwen2.5 系列模型的工作，实际上很多结论都是 Qwen2.5-based-xxx）

## 一些实验的观察

参考之前的工作，高熵 token 更容易保持模型的熵，我们也简单在 reward-shaping 阶段，鼓励**答案正确且出现高熵 token 的 response**，对比 baseline（只有答案正确的 reward），二者的熵如下：

### Baseline（只有答案正确的 reward）
- V 型曲线，后续涨不回初始 entropy

### 答案正确且有 high-entropy-token 鼓励的 reward
- V 型曲线，后续的熵超过初始曲线

## 正样本权重的问题

笔者觉得关键还是 **positive-sample 的权重**。

当前的训练方法（GRPO 等等），正确样本的 advantage 都是一样的，这个显然不太对，而且：
- **logp 越高的样本越容易被强化**（logp 低的样本越容易被忽略）
- 进而导致只能采样出来 logp 高的 sample（多样性下降）
- **赢的漂亮的方法加大奖励**（但 logp 可能会比较低）

### 改进思路

如果有更好的质量评估模型（长度/思考密度/格式/流畅度等等），对正确样本里面 **logp 低的样本赋予更高的权重**，可能也能缓解 entropy 下降的问题。

类似 **MCTS 里面的 PUCT 准则**：
- 忽略 logp 是不太对的
- 但 logp 需要有其他分数平衡
- 避免只选 logp 高的 node（容易忽略 logp 低但 node.Q 高的 node）

### 缓解熵下降的两条路

1. **修正正样本的 advantage**：对正样本里面 logp 低的样本有 uplift [11]
2. **修正 token-advantage** [1][12]

## 探索效率的权衡

熵高更容易保持探索，但**探索效率是否高也是一个问题**。

熵高了探索效率可能会下降（比如 [1] 中，high-entropy-token 的优化在 1k-step 后有显著优势，前 1k-step 基本重合，但实际应用估计也没人想训这么多的 step）。

> 这里 step 指参数更新 step，rollout-steps 相对更少一些

## 参考文献

1. [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2505.xxxxx)
2. [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.xxxxx)
3. [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.xxxxx)
4. [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2505.xxxxx)
5. [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning](https://arxiv.org/abs/2505.xxxxx)
6. [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319)
7. [SynLogic](https://github.com/MiniMax-AI/SynLogic/tree/main)
8. [DRA-GRPO](https://github.com/xiwenc1/DRA-GRPO)
9. [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2505.xxxxx)
10. 唐国梁Tommy：RL与SFT的参数更新之谜：强化学习仅更新一小部分参数
11. [Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening](https://arxiv.org/abs/2505.xxxxx)
12. [SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization](https://arxiv.org/abs/2505.xxxxx)
