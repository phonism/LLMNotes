---
layout: post
title: "LLM-RL 训练稳定性：根因分析与解决方案"
date: 2025-12-19
author: Qi Lu
tags: [RL, RLHF, PPO, GRPO, GSPO, 训练稳定性, Importance Sampling]
---

## 引言

在大语言模型的强化学习（LLM-RL）训练中，经常会发现训练曲线在稳定上升一段时间后，突然崩溃。无论是复杂的推理 RL 还是多轮工具调用的 Agentic RL，很多从业者都遇到过这种神秘的训练崩溃。

这篇博客将综合 ByteDance 的 [When Speed Kills Stability](https://richardli.xyz/rl-collapse)、Qwen 团队的 [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374)、vLLM 的 [Bitwise Consistent Training](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) 等多篇重要工作，系统性地剖析 LLM-RL 训练不稳定的根因，并总结实用的解决方案。

**符号约定**：

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 当前正在优化的策略（训练引擎计算） |
| $\pi_{\text{old}}$ | 采样时的策略（训练引擎计算，但参数是旧的） |
| $\pi_\text{vllm}$ | 推理引擎计算的策略（vLLM/SGLang，数值上与 $\pi$ 有差异） |
| $\pi_{\text{ref}}$ | 参考策略（KL 正则的锚点，通常是 SFT 模型） |

核心区分：$\pi$ vs $\pi_\text{vllm}$ 是**同一参数在不同引擎上的数值差异**，$\pi_\theta$ vs $\pi_{\text{old}}$ 是**同一引擎在不同时刻的参数差异**。

## 问题现象：突如其来的崩溃

典型的崩溃模式如下：

1. **训练奖励**：稳步上升 → 突然骤降或剧烈震荡
2. **梯度范数**：正常范围 → 瞬间爆炸
3. **PPL（困惑度）**：保持稳定 → 急剧飙升
4. **熵**：逐渐下降 → 异常波动

最令人困惑的是，这种崩溃往往是**不可预测**的——同样的代码、同样的超参数，在不同的 GPU 上可能表现完全不同。

## 根因分析

LLM-RL 训练不稳定有**两个层面的根因**：

1. **系统层面**：Training-Inference Mismatch（推理引擎 $\pi_\text{vllm}$ vs 训练引擎 $\pi$ 的数值差异）
2. **算法层面**：Token-Sequence Mismatch（token-level 优化目标 vs sequence-level 奖励）

这两个问题相互独立但会叠加放大。下面分别分析。

### 根因一：Training-Inference Mismatch

**速度与一致性的矛盾**

现代 LLM-RL 系统通常使用**高速推理引擎**（如 vLLM、SGLang）进行 rollout 采样，而使用**训练框架**（如 FSDP、Megatron-LM）进行参数更新。这两类系统有着截然不同的优化目标：

| 系统 | 优化目标 | 典型技术 |
|------|----------|----------|
| 推理引擎 | 吞吐量最大化 | Speculative Decoding, INT8/FP8, 批次变体 CUDA 核心 |
| 训练框架 | 数值稳定性 | FP32 Master Weights, 确定性算子 |

这种优化目标的分歧导致了**不可避免的数值不一致**。即使参数完全相同，推理引擎计算的 $\pi_\text{vllm}(y\mid x)$ 和训练引擎计算的 $\pi(y\mid x)$ 也会产生差异。

**实际梯度 vs 理论梯度**

理论上，on-policy 的策略梯度应该是：

$$\mathbb{E}_{y \sim \pi_\theta} \left[ R(x,y) \nabla_\theta \log \pi_\theta(y|x) \right]$$

但实际上，由于样本来自推理引擎 $\pi_\text{vllm}$：

$$\mathbb{E}_{y \sim \pi_\text{vllm}} \left[ R(x,y) \nabla_\theta \log \pi_\theta(y|x) \right]$$

这意味着：**你以为在做 on-policy 训练，实际上是在做 off-policy 训练。**

### 根因二：Token-Sequence Mismatch

主流 RL 算法（PPO、GRPO）使用 **token-level 优化目标**，但奖励是 **sequence-level** 的。

**一阶近似的理论基础**：Sequence-level IS weight 可以展开为：

$$\frac{\pi_\theta(y|x)}{\pi_\text{vllm}(y|x)} = \prod_{t=1}^{|y|}(1+\delta_t) \approx 1 + \sum_{t=1}^{|y|}\delta_t$$

其中 $\delta_t = \frac{\pi_\theta(y_t \mid s_t)}{\pi_\text{vllm}(y_t \mid s_t)} - 1$。这说明 token-level 目标是 sequence-level 目标的**一阶近似**，忽略了 $O(\delta^2)$ 的高阶项。

**IS weight 的分解**：Token-level IS weight 可以分解为两个因子：

$$\frac{\pi_\theta(y_t|s_t)}{\pi_\text{vllm}(y_t|s_t)} = \underbrace{\frac{\pi_{\text{old}}(y_t|s_t)}{\pi_\text{vllm}(y_t|s_t)}}_{\text{Training-Inference Discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|s_t)}{\pi_{\text{old}}(y_t|s_t)}}_{\text{Policy Staleness}}$$

- **Training-Inference Discrepancy**：推理引擎 $\pi_\text{vllm}$ 和训练引擎 $\pi$ 的数值差异（根因一）
- **Policy Staleness**：mini-batch 处理过程中策略的漂移

这个分解说明：**两个根因通过乘法叠加**，任何一个偏大都会导致 IS weight 偏离 1，进而破坏一阶近似的有效性。

## 具体表现与挑战

### 挑战一：低概率 Token 陷阱

**Mismatch 在低概率 token 处最为严重**。当 vLLM 采样到一个概率接近 0 的 token 时，FSDP 计算的概率可能比 vLLM 低好几个数量级，导致：

- PPL 爆炸（分母趋近于 0）
- 梯度爆炸
- 训练崩溃

这解释了为什么多轮工具调用场景特别容易崩溃——工具返回的 OOD（Out-of-Distribution）文本会导致模型产生更多低概率 token。

### 挑战二：硬件差异放大问题

不同 GPU 架构的 mismatch 程度差异巨大：

$$\text{vllm-kl}: \quad \text{H20} < \text{L20} < \text{A100}$$

- H20: $5 \times 10^{-4}$ ~ $10^{-3}$
- L20: $10^{-3}$ ~ $10^{-2}$
- A100: $10^{-2}$ ~ $1$

同样的代码在 L20 上崩溃后，从 checkpoint 恢复到 H20 上可以立即稳定训练！

### 挑战三：高方差与 Entropy Collapse

**高方差问题**：长 CoT 场景下，token 级 IS 权重的累积会导致方差爆炸：

$$\prod_{t=1}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} = \exp\left(\sum_t \log \rho_t\right)$$

logprob 差在长序列上**线性累积**，exp 后变成**指数级差异**。后果是梯度由极少数样本、极少数 token 主导，对 MoE 模型尤其灾难——少量极端更新可能打坏 expert routing。

**Entropy Collapse**：RL 优化倾向于增强高奖励 token 的概率，同时压缩低概率 token，导致策略熵持续下降。当熵接近 0 时：
- 探索能力丧失
- 多样性崩溃
- 无法发现新的解决方案

研究表明 policy performance 是以 entropy 消耗为代价的，存在理论上限。

## 解决方案

### 方案一：Sequence-Level Importance Sampling

正确的无偏估计器需要在整个序列上应用 importance ratio：

$$g_{\text{seq}} = \mathbb{E}_{y \sim \pi_\text{vllm}} \left[ \frac{\pi_\theta(y|x)}{\pi_\text{vllm}(y|x)} \cdot R(x,y) \cdot \nabla \log \pi_\theta(y|x) \right]$$

实践中有两种变体：

+ **Truncated IS (TIS)**：对 ratio 进行截断 $$\rho(y) \gets \min(\rho(y), C)$$
+ **Masked IS (MIS)**：对超过阈值的序列直接 mask $$\rho(y) \gets \rho(y) \cdot \mathbb{I}\{\rho(y) \le C\}$$

实验表明 **MIS 比 TIS 效果更好**，不仅稳定了训练，还超过了崩溃前的峰值性能。

### 方案二：Off-Policy Sequence Masking（DeepSeek-V3.2）

DeepSeek-V3.2 采用了更精细的 masking 策略：

$$M_{i,t} = \begin{cases} 0 & \text{if } \hat{A}_{i,t} < 0 \text{ and } \frac{1}{\lvert o_i \rvert}\sum \log \frac{\pi_{\text{old}}}{\pi_\theta} > \delta \\ 1 & \text{otherwise} \end{cases}$$

核心思路：**只 mask 那些 advantage 为负且 off-policy 程度超过阈值 $\delta$ 的序列**。

这里用 $\frac{1}{\lvert o_i \rvert}\sum \log \frac{\pi_{\text{old}}}{\pi_\theta}$ 来衡量 off-policy 程度，本质上是 **per-token 平均 KL 散度**（等价于 log ratio 的几何平均）。这种长度归一化避免了长序列被系统性丢弃的问题。

**为什么只 mask 负 advantage？** 正 advantage 的样本即使 off-policy，仍然提供有用的梯度方向；而负 advantage 的 off-policy 样本可能引入有害的梯度噪声。

DeepSeek-V3.2 还引入了配套的稳定化技术：

**Keep Routing（MoE 专用）**：推理和训练框架的 expert routing 可能不一致。解决方案是保存推理时的 routing 路径，训练时强制使用相同路径。

**Keep Sampling Mask**：top-p/top-k 采样会截断低概率 token，导致 $\pi_{\text{old}}$ 和 $\pi_\theta$ 的 action space 不一致。解决方案是保存采样时的 truncation mask，训练时对 $\pi_\theta$ 应用相同的 mask。

**Unbiased KL Estimation**：标准 K3 estimator 在 off-policy 设置下有偏。修正公式：
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_\theta}{\pi_{\text{old}}} \left[ \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1 \right]$$

### 方案三：Bitwise Consistent Training

另一条路径是**让推理和训练完全一致**。

核心思路：
1. 审计每一个 forward pass 中的 kernel 调用
2. 将 vLLM 的 fused operations（SiLU MLPs、RMSNorm）导入训练框架
3. 实现对应的 backward pass

实验表明，启用 bitwise 一致性后：
- 收敛更快
- 最终奖励更高
- 训练更稳定

但代价是约 **2.4x 的性能下降**。

### 方案四：IcePop（Token-Level Discrepancy Masking）

Ring-1T 提出的 IcePop 在 **token 粒度**上处理 mismatch，与前面的 sequence-level 方法形成互补。

**核心思路**：定义 token-level 的 ratio $k_{i,t} = \frac{\pi(y_t \mid s_t)}{\pi_\text{vllm}(y_t \mid s_t)}$，对超出合理范围的 token 进行 masking：

$$M(k) = \begin{cases} k & \text{if } k \in [\alpha, \beta] \\ 0 & \text{otherwise} \end{cases}$$

典型参数：$\alpha = 0.5$，$\beta = 5.0$。

**双向截断**：与只关注 $k > C$ 的 sequence-level MIS 不同，IcePop 同时处理两个方向：
- $k > \beta$：训练概率远大于推理概率（可能导致梯度爆炸）
- $k < \alpha$：训练概率远小于推理概率（可能导致 PPL 爆炸）

**为什么 token-level 有效？** 在 MoE 模型中，expert routing 的差异会导致 mismatch 在不同 token 位置分布不均匀。Token-level masking 可以精准剔除问题 token，而不是丢弃整个序列。

**与 sequence-level 的对比**：

| 方法 | 粒度 | 优势 | 劣势 |
|------|------|------|------|
| Seq-MIS | 序列 | 理论无偏 | 可能丢弃过多数据 |
| IcePop | Token | 细粒度控制 | 未修正 state occupancy |

实践中可以结合使用：先用 IcePop 处理极端 token，再用 sequence-level 方法处理整体偏移。

### 方案五：GSPO（序列级 IS）

GSPO（Group Sequence Policy Optimization）将 IS 操作提升到序列级别：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right)^{1/\lvert y_i \rvert}$$

**核心改进**：
- 先对 sequence-level ratio 做**长度归一化**，再 clip
- 同一序列的所有 token 共用同一个 IS 权重

**与 GRPO 的区别**：

| 维度 | GRPO | GSPO |
|------|------|------|
| IS 粒度 | Token-level | Sequence-level |
| Clip 对象 | 每个 token 的 ratio | 整个序列的归一化 ratio |
| 长序列稳定性 | 差（方差爆炸） | 好（长度归一化） |

**优势**：
- 避免 token 级权重累乘的方差爆炸
- 对 MoE 模型更稳定
- 简化 RL 基础设施设计

### 方案六：多次采样估计（MoE 专用）

KAT-Coder 团队提出了一个不同视角：对于 MoE 模型，**采样噪声本身是导致训练不稳定的主导因素**，而非训推不一致。

**噪声来源分析**：

| 模型类型 | 训推差异 | 推理噪声方差 | 训练噪声方差 |
|----------|----------|--------------|--------------|
| Dense | ~0.002 | ~$10^{-5}$ | 0（Megatron 确定性） |
| MoE | ~0.008 | ~$10^{-3}$ | ~$10^{-7}$（scatter_add 随机性） |

MoE 的推理噪声方差比 Dense 高两个数量级，这才是不稳定的主因。

**核心方法**：计算 $\pi_{\text{old}}$ 时，直接用**推理引擎**重复计算 n 次（n=8），取平均值：

$$\hat{\pi}_{\text{old}}(y \mid x) = \frac{1}{n} \sum_{i=1}^{n} \pi_{\text{inference}}^{(i)}(y \mid x)$$

**关键优势**：
- 获得**无偏且方差缩小 n 倍**的估计
- **无需训练引擎 recompute**，直接用推理引擎
- 在异步框架下，多次采样时间可与 rollout 重叠
- KV cache 命中率接近 100%
- 端到端实际**减少 10-20% 训练时间**

**与其他方案对比**：

| 方案 | 问题 |
|------|------|
| Routing Replay | 大规模 agentic 场景难以保证 prefix cache 命中 |
| 截断 IS (TIS) | 对截断边界敏感，不解决估计偏差根因 |
| 确定性推理 | 需深度改造推理引擎，吞吐下降 40-70% |
| 多次采样估计 | 无超参，工程友好，效果最优 |

在 Qwen3-235B-A22B 上的实验表明，recompute 和 rollout_logprob 方法在 60-80 步后 reward 崩溃，而该方法保持稳定增长且优于 TIS。

### 方案七：工程调优

一些实用的工程手段：

| 方法 | 效果 | 适用场景 |
|------|------|----------|
| 降低 top-p | 减少低概率 token | 牺牲探索性 |
| 更换 GPU | H20 最稳定 | 硬件可选时 |
| 禁用 Cascade Attention | 显著降低 A100 上的 mismatch | A100 用户 |
| FP32 LM Head | 轻微改善 | 效果有限 |

## 监控指标

**vllm-kl** 是一个重要的早期预警指标：

$$\text{vllm-kl} = \mathbb{E}_{s, a \sim \pi_\text{vllm}} \left[ \log \frac{\pi_\text{vllm}(a|s)}{\pi(a|s)} \right]$$

建议同时监控：
- **vllm-kl**：mismatch 程度
- **fsdp-ppl**：训练引擎困惑度
- **梯度范数**：稳定性指标
- **熵**：策略分布健康度

当 vllm-kl 出现 spike 时，往往预示着即将发生的崩溃。

## 实践建议

1. **接受 mismatch 是不可避免的**：这是速度与一致性的根本权衡，不是临时 bug。

2. **使用 sequence-level 修正**：token-level IS 理论上有偏，在复杂任务上会失败。推荐 MIS 或 Geo-RS。

3. **监控 vllm-kl**：这是最直接的健康指标。

4. **验证硬件影响**：在目标硬件上测试，结果可能不可移植。

5. **对于 MoE 模型**：考虑 Routing Replay 来稳定 expert routing。

6. **权衡方案选择**：
   - 追求最高性能 → Bitwise Consistent Training（牺牲速度）
   - 追求实用平衡 → Sequence-Level MIS + 适当的 top-p

## 总结

LLM-RL 训练的稳定性问题，本质上是**现代系统架构分工**带来的副作用。推理引擎和训练框架各自为了效率做出的优化，在 RL 的闭环反馈中被放大成了系统性的不稳定。

理解这个问题的关键洞见是：
- **看似 on-policy 实则 off-policy**
- **token-level 修正不够，需要 sequence-level**
- **低概率 token 是最薄弱的环节**
- **高方差和 entropy collapse 需要专门处理**

随着 Reasoning RL 和 Agentic RL 的发展，这个问题只会越来越重要。希望这篇文章能帮助你在实践中少踩一些坑。

## 参考资料

1. [When Speed Kills Stability](https://richardli.xyz/rl-collapse)
2. [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374)
3. [Bitwise Consistent Train-Inference](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html)
4. [DeepSeek-V3.2 Technical Report](https://arxiv.org/abs/2512.02556)
5. [Ring-1T: Scaling RL to Trillion Parameters](https://arxiv.org/abs/2510.18855)
6. [GSPO: Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
7. [KAT-Coder: MoE 模型 RL 训练稳定性](https://kwaikat.github.io/kwaikat-blog/posts/katcoder_1201/)
