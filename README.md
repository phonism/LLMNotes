# LLMNotes

LLM 与强化学习学习笔记。

**Blog**: [https://phonism.github.io/LLMNotes/](https://phonism.github.io/LLMNotes/)

## 内容

### Transformer / LLM

| Part | 主题 | 内容 |
|:----:|------|------|
| I | 基础理论 | Roofline、计算分析、Scaling Law |
| II | 核心组件 | Tokenizer、RoPE、SwiGLU |
| III | 注意力机制 | FlashAttention、MLA、稀疏/线性注意力 |
| IV | MoE | 专家混合架构 |
| V | 训练 | 数据工程、分布式训练、Muon |
| VI | 评测 | Benchmark 设计 |
| VII | 部署 | 量化、KV Cache、投机解码 |
| VIII | 前沿 | 多模态、CoT、MCTS |

### 强化学习

| 章节 | 主题 | 内容 |
|:----:|------|------|
| 1 | 基础知识 | MDP、价值函数、优势函数 |
| 2 | Value-Based | Bellman、TD/MC、DQN |
| 3 | Policy-Based | Policy Gradient、GAE、PPO |
| 4 | Model-Based & MARL | MCTS、AlphaZero、Nash 均衡 |
| 5-6 | LLM 对齐 | RLHF、DPO、GRPO、PRM |

## 构建

```bash
# 依赖: XeLaTeX (TeX Live 2024+ / MacTeX)

cd notes/Transformers && ./build.sh   # -> Transformers_Notes.pdf
cd notes/RL && ./build.sh             # -> RL_Notes.pdf
```

## License

MIT
