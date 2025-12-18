# LLMNotes

大语言模型学习笔记与演示文稿，涵盖 Transformer 架构、强化学习等核心主题。

## 内容概览

### Transformer / LLM 笔记

系统性覆盖大语言模型从基础理论到前沿应用的完整知识体系：

| Part | 主题 | 内容 |
|------|------|------|
| I | 基础理论 | Roofline 模型、Transformer 计算分析、Scaling Law |
| II | 核心组件 | Tokenizer (BPE)、位置编码 (RoPE)、门控机制 (SwiGLU) |
| III | 注意力机制 | FlashAttention、MLA、稀疏注意力、线性注意力 |
| IV | 模型架构 | MoE 专家混合 |
| V | 训练技术 | 数据工程、分布式训练 (DP/TP/PP/ZeRO)、Muon 优化器 |
| VI | 评测体系 | Benchmark 设计与分析 |
| VII | 部署优化 | 量化 (PTQ/QAT)、推理优化 (KV Cache、投机解码) |
| VIII | 前沿应用 | 多模态、推理能力 (CoT/MCTS) |

### 强化学习笔记

从基础到前沿的 RL 完整学习路径：

| 章节 | 主题 | 内容 |
|------|------|------|
| 1 | 绪论 | MDP、Markov 性质、价值函数、优势函数 |
| 2 | 基于价值的 RL | Bellman 方程、DP、MC/TD、Q-Learning、DQN |
| 3 | 基于策略的 RL | Policy Gradient、REINFORCE、GAE、PPO |
| 4 | 模型与多智能体 | MBRL、MCTS、AlphaGo/Zero、Nash 均衡 |
| 5 | LLM 与 RL | RLHF、DPO 推导、GRPO、PRM、Long CoT RL |
| 6 | 总结与展望 | 算法分类、演进脉络、未来方向 |

### 演示文稿

- **RL Slides**: 强化学习核心概念速览
- **Transformer Slides**: Transformer 架构升级之路

## 项目结构

```
LLMNotes/
├── notes/
│   ├── Transformers/          # LLM 完整笔记
│   │   ├── main.tex
│   │   ├── sections/          # 各章节
│   │   ├── slides/            # 配套 slides
│   │   └── build.sh
│   └── RL/                    # 强化学习笔记
│       ├── main.tex
│       ├── preamble.tex
│       ├── chapters/          # 各章节
│       └── build.sh
├── slides/
│   └── RL/                    # RL slides
│       ├── main.tex
│       └── build.sh
└── README.md
```

## 快速开始

### 依赖

- **必需**: XeLaTeX (TeX Live 2024+ 或 MacTeX)
- **可选**: Python 3 + Pygments (代码高亮)

### 编译

```bash
# 编译 Transformer 笔记
cd notes/Transformers && ./build.sh

# 编译 RL 笔记
cd notes/RL && ./build.sh

# 编译 Slides
cd slides/RL && ./build.sh
cd notes/Transformers/slides && ./build.sh
```

编译产物：
- `notes/Transformers/Transformers_Notes.pdf`
- `notes/RL/RL_Notes.pdf`
- `slides/RL/RL_Slides.pdf`
- `notes/Transformers/slides/Transformer_Slides.pdf`

## 写作风格

- **深入浅出**: 入门读者能看懂结论，进阶读者能跟完推导
- **问题驱动**: 先说清楚"要解决什么问题"，再展开技术细节
- **推导完整**: 展示完整步骤，不跳步，关键变换加文字说明
- **知识串联**: 概念之间建立联系，形成知识网络

## License

MIT
