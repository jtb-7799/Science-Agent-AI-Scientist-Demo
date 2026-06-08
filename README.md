# AI for Science Agents Demo

本仓库包含 5 个用于企业 demo 展示的 AI for Science Agent 项目，覆盖自动科研、文献驱动研究想法生成、知识图谱推理、实验树搜索和生物医学发现等方向。

## 项目结构

```text
work_code/
├── AI-Scientist/        # 自动生成科研 idea、实验、论文和审稿
├── AI-scientist-v2/     # 基于 Best-First Tree Search 的自动实验探索
├── ResearchAgent/       # 基于文献和知识库的研究问题/方法/实验设计生成
├── SCIAGENTS/           # 基于知识图谱的多智能体科学假设生成
└── Robin/               # 面向生物医学/药物发现的多智能体流程
```

## 5 个 Demo 简介

### AI-Scientist

模板驱动的端到端 AI Scientist 系统，流程包括 idea generation、novelty check、代码实验、绘图、论文写作和 LLM 审稿。

入口示例：

```bash
cd AI-Scientist
python launch_scientist.py --model "tb-gpt-4o" --experiment nanoGPT_lite --num-ideas 2
```

### AI-scientist-v2

去模板化的科研自动化系统，核心是 agentic tree search。系统会自动生成实验代码、执行实验、评估结果，并进入 baseline、调参、创新实验、消融实验等阶段。

入口示例：

```bash
cd AI-scientist-v2
python launch_scientist_bfts.py --model "tb-gpt-4o" --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json
```

### ResearchAgent

从 Semantic Scholar paper IDs 和本地 knowledge store 出发，迭代生成并验证 research problem、method proposal 和 experiment design。

入口示例：

```bash
cd ResearchAgent
python ./code/main.py --data-path ./data/papers.jsonl --knowledge-path ./data/knowledge.jsonl --model-name tb-gpt-4o
```

### SCIAGENTS

基于知识图谱和多智能体协作的科学发现 demo。系统从知识图谱中采样概念路径，并由 Planner、Ontologist、Scientist、Critic 等角色协作生成科研假设。

核心目录：

```text
SCIAGENTS/SciAgentsDiscovery/
```

### Robin

面向生物医学和药物发现的多智能体系统，包含治疗候选物生成、实验 assay 选择、文献评估、候选排序和 Finch notebook 数据分析 agent。

核心目录：

```text
Robin/robin/
Robin/finch/
```

## API 配置

请使用环境变量配置 API，不要把真实 key 写入代码或提交到 GitHub。

```bash
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=****
AZURE_OPENAI_DEPLOYMENT=****
S2_API_KEY=your-semantic-scholar-api-key
```

Python 示例：

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "****"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
```

Semantic Scholar API 建议按 `1 request / second` 做限速。
