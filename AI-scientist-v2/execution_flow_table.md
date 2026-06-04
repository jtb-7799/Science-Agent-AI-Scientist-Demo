# AI Scientist-v2 代码执行流程表

## 完整执行流程

| 步骤 | 标题 | 扮演角色 | 核心文件 | 输入 | 核心逻辑 | 输出 |
|------|------|----------|----------|------|----------|------|
| 1 | 创意生成 | 科研大脑——决定研究什么 | `perform_ideation_temp_free.py` + `semantic_scholar.py` | `.md` 主题描述文件 | LLM 在 ReAct 循环中交替「思考 → 搜索文献 → 反思」，最多 20 轮 × 5 次反思，最终输出结构化 JSON | 研究创意 `.json`（Name / Title / Hypothesis / Experiments / Risk Factors） |
| 2 | 总调度 | 项目经理——串联全流程 | `launch_scientist_bfts.py` | 创意 `.json` + `bfts_config.yaml` | 解析命令行参数 → JSON 转 Markdown → 编辑配置文件 → 依次调用实验/图表/撰写/评审模块 | `experiments/日期_创意/` 目录初始化 |
| 3 | BFTS 树搜索 | 实验导演——管理 4 阶段探索 | `perform_experiments_bfts_with_agentmanager.py` + `agent_manager.py` | 研究创意 + 配置 | AgentManager 主循环：Stage 1(基线) → 2(调参) → 3(创新) → 4(消融)，每阶段 LLM 评估完成度后自动切换，最佳节点作为下一阶段种子 | 4 个 Journal（每个阶段一棵实验树） |
| 4 | 单步搜索 | 实验引擎——心跳式探索 | `parallel_agent.py :: step()` | 当前树状态 + 阶段上下文 | (1) 选节点：50%概率修bug / 50%概率Best-First改进 (2) 对 N 个选中节点并行：草稿/调试/改进 (3) 子进程执行代码 (4) LLM+VLM评估 (5) 结果入树 | 树扩展 N 个新节点 |
| 5 | LLM 代码生成 | 实验员——写代码 | `parallel_agent.py :: _draft / _debug / _improve` + `backend/__init__.py` | 节点状态 + 历史上下文 + 研究创意 | 构造结构化 Prompt（Introduction + Research idea + Previous code + Error info + Instructions）→ backend.query() → extract_code() 从 LLM 回复中提取 Python 代码 | 一段可执行的 Python 实验脚本 |
| 6 | 代码沙箱执行 | 安全沙箱——隔离运行 | `interpreter.py` | LLM 生成的 Python 代码字符串 | 创建隔离子进程 → 3个Queue通信（code_inq / result_outq / event_outq）→ exec(compile(code)) → 超时 SIGINT → SIGKILL 兜底 | ExecutionResult（stdout / 耗时 / 异常类型 / 堆栈） |
| 7 | 节点评估 | 质检员——判断好坏 | `parallel_agent.py :: _evaluate_node` | 代码 + 运行输出 + 图表 | (1) LLM反馈：判断 is_bug? → 标记 buggy/good (2) VLM分析：图表质量/有效性 (3) 指标提取：从日志解析数值 metric | 带评估标签的 Node（metric / is_buggy / vlm_feedback） |
| 8 | 多种子评估 | 统计员——确保可复现 | `agent_manager.py :: _run_multi_seed_evaluation` | Stage 最佳节点 | 用不同随机种子重复运行最佳实验 N 次 → 计算均值和标准差 | 带误差线的最终指标 |
| 9 | 图表汇总 | 可视化设计师 | `perform_plotting.py` | 所有 Stage 的实验摘要 + .npy 数据文件 | LLM 读取所有摘要 → 生成一个 Python 聚合脚本 → 执行脚本 → 合并多 Stage 图表为最终论文图 | `aggregated_plots/` 中的最终论文图表 |
| 10 | 引用收集 | 文献管理员 | `perform_icbinb_writeup.py :: gather_citations` | 论文草稿 | 循环 20 轮：LLM 识别需引用位置 → Semantic Scholar 搜索 → LLM 筛选 → 更新 .bib | `references.bib` 文件 |
| 11 | 论文撰写 | 论文写手 | `perform_icbinb_writeup.py` | 创意 + 实验摘要 JSON + 图表 + 引用 | LLM(small) 写初稿 → pdflatex 编译 → chktex 检查 → VLM 审图 → LLM(big) 反思修改 → 最多重试 3 次 | LaTeX 源码 + 最终 PDF |
| 12 | 论文评审 | 模拟审稿人 | `perform_llm_review.py` + `perform_vlm_review.py` | 最终 PDF | LLM 评审：科学贡献/实验合理性/结果可信度/写作质量。VLM 评审：图表标题一致性/清晰度/重复检测 | `review_text.txt` + `review_img_cap_ref.json` |

## 4 个实验阶段详解

| 阶段 | 名称 | 目标 | 节点类型 | 停止条件 | 知识传递 |
|------|------|------|----------|----------|----------|
| Stage 1 | 初步实现 | 跑通基线代码 | draft（从零生成） + debug（修复bug） + improve（改进） | 基础代码成功运行 + 达到 max_iters | 无（冷启动） |
| Stage 2 | 超参调优 | 优化关键参数，建立鲁棒基线 | hyperparameter_tuning（调整 lr / epochs / batch_size） + 引入 2+ 新 HuggingFace 数据集 | 训练曲线收敛 + 多数据集成功运行 | Stage 1 最佳节点 → Stage 2 种子代码 |
| Stage 3 | 创意研究 | 自由探索新改进方向 | improve（改进） + debug（修复） + 使用 3 个 HF 数据集 | 计算预算耗尽；实验太快 → 建议增加复杂度 | Stage 2 最佳节点 → Stage 3 基线 |
| Stage 4 | 消融实验 | 系统性分析各组件贡献 | ablation（移除/替换组件） + 多数据集 | 计算预算耗尽 | Stage 3 最佳节点 → Stage 4 分析对象 |

## Node 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str (uuid) | 节点唯一标识 |
| `parent` | Node | 父节点引用 |
| `children` | set[Node] | 子节点集合 |
| `plan` | str | LLM 生成的实验计划（自然语言） |
| `code` | str | 实验 Python 代码 |
| `plot_code` | str | 绘图 Python 代码 |
| `step` | int | 所属的 step 编号 |
| `metric` | MetricValue | 实验指标值（用于 Best-First 选优） |
| `is_buggy` | bool | 代码是否有 bug |
| `debug_depth` | int | 已尝试修复的次数（超过 max_debug_depth 则放弃） |
| `term_out` | list[str] | 代码运行输出（stdout） |
| `exec_time` | float | 运行耗时 |
| `exc_type` | str | 异常类型（None = 正常运行） |
| `analysis` | str | LLM 对结果的分析 |
| `vlm_feedback_summary` | str | VLM 对图表的反馈摘要 |
| `plot_analyses` | list | VLM 对每张图的详细分析 |
| `plots` | list[str] | 生成的图表文件路径 |

## LLM 三类调用

| 调用类型 | 模型配置 | 用途 | 输入 | 输出 |
|----------|----------|------|------|------|
| 代码生成 (`code`) | `agent.code.model` | 写实验/绘图/解析脚本 | 结构化 Prompt（含历史代码 + 反馈） | Python 代码 + 自然语言计划 |
| 文本反馈 (`feedback`) | `agent.feedback.model` | 评估代码是否有 bug + 指标提取 | 代码 + stdout + 异常信息 | `{is_bug, summary}` 或 `{metric_names, data}` |
| 视觉反馈 (`vlm_feedback`) | `agent.vlm_feedback.model` | 分析生成图表的质量 | 图表 PNG 图片 | `{plot_analyses, valid_plots_received, summary}` |

## 关键配置参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `num_workers` | `bfts_config.yaml :: agent` | 4 | 每步并行处理的节点数 |
| `steps` | `bfts_config.yaml :: agent` | 5 | 每个子阶段的最大迭代步数 |
| `num_drafts` | `bfts_config.yaml :: agent.search` | 3 | Stage 1 初始并行根节点数 |
| `debug_prob` | `bfts_config.yaml :: agent.search` | 0.5 | 选 buggy 节点修复的概率 |
| `max_debug_depth` | `bfts_config.yaml :: agent.search` | 3 | 单个节点最大修复尝试次数 |
| `num_seeds` | `bfts_config.yaml :: agent.multi_seed_eval` | 3 | 多种子评估的重复次数 |
| `timeout` | `bfts_config.yaml :: exec` | 3600 | 单次代码执行的超时秒数 |
| `stage1_max_iters` | `bfts_config.yaml :: agent.stages` | 20 | Stage 1 最大迭代数 |
| `stage2_max_iters` | `bfts_config.yaml :: agent.stages` | 12 | Stage 2 最大迭代数 |
| `stage3_max_iters` | `bfts_config.yaml :: agent.stages` | 12 | Stage 3 最大迭代数 |
| `stage4_max_iters` | `bfts_config.yaml :: agent.stages` | 18 | Stage 4 最大迭代数 |
