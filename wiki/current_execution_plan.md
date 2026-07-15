# 当前执行计划：B 榜无 doc_ids 闭环

更新时间：2026-07-15 15:18（Asia/Shanghai）

## 目标

在不使用真实 `doc_ids` 参与检索和作答的前提下，用当前最佳 locator `attempt_43 / canonical_score_inherit` 完成 78 道严格盲测候选题的答题闭环，并与历史 `attempt_31` 和 A 榜 v20 参考答案向量进行可复现对比。

本轮不把代理一致率表述成官方准确率。项目中尚无官方 B 榜题目或标签，所有结果都必须标明数据范围和参考口径。

## 当前基线

- 当前分支：`codex/b-board-optimization-iteration`
- 计划开始时 HEAD：`c4b7521dd326842bab649147d15e8005ed592248`
- 严格无 doc_ids 候选题：78 题；特殊诊断题：22 题
- 最佳 locator：`attempt_43 / canonical_score_inherit`
- 最佳 locator 指标：`doc_recall@5=0.897436`、`doc_recall@10=0.910256`
- 历史无 doc_ids 答题：`attempt_31`，代理一致率 `0.782051`，总 Token `281810`
- A 榜当前参考向量：v20，用户外测记录 `88%`，总 Token `211749`

## 执行顺序

### 阶段 0：固化计划与基线

1. 将本计划与动作日志纳入 Git。
2. 将 `.DS_Store` 加入忽略规则，避免混入提交。
3. 显式检查 staged 文件、敏感信息和空白错误后提交。

验收：计划提交必须早于模型调用和新一轮实验。

### 阶段 1：运行前预检与历史快照

1. 只验证 `.env` 是否包含非空的 `LLM_API_KEY`、`LLM_API_BASE`、`LLM_MODEL`，不输出配置值。
2. 验证四个严格盲测领域的 parsed/index 产物存在。
3. 将现有 `no_docids_clean_subset_run` 备份为带 `attempt_31` 标识的目录。
4. 做最小 API 预检；失败时先定位配置、网络或限流问题，不直接启动全量任务。

### 阶段 2：执行最佳策略答题

运行：

```bash
PYTHONPATH=src /opt/miniconda3/envs/afa-autoresearch/bin/python \
  scripts/run_b_board_migration_loop.py \
  --answer-only \
  --answer-workers 5
```

要求：

- 从 `artifacts/b_board_migration/best_strategy_config.json` 读取 `attempt_43`。
- 真实 `doc_ids` 只允许用于运行后的评估字段，不得参与 locator、retrieval 或 answer。
- 使用逐题 checkpoint；中断后优先恢复，不重复消耗已完成题目的 Token。

### 阶段 3：验证与对比

必须检查：

1. `run_manifest.json` 的 attempt、题数、失败题和 Token。
2. `answer.csv` 是否为 78 题加 1 行 summary，答案是否合法且非空。
3. evidence audit 的 supported、weak、unsupported、contradicted、format conflict。
4. `attempt_43` 相对 `attempt_31` 的答案变化、代理一致率变化、证据支持变化和 Token 变化。
5. 相对 v20 参考向量的差异只作为回归代理，不作为官方分数。

对比结果写入 `artifacts/b_board_migration/comparisons/`，摘要写入受 Git 管理的动作日志。

### 阶段 4：结果驱动的定向处理

根据阶段 3 的真实结果，只处理证据明确的主要问题：

1. locator 已命中文档但 evidence 失败：优先改 evidence retrieval、unit type 或邻居扩展。
2. evidence 足够但答案不支持：优先改题型 prompt、final consistency 或答案格式。
3. 高 Token 且答案/证据无改善：收紧 rescue、TopK 或重复调用。
4. 如果问题需要大规模重构或再次全量付费运行，先在受影响小题集验证。

本轮至少完成一份逐题差异清单和按领域聚合结论；是否修改求解器，以结果是否给出清晰根因为准，禁止为“有改动”而盲目改代码。

### 阶段 5：收尾

1. 更新动作日志、README 当前状态和必要的复现说明。
2. 运行 Python/JSON/CSV 低成本验证。
3. 显式提交本轮代码、文档和日志；`artifacts/` 继续作为本地产物，不纳入 Git。

## 完成标准

- 计划先提交到 Git。
- `attempt_43` 78 题运行完成，或失败原因、已完成题数和可恢复方式有可靠记录。
- 历史 `attempt_31` 产物未丢失。
- 有逐题和领域级对比结果。
- 所有动作、命令、结果和异常均记录在 `wiki/execution_log.md`。
- 最终工作区没有意外文件或敏感信息进入提交。

## 执行结果（2026-07-15）

- 计划提交：`2b276e4 docs: record B-board execution plan`
- 对比与修复提交：`9c961f2 fix: preserve locator ranking for blind answers`
- attempt_43 首轮：78/78，0 失败，v20 代理匹配 `62/78`，Token `375025`
- 定向修复：避免 regulatory/research 的通用 alias shortlist 覆盖 locator 原始排名
- 定向重跑：仅重跑受影响的 21 题，其余 57 题从 checkpoint 复用
- 最终结果：78/78，0 失败，v20 代理匹配 `65/78`，supported `60/78`，Token `266364`
- 相对 attempt_31：代理匹配 `+4`、supported `+8`、Token `-15446`
- 结论：本轮目标已完成；结果仍是 Group A mask 实验，不是官方 B 榜成绩
