# 执行动作日志

本文件按时间追加记录关键动作、命令、结果和异常。日志不得记录 API Key、Token 或其他密钥值；大体积运行产物保存在 `artifacts/`，这里只记录可复核的路径与摘要。

## 2026-07-15 15:18 +08:00｜计划与基线固化

- 执行者：Codex
- 分支：`codex/b-board-optimization-iteration`
- 起始 HEAD：`c4b7521dd326842bab649147d15e8005ed592248`
- 用户目标：先将执行计划提交到 Git，再直接完成后续运行，并持续记录动作。
- 状态检查：tracked/staged 文件无改动；仅存在未跟踪的 `.DS_Store`。
- 环境检查：`.env` 存在，`LLM_API_KEY`、`LLM_API_BASE`、`LLM_MODEL` 均为非空；未输出任何配置值。
- 数据检查：严格无 doc_ids 实验所需的 regulatory、financial_reports、insurance、research parsed/index 路径均存在。
- 决策：先提交计划、日志和 `.DS_Store` 忽略规则；模型调用、历史快照和对比分析在该提交之后执行。
- 计划文档：`wiki/current_execution_plan.md`

## 2026-07-15 15:20 +08:00｜计划提交完成

- 提交：`2b276e4 docs: record B-board execution plan`
- 提交范围：`.gitignore`、`wiki/current_execution_plan.md`、`wiki/execution_log.md`
- 检查：`git diff --cached --check` 通过；staged diff 未发现疑似 API Key 或 Bearer Token。
- 结论：满足“计划先提交、实验后执行”的顺序要求。

## 2026-07-15 15:22 +08:00｜历史快照与模型预检

- 历史快照：将 `artifacts/b_board_migration/no_docids_clean_subset_run` 移至 `artifacts/b_board_migration/no_docids_clean_subset_run_attempt31_20260715`，保留 attempt_31 的答案、证据、Token 和审计结果。
- 模型预检：通过 OpenAI-compatible `/chat/completions` 发起最小 JSON 请求，模型为 `qwen3.7-plus`。
- 预检结果：成功；prompt/completion/total tokens 为 `30/134/164`。
- 安全说明：日志未记录 API Base 或 API Key 的值。
- 下一动作：启动 attempt_43 的 78 题无 doc_ids 答题；并行 worker 数为 5。

## 2026-07-15 16:08 +08:00｜attempt_43 首轮闭环完成

- 命令：`PYTHONPATH=src /opt/miniconda3/envs/afa-autoresearch/bin/python scripts/run_b_board_migration_loop.py --answer-only --answer-workers 5`
- 退出状态：成功，78/78 题完成，`failed_count=0`，无 `failed_answers.jsonl`。
- 运行身份：`attempt_43 / canonical_score_inherit`。
- 代理一致率：相对 v20 参考向量为 `62/78 = 0.794872`；该值不是官方 B 榜准确率。
- Token：`375025`。
- evidence audit：supported `56`、weak `3`、unsupported `12`、contradicted `1`、format conflict `6`。
- 产物：`artifacts/b_board_migration/no_docids_clean_subset_run/`。

## 2026-07-15 16:10 +08:00｜attempt_31 / attempt_43 / v20 三路对比

- 新增工具：`scripts/compare_b_board_answer_runs.py`。
- 完整性检查：两轮各 78 题、0 失败；QID 集、领域分布、答案合法性、Token 行/summary/manifest、evidence audit 汇总均一致。
- 对比产物：`artifacts/b_board_migration/comparisons/attempt43_vs_attempt31_v20_20260715/`。
- 结果：代理匹配 `61 -> 62`，supported `52 -> 56`，但 supported-or-weak `64 -> 59`，Token `281810 -> 375025`（增加 `93215`）。
- 领域信号：财报代理匹配 `9 -> 13`；研报代理匹配 `18 -> 16`，研报 Token `11504 -> 106216`。
- 注意：v20 的可比 78 题 Token 是 `163403`，不能把 v20 全量 100 题的 `211749` 直接与 78 题运行比较。

## 2026-07-15 16:14 +08:00｜定位 alias_pruned 结构性问题

- 现象：`res_a_017`、`res_a_020` 的 locator 排名已把真实相关文档排在前两位，但答题前 `alias_pruned` 只保留标题命中通用“深度报告”别名的其他文档。
- 影响：两题答案由 v20 一致的 `ABD` 变为 `AC`，证据由 supported 变为 weak，合计额外消耗 `94870` Token。
- 同类问题：regulatory 中通用“定期报告/规定”别名也会覆盖原始 locator 排名；实际共有 19 道 regulatory 和 2 道 research 的答题文档选择会受修复影响。
- 修复：`alias_pruned` 只在 financial_reports/insurance 保留 alias shortlist；regulatory/research 直接沿用 locator 排名的 topK。
- 测试：新增 `tests/test_b_board_doc_selection.py`，覆盖 research、regulatory、insurance 及 expanded_topk；4 项测试通过。
- 下一动作：提交修复与对比工具，备份首轮 attempt_43，再只重跑上述 21 道受影响题。

## 2026-07-15 16:16 +08:00｜修复提交

- 提交：`9c961f2 fix: preserve locator ranking for blind answers`
- 提交内容：alias shortlist 修复、三路对比工具、4 项单元测试、阶段日志。
- 验证：`unittest` 4/4 通过；相关 Python 文件 `py_compile` 通过；staged secret scan 通过。

## 2026-07-15 16:27 +08:00｜21 题定向恢复完成

- 首轮快照：`artifacts/b_board_migration/no_docids_clean_subset_run_attempt43_raw_20260715/`。
- 恢复方式：从首轮 checkpoint 中移除 19 道 regulatory 和 2 道 research 受影响题，保留其余 57 题，再执行同一 `--answer-only --answer-workers 5` 命令。
- 完成状态：78/78，`failed_count=0`。
- 最终代理一致率：相对 v20 为 `65/78 = 0.833333`；不是官方 B 榜准确率。
- 最终 Token：`266364`；本次定向重跑的 21 题实际消耗 `65705` Token。
- 本轮全部模型调用消耗：预检 `164` + 首轮 `375025` + 定向重跑 `65705` = `440894` Token。
- 最终 evidence audit：supported `60`、weak `1`、unsupported `11`、contradicted `1`、format conflict `5`。
- 按领域：财报 supported `14/20`，保险 `12/20`，法规 `18/20`，研报 `16/18`。

## 2026-07-15 16:29 +08:00｜最终对比与停止条件

- 相对 attempt_31：代理匹配 `61 -> 65`，supported `52 -> 60`，Token `281810 -> 266364`。
- 相对首轮 attempt_43：代理匹配 `62 -> 65`，supported `56 -> 60`，Token `375025 -> 266364`。
- 最终对比：`artifacts/b_board_migration/comparisons/attempt43_rank_preserved_vs_attempt31_v20_20260715/`。
- 修复前后对比：`artifacts/b_board_migration/comparisons/attempt43_rank_preserved_vs_attempt43_raw_20260715/`。
- 剩余代理回归：`fin_a_003`、`ins_a_016`。其中 `ins_a_016` 已定位到完整的四个真实文档，但模型仍输出单项答案；继续按 v20 答案打补丁会形成 A 榜过拟合，本轮不做硬编码。
- 下一优先级：基于 evidence audit 处理 5 个格式冲突和 11 个 unsupported case，先做保险多选一致性与财报 selected gate，不再围绕参考答案做逐题规则堆叠。
