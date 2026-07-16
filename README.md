# AFAC2026 赛道四：金融长文本 Agent

本仓库用于实现 AFAC2026 赛道四的金融长文本问答系统。当前已打通五个领域的 Group A 100 题链路，并完成首版无 `doc_ids` 文档定位与答题迁移实验：支持文档清单构建、领域化解析与分块、BM25 检索、evidence gate、低置信 rescue、`answer.csv`/`evidence.json`/Token 留痕，以及按题 checkpoint 的可恢复运行。

## 当前进度

- 已完成共享骨架与五领域插件：`regulatory`、`financial_reports`、`insurance`、`research`、`financial_contracts`
- 已打通 Group A 100 题 baseline + 动态低置信 rescue 全链路
- 已支持 `.env` 读取模型配置
- 已支持运行结果、token 统计、证据链导出
- 已实现 B 榜迁移所需的无 `doc_ids` locator，并在 Group A 的 78 道严格盲测候选题上完成闭环
- 当前主要待处理项是保险/财报的证据支持与答案格式冲突，以及实际 Group B 数据接入

## 当前最佳版本快照

### Group A 当前参考版本

截至 `2026-06-30`，当前已记录的 Group A 最优参考版本为：

- 本地产物目录：`artifacts/submissions/group_a_candidate_accuracy_first_v20_20260630/`
- 用户外测记录准确率：`88%`
- total tokens：`211,749`
- 说明：v20 在保持答案不变的前提下，将部分高 Token 合同题改为字段级本地规则；仓库本地没有官方标签，不能自行重算真实准确率

### 无 doc_ids 迁移快照

截至 `2026-07-15`，`attempt_43 / canonical_score_inherit` 在 78 道严格盲测候选题上的最新结果为：

- v20 参考向量匹配：`65/78 = 83.33%`
- evidence supported：`60/78 = 76.92%`
- supported or weak：`61/78 = 78.21%`
- total tokens：`266,364`
- 相对历史 `attempt_31`：参考向量匹配 `+4`、supported `+8`、Token `-15,446`
- 运行产物：`artifacts/b_board_migration/no_docids_clean_subset_run/`
- 对比产物：`artifacts/b_board_migration/comparisons/attempt43_rank_preserved_vs_attempt31_v20_20260715/`

以上匹配率只是相对 v20 的回归代理，不是官方 B 榜准确率。当前工作区尚无官方 Group B 题目或标签。

### 历史快照

`artifacts/submissions/group_a_current_best_merged_20260629/` 是 2026-06-29 记录的 `71%` 历史版本，总 Token `1,305,879`，保留用于回溯，不再作为当前最佳版本。

后续优化重点是：修复 insurance 多选格式冲突和缺失证据，继续提升 financial_reports 的 selected evidence 完整性，并在真实 Group B 数据到位后验证 locator 泛化能力。当前研究路线与晋级规则见 `wiki/accuracy_token_research_plan.md`；实验结果追加记录在 `wiki/accuracy_token_experiment_log.md`。

## 目录结构

```text
.
├── docs/                       # 赛题说明与整理文档
├── public_dataset_upload/      # 原始比赛数据（本地，不提交）
├── scripts/                    # 各阶段运行脚本
├── src/afa_agent/              # 共享骨架与领域插件
├── artifacts/                  # 解析、索引、运行产物（本地，不提交）
├── PLAN.md                     # 当前实施方案
├── README_BASE.md              # 赛题基础说明稿
└── requirements.txt
```

## 环境准备

建议使用 Python 3.11+。

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

项目默认从 `.env` 读取模型配置，当前支持 OpenAI 风格兼容接口。需要至少包含：

```env
LLM_API_KEY=...
LLM_API_BASE=...
LLM_MODEL=...
```

如果要跑 `autoresearch + loop engine + mineru`，建议使用独立的 `conda` 环境：

```bash
conda create -n afa-autoresearch python=3.11 -y
conda activate afa-autoresearch
python -m pip install -U pip
python -m pip install -r requirements.txt
```

仓库中已导出环境文件：

```bash
conda env create -f environment.autoresearch.yml
```

## 运行方式

### 1. 构建文档清单

```bash
python3 scripts/build_manifest.py
```

输出：

- `artifacts/manifest/dataset_manifest.json`

### 2. 解析法规域文档

```bash
python3 scripts/parse_domain.py --domain regulatory
```

说明：

- 优先使用 `regulatory/txt`
- 缺失时回退到 `regulatory/html`
- 附件 PDF 使用 `pypdf` 做文本兜底抽取

输出：

- `artifacts/parsed/regulatory/parsed.json`

### 3. 构建法规域索引

```bash
python3 scripts/build_index.py --domain regulatory
```

输出：

- `artifacts/index/regulatory/index.json`

### 4. 运行法规域 A 榜

全量运行：

```bash
python3 scripts/run_answering.py --domain regulatory --split A
```

只跑部分题：

```bash
python3 scripts/run_answering.py --domain regulatory --split A --limit 2
```

只重跑单题：

```bash
python3 scripts/run_answering.py --domain regulatory --split A --qid reg_a_004
```

从已有 run 目录继续：

```bash
python3 scripts/run_answering.py --domain regulatory --split A --resume-run-dir artifacts/runs/<run_id>
```

### 5. 评测与导出

```bash
python3 scripts/evaluate.py --run-dir artifacts/runs/<run_id>
python3 scripts/export_submission.py --run-dir artifacts/runs/<run_id>
```

### 6. AutoResearch 单阶段实验

```bash
python3 scripts/run_autoresearch.py \
  --stage pdf_parse \
  --domains regulatory \
  --dataset-slice dev_mini \
  --baseline-config configs/autoresearch/default_strategy.json \
  --candidate-set configs/autoresearch/candidate_sets.json
```

说明：

- `configs/autoresearch/default_strategy.json` 定义当前基线策略
- `configs/autoresearch/candidate_sets.json` 定义每个阶段的候选变体
- `configs/autoresearch/dataset_slices.json` 定义 `dev_mini / dev_stage / full_group_a`
- 结果输出到 `artifacts/autoresearch/experiments/<experiment_id>/`

### 7. Loop Engine 串行迭代

```bash
python3 scripts/run_loop_engine.py --plan-config configs/autoresearch/plan_config.json
```

说明：

- 默认按 `pdf_parse -> segmentation -> retrieval -> rule_layer -> answering` 顺序推进
- 每阶段先在小题集上挑选最优候选，再在 `full_group_a` 上做验证
- 每轮都会生成 `experiment_manifest.json`、`candidate_config.json`、`aggregate_metrics.json`、`ranking.json`、`notes.md`

### 8. PDF 解析后端切换

默认 PDF 仍使用 `pypdf`。如果环境中已安装 `mineru` 或 `magic-pdf`，可以在策略配置中切换：

```json
{
  "domains": {
    "__all__": {
      "pdf_parse": {
        "pdf_backend": "mineru",
        "mineru_backend": "pipeline",
        "mineru_method": "auto",
        "mineru_lang": "ch"
      }
    }
  }
}
```

当前实现会优先尝试 `mineru`，失败时自动回退到 `pypdf`，不会阻断主流程。

注意：

- 基础版 `mineru` 默认可能无法直接跑本地高精度后端
- 如果要启用本地 `pipeline/hybrid`，通常还需要补装 `mineru[pipeline]` 或等效依赖
- 如果未来改走远端服务，也可以继续通过策略配置切换 `backend`

### 9. 使用 MinerU 预解析项目 PDF

如果希望把项目中的 PDF 先统一解析成持久化中间产物，再交给后续 `build_manifest / parse_domain` 使用，可以运行：

```bash
python3 scripts/parse_pdfs_with_mineru.py --domain research --limit 1
python3 scripts/parse_pdfs_with_mineru.py --domain research
```

常用参数：

```bash
python3 scripts/parse_pdfs_with_mineru.py \
  --domain research \
  --doc-id pack2_text01 \
  --backend pipeline \
  --method txt \
  --model-source modelscope \
  --timeout-seconds 1800 \
  --force
```

MinerU 产物目录约定：

- `artifacts/mineru/manifest.json`
- `artifacts/mineru/docs/<domain>/<doc_id>/raw_output/`
- `artifacts/mineru/docs/<domain>/<doc_id>/normalized/content.md`
- `artifacts/mineru/docs/<domain>/<doc_id>/normalized/content.txt`
- `artifacts/mineru/docs/<domain>/<doc_id>/logs/stdout.log`
- `artifacts/mineru/docs/<domain>/<doc_id>/logs/stderr.log`
- `artifacts/mineru/docs/<domain>/<doc_id>/meta.json`

说明：

- `raw_output/` 保存 MinerU 原始输出，便于排查表格、图片、json 和 markdown 结构
- `normalized/` 保存后续流程稳定消费的统一文本入口
- `meta.json` 记录命令、返回码、耗时、日志路径、选择了哪个主文本文件
- 运行 `scripts/build_manifest.py` 后，如果某个 PDF 已经有成功的 MinerU 结果，manifest 会优先引用 `normalized/content.md`

## 运行产物

每次运行都会生成一个 `artifacts/runs/<run_id>/` 目录，包含：

- `answer.csv`
- `answers.json`
- `evidence.json`
- `logs.jsonl`
- `metrics.json`
- `run_config.json`
- `token_usage.json`
- `wrong_cases.json`

当前已完成一版法规域 A 榜运行结果：

- [artifacts/runs/regulatory_a_20260622_200749/answer.csv](/Users/abandon/Documents/AFA_ww/artifacts/runs/regulatory_a_20260622_200749/answer.csv)
- [artifacts/runs/regulatory_a_20260622_200749/evidence.json](/Users/abandon/Documents/AFA_ww/artifacts/runs/regulatory_a_20260622_200749/evidence.json)

该次运行共完成 `20` 题，累计 `251616` tokens。

`autoresearch` 相关产物会额外写入：

- `artifacts/autoresearch/experiments/<experiment_id>/`
- `artifacts/autoresearch/leaderboard.json`
- `artifacts/autoresearch/loops/<loop_id>/`

## 当前实现要点

- 不使用向量检索
- A 榜标准链路使用题目给定 `doc_ids`；无 `doc_ids` 迁移链路先由 locator 生成候选文档
- 法规域采用 `规则限定 + BM25 + 相邻条款回溯`
- 按选项独立判别，再按题型约束生成最终答案
- 强制记录所有模型调用的 token
- 支持中断恢复，避免长跑结果丢失

## 下一步计划

- 优先审计 insurance 的多选格式冲突和缺失 selected evidence
- 继续提升 financial_reports 表格/指标 evidence 完整性
- 对高 Token case 收紧无收益的 rescue 和一致性复核
- 接入真实 Group B 题目后，验证 locator 泛化并单独处理文档顺序依赖题
