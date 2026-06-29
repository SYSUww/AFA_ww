# AFAC2026 赛道四：金融长文本 Agent

本仓库用于实现 AFAC2026 赛道四的金融长文本问答系统。当前已经搭好首版工程骨架，并完成 `regulatory` 领域 A 榜的可运行基线：支持文档清单构建、法规解析、BM25 检索、逐选项判别、`answer.csv` 导出、`evidence.json` 留痕，以及按题 checkpoint 的可恢复运行。

## 当前进度

- 已完成共享骨架与法规域插件
- 已打通 `regulatory` A 榜全链路
- 已支持 `.env` 读取模型配置
- 已支持运行结果、token 统计、证据链导出
- 已预留多领域扩展接口，下一优先域为 `financial_reports`

## 当前最佳版本快照

截至 `2026-06-29`，当前效果最好的 Group A 100 题合并产物为：

- 本地产物目录：`artifacts/submissions/group_a_current_best_merged_20260629/`
- 提交文件：`answer.csv`
- 合并方式：
  - 以 `artifacts/submissions/group_a_20260628_preprocessed_loop_full/answer.csv` 的 100 题顺序为准
  - 用 `artifacts/submissions/risk53_current_best_20260629/answer.csv` 覆盖其中 53 个风险题
  - 其余 47 题保留原答案
- 当前人工/外部评测反馈准确率：`71%`
- token 消耗：
  - prompt tokens: `1,053,990`
  - completion tokens: `251,889`
  - total tokens: `1,305,879`

这版通过 evidence gate / rescue / answer finalization / regulatory supplemental rescue 提升了准确率，但 token 消耗明显增加，导致综合得分下降。后续优化重点应放在降低高置信 case 的重复 LLM 调用、压缩 insurance 和 financial_contracts 的多选复核 prompt，以及只对低确定性 case 触发二次检索和二次回答。

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
- A 榜直接使用题目给定 `doc_ids`
- 法规域采用 `规则限定 + BM25 + 相邻条款回溯`
- 按选项独立判别，再按题型约束生成最终答案
- 强制记录所有模型调用的 token
- 支持中断恢复，避免长跑结果丢失

## 下一步计划

- 提升法规标题与条款结构抽取质量
- 增强法规域规则召回，降低高 token 题成本
- 接入 `financial_reports` 领域插件
- 为后续 B 榜预留候选文档召回模块
