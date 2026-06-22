# AFAC2026 赛道四：金融长文本 Agent

本仓库用于实现 AFAC2026 赛道四的金融长文本问答系统。当前已经搭好首版工程骨架，并完成 `regulatory` 领域 A 榜的可运行基线：支持文档清单构建、法规解析、BM25 检索、逐选项判别、`answer.csv` 导出、`evidence.json` 留痕，以及按题 checkpoint 的可恢复运行。

## 当前进度

- 已完成共享骨架与法规域插件
- 已打通 `regulatory` A 榜全链路
- 已支持 `.env` 读取模型配置
- 已支持运行结果、token 统计、证据链导出
- 已预留多领域扩展接口，下一优先域为 `financial_reports`

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
