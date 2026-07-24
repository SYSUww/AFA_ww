# 通用检索查询生成器设计与首轮实现（2026-07-24）

## 1. 从100题审计得到的通用规律

两份逐题审计不能直接转成 `QID -> 关键词` 配置，但能抽象出稳定的语义结构：

```text
文档/实体锚点
+ 指标、条款或事件主题
+ 关系/动作/运算符
+ 年份、期间、期限和数值
+ 范围、极性和例外
```

选择题不能只搜索选项原句，至少要区分：

1. **support**：保留选项结论值，召回直接支持或直接否定命题的原文；
2. **broad**：去掉候选结论数值，查材料实际披露的指标、条款和适用范围；
3. **contrast**：按通用范围、极性或方向词寻找反证和例外；
4. **scope_check**：遇到“未提及、不存在、均未”等存在性否定时，要求限定文档/章节的完整覆盖。

计算题额外生成 **coverage** 查询：

```text
每个动态抽取的实体
× 题面期间
× 原始指标/条款
+ 公式、单位、表格行
```

它的目的不是提前写入答案，而是把“实体×期间×指标”所需原始输入尽量在首轮覆盖，减少模型报缺值后的无效扩检索。

## 2. Module、Interface 与 Seam

实现位于：

- `src/afa_agent/retrieval_query.py`
- `src/afa_agent/strategy.py`

这是一个纯进程内、无I/O的深 module。外部 seam 只有一个核心 interface：

```python
generate_retrieval_plan(
    RetrievalRequest(...),
    max_queries=12,
) -> RetrievalPlan
```

`RetrievalRequest` 有且只有：

```text
domain
question
option_text
question_type
answer_format
document_hints
```

明确没有：

```text
qid
label / expected_answer / pred_answer
answer_parts
doc_id / evidence_id
```

生成器实现也没有任何固定题号、固定公司、固定年份、固定答案或 corpus unit 映射。公司、产品、法规、年份和数值只能从当前题面/选项动态抽取；领域词典只包含跨题复用的指标、条款、关系和范围概念。

## 3. 返回结构

`RetrievalPlan` 包含：

- `version=semantic_slots_v1`
- `SemanticSlots`
  - `anchors`
  - `topics`
  - `periods`
  - `quantities`
  - `relations`
  - `scopes`
  - `exceptions`
  - `articles`
  - `atoms`
- 带来源说明的 `QueryVariant`
  - `primary`
  - `support`
  - `broad`
  - `contrast`
  - `scope_check`
  - `coverage`
- `requires_scope_check`

查询计划可完整序列化进后续实验 artifact，便于审计“哪些词来自题面、哪个通道发起了检索”，不再只保存一串无法归因的 query 字符串。

## 4. 与当前生产链的关系

`strategy.py` 新增显式配置：

```json
{
  "retrieval": {
    "query_generator": "semantic_slots_v1",
    "semantic_query_max_variants": 12
  }
}
```

默认值仍为：

```json
{
  "query_generator": "legacy"
}
```

因此本次实现不会静默改变当前生产答案。只有新的隔离实验显式设置 `semantic_slots_v1` 才使用新生成器；已有不可变运行和A13 artifact均不受影响。

计算题 runner 目前尚未切换到该计划。生成器已经能够产生计算 `coverage` 查询，但正式接入前必须先完成只读 Recall/token 回放，不能直接覆盖现有 `_calculation_semantic_query_terms` 和错误分流。

## 5. 首轮验证

### 5.1 单元与回归测试

- 新增12项生成器测试：
  - interface 无QID/答案字段；
  - QID变化不改变查询；
  - 财报支持/宽查/反证；
  - 保险公式运算符；
  - 监管期限/模态/动作；
  - 否定存在性 scope check；
  - 计算题多实体 coverage；
  - legacy 默认行为不变；
  - 未知生成器配置显式拒绝，不静默回退。
- 生成器、相关配置、检索完整性和B榜 runner 测试共149项通过。

### 5.2 全部100题无模型回放

读取题面、按选择题选项和计算题分别生成计划：

| 指标 | 结果 |
|---|---:|
| 题目数 | 100 |
| 查询计划数 | 320 |
| 查询中出现QID | 0 |
| 计算题生成coverage | 26/26 |
| 每计划平均查询数 | 3.712 |
| 单计划最大查询数 | 12 |

通道数量：

```text
primary     320
support     411
broad       320
contrast     97
coverage     38
scope_check   2
```

这些数字只证明生成器能稳定运行且不泄漏QID，不证明检索召回已经优于当前策略。

## 6. 当前局限

1. 实体抽取仍是确定性文本规则。公司/法规标题较稳定，行业研究中隐含主体或没有显式实体的题仍主要依赖主题词。
2. `contrast` 使用通用极性替换，能提供反证方向，但不保证每个改写都符合自然语言最佳表达。
3. `scope_check` 只声明需要完整覆盖；真正证明“文档没有该条款”仍需要全文/章节覆盖门禁，不能靠Top-K未命中。
4. coverage 查询只解决召回，不解决运算符、年份绑定、单位和公式方向。`ins_b_003`、`fin_b_018`、`res_b_012`仍必须由本地语义校验处理。
5. 当前只完成A13证据重叠代理回放，没有消耗Qwen做答案回归；代理证据不是官网gold，因此不能直接晋级生产。

## 7. 下一步验收

先做纯离线检索回放：

- 对两份审计中的决定性 unit 计算 Recall@5/10、MRR；
- 比较 legacy 与 `semantic_slots_v1` 的必要槽位覆盖率；
- 统计首轮证据字符数和预计prompt token；
- 检查支持/宽查/反证是否增加无关文档；
- 计算“首轮已覆盖但仍补检索”的假重试率。

只有在不使用QID白名单、不读取冻结答案、全量统一配置下，召回或token指标稳定改善，才进入新的隔离Qwen运行。

## 8. 三轮离线检索回放结果

固定条件：

- 100题中95题可评估，5题因A13没有可达的非题面证据而跳过；
- locator选中文档、索引、最终Top10均固定；
- gold代理为A13 `used_evidence_ids` 中同时存在于索引和locator文档内的unit；
- 全程0次模型调用，不读取冻结答案，不按QID配置；
- query token只衡量检索工作量，不等于提交模型Token。

| 轮次 | 融合方式 | Any Recall@10 | Coverage@10 | MRR@10 | 平均查询数 | 结论 |
|---|---|---:|---:|---:|---:|---|
| legacy | 当前首轮检索 | 0.863158 | 0.518091 | 0.578901 | 4.673684 | 基线 |
| A1 | 全语义通道等权RRF | 0.810526 | 0.465710 | 0.528446 | 11.557895 | 精准证据被通用词稀释，拒绝 |
| A2 | 保留legacy Top8，语义固定补2条 | 0.852632 | 0.503308 | 0.577849 | 9.252632 | 回退收窄但仍为负，拒绝 |
| A3 | legacy `1/rank` + 补证 `0.11/rank` | 0.873684 | 0.529933 | 0.579077 | 9.252632 | Top10有正向研究信号 |

A3的95题Coverage@10结果为：

```text
提升 2
持平 93
下降 0
```

两道收益题分别体现了两类通用价值：

- 多产品条款题：从选项动态抽取产品锚点，补齐未进入Top10的产品证据；
- 计算题：将年份、指标、关系和单位组合成coverage查询，补回原始计算输入。

但A3仍不能直接打开生产：

1. A13证据并集含支持证据和噪声，不是真实标注；
2. `Any Recall@5` 从0.800000降到0.789474，说明补证会轻微重排头部；
3. 平均查询数增加97.97%，检索query token增加58.67%；
4. 最终证据文本只增加1.24%，说明模型输入增量可控，但仍需真实evidence-gate或独立题集验证；
5. 本方向已完成3轮，不能继续在同一代理集调权重，避免对A13过拟合。

因此当前结论是：`semantic_slots_v1`适合作为低权重补证/证据不足救援候选，不适合替代legacy主检索。生产默认继续保持`legacy`。
