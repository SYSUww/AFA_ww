# 准确性与 Token 研究计划（禁止 Embedding）

更新时间：2026-07-16（Asia/Shanghai）

状态：准备执行

## 1. 研究目标

本阶段只优化两个核心目标：

1. 准确性优先；
2. 在准确性不下降的前提下降低模型 Token。

采用字典序决策，不把准确性和 Token 混成一个可相互抵消的加权分数：

1. 先检查运行完整性与答案格式；
2. 再比较官方准确率；没有官方标签时，使用人工证据复核、代理答案一致率和 evidence audit，但必须分别报告；
3. 准确性不退步的候选中，再选择 Token 更低的方案；
4. 运行耗时、磁盘占用和代码整洁度只作为工程指标，不得冒充 Token 优化。

## 2. 硬约束

### 2.1 禁止项

- 禁止 embedding、向量数据库和 dense retrieval；
- 禁止依赖向量表示的 reranker、cross-encoder reranker 或语义缓存；
- 禁止按 qid 或参考答案硬编码规则；
- 禁止将 v20 一致率或 Group A mask 结果表述为官方 B 榜准确率；
- 禁止在最小题集未通过时直接发起全量付费运行；
- 禁止一次实验同时修改多个主要变量。

### 2.2 允许项

- BM25、字段化 BM25、字符或词粒度的词法检索；
- jieba 领域词典、同义词表、别名表、正则和精确匹配；
- 标题、章节、条款、表格、公式和字段结构化；
- 多个纯词法 query 的 max-score、RRF 等排名融合；
- 本地数值计算、规则判断和证据完整性检查；
- 基于明确规则的动态 TopK、rescue 和模型调用路由；
- 完全相同输入的精确缓存。

## 3. 基线注册表

### 3.1 B0-score：Group A 外测参考

- 版本：`artifacts/submissions/group_a_candidate_accuracy_first_v20_20260630/`
- 用户外测记录：`88%`
- 题数：100
- Token：`211,749`
- 说明：仓库本地没有官方标签，不能自行重算该准确率；该版本没有完成全量 API 复跑，其中 7 道合同题被规则化为 0 Token，因此它是历史外测参考，不是完整可复现的流水线基线。

### 3.2 B0-repro：历史全链路复现

- 版本：`artifacts/reproducible_runs/group_a_dynamic_confidence_20260630_180128/`
- 题数：100
- Token：`593,846`
- 相对 v20 答案差异：6 题
- 说明：用于理解完整 baseline + rescue 成本，不继承 v20 的 88% 外测结论。

### 3.3 B0-mask：当前无 docids 操作基线

- 版本：`attempt_43 / canonical_score_inherit`
- 产物：`artifacts/b_board_migration/no_docids_clean_subset_run/`
- 严格盲测候选题：78
- 相对 v20 代理匹配：`65/78`
- Token：`266,364`，其中 prompt Token `204,352`
- evidence audit：supported `60`、weak `1`、unsupported `11`、contradicted `1`、format conflict `5`
- 零 Token 题：39
- 说明：最终 78 题中有 57 题来自定向恢复，当前 checkpoint 没有 fingerprint；它可以作为当前操作基线，但 R0 完成后必须生成一轮不可混用 checkpoint 的新 `B0-run`，再作为后续正式实验对照。

| 领域 | 题数 | Token | 平均每题 | supported |
|---|---:|---:|---:|---:|
| financial_reports | 20 | 99,648 | 4,982 | 14/20 |
| insurance | 20 | 86,014 | 4,301 | 12/20 |
| regulatory | 20 | 69,356 | 3,468 | 18/20 |
| research | 18 | 11,346 | 630 | 16/18 |

注意：`supported=60/78` 不能直接当作准确率。部分判断题输出 B 时，当前审计会把缺少 B 选项 gate 误判为 unsupported；部分本地规则题也没有统一的 provenance。P0 必须先校准审计口径。

另需注意：v20 中存在来源标注为 multi、但答案只有一个字母的历史输出。v20 一致率不能用于否决格式契约修复。

## 4. 统一实验协议

### 4.1 Run 身份

每个 run 必须记录并校验：

- Git branch 与 commit；
- 题目集合及内容 hash；
- parsed/index 文件路径与内容 hash；
- strategy 配置内容 hash，而不只是绝对路径；
- 模型名、temperature、prompt 模板版本；
- 代码版本、开始时间、完成题数、失败题；
- prompt、completion、total Token；
- 是否恢复，以及恢复来源的 fingerprint。

checkpoint 只有在 fingerprint 完全一致时才能自动恢复。跨策略复用必须显式指定来源和 qid，不能静默按 qid 混入旧结果。

### 4.2 指标分层

硬门槛：

- 预期 qid 全部完成；
- `failed_count=0`；
- 空答案、非法单选、非法多选、非法判断题均为 0；
- Token 汇总与逐题之和一致；
- no-docids 路径不得读取真实 doc_ids 的值或数量参与定位、检索和作答。

准确性指标：

- 官方标签准确率，若可获得；
- 人工证据复核结论；
- 相对 v20 的答案一致率，仅作回归代理；
- calibrated supported、weak、unsupported、contradicted；
- 按领域、题型和错因拆分的变化。

Token 指标：

- 总 Token、平均每题、P50、P90、最大值；
- 按领域、题型、初答、rescue、consistency retry 拆分；
- 每个答案改善或证据改善对应的新增 Token；
- 精确缓存命中节省与单次算法 Token 分开记录。

### 4.3 晋级与淘汰

- 有官方标签时，准确率下降的候选立即淘汰；
- 没有官方标签时，守护题不得回归，contradicted 不得增加，校准后的 supported-or-weak 不得下降；
- 格式修复后，format conflict 必须为 0；
- Token-only 实验要求答案向量不退化、任一领域 proxy/support 不下降；
- Token-only 候选只有在受影响领域下降至少 10%，或全局下降至少 5% 时才晋级；
- 准确性明确提升时，受影响题 Token 可临时增加不超过 20%，随后必须进入 Token 压缩阶段；
- 连续两轮最小题集无准确性或证据收益时，停止该方向；
- 每轮只改变一个主要变量，并记录 promoted/rejected 原因。

现有 `proxy_v1` 将格式、证据覆盖、检索聚焦、规则一致与 Token 混合成一个分数，只保留为诊断列，不再单独用于选出冠军。

### 4.4 前瞻式冻结切片

当前 `dev_mini` 包含在 `dev_stage` 中，二者又包含在 `full_group_a` 中，不能承担独立 gate/holdout 的职责。R0 必须新增不重叠的 v2 切片并提交 qid 列表与 SHA256。

每个领域 20 题按 domain、answer format、题型、文档数和事实/计算/比较/条款标签分层：

- dev：10 题，用于调参与逐题分析；
- gate：5 题，与 dev 不重叠，只允许根据汇总指标晋级；
- holdout：5 题，与 dev/gate 不重叠，每个研究批次最多运行一次；
- smoke：从 dev 中固定选 2 题，只检查链路，不是独立评估集。

项目历史上已经查看过全部 Group A 题，因此该切分只能称为“从现在开始冻结的前瞻式回归集”，不能宣称真正未知测试集。B 榜模拟另建删除真实 docids 后的 `blind_mask_dev/gate` 文件。

生成器与评估器物理分开：

```text
masked questions -> candidate runner -> sealed answers
                                      -> evaluator <- reference answers / true docids
```

candidate runner 不接受参考答案参数，blind Question 对象中不得存在真实 docids；单题参考答案和真实 docids 只能由独立 evaluator 在答案文件关闭后读取。

## 5. 研究优先级

| 优先级 | 编号 | 研究项 | 主要目标 | 是否需要模型调用 |
|---|---|---|---|---|
| P0 | R0 | 固化 fingerprint、完整性和恢复契约 | 实验可信度 | 否 |
| P0 | R1 | 答案格式与 LLM JSON 类型硬契约 | 准确性 | 否 |
| P0 | R2 | 检索边界、重复 unit 与 no-docids 隔离 | 准确性 | 否 |
| P0 | R3 | 校准 evidence audit 与规则 provenance | 指标可信度 | 否 |
| P0 | A1 | 保险多选联合裁决 | 准确性 | 小题集需要 |
| P0 | A2 | 财报 selected evidence 与结构化指标 | 准确性 | 小题集需要 |
| P1 | A3 | 多 query 纯词法融合 | 准确性 | 先离线，赢家需要 |
| P1 | T1 | 抽取式证据压缩与动态 TopK | Token | 需要 A/B |
| P1 | T2 | 共享 evidence pool 与动态调用 | Token | 需要 A/B |
| P1 | T3 | 可复算规则与本地计算 early-stop | 两者 | 小题集需要 |
| P1 | B1 | 无 docids 词法 locator | B 榜准确性 | 先离线，赢家需要 |
| P2 | E1 | retriever/HTTP session 复用与代码拆分 | 运行效率 | 否 |

## 6. 分阶段执行计划

### 阶段 0：零 Token 的可信度与正确性修复

#### R0：实验身份与恢复契约

改动：

- 为普通 run、B 榜迁移 run 统一生成 fingerprint；
- checkpoint 改成原子写入，失败题和完成题状态分开；
- 恢复前验证 domain、split、题目、索引、策略、模型和代码版本；
- 失败题必须进入完整性检查和准确率分母，禁止用成功题子集产生看似更高的指标。
- 新增互不重叠的 `dataset_slices_v2.json` 与对应 `plan_config_v2.json`；
- R0 完成后生成全新的 `B0-run`，不得复用旧 checkpoint。

验收：构造不同策略、不同索引和错误 run-dir，均必须拒绝静默恢复。

#### R1：答案与模型输出硬契约

改动：

- 最终答案格式校验必须在所有策略覆盖之后执行；
- 修复 supported-only 将合法多选重新缩成单项的问题；
- 对 LLM JSON 做 schema/type 校验，字符串 `"false"` 不得判为真；
- 格式修复优先本地完成，不为 JSON 或字母格式额外调用模型；
- 非法响应不得静默回退到首选项，必须记录错误或进入受控 fallback。

最小回放：

- `fin_a_011`
- `ins_a_009`、`ins_a_012`、`ins_a_014`、`ins_a_016`
- 合成 JSON/type 测试

验收：5 个现有 format conflict 全部合法；非目标合法题答案不变；模型 Token 为 0。

#### R2：检索完整性与盲测隔离

改动：

- neighbor expansion 只能扩展同一 doc_id 的相邻单元；
- parsed/index 构建时强制 unit_id 唯一；
- 高亮单元改为字段权重或显式新 ID，不重复注入同一证据；
- no-docids locator 使用不含真实 doc_ids 的数据结构；真实 docids 仅交给运行后 evaluator；
- 加入“删除真实 docids 后候选排序完全一致”的不变性测试。

验收：跨文档邻居为 0、重复 unit_id 为 0，doc recall 不下降，证据总字数不增加。

#### R3：校准 evidence audit

重点：

- 判断题 B 应由“A 被证据反驳”或明确的 false evidence 表示支持；
- 本地规则统一输出 evidence IDs、中间变量、适用条件和置信来源；
- 区分“答案可能错”和“审计信息缺失”，不能统一标成 unsupported。

最小回放：

- 判断题：`fin_a_003/013/018`、`reg_a_010`、`res_a_006/013`
- 保险本地规则：`ins_a_001/003/006/020`

验收：人工逐题复核与审计结论一致；答案和 Token 不变。

### 阶段 1：准确性专项

#### A1：保险多选联合裁决

问题集：`ins_a_009/012/014/016`。

守护集：`ins_a_002/005/007/008/010/015`。

实验顺序：

1. 构建去重的 question-level evidence pool，但保持当前调用方式；
2. 当多选只有一个 supported 选项时，仅对第二候选执行一次针对性词法 rescue；
3. 比较逐选项判断和一次整题受约束裁决；
4. 只有证据完整时才允许规则或模型输出最终选项。

晋级：4 个格式冲突归零，selected gate 覆盖提升，守护题无回归；新增调用未改善任何题则淘汰。

#### A2：财报 selected evidence

问题集：`fin_a_003/008/011/013/015/018`。

守护集：`fin_a_002/005/009/012/016`。

实验顺序：

1. 统一公司、年份、指标、数值和单位；
2. 构建 `指标-年份-数值-单位-doc_id` 结构化 evidence card；
3. 对同比、占比和排序题配对当期/基期数据并用 Decimal 复算；
4. 只在字段缺失或冲突时交给模型判断。

晋级：contradicted `1 -> 0`、format conflict `1 -> 0`、相关证据进入 Top5、守护题无回归。Token 增加超过 20% 且没有准确性或支持提升则淘汰。

#### A3：无 embedding 的多 query 融合

只比较纯词法方案：

- question + option；
- 实体、年份、指标、金额 query；
- 标题/字段/条款 query；
- max-score 与 RRF 融合。

先离线比较 doc recall@5、selected evidence recall 和重复率，只对赢家发起模型 A/B。融合只增加重复 evidence、或 recall 下降时立即淘汰。

领域顺序：financial_reports、insurance 优先；regulatory 只处理高 Token bad case；research 冻结为回归守护域。

### 阶段 2：准确率冻结后的 Token 优化

#### T1：抽取式证据压缩与动态 TopK

禁止用 LLM 摘要证据。采用：

- unit_id 和文本去重；
- 命中词附近句窗；
- 强制保留标题、主体、年份、数字、单位、否定词、条件和例外；
- 财报使用结构化指标卡；
- 保险使用责任、条件、公式、限额卡；
- 法规使用主体、动作、期限、例外卡。

比较原始长度的 100% / 80% / 65%。任何答案变化、关键数字或否定条件丢失、supported 下降均回退。

动态 TopK：`3/4 -> coverage gate -> 6/8 -> 仍缺关键字段才到 12`。连续一轮没有新增文档、关键字段、公式或条款时停止 rescue。

#### T2：共享 evidence pool 与动态调用

1. 先共享去重 evidence pool，保持原调用次数；
2. 再试一次整题调用判断全部选项；
3. 只对 uncertain 选项做单项复核；
4. research 暂不改调用方式。

止损：多选答案、option label、gate 状态任一退化即回退。

#### T3：可靠规则与本地计算 early-stop

允许免调用的规则必须满足：证据充分、适用条件明确、可本地复算。

- 财报：同比、比例、排序、单位换算；
- 保险：给付公式、免赔额、上限、max/min；
- 法规：期限、金额、主体、明确否定；
- 合同：发行规模、期限、评级等字段。

规则必须输出 evidence IDs、中间变量、计算结果和适用条件；规则与人工证据结论冲突时关闭该规则，不得按 qid 打补丁。

Token 阶段首轮目标：全局不低于 `65/78` 代理匹配、校准后支持指标不退步、format conflict 为 0，总 Token 至少下降 5%，即不高于约 `253,046`。

### 阶段 3：无 docids 词法 locator

只研究：

- fielded BM25 文档 profile；
- 标题、文件名、主体、年份、编号和金额精确命中；
- 字符 n-gram BM25 处理 OCR 和别名差异；
- 多 query RRF；
- canonical 去重、文档配额和动态 TopK。

必须先在不含真实 docids 的 Question 数据上运行，再由独立 evaluator 计算 recall。晋级门槛不低于当前 `doc_recall@5=0.897436`、`doc_recall@10=0.910256`，并通过下游答案和 Token 验证。

### 阶段 4：全量验证与冻结

执行顺序：

1. 合成单测和零 Token 回放；
2. bad case 最小集；
3. 领域守护集；
4. Group A 全量；
5. 78 题 no-docids mask；
6. 真实 Group B 数据到位后再做官方验证。

最终保留 Pareto 前沿，不只保留一个总分：

- 最高准确性版本；
- 同准确性最低 Token 版本；
- 更高准确性但 Token 略高的候选版本。

## 7. 第一批开工范围

第一批只做 P0，禁止模型调用：

1. 固化互不重叠的 v2 切片、qid hash 和晋级配置；
2. 新增运行 fingerprint、恢复校验和原子 checkpoint；
3. 修复最终答案格式覆盖问题；
4. 增加 LLM JSON schema/type 校验；
5. 修复同文档 neighbor 边界与重复 unit_id；
6. 隔离 no-docids 运行数据和真实 docids evaluator；
7. 校准判断题与本地规则的 evidence audit；
8. 补齐对应单元测试和离线回放；
9. 全部通过后，单独申请一次全新 `B0-run` 的模型预算，再决定是否开启 A1/A2。

第一批完成标准：

- 不调用模型、不新增 Token 消耗；
- 所有现有测试和新增测试通过；
- 5 个 format conflict 的本地回放均输出合法格式；
- 跨文档邻居、重复 unit ID、错误 checkpoint 复用均有回归测试；
- 审计口径经过人工逐题核对；
- 形成第一份 promoted/rejected 实验记录。
