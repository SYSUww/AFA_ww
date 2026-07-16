# 准确性与 Token 实验记录

本文件按实验追加记录。禁止记录 API Key、API Base 等敏感值。代理指标、人工复核和官方结果必须分开表述。

## 基线注册表｜2026-07-16

- Git branch：`codex/b-board-optimization-iteration`
- Git commit：`55b2154`
- `B0-score`：v20，用户外测记录 `88%`，100 题 Token `211,749`；没有完整 API 复跑，不作为流水线 Token 基线
- `B0-repro`：历史全链路复现，100 题 Token `593,846`，相对 v20 差异 6 题
- `B0-mask`：`attempt_43 / canonical_score_inherit`
- no-docids 题数：78
- v20 代理匹配：`65/78`
- Token：`266,364`
- evidence audit：supported `60`、weak `1`、unsupported `11`、contradicted `1`、format conflict `5`
- 说明：以上 no-docids 指标来自 Group A mask，不是官方 B 榜结果；当前 checkpoint 无 fingerprint，R0 后需要全新生成 `B0-run`。

## 实验模板

### `<experiment_id>`｜`<时间>`

- 状态：planned / running / promoted / rejected / failed
- 研究编号：R0 / R1 / R2 / R3 / A1 / A2 / A3 / T1 / T2 / T3 / B1
- 假设：
- 唯一主要变量：
- Git commit：
- Run fingerprint：
- 配置路径与 hash：
- 输入、parsed、index 路径与 hash：
- 模型与 temperature：
- 数据切片与 qids：
- 对照 run：
- 候选 run：
- 完成题数 / 失败题数：
- 官方准确率：未获得 / 数值
- 人工证据复核：
- v20 代理一致率：
- supported / weak / unsupported / contradicted / format conflict：
- prompt / completion / total Token：
- 按领域 Token：
- 答案变化清单：
- 证据变化清单：
- Token 变化与原因：
- 风险与异常：
- 晋级门槛检查：
- 结论：promoted / rejected
- 下一步：

## P0-R0｜2026-07-16｜运行身份、盲测隔离与冻结切片

- 状态：promoted（工程可信度门槛，不代表模型准确率提升）
- 研究编号：R0 / R2
- 唯一主要变量：运行与评估完整性契约
- Git branch：`codex/accuracy-token-optimization`
- Git commit：`89c79d4`
- 数据切片：5 个领域各 dev/gate/holdout=`10/5/5`，smoke=`2` 且属于 dev；互不重叠检查通过
- 完成题数 / 失败题数：合成回归 4 / 0
- 官方准确率：未获得
- 模型与 Token：未调用模型；prompt/completion/total 新增均为 `0`
- 验证：普通 run 与 B-board run 均覆盖 fingerprint mismatch；旧 manifest 无 fingerprint 拒绝恢复；blind locator 对真实 doc_ids 的值和数量不变；失败题进入预期分母且 `promotion_score=null`
- 结论：promoted；后续正式实验必须使用新指纹和新 run dir
- 下一步：明确付费预算后运行全新 `B0-run`

## P0-R1｜2026-07-16｜答案与 JSON 类型契约

- 状态：promoted
- 研究编号：R1
- 唯一主要变量：最终答案与模型 JSON 类型契约
- Git commit：`89c79d4`
- 官方准确率：未获得
- 零模型回放：`fin_a_011 D -> AD`、`ins_a_009 C -> AC`、`ins_a_012 A -> AB`、`ins_a_014 A -> AB`、`ins_a_016 D -> AD`
- 验证：5 个历史 multi 冲突全部变为至少 2 个合法选项；字符串 `"false"` 被拒绝并重试；连续非法 fallback 显式失败而非静默选 A
- 模型与 Token：未调用模型；新增 Token `0`
- 风险：回放只验证格式契约，不等于官方答案正确
- 结论：promoted；进入全新 B0-run 验证下游答案向量

## P0-R2｜2026-07-16｜检索边界与唯一单元

- 状态：promoted
- 研究编号：R2
- 唯一主要变量：检索单元完整性
- Git commit：`89c79d4`
- insurance 标准索引：16 文档、610 单元、610 唯一 ID、缺失 0、重复 0
- research 标准索引：15 文档、940 单元、940 唯一 ID、缺失 0、重复 0
- 验证：通用/监管 neighbor expansion 跨文档为 0；两域 retriever 初始化和 top_k=3 BM25 查询成功
- 模型与 Token：未调用模型；新增 Token `0`
- 风险：重建后尚未发起模型答案回归，doc recall 和下游准确性等待 B0-run
- 结论：promoted

## P0-R3｜2026-07-16｜Evidence audit 校准与规则 provenance

- 状态：promoted（审计口径）
- 研究编号：R3
- 唯一主要变量：证据支持状态的判定语义
- Git commit：`89c79d4`
- 对照：旧 78 题答案产物；答案与 Token 完全不改，仅离线重算 audit
- 旧口径：supported `60`、weak `1`、unsupported `11`、contradicted `1`、format conflict `5`
- 新口径：supported `66`、weak `1`、unsupported `5`、contradicted `1`、format conflict `5`
- 人工核对：`fin_a_003/013/018`、`reg_a_010`、`res_a_006/013` 的 B 均映射为“被评估的 A 陈述为 false”，并继承 A 的 evidence gate；6 题不再因缺少虚构的 B gate 被误判
- 规则 provenance：schema v1、稳定 rule ID/version、strict decision、最终 evidence ID 子集、inputs/conditions/可复算 outputs 和 allowlist validator；旧 artifact 与未知/篡改规则继续 unsupported
- 模型与 Token：未调用模型；新增 Token `0`
- 风险：旧 `ins_a_001/003/006/020` 仍缺结构化 provenance，保持 unsupported；当前仅新放行共享免赔额规则
- 结论：promoted；审计改善不得表述为准确率提升
