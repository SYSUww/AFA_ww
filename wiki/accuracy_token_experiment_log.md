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
