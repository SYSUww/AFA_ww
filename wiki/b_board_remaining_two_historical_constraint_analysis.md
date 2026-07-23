# B 榜剩余两道错题：历史官网分数约束分析

> 分析日期：2026-07-23（Asia/Shanghai）
> 当前假设：`i032` 的 `fin_b_005=AC` 命中官网答案，因此
> `i032` 的答案准确率为 98%，只剩 2 道错题。
> 分析边界：只使用仓库内官网成绩记录、提交 CSV、候选 manifest、
> 原始抽取材料和既有固定审计结果；未进行官网提交，也未把代理分数当作官网标签。

## 1. 结论

历史总分**不能唯一定位**剩余两道错题，但可以得到一个此前未被完整利用的强约束：

```text
B = {fc_b_019, ins_b_006, ins_b_012, ins_b_016}

剩余 2 道错题中，至少 1 道、至多 2 道位于 B。
```

原因是：

1. 提交 006→007 的五题组合使准确率从 94% 到 97%，总效果为 `+3`；
2. 五题中的 `ins_b_017=ABCD` 后来已由单题消融确认正确，因此
   `ins_b_017: AD→ABCD` 必然贡献 `+1`；
3. 所以另外四题的历史合计效果严格等于 `+2`；
4. 四道多选题的新旧答案都不同，在等权、完全匹配、无部分分的计分假设下，
   四题只能是：
   - 2 题由错改对、2 题新旧都错；或
   - 3 题由错改对、1 题由对改错。

这两种情况分别意味着当前四题中有 2 题或 1 题错误。

特别需要否定两个看似直观、但方程并不支持的结论：

- **不能证明剩余两错分别位于“94%→97% 五题组”和“三道百分号组”，各恰好一道。**
  历史方程允许两错都在上述四题集合，也允许一错在四题集合、另一错在此前从未变化的任意题。
- **不能一般性证明当前错题的旧答案就是正确答案。**
  若四题集合当前有 2 错，对应的是 `(p,m,z)=(2,0,2)`，两道错题的新旧答案都错；
  只有当四题集合当前恰好 1 错时，才是 `(3,1,0)`，这唯一错题的旧答案才正确。

结合原文，`ins_b_006=BCD`、`ins_b_012=BCD`、`ins_b_016=ABD`
都有逐项正证据和未选项排除证据。如果额外接受这三题均正确，则历史方程会**唯一推出**：

```text
fc_b_019: BD → ABCD 的效果 = -1
即官网口径下 BD 正确、当前 ABCD 错误。
```

但这还不能宣布 `fc_b_019=BD` 已确认，因为长安银行原文又逐字支持 A、B、C、D。
因此当前最准确的说法是：

- `fc_b_019` 是剩余错题中的**第一优先官网标签冲突候选**；
- 四题集合 `B` 是严格候选集合，至少命中一道；
- 第二道错题仍不能唯一定位。若 `fc_b_019=BD` 成立，则第二道一定在
  其余 95 道中的未锁定题；优先审查 `res_b_017`、`res_b_014`，
  其次为依赖否定性检索的 `fc_b_007`、`ins_b_007`、`ins_b_010`、`ins_b_013`。

## 2. 输入完整性与假设

### 2.1 文件盘点

`artifacts/b_board_actual/candidates` 下共扫描：

- `submit.csv`：119 个；
- `candidate_manifest.json`：3 个；
- 按路径排序后，对每个 `submit.csv` 的 `SHA-256 + 路径` 清单再做 SHA-256：
  `86219d9b7aaa04734579a01da3f8e426edfa76dba45854f3fb7960b86e2cd02f`；
- 对 3 个 manifest 的同类清单摘要：
  `030690e062d3f2b30074bd7116c4b6f5c0f11bb1c1d716cbf8a97947214974b9`。

119 个 CSV 中，只有下表 8 个文件在
`wiki/b_board_submission_score_log.md` 中绑定了真实官网结果；
其余是未提交候选、离线迭代或组合产物，不能增加官网分数方程。
`i032` 也尚未提交，其 manifest 明确写着 `official_result: null`，
见 `artifacts/b_board_actual/candidates/i032_evidence_full_year_dividend/fin005_ac_single_v1/candidate_manifest.json:28-32`。

### 2.2 计分假设

以下约束依赖：

- 共 100 题、每题等权；
- 答案完全匹配；
- 无部分分；
- 同一道多选题只有一个正确选项集合；
- 日志中的用户反馈分数与记录的 SHA-256 文件一一对应；
- 本文按用户要求额外假设 `fin_b_005=AC` 正确，故 `i032=98%`。

数值题可能存在字符串等价，例如官网已表现为 `67.10` 与 `67.1`
得分相同；因此本文不会把“效果为 0”自动解释为两个字符串都错。

## 3. 官网提交重建

| 提交 | 官网准确率 | 文件 SHA-256 | 相对比较基线的答案变化 |
|---|---:|---|---|
| 001 | 91% | `05cbfdc9a6e7ad384ab7381d28b497c27b52793362ad454970fd071ad0e737ed` | 初始基线 |
| 002 | 92% | `678cda15a5616611e4048944d481407ba3b506f600367f10ad49698530e6b8fa` | `fc_b_003 ABCD→ACD`；`reg_b_001 AD→ACD`；`res_b_004 AD→BCD`；`res_b_012 67.10→67.1` |
| 003 | 93% | `09707c1865793f79886053de0cf017c10732adcd5c5868412572761266b45516` | `fin_b_013`、`fin_b_017`、`res_b_005` 增加 `%` |
| 004 | 91% | `fd16a0a629339c1bc476d5c1db9e201c4d71f91b4b7d0e90e58a230e684f4442` | 相对 003：`reg_b_001 ACD→AD`、`res_b_004 BCD→BD` |
| 005 | 92% | `a077b514650c89b3c71c973f5430d55589065039eb37aca901ce738459750251` | 相对 003：仅 `reg_b_001 ACD→AD` |
| 006 | 94% | `b3eb494684225cdb60efffed409cb49a60e1e0d94e1d7ce276d98ee99e355f1a` | 相对 003：仅 `fc_b_003 ACD→ABCD` |
| 007 | 97% | `497658e3e76c0df3bb28134a4d1b2d442bb5b5602ea4654d7d0ce019aecb400d` | 相对 006：`fc_b_019` 与 4 道保险题，共 5 题 |
| 008 | 96%（由综合分反解） | `56cd46ae731ac75744f36f8bc682118ed619c7b99a57490d667117f96e414889` | 相对 007：仅 `ins_b_017 ABCD→ABC` |
| i032（假设） | 98% | `8449237193e0895e05aa1166566e7b1213206510d6af91d6fcd4de0893b54d54` | 相对 007：仅 `fin_b_005 ACD→AC` |

成绩、路径和日志 SHA-256 的第一手记录分别见：

- 提交 001：`wiki/b_board_submission_score_log.md:13-33`；
- 提交 002：`wiki/b_board_submission_score_log.md:63-91`；
- 提交 003：`wiki/b_board_submission_score_log.md:101-129`；
- 提交 004/005：`wiki/b_board_submission_score_log.md:284-330`；
- 提交 006：`wiki/b_board_submission_score_log.md:373-393`；
- 提交 007：`wiki/b_board_submission_score_log.md:416-432`；
- 提交 008：`wiki/b_board_submission_score_log.md:459-485`；
- i032 的单题变化和 SHA：上述 manifest `:9-16,28-32`。

本轮重新计算出的 CSV 行也与日志一致：

- 提交 006 的 `fc_b_003=ABCD`：
  `artifacts/b_board_actual/candidates/i023_gpt56_suspect_case_optimization/diff7_fc_b_003_rollback_abcd_v1/submit.csv:5`；
- 提交 007 的五题答案：
  `artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/submit.csv:21,48,54,58-59`；
- 提交 008 的 `ins_b_017=ABC`：
  `artifacts/b_board_actual/candidates/i031_accuracy_boundary_d_rollback/ins017_drop_d_single_a2/submit.csv:59`；
- i032 的 `fin_b_005=AC`：
  `artifacts/b_board_actual/candidates/i032_evidence_full_year_dividend/fin005_ac_single_v1/submit.csv:27`。

## 4. 差分方程

记 `C(q, a)∈{0,1}` 表示题 `q` 使用答案 `a` 时是否得 1 分。

### 4.1 已经能够单题锁定的答案

提交 003→005：

```text
C(reg_b_001, AD) - C(reg_b_001, ACD) = 92 - 93 = -1
```

所以 `reg_b_001=ACD` 正确、`AD` 错误。

提交 004→005 只有 `res_b_004` 不同：

```text
C(res_b_004, BD) - C(res_b_004, BCD) = 91 - 92 = -1
```

所以 `res_b_004=BCD` 正确、`BD` 错误。

提交 003→006：

```text
C(fc_b_003, ABCD) - C(fc_b_003, ACD) = 94 - 93 = +1
```

所以 `fc_b_003=ABCD` 正确、`ACD` 错误。

提交 007→008：

```text
C(ins_b_017, ABC) - C(ins_b_017, ABCD) = 96 - 97 = -1
```

所以 `ins_b_017=ABCD` 正确、`ABC` 错误。提交 008 的 manifest
也记录了反解后的 `accuracy_score=96`，见
`artifacts/b_board_actual/candidates/i031_accuracy_boundary_d_rollback/ins017_drop_d_single_a2/candidate_manifest.json:28-37`。

本文假设再锁定 `fin_b_005=AC` 正确、`ACD` 错误。

因此 i032 中可排除为剩余错题的 5 个 qid 是：

```text
fc_b_003, reg_b_001, res_b_004, ins_b_017, fin_b_005
```

### 4.2 早期四题组合

提交 001→002 总效果为 `+1`。代入已经锁定的三项：

```text
fc_b_003: ABCD→ACD = -1
reg_b_001: AD→ACD  = +1
res_b_004: AD→BCD  = +1
--------------------------------
前三项合计              +1
```

因此：

```text
C(res_b_012, 67.1) - C(res_b_012, 67.10) = 0
```

这只确认两个数值表示得分等价，不足以单独证明该题内容正确。
日志的同一结论见 `wiki/b_board_submission_score_log.md:395-406`。

### 4.3 三道百分号题

提交 002→003：

```text
effect(fin_b_013 加 %)
+ effect(fin_b_017 加 %)
+ effect(res_b_005 加 %)
= +1
```

没有后续单题消融，故不能从官网总分确定三项各自贡献。
不过原文计算审计强支持当前值：

- `fin_b_013` 的复算见
  `artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/evaluation_gpt56_full100_evidence_audit_v10/confidence_audit.jsonl:33`；
- `fin_b_017` 见同文件 `:37`；
- `res_b_005` 见同文件 `:85`。

因此三题仍在数学上的未锁定集合内，但不是当前最高风险。

### 4.4 94%→97% 五题组合的更新后方程

提交 006→007 的真实 CSV 差分为：

```text
fc_b_019: BD→ABCD
ins_b_006: CD→BCD
ins_b_012: BD→BCD
ins_b_016: BD→ABD
ins_b_017: AD→ABCD
```

总效果为 `+3`。`ins_b_017=ABCD` 已被后续单题实验确认；
在唯一完全匹配答案假设下，`AD` 必然错误，所以该项效果为 `+1`。
于是：

```text
effect(fc_b_019)
+ effect(ins_b_006)
+ effect(ins_b_012)
+ effect(ins_b_016)
= +2
```

设四题中：

- `p` 道为旧错、新对，效果 `+1`；
- `m` 道为旧对、新错，效果 `-1`；
- `z` 道为旧错、新错，效果 `0`。

则：

```text
p - m = 2
p + m + z = 4
```

只有两组非负整数解：

| p | m | z | 当前四题中的错题数 `m+z` |
|---:|---:|---:|---:|
| 2 | 0 | 2 | 2 |
| 3 | 1 | 0 | 1 |

所以四题中至少 1 错、至多 2 错。这是本轮最强的纯官网历史约束。

## 5. 与原文证据交叉后的候选优先级

### P0：`fc_b_019`

这是最值得做单题官网消融的候选，但当前状态是“强历史约束与强原文证据冲突”，
而不是已经证明答案错。

支持当前 `ABCD` 的原文：

- A：信用风险主要集中于贷款、债券投资及承诺与担保：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:129-157`；
- B：流动性风险来源：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:159-163`；
- C：董事会及风险管理委员会：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:1020-1022`；
- D：业务部门、风险主管部门、审计部门三道防线：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:1056-1058`。

固定 GPT-5.6 审计也给 `ABCD` 99/high，见
`artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/evaluation_gpt56_full100_evidence_audit_v10/confidence_audit.jsonl:19`。

但是三道保险题同样有接近封闭的证据：

- `ins_b_006=BCD`：
  B 的“恐怖袭击”见 `insurance/8.md:96-100`，
  C 见 `insurance/10.md:61-66`，
  D 见 `insurance/12.md:79-90`；
  固定审计逐项结论见 `confidence_audit.jsonl:46`。
- `ins_b_012=BCD`：
  B 见 `insurance/11.md:50-58`，
  C 见 `insurance/12.md:77-90`，
  D 见 `insurance/14.md:45-52`；
  固定审计逐项结论见 `confidence_audit.jsonl:52`。
- `ins_b_016=ABD`：
  A 见 `insurance/2.md:112-124`，
  B 见 `insurance/4.md:145-161`，
  D 见 `insurance/16.md:135-147`；
  固定审计逐项结论见 `confidence_audit.jsonl:56`。

如果把这三道保险证据视为与官网标签完全一致，则它们三项历史效果都是 `+1`，
四题方程会迫使 `fc_b_019` 的效果为 `-1`，即旧 `BD` 对、当前 `ABCD` 错。
这就是把 `fc_b_019` 列为 P0 的原因。

### P0b：`ins_b_006`、`ins_b_012`、`ins_b_016`

不能从严格集合中删除这三题。若 `fc_b_019=ABCD` 实际是官网正确答案，
三道保险题的历史效果总和就只能为 `+1`，从而三题中至少 1 道当前错误。
原文审计目前找不到哪一道应错，说明这里可能存在隐藏标签边界、产品映射口径
或官方答案本身与直接材料不一致。

按现有证据强度，三题内部没有足以支持直接改答的明显优先级；
固定审计分别为 99、98、99，1 分差异不具备可用的排序意义，
不足以支持修改其中任何一题。

### P1：第二道错题的范围候选

如果 `fc_b_019=BD` 最终成立，四题方程同时推出三道保险当前答案均正确，
那么另一道错题一定在集合 `B` 之外。当前优先级：

1. `res_b_017=AC`：跨行业类比把 L3 扩到“完全自动驾驶”，
   固定审计 `59/low`、错误可能性 46，见 `confidence_audit.jsonl:97`；
   原文全量复核仍只判“条件通过”，见
   `wiki/b_board_97_evidence_accuracy_audit.md:147-152`。
2. `res_b_014=CD`：D 的“金融工具”和释放即期消费是强概括，
   固定审计 `0/blocked`、错误可能性 43，见 `confidence_audit.jsonl:94`；
   原文全量复核同样只判“条件通过”，见
   `wiki/b_board_97_evidence_accuracy_audit.md:139-145`。
3. `fc_b_007=BC`、`ins_b_007=BC`、`ins_b_010=AB`、`ins_b_013=BD`：
   当前选项有正证据，但未选项排除依赖完整章节或合同否定性检索；
   既有全量审计风险清单见
   `wiki/b_board_97_evidence_accuracy_audit.md:189-193`。
4. `res_b_008=AC`、`res_b_019=AB`：
   封存证据审计分别为 45、44 的错误可能性，见
   `confidence_audit.jsonl:88,99`；完整原文审计认为可闭环，
   因此仅列次级观察，不建议直接改答。

`fin_b_013`、`fin_b_017`、`res_b_005` 虽然历史 `%` 组合尚未拆分，
但可独立复算且固定审计为 high，优先级低于上述语义边界题。

## 6. 为什么不能推出“两组各错一题”

把“百分号组”记为：

```text
F = {fin_b_013, fin_b_017, res_b_005}
```

官网只给出这组三次格式变化的合计效果 `+1`，没有给出当前组内错题数。
这条方程与“i032 当前只剩 2 错”联立后，仍至少存在以下三类合法情况：

1. `B` 中 2 错、`F` 中 0 错、其余题 0 错；
2. `B` 中 1 错、`F` 中 1 错、其余题 0 错；
3. `B` 中 1 错、`F` 中 0 错、从未发生答案变化的题中 1 错。

三类都不违反任何官网总分。尤其是提交 003→i032 的总提升 `+5` 已完全由
`fc_b_003` 的 `+1`、五题组的 `+3` 和假设成立的 `fin_b_005 +1` 解释；
百分号组在提交 003 和 i032 中答案相同，不会为这个差值增加新约束。

所以“两个差分组各恰好一错”只是三类合法情况中的一种，不是历史分数结论。

## 7. 为什么无法唯一定位

在 i032=98% 假设下，已锁定 5 道当前正确题，剩余候选全集有 95 道。
历史官网分数对当前 2 道错题只额外增加了一个有区分力的集合约束：

```text
1 <= wrong_count({fc_b_019, ins_b_006, ins_b_012, ins_b_016}) <= 2
```

因此仍存在两类合法世界：

1. 四题集合中恰好 2 道错，其余 91 道全对；
2. 四题集合中恰好 1 道错，另外 91 道中恰好 1 道错。

所有从提交 001 到 i032 始终未变化的题，在官网总分方程中具有相同系数，
无法彼此区分。未提交候选没有外部成绩，也不能增加方程秩。
所以仅凭现有历史成绩，不可能唯一恢复两个 qid。

## 8. 下一次提交的最高信息增益设计

若用户决定再消耗一次官网机会，最有价值的是基于 i032 仅做：

```text
fc_b_019: ABCD → BD
```

其余 99 题冻结。三种结果分别意味着：

| 得分 | 可确认结论 |
|---:|---|
| 99% | `BD` 正确、`ABCD` 错；三道保险当前答案同时被历史方程锁定正确；只剩 1 道错在集合 B 外 |
| 98% | `BD` 与 `ABCD` 都未命中；`fc_b_019` 是错题，且三道保险中恰有 1 道当前也错 |
| 97% | `ABCD` 正确、`BD` 错；剩余 2 道中至少 1 道位于三道保险题 |

这比测试未经官网出现的 `fc_b_019=ABC` 信息增益更高：
`BD` 是真实 94% 提交中的旧答案，已有历史方程可以与新结果联立；
`ABC` 只来自离线边界猜测，没有对应官网分数。

在提交前仍建议先对四题做一次“按官方题干字面，而非一般语义”的人工盲审。
本报告不创建候选、不修改答案，也不建议在没有合法替代组合时直接提交
`res_b_014` 或 `res_b_017`。

## 9. 最终判断

- 能否找到两个确定错题：**暂时不能**。
- 能否证明两个历史差分组各错一题：**不能**。
- 当前错题的旧答案是否必然正确：**不必然**；只有四题集合恰好 1 错时才成立。
- 能否显著缩小范围：**可以**。至少一错位于
  `fc_b_019 / ins_b_006 / ins_b_012 / ins_b_016`。
- 当前最可能的单题：**`fc_b_019`**，候选替代应先测试历史旧值 `BD`，
  而不是未经官网约束的 `ABC`。
- 第二题：尚未形成可提交级别的唯一答案；证据审计优先
  `res_b_017`、`res_b_014`，再审依赖否定性检索的四题。
- 最大风险：历史分数约束与原文逐字证据出现真实冲突，说明剩余错误很可能不是
  “材料里明显答错”，而是官方选项边界、隐藏标签口径或题目生成标签本身的问题。
