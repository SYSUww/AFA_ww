# B 榜 97% 基线：最可能三道错题证据排序

> 日期：2026-07-23（Asia/Shanghai）
> 对象：官网已得 97% 的
> `artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/submit.csv`
> 重要边界：`fin_b_005=AC` 候选**尚未提交官网**，因此 98% 不是事实，
> 只是“若该修正命中隐藏答案”的条件假设。
> 下列置信度是证据排序分，不是经过历史样本校准的真实概率，不能相乘或解释为官网命中率。

## 1. 当前最可能的三题组合

| 排名 | qid | 97%版本答案 | 优先替代 | 证据排序置信度 | 判定 |
|---:|---|---|---|---:|---|
| 1 | `fin_b_005` | `ACD` | `AC` | 95/100 | 原文和全年口径直接反证 D，但尚未官网验证 |
| 2 | `fc_b_019` | `ABCD` | 优先验证历史旧值 `BD` | 65/100 | 历史总分方程强推，但与原文逐字支持 ABCD 冲突 |
| 3 | `fin_b_013` | `40.05%；10.10` | `40.05；10.10` | 42/100 | 题干明确“不带单位”，且百分号组三题净增仅 +1 |

当前最可能的 97% 基线三错组合为：

```text
fin_b_005 / fc_b_019 / fin_b_013
```

这是一组用于安排审计和单题消融顺序的工作假设，不是三个已确认错误。
其中只有 `fin_b_005` 达到“可以优先生成单题候选”的证据强度；
`fc_b_019` 适合做官网标签冲突验证；`fin_b_013` 有合法格式替代，
但证据强度不足以与前两题捆绑提交。

## 2. `fin_b_005`：95/100

### 证据

97% 文件当前为 `ACD`：

`artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/submit.csv:27`

宁德时代原始年报明确：

- 2025 年中期每 10 股分红 10.07 元：
  `artifacts/extracted_cleaned/financial_reports/annual_catl_2025_report.md:890-892`；
- 年末表中的 69.57 元是在扣除中期分红后“本次剩余待分配”的金额：
  `artifacts/extracted_cleaned/financial_reports/annual_catl_2025_report.md:908-910`。

所以宁德时代全年为：

```text
10.07 + 69.57 = 79.64 元/10股
```

美的集团年报直接写明全年 43 元，由中期 5 元和年末 38 元组成：

`artifacts/extracted_cleaned/financial_reports/annual_midea_2025_report.md:47`

因此两者全年差额为：

```text
79.64 - 43 = 36.64
```

D 所称 26.57 使用了宁德时代“剩余分红”而非全年分红，故应排除 D，
答案应为 `AC`。完整证据审计结论见
`wiki/b_board_97_evidence_accuracy_audit.md:115-125`。

### 为什么不是 100

- 官网没有逐题标签；
- `AC` 候选 manifest 仍是 `official_result: null`：
  `artifacts/b_board_actual/candidates/i032_evidence_full_year_dividend/fin005_ac_single_v1/candidate_manifest.json:28-32`；
- 因此“修正后 98%”只是条件预测，不可写成真实成绩。

## 3. `fc_b_019`：65/100

### 历史总分约束

提交 006→007 从 94% 提升至 97%，共改变：

```text
fc_b_019: BD→ABCD
ins_b_006: CD→BCD
ins_b_012: BD→BCD
ins_b_016: BD→ABD
ins_b_017: AD→ABCD
```

记录见 `wiki/b_board_submission_score_log.md:416-439`。

后续单题提交已经确认 `ins_b_017=ABCD` 正确：

`wiki/b_board_submission_score_log.md:459-485`

因此 `ins_b_017: AD→ABCD` 必然贡献 `+1`，其余四题历史效果合计只能为 `+2`。
严格枚举只有两种结构：

```text
2题 +1，2题 0
或
3题 +1，1题 -1
```

故当前
`{fc_b_019, ins_b_006, ins_b_012, ins_b_016}`
中至少一题、至多两题错误。完整推导见
`wiki/b_board_remaining_two_historical_constraint_analysis.md:216-260`。

三道保险题的当前答案都有直接条款闭环：

- `ins_b_006=BCD`：
  `artifacts/extracted_cleaned/insurance/8.md:96-100`、
  `insurance/10.md:61-66`、`insurance/12.md:79-90`；
- `ins_b_012=BCD`：
  `insurance/11.md:50-58`、`insurance/12.md:77-90`、
  `insurance/14.md:45-52`；
- `ins_b_016=ABD`：
  `insurance/2.md:112-124`、`insurance/4.md:145-161`、
  `insurance/16.md:135-147`。

若三题均与官网标签一致，则四题方程唯一迫使：

```text
effect(fc_b_019: BD→ABCD) = -1
```

即官网口径下旧值 `BD` 正确、当前 `ABCD` 错误。

### 冲突证据

长安银行原文又直接支持四项：

- A：信用风险集中在贷款、债券投资及承诺与担保：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:129-157`；
- B：流动性风险来源：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:159-163`；
- C：董事会及下设风险管理委员会：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:1020-1022`；
- D：业务、风险主管、审计三道防线：
  `artifacts/extracted_cleaned/financial_contracts/text12.md:1056-1058`。

所以 65/100 来自“强历史约束 + 强文本冲突”：

- 它是最值得测试的官网标签差异题；
- 但不能仅凭当前证据宣布 `BD` 已确认；
- 若再提交，优先单测历史真实出现过的 `BD`，而不是无官网方程的 `ABC`。

## 4. `fin_b_013`：42/100

当前答案为 `40.05%；10.10`，题型是两空计算题，不受多选题“至少两个选项”
约束。原始金额复算支持数值 `40.05` 和 `10.10`，风险只在第一空是否应带 `%`。

- 题干明确要求“答案格式为同比增幅；占比提高百分点，均保留两位小数、不带单位”，
  见
  `artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/answers.json:11991-11995`；
- 当前第一空却是 `40.05%`，见同文件 `answers.json:11683-11690`；
- 提交 002→003 同时给 `fin_b_013`、`fin_b_017`、`res_b_005` 增加 `%`，
  三题合计只提升 1 分，见 `wiki/b_board_submission_score_log.md:123-158`。

一个与官网总分完全一致的合法解释是：

```text
fin_b_013 加 % = -1
fin_b_017 加 % = +1
res_b_005 加 % = +1
合计 = +1
```

这不是唯一解释；平台也可能把 `%` 视为可归一化格式。因此本题只是第三嫌疑，
不能直接宣称去掉 `%` 正确。

**错题置信度：42/100。**

### 多选题至少两个选项的硬约束

该硬约束会显著降低此前语义边界题的嫌疑：

- `res_b_017`：B、D 有明确反证；既然至少两个选项正确，A、C 就是唯一可行组合，
  当前 `AC` 应保留；
- `res_b_014`：A、B 被反证，C 明确成立；至少两项正确会反向加强 D，
  当前 `CD` 应保留；
- `res_b_019`：A 明确成立、D 被反证，至少还需 B/C 之一；当前 `AB`
  仍是证据更强的合法组合。

因此不能因为某个选项存在范围风险，就把这些题收缩成非法的单字母答案。

## 5. 替代情景：若四题集合 B 内有两错

严格历史约束允许：

```text
B = {fc_b_019, ins_b_006, ins_b_012, ins_b_016}
```

当前在 B 内恰好错两题。此时 97% 基线最可能三错组合应把
`fin_b_013` 替换为一题保险题：

```text
fin_b_005 / fc_b_019 / one_of(ins_b_006, ins_b_012, ins_b_016)
```

需要强调两点：

1. 纯方程并不强制 `fc_b_019` 一定是 B 内错题；理论上也允许两道保险题错误、
   `fc_b_019=ABCD` 正确。
2. 若 B 内确有两错，则对应历史结构为 `(p,m,z)=(2,0,2)`：
   两道当前错题的旧答案也都错，不能直接回滚旧值。

三道保险题目前的原文证据都很强，固定审计分别为 99、98、99，
不足以可靠排序。若必须选一个先重审，可从 `ins_b_006` 开始，
原因不是模型分更低，而是“恐怖活动/恐怖袭击”的同义口径最可能出现隐藏标签边界；
这仍只是审计顺序，不是修改建议。

## 6. 最终使用建议

| 用途 | 建议 |
|---|---|
| 下一份官网候选 | 仍优先 `fin_b_005: ACD→AC`；其 98% 结果尚待官网验证 |
| 下一轮本地证据审计 | `fc_b_019` 与三道保险题做官方题干字面盲审 |
| 若验证 `fin_b_005` 后确为 98% | 单题测试 `fc_b_019: ABCD→BD` 的信息增益最高 |
| 暂不建议直接提交 | `fin_b_013`，因为百分号组三题尚未完成单题归因 |

当前排序的核心不是“模型认为哪题可疑”，而是：

```text
直接原文反证强度
× 历史官网分数约束
× 单题消融可归因性
```

因此 `95 / 65 / 42` 只代表相对审计优先级和证据成熟度，不代表三题各自真实错误概率。
