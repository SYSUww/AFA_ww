# B榜题目说明

> 2026-07-23 更新：本文件中的八列提交示例已被同目录 [`new.md`](new.md) 的九列格式和新评分制度取代。仓库仅继续把旧 `submit.csv` 用作答案槽位模板，正式提交不得沿用旧表头。

## 任务背景

B榜包含金融合同、财务报告、保险、监管规则、研究报告等领域的问题。参赛者需要根据题目内容和给定材料，回答每道题对应的答案。

## 题目数据字段

每道题包含以下字段：

```json
{
  "qid": "",
  "domain": "",
  "split": "B",
  "question": "",
  "type": "",
  "options": {}
}
```

字段说明：

- `qid`：题目唯一编号。
- `domain`：题目所属领域。
- `split`：数据划分，B榜题目固定为 `B`。
- `question`：题目文本。
- `type`：题目类型，包括单选题、多选题、判断题、计算题或抽取题。
- `options`：选择题或判断题选项；计算题或抽取题该字段为空对象 `{}`。

## 作答要求

### 选择题和判断题

答案必须填写选项的大写字母。多选题必须填写所有正确选项，并按 `A`、`B`、`C`、`D` 的顺序连续书写，不使用逗号、空格或其他分隔符。

示例格式：

```text
ACD
```

### 计算题和抽取题

计算题和抽取题必须填写计算结果或抽取结果，格式要求如下：

- 题目问天数、笔数、分值等单位类结果时，只填写数字，不填写单位，保留两位小数。
- 百分数答案必须填写百分号 `%`，并保留两位小数。
- 日期答案必须使用中文日期格式：`YYYY年M月D日`，例如 `2026年1月1日`。
- 文本类答案必须按题目要求填写完整文本，不添加无关说明。
- 排序类答案中的大于号必须使用英文半角字符 `>`，前后不加空格。
- 多答案题必须按题目要求的顺序填写多个答案部分，不能乱序。

请参照 `submit.csv` 给出的全部题目的提交样例。

## 提交格式

提交文件必须使用 CSV 格式，请参照 `submit.csv` 的列结构和填写方式。表头必须为：

```csv
qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,completion_tokens,total_tokens
```

字段说明：

- `qid`：题目编号，必须与题目数据中的 `qid` 一致。
- `answer_1` 至 `answer_4`：答案字段。单答案题只填写 `answer_1`；多答案题按顺序填写多个字段，未使用的答案字段留空。
- `prompt_tokens`：该题推理过程消耗的 prompt token 数，必须为非负整数。
- `completion_tokens`：该题输出消耗的 completion token 数，必须为非负整数。
- `total_tokens`：该题总 token 数，必须为 `prompt_tokens` 与 `completion_tokens` 之和，且必须为非负整数。

提交文件必须包含一行 `summary` 汇总 token，并放在表头之后：

```csv
qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,completion_tokens,total_tokens
summary,,,,,123456,789,124245
```

`summary` 行中，`prompt_tokens`、`completion_tokens` 和 `total_tokens` 分别填写本次提交的总 prompt token 数、总 completion token 数和总 token 数，其中 `total_tokens` 必须等于 `prompt_tokens + completion_tokens`。

## 提交示例

以下示例仅展示格式，不代表真实题目或真实答案：

```csv
qid,answer_1,answer_2,answer_3,answer_4,prompt_tokens,completion_tokens,total_tokens
summary,,,,,123456,789,124245
example_b_001,ACD,,,,1000,20,1020
example_b_002,12.34%,56.78%,,,1200,30,1230
example_b_003,2026年1月1日,,,,900,15,915
example_b_004,100,,,,800,10,810
```
