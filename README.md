# AFAC2026 赛道四：金融长文本 Agent 的动态记忆压缩与高效问答

## 赛题简介

AFAC2026 赛道四，聚焦金融长文档问答任务。参赛者需要在**不修改基座模型参数**的前提下，设计 Agent 层面的记忆流转、动态压缩和上下文优化策略。

### 五类金融文本

| 领域 | 文档数 | A 组题数 |
|------|--------|----------|
| insurance（保险条款） | 16 | 20 |
| regulatory（监管法规） | 26 | 20 |
| financial_contracts（金融合同） | 14 | 20 |
| financial_reports（财务报表） | 10 | 20 |
| research（行业研报） | 20 | 20 |

### 题目类型

- **单选题（mcq）**：从 A/B/C/D 中选择唯一正确答案
- **多选题（multi）**：从 A/B/C/D 中选择所有正确答案，按字母顺序排列（如 ABC）
- **判断题（tf）**：A/B 表示正确/错误

### 评测指标

```
FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)
TokenScore = max(0, min(1, (5,000,000 - TotalTokens) / 5,000,000))
```

准确率决定主体得分，Token 效率最多影响 30% 的加权系数。



## 关键约束

- 不得修改基座模型参数
- 不得使用其他开源或闭源模型替代官方指定模型
- Token 预算：5,000,000

## 参考链接

- [赛题页面](https://tianchi.aliyun.com/competition/entrance/532486/information)