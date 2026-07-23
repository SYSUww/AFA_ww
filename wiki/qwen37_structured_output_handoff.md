# Qwen3.7 Structured Output 交接说明

## 目的

降低 Qwen3.7 在 B 榜计算题中因 JSON 结构漂移造成的失败，同时保留证据
grounding、计算重放、答案冻结、完整 Token 记录和 answer/reasoning 两阶段解耦。

本说明只约束后续实现，不要求暂停、删除或修改已经启动的不可变运行。

## 已确认的当前状态

- `src/afa_agent/client.py` 的 `OpenAICompatibleClient.chat_json` 目前固定发送：

  ```json
  {"response_format": {"type": "json_object"}}
  ```

- `json_object` 只保证返回 JSON 对象，不能保证字段、类型、必填项和不同 `op`
  的参数结构符合执行器契约。
- 当前已观察到的结构漂移包括：
  - 常量使用 `value`，而执行器要求 `literal`；
  - 通用 `args` 返回对象而非数组；
  - `count_gte` 的 `items/threshold` 结构漂移；
  - `pct_change` 缺少 `new/old` 或使用错误字段；
  - 派生结果被重复声明为必须在证据中逐字出现的变量。
- 当前工作区已经存在确定性计划归一化框架，并在一次定向计算 badcase
  运行中让 14 道目标题中的 12 道保存了冻结答案产物。该结果是局部运行证据，
  不能外推为完整 100 题通过。
- 当前 OpenAI-compatible 服务是否原生支持
  `response_format.type=json_schema` 与 `strict=true` 尚未验证。不得在能力探针
  通过前把它写成已支持事实。

## 目标架构

```text
Qwen 生成 CalculationPlan
  -> 原生 strict JSON Schema（服务支持时）
     或 json_object + 本地 JSON Schema（不支持时）
  -> 确定性计划归一化
  -> Schema 再校验
  -> 证据 grounding
  -> 本地 Decimal 计算重放
  -> 冻结答案及证据 artifact
  -> 独立 reasoning 阶段
```

JSON Schema 负责结构，不负责判断证据真实性、单位来源、公式方向或答案正确性。
这些语义门禁必须继续由本地 grounding 和计算重放承担。

## 实施顺序

1. **先做最小能力探针**
   - 使用当前 Qwen3.7 固定快照和脱敏测试消息；
   - 测试当前服务是否接受 `json_schema + strict`；
   - 不输出 API Key、完整服务凭据或敏感响应；
   - 记录 HTTP 状态、是否遵循 Schema、模型名和原始 usage。

2. **扩展客户端但不静默降级**
   - 给 `chat_json` 增加可选的 `response_schema` 或独立方法；
   - 明确记录实际模式：
     - `native_json_schema_strict`
     - `json_object_local_schema`
   - 服务拒绝 strict Schema 时，可以显式选择本地校验模式，但不能把降级结果
     标记成原生 Structured Output。

3. **定义版本化 CalculationPlan Schema**
   - 顶层必填：
     `variables`、`steps`、`outputs`、`decision_summary`；
   - 使用 `additionalProperties: false`；
   - `op` 使用白名单枚举；
   - 为不同运算使用 `oneOf` 或等价分支：
     - 普通算术的 `args` 必须为数组；
     - `pct_change`、`pct_point_delta` 必须使用 `new/old`；
     - `count_gte`、`count_gt` 必须包含数组参数及阈值；
     - 常量节点只能使用 `literal`；
     - 输出格式只能使用执行器支持的枚举。

4. **保留确定性程序兜底**
   - 只允许不改变语义的规范化：
     - `value` 常量节点转 `literal`；
     - 无歧义具名参数转执行器参数结构；
     - 重复派生变量转已有 step 引用；
     - 删除不影响任何输出的节点；
     - 直接数值输出的纯数字 `text` 转 `decimal`。
   - 不得创造数值、交换运算方向、猜测单位、补写证据或修改答案。

5. **错误分流**
   - Schema 错误先走本地规范化或结构修复，不触发无意义的证据扩检索；
   - grounding 缺失才允许定向检索；
   - 公式或运算方向存在歧义时重新调用模型；
   - answer 阶段通过后立即冻结并持久化，reasoning 失败不得重跑答案阶段。

## Prompt 约束

即使使用 Structured Output，Prompt 仍需明确：

- `variables` 只能包含题目或证据逐字出现的原始数值、日期和单位；
- 差值、比率、合计、最大值、最小值、间隔天数等派生量必须写入 `steps`；
- 证据没有逐字单位时 `unit` 必须为空，不得根据常识补单位；
- 常量必须使用 `literal`；
- 禁止 `assign` 等执行器不支持的操作。

应给 Qwen 提供少量合法/非法 JSON 对照，避免继续依赖 GPT 系列模型能够理解的
隐式边界。

## 验收标准

- 能明确证明当前服务使用的是原生 strict Schema，或明确标记为本地 Schema 模式；
- 现有合法计划不发生答案、操作数顺序或单位语义漂移；
- 已知结构 badcase 的修复过程全部记录在
  `decision_trace.calculation_plan_normalizations`；
- 修复后的计划必须再次通过 Schema、grounding、Decimal 重放和答案格式校验；
- answer/reasoning 阶段产物、失败和 usage 分开持久化；
- 所有成功及失败 API 调用的
  `prompt_tokens + completion_tokens = total_tokens`；
- 不把理论 99% 参考答案注入 Qwen Prompt，也不以本地匹配冒充官网准确率；
- 每次实验先读取 `wiki/b_board_actual_loop_log.md`，每个方向最多三轮，并记录
  做法、效果、失败分析和下一步；
- 通过相关单测、完整测试、`compileall` 和 `git diff --check` 后，才可推送有效分支。

## 非目标

- Structured Output 不替代证据检索；
- Structured Output 不替代计算执行器；
- Structured Output 不证明隐藏答案正确；
- GPT-5.6 仍只用于冻结 reasoning 的离线影子评估，不进入答案生产链。
