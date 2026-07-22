# B 榜新评分制度迁移

更新时间：2026-07-23（Asia/Shanghai）

状态：代码契约与本地回归已迁移；当前模型按用户要求暂时保持 `gpt-5.5`，尚未生成白名单模型的新制度提交产物。

## 1. 当前规则

规则来源：[`upload_b/new.md`](../upload_b/new.md)。

```text
总分 = accuracy × 0.6 + reasoning × 0.2 + token_efficiency × 0.2
```

- 正式 CSV 至少包含 `qid,answer1,answer2,answer3,answer4,prompt_tokens,completion_tokens,total_tokens,reasoning`。
- `summary` 行建议保留；若存在，必须与逐题 usage 求和一致。
- reasoning 少于 20 字直接计 0，且需要能具体支持 answer。
- 只允许 Qwen3.5/Qwen3.6 系列；usage 必须来自 API 原始响应，相关重试、验证、纠错和摘要调用都要逐题累计。
- Token 分：`0 -> 0`；`1..499999` 线性递增；`500000..5000000 -> 100`；`5000001..10000000` 线性递减；再高为 0。

## 2. 已落地的工程保护

- B 榜 writer 固定生成九列 CSV；旧八列 `upload_b/submit.csv` 只作为答案槽位模板兼容读取。
- 通用 validator 接受额外列和可选 summary；生产 runner/merge 使用 audit-ready 模式，拒绝短 reasoning 和零 usage。
- 每题增加最终 reasoning 生成调用，要求白名单模型保持答案不变，并把该次 usage 加入本题。
- API 客户端在每次成功响应后立即写入按题调用账本；runner 以账本汇总覆盖求解器内部聚合，避免 JSON 重试失败导致 usage 丢失。
- 新 runner 启动时拒绝非 Qwen3.5/Qwen3.6 模型；旧 checkpoint 因 runner/prompt 指纹升级不能安全续跑。
- composite merge 校验源 run 模型、逐次调用账本、逐题合计和 reasoning，旧 GPT/零 Token 产物不会被标记为可提交。
- 本地提供新 Token 分段函数和 60/20/20 总分计算，覆盖所有区间边界。

## 3. 旧产物结论

- 提交 006 的外部准确率为 94%，总 Token 为 `982494`。
- 用户反馈提交 007 的平台准确率为 **97%**，相对提交 006 提升 3 个百分点；对应文件总 Token 为 `931605`，Token 维度处于 100 分区间。
- 提交 007 的五道变化题合计净增 3 题，但不能仅凭总准确率断言每道题各自正确；当前答案序列可作为新的外部回归基线。
- 提交 007 仍是旧八列、无提交 reasoning、源模型 `gpt-5.5`，且 answers 中有 47 题为零 Token，因此不能直接包装成新制度提交。
- 如果只用于公式估算：97% accuracy、Token 分 100、reasoning 分 0 时，总分为 `78.2`。这不是平台新制度总分实测结果。

## 4. 新执行策略

1. 当前按用户要求保留 `.env` 的 `gpt-5.5` 配置，不修改 endpoint、key 或模型；在此状态下不宣称生成物满足新白名单。
2. 以 97% 文件作为答案回归基线，继续做本地证据审计和 reasoning/usage 工程验证，但不把研究产物标记为新制度提交就绪。
3. 准备正式新制度提交时，再切换到出题方允许的 Qwen3.5/Qwen3.6 服务，并使用全新 run 目录，不复用旧 answers/checkpoint，也不手工补 reasoning 或 Token。
4. 正式运行前先做小样本，检查逐题调用账本、reasoning 质量、answer 不变约束和 CSV 审计，再授权全量付费运行。
5. Token 总量目标保持在 50 万至 500 万的满分区间；不为凑 Token 人为填充调用，也不再把低于 50 万视为当然更优。
6. 如果外部 evaluator/纠错模型的结果影响最终答案，必须把相关逐题 usage 合并到最终账本；未完成归集前，其 composite 只能用于研究，不能标记为提交就绪。

## 5. 验证命令

```bash
env PYTHONPATH=src /opt/miniconda3/envs/afa-autoresearch/bin/python -m unittest discover -s tests -p 'test_b_board*.py' -v
env PYTHONPATH=src /opt/miniconda3/envs/afa-autoresearch/bin/python -m unittest discover -s tests -v
python3 -m compileall -q src scripts
git diff --check
```
