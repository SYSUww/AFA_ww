# B 榜合规修复与总分优化执行日志

本日志只记录本轮 `codex/b-board-compliance-repair` 独立工作区的动作。官网准确率、
本地 pseudo99 匹配率、GPT-5.6 影子分和 Token 效率分必须分别标注，不得互相替代。

## 2026-07-25 启动记录

```json
{
  "experiment_id": "b-compliance-repair-loop-r0-baseline-freeze",
  "status": "in_progress",
  "branch": "codex/b-board-compliance-repair",
  "base_commit": "3d1d2ceb739db3525e814deb895549ce51cc6055",
  "history_reviewed": {
    "loop_log_sha256": "b2195b22bdfd8931c5c687e381b9ab06cdc6f88c9f677ca6441c2da77a015aaa",
    "registry_sha256": "5011c104139838647ff25a0498f08f62c08a519411abbd0e0f10f49cf7353dde",
    "new_md_sha256": "e59a7b25f4abc3fd456d16b06bd10e97435e71bdfd7794ce4af3f0829999ef19",
    "structured_output_handoff_sha256": "01eca8acd3951078d96d3f53cbe3a6f966034849bfd46097543bc9161cdbc9c1"
  },
  "baseline": {
    "artifact": "/Users/abandon/Documents/AFA_ww_compliance_choice/artifacts/b_board_actual/retrieval_llm_baseline/full100_submit_v1",
    "run_manifest_sha256": "ffe93dc4dd6816161baea69905ac13726c67faebf39c546a745c2054dc302589",
    "submit_sha256": "36cc0aa25ff878e3c7f1d43af8f7473be513f734e29d889fda851cfd25d39251",
    "question_count": 100,
    "answered_question_count": 100,
    "failed_question_count": 0,
    "raw_call_count": 172,
    "additional_call_count": 72,
    "total_tokens": 1067222,
    "token_efficiency_score": 78.65556,
    "pseudo99_equivalent_match_percent": 74.0,
    "official_accuracy": null,
    "strict_reasoning_score": null
  },
  "promotion_gate": {
    "formula": "accuracy_proxy * 0.5 + reasoning_shadow * 0.3 + token_efficiency * 0.2",
    "hard_rule": "candidate_total_score >= baseline_total_score",
    "equal_total_tie_breakers": [
      "lower_retry_count",
      "lower_failure_count",
      "fewer_audit_issues",
      "higher_stability"
    ],
    "note": "pseudo99 与 GPT-5.6 均为离线代理，不冒充官网得分。"
  },
  "parallel_tracks": [
    "calculation_freeze_usage_retry_fingerprint",
    "generic_retrieval_adjacent_chunk_merge",
    "strict_gpt56_reasoning_judge_v2",
    "total_score_non_regression_gate"
  ],
  "direction_attempt_limit": 3,
  "official_submission_count": 0
}
```

### A1 结果与 A2 假设

```json
{
  "experiment_id": "b-compliance-repair-choice-conclusion-equivalence-a1-full100-result",
  "direction": "generic_choice_conclusion_contract_normalization",
  "direction_attempt_count": 1,
  "status": "completed_not_promoted",
  "metrics": {
    "question_count": 100,
    "answered_question_count": 88,
    "failed_question_count": 12,
    "raw_call_count": 140,
    "format_consistency_retry_count": 3,
    "reasoning_only_retry_count": 37,
    "token_usage": {
      "prompt_tokens": 558110,
      "completion_tokens": 184299,
      "total_tokens": 742409
    },
    "token_efficiency_score": 85.15182,
    "pseudo99_equivalent_match_count": 61,
    "official_accuracy": null,
    "compliance_audit_passed": true,
    "unobservable_usage_risk": false
  },
  "relative_to_schema_v2": {
    "raw_call_delta": -31,
    "total_token_delta": -140501,
    "total_token_delta_percent": -15.91,
    "answered_question_delta": 2,
    "pseudo99_match_delta": -1
  },
  "promotion_result": "rejected_incomplete_and_below_retained_accuracy_proxy",
  "official_submission_count": 0
}
```

开始 A2 前再次复核日志与原始响应。`new.md` 要求 reasoning 对答案提供逻辑、
完整性和清晰度支撑，并未要求必须逐字以 `结论：...` 结束。最初的宽松离线方案
曾把 31 道无 marker 响应均视为可接受；独立复审发现其中包含“答案含 D、正文却
明确写 D 错误”等真实冲突，因此该投影作废，未进入模型实验。

A2 最终收窄为：所有题型都只有在同一响应末尾存在可机械解析、且与
answer_parts 一致的答案提示时才免重试；任意位置出现同值、子串命中或中间值
都不算一致性证据；普通“计算结果显示”等叙述也不视为答案提示。任何字段均不
追加、不修改。安全回放仅能避免 2 次调用、节省 5197 Token，不再宣称能够恢复
旧失败样本。
GPT-5.6 严格影子 Judge 的三个校准尝试均为 HTTP 502，因此该维度仍不可数值化，
不得用本地假分替代。

```json
{
  "experiment_id": "b-compliance-repair-choice-conclusion-equivalence-a2-final-safe-projection",
  "direction": "generic_choice_conclusion_contract_normalization",
  "direction_attempt_count": 2,
  "status": "implementation_review",
  "history_reviewed_before_attempt": true,
  "superseded_unsafe_projection": {
    "avoidable_calls": 31,
    "reason": "independent review found answer/reasoning semantic conflicts"
  },
  "safe_offline_projection_on_a1": {
    "avoidable_calls": 2,
    "avoidable_tokens": 5197,
    "projected_total_tokens": 737212,
    "projected_token_efficiency_score": 85.25576,
    "projected_answered_question_count": 88,
    "projected_pseudo99_equivalent_match_count": 61
  },
  "promotion_gate": "complete and observable; Token improvement cannot compensate a material accuracy or reasoning regression.",
  "official_accuracy": null,
  "official_submission_count": 0
}
```

### A1 全量运行

离线回放确认 Schema v2 的 29 道首次响应可避免 29 次 reasoning-only 调用，
对应已观测 107062 Token；149 项生产链专项测试、持久化及 assembler 回放通过，
独立审查为 GO。以下运行使用冻结提交与新目录。

```json
{
  "experiment_id": "b-compliance-repair-choice-conclusion-equivalence-a1-full100",
  "direction": "generic_choice_conclusion_contract_normalization",
  "direction_attempt_count": 1,
  "status": "running",
  "frozen_source_commit": "f8ec89b4e979ca642f7d2b82519ad5480dbed7e1",
  "run_config": {
    "calculation_mode": "direct",
    "document_candidate_strategy": "anchor_first",
    "evidence_quota_strategy": "primary_guard",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "workers": 8,
    "max_format_retries": 1
  },
  "run_dir": "artifacts/b_board_actual/compliance_repair/choice_conclusion_equivalence_a1_full100",
  "official_accuracy": null,
  "official_submission_count": 0
}
```

## 2026-07-25 Schema v2 全量复现结果与重试方向 A1

Schema 兼容修复通过真实 Qwen 验证：全部 26 道计算题均进入模型，不再出现
`regex_converter` HTTP 400。该生产候选复现仍未晋级，因为 14 题最终契约失败，
且 67 次 reasoning-only 调用显著抬高 Token。

```json
{
  "experiment_id": "b-compliance-repair-final-direct-anchor-first-full100-schema-v2-result",
  "status": "completed_not_promoted",
  "metrics": {
    "question_count": 100,
    "answered_question_count": 86,
    "failed_question_count": 14,
    "raw_call_count": 171,
    "format_consistency_retry_count": 4,
    "reasoning_only_retry_count": 67,
    "token_usage": {
      "prompt_tokens": 683116,
      "completion_tokens": 199794,
      "total_tokens": 882910
    },
    "token_efficiency_score": 82.3418,
    "pseudo99_equivalent_match_count": 62,
    "official_accuracy": null,
    "compliance_audit_passed": true,
    "unobservable_usage_risk": false
  },
  "same_scope_selected_a2": {
    "pseudo99_equivalent_match_count": 70,
    "total_tokens": 600848,
    "token_efficiency_score": 87.98304
  },
  "promotion_result": "rejected_incomplete_proxy_and_token_regression",
  "official_submission_count": 0
}
```

在开始下一个方向前已复核本日志，并离线审计两次全量运行的原始响应。上一轮
60 次 reasoning-only 中有 36 次仅因 `answer_parts=["AD"]` 与
`reasoning` 末尾 `结论：A；D` 的分隔符差异触发，消耗 149075 Token；
Schema v2 首次响应中同类情况有 29 题。A1 只在校验时把合法多选字母之间的
中文分号、顿号、逗号或空格视为等价，不修改答案或 reasoning，不接受不同字母
集合，也不注入 QID、公司、年份或参考答案。

```json
{
  "experiment_id": "b-compliance-repair-choice-conclusion-equivalence-a1",
  "direction": "generic_choice_conclusion_contract_normalization",
  "direction_attempt_count": 1,
  "status": "implementation",
  "history_reviewed_before_attempt": true,
  "promotion_gate": "complete and observable; material Token reduction may promote only when proxy accuracy and strict reasoning quality do not materially regress.",
  "official_accuracy": null,
  "official_submission_count": 0
}
```

## 2026-07-25 Qwen Schema 兼容修复后的 100 题复现

开始前已复核本日志。该运行仍是对已选中 `generic_anchor_first_document_candidates`
A2 的生产候选复现，不是第四次方向尝试。相对上一失败运行只修改 Schema
兼容性：删除服务不支持的负向前瞻，纯标点拒绝移入本地确定性校验；检索、
Prompt、模型、并发度、重试上限及输出契约均保持不变。

```json
{
  "experiment_id": "b-compliance-repair-final-direct-anchor-first-full100-schema-v2",
  "status": "running",
  "experiment_role": "promotion_reproduction_not_new_direction_attempt",
  "frozen_source_commit": "d24c196b777919e4624a67f1ac5232f2494607ab",
  "run_config": {
    "calculation_mode": "direct",
    "document_candidate_strategy": "anchor_first",
    "evidence_quota_strategy": "primary_guard",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "workers": 8,
    "max_format_retries": 1
  },
  "promotion_gate": "100/100 complete, all usage observable, compliance audit passes, and weighted proxy does not materially regress.",
  "run_dir": "artifacts/b_board_actual/compliance_repair/final_direct_anchor_first_full100_schema_v2",
  "official_accuracy": null,
  "official_submission_count": 0
}
```

## 2026-07-25 生产候选复现失败：Qwen 原生 Schema 不支持前瞻正则

本轮只复现已选中的 `generic_anchor_first_document_candidates` A2，不计为新的
方向尝试。运行目录和冻结源码均未在运行中修改。

```json
{
  "experiment_id": "b-compliance-repair-final-direct-anchor-first-full100-result",
  "status": "completed_not_promoted",
  "experiment_role": "promotion_reproduction_not_new_direction_attempt",
  "frozen_source_commit": "3c37915fc88168c7ed74079c40615fb506718b2c",
  "metrics": {
    "question_count": 100,
    "answered_question_count": 58,
    "failed_question_count": 42,
    "raw_call_count": 134,
    "transport_attempt_count": 160,
    "format_retry_count": 0,
    "call_purpose_counts": {
      "initial_answer": 74,
      "reasoning_only_retry_from_frozen_answer": 60
    },
    "token_usage": {
      "prompt_tokens": 525360,
      "completion_tokens": 143033,
      "total_tokens": 668393
    },
    "unobservable_usage_risk": true
  },
  "failure_analysis": {
    "native_schema_http_400": 26,
    "reasoning_missing_explicit_conclusion": 9,
    "answer_reasoning_conclusion_mismatch": 7,
    "root_cause": "DashScope JSON Schema regex converter rejects negative lookahead in calculation/extraction answer item pattern before generation."
  },
  "official_accuracy": null,
  "official_submission_count": 0,
  "promotion_result": "rejected_incomplete_and_unobservable",
  "next_step": "Remove the provider-incompatible regex from native Schema, preserve minLength, enforce punctuation-only rejection in deterministic local validation, then rerun in a fresh immutable directory."
}
```

## 2026-07-25 冻结答案、仅修 reasoning：F2 历史重试 9 题

F2 在每次新方向前复核了本日志，并沿用通用的选项主体文档约束检索；不按 QID、
题干、公司、年份或参考答案写规则。范围仅用于复现实验，不进入生成提示。

```json
{
  "experiment_id": "b-compliance-repair-frozen-reasoning-f2-retry9",
  "status": "completed_inconclusive",
  "direction_attempt_count": 2,
  "scope": {
    "question_count": 9,
    "source": "R2中曾触发格式或一致性重试的题目",
    "official_ground_truth": false
  },
  "metrics": {
    "answered_question_count": 9,
    "failed_question_count": 0,
    "raw_call_count": 14,
    "format_retry_count": 5,
    "call_purpose_counts": {
      "initial_answer": 9,
      "format_consistency_retry": 5,
      "reasoning_only_retry_from_frozen_answer": 0
    },
    "token_usage": {
      "prompt_tokens": 45889,
      "completion_tokens": 24691,
      "total_tokens": 70580
    },
    "pseudo99_equivalent_match": "4/9",
    "same_scope_r2": {
      "raw_call_count": 18,
      "format_retry_count": 9,
      "total_tokens": 92169
    }
  },
  "observations": [
    "候选完整且所有usage均来自provider原始字段",
    "相对R2同范围少4次调用、少21589 Token",
    "五次失败主要是reasoning缺少明确最终结论，而非答案与结论冲突",
    "冻结答案后的reasoning-only调用实际为0，因此不能把随机波动归因于该机制"
  ],
  "promotion_result": "not_promoted",
  "official_submission_count": 0
}
```

## 2026-07-25 冻结答案、仅修 reasoning：F3 最后一轮计划

```json
{
  "experiment_id": "b-compliance-repair-frozen-reasoning-f3-retry9",
  "status": "running",
  "direction_attempt_count": 3,
  "history_reviewed_before_attempt": true,
  "material_delta": "答案形状有效时，reasoning缺少明确结论或结论与答案不一致，均冻结初始Qwen答案并仅重生成reasoning",
  "safety_contract": {
    "frozen_answer_owner": "initial_qwen_response",
    "reasoning_owner": "qwen_reasoning_only_retry",
    "semantic_code_repair": false,
    "qid_or_reference_in_generation": false
  },
  "run_config": {
    "workers": 8,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "max_format_retries": 1
  },
  "promotion_gate": "9题完整、usage可观测、reasoning-only确实触发，且相对R2同范围Token下降同时代理准确率无大幅下降",
  "official_submission_count": 0
}
```

### F3 结果

```json
{
  "status": "completed_promising",
  "answered_question_count": 9,
  "failed_question_count": 0,
  "raw_call_count": 12,
  "format_retry_count": 3,
  "call_purpose_counts": {
    "initial_answer": 9,
    "reasoning_only_retry_from_frozen_answer": 3
  },
  "token_usage": {
    "prompt_tokens": 34976,
    "completion_tokens": 17335,
    "total_tokens": 52311
  },
  "pseudo99_equivalent_match": "4/9",
  "same_scope_r2": {
    "raw_call_count": 18,
    "format_retry_count": 9,
    "total_tokens": 92169,
    "pseudo99_equivalent_match": "4/9"
  },
  "relative_to_r2": {
    "raw_calls": "-33.33%",
    "total_tokens": "-43.24%",
    "pseudo99_match_delta": 0
  },
  "compliance_audit": "passed",
  "official_accuracy": null,
  "promotion_result": "provisional_pass_expand_to_full100",
  "caveat": "严格GPT-5.6影子reasoning分因provider HTTP 502仍不可用",
  "official_submission_count": 0
}
```

F3 是该方向的第三次也是最后一次材料性尝试。后续只做同一冻结实现的全量验证，
不再调整该方向机制；若全量不完整或其他维度大幅下降，则不晋级。

## 2026-07-25 组合候选全量 100 题验证

```json
{
  "experiment_id": "b-compliance-repair-full100-v1",
  "status": "running",
  "validated_components": [
    "通用选项主体文档约束与逐选项证据保留",
    "答案形状有效时冻结初始Qwen答案并仅修复reasoning结论",
    "模块化joint输出契约"
  ],
  "run_config": {
    "workers": 8,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "max_format_retries": 1
  },
  "promotion_gate": "100题完整、usage可观测、合规审计通过；代理准确率、Token、reasoning任一提升时其余维度不得大幅下降",
  "official_submission_count": 0
}
```

### 全量结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 94,
  "failed_question_count": 6,
  "raw_call_count": 127,
  "format_retry_count": 28,
  "call_purpose_counts": {
    "initial_answer": 99,
    "reasoning_only_retry_from_frozen_answer": 24,
    "format_consistency_retry": 4
  },
  "token_usage": {
    "prompt_tokens": 485859,
    "completion_tokens": 163551,
    "total_tokens": 649410
  },
  "token_efficiency_score": 87.0118,
  "pseudo99_equivalent_match": "61/100",
  "same_scope_baseline": {
    "answered_question_count": 100,
    "raw_call_count": 172,
    "total_tokens": 1067222,
    "token_efficiency_score": 78.65556,
    "pseudo99_equivalent_match": "74/100"
  },
  "compliance_audit": "passed",
  "unobservable_usage_risk": true,
  "unobservable_usage_cause": "fin_b_018 provider ReadTimeout before observable response",
  "promotion_result": "rejected",
  "reasons": [
    "candidate_incomplete",
    "pseudo_accuracy_declined_13_points",
    "one_provider_timeout_has_unobservable_usage_risk",
    "strict_reasoning_score_unavailable"
  ],
  "official_accuracy": null,
  "official_submission_count": 0
}
```

结论：冻结答案后的 reasoning-only 兜底显著减少重试 Token，但与选项文档约束检索
组合后，全量代理准确率下降过大。按照“单维提升不得让其他维度大幅下降”的门槛，
组合候选拒绝晋级；未通过全量门槛的选项文档约束检索不得成为生产默认值。

## 2026-07-25 恢复生产默认检索并复测失败簇

全量结果表明选项主体文档约束虽在 19 道历史错题样本上有效，但在 100 题上导致
代理匹配从 74 降到 61。代码因此恢复到全量基线使用的
`semantic_slots_entity_coverage_option_quota_doc_union_primary_guard_v5`；
未晋级方向只保留在日志与冻结实验中，不进入生产默认链路。

```json
{
  "experiment_id": "b-compliance-repair-legacy-retrieval-failure6-a1",
  "status": "running",
  "history_reviewed_before_attempt": true,
  "scope": {
    "question_count": 6,
    "source": "full100_v1最终失败簇",
    "official_ground_truth": false
  },
  "run_config": {
    "workers": 6,
    "retrieval_policy": "semantic_slots_entity_coverage_option_quota_doc_union_primary_guard_v5",
    "output_contract": "joint",
    "calculation_mode": "direct",
    "max_format_retries": 1
  },
  "goal": "区分失败是检索分布偏移还是仍需通用格式兜底",
  "official_submission_count": 0
}
```

### A1 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 4,
  "failed_question_count": 2,
  "raw_call_count": 12,
  "format_retry_count": 6,
  "token_usage": {
    "prompt_tokens": 43440,
    "completion_tokens": 19788,
    "total_tokens": 63228
  },
  "resolved_from_previous_failure_set": [
    "fin_b_016",
    "fin_b_018",
    "fin_b_019",
    "ins_b_003"
  ],
  "remaining_failure_types": {
    "reasoning_missing_explicit_conclusion": 1,
    "answer_slot_format_error": 1
  },
  "observations": [
    "旧检索使ins_b_003由失败恢复并得到366.00",
    "六题均触发第二次调用，仍有明确Token优化空间",
    "剩余两类失败均可用题型无关的输出契约处理，不需要QID规则"
  ],
  "promotion_result": "not_promoted_incomplete",
  "official_submission_count": 0
}
```

## 2026-07-25 全链路合规阻断修复与 H3 前置验收

本轮只修改、审计和测试代码，没有调用比赛模型，也没有提交官网。

```json
{
  "experiment_id": "b-compliance-repair-pre-h3-audit",
  "status": "preflight_passed_with_external_judge_blocked",
  "history_reviewed_before_attempt": true,
  "official_submission_count": 0,
  "model_call_count": 0,
  "official_accuracy": null,
  "changes": {
    "generation_source_closure": {
      "direct_loaded_source_files": 21,
      "verified_loaded_source_files": 28,
      "hardcoded_b_qid_count": 0,
      "prohibited_solver_import_count": 0,
      "prohibited_solver_module_loaded_count": 0
    },
    "direct_generation": [
      "首轮有效答案在模型响应后立即冻结，并在reasoning-only调用前独立持久化",
      "每次提供商调用前写入fsync call intent；恢复时遇到未匹配intent直接判定usage不可观测，不自动重发",
      "answer字段单独校验，拒绝标点占位；reasoning结论必须与冻结答案逐字一致",
      "format retry与reasoning-only调用分开计数"
    ],
    "verified_calculation": [
      "移除solver导入，只保留通用确定性计算执行器",
      "计划与reasoning调用均有pre-call intent和transport审计",
      "reasoning调用上限为2，停止路径与实际行为一致"
    ],
    "reasoning_judge": [
      "每个stage调用前持久化intent，checkpoint与ledger绑定随机run_instance_id",
      "复制输出目录、孤儿ledger、未匹配intent均失败关闭且不重发",
      "running清单可从canonical checkpoint重建派生镜像，complete清单严格验镜像"
    ],
    "research_isolation": [
      "document_balanced、adaptive_multi_report_calculation、metric_slot_coverage显式标为research-only",
      "research-only策略必须显式开关，且不能与primary_guard组合",
      "assembler复验运行绑定、源码指纹、冻结答案、调用intent、transport与research标志"
    ],
    "transport_usage": [
      "只对明确的HTTP 429、生成前零usage拒绝做transport重试",
      "保存transport_attempt_count与transport_rejections，并由runner、Judge、assembler共同复验"
    ]
  },
  "verification": {
    "focused_tests": {
      "passed": 176,
      "failed": 0
    },
    "full_repository_tests": {
      "passed": 499,
      "skipped": 1,
      "failed": 1,
      "failure": "独立worktree中的dataset_manifest绝对路径夹具仍指向原工作区；同一测试在原工作区通过，不属于本轮运行链路回归"
    },
    "compileall": "passed",
    "git_diff_check": "passed",
    "strict_gpt56_judge": "provider_http_502_unavailable"
  },
  "decision": "等待三路只读复核确认无P0/P1后提交冻结源码，再启动verified calculation H3；H3是该方向第3次也是最后一次尝试。"
}
```

### 第四轮独立复核与冻结前最终结果

```json
{
  "status": "go_pending_frozen_commit",
  "official_submission_count": 0,
  "model_call_count": 0,
  "official_accuracy": null,
  "independent_reviews": {
    "judge_ledger_and_recovery": "GO",
    "verified_schema_retry_and_assembler": "GO",
    "source_provenance_and_prompt_audit": "GO pending final negative-test confirmation"
  },
  "closed_findings": [
    "第二阶段timeout按started intent减observed stage计算不可观测风险，保留已观测usage且不重发",
    "并发失败镜像按qid和error_type稳定排序，逆序完成后可零调用恢复",
    "终态429失败transport与末尾call intent对账；timeout等不可观测失败源禁止进入最终组装",
    "verified真实reasoning结论使用专用契约；checkpoint SHA、运行绑定、证据绑定、Decimal replay和raw call前缀全部复验",
    "verified全局模式仅对计算题启用，非计算题按direct冻结答案契约审计",
    "run fingerprint统一排除created_at并由runner、evaluator、assembler共用",
    "Prompt审计覆盖模块Prompt、prompt/message函数及inline system message；真实direct和verified源码闭包扫描均为0",
    "组装器按call purpose计算格式重试，失败transport attempts/rejections计入最终审计"
  ],
  "verification": {
    "full_test_case_count": 509,
    "full_test_status": "passed_with_one_existing_skip",
    "compileall": "passed",
    "git_diff_check": "passed",
    "active_baseline_process": false,
    "strict_gpt56_remote_call": "not_run_provider_previously_returned_http_502"
  },
  "h3_contract": {
    "direction": "verified_calculation_full_coverage",
    "direction_attempt_count": 3,
    "document_candidate_strategy": "anchor_first",
    "evidence_quota_strategy": "primary_guard",
    "calculation_mode": "verified",
    "output_contract": "joint",
    "workers": 8,
    "max_format_retries": 1,
    "scope": "26 calculation questions",
    "run_dir_policy": "new_empty_immutable_directory"
  },
  "next_step": "提交并推送冻结源码；确认最终只读复核GO后启动H3。"
}
```

### H3 冻结快照

```json
{
  "status": "running",
  "experiment_id": "b-compliance-repair-verified-calculation-h3-anchor-first-calc26",
  "direction_attempt_count": 3,
  "history_reviewed_before_attempt": true,
  "frozen_source_commit": "3c37915fc88168c7ed74079c40615fb506718b2c",
  "frozen_source_remote": "origin/codex/b-board-compliance-repair",
  "frozen_source_tests": "passed",
  "official_submission_count": 0,
  "official_accuracy": null,
  "run_dir": "artifacts/b_board_actual/compliance_repair/verified_calculation_h3_anchor_first_calc26"
}
```

### H3 结果：verified calculation 第三轮关闭

```json
{
  "experiment_id": "b-compliance-repair-verified-calculation-h3-anchor-first-calc26",
  "status": "completed_not_promoted",
  "direction": "verified_calculation_full_coverage",
  "direction_attempt_count": 3,
  "direction_closed": true,
  "history_reviewed_before_attempt": true,
  "official_accuracy": null,
  "official_submission_count": 0,
  "metrics": {
    "question_count": 26,
    "answered_question_count": 24,
    "failed_question_count": 2,
    "raw_call_count": 50,
    "format_retry_count": 0,
    "transport_attempt_count": 50,
    "transport_rejection_count": 0,
    "token_usage": {
      "prompt_tokens": 312953,
      "completion_tokens": 97467,
      "total_tokens": 410420
    },
    "pseudo99_equivalent_match": "15/26",
    "pseudo99_equivalent_accuracy_percent": 57.692308,
    "sample_token_efficiency_score_not_full_run_comparable": 82.084,
    "compliance_audit_passed": true,
    "unobservable_usage_risk": false
  },
  "comparison_to_h2": {
    "answered_question_delta": 3,
    "pseudo99_match_delta": 3,
    "token_delta": 40695,
    "token_delta_percent": 11.00682
  },
  "failed_questions": [
    {
      "qid": "reg_b_016",
      "error_code": "calculation_decimal_replay_error",
      "error": "Unknown reference: 4999"
    },
    {
      "qid": "res_b_012",
      "error_code": "calculation_decimal_replay_error",
      "error": "sub amount unit mismatch: 亿元 vs 万元"
    }
  ],
  "pseudo99_mismatch_qids": [
    "fin_b_013",
    "fin_b_014",
    "fin_b_015",
    "fin_b_016",
    "fin_b_019",
    "ins_b_003",
    "ins_b_018",
    "ins_b_019",
    "reg_b_016",
    "res_b_005",
    "res_b_012"
  ],
  "promotion_result": "rejected",
  "reasons": [
    "candidate_incomplete",
    "pseudo_accuracy_materially_below_direct_baseline",
    "token_usage_increased_vs_h2",
    "strict_reasoning_score_unavailable"
  ],
  "decision": "不生成提交文件，不继续尝试verified calculation方向；后续回到direct生产链并优化通用检索、证据合并与首次调用质量。"
}
```

## 2026-07-25 A2 配置合规重放：最终 100 题候选

该运行不是 `generic_anchor_first_document_candidates` 的第四次方向试验，而是对该
方向已选中的 A2 配置，在完成 provenance、冻结答案、usage、intent、transport 和
Prompt 审计修复后做一次全量可复现验收。检索和生成超参数保持 A2 不变。

```json
{
  "experiment_id": "b-compliance-repair-final-direct-anchor-first-full100",
  "status": "running",
  "history_reviewed_before_attempt": true,
  "experiment_role": "promotion_reproduction_not_new_direction_attempt",
  "selected_direction": "generic_anchor_first_document_candidates",
  "selected_direction_attempt": 2,
  "frozen_source_commit": "3c37915fc88168c7ed74079c40615fb506718b2c",
  "run_config": {
    "document_candidate_strategy": "anchor_first",
    "evidence_quota_strategy": "primary_guard",
    "calculation_mode": "direct",
    "output_contract": "joint",
    "evidence_compaction": "off",
    "workers": 8,
    "max_format_retries": 1
  },
  "promotion_gate": "100/100完整、usage可观测、合规审计通过；相对旧A2的70/100 proxy和600848 Token不得显著退化。",
  "official_accuracy": null,
  "official_submission_count": 0,
  "run_dir": "artifacts/b_board_actual/compliance_repair/final_direct_anchor_first_full100"
}
```

## 2026-07-25 自适应多报告计算证据 A3：指标槽覆盖

本轮开始前已复核本日志中的 A1/A2。A3 只新增通用的“主体 × 指标 ×
数值承载片段”覆盖，不包含 QID、公司、年份或答案常量；这是该方向第 3
次且最后一次模型实验。

```json
{
  "experiment_id": "b-compliance-repair-adaptive-multi-report-quota-a3-metric-slots-result",
  "status": "completed_not_promoted",
  "direction": "adaptive_multi_report_calculation_evidence_quota",
  "direction_attempt_count": 3,
  "metrics": {
    "question_count": 3,
    "answered_question_count": 2,
    "failed_question_count": 1,
    "raw_call_count": 5,
    "format_retry_count": 2,
    "token_usage": {
      "prompt_tokens": 19240,
      "completion_tokens": 7899,
      "total_tokens": 27139
    },
    "pseudo99_equivalent_match": "2/3",
    "official_accuracy": null,
    "unobservable_usage_risk": false,
    "compliance_audit_passed": true
  },
  "per_question": {
    "fin_b_015": {
      "status": "answered",
      "answer": "宁德时代>美的集团;19.75",
      "raw_call_count": 1,
      "token_total": 3879,
      "pseudo99_equivalent": true
    },
    "fin_b_016": {
      "status": "failed",
      "raw_call_count": 2,
      "token_total": 9296,
      "failure": "两次模型响应均把四主体排序写成两个仅含大于号的槽位，未形成合法答案"
    },
    "fin_b_019": {
      "status": "answered",
      "answer": "比亚迪>宁德时代>美的集团;0.84",
      "raw_call_count": 2,
      "token_total": 13964,
      "pseudo99_equivalent": true
    }
  },
  "promotion_result": "rejected_incomplete",
  "reasons": [
    "candidate_incomplete",
    "strict_reasoning_score_unavailable",
    "metric value binding audit found possible wrong-context promotion"
  ],
  "decision": "该方向三轮已用完并关闭；metric_slot_coverage保持research-only，不进入默认生产链路。",
  "official_submission_count": 0
}
```

运行后只做了离线安全修复，没有发起第 4 次模型实验：指标名与值必须在局部
上下文中绑定，排除阈值/担保语境，删除表头 fallback/boost，并仅允许显式
多文档 research 策略启用。计算题 schema 也在 provider 层拒绝长度不足的
答案槽，避免把 `>` 当作合法主体结果。

## 2026-07-25 GPT-5.6 严格影子 Judge v2 第三次探针

A3 继续只评估 5 条人工哨兵 reasoning，不读取赛题、答案或检索证据。4 条
达到最短长度的哨兵均在调用前端返回 HTTP 502，provider usage 为 0；
短文本哨兵按固定规则直接记 0。校准未通过，因此本轮仍不能生成可信的
reasoning 分或最终总分，也不会以全 0 结果晋级候选。错误产物已脱敏，影子
Judge Token 不计入提交 Token。

## 2026-07-25 全链路合规阻断修复与 H3 前置验收

本阶段没有调用赛题生产模型、没有官网提交，也没有产生新的准确率。修复目标是
让下一次不可变实验的答案、reasoning、usage 和源码依赖都可复核。

```json
{
  "audit_id": "b-compliance-repair-provenance-usage-preflight-v2",
  "status": "passed_local_preflight_pending_parallel_review",
  "repairs": [
    "direct与verified生成入口均延迟加载旧插件；冷启动solver模块数为0",
    "源码指纹改为实际运行时afa_agent模块闭包，direct覆盖21个文件、verified覆盖28个文件",
    "两个闭包中固定B榜QID与solver import均为0",
    "固定公司/产品常量改为从题面动态抽取的通用模式",
    "research-only证据策略必须显式opt-in，组装器永久拒绝",
    "每次Qwen调用前原子写call intent；未观测响应时fail-closed且不重发",
    "429 pre-generation rejection与模型格式重试、reasoning-only调用分开计数",
    "direct答案字段合法后立即写独立冻结checkpoint；reasoning-only失败不改答案",
    "reasoning结论必须逐字等于按中文分号连接的answer_parts",
    "计算/抽取Schema拒绝仅由分隔符组成的槽位",
    "verified reasoning最多调用2次，超过预算不再发送",
    "assembler按call purpose重建reasoning-only最终payload并核验冻结答案、intent与provider usage",
    "GPT-5.6 Judge增加pre-call intent、随机run instance、路径绑定和运行态mirror重建"
  ],
  "tests": {
    "targeted_tests": "176 passed",
    "full_repository": "499 passed, 1 skipped, 1 pre-existing absolute-path fixture failure",
    "path_sensitive_fixture_check_in_original_root": "passed",
    "compileall": "passed",
    "git_diff_check": "passed"
  },
  "official_accuracy": null,
  "official_submission_count": 0
}
```

全仓唯一失败项是 `test_research_slices_v2`：其未跟踪数据清单写死
`/Users/abandon/Documents/AFA_ww`，因此在独立 worktree 生成绝对来源路径；
同一测试在原仓目录通过。该问题与 B 榜生产链无关，未通过修改断言掩盖。

## 2026-07-25 锚点优先文档候选 A1：历史正向差分 20 题

```json
{
  "experiment_id": "b-compliance-repair-anchor-first-docs-a1-positive20",
  "status": "planned",
  "direction": "generic_anchor_first_document_candidates",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "history_findings": {
    "current_union_strategy_pseudo99": "61/100",
    "historical_anchor_first_strategy_pseudo99": "74/100",
    "historical_correct_current_wrong_count": 20,
    "current_correct_historical_wrong_count": 7,
    "score_type": "offline pseudo99 equivalent match, not official accuracy"
  },
  "hypothesis": "题目主体或文档标题锚点已可靠命中文档时，不再混入全局BM25发现的其他高分文档，可减少跨主体、跨年份和跨合同污染；未命中锚点时仍回退全局发现。",
  "safety_contract": {
    "qid_or_reference_in_generation": false,
    "fixed_question_or_answer_mapping": false,
    "fixed_company_or_year_mapping": false,
    "generation_scope_qids_used_only_for_offline_evaluation": true,
    "model": "qwen3.7-plus-2026-05-26"
  },
  "scope": {
    "question_count": 20,
    "selection": "historical anchor-first correct and current union mismatch",
    "official_ground_truth": false
  },
  "promotion_gate": "20/20完成且usage可观测；相对同切片当前候选proxy有实质提升，Token或reasoning仅允许小幅下降；通过后才扩100题。",
  "official_submission_count": 0
}
```

### A1 结果与 A2 扩展

```json
{
  "status": "completed_promising",
  "metrics": {
    "answered_question_count": 20,
    "failed_question_count": 0,
    "raw_call_count": 23,
    "format_retry_count": 3,
    "token_usage": {
      "prompt_tokens": 92454,
      "completion_tokens": 40583,
      "total_tokens": 133037
    },
    "pseudo99_equivalent_match": "9/20",
    "official_accuracy": null,
    "unobservable_usage_risk": false,
    "compliance_audit_passed": true
  },
  "same_scope_current_anchor_union": {
    "answered_question_count": 19,
    "failed_question_count": 1,
    "raw_call_count": 23,
    "total_tokens": 136662,
    "pseudo99_equivalent_match": "0/20"
  },
  "relative_delta": {
    "pseudo99_match": "+9题",
    "total_tokens": "-2.65%",
    "completion": "+1题"
  },
  "interpretation": "A1在预先冻结的历史正向差分切片上同时提高proxy、完整性并小幅降低Token，但只复现旧运行20个历史匹配中的9个，不能把旧74/100当成本轮准确率，也不能排除切片选择偏差。",
  "promotion_result": "provisional_pass_expand_to_full100",
  "artifact_path": "artifacts/b_board_actual/compliance_repair/anchor_first_a1_positive20",
  "official_submission_count": 0
}
```

```json
{
  "experiment_id": "b-compliance-repair-anchor-first-docs-a2-full100",
  "status": "planned",
  "direction": "generic_anchor_first_document_candidates",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": "仅扩展评估范围到100题；检索、Prompt、Schema、重试和后处理均与A1冻结一致。",
  "promotion_gate": "100/100完整且usage可观测；相对anchor_union full100的61/100 proxy有实质提升，Token/reasoning不得大幅下降。",
  "official_submission_count": 0
}
```

### A2 结果与 A3 最后一轮

```json
{
  "status": "completed_promising",
  "metrics": {
    "answered_question_count": 100,
    "failed_question_count": 0,
    "raw_call_count": 108,
    "format_retry_count": 8,
    "token_usage": {
      "prompt_tokens": 434744,
      "completion_tokens": 166104,
      "total_tokens": 600848
    },
    "token_efficiency_score": 87.98304,
    "pseudo99_equivalent_match": "70/100",
    "official_accuracy": null,
    "unobservable_usage_risk": false,
    "compliance_audit_passed": true
  },
  "same_scope_anchor_union": {
    "answered_question_count": 99,
    "failed_question_count": 1,
    "raw_call_count": 107,
    "total_tokens": 563536,
    "token_efficiency_score": 88.72928,
    "pseudo99_equivalent_match": "61/100"
  },
  "weighted_proxy_delta_excluding_reasoning": {
    "accuracy_component": 4.5,
    "token_component": -0.149248,
    "net": 4.350752,
    "reasoning_drop_needed_to_erase_gain": 14.502507
  },
  "by_domain_match": {
    "financial_contracts": "16/20",
    "financial_reports": "6/20",
    "insurance": "13/20",
    "regulatory": "19/20",
    "research": "16/20"
  },
  "promotion_result": "provisional_pass_pending_strict_reasoning_judge",
  "artifact_path": "artifacts/b_board_actual/compliance_repair/anchor_first_a2_full100",
  "official_submission_count": 0
}
```

```json
{
  "experiment_id": "b-compliance-repair-anchor-first-docs-a3-balanced-evidence-full100",
  "status": "planned",
  "direction": "generic_anchor_first_document_candidates",
  "direction_attempt_count": 3,
  "history_reviewed_before_attempt": true,
  "material_delta": "在A2锚点优先文档不变的前提下，将TopK保留规则从primary前4+每文档1条改为每锚点文档2条+每选项1条；其余全部冻结。",
  "hypothesis": "多主体、多年份和多合同题需要每个候选文档至少两条互补证据；按文档平衡能减少同一文档占满前四位导致的跨主体证据缺失。",
  "safety_contract": {
    "qid_or_reference_in_generation": false,
    "fixed_question_or_answer_mapping": false,
    "fixed_company_or_year_mapping": false
  },
  "promotion_gate": "100/100完整、usage可观测，代理总分相对A2不下降；这是本方向第3轮，结束后无论结果均关闭方向。",
  "official_submission_count": 0
}
```

### A3 结果：方向关闭

```json
{
  "status": "completed_not_promoted",
  "metrics": {
    "answered_question_count": 100,
    "failed_question_count": 0,
    "raw_call_count": 106,
    "format_retry_count": 6,
    "token_usage": {
      "prompt_tokens": 421480,
      "completion_tokens": 159989,
      "total_tokens": 581469
    },
    "token_efficiency_score": 88.37062,
    "pseudo99_equivalent_match": "65/100",
    "official_accuracy": null,
    "unobservable_usage_risk": false,
    "compliance_audit_passed": true
  },
  "relative_to_a2": {
    "pseudo99_match": "-5题",
    "total_tokens": "-19379",
    "token_efficiency_score": 0.38758,
    "weighted_proxy_delta_excluding_reasoning": -2.422484
  },
  "effect": "全局每文档预留两条证据改善了fin_b_016与fin_b_019，但损害了保险、监管、研究和其他计算题；Token收益不足以抵消准确率代理下降。",
  "promotion_result": "rejected_accuracy_regression",
  "direction_status": "closed_after_three_attempts",
  "retained_candidate": "anchor_first_a2_full100",
  "artifact_path": "artifacts/b_board_actual/compliance_repair/anchor_first_a3_balanced_full100",
  "official_submission_count": 0
}
```

## 2026-07-25 多报告计算题自适应证据配额 A1

```json
{
  "experiment_id": "b-compliance-repair-adaptive-multi-report-quota-a1-slice3",
  "status": "planned",
  "direction": "adaptive_multi_report_calculation_evidence_quota",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "hypothesis": "只有金融年报计算题且锚点命中多个报告时，需要每个报告至少两条互补证据；其他题继续A2的primary guard，可保留Token和准确率。",
  "offline_frozen_replay": {
    "activation_count": 3,
    "activation_rule": "domain=financial_reports AND answer_format=calculation AND selected_doc_count>1",
    "a2_wrong_a3_correct": [
      "fin_b_016",
      "fin_b_019"
    ],
    "both_wrong": [
      "fin_b_015"
    ],
    "a2_correct_a3_wrong": [],
    "oracle_composed_proxy": "72/100"
  },
  "safety_contract": {
    "runtime_uses_qid": false,
    "runtime_uses_reference_answer": false,
    "fixed_company_or_year_mapping": false,
    "route_inputs": [
      "domain",
      "answer_format",
      "selected_doc_count"
    ]
  },
  "scope": {
    "question_count": 3,
    "selection": "all current questions satisfying the generic activation rule",
    "official_ground_truth": false
  },
  "promotion_gate": "3/3完整、usage可观测；相对A2同切片proxy不下降且至少修复1题后扩100题。",
  "official_submission_count": 0
}
```

### A1 结果与 A2 重跑

```json
{
  "status": "completed_not_promoted",
  "metrics": {
    "answered_question_count": 2,
    "failed_question_count": 1,
    "raw_call_count": 3,
    "format_retry_count": 1,
    "token_usage": {
      "prompt_tokens": 12138,
      "completion_tokens": 5222,
      "total_tokens": 17360
    },
    "pseudo99_equivalent_match": "1/3",
    "official_accuracy": null,
    "unobservable_usage_risk": true
  },
  "failure": {
    "qid": "fin_b_015",
    "type": "ReadTimeout",
    "read_timeout_seconds": 360,
    "provider_usage_observed": false
  },
  "completed_results": {
    "fin_b_019": "0.84, matches proxy",
    "fin_b_016": "66.85, does not match proxy"
  },
  "promotion_result": "rejected_incomplete_and_unobservable",
  "next_step": "A2使用新run目录原样重跑3题；不在不可观测A1内自动重发。",
  "official_submission_count": 0
}
```

```json
{
  "experiment_id": "b-compliance-repair-adaptive-multi-report-quota-a2-slice3",
  "status": "planned",
  "direction": "adaptive_multi_report_calculation_evidence_quota",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": "无算法变化，仅在全新不可变run中重跑因provider超时而不可判定的A1范围。",
  "promotion_gate": "3/3完整、usage可观测；相对A2生产候选同切片至少净修复1题。",
  "official_submission_count": 0
}
```

### A2 结果

```json
{
  "experiment_id": "b-compliance-repair-adaptive-multi-report-quota-a2-slice3-result",
  "status": "completed_not_promoted",
  "direction": "adaptive_multi_report_calculation_evidence_quota",
  "direction_attempt_count": 2,
  "metrics": {
    "answered_question_count": 2,
    "failed_question_count": 1,
    "raw_call_count": 2,
    "format_retry_count": 0,
    "token_usage": {
      "prompt_tokens": 8908,
      "completion_tokens": 3174,
      "total_tokens": 12082
    },
    "pseudo99_equivalent_match": "2/3",
    "official_accuracy": null,
    "unobservable_usage_risk": true
  },
  "completed_results": {
    "fin_b_019": "0.84, matches proxy",
    "fin_b_016": "76.92, matches proxy"
  },
  "failure": {
    "qid": "fin_b_015",
    "error_type": "ReadTimeout",
    "read_timeout_seconds": 360,
    "provider_usage_observed": false
  },
  "promotion_result": "rejected_incomplete_and_unobservable",
  "decision": "两轮均在fin_b_015产生不可观测超时，不做同配置第三次抽样；先修raw usage/不可变产物审计门禁，再以实体×指标槽覆盖作为具有实质变化的A3。",
  "official_submission_count": 0
}
```

### A3 计划：实体 × 指标槽覆盖

```json
{
  "experiment_id": "b-compliance-repair-adaptive-multi-report-quota-a3-metric-slots",
  "status": "planned",
  "direction": "adaptive_multi_report_calculation_evidence_quota",
  "direction_attempt_count": 3,
  "history_reviewed_before_attempt": true,
  "material_delta": "不再按每文档固定取前两条；从通用检索计划抽取计算子句前的披露指标，对每个anchor选中文档按年份×指标检索，并优先保留首条含实际数值而非仅表头的证据。",
  "safety_contract": {
    "qid_or_company_or_year_rule": false,
    "reference_or_official_lock_in_generation": false,
    "answer_semantic_postprocessing": false,
    "raw_ledger_append_only": true,
    "provider_usage_reconciled": true
  },
  "offline_preflight": {
    "fin_b_015_required_metric_slots": [
      "营业收入",
      "经营活动产生的现金流量净额"
    ],
    "coverage": "宁德时代和美的集团各自的营业收入与经营现金流数值均进入Top5",
    "model_calls": 0
  },
  "scope": {
    "question_count": 3,
    "selection": "所有满足financial_reports+calculation+多anchor文档的题目",
    "official_ground_truth": false
  },
  "promotion_gate": "3/3完整、usage全部可观测、严格无字段改写；相对anchor-first A2同范围至少净修复1题且Token或reasoning无大幅回归。",
  "official_submission_count": 0
}
```

## 2026-07-25 继续实验前的 provenance / usage 审计门禁

三路只读审计在模型实验结束后发现以下阻断项；在修复并通过回归测试前，不再启动
Qwen 候选实验，也不把现有 `compliance.passed` 当成完整合规证明：

- raw checkpoint 校验失败、线程未知异常或 verified 中间崩溃可能以空 `calls`
  覆盖已观测账本，低报 Token；
- 同一 run 目录缺少独占锁，重复 QID 也未拒绝，存在并发重复调用与 last-writer-wins；
- manifest 的 `all_observed_usage_from_provider_raw_fields` 是常量，评估器只对复制字段，
  没有逐调用核对 provider `raw_response.usage`；
- 默认 joint 校验会修改最终 answer / reasoning；与 Prompt 的“不会改写”承诺不一致，
  必须改为生产默认只接受模型原始可提交 payload，历史恢复逻辑仅保留显式 research-only；
- 静态合规扫描未覆盖真实 import graph，verified 重放、checkpoint usage、影子 Judge
  fingerprint / partial resume、promotion 非法输入均有验证缺口；
- 新增检索策略参数破坏三个旧测试 fixture，当前相关测试为 68 pass + 3 error。

修复验收条件：故障注入证明原始账本不被覆盖、所有 usage 逐级对账、默认生成字段不被
代码改写、并发 run 被拒绝、影子 Judge 与 promotion 输入严格校验、相关与全量测试通过。

## 2026-07-25 历史高匹配候选合规归因审计

只读审计确认，A13 的 96/100 与三题 overlay 后的 99/100 都是本地
pseudo99 匹配，不是官网准确率。A13 100题中有56题证据 metadata 带历史定向
规则标志，包含预写 True/False、固定题干、固定公司/年份/数字或固定产品 bundle；
这些机制不进入本分支的生成链路。三道计算题的底层能力——直接披露汇总值识别、
目标报告年份绑定、保险条款的 `max` 运算符绑定——是答案盲通则，可作为后续新方向；
但旧候选按固定 QID 覆盖三题，不能直接复用。未发现 pseudo99 或官网 locks 进入
Qwen Prompt；它们只存在于冻结结果后的评估层。

## 2026-07-25 模型字段组装与数值格式兜底 A1

本轮在开始前复核本日志。reasoning-only 严格 Schema 改为由 Qwen 同时返回
`reasoning` 正文和枚举锁定的 `conclusion`，代码只机械拼接两个模型字段；
计算/抽取槽仅对纯数字做单位剥离、ASCII 百分号和题面要求精度格式化。
若 `answer_parts` 结构无效但同一次 Qwen reasoning 有合法明确结论，允许从同一响应
恢复答案字段；答案字段本身有效时绝不改用 reasoning 覆盖，且所有转换写入 provenance。

```json
{
  "experiment_id": "b-compliance-repair-contract-recovery-a1-failure2",
  "status": "running",
  "history_reviewed_before_attempt": true,
  "direction_attempt_count": {
    "model_field_reasoning_assembly": 1,
    "deterministic_contract_recovery": 1
  },
  "scope": {
    "question_count": 2,
    "source": "旧检索A1稳定剩余失败类型",
    "official_ground_truth": false
  },
  "safety_contract": {
    "qid_or_reference_rules": false,
    "semantic_correction": false,
    "shape_valid_answer_remains_authoritative": true,
    "same_response_recovery_only": true
  },
  "promotion_gate": "2/2完整、usage可观测、合规审计通过，且调用数和Token不高于A1同范围",
  "official_submission_count": 0
}
```

### A1 结果

```json
{
  "status": "completed_promising",
  "answered_question_count": 2,
  "failed_question_count": 0,
  "raw_call_count": 3,
  "format_retry_count": 1,
  "token_usage": {
    "prompt_tokens": 7873,
    "completion_tokens": 4601,
    "total_tokens": 12474
  },
  "same_scope_previous": {
    "answered_question_count": 0,
    "raw_call_count": 4,
    "total_tokens": 17723,
    "pseudo99_equivalent_match": "0/2"
  },
  "pseudo99_equivalent_match": "1/2",
  "relative_delta": {
    "raw_calls": "-25.00%",
    "total_tokens": "-29.62%",
    "pseudo99_match": "+1"
  },
  "compliance_audit": "passed",
  "reasoning_quality_risk": "fin_b_014正文出现冻结答案和反推等流程元话语",
  "promotion_result": "numeric_normalization_pass; reasoning_assembly_needs_prompt_a2",
  "official_submission_count": 0
}
```

### reasoning 模型字段组装 A2 计划

```json
{
  "experiment_id": "b-compliance-repair-reasoning-assembly-a2-fin014",
  "status": "running",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": "禁止冻结答案、指定结论、前一阶段、提示词、数据集、测试、反推等流程元话语；证据不足只能客观说明缺失事实",
  "scope": {
    "question_count": 1,
    "official_ground_truth": false
  },
  "promotion_gate": "完整、usage可观测、reasoning无流程元话语且Token不显著上升",
  "official_submission_count": 0
}
```

### A2 结果与 A3 最后一轮

A2 完整并通过结构契约，但 reasoning 仍出现“指定结论、反推、强制输出”等流程
元话语，且 Token 从 A1 的 8462 增至 8644，拒绝晋级。A3 不再把结论值写入消息
正文，只通过原生严格 JSON Schema 的单值枚举传递；提示改为只写证据事实、计算、
缺失指标和自然收束。这是该方向第三次也是最后一次材料性尝试。

```json
{
  "experiment_id": "b-compliance-repair-reasoning-assembly-a3-fin014",
  "status": "running",
  "direction_attempt_count": 3,
  "history_reviewed_before_attempt": true,
  "material_delta": "结论值仅存在于严格Schema枚举，不在messages正文出现；采用正向证据摘要模板",
  "promotion_gate": "完整、usage可观测、reasoning无流程元话语，Token相对A1增幅不超过5%",
  "official_submission_count": 0
}
```

### A3 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 1,
  "failed_question_count": 0,
  "raw_call_count": 2,
  "total_tokens": 9139,
  "process_meta_language_present": false,
  "quality_problems": [
    "reasoning过长",
    "使用检索证据之外的常识和数值",
    "证据不足时仍产生大段探索性文本"
  ],
  "promotion_result": "rejected_reasoning_quality_and_token_increase",
  "direction_closed_after_three_material_attempts": true,
  "official_submission_count": 0
}
```

## 2026-07-25 同响应 reasoning 结论组装 A1

当 Qwen 首轮 `answer_parts` 通过完整答案形状校验、reasoning 正文不少于 20 字且
仅缺少显式结论时，代码把同一响应的模型答案机械附加为
`结论：<answer_parts>`。不重新检索、不调用第二个模型、不改变答案语义；
若 reasoning 已有不同结论，则仍进入冻结答案后的 reasoning-only 模型调用。

```json
{
  "experiment_id": "b-compliance-repair-same-response-conclusion-a1",
  "status": "running",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "scope": {
    "question_count": 2,
    "source": "fin_b_014与ins_b_003历史缺结论样本",
    "official_ground_truth": false
  },
  "safety_contract": {
    "shape_valid_answer_required": true,
    "reasoning_missing_conclusion_only": true,
    "semantic_correction": false,
    "answer_reasoning_mismatch_not_assembled": true
  },
  "promotion_gate": "2/2完整、每题仅1次调用、合规审计通过、代理答案不下降",
  "official_submission_count": 0
}
```

### A1 结果

```json
{
  "status": "completed_expand_to_full100",
  "answered_question_count": 2,
  "failed_question_count": 0,
  "raw_call_count": 2,
  "format_retry_count": 0,
  "token_usage": {
    "prompt_tokens": 7728,
    "completion_tokens": 5178,
    "total_tokens": 12906
  },
  "historical_same_scope": {
    "raw_call_count": 4,
    "total_tokens": 21915,
    "pseudo99_equivalent_match": "1/2"
  },
  "relative_delta": {
    "raw_calls": "-50.00%",
    "total_tokens": "-41.11%"
  },
  "pseudo99_equivalent_match": "0/2",
  "compliance_audit": "passed",
  "causal_accuracy_note": "代码未改变首轮Qwen答案；样本代理下降来自不同首轮响应，但仍需全量门槛验证",
  "promotion_result": "not_yet_promoted_expand_scope",
  "official_submission_count": 0
}
```

## 2026-07-25 生产默认检索 + 通用契约兜底全量验证 V2

```json
{
  "experiment_id": "b-compliance-repair-full100-v2-legacy-retrieval",
  "status": "running",
  "history_reviewed_before_attempt": true,
  "validated_components": [
    "生产默认v5通用检索",
    "同响应缺结论机械组装",
    "数值槽单位与精度确定性格式化",
    "仅在真实结论冲突时触发reasoning-only"
  ],
  "run_config": {
    "workers": 8,
    "output_contract": "joint",
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "max_format_retries": 1
  },
  "promotion_gate": "100题完整、usage可观测、合规审计通过；相对74/100代理准确率不得大幅下降，Token需明显改善",
  "official_submission_count": 0
}
```

### V2 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 99,
  "failed_question_count": 1,
  "raw_call_count": 107,
  "format_retry_count": 7,
  "call_purpose_counts": {
    "initial_answer": 100,
    "reasoning_only_retry_from_frozen_answer": 5,
    "format_consistency_retry": 2
  },
  "postprocessing_question_counts": {
    "none": 68,
    "same_response_conclusion_assembly": 24,
    "model_field_assembly": 5,
    "deterministic_format_normalization": 2,
    "same_response_contract_recovery": 1
  },
  "token_usage": {
    "prompt_tokens": 399370,
    "completion_tokens": 164166,
    "total_tokens": 563536
  },
  "token_efficiency_score": 88.72928,
  "pseudo99_equivalent_match": "61/100",
  "same_scope_baseline": {
    "answered_question_count": 100,
    "raw_call_count": 172,
    "total_tokens": 1067222,
    "token_efficiency_score": 78.65556,
    "pseudo99_equivalent_match": "74/100"
  },
  "relative_delta": {
    "raw_calls": "-37.79%",
    "total_tokens": "-47.20%",
    "token_efficiency_score": "+10.07372",
    "pseudo99_match": "-13"
  },
  "compliance_audit": "passed",
  "unobservable_usage_risk": false,
  "promotion_result": "rejected_accuracy_decline_and_incomplete",
  "official_accuracy": null,
  "official_submission_count": 0
}
```

结论：Token 机制有效，但 13 点代理准确率下降超过允许范围，不能晋级生产提交。
下一方向必须回到证据召回与排序；使用历史高准确率证据只做离线诊断，生成链路仍
不加载参考答案、QID规则或官方锁。

### A2 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 20,
  "failed_question_count": 6,
  "raw_call_count": 39,
  "format_retry_count": 14,
  "token_usage": {
    "prompt_tokens": 163948,
    "completion_tokens": 73325,
    "total_tokens": 237273
  },
  "pseudo99_equivalent_match": "8/26",
  "official_accuracy": null,
  "unobservable_usage_risk": true,
  "failure_analysis": {
    "answer_reasoning_conclusion_mismatch": 3,
    "answer_slot_format_error": 2,
    "provider_http_500_unknown_usage": 1,
    "merged_components": "393 -> 338",
    "conclusion": "限制传递合并略增代理匹配，但没有改善完成率或追加调用，不能晋级。"
  },
  "promotion_result": "rejected",
  "next_step": "相邻合并方向暂止于A2，不浪费第三轮；另开verified_calculation_full_coverage H2，隔离计算答案冻结与reasoning解耦效果。"
}
```

## 2026-07-25 Verified Calculation H2：全 26 道计算题

```json
{
  "experiment_id": "b-compliance-repair-verified-calculation-h2-calc26",
  "status": "running",
  "direction": "verified_calculation_full_coverage",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "prior_attempt": "verified_calculation_full_coverage_h1_calc26",
  "scope": {
    "question_count": 26,
    "calculation_mode": "verified",
    "evidence_compaction": "off",
    "workers": 6
  },
  "contract": [
    "Qwen CalculationPlan",
    "local schema normalization",
    "evidence grounding",
    "Decimal replay",
    "immediate frozen checkpoint",
    "reasoning-only Qwen call"
  ],
  "source_sha256": {
    "runner": "4e5e1523471d0cdc062b678bdbacfe341e4569e5dec66bf5f9ccae747bef4ba6",
    "verified_calculation": "eeeb58ca8af359bcb5c7ea037848c705d4e29f0a5b93e0c67914b6061071a3f3",
    "calculation_executor": "4fe6b4e3f80bc4b388420fd0030f4308a5bbb9400e50b0809ed11e02ef8e7745",
    "calculation_schema": "391b4df78fa605a105361fec462cce55cf78c71ddbf0e687241fcb8af8218cd4",
    "reasoning_schema": "43df93d89abe1f135ef03dc58861b0b1c6e385a85b54d8218acb7edd5f8090ad",
    "semantic_runner": "1911858924ed20db65771d64bd3a43e7acb0512534aa63775248d88ce9957a04",
    "client": "581ff494ef0b8c7e8e7aad33599303d105c71adb7d5c215896768b9ee850dd81"
  },
  "official_submission_count": 0
}
```

### A1 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 21,
  "failed_question_count": 5,
  "raw_call_count": 39,
  "format_retry_count": 14,
  "token_usage": {
    "prompt_tokens": 163113,
    "completion_tokens": 73376,
    "total_tokens": 236489
  },
  "pseudo99_equivalent_match": "7/26",
  "official_accuracy": null,
  "unobservable_usage_risk": true,
  "failure_analysis": {
    "answer_reasoning_conclusion_mismatch": 4,
    "read_timeout_with_unknown_usage": 1,
    "compaction_input_hits": 777,
    "output_blocks": 260,
    "merged_blocks": 77,
    "merged_components": 393,
    "root_cause": "同一parent的相邻chunk发生传递闭包，单块平均合入过多片段；prepare阶段从块前缀截断1800字，可能丢失原高排名锚点。"
  },
  "promotion_result": "rejected",
  "reasons": [
    "candidate_incomplete",
    "unobservable_usage_risk",
    "strict_reasoning_score_unavailable"
  ],
  "next_step": "A2把每个合并块限制为最多2个component，并围绕原始最高排名chunk取邻居；所有剩余component仍作为独立块参与TopK。"
}
```

## 2026-07-25 选项实体文档约束 R1：A1 五道失败题

```json
{
  "experiment_id": "b-compliance-repair-option-doc-scope-r1-failed5",
  "status": "running",
  "direction": "option_entity_document_scoping",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "offline_diagnosis": [
    "旧融合器先保留4个primary与最多6个document hit，Top10常在加入option hit之前耗尽。",
    "选项检索虽包含选项文本，但跨全部候选文档排序，可能把另一产品的同名条款误当作当前选项证据。",
    "focused query只使用完整命题atom，丢掉题干中的主题、期间、范围与例外词。"
  ],
  "generic_change": [
    "从选项文本抽取主体，先定位候选文档，再在该文档范围检索该选项。",
    "focused query同时使用命题atom与主题/期间/范围组合，不读取QID或历史答案。",
    "TopK先保留2个primary，再按选项各保留1个证据，最后补文档覆盖与融合得分。"
  ],
  "scope": {
    "question_count": 5,
    "source": "reasoning-canonical A1 response-contract失败题",
    "official_ground_truth": false
  },
  "run": {
    "workers": 5,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "max_format_retries": 1
  },
  "promotion_gate": "完整且usage可观测；与同题旧基线相比代理准确率或重试/Token至少一项明显改善，其他维度不得大幅下降。",
  "official_submission_count": 0
}
```

### R1 结果

```json
{
  "status": "completed_promising",
  "answered_question_count": 5,
  "failed_question_count": 0,
  "raw_call_count": 6,
  "actual_retry_count": 1,
  "token_usage": {
    "prompt_tokens": 17183,
    "completion_tokens": 10812,
    "total_tokens": 27995
  },
  "same_scope_baseline": {
    "raw_call_count": 9,
    "actual_retry_count": 4,
    "total_tokens": 50288,
    "pseudo99_equivalent_match": "0/5"
  },
  "relative_delta": {
    "raw_calls": "-33.33%",
    "total_tokens": "-44.33%",
    "pseudo99_equivalent_match": "+3题"
  },
  "pseudo99_equivalent_match": "3/5",
  "official_accuracy": null,
  "unobservable_usage_risk": false,
  "promotion_result": "provisional_pass_expand_scope",
  "remaining_mismatches": [
    "fin_b_007",
    "ins_b_004"
  ],
  "official_submission_count": 0
}
```

## 2026-07-25 选项实体文档约束 R2：19 道多选 badcase

```json
{
  "experiment_id": "b-compliance-repair-option-doc-scope-r2-choice19",
  "status": "running",
  "direction": "option_entity_document_scoping",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": "仅扩大样本，不修改R1检索与输出契约。",
  "scope": {
    "question_count": 19,
    "source": "full100_submit_v1中与pseudo99不一致的多选题",
    "official_ground_truth": false
  },
  "run": {
    "workers": 8,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "joint",
    "max_format_retries": 1
  },
  "official_submission_count": 0
}
```

### R2 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 18,
  "failed_question_count": 1,
  "raw_call_count": 28,
  "actual_retry_count": 9,
  "token_usage": {
    "prompt_tokens": 93822,
    "completion_tokens": 44434,
    "total_tokens": 138256
  },
  "same_scope_baseline": {
    "answered_question_count": 19,
    "raw_call_count": 34,
    "actual_additional_call_count": 15,
    "total_tokens": 195766,
    "pseudo99_equivalent_match": "0/19"
  },
  "relative_delta": {
    "raw_calls": "-17.65%",
    "total_tokens": "-29.38%",
    "pseudo99_equivalent_match": "+8题"
  },
  "pseudo99_equivalent_match": "8/19",
  "official_accuracy": null,
  "unobservable_usage_risk": false,
  "promotion_result": "rejected_for_incompleteness",
  "failure": "reg_b_001 answer_parts与reasoning末尾结论连续两次不一致",
  "next_step": "R3用已修复的reasoning单一事实源消除双写冲突；检索配置保持R2不变。",
  "official_submission_count": 0
}
```

## 2026-07-25 集成 R3：选项文档约束 + reasoning 单一事实源

```json
{
  "experiment_id": "b-compliance-repair-integrated-r3-choice19",
  "status": "running",
  "direction": "option_scope_plus_reasoning_canonical",
  "direction_attempt_count": {
    "option_entity_document_scoping": 3,
    "reasoning_canonical_output_contract": 2
  },
  "history_reviewed_before_attempt": true,
  "material_delta_from_r2": "仅将joint双字段切换为修复后的reasoning-canonical单字段；检索不变。",
  "scope": {
    "question_count": 19,
    "official_ground_truth": false
  },
  "run": {
    "workers": 8,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "reasoning-canonical",
    "max_format_retries": 1
  },
  "official_submission_count": 0
}
```

### R3 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 16,
  "failed_question_count": 3,
  "raw_call_count": 22,
  "actual_retry_count": 3,
  "token_usage": {
    "prompt_tokens": 72173,
    "completion_tokens": 36781,
    "total_tokens": 108954
  },
  "same_scope_baseline_total_tokens": 195766,
  "relative_token_delta": "-44.34%",
  "pseudo99_equivalent_match": "8/19",
  "official_accuracy": null,
  "unobservable_usage_risk": false,
  "promotion_result": "rejected",
  "reason": "单字段消除了双写冲突并显著降Token，但fc_b_007、ins_b_010、reg_b_001均连续给出单选式结论，违反多选至少两项，候选不完整。",
  "decision": "reasoning-canonical不作为默认生产契约；保留研究实现与日志。生产链继续采用joint首答，并研究仅在answer/reasoning冲突时冻结答案后补写reasoning。",
  "official_submission_count": 0
}
```

## 2026-07-25 外部资料碰撞

- Qwen 官方结构化输出文档指出：Schema 会随每个模型轮次重复传输，且每次
  校验重试都是完整模型轮次，因此应保持 Schema 简单并尽量首轮命中。
- Qwen Cloud/阿里云文档建议明确描述字段、类型与示例，并在 JSON object
  模式下做本地 Schema 校验；本项目已有原生 strict Schema 实测，因此继续保留
  原生模式，同时避免通过额外模型做语义 JSON 修复。
- ACL 2025 的 set-wise retrieval 工作强调复杂问题需要让最终 passage 集合共同覆盖
  多个信息需求，而不是仅按单段相关性取 TopK；这与本轮“每个选项保留证据位”的
  R1/R2 实测方向一致。

可执行的新方向不是继续扩大 Schema，而是：joint 首答若答案形状合法但 reasoning
结论不一致，立即冻结模型自产答案，只重生成 reasoning；不得再次运行答案阶段。

## 2026-07-25 冻结答案后仅补 reasoning F1：reg_b_001

```json
{
  "experiment_id": "b-compliance-repair-frozen-reasoning-f1-reg001",
  "status": "running",
  "direction": "frozen_answer_reasoning_only_retry",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "hypothesis": "当joint首答的answer_parts形状合法、仅与reasoning末尾结论冲突时，冻结该Qwen自产答案并用第二次Qwen调用只生成reasoning，可避免答案阶段重跑和答案漂移。",
  "safety_contract": {
    "answer_source": "initial_qwen_answer_parts",
    "answer_rerun": false,
    "reasoning_call_can_modify_answer": false,
    "reasoning_call_thinking": false,
    "semantic_code_repair": false
  },
  "scope": {
    "question_count": 1,
    "source": "R2唯一response-contract失败题",
    "official_ground_truth": false
  },
  "official_submission_count": 0
}
```

### F1 结果

本次首答本身一致，未触发 fallback：1/1 完成、1 次调用、4,056 Token，
答案为 `AD`，与 pseudo99 的 `ACD` 不一致。该结果只能证明不需要重试时不会
增加调用，不能证明 reasoning-only fallback 的线上效果。

## 2026-07-25 冻结答案后仅补 reasoning F2：历史重试 9 题

```json
{
  "experiment_id": "b-compliance-repair-frozen-reasoning-f2-retry9",
  "status": "running",
  "direction": "frozen_answer_reasoning_only_retry",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": "机制不变；扩大到R2中发生过第二次调用的9题。",
  "scope": {
    "question_count": 9,
    "official_ground_truth": false
  },
  "run": {
    "workers": 8,
    "output_contract": "joint",
    "max_format_retries": 1
  },
  "official_submission_count": 0
}
```

## 2026-07-25 相邻证据合并 A2：限制传递合并

```json
{
  "experiment_id": "b-compliance-repair-adjacent-merge-a2-mismatch26",
  "status": "running",
  "direction": "generic_adjacent_evidence_compaction",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "material_delta": {
    "max_components_per_block": 2,
    "split_strategy": "around_best_original_rank",
    "all_unselected_components_retained": true,
    "workers": 6
  },
  "source_sha256": {
    "runner": "4e5e1523471d0cdc062b678bdbacfe341e4569e5dec66bf5f9ccae747bef4ba6",
    "compaction": "303fab1b876aad99d2d74860e82456d2f047553552c7a38da8d193dd792598c6",
    "baseline_pipeline": "b56307669d69687f6c2f46c8bcb438be1e8cdf7ade752419f252d22f5e452b6d",
    "query_generator": "fc334ae2ae992fd879e7ca6ffce0a6eeac7bc2c79de3438a66359d124b2a002d"
  },
  "official_submission_count": 0
}
```

## 2026-07-25 基线追加调用离线归因

对冻结的 `full100_submit_v1` 172 次调用做离线回放，不调用模型、不修改答案：

| 追加调用原因 | 次数 | Token | 本轮处理 |
|---|---:|---:|---|
| 可由非语义格式/确定性校验消除 | 29 | 182,126 | 代码兜底或放宽纯机械门禁 |
| answer-reasoning / 计算语义冲突 | 8 | 49,660 | 保留一次内容重试，不由代码裁决 |
| 证据不足 | 27 | 212,932 | 提升新增证据在首轮检索中的位次 |
| 结构中途闭合等其他输出问题 | 8 | 62,006 | Prompt 完整性约束，必要时重调 |
| 合计 | 72 | 506,724 | 目标是尽量一次完成，仍保留审计安全兜底 |

离线估算：若只安全消除 29 次非语义追加调用，调用数可由 172 降至 143，
总 Token 可由 1,067,222 降至约 885,096。该数值是回放估算，不是新运行实测。

## 2026-07-25 相邻证据合并 A1：26 道历史 mismatch 样本

```json
{
  "experiment_id": "b-compliance-repair-adjacent-merge-a1-mismatch26",
  "status": "running",
  "direction": "generic_adjacent_evidence_compaction",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "hypothesis": "从三倍深度候选池中，将同文档真实相邻且章节兼容的表头+数据行、条件+条款、定义+例外合并为一个TopK块，可让二轮补充证据在首轮进入上下文，同时减少重叠文本。",
  "scope": {
    "question_count": 26,
    "source": "full100_submit_v1 pseudo99 mismatch set",
    "official_ground_truth": false
  },
  "run": {
    "workers": 8,
    "calculation_mode": "direct",
    "evidence_compaction": "adjacent",
    "compaction_pool_multiplier": 3,
    "final_top_k": 10,
    "max_format_retries": 1
  },
  "source_sha256": {
    "runner": "861cd73d1fd80b78cd099fcc145e07723d872783e2e9464eb65e50107b57ac7a",
    "compaction": "972ce9e580ed46c56653d166bf11e0b0e8cdf7ade752419f252d22f5e452b6d",
    "baseline_pipeline": "b56307669d69687f6c2f46c8bcb438be1e8cdf7ade752419f252d22f5e452b6d",
    "query_generator": "fc334ae2ae992fd879e7ca6ffce0a6eeac7bc2c79de3438a66359d124b2a002d"
  },
  "promotion_gate": "weighted proxy total must not decline",
  "official_submission_count": 0
}
```

## 2026-07-25 Qwen3.7 原生 Schema 能力探针

```json
{
  "experiment_id": "b-compliance-repair-qwen37-schema-probe-a1",
  "status": "completed",
  "scope": "deidentified_capability_probe",
  "model": "qwen3.7-plus-2026-05-26",
  "temperature": 0.0,
  "response_format_mode": "native_json_schema_strict",
  "schema_adherent": true,
  "token_usage": {
    "prompt_tokens": 46,
    "completion_tokens": 472,
    "total_tokens": 518
  },
  "contains_competition_question_or_answer": false,
  "included_in_submission_token_usage": false,
  "artifact": "artifacts/b_board_actual/compliance_repair/qwen37_schema_probe_v1.json",
  "official_submission_count": 0
}
```

## 2026-07-25 计算题 verified H2：26 题完整覆盖

```json
{
  "experiment_id": "b-compliance-repair-verified-calculation-h2-calc26",
  "status": "completed_not_promoted",
  "direction": "verified_calculation_full_coverage",
  "direction_attempt_count": 2,
  "history_reviewed_before_attempt": true,
  "metrics": {
    "answered_question_count": 21,
    "failed_question_count": 5,
    "raw_call_count": 47,
    "actual_retry_count": 0,
    "call_purpose_counts": {
      "calculation_plan": 26,
      "verified_calculation_reasoning": 21
    },
    "token_usage": {
      "prompt_tokens": 285855,
      "completion_tokens": 83870,
      "total_tokens": 369725
    },
    "pseudo99_equivalent_match": "12/26",
    "official_accuracy": null,
    "unobservable_usage_risk": false
  },
  "failure_analysis": {
    "financial_reports_match": "0/8",
    "primary_issue": "严格grounding、单位和值类型校验拒绝了5题；已通过校验的计划仍存在年份、口径和公式绑定错误。",
    "manifest_note": "旧manifest把21次强制reasoning调用误计为format retry；现已按call purpose修复，真实重试为0。"
  },
  "promotion_result": "rejected",
  "reasons": [
    "candidate_incomplete",
    "pseudo_accuracy_declined_materially",
    "strict_reasoning_score_unavailable"
  ],
  "decision": "默认计算模式继续保持direct；不以verified H2生成提交文件。",
  "official_submission_count": 0
}
```

## 2026-07-25 GPT-5.6 严格影子 Judge v2 校准

两次校准均只对人工哨兵 reasoning 做影子评测，不读取答案和证据，不计入提交
Token。A1 使用原生严格 Schema，A2 使用本地 JSON 约束；两次均在服务端返回
HTTP 502，四个需要模型评测的哨兵全部 `judge_error`，可观测 usage 均为 0。
因此当前只能确认失败时按 0 分处理的保守契约，不能声称 Judge v2 已完成数值校准，
也不能用其结果晋级任何候选。

## 2026-07-25 reasoning 单一事实源 A1：选择题

```json
{
  "experiment_id": "b-compliance-repair-reasoning-canonical-a1-choice19",
  "status": "running",
  "direction": "reasoning_canonical_output_contract",
  "direction_attempt_count": 1,
  "history_reviewed_before_attempt": true,
  "hypothesis": "只让Qwen返回最终reasoning，并从其末尾结论原样抽取答案，可消除answer_parts与reasoning双写冲突，同时减少输出Schema和完成Token。",
  "safety_contract": {
    "model_owned_field": "reasoning",
    "answer_source": "unchanged_model_reasoning_conclusion",
    "semantic_code_repair": false,
    "qid_or_reference_in_generation": false
  },
  "scope": {
    "question_count": 19,
    "source": "full100_submit_v1中与pseudo99不一致的多选题",
    "official_ground_truth": false
  },
  "run": {
    "workers": 6,
    "calculation_mode": "direct",
    "evidence_compaction": "off",
    "output_contract": "reasoning-canonical",
    "max_format_retries": 1
  },
  "promotion_gate": "候选必须完整、usage可观测；代理总分不下降，或Token明显下降且准确率/推理质量仅小幅波动。",
  "official_submission_count": 0
}
```

### A1 结果

```json
{
  "status": "completed_not_promoted",
  "answered_question_count": 14,
  "failed_question_count": 5,
  "raw_call_count": 26,
  "actual_retry_count": 7,
  "token_usage": {
    "prompt_tokens": 88257,
    "completion_tokens": 48683,
    "total_tokens": 136940
  },
  "same_scope_baseline": {
    "raw_call_count": 34,
    "total_tokens": 195766
  },
  "relative_delta": {
    "raw_calls": "-23.53%",
    "total_tokens": "-30.05%"
  },
  "pseudo99_equivalent_match": "5/19",
  "official_accuracy": null,
  "unobservable_usage_risk": false,
  "promotion_result": "rejected",
  "reasons": [
    "candidate_incomplete",
    "five_response_contract_failures",
    "strict_reasoning_score_unavailable"
  ],
  "audit_findings": [
    "canonical重试提示仍引用joint契约，导致多选被误写为多槽",
    "raw response checkpoint在进程恢复时未被重载，可能重复调用并覆盖usage"
  ],
  "next_step": "先修两个审计阻断问题并增加崩溃恢复测试；不得直接把A1结果用于提交。",
  "official_submission_count": 0
}
```
