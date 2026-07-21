## B0-actual

- recorded_at: `2026-07-21T18:19:02+00:00`

```json
{
  "artifact_paths": {
    "answers": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/b0_actual_gpt55_attempt43_composite_v1/answers.json",
    "run_manifest": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/b0_actual_gpt55_attempt43_composite_v1/run_manifest.json",
    "submission": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/b0_actual_gpt55_attempt43_composite_v1/submit.csv"
  },
  "event": "b0_complete",
  "experiment_id": "B0-actual",
  "question_count": 100,
  "recorded_at": "2026-07-21T18:19:02+00:00",
  "status": "complete",
  "token_usage": {
    "completion_tokens": 181468,
    "prompt_tokens": 1728663,
    "total_tokens": 1910131
  }
}
```

## B0-actual-integrity

- recorded_at: `2026-07-21T18:19:37+00:00`

```json
{
  "event": "b0_integrity_correction",
  "experiment_id": "B0-actual-integrity",
  "reason": "Earlier b0_complete entry reported the expected submit path instead of the manifest value; this append-only correction is authoritative.",
  "recorded_at": "2026-07-21T18:19:37+00:00",
  "status": "submission_invalid",
  "submission_path": null,
  "submission_valid": false,
  "submission_validation_failures": [
    {
      "error": "fc_b_005.answer_2: percentage slot requires exactly two decimals and '%' suffix",
      "qid": "fc_b_005"
    },
    {
      "error": "fin_b_013.answer_1: numeric/date slot requires two-decimal number or valid Chinese date",
      "qid": "fin_b_013"
    },
    {
      "error": "fin_b_015.answer_1: ordering slot requires non-empty labels joined by half-width '>'",
      "qid": "fin_b_015"
    },
    {
      "error": "fin_b_017.answer_2: numeric/date slot requires two-decimal number or valid Chinese date",
      "qid": "fin_b_017"
    },
    {
      "error": "fin_b_019.answer_1: ordering slot requires non-empty labels joined by half-width '>'",
      "qid": "fin_b_019"
    },
    {
      "error": "res_b_005.answer_1: numeric/date slot requires two-decimal number or valid Chinese date",
      "qid": "res_b_005"
    }
  ]
}
```

## b0-run-integrity-reg016-v2

- recorded_at: `2026-07-21T18:20:34+00:00`

```json
{
  "answered_question_count": 0,
  "change_vector": {
    "material_delta": "count_threshold_reference_support",
    "strategy": "run_integrity"
  },
  "direction_id": "run_integrity",
  "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
  "evaluator_model": "gpt-5.5",
  "experiment_id": "b0-run-integrity-reg016-v2",
  "generator_model": "gpt-5.5",
  "hypothesis": "Calculation failures must preserve diagnostics and official slot rendering must be deterministic.",
  "material_delta": {
    "implementation": "count_threshold_reference_support"
  },
  "pipeline_stage": "runtime",
  "reason": "Output format requested by the model did not satisfy the official two-decimal slot.",
  "recorded_at": "2026-07-21T18:20:34+00:00",
  "root_cause_cluster": "integrity",
  "run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/repairs/reg_b_016_count_threshold_v2",
  "run_status": "incomplete",
  "status": "blocked_technical",
  "target_qids": [
    "reg_b_016"
  ],
  "token_accounting_complete": false,
  "token_usage": {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

## b0-run-integrity-reg016-v3

- recorded_at: `2026-07-21T18:20:34+00:00`

```json
{
  "answered_question_count": 0,
  "change_vector": {
    "material_delta": "official_slot_schema_override",
    "strategy": "run_integrity"
  },
  "direction_id": "run_integrity",
  "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
  "evaluator_model": "gpt-5.5",
  "experiment_id": "b0-run-integrity-reg016-v3",
  "generator_model": "gpt-5.5",
  "hypothesis": "Calculation failures must preserve diagnostics and official slot rendering must be deterministic.",
  "material_delta": {
    "implementation": "official_slot_schema_override"
  },
  "pipeline_stage": "runtime",
  "reason": "A subsequent model plan omitted a decimal input; failure diagnostics were not yet persisted.",
  "recorded_at": "2026-07-21T18:20:34+00:00",
  "root_cause_cluster": "integrity",
  "run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/repairs/reg_b_016_slot_schema_v3",
  "run_status": "incomplete",
  "status": "blocked_technical",
  "target_qids": [
    "reg_b_016"
  ],
  "token_accounting_complete": false,
  "token_usage": {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

## b0-run-integrity-reg016-v4

- recorded_at: `2026-07-21T18:20:34+00:00`

```json
{
  "answered_question_count": 1,
  "change_vector": {
    "material_delta": "failure_diagnostics_and_slot_schema",
    "strategy": "run_integrity"
  },
  "direction_id": "run_integrity",
  "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
  "evaluator_model": "gpt-5.5",
  "experiment_id": "b0-run-integrity-reg016-v4",
  "generator_model": "gpt-5.5",
  "hypothesis": "Calculation failures must preserve diagnostics and official slot rendering must be deterministic.",
  "material_delta": {
    "implementation": "failure_diagnostics_and_slot_schema"
  },
  "pipeline_stage": "runtime",
  "reason": "Generated a grounded, replayable 2.00 answer and restored 100/100 artifact coverage.",
  "recorded_at": "2026-07-21T18:20:34+00:00",
  "root_cause_cluster": "integrity",
  "run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/repairs/reg_b_016_diagnostic_v4",
  "run_status": "complete",
  "status": "accepted",
  "target_qids": [
    "reg_b_016"
  ],
  "token_accounting_complete": true,
  "token_usage": {
    "completion_tokens": 2171,
    "prompt_tokens": 29173,
    "total_tokens": 31344
  }
}
```

## b-loop-calculation_executor-a1-typed_grounded_calc_v2

- recorded_at: `2026-07-21T18:22:24+00:00`

```json
{
  "attempt_index": 1,
  "direction_id": "calculation_executor",
  "event": "attempt_planned",
  "experiment_id": "b-loop-calculation_executor-a1-typed_grounded_calc_v2",
  "history_decision": "execute",
  "history_reason": "No comparable historical experiment was found",
  "material_delta": {
    "calculation_trace_schema": 2,
    "deterministic_slot_format": true,
    "threshold_refs": true
  },
  "pipeline_stage": "calculation",
  "recorded_at": "2026-07-21T18:22:24+00:00",
  "root_cause_cluster": "calculation",
  "status": "running",
  "target_qids": [
    "fc_b_005",
    "fin_b_013",
    "fin_b_015",
    "fin_b_017",
    "fin_b_019",
    "res_b_005"
  ]
}
```

## B0-actual-evaluation

- recorded_at: `2026-07-21T18:26:43+00:00`

```json
{
  "evaluated_answer_count": 99,
  "event": "evaluation_invalid",
  "experiment_id": "B0-actual-evaluation",
  "failure_count": 1,
  "recorded_at": "2026-07-21T18:26:43+00:00",
  "sentinel_validation": {
    "failures": [],
    "passed": true,
    "sentinel_count": 6
  },
  "status": "invalid"
}
```

## b-loop-calculation_executor-a1-typed_grounded_calc_v2

- recorded_at: `2026-07-21T18:29:05+00:00`

```json
{
  "answered_qids": [
    "fin_b_017",
    "res_b_005"
  ],
  "attempt_index": 1,
  "candidate_confidence": {
    "by_domain": {
      "financial_reports": {
        "median": 0,
        "minimum": 0,
        "p10": 0,
        "question_count": 1
      },
      "research": {
        "median": 0,
        "minimum": 0,
        "p10": 0,
        "question_count": 1
      }
    },
    "median": 0,
    "minimum": 0,
    "p10": 0,
    "question_count": 2,
    "tiers": {
      "blocked": 2
    }
  },
  "candidate_fingerprint": {
    "components": {
      "context": {
        "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
        "evaluator_model": "gpt-5.5",
        "generator_model": "gpt-5.5"
      },
      "identity": {
        "change_vector": {
          "grounding_required": true,
          "slot_schema": "official_template",
          "strategy": "typed_grounded_calc_v2"
        },
        "domains": [],
        "hypothesis": "typed evidence grounding plus deterministic official slot rendering removes invalid calculation answers.",
        "pipeline_stage": "calculation",
        "question_types": [
          "calculation"
        ],
        "root_cause_cluster": "calculation",
        "target_qids": [
          "fc_b_005",
          "fin_b_013",
          "fin_b_015",
          "fin_b_017",
          "fin_b_019",
          "res_b_005"
        ]
      }
    },
    "context_sha256": "fb8ccbeedeb2b9d54db9a8a389961275c1cf689059baaf6a3b8445af28fb8748",
    "direction_sha256": "96fbffc02ba6fcf1cb517889811514e5cff3690e3c85ba763701662958fce3cc",
    "schema_version": 1,
    "semantic_sha256": "ae24de61863d0c569cd209f5c0782575e4ca535de7f6d5a84edcae00743cc265",
    "sha256": "31acb6021d1c6777c32d3809780c0aa97facf4a7adff6bbd847dc6d1c94c3256"
  },
  "change_vector": {
    "grounding_required": true,
    "slot_schema": "official_template",
    "strategy": "typed_grounded_calc_v2"
  },
  "direction_id": "calculation_executor",
  "evaluation_token_usage": {
    "completion_tokens": 4247,
    "prompt_tokens": 8043,
    "total_tokens": 12290
  },
  "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
  "evaluator_model": "gpt-5.5",
  "experiment_id": "b-loop-calculation_executor-a1-typed_grounded_calc_v2",
  "failed_qids": [
    "fc_b_005",
    "fin_b_013",
    "fin_b_015",
    "fin_b_019"
  ],
  "generation_token_usage": {
    "completion_tokens": 11958,
    "prompt_tokens": 99294,
    "total_tokens": 111252
  },
  "generator_model": "gpt-5.5",
  "hypothesis": "Typed evidence grounding plus deterministic official slot rendering removes invalid calculation answers.",
  "material_delta": {
    "calculation_trace_schema": 2,
    "deterministic_slot_format": true,
    "threshold_refs": true
  },
  "next_root_causes": [
    "calculation_unit_semantics",
    "calculation_directionality",
    "iterative_variable_retrieval"
  ],
  "pipeline_stage": "calculation",
  "promotion_result": "rejected",
  "question_types": [
    "calculation"
  ],
  "reason": "4/6 targets lacked required evidence; both generated answers were contradicted by the fixed evaluator.",
  "recorded_at": "2026-07-21T18:29:05+00:00",
  "registry_schema_version": 1,
  "root_cause_cluster": "calculation",
  "run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/candidates/calculation_executor/typed_grounded_calc_v2_invalid6",
  "status": "rejected",
  "target_qids": [
    "fc_b_005",
    "fin_b_013",
    "fin_b_015",
    "fin_b_017",
    "fin_b_019",
    "res_b_005"
  ]
}
```

## B0-actual-evaluation

- recorded_at: `2026-07-21T18:29:10+00:00`

```json
{
  "artifact_paths": {
    "confidence_audit": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/b0_actual_gpt55_attempt43_composite_v1/evaluation/confidence_audit.json",
    "evaluator_manifest": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/b0_actual_gpt55_attempt43_composite_v1/evaluation/evaluator_manifest.json"
  },
  "confidence": {
    "low_confidence_qids": [
      "fc_b_001",
      "fc_b_002",
      "fc_b_003",
      "fc_b_004",
      "fc_b_005",
      "fc_b_006",
      "fc_b_007",
      "fc_b_008",
      "fc_b_009",
      "fc_b_010",
      "fc_b_011",
      "fc_b_012",
      "fc_b_014",
      "fc_b_015",
      "fc_b_017",
      "fc_b_018",
      "fc_b_020",
      "fin_b_001",
      "fin_b_002",
      "fin_b_003",
      "fin_b_004",
      "fin_b_005",
      "fin_b_008",
      "fin_b_009",
      "fin_b_010",
      "fin_b_012",
      "fin_b_013",
      "fin_b_014",
      "fin_b_015",
      "fin_b_016",
      "fin_b_017",
      "fin_b_018",
      "fin_b_019",
      "fin_b_020",
      "ins_b_001",
      "ins_b_003",
      "ins_b_006",
      "ins_b_007",
      "ins_b_008",
      "ins_b_011",
      "ins_b_013",
      "ins_b_015",
      "ins_b_016",
      "ins_b_018",
      "ins_b_019",
      "reg_b_001",
      "reg_b_003",
      "reg_b_004",
      "reg_b_007",
      "reg_b_008",
      "reg_b_014",
      "reg_b_018",
      "reg_b_021",
      "reg_b_024",
      "reg_b_025",
      "reg_b_026",
      "reg_b_027",
      "res_b_002",
      "res_b_003",
      "res_b_004",
      "res_b_005",
      "res_b_006",
      "res_b_007",
      "res_b_008",
      "res_b_009",
      "res_b_010",
      "res_b_011",
      "res_b_012",
      "res_b_014",
      "res_b_016",
      "res_b_017",
      "res_b_018",
      "res_b_019",
      "res_b_020"
    ],
    "minimum": 0,
    "p10": 0,
    "question_count": 100,
    "tiers": {
      "blocked": 70,
      "high": 12,
      "low": 4,
      "medium": 14
    }
  },
  "dynamic_direction_ids": [
    "dynamic_calculation_1759e98acc",
    "dynamic_chunking_2f1d16baa9",
    "dynamic_citation_86ea677bf3",
    "dynamic_decision_dfb5a6283c",
    "dynamic_evidence_12f3abc51b",
    "dynamic_format_46ca269ec7",
    "dynamic_parsing_99264930f7",
    "dynamic_retrieval_3f6ba9077e",
    "dynamic_stability_efa72de843",
    "dynamic_unknown_eeb29bdf45"
  ],
  "event": "evaluation_complete",
  "experiment_id": "B0-actual-evaluation",
  "recorded_at": "2026-07-21T18:29:10+00:00",
  "registry": {
    "direction_attempt_counts": {
      "calculation_executor": 1,
      "manifest_0a1189f82e2d": 1,
      "run_integrity": 3,
      "unknown": 5
    },
    "experiment_count": 10,
    "statuses": {
      "accepted": 1,
      "blocked_technical": 2,
      "legacy_transferable": 6,
      "rejected": 1
    }
  },
  "status": "complete"
}
```

## B0-actual-root-cause-analysis

- recorded_at: `2026-07-21T18:30:11+00:00`

```json
{
  "baseline_evaluation_tokens": 557089,
  "blocked_hard_failure_count": 62,
  "blocked_semantic_count": 8,
  "confidence_tiers": {
    "blocked": 70,
    "high": 12,
    "low": 4,
    "medium": 14
  },
  "event": "b0_root_cause_analysis",
  "experiment_id": "B0-actual-root-cause-analysis",
  "hard_failure_counts": {
    "calculation_not_grounding_verified": 25,
    "format_forced": 19,
    "invalid_answer_slot_1": 4,
    "invalid_answer_slot_2": 5,
    "no_supported_fallback": 21
  },
  "priority_decision": "Eliminate deterministic hard failures before semantic prompt tuning.",
  "recorded_at": "2026-07-21T18:30:11+00:00",
  "sentinels_passed": true,
  "status": "complete"
}
```

## b-loop-calculation_executor-a2-typed-units-v2

- recorded_at: `2026-07-21T18:30:11+00:00`

```json
{
  "attempt_index": 2,
  "direction_id": "calculation_executor",
  "event": "attempt_planned",
  "experiment_id": "b-loop-calculation_executor-a2-typed-units-v2",
  "history_decision": "refine_existing",
  "history_similarity": 0.88,
  "material_delta": {
    "percent_denominator_normalization": true,
    "ratio_percent_rendering": true
  },
  "recorded_at": "2026-07-21T18:30:11+00:00",
  "status": "running",
  "target_qids": [
    "fin_b_017"
  ]
}
```

## b-loop-calculation_executor-a2-typed-units-v2

- recorded_at: `2026-07-21T18:47:06+00:00`

```json
{
  "attempt_index": 2,
  "blind_evaluation_token_usage": {
    "completion_tokens": 284,
    "prompt_tokens": 3808,
    "total_tokens": 4092
  },
  "blind_pair": {
    "candidate_label": "A",
    "evaluation": {
      "confidence": 98,
      "prompt_version": "b_blind_pair_v1",
      "reason": "证据中2025年EBITDA为338,931百万元、EBITDA率为32.3%、营业收入为1,050,187百万元。隐含营业收入=338,931÷0.323=1,049,321.9814，保留两位为1049321.98。绝对相对偏差=|1,049,321.9814-1,050,187|÷1,050,187=0.00082368，即0.082368%，按百分数保留两位且不带单位为0.08。A的数值和格式均符合要求；B第二项输出0.00%既未正确转换为百分数保留两位，也带了题目要求不带的单位符号。",
      "winner": "A"
    },
    "incumbent_label": "B",
    "prompt_fingerprint": "bd74aac0b37d7f88a99eae50c5c9fe6fa5e5779a835568d58d7592acf2de79f4",
    "token_usage": {
      "completion_tokens": 284,
      "prompt_tokens": 3808,
      "total_tokens": 4092
    }
  },
  "candidate_fingerprint": {
    "components": {
      "context": {
        "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
        "evaluator_model": "gpt-5.5",
        "generator_model": "gpt-5.5"
      },
      "identity": {
        "change_vector": {
          "causal_paired_gate": true,
          "percent_denominator_normalization": true,
          "ratio_percent_rendering": true,
          "strategy": "typed_units_v2"
        },
        "domains": [
          "financial_reports"
        ],
        "hypothesis": "typed percent/ratio propagation and deterministic bare-percent rendering make calculation answers evidence-grounded and submission-valid.",
        "pipeline_stage": "calculation",
        "question_types": [
          "calculation"
        ],
        "root_cause_cluster": "calculation",
        "target_qids": [
          "fin_b_017"
        ]
      }
    },
    "context_sha256": "fb8ccbeedeb2b9d54db9a8a389961275c1cf689059baaf6a3b8445af28fb8748",
    "direction_sha256": "9e1022e9c64412c4538cfd9d16ac17703105bdf57929a6a74acd48dbc9bf1fd7",
    "schema_version": 1,
    "semantic_sha256": "6bc7bb40709a6665cd9309b2d403e8568bf40721727f32a723a1fce765a50184",
    "sha256": "8b0ee8e6945dc439eaa0a539911b360b38697d761f8f4fc37db342837d9f2f87"
  },
  "change_vector": {
    "causal_paired_gate": true,
    "percent_denominator_normalization": true,
    "ratio_percent_rendering": true,
    "strategy": "typed_units_v2"
  },
  "changed_answer_qids": [
    "fin_b_017"
  ],
  "confidence_after": {
    "fin_b_017": 98
  },
  "confidence_before": {
    "fin_b_017": 0
  },
  "diagnostic_call_token_usage": {
    "completion_tokens": 964,
    "prompt_tokens": 1644,
    "total_tokens": 2608
  },
  "direction_id": "calculation_executor",
  "domains": [
    "financial_reports"
  ],
  "effective_branch": "codex/b榜-loop-i001-calculation-typed-units",
  "evaluator_failed_attempts_not_in_usage": 3,
  "evaluator_fingerprint": "3a31f89c4764adb6e081781ba1ef9db8307ec75402558a1c1f74b48ed97f539f",
  "evaluator_model": "gpt-5.5",
  "event": "iteration_promoted",
  "experiment_id": "b-loop-calculation_executor-a2-typed-units-v2",
  "full_evaluation_token_usage_recorded": {
    "completion_tokens": 87752,
    "prompt_tokens": 468276,
    "total_tokens": 556028
  },
  "generation_token_usage": {
    "completion_tokens": 2589,
    "prompt_tokens": 12827,
    "total_tokens": 15416
  },
  "generator_model": "gpt-5.5",
  "hypothesis": "Typed percent/ratio propagation and deterministic bare-percent rendering make calculation answers evidence-grounded and submission-valid.",
  "integrity": {
    "artifact_complete": true,
    "invalid_submission_after": 5,
    "invalid_submission_before": 6,
    "passed": true
  },
  "iteration_run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/iterations/iteration_001_typed_units_v2",
  "material_delta": {
    "official_bare_percent_slot": true,
    "percent_points_to_ratio_conversion": true,
    "typed_value_kinds": true
  },
  "next_root_causes": [
    "calculation_directionality",
    "iterative_variable_retrieval",
    "choice_evidence_contract"
  ],
  "pipeline_stage": "calculation",
  "promotion_result": "promoted",
  "question_types": [
    "calculation"
  ],
  "raw_fixed_evaluator_drift": {
    "policy": "all 100 raw evaluations retained; causal gate normalizes unchanged answer subjects to incumbent scores",
    "unchanged_score_drift_count": 33,
    "unchanged_tier_drift_count": 11
  },
  "reason": "fin_b_017 improved from blocked 0 to high 98; blind judge preferred candidate at confidence 98; invalid submission qids fell from 6 to 5 with no new hard failure.",
  "recorded_at": "2026-07-21T18:47:06+00:00",
  "registry_schema_version": 1,
  "root_cause_cluster": "calculation",
  "round_gate": {
    "metrics": {
      "causal_changed_qids": [
        "fin_b_017"
      ],
      "domain_metrics": {
        "financial_contracts": {
          "blocked_low_after": 17,
          "blocked_low_before": 17,
          "low_tier_improved_qids": [],
          "p10_after": 0,
          "p10_before": 0,
          "p10_delta": 0
        },
        "financial_reports": {
          "blocked_low_after": 16,
          "blocked_low_before": 17,
          "low_tier_improved_qids": [
            "fin_b_017"
          ],
          "p10_after": 0,
          "p10_before": 0,
          "p10_delta": 0
        },
        "insurance": {
          "blocked_low_after": 11,
          "blocked_low_before": 11,
          "low_tier_improved_qids": [],
          "p10_after": 0,
          "p10_before": 0,
          "p10_delta": 0
        },
        "regulatory": {
          "blocked_low_after": 12,
          "blocked_low_before": 12,
          "low_tier_improved_qids": [],
          "p10_after": 0,
          "p10_before": 0,
          "p10_delta": 0
        },
        "research": {
          "blocked_low_after": 17,
          "blocked_low_before": 17,
          "low_tier_improved_qids": [],
          "p10_after": 0,
          "p10_before": 0,
          "p10_delta": 0
        }
      },
      "evaluated_qid_count": 100,
      "expected_qid_count": 100,
      "has_domain_p10_gain": false,
      "has_low_tier_improvement": true,
      "new_hard_failures": {},
      "unchanged_raw_score_drift": {
        "fc_b_003": -5,
        "fc_b_013": -3,
        "fc_b_016": -5,
        "fc_b_018": 5,
        "fc_b_019": -15,
        "fin_b_007": -2,
        "fin_b_009": 20,
        "fin_b_010": 5,
        "fin_b_011": 5,
        "ins_b_002": -10,
        "ins_b_004": -2,
        "ins_b_005": 10,
        "ins_b_006": -10,
        "ins_b_007": 5,
        "ins_b_008": -5,
        "ins_b_010": 3,
        "ins_b_012": 22,
        "ins_b_014": 8,
        "ins_b_016": 18,
        "ins_b_017": -35,
        "ins_b_020": -3,
        "reg_b_005": 8,
        "reg_b_008": 15,
        "reg_b_009": 10,
        "reg_b_013": -20,
        "reg_b_015": -6,
        "reg_b_016": -5,
        "reg_b_017": -7,
        "reg_b_020": 10,
        "reg_b_023": -6,
        "res_b_001": 8,
        "res_b_013": -13,
        "res_b_015": -8
      },
      "unchanged_raw_tier_drift": [
        "fc_b_019",
        "fin_b_009",
        "ins_b_005",
        "ins_b_012",
        "ins_b_016",
        "ins_b_017",
        "reg_b_005",
        "reg_b_008",
        "reg_b_015",
        "reg_b_017",
        "res_b_015"
      ]
    },
    "reasons": [],
    "valid": true
  },
  "run_dir": "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/candidates/calculation_executor/typed_units_v2_fin_b_017",
  "single_candidate_confidence": {
    "fin_b_017": 100
  },
  "status": "accepted",
  "target_qids": [
    "fin_b_017"
  ],
  "tests": {
    "count": 103,
    "passed": true
  },
  "token_accounting_complete": false
}
```
