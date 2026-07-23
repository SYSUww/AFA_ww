from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from afa_agent.b_board.calculation import CalculationExecutor
from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.runner import (
    RUN_MODE_RESEARCH,
    RUN_MODE_SUBMISSION,
    SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION,
    SUBMISSION_REASONING_FEEDBACK_SYSTEM_PROMPT,
    SUBMISSION_REASONING_PROMPT_VERSION,
    SUBMISSION_REASONING_REFINE_POLICY_VERSION,
    SUBMISSION_REASONING_REFINE_PROMPT_VERSION,
    SUBMISSION_REASONING_REFINE_SYSTEM_PROMPT,
    SUBMISSION_REASONING_SYSTEM_PROMPT,
    BAnswerArtifact,
    BAnswerGenerationError,
    BBoardActualRunner,
    _artifact_from_dict,
    _calculation_evidence_payload,
    _calculation_semantic_constraints,
    _calculation_semantic_query_terms,
    _is_calculation_plan_structure_error,
    _normalize_calculation_plan_structure,
    _normalize_calculation_numeric_literals,
    _validate_aggregate_intensity_plan_binding,
    _validate_calculation_summary_output_consistency,
    _validate_calculation_variable_period_binding,
    _validate_calculation_result_semantics,
    _validate_insurance_surrender_rate_binding,
)
from afa_agent.client import LLMResponse
from afa_agent.config import ModelConfig, RunConfig
from afa_agent.models import TokenUsage
from afa_agent.run_metadata import RunFingerprintError, validate_resume_fingerprint
from scripts import run_b_board_actual


def _question() -> BQuestion:
    return BQuestion(
        qid="q1",
        domain="regulatory",
        split="B",
        question="该说法是否正确？",
        options={"A": "正确", "B": "错误"},
        answer_format="tf",
        type="判断题",
        answer_slots=1,
        answer_slot_templates=("A",),
    )


def _artifact() -> BAnswerArtifact:
    return BAnswerArtifact(
        qid="q1",
        domain="regulatory",
        answer_format="tf",
        answer_slot_count=1,
        answer_parts=["A"],
        used_evidence_ids=["u1"],
        evidence_items=[{"unit_id": "u1", "text": "证据"}],
        decision_summary="证据明确支持题干中的监管要求成立，因此选择正确选项A。",
        decision_trace={},
        calculation_trace={},
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        locator={},
    )


def _model(model_name: str) -> ModelConfig:
    return ModelConfig(
        api_key="secret",
        api_base="https://example.invalid/v1",
        model_name=model_name,
        temperature=0.0,
    )


class _QueuedClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.messages.append(messages)
        return self.responses.pop(0)


def _response(content: str, prompt: int, completion: int) -> LLMResponse:
    return LLMResponse(
        content=content,
        token_usage=TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        raw_payload={},
    )


class BBoardRunnerModeTests(unittest.TestCase):
    def test_direct_numeric_output_normalizes_qwen_text_percentage_literal(self) -> None:
        plan = {
            "variables": [
                {
                    "name": "毛利率",
                    "value": "5.55%",
                    "value_type": "text",
                    "unit": "%",
                    "evidence_ids": ["u1"],
                }
            ],
            "steps": [],
            "outputs": [{"source": "毛利率", "format": "percent2"}],
        }

        normalized, changes = _normalize_calculation_numeric_literals(plan)
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=1,
            evidence_text_by_id={"u1": "报告期内主营业务毛利率为5.55%。"},
            expected_slot_templates=("999999.99%",),
            expected_percent_suffixes=(True,),
        )

        self.assertEqual(plan["variables"][0]["value_type"], "text")
        self.assertEqual(normalized["variables"][0]["value_type"], "decimal")
        self.assertEqual(result.answer_parts, ("5.55%",))
        self.assertEqual(changes[0]["reason"], "direct_numeric_output_literal")

    def test_numeric_literal_normalization_does_not_retype_non_numeric_text(self) -> None:
        plan = {
            "variables": [
                {
                    "name": "排序",
                    "value": "甲>乙",
                    "value_type": "text",
                    "unit": "",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "说明",
                    "value": "5.55%",
                    "value_type": "text",
                    "unit": "%",
                    "evidence_ids": ["u1"],
                },
            ],
            "steps": [],
            "outputs": [
                {"source": {"ref": "排序"}, "format": "text"},
                {"source": {"ref": "说明"}, "format": "raw"},
            ],
        }

        normalized, changes = _normalize_calculation_numeric_literals(plan)

        self.assertEqual(normalized, plan)
        self.assertEqual(changes, [])

    def test_structure_normalization_repairs_args_literals_and_prunes_unused_derived_values(
        self,
    ) -> None:
        plan = {
            "variables": [
                {
                    "name": "金额",
                    "value": "10",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "数量",
                    "value": "2",
                    "value_type": "decimal",
                    "unit": "",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "未落地派生值",
                    "value": "500",
                    "value_type": "decimal",
                    "unit": "",
                    "evidence_ids": ["u1"],
                },
            ],
            "steps": [
                {
                    "id": "s1",
                    "op": "div",
                    "args": {
                        "a": {"ref": "金额"},
                        "b": {"ref": "数量"},
                    },
                },
                {
                    "id": "s2",
                    "op": "mul",
                    "args": {
                        "a": {"ref": "s1"},
                        "b": {
                            "value": "100",
                            "value_type": "decimal",
                        },
                    },
                },
            ],
            "outputs": [{"source": "s2", "format": "decimal2"}],
        }

        normalized, changes = _normalize_calculation_plan_structure(plan)
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=1,
            evidence_text_by_id={"u1": "金额为10元，数量为2。"},
        )

        self.assertEqual(result.answer_parts, ("500.00",))
        self.assertEqual(
            [item["name"] for item in normalized["variables"]],
            ["金额", "数量"],
        )
        self.assertIsInstance(normalized["steps"][0]["args"], list)
        self.assertEqual(
            normalized["steps"][1]["args"][1]["literal"],
            "100",
        )
        self.assertEqual(
            {
                item["reason"]
                for item in changes
            },
            {
                "default_empty_decision_summary",
                "default_empty_supporting_evidence_ids",
                "named_args_to_ordered_schema",
                "numeric_literal_default_empty_unit",
                "prune_non_output_dependencies",
                "value_object_to_literal",
            },
        )

    def test_structure_normalization_rewrites_explicit_step_aliases(self) -> None:
        plan = {
            "variables": [
                {
                    "name": "总额",
                    "value": "100",
                    "value_type": "decimal",
                    "unit": "亿元",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "占比",
                    "value": "60",
                    "value_type": "decimal",
                    "unit": "%",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "派生总额",
                    "value": "60",
                    "value_type": "decimal",
                    "unit": "亿元",
                    "evidence_ids": ["s1"],
                },
            ],
            "steps": [
                {
                    "id": "s1",
                    "op": "mul",
                    "args": {
                        "a": {"ref": "总额"},
                        "b": {"ref": "占比"},
                    },
                }
            ],
            "outputs": [{"source": "派生总额", "format": "decimal2"}],
        }

        normalized, changes = _normalize_calculation_plan_structure(plan)
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=1,
            evidence_text_by_id={"u1": "总额为100亿元，占比为60%。"},
        )

        self.assertEqual(result.answer_parts, ("60.00",))
        self.assertNotIn(
            "派生总额",
            [item["name"] for item in normalized["variables"]],
        )
        self.assertEqual(normalized["outputs"][0]["source"], {"ref": "s1"})
        self.assertIn(
            "derived_variable_step_aliases",
            [item["reason"] for item in changes],
        )

    def test_structure_normalization_repairs_count_and_inline_sort(self) -> None:
        plan = {
            "variables": [
                {
                    "name": "甲",
                    "value": "1",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "乙",
                    "value": "2",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "门槛",
                    "value": "1",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
            ],
            "steps": [
                {
                    "id": "count",
                    "op": "count_gte",
                    "args": {
                        "items": [{"ref": "甲"}, {"ref": "乙"}],
                        "threshold": {"ref": "门槛"},
                    },
                }
            ],
            "outputs": [
                {
                    "source": "sort_desc",
                    "format": "text",
                    "items": [
                        {"label": "甲", "source": {"ref": "甲"}},
                        {"label": "乙", "source": {"ref": "乙"}},
                    ],
                },
                {"source": "count", "format": "decimal0"},
            ],
        }

        normalized, changes = _normalize_calculation_plan_structure(plan)
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=2,
            evidence_text_by_id={"u1": "甲为1元，乙为2元，门槛为1元。"},
        )

        self.assertEqual(result.answer_parts, ("乙>甲", "2"))
        count = next(
            item for item in normalized["steps"] if item["id"] == "count"
        )
        self.assertEqual(count["threshold"], {"ref": "门槛"})
        self.assertIsInstance(count["args"], list)
        self.assertIn(
            "inline_output_sort_to_step",
            [item["reason"] for item in changes],
        )

    def test_structure_normalization_clears_absent_same_scale_units_only_for_ratio(
        self,
    ) -> None:
        plan = {
            "variables": [
                {
                    "name": "甲现金流",
                    "value": "20",
                    "value_type": "decimal",
                    "unit": "千元",
                    "evidence_ids": ["a"],
                },
                {
                    "name": "甲收入",
                    "value": "100",
                    "value_type": "decimal",
                    "unit": "千元",
                    "evidence_ids": ["a"],
                },
            ],
            "steps": [
                {
                    "id": "现金流率",
                    "op": "div",
                    "args": [
                        {"ref": "甲现金流"},
                        {"ref": "甲收入"},
                    ],
                }
            ],
            "outputs": [
                {
                    "source": {"ref": "现金流率"},
                    "format": "percent2",
                }
            ],
        }

        normalized, changes = _normalize_calculation_plan_structure(
            plan,
            evidence_text_by_id={"a": "甲现金流为20，甲收入为100。"},
        )
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=1,
            evidence_text_by_id={"a": "甲现金流为20，甲收入为100。"},
        )

        self.assertEqual(result.answer_parts, ("20.00%",))
        self.assertEqual(
            [item["unit"] for item in normalized["variables"]],
            ["", ""],
        )
        self.assertIn(
            "clear_absent_same_scale_ratio_units",
            [item["reason"] for item in changes],
        )

    def test_schema_errors_skip_retrieval_but_grounding_errors_do_not(self) -> None:
        self.assertTrue(
            _is_calculation_plan_structure_error(
                ValueError("mul args must be a list")
            )
        )
        self.assertFalse(
            _is_calculation_plan_structure_error(
                ValueError(
                    "Variables are not grounded in cited evidence: 金额[value_not_found]"
                )
            )
        )

    def test_calculation_semantic_query_terms_cover_rule_evidence(self) -> None:
        self.assertIn(
            "中期分红",
            _calculation_semantic_query_terms(
                "两家公司2025年度全年每10股分红差额是多少？"
            ),
        )
        self.assertIn(
            "给付条件",
            _calculation_semantic_query_terms(
                "不同情形下累计身故保险金合计是多少？"
            ),
        )
        self.assertEqual(
            _calculation_semantic_query_terms("营业收入是多少？"),
            "",
        )

    def test_aggregate_intensity_constraint_is_derived_from_question_and_evidence(
        self,
    ) -> None:
        constraints = _calculation_semantic_constraints(
            (
                "若2026年国内新能源乘用车销量与2025年持平，但单车带电量"
                "从2025年的水平提升至56kWh，则动力电池需求同比增速是多少？"
            ),
            [
                {
                    "doc_id": "__question__",
                    "evidence_id": "question:q1",
                    "title": "题目",
                    "text": "单车带电量提升至56kWh",
                },
                {
                    "doc_id": "report",
                    "evidence_id": "u1",
                    "title": "2025 | 2026E | 2027E",
                    "text": (
                        "乘用车单车带电量(kwh) | 45.8 | 52.2 | 53.1\n"
                        "国内:纯电动销量(万辆) | 811.6 | 811.6 | 876.5"
                    ),
                }
            ],
        )

        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]["evidence_id"], "u1")
        self.assertEqual(
            constraints[0]["question_evidence_id"],
            "question:q1",
        )
        self.assertIn("45.8kWh", constraints[0]["constraint"])
        self.assertIn("56kWh", constraints[0]["constraint"])
        self.assertIn("不得把总体", constraints[0]["constraint"])
        self.assertEqual(constraints[0]["baseline_value"], "45.8")
        self.assertEqual(constraints[0]["target_value"], "56")

    def test_aggregate_intensity_constraint_does_not_fire_without_flat_volume(
        self,
    ) -> None:
        self.assertEqual(
            _calculation_semantic_constraints(
                "预计乘用车单车带电量提升至56kWh，需求是多少？",
                [],
            ),
            [],
        )

    def test_aggregate_intensity_gate_rejects_different_scope_denominator(
        self,
    ) -> None:
        constraints = [
            {
                "type": "unchanged_aggregate_volume",
                "evidence_id": "table",
                "question_evidence_id": "question:q1",
                "base_year": "2025",
                "baseline_value": "45.8",
                "target_value": "56",
                "unit": "kWh",
                "constraint": "test",
            }
        ]
        wrong_plan = {
            "variables": [
                {
                    "name": "2025总体单车带电量",
                    "value": "45.8",
                    "value_type": "decimal",
                    "unit": "kwh",
                    "evidence_ids": ["table"],
                },
                {
                    "name": "2026总体单车带电量",
                    "value": "56",
                    "value_type": "decimal",
                    "unit": "kWh",
                    "evidence_ids": ["question:q1"],
                },
                {
                    "name": "不同口径总量",
                    "value": "733",
                    "value_type": "decimal",
                    "unit": "GWh",
                    "evidence_ids": ["other"],
                },
            ],
            "steps": [
                {
                    "id": "wrong",
                    "op": "pct_change",
                    "new": {"ref": "2026总体单车带电量"},
                    "old": {"ref": "不同口径总量"},
                }
            ],
            "outputs": [{"source": {"ref": "wrong"}, "format": "percent2"}],
        }
        with self.assertRaisesRegex(
            ValueError,
            "whose old dependency contains evidence baseline 45.8kWh",
        ):
            _validate_aggregate_intensity_plan_binding(
                wrong_plan,
                constraints,
            )

        correct_plan = {
            **wrong_plan,
            "steps": [
                {
                    "id": "correct",
                    "op": "pct_change",
                    "new": {"ref": "2026总体单车带电量"},
                    "old": {"ref": "2025总体单车带电量"},
                }
            ],
            "outputs": [{"source": {"ref": "correct"}, "format": "percent2"}],
        }
        _validate_aggregate_intensity_plan_binding(correct_plan, constraints)

    def test_calculation_summary_must_contain_replayed_numeric_output(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "decision_summary does not contain replayed output: -90.07%",
        ):
            _validate_calculation_summary_output_consistency(
                {"decision_summary": "最终同比增速为18.10%。"},
                ("-90.07%",),
            )
        _validate_calculation_summary_output_consistency(
            {"decision_summary": "最终同比增速为22.27%。"},
            ("22.27%",),
        )

    def test_calculation_evidence_payload_expands_progressively(self) -> None:
        evidence = [
            {
                "unit_id": "question:q1",
                "doc_id": "__question__",
                "title_path": ["题目"],
                "text": "题目",
            },
            *[
                {
                    "unit_id": f"u{index}",
                    "doc_id": f"d{index}",
                    "title_path": [],
                    "text": f"证据{index}",
                }
                for index in range(1, 12)
            ],
        ]

        first = _calculation_evidence_payload(
            evidence,
            max_non_question_hits=8,
        )
        second = _calculation_evidence_payload(
            evidence,
            max_non_question_hits=16,
        )

        self.assertEqual(len(first), 9)
        self.assertEqual(first[0]["evidence_id"], "question:q1")
        self.assertEqual(first[-1]["evidence_id"], "u8")
        self.assertEqual(len(second), 12)
        self.assertEqual(second[-1]["evidence_id"], "u11")

    def test_percentage_point_question_rejects_ratio_output(self) -> None:
        question = BQuestion(
            qid="q-pct-point",
            domain="fin",
            split="B",
            question="两家公司的指标相差多少个百分点？",
            options={},
            answer_format="freeform",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("0.00",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "percentage-point output must have percent_points value_kind",
        ):
            _validate_calculation_result_semantics(
                question,
                {"outputs": [{"value_kind": "ratio", "value": "0.10"}]},
            )
        _validate_calculation_result_semantics(
            question,
            {"outputs": [{"value_kind": "percent_points", "value": "10.00"}]},
        )

    def test_insurance_surrender_rate_rejects_adjacent_year_literal(self) -> None:
        question = BQuestion(
            qid="ins_b_019",
            domain="insurance",
            split="B",
            question=(
                "国寿增益宝在第5个保单年度解除，个人账户价值50万元，"
                "犹豫期后退保可退还多少？"
            ),
            options={},
            answer_format="freeform",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("0.00",),
        )
        evidence = {
            "u1": (
                "退保费用占个人账户价值的比例为：保单年度 | 退保费用比例 "
                "第四年 | 1% 第五年 | 1% 第六年及以后 | 0%"
            )
        }
        wrong_plan = {
            "variables": [
                {
                    "name": "国寿增益宝个人账户价值",
                    "value": "50",
                    "value_type": "decimal",
                    "unit": "万元",
                    "evidence_ids": ["question:ins_b_019"],
                }
            ],
            "steps": [
                {
                    "id": "fee",
                    "op": "mul",
                    "args": [
                        {"ref": "国寿增益宝个人账户价值"},
                        {
                            "literal": "0",
                            "value_type": "decimal",
                            "unit": "%",
                        },
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "第5个保单年度的证据费率为1%",
        ):
            _validate_insurance_surrender_rate_binding(
                question,
                wrong_plan,
                evidence,
            )

        correct_plan = {
            **wrong_plan,
            "steps": [
                {
                    "id": "fee",
                    "op": "mul",
                    "args": [
                        {"ref": "国寿增益宝个人账户价值"},
                        {
                            "literal": "1.00",
                            "value_type": "decimal",
                            "unit": "%",
                        },
                    ],
                }
            ],
        }
        _validate_insurance_surrender_rate_binding(
            question,
            correct_plan,
            evidence,
        )

    def test_dated_variable_requires_evidence_from_same_target_date(self) -> None:
        question = BQuestion(
            qid="fc_b_005",
            domain="financial_contracts",
            split="B",
            question=(
                "两次评估基准日2023年6月30日和2023年12月31日的"
                "评估增值率分别是多少？"
            ),
            options={},
            answer_format="freeform",
            type="计算题",
            answer_slots=2,
            answer_slot_templates=("0.00%", "0.00%"),
        )
        wrong_plan = {
            "variables": [
                {
                    "name": "2023年6月30日评估增值率",
                    "value": "1468.47",
                    "value_type": "decimal",
                    "unit": "%",
                    "evidence_ids": ["june"],
                },
                {
                    "name": "2023年12月31日评估增值率",
                    "value": "1468.47",
                    "value_type": "decimal",
                    "unit": "%",
                    "evidence_ids": ["june"],
                },
            ]
        }
        evidence = {
            "june": "评估基准日为2023年6月30日，增值率1468.47%。",
            "december": "加期评估基准日为2023年12月31日，增值率740.58%。",
        }

        with self.assertRaisesRegex(
            ValueError,
            "2023年12月31日",
        ):
            _validate_calculation_variable_period_binding(
                question,
                wrong_plan,
                evidence,
            )

        correct_plan = {
            "variables": [
                wrong_plan["variables"][0],
                {
                    **wrong_plan["variables"][1],
                    "value": "740.58",
                    "evidence_ids": ["december"],
                },
            ]
        }
        _validate_calculation_variable_period_binding(
            question,
            correct_plan,
            evidence,
        )

    def test_reasoning_prompt_requires_explicit_auditable_structure(self) -> None:
        self.assertEqual(
            SUBMISSION_REASONING_PROMPT_VERSION,
            "b_submission_reasoning_v3_qwen37_grounded",
        )
        self.assertIn("定位—关键事实—推导—结论", SUBMISSION_REASONING_SYSTEM_PROMPT)
        self.assertIn("frozen_answer_parts", SUBMISSION_REASONING_SYSTEM_PROMPT)
        self.assertIn('grounding_status="insufficient"', SUBMISSION_REASONING_SYSTEM_PROMPT)

    def test_reasoning_generation_preserves_answer_and_receives_verified_trace(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="qwen3.7-plus"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"answer_parts":["A"],"grounding_status":"supported",'
                    '"missing_support":[],"reasoning":"定位监管要求后，证据明确给出适用条件，'
                    '该条件与题干陈述一致，因而判断成立，最终答案为A。"}',
                    10,
                    2,
                )
            ]
        )
        artifact = _artifact()

        result = runner._attach_submission_reasoning(_question(), artifact)

        self.assertEqual(result.answer_parts, ["A"])
        self.assertIn("最终答案为A", result.decision_summary)
        self.assertEqual(
            result.token_usage,
            {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        )
        payload = runner.client.messages[0][1]["content"]
        self.assertIn('"frozen_answer_parts": ["A"]', payload)
        self.assertIn('"verified_calculation_trace": {}', payload)
        trace = result.decision_trace["submission_reasoning"]
        self.assertEqual(trace["grounding_status"], "supported")
        self.assertEqual(trace["attempt_count"], 1)
        self.assertEqual(
            trace["token_usage"],
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    def test_reasoning_generation_rescues_once_after_insufficient_evidence(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="qwen3.7-plus"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"answer_parts":["A"],"grounding_status":"insufficient",'
                    '"missing_support":["缺少适用条件"],"reasoning":""}',
                    10,
                    2,
                ),
                _response(
                    '{"answer_parts":["A"],"grounding_status":"supported",'
                    '"missing_support":[],"reasoning":"定位监管要求后，补充证据明确给出适用条件，'
                    '该条件与题干陈述一致，因而判断成立，最终答案为A。"}',
                    12,
                    3,
                ),
            ]
        )
        runner._rescue_submission_reasoning_evidence = lambda *_args: [
            {
                "unit_id": "u2",
                "doc_id": "d1",
                "title_path": ["补充条款"],
                "text": "补充证据明确给出适用条件。",
            }
        ]

        result = runner._attach_submission_reasoning(_question(), _artifact())

        self.assertEqual(len(runner.client.messages), 2)
        self.assertNotIn("u2", result.used_evidence_ids)
        self.assertEqual(
            [item["unit_id"] for item in result.evidence_items],
            ["u1"],
        )
        self.assertEqual(
            [item["unit_id"] for item in result.reasoning_evidence_items],
            ["u1", "u2"],
        )
        self.assertEqual(
            result.token_usage,
            {"prompt_tokens": 32, "completion_tokens": 10, "total_tokens": 42},
        )
        trace = result.decision_trace["submission_reasoning"]
        self.assertEqual(trace["attempt_count"], 2)
        self.assertEqual(trace["rescued_evidence_ids"], ["u2"])

    def test_reasoning_refinement_prompts_match_new_md_dimensions_and_freeze_answer(self) -> None:
        self.assertEqual(
            SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION,
            "b_submission_reasoning_feedback_v2_prioritized",
        )
        for dimension in ("logical", "completeness", "clarity"):
            self.assertIn(dimension, SUBMISSION_REASONING_FEEDBACK_SYSTEM_PROMPT)
        self.assertEqual(
            SUBMISSION_REASONING_REFINE_PROMPT_VERSION,
            "b_submission_reasoning_refine_v2_minimal_verified",
        )
        self.assertIn("冻结答案", SUBMISSION_REASONING_REFINE_SYSTEM_PROMPT)
        self.assertIn("定位—关键事实—推导—结论", SUBMISSION_REASONING_REFINE_SYSTEM_PROMPT)
        self.assertEqual(
            SUBMISSION_REASONING_REFINE_POLICY_VERSION,
            "b_submission_reasoning_refine_policy_v3_conservative",
        )

    def test_reasoning_refinement_preserves_answer_and_sums_both_raw_usages(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="gpt-5.5"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"logical_issues":["因果链断裂"],"completeness_issues":[],'
                    '"clarity_issues":[],"verification_questions":["为何排除B"],'
                    '"must_preserve_facts":["证据支持A"]}',
                    10,
                    2,
                ),
                _response(
                    '{"answer_parts":["A"],"reasoning":"定位题干中的监管要求；关键证据直接支持该要求成立，与错误选项B的表述不符；因此从事实可推得判断为正确，最终答案为A。"}',
                    12,
                    3,
                ),
            ]
        )

        result = runner.refine_submission_reasoning(_question(), _artifact())

        self.assertEqual(result.answer_parts, ["A"])
        self.assertIn("最终答案为A", result.decision_summary)
        self.assertEqual(
            result.token_usage,
            {"prompt_tokens": 32, "completion_tokens": 10, "total_tokens": 42},
        )
        trace = result.decision_trace["submission_reasoning_refinement"]
        self.assertTrue(trace["answer_parts_preserved"])
        self.assertEqual(
            trace["feedback_prompt_version"], SUBMISSION_REASONING_FEEDBACK_PROMPT_VERSION
        )
        self.assertEqual(trace["refine_prompt_version"], SUBMISSION_REASONING_REFINE_PROMPT_VERSION)
        self.assertEqual(len(runner.client.messages), 2)

    def test_reasoning_refinement_preserves_original_when_feedback_has_no_issues(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="gpt-5.5"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"logical_issues":[],"completeness_issues":[],"clarity_issues":[],'
                    '"verification_questions":[],"must_preserve_facts":["证据支持A"]}',
                    10,
                    2,
                )
            ]
        )
        artifact = _artifact()
        original_reasoning = artifact.decision_summary

        result = runner.refine_submission_reasoning(_question(), artifact)

        self.assertEqual(result.decision_summary, original_reasoning)
        self.assertEqual(result.answer_parts, ["A"])
        self.assertEqual(
            result.token_usage,
            {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        )
        trace = result.decision_trace["submission_reasoning_refinement"]
        self.assertEqual(trace["mode"], "preserved_no_material_issues")
        self.assertIsNone(trace["refine_token_usage"])
        self.assertEqual(len(runner.client.messages), 1)

    def test_reasoning_refinement_preserves_single_unverified_completeness_issue(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="gpt-5.5"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"logical_issues":[],"completeness_issues":["缺少一条直接事实"],'
                    '"clarity_issues":[],"verification_questions":["证据是否存在"],'
                    '"must_preserve_facts":["证据支持A"]}',
                    10,
                    2,
                )
            ]
        )
        artifact = _artifact()
        original_reasoning = artifact.decision_summary

        result = runner.refine_submission_reasoning(_question(), artifact)

        self.assertEqual(result.decision_summary, original_reasoning)
        trace = result.decision_trace["submission_reasoning_refinement"]
        self.assertEqual(trace["mode"], "preserved_conservative_gate")
        self.assertEqual(trace["policy_version"], SUBMISSION_REASONING_REFINE_POLICY_VERSION)
        self.assertEqual(len(runner.client.messages), 1)

    def test_reasoning_refinement_rejects_answer_change_and_reports_all_usage(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.config = SimpleNamespace(model=SimpleNamespace(model_name="gpt-5.5"))
        runner.client = _QueuedClient(
            [
                _response(
                    '{"logical_issues":["因果链断裂"],"completeness_issues":[],"clarity_issues":[],'
                    '"verification_questions":[],"must_preserve_facts":["证据支持A"]}',
                    10,
                    2,
                ),
                _response(
                    '{"answer_parts":["B"],"reasoning":"这是一段长度足够但错误改变冻结答案的推理摘要，必须被硬门禁拒绝。"}',
                    12,
                    3,
                ),
            ]
        )

        with self.assertRaisesRegex(BAnswerGenerationError, "changed answer_parts") as raised:
            runner.refine_submission_reasoning(_question(), _artifact())

        self.assertEqual(
            raised.exception.token_usage,
            {"prompt_tokens": 32, "completion_tokens": 10, "total_tokens": 42},
        )

    def test_cli_defaults_to_submission_and_accepts_research(self) -> None:
        with mock.patch.object(sys, "argv", ["run_b_board_actual.py"]):
            self.assertEqual(run_b_board_actual.parse_args().run_mode, RUN_MODE_SUBMISSION)
        with mock.patch.object(
            sys, "argv", ["run_b_board_actual.py", "--run-mode", RUN_MODE_RESEARCH]
        ):
            self.assertEqual(run_b_board_actual.parse_args().run_mode, RUN_MODE_RESEARCH)

    def test_default_submission_mode_rejects_non_allowlisted_model(self) -> None:
        config = RunConfig(model=_model("gpt-5.5"))
        with mock.patch(
            "afa_agent.b_board.runner.build_run_config", return_value=config
        ), self.assertRaisesRegex(ValueError, "requires a Qwen3.5/Qwen3.6/Qwen3.7 model"):
            BBoardActualRunner(questions=[])

    def test_research_mode_allows_non_allowlisted_model(self) -> None:
        config = RunConfig(model=_model("gpt-5.5"))
        migration = SimpleNamespace(load_domain_payloads=lambda *_args: {})
        attempt = SimpleNamespace(attempt_id="attempt_43")
        with mock.patch(
            "afa_agent.b_board.runner.build_run_config", return_value=config
        ), mock.patch(
            "afa_agent.b_board.runner.OpenAICompatibleClient"
        ), mock.patch(
            "afa_agent.b_board.runner.CalculationExecutor"
        ), mock.patch(
            "afa_agent.b_board.runner._migration_module", return_value=migration
        ), mock.patch(
            "afa_agent.b_board.runner._find_locator_attempt", return_value=attempt
        ):
            runner = BBoardActualRunner(questions=[], run_mode=RUN_MODE_RESEARCH)

        self.assertEqual(runner.run_mode, RUN_MODE_RESEARCH)

    def test_research_run_only_writes_research_csv_and_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "research"
            runner = self._lightweight_runner(RUN_MODE_RESEARCH, "gpt-5.5")

            manifest = runner.run(run_dir=run_dir, workers=1)

            self.assertTrue((run_dir / "research_submit.csv").is_file())
            self.assertFalse((run_dir / "submit.csv").exists())
            self.assertFalse(manifest["submission_eligible"])
            self.assertIsNone(manifest["submission_path"])
            self.assertEqual(
                manifest["research_submission_path"],
                str((run_dir / "research_submit.csv").resolve()),
            )
            self.assertEqual(
                manifest["submission_ineligibility_reasons"],
                [
                    "research_mode_is_not_submission_eligible",
                    "model_is_not_qwen3.5_qwen3.6_or_qwen3.7",
                ],
            )
            self.assertTrue((run_dir / "usage_ledger.jsonl").is_file())
            self.assertEqual(
                manifest["usage_ledger_path"],
                str((run_dir / "usage_ledger.jsonl").resolve()),
            )

    def test_submission_run_writes_submit_csv_and_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "submission"
            runner = self._lightweight_runner(RUN_MODE_SUBMISSION, "qwen3.5-plus")

            manifest = runner.run(run_dir=run_dir, workers=1)

            self.assertTrue((run_dir / "submit.csv").is_file())
            self.assertFalse((run_dir / "research_submit.csv").exists())
            self.assertTrue(manifest["submission_eligible"])
            self.assertEqual(
                manifest["submission_path"], str((run_dir / "submit.csv").resolve())
            )
            self.assertIsNone(manifest["research_submission_path"])
            self.assertEqual(manifest["submission_ineligibility_reasons"], [])
            self.assertTrue((run_dir / "usage_ledger.jsonl").is_file())

    def test_qwen37_submission_run_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "submission-qwen37"
            runner = self._lightweight_runner(
                RUN_MODE_SUBMISSION, "qwen3.7-plus-2026-05-26"
            )

            manifest = runner.run(run_dir=run_dir, workers=1)

            self.assertTrue(manifest["submission_eligible"])
            self.assertEqual(manifest["submission_ineligibility_reasons"], [])

    def test_resume_preserves_prior_failed_call_usage_in_final_qid_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "resume-usage"
            failing_runner = self._lightweight_runner(
                RUN_MODE_SUBMISSION, "qwen3.7-plus"
            )
            failure_call = {
                "call_index": 1,
                "model_name": "qwen3.7-plus",
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }

            def fail_once(*_args):
                raise BAnswerGenerationError(
                    "invalid first response",
                    token_usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                    diagnostics=[
                        {"stage": "api_usage_ledger", "calls": [failure_call]}
                    ],
                )

            failing_runner.answer_one = fail_once
            first_manifest = failing_runner.run(run_dir=run_dir, workers=1)
            self.assertEqual(first_manifest["status"], "incomplete")

            succeeding_runner = self._lightweight_runner(
                RUN_MODE_SUBMISSION, "qwen3.7-plus"
            )
            succeeded = _artifact()
            succeeded.decision_trace = {
                "api_usage_ledger": {
                    "call_count": 1,
                    "calls": [
                        {
                            "call_index": 1,
                            "model_name": "qwen3.7-plus",
                            "token_usage": dict(succeeded.token_usage),
                        }
                    ],
                }
            }
            succeeding_runner.answer_one = lambda *_args: succeeded

            with mock.patch(
                "afa_agent.b_board.runner.validate_resume_fingerprint"
            ):
                final_manifest = succeeding_runner.run(run_dir=run_dir, workers=1)
            ledger_rows = [
                json.loads(line)
                for line in (run_dir / "usage_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(final_manifest["status"], "complete")
        self.assertEqual(
            final_manifest["generation_token_usage"],
            {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
        )
        self.assertEqual(
            final_manifest["failed_token_usage"],
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(final_manifest["retry_failure_count"], 1)
        self.assertEqual(len(ledger_rows), 1)
        self.assertEqual(ledger_rows[0]["status"], "success")
        self.assertEqual(ledger_rows[0]["call_count"], 2)
        self.assertEqual(
            ledger_rows[0]["token_usage"],
            {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
        )

    def test_reasoning_failure_resume_does_not_rerun_frozen_answer_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "reasoning-resume"
            answer_call_count = 0
            first_runner = self._lightweight_runner(
                RUN_MODE_SUBMISSION, "qwen3.7-plus"
            )
            answer_artifact = _artifact()
            answer_artifact.decision_trace = {
                "answer_api_usage_ledger": {
                    "call_count": 1,
                    "calls": [
                        {
                            "call_index": 1,
                            "model_name": "qwen3.7-plus",
                            "token_usage": dict(answer_artifact.token_usage),
                        }
                    ],
                }
            }

            def answer_once(*_args):
                nonlocal answer_call_count
                answer_call_count += 1
                return _artifact_from_dict(answer_artifact.to_dict())

            reasoning_failure_call = {
                "call_index": 1,
                "model_name": "qwen3.7-plus",
                "token_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            }

            def fail_reasoning(*_args):
                raise BAnswerGenerationError(
                    "reasoning evidence insufficient",
                    token_usage={
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                    },
                    diagnostics=[
                        {
                            "stage": "api_usage_ledger",
                            "pipeline_stage": "reasoning",
                            "calls": [reasoning_failure_call],
                        }
                    ],
                )

            first_runner.answer_one = answer_once
            first_runner.reasoning_one = fail_reasoning
            first_manifest = first_runner.run(run_dir=run_dir, workers=1)

            self.assertEqual(first_manifest["status"], "incomplete")
            self.assertEqual(first_manifest["answer_completed_count"], 1)
            self.assertEqual(first_manifest["reasoning_completed_count"], 0)
            self.assertEqual(first_manifest["reasoning_failed_qids"], ["q1"])
            self.assertEqual(answer_call_count, 1)
            self.assertEqual(
                len(
                    json.loads(
                        (run_dir / "answer_artifacts.json").read_text()
                    )
                ),
                1,
            )
            self.assertEqual(json.loads((run_dir / "answers.json").read_text()), [])
            self.assertFalse((run_dir / "submit.csv").exists())

            second_runner = self._lightweight_runner(
                RUN_MODE_SUBMISSION, "qwen3.7-plus"
            )

            def must_not_rerun_answer(*_args):
                raise AssertionError("frozen answer stage was rerun")

            def finish_reasoning(_question, frozen):
                result = _artifact_from_dict(frozen.to_dict())
                current_usage = {
                    "prompt_tokens": 6,
                    "completion_tokens": 2,
                    "total_tokens": 8,
                }
                result.token_usage = {
                    "prompt_tokens": frozen.token_usage["prompt_tokens"] + 6,
                    "completion_tokens": frozen.token_usage["completion_tokens"] + 2,
                    "total_tokens": frozen.token_usage["total_tokens"] + 8,
                }
                result.decision_trace = {
                    **result.decision_trace,
                    "reasoning_api_usage_ledger": {
                        "call_count": 1,
                        "calls": [
                            {
                                "call_index": 1,
                                "model_name": "qwen3.7-plus",
                                "token_usage": current_usage,
                            }
                        ],
                    },
                }
                return result

            second_runner.answer_one = must_not_rerun_answer
            second_runner.reasoning_one = finish_reasoning
            with mock.patch(
                "afa_agent.b_board.runner.validate_resume_fingerprint"
            ):
                final_manifest = second_runner.run(run_dir=run_dir, workers=1)
            ledger_rows = [
                json.loads(line)
                for line in (run_dir / "usage_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(answer_call_count, 1)
        self.assertEqual(final_manifest["status"], "complete")
        self.assertEqual(final_manifest["answer_retry_failure_count"], 0)
        self.assertEqual(final_manifest["reasoning_retry_failure_count"], 1)
        self.assertEqual(
            final_manifest["generation_token_usage"],
            {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        )
        self.assertEqual(ledger_rows[0]["status"], "success")
        self.assertEqual(ledger_rows[0]["call_count"], 3)
        self.assertEqual(
            ledger_rows[0]["token_usage"],
            {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        )

    def test_run_mode_changes_fingerprint_and_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_root = root / "parsed"
            index_root = root / "index"
            parsed_root.mkdir()
            index_root.mkdir()
            strategy_path = root / "strategy.json"
            strategy_path.write_text('{"version": "test"}', encoding="utf-8")
            runner = self._lightweight_runner(RUN_MODE_RESEARCH, "qwen3.5-plus")
            del runner._build_fingerprint
            runner.parsed_root = parsed_root
            runner.index_root = index_root
            runner.strategy_path = strategy_path
            runner.locator_attempt_id = "attempt_43"
            runner.calculation_top_k = 18
            runner.attempt = SimpleNamespace(to_dict=lambda: {"attempt_id": "attempt_43"})
            git_state = {
                "branch": "codex/test",
                "commit": "a" * 40,
                "dirty": False,
                "dirty_diff_sha256": "b" * 64,
                "untracked_paths": [],
            }
            with mock.patch(
                "afa_agent.run_metadata.collect_git_state", return_value=git_state
            ):
                research_fingerprint = runner._build_fingerprint(runner.questions, workers=1)
                runner.run_mode = RUN_MODE_SUBMISSION
                submission_fingerprint = runner._build_fingerprint(runner.questions, workers=1)

            self.assertEqual(
                research_fingerprint["components"]["arguments"]["run_mode"],
                RUN_MODE_RESEARCH,
            )
            self.assertEqual(
                submission_fingerprint["components"]["arguments"]["run_mode"],
                RUN_MODE_SUBMISSION,
            )
            with self.assertRaisesRegex(RunFingerprintError, "arguments"):
                validate_resume_fingerprint(
                    {"fingerprint": research_fingerprint}, submission_fingerprint
                )

    @staticmethod
    def _lightweight_runner(run_mode: str, model_name: str) -> BBoardActualRunner:
        runner = object.__new__(BBoardActualRunner)
        question = _question()
        runner.questions = [question]
        runner.question_by_qid = {question.qid: question}
        runner.run_mode = run_mode
        runner.locator_attempt_id = "attempt_43"
        runner.config = SimpleNamespace(model=_model(model_name))
        runner.locate = lambda _questions: {question.qid: {"qid": question.qid}}
        runner.answer_one = lambda _question, _locator: _artifact()
        runner.reasoning_one = (
            lambda _question, artifact: _artifact_from_dict(artifact.to_dict())
        )
        runner._build_fingerprint = lambda _questions, _workers: {
            "schema_version": 1,
            "sha256": "unit-test",
            "components": {"arguments": {"run_mode": run_mode}},
        }
        return runner


if __name__ == "__main__":
    unittest.main()
