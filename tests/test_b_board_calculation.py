from __future__ import annotations

import unittest
from datetime import date

from afa_agent.b_board.calculation import (
    BusinessCalendar,
    CalculationExecutor,
    CalculationPlanError,
    _answers_differ_only_in_format,
)
from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.runner import (
    CALCULATION_SYSTEM_PROMPT,
    _calculation_operator_shape_hint,
    _calculation_plan_reports_missing_required_input,
    _calculation_retry_query,
    _diagnostic_phrase_evidence,
    _merge_calculation_evidence,
)


class BBoardCalculationTests(unittest.TestCase):
    def test_operator_shape_hint_is_question_conditioned_and_answer_blind(self):
        hint = _calculation_operator_shape_hint(
            "计算同比增速，并按指标从高到低排序；最晚应从何时开始公示。"
        )

        self.assertIn('"op":"pct_change"', hint)
        self.assertIn('"op":"sort_desc"', hint)
        self.assertIn("date_add_days", hint)
        self.assertNotIn("22.27", hint)
        self.assertEqual(
            _calculation_operator_shape_hint(
                "后续公告每30日一次，两次公告之间间隔多少日？"
            ),
            "",
        )
        self.assertEqual(
            _calculation_operator_shape_hint("直接提取报告中的金额。"),
            "",
        )

    def test_admitted_missing_input_is_distinct_from_shape_failure(self):
        self.assertTrue(
            _calculation_plan_reports_missing_required_input(
                {
                    "decision_summary": (
                        "给定证据中没有任何一家公司的资产负债率，无法计算。"
                    )
                }
            )
        )
        self.assertFalse(
            _calculation_plan_reports_missing_required_input(
                {
                    "decision_summary": (
                        "材料未直接提供合计，但分项完整，相加后可以计算。"
                    )
                }
            )
        )

    def test_string_source_matching_variable_name_is_treated_as_reference(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "后续公告周期",
                        "value": "30",
                        "value_type": "decimal",
                        "unit": "日",
                        "evidence_ids": ["question"],
                    }
                ],
                "steps": [],
                "outputs": [{"source": "后续公告周期", "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"question": "后续每30日公告一次"},
            expected_slot_templates=("999999.99",),
        )

        self.assertEqual(result.answer_parts, ("30.00",))
        self.assertEqual(result.trace["outputs"][0]["value_kind"], "amount")

    def test_grounding_accepts_exact_chinese_integer_wording(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "后续公告周期",
                        "value": "30",
                        "value_type": "decimal",
                        "unit": "日",
                        "evidence_ids": ["rule"],
                    }
                ],
                "steps": [],
                "outputs": [
                    {
                        "source": "后续公告周期",
                        "format": "decimal0",
                    }
                ],
            },
            expected_slots=1,
            evidence_text_by_id={"rule": "此后每三十日应当公告一次。"},
        )

        self.assertEqual(result.answer_parts, ("30",))
        self.assertTrue(result.trace["grounding_verified"])

    def test_format_migration_guard_rejects_value_changes(self):
        self.assertTrue(_answers_differ_only_in_format(["40.05", "67.10"], ["40.05%", "67.1"]))
        self.assertFalse(_answers_differ_only_in_format(["0.08"], ["8.00%"]))

    def test_explicit_question_precision_overrides_generic_numeric_slot(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "ordinary_users",
                        "value": "67.051351",
                        "value_type": "decimal",
                        "unit": "万人",
                        "evidence_ids": ["question"],
                    }
                ],
                "steps": [],
                "outputs": [{"source": {"ref": "ordinary_users"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"question": "普通用户人数为67.051351万人"},
            expected_slot_templates=("999999.99",),
            expected_numeric_decimal_places=1,
        )

        self.assertEqual(result.answer_parts, ("67.1",))
        self.assertEqual(result.trace["outputs"][0]["format"], "decimal1")
        self.assertEqual(result.trace["outputs"][0]["question_decimal_places"], 1)
        self.assertEqual(result.trace["outputs"][0]["question_rounded_value"], "67.1")

    def test_question_no_percent_instruction_overrides_percent_slot(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "return_rate",
                        "value": "7.64951",
                        "value_type": "decimal",
                        "unit": "%",
                        "evidence_ids": ["report"],
                    }
                ],
                "steps": [],
                "outputs": [{"source": {"ref": "return_rate"}, "format": "percent2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"report": "近似资产收益率为7.64951%"},
            expected_slot_templates=("999999.99%",),
            expected_numeric_decimal_places=2,
            expected_percent_suffixes=(False,),
        )

        self.assertEqual(result.answer_parts, ("7.65",))
        self.assertEqual(result.trace["outputs"][0]["format"], "percent2_bare")

    def test_legacy_trace_revalidation_prunes_helpers_and_converts_percent_ratio(self):
        result = CalculationExecutor().replay_legacy_trace(
            {
                "schema_version": 1,
                "variables": [
                    {
                        "name": "premium",
                        "value": "80",
                        "value_type": "decimal",
                        "unit": "万元",
                        "evidence_ids": ["question"],
                    },
                    {
                        "name": "refund_ratio",
                        "value": "0.75",
                        "value_type": "decimal",
                        "unit": "比例",
                        "evidence_ids": ["rule"],
                    },
                    {
                        "name": "narrative_helper",
                        "value": "旧版描述无需参与计算",
                        "value_type": "text",
                        "unit": "",
                        "evidence_ids": ["missing"],
                    },
                ],
                "steps": [
                    {
                        "id": "refund",
                        "op": "mul",
                        "args": [{"ref": "premium"}, {"ref": "refund_ratio"}],
                        "result": "60.00",
                    }
                ],
                "outputs": [
                    {
                        "slot": 1,
                        "source": {"ref": "refund"},
                        "format": "decimal2",
                        "value": "60.00",
                    }
                ],
                "replay_verified": True,
            },
            expected_slots=1,
            evidence_text_by_id={
                "question": "累计所交保险费80万元",
                "rule": "保单账户累计收益的75%",
            },
            expected_slot_templates=("999999.99",),
        )
        self.assertEqual(result.answer_parts, ("60.00",))
        self.assertTrue(result.trace["grounding_verified"])
        self.assertTrue(result.trace["revalidation"]["answer_preserved"])
        self.assertEqual(
            result.trace["revalidation"]["pruned_variable_names"],
            ["narrative_helper"],
        )
        self.assertEqual(
            result.trace["revalidation"]["converted_percent_ratio_variables"][0][
                "literal_percent_value"
            ],
            "75.00",
        )

    def test_legacy_trace_revalidation_rejects_ungrounded_ratio(self):
        with self.assertRaisesRegex(CalculationPlanError, "cannot be grounded"):
            CalculationExecutor().replay_legacy_trace(
                {
                    "schema_version": 1,
                    "variables": [
                        {
                            "name": "ratio",
                            "value": "0.75",
                            "value_type": "decimal",
                            "unit": "比例",
                            "evidence_ids": ["rule"],
                        }
                    ],
                    "steps": [],
                    "outputs": [
                        {
                            "source": {"ref": "ratio"},
                            "format": "decimal2",
                            "value": "0.75",
                        }
                    ],
                },
                expected_slots=1,
                evidence_text_by_id={"rule": "条款未披露具体比例"},
                expected_slot_templates=("999999.99",),
            )

    def test_legacy_trace_can_apply_current_percent_contract_deterministically(self):
        result = CalculationExecutor().replay_legacy_trace(
            {
                "schema_version": 2,
                "variables": [
                    {
                        "name": "growth",
                        "value": "40.0461",
                        "value_type": "decimal",
                        "unit": "%",
                        "evidence_ids": ["report"],
                    }
                ],
                "steps": [],
                "outputs": [
                    {
                        "source": {"ref": "growth"},
                        "format": "decimal2",
                        "value": "40.05",
                    }
                ],
            },
            expected_slots=1,
            evidence_text_by_id={"report": "同比增幅为40.0461%"},
            expected_slot_templates=("999999.99",),
            expected_numeric_decimal_places=2,
            expected_percent_suffixes=(True,),
            preserve_incumbent_answer=False,
        )

        self.assertEqual(result.answer_parts, ("40.05%",))
        self.assertFalse(result.trace["revalidation"]["answer_preserved"])
        self.assertTrue(result.trace["revalidation"]["format_change_allowed"])

    def test_legacy_format_migration_rejects_numeric_value_change(self):
        with self.assertRaisesRegex(CalculationPlanError, "changed the incumbent value"):
            CalculationExecutor().replay_legacy_trace(
                {
                    "schema_version": 2,
                    "variables": [
                        {
                            "name": "growth",
                            "value": "22.27",
                            "value_type": "decimal",
                            "unit": "%",
                            "evidence_ids": ["report"],
                        }
                    ],
                    "steps": [],
                    "outputs": [
                        {
                            "source": {"ref": "growth"},
                            "format": "decimal2",
                            "value": "8.00",
                        }
                    ],
                },
                expected_slots=1,
                evidence_text_by_id={"report": "同比增幅为22.27%"},
                expected_slot_templates=("999999.99",),
                expected_percent_suffixes=(True,),
                preserve_incumbent_answer=False,
            )

    def test_bare_table_amount_requires_blank_declared_unit(self):
        plan = {
            "variables": [
                {
                    "name": "overseas_revenue",
                    "value": "310,740,988,000.00",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["table"],
                }
            ],
            "steps": [],
            "outputs": [{"source": {"ref": "overseas_revenue"}, "format": "decimal2"}],
        }
        with self.assertRaisesRegex(CalculationPlanError, r"overseas_revenue\[unit_not_found\]"):
            CalculationExecutor().execute(
                plan,
                expected_slots=1,
                evidence_text_by_id={"table": "境外 | 310,740,988,000.00 | 38.65%"},
            )

        plan["variables"][0]["unit"] = ""
        result = CalculationExecutor().execute(
            plan,
            expected_slots=1,
            evidence_text_by_id={"table": "境外 | 310,740,988,000.00 | 38.65%"},
        )
        self.assertEqual(result.answer_parts, ("310740988000.00",))
        self.assertTrue(result.trace["grounding_verified"])
        self.assertIn("表格只有裸金额", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("百分点差必须使用 pct_point_delta", CALCULATION_SYSTEM_PROMPT)

    def test_ascii_energy_unit_grounding_is_case_insensitive(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "单车带电量",
                        "value": "56",
                        "value_type": "decimal",
                        "unit": "kwh",
                        "evidence_ids": ["question"],
                    }
                ],
                "steps": [],
                "outputs": [
                    {"source": {"ref": "单车带电量"}, "format": "decimal2"}
                ],
            },
            expected_slots=1,
            evidence_text_by_id={"question": "单车带电量提升至56kWh"},
            expected_slot_templates=("999999.99",),
        )

        self.assertEqual(result.answer_parts, ("56.00",))
        self.assertTrue(result.trace["grounding_verified"])

    def test_diagnostic_retry_query_and_evidence_merge(self):
        question = BQuestion(
            qid="q1",
            domain="financial_reports",
            split="B",
            question="计算境外收入同比增幅",
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("999999.99",),
        )
        query = _calculation_retry_query(
            question,
            {"decision_summary": "缺少2025年境外收入原始金额"},
            CalculationPlanError("Unknown reference: overseas_2025"),
        )
        self.assertIn("缺少2025年境外收入原始金额", query)
        self.assertIn("Unknown reference", query)

        merged, added = _merge_calculation_evidence(
            [{"unit_id": "q", "text": "question"}, {"unit_id": "u1"}],
            [{"unit_id": "u1"}, {"unit_id": "u2"}, {"unit_id": "u3"}],
            max_items=3,
        )
        self.assertEqual([item["unit_id"] for item in merged], ["q", "u1", "u2"])
        self.assertEqual(added, ["u2"])

    def test_phrase_overlay_prioritizes_rare_missing_variable_unit(self):
        class FakeRetriever:
            units = [
                {
                    "unit_id": "generic",
                    "doc_id": "annual_byd_2025_report",
                    "title_path": [],
                    "text": "营业收入合计 803,964,958,000.00",
                    "unit_type": "metric_row",
                },
                {
                    "unit_id": "overseas",
                    "doc_id": "annual_byd_2025_report",
                    "title_path": ["分地区"],
                    "text": "2025年 分地区 境外 310,740,988,000.00",
                    "unit_type": "paragraph",
                },
            ]

        hits = _diagnostic_phrase_evidence(
            FakeRetriever(),  # type: ignore[arg-type]
            ["annual_byd_2025_report"],
            "比亚迪2025年缺少分地区境外营业收入",
            top_k=2,
        )
        self.assertEqual(hits[0]["unit_id"], "overseas")
        self.assertEqual(
            hits[0]["metadata"]["retrieval_source"], "phrase_constrained_v2"
        )

        class DateRetriever:
            units = [
                {
                    "unit_id": "june",
                    "doc_id": "text08",
                    "title_path": [],
                    "text": "2023年6月30日评估增值率1468.47%",
                },
                {
                    "unit_id": "december",
                    "doc_id": "text08",
                    "title_path": [],
                    "text": "2023年12月31日评估增值率740.58%",
                },
            ]

        dated = _diagnostic_phrase_evidence(
            DateRetriever(),  # type: ignore[arg-type]
            ["text08"],
            "缺少2023年12月31日评估增值率",
            top_k=1,
        )
        self.assertEqual(dated[0]["unit_id"], "december")

    def test_decimal_mean_rounds_only_at_output(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "a", "value": "10.005", "unit": "亿元", "evidence_ids": ["u1"]},
                    {"name": "b", "value": "20.006", "unit": "亿元", "evidence_ids": ["u2"]},
                ],
                "steps": [{"id": "mean", "op": "mean", "args": [{"ref": "a"}, {"ref": "b"}]}],
                "outputs": [{"source": {"ref": "mean"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"u1": "金额为10.005亿元", "u2": "金额为20.006亿元"},
            expected_slot_templates=("999999.99",),
        )
        self.assertEqual(result.answer_parts, ("15.01",))
        self.assertEqual(result.used_evidence_ids, ("u1", "u2"))
        self.assertTrue(result.trace["replay_verified"])
        self.assertTrue(result.trace["grounding_verified"])

    def test_cross_scale_currency_subtraction_requires_explicit_conversion(self):
        plan = {
            "variables": [
                {"name": "total_gmv", "value": "100", "unit": "亿元", "evidence_ids": ["q"]},
                {"name": "self_share", "value": "60", "unit": "%", "evidence_ids": ["q"]},
                {"name": "app_share", "value": "35", "unit": "%", "evidence_ids": ["q"]},
                {"name": "members", "value": "24.01", "unit": "万人", "evidence_ids": ["q"]},
                {"name": "member_spend", "value": "2960", "unit": "元", "evidence_ids": ["q"]},
            ],
            "steps": [
                {
                    "id": "app_gmv",
                    "op": "mul",
                    "args": [{"ref": "total_gmv"}, {"ref": "self_share"}, {"ref": "app_share"}],
                },
                {
                    "id": "member_gmv",
                    "op": "mul",
                    "args": [{"ref": "members"}, {"ref": "member_spend"}],
                },
                {
                    "id": "ordinary_gmv",
                    "op": "sub",
                    "args": [{"ref": "app_gmv"}, {"ref": "member_gmv"}],
                },
            ],
            "outputs": [{"source": {"ref": "ordinary_gmv"}, "format": "decimal1"}],
        }

        with self.assertRaisesRegex(CalculationPlanError, "sub amount unit mismatch: 亿元 vs 万元"):
            CalculationExecutor().execute(
                plan,
                expected_slots=1,
                evidence_text_by_id={
                    "q": "总GMV为100亿元，自营占60%，APP占35%，会员24.01万人，人均2960元"
                },
            )

    def test_explicit_currency_scale_conversion_replays_res_b_012(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "total_gmv", "value": "100", "unit": "亿元", "evidence_ids": ["q"]},
                    {"name": "self_share", "value": "60", "unit": "%", "evidence_ids": ["q"]},
                    {"name": "app_share", "value": "35", "unit": "%", "evidence_ids": ["q"]},
                    {"name": "members", "value": "24.01", "unit": "万人", "evidence_ids": ["q"]},
                    {"name": "member_spend", "value": "2960", "unit": "元", "evidence_ids": ["q"]},
                    {"name": "ordinary_ratio", "value": "70", "unit": "%", "evidence_ids": ["q"]},
                ],
                "steps": [
                    {
                        "id": "app_gmv_yi",
                        "op": "mul",
                        "args": [{"ref": "total_gmv"}, {"ref": "self_share"}, {"ref": "app_share"}],
                    },
                    {"id": "app_gmv_wan", "op": "mul", "args": [{"ref": "app_gmv_yi"}, 10000]},
                    {
                        "id": "member_gmv_wan",
                        "op": "mul",
                        "args": [{"ref": "members"}, {"ref": "member_spend"}],
                    },
                    {
                        "id": "ordinary_gmv_wan",
                        "op": "sub",
                        "args": [{"ref": "app_gmv_wan"}, {"ref": "member_gmv_wan"}],
                    },
                    {
                        "id": "ordinary_spend",
                        "op": "mul",
                        "args": [{"ref": "member_spend"}, {"ref": "ordinary_ratio"}],
                    },
                    {
                        "id": "ordinary_users",
                        "op": "div",
                        "args": [{"ref": "ordinary_gmv_wan"}, {"ref": "ordinary_spend"}],
                    },
                ],
                "outputs": [{"source": {"ref": "ordinary_users"}, "format": "decimal1"}],
            },
            expected_slots=1,
            evidence_text_by_id={
                "q": "总GMV为100亿元，自营占60%，APP占35%，会员24.01万人，人均2960元，普通用户为70%"
            },
        )

        self.assertEqual(result.answer_parts, ("67.1",))

    def test_calculation_prompt_requires_explicit_amount_scale_conversion(self):
        self.assertIn("1亿元=10000万元", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("1万人×1元=1万元", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("1亿元乘100000000", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("max(差额,0)", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("全年金额=中期已实施金额+年末剩余金额", CALCULATION_SYSTEM_PROMPT)
        self.assertIn("supporting_evidence_ids", CALCULATION_SYSTEM_PROMPT)

    def test_count_scale_conversion_exposes_mixed_yuan_and_wanyuan_subtraction(
        self,
    ):
        plan = {
            "variables": [
                {
                    "name": "APP自营GMV",
                    "value": "21",
                    "unit": "亿元",
                    "evidence_ids": ["q"],
                },
                {
                    "name": "会员人数",
                    "value": "24.01",
                    "unit": "万人",
                    "evidence_ids": ["q"],
                },
                {
                    "name": "人均消费",
                    "value": "2960",
                    "unit": "元",
                    "evidence_ids": ["q"],
                },
            ],
            "steps": [
                {
                    "id": "人数",
                    "op": "mul",
                    "args": [
                        {"ref": "会员人数"},
                        {"literal": "10000", "value_type": "decimal"},
                    ],
                },
                {
                    "id": "会员消费",
                    "op": "mul",
                    "args": [
                        {"ref": "人数"},
                        {"ref": "人均消费"},
                    ],
                },
                {
                    "id": "APP自营GMV万元",
                    "op": "mul",
                    "args": [
                        {"ref": "APP自营GMV"},
                        {"literal": "10000", "value_type": "decimal"},
                    ],
                },
                {
                    "id": "普通用户消费",
                    "op": "sub",
                    "args": [
                        {"ref": "APP自营GMV万元"},
                        {"ref": "会员消费"},
                    ],
                },
            ],
            "outputs": [
                {
                    "source": {"ref": "普通用户消费"},
                    "format": "decimal2",
                }
            ],
        }

        with self.assertRaisesRegex(
            CalculationPlanError,
            "sub amount unit mismatch: 万元 vs 元",
        ):
            CalculationExecutor().execute(
                plan,
                expected_slots=1,
                evidence_text_by_id={
                    "q": "APP自营GMV为21亿元，会员人数24.01万人，人均消费2960元。"
                },
            )

    def test_percentage_and_sorting(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "old", "value": "40", "evidence_ids": ["u1"]},
                    {"name": "new", "value": "50", "evidence_ids": ["u2"]},
                    {"name": "x", "value": "2.1", "evidence_ids": ["u3"]},
                    {"name": "y", "value": "3.2", "evidence_ids": ["u4"]},
                ],
                "steps": [
                    {
                        "id": "growth",
                        "op": "pct_change",
                        "new": {"ref": "new"},
                        "old": {"ref": "old"},
                    },
                    {"id": "ranking", "op": "sort_desc", "items": [{"label": "甲", "source": {"ref": "x"}}, {"label": "乙", "source": {"ref": "y"}}]},
                ],
                "outputs": [
                    {"source": {"ref": "ranking"}, "format": "text"},
                    {"source": {"ref": "growth"}, "format": "percent2"},
                ],
            },
            expected_slots=2,
            evidence_text_by_id={
                "u1": "基期40",
                "u2": "当期50",
                "u3": "甲为2.1",
                "u4": "乙为3.2",
            },
            expected_slot_templates=("公司名称>公司名称", "999999.99%"),
        )
        self.assertEqual(result.answer_parts, ("乙>甲", "25.00%"))
        self.assertEqual(
            result.trace["steps"][0]["operand_roles"],
            {"new": {"ref": "new"}, "old": {"ref": "old"}},
        )

    def test_directional_operation_rejects_ambiguous_positional_args(self):
        with self.assertRaisesRegex(CalculationPlanError, "named 'new' and 'old'"):
            CalculationExecutor().execute(
                {
                    "variables": [
                        {"name": "new", "value": "56", "evidence_ids": ["q"]},
                        {"name": "old", "value": "45.8", "evidence_ids": ["q"]},
                    ],
                    "steps": [
                        {
                            "id": "growth",
                            "op": "pct_change",
                            "args": [{"ref": "old"}, {"ref": "new"}],
                        }
                    ],
                    "outputs": [{"source": {"ref": "growth"}, "format": "percent2"}],
                },
                expected_slots=1,
            )

    def test_versioned_business_calendar(self):
        calendar = BusinessCalendar(
            holidays=frozenset({date(2026, 4, 6)}),
            working_weekends=frozenset(),
        )
        result = CalculationExecutor(calendar).execute(
            {
                "variables": [{"name": "d", "value": "2026-04-03", "value_type": "date", "evidence_ids": ["u1"]}],
                "steps": [{"id": "next", "op": "next_workday", "args": [{"ref": "d"}]}],
                "outputs": [{"source": {"ref": "next"}, "format": "date_cn"}],
            },
            expected_slots=1,
            evidence_text_by_id={"u1": "日期为2026年4月3日"},
            expected_slot_templates=("999999.99",),
        )
        self.assertEqual(result.answer_parts, ("2026年4月7日",))

    def test_named_date_arguments_preserve_semantic_roles(self):
        executor = CalculationExecutor()
        plan = {
            "variables": [
                {
                    "name": "accepted",
                    "value": "2026年4月1日",
                    "value_type": "date",
                    "unit": "",
                    "evidence_ids": ["q"],
                },
                {
                    "name": "period",
                    "value": "90",
                    "value_type": "decimal",
                    "unit": "日",
                    "evidence_ids": ["q"],
                },
            ],
            "steps": [
                {
                    "id": "deadline",
                    "op": "date_add_days",
                    "args": {
                        "date": {"ref": "accepted"},
                        "days": {"ref": "period"},
                    },
                },
                {
                    "id": "elapsed",
                    "op": "days_between",
                    "args": {
                        "end": {"ref": "deadline"},
                        "start": {"ref": "accepted"},
                    },
                },
            ],
            "outputs": [
                {"source": {"ref": "deadline"}, "format": "date_cn"},
                {"source": {"ref": "elapsed"}, "format": "decimal0"},
            ],
        }
        result = executor.execute(
            plan,
            expected_slots=2,
            evidence_text_by_id={"q": "2026年4月1日，期限90日"},
        )
        self.assertEqual(result.answer_parts, ("2026年6月30日", "90"))

    def test_non_date_operation_rejects_mapping_args(self):
        plan = {
            "variables": [
                {
                    "name": "a",
                    "value": "1",
                    "value_type": "decimal",
                    "unit": "",
                    "evidence_ids": ["q"],
                }
            ],
            "steps": [
                {"id": "x", "op": "add", "args": {"left": {"ref": "a"}}}
            ],
            "outputs": [{"source": {"ref": "x"}, "format": "decimal0"}],
        }
        with self.assertRaisesRegex(CalculationPlanError, "add args must be a list"):
            CalculationExecutor().execute(
                plan,
                expected_slots=1,
                evidence_text_by_id={"q": "1"},
            )

    def test_missing_evidence_and_unknown_operation_fail(self):
        with self.assertRaisesRegex(CalculationPlanError, "no evidence"):
            CalculationExecutor().execute(
                {"variables": [{"name": "a", "value": "1", "evidence_ids": []}], "steps": [], "outputs": []},
                expected_slots=0,
            )
        with self.assertRaisesRegex(CalculationPlanError, "Unsupported operation"):
            CalculationExecutor().execute(
                {
                    "variables": [{"name": "a", "value": "1", "evidence_ids": ["u1"]}],
                    "steps": [{"id": "x", "op": "eval", "args": [{"ref": "a"}]}],
                    "outputs": [{"source": {"ref": "a"}, "format": "raw"}],
                },
                expected_slots=1,
            )

    def test_grounding_rejects_scaled_or_hallucinated_value(self):
        plan = {
            "variables": [
                {
                    "name": "rate",
                    "value": "0.0555",
                    "unit": "%",
                    "evidence_ids": ["u1"],
                }
            ],
            "steps": [],
            "outputs": [{"source": {"ref": "rate"}, "format": "percent2"}],
        }
        with self.assertRaisesRegex(CalculationPlanError, "not grounded"):
            CalculationExecutor().execute(
                plan,
                expected_slots=1,
                evidence_text_by_id={"u1": "主营业务毛利率为5.55%"},
                expected_slot_templates=("999999.99%",),
            )

    def test_count_threshold_can_reference_grounded_variable(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "a", "value": "4999", "evidence_ids": ["q"]},
                    {"name": "b", "value": "5000", "evidence_ids": ["q"]},
                    {"name": "c", "value": "8000", "evidence_ids": ["q"]},
                    {"name": "threshold", "value": "5000", "evidence_ids": ["q"]},
                ],
                "steps": [
                    {
                        "id": "count",
                        "op": "count_gte",
                        "args": [{"ref": "a"}, {"ref": "b"}, {"ref": "c"}],
                        "threshold": {"ref": "threshold"},
                    }
                ],
                "outputs": [{"source": {"ref": "count"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"q": "金额分别为4999元、5000元和8000元，门槛为5000元"},
            expected_slot_templates=("999999.99",),
        )
        self.assertEqual(result.answer_parts, ("2.00",))
        self.assertEqual(result.trace["outputs"][0]["requested_format"], "decimal2")

    def test_official_slot_schema_overrides_model_requested_format(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "count", "value": "2", "evidence_ids": ["q"]}
                ],
                "steps": [],
                "outputs": [{"source": {"ref": "count"}, "format": "decimal0"}],
            },
            expected_slots=1,
            evidence_text_by_id={"q": "共需核实2笔"},
            expected_slot_templates=("999999.99",),
        )
        self.assertEqual(result.answer_parts, ("2.00",))
        self.assertEqual(result.trace["outputs"][0]["requested_format"], "decimal0")
        self.assertEqual(result.trace["outputs"][0]["format"], "decimal2")

    def test_percent_points_are_converted_for_division_and_ratio_output(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "ebitda", "value": "338931", "unit": "百万元", "evidence_ids": ["e"]},
                    {"name": "rate", "value": "32.3", "unit": "%", "evidence_ids": ["e"]},
                    {"name": "revenue", "value": "1050187", "unit": "百万元", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {"id": "implied", "op": "div", "args": [{"ref": "ebitda"}, {"ref": "rate"}]},
                    {"id": "gap", "op": "sub", "args": [{"ref": "implied"}, {"ref": "revenue"}]},
                    {"id": "abs_gap", "op": "abs", "args": [{"ref": "gap"}]},
                    {"id": "relative", "op": "div", "args": [{"ref": "abs_gap"}, {"ref": "revenue"}]},
                ],
                "outputs": [
                    {"source": {"ref": "implied"}, "format": "decimal2"},
                    {"source": {"ref": "relative"}, "format": "percent2"},
                ],
            },
            expected_slots=2,
            evidence_text_by_id={"e": "EBITDA为338931百万元，EBITDA率32.3%，营业收入1050187百万元"},
            expected_slot_templates=("999999.99", "999999.99"),
        )
        self.assertEqual(result.answer_parts, ("1049321.98", "0.08"))
        self.assertEqual(
            result.trace["steps"][0]["unit_conversions"],
            [{"argument": "denominator", "from": "percent_points", "to": "ratio"}],
        )

    def test_percent_points_divided_by_percent_points_cancel_to_ratio(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "income", "value": "12", "unit": "万元", "evidence_ids": ["q"]},
                    {"name": "refund_rate", "value": "75", "unit": "%", "evidence_ids": ["e"]},
                    {"name": "percent_base", "value": "100", "unit": "%", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {
                        "id": "ratio",
                        "op": "div",
                        "args": [
                            {"ref": "refund_rate"},
                            {"ref": "percent_base"},
                        ],
                    },
                    {
                        "id": "refund",
                        "op": "mul",
                        "args": [{"ref": "income"}, {"ref": "ratio"}],
                    },
                ],
                "outputs": [{"source": {"ref": "refund"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"q": "累计收益12万元", "e": "返还75%，百分比基数100%"},
            expected_slot_templates=("999999.99",),
        )

        self.assertEqual(result.answer_parts, ("9.00",))
        self.assertEqual(result.trace["steps"][0]["value_kind"], "ratio")
        self.assertEqual(
            result.trace["steps"][0]["unit_conversions"],
            [
                {"argument": "numerator", "from": "percent_points", "to": "ratio"},
                {"argument": "denominator", "from": "percent_points", "to": "ratio"},
            ],
        )

    def test_amount_multiplied_by_percent_points_converts_percent_to_ratio(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "premium", "value": "100", "unit": "万元", "evidence_ids": ["q"]},
                    {"name": "rate", "value": "99", "unit": "%", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {
                        "id": "refund",
                        "op": "mul",
                        "args": [{"ref": "premium"}, {"ref": "rate"}],
                    }
                ],
                "outputs": [{"source": {"ref": "refund"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"q": "累计保费100万元", "e": "给付比例99%"},
            expected_slot_templates=("999999.99",),
        )

        self.assertEqual(result.answer_parts, ("99.00",))
        self.assertEqual(result.trace["steps"][0]["value_kind"], "amount")
        self.assertEqual(
            result.trace["steps"][0]["unit_conversions"],
            [{"argument": "argument_2", "from": "percent_points", "to": "ratio"}],
        )

    def test_common_named_argument_objects_and_percent_suffix_are_normalized(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "ebitda", "value": "338931", "unit": "百万元", "evidence_ids": ["e"]},
                    {"name": "rate", "value": "32.3%", "unit": "%", "evidence_ids": ["e"]},
                    {"name": "reported", "value": "1050187", "unit": "百万元", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {
                        "id": "implied",
                        "op": "div",
                        "args": {
                            "numerator": {"ref": "ebitda"},
                            "denominator": {"ref": "rate"},
                        },
                    },
                    {
                        "id": "deviation",
                        "op": "pct_change",
                        "args": {"new": {"ref": "implied"}, "old": {"ref": "reported"}},
                    },
                    {
                        "id": "absolute_deviation",
                        "op": "abs",
                        "args": {"value": {"ref": "deviation"}},
                    },
                ],
                "outputs": [
                    {"source": {"ref": "implied"}, "format": "decimal2"},
                    {"source": {"ref": "absolute_deviation"}, "format": "percent2"},
                ],
            },
            expected_slots=2,
            evidence_text_by_id={
                "e": "EBITDA为338931百万元，EBITDA率32.3%，营业收入1050187百万元"
            },
            expected_slot_templates=("999999.99", "999999.99"),
        )

        self.assertEqual(result.answer_parts, ("1049321.98", "0.08"))

    def test_pct_point_delta_converts_ratio_inputs_to_percentage_points(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "catl_cash", "value": "133219982", "evidence_ids": ["e"]},
                    {"name": "catl_revenue", "value": "423701834", "evidence_ids": ["e"]},
                    {"name": "midea_cash", "value": "53345930", "evidence_ids": ["e"]},
                    {"name": "midea_revenue", "value": "456451731", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {"id": "catl_rate", "op": "div", "args": [{"ref": "catl_cash"}, {"ref": "catl_revenue"}]},
                    {"id": "midea_rate", "op": "div", "args": [{"ref": "midea_cash"}, {"ref": "midea_revenue"}]},
                    {
                        "id": "delta_pp",
                        "op": "pct_point_delta",
                        "new": {"ref": "catl_rate"},
                        "old": {"ref": "midea_rate"},
                    },
                ],
                "outputs": [{"source": {"ref": "delta_pp"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={
                "e": "宁德时代现金流133219982、收入423701834；美的现金流53345930、收入456451731"
            },
            expected_slot_templates=("999999.99",),
        )

        self.assertEqual(result.answer_parts, ("19.75",))
        self.assertEqual(
            result.trace["steps"][2]["unit_conversions"],
            [
                {"argument": "new", "from": "ratio", "to": "percent_points"},
                {"argument": "old", "from": "ratio", "to": "percent_points"},
            ],
        )

    def test_dimensionless_one_minus_percent_points_uses_ratio_units(self):
        result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "one", "value": "1", "evidence_ids": ["q"]},
                    {"name": "debt_rate", "value": "61.17", "unit": "%", "evidence_ids": ["e"]},
                    {"name": "roe", "value": "19.70", "unit": "%", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {"id": "equity_ratio", "op": "sub", "args": [{"ref": "one"}, {"ref": "debt_rate"}]},
                    {"id": "equity_multiplier", "op": "div", "args": [{"ref": "one"}, {"ref": "equity_ratio"}]},
                    {"id": "approx_roa", "op": "div", "args": [{"ref": "roe"}, {"ref": "equity_multiplier"}]},
                ],
                "outputs": [
                    {"source": {"ref": "equity_multiplier"}, "format": "decimal2"},
                    {"source": {"ref": "approx_roa"}, "format": "decimal2"},
                ],
            },
            expected_slots=2,
            evidence_text_by_id={"q": "公式为1÷(1-资产负债率)", "e": "资产负债率61.17%，净资产收益率19.70%"},
            expected_slot_templates=("999999.99", "999999.99"),
        )

        self.assertEqual(result.answer_parts, ("2.58", "7.65"))
        self.assertEqual(
            result.trace["steps"][0]["unit_conversions"],
            [{"argument": "right", "from": "percent_points", "to": "ratio"}],
        )

    def test_readme_percent_suffix_formats_ratio_and_percent_points_correctly(self):
        ratio_result = CalculationExecutor().execute(
            {
                "variables": [
                    {"name": "gap", "value": "0.00082368", "evidence_ids": ["e"]},
                    {"name": "whole", "value": "1", "evidence_ids": ["e"]},
                ],
                "steps": [
                    {
                        "id": "relative",
                        "op": "div",
                        "args": [{"ref": "gap"}, {"ref": "whole"}],
                    }
                ],
                "outputs": [{"source": {"ref": "relative"}, "format": "percent2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"e": "绝对相对偏差分子为0.00082368，整体为1"},
            expected_slot_templates=("999999.99",),
            expected_numeric_decimal_places=2,
            expected_percent_suffixes=(True,),
        )
        points_result = CalculationExecutor().execute(
            {
                "variables": [
                    {
                        "name": "growth",
                        "value": "40.0461",
                        "unit": "%",
                        "evidence_ids": ["e"],
                    },
                ],
                "steps": [],
                "outputs": [{"source": {"ref": "growth"}, "format": "decimal2"}],
            },
            expected_slots=1,
            evidence_text_by_id={"e": "同比增幅为40.0461%"},
            expected_slot_templates=("999999.99",),
            expected_numeric_decimal_places=2,
            expected_percent_suffixes=(True,),
        )

        self.assertEqual(ratio_result.answer_parts, ("0.08%",))
        self.assertEqual(points_result.answer_parts, ("40.05%",))


if __name__ == "__main__":
    unittest.main()
