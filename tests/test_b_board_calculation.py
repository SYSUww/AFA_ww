from __future__ import annotations

import unittest
from datetime import date

from afa_agent.b_board.calculation import (
    BusinessCalendar,
    CalculationExecutor,
    CalculationPlanError,
)


class BBoardCalculationTests(unittest.TestCase):
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
                    {"id": "growth", "op": "pct_change", "args": [{"ref": "new"}, {"ref": "old"}]},
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


if __name__ == "__main__":
    unittest.main()
