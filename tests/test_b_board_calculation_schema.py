from __future__ import annotations

import unittest

from jsonschema import Draft202012Validator

from afa_agent.b_board.calculation import CalculationExecutor
from afa_agent.b_board.calculation_schema import (
    CALCULATION_PLAN_SCHEMA,
    CALCULATION_PLAN_SCHEMA_VERSION,
    validate_calculation_plan_schema,
)
from afa_agent.b_board.runner import _normalize_calculation_plan_structure


class CalculationPlanSchemaTests(unittest.TestCase):
    def test_versioned_schema_is_valid_draft_2020_12(self) -> None:
        Draft202012Validator.check_schema(CALCULATION_PLAN_SCHEMA)
        self.assertEqual(
            CALCULATION_PLAN_SCHEMA_VERSION,
            "calculation_plan_v1",
        )

    def test_deterministic_repairs_run_before_local_schema_gate(self) -> None:
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
            ],
            "steps": [
                {
                    "id": "单价",
                    "op": "div",
                    "args": {
                        "a": {"ref": "金额"},
                        "b": {"ref": "数量"},
                    },
                },
                {
                    "id": "放大",
                    "op": "mul",
                    "args": [
                        {"ref": "单价"},
                        {"value": "100", "value_type": "decimal"},
                    ],
                },
            ],
            "outputs": [{"source": "放大", "format": "decimal2"}],
        }

        normalized, changes = _normalize_calculation_plan_structure(plan)
        validate_calculation_plan_schema(normalized)
        result = CalculationExecutor().execute(
            normalized,
            expected_slots=1,
            evidence_text_by_id={"u1": "金额为10元，数量为2。"},
        )

        self.assertEqual(result.answer_parts, ("500.00",))
        reasons = {item["reason"] for item in changes}
        self.assertIn("named_args_to_ordered_schema", reasons)
        self.assertIn("value_object_to_literal", reasons)
        self.assertIn("numeric_literal_default_empty_unit", reasons)
        self.assertIn("default_empty_supporting_evidence_ids", reasons)
        self.assertIn("default_empty_decision_summary", reasons)

    def test_ambiguous_directional_shape_reaches_retry_gate(self) -> None:
        plan = {
            "variables": [
                {
                    "name": "新值",
                    "value": "110",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
                {
                    "name": "旧值",
                    "value": "100",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["u1"],
                },
            ],
            "steps": [
                {
                    "id": "增幅",
                    "op": "pct_change",
                    "args": [{"ref": "新值"}, {"ref": "旧值"}],
                }
            ],
            "outputs": [{"source": "增幅", "format": "percent2"}],
            "supporting_evidence_ids": [],
            "decision_summary": "",
        }

        normalized, _ = _normalize_calculation_plan_structure(plan)
        with self.assertRaisesRegex(
            ValueError,
            "CalculationPlan schema violation",
        ):
            validate_calculation_plan_schema(normalized)


if __name__ == "__main__":
    unittest.main()
