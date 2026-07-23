from __future__ import annotations

import unittest

from jsonschema import Draft202012Validator

from afa_agent.b_board.reasoning_schema import (
    REASONING_FEEDBACK_SCHEMA,
    REASONING_FEEDBACK_SCHEMA_VERSION,
    REASONING_REFINE_SCHEMA,
    REASONING_REFINE_SCHEMA_VERSION,
    normalize_reasoning_feedback_payload,
    normalize_reasoning_refine_payload,
    validate_reasoning_feedback_schema,
    validate_reasoning_refine_schema,
)


class ReasoningRefinementSchemaTests(unittest.TestCase):
    def test_versioned_schemas_are_valid_draft_2020_12(self) -> None:
        Draft202012Validator.check_schema(REASONING_FEEDBACK_SCHEMA)
        Draft202012Validator.check_schema(REASONING_REFINE_SCHEMA)
        self.assertEqual(
            REASONING_FEEDBACK_SCHEMA_VERSION,
            "reasoning_feedback_v1",
        )
        self.assertEqual(
            REASONING_REFINE_SCHEMA_VERSION,
            "reasoning_refine_v1",
        )

    def test_feedback_representation_drift_is_normalized(self) -> None:
        normalized, changes = normalize_reasoning_feedback_payload(
            {
                "logical_issues": "因果链缺少中间推导",
                "completeness_issues": None,
                "clarity_issues": [],
                "verification_questions": [],
                "must_preserve_facts": "证据直接支持A",
                "comment": "非契约字段",
            }
        )

        validate_reasoning_feedback_schema(normalized)
        self.assertEqual(
            normalized["logical_issues"],
            ["因果链缺少中间推导"],
        )
        self.assertEqual(normalized["completeness_issues"], [])
        self.assertEqual(
            normalized["must_preserve_facts"],
            ["证据直接支持A"],
        )
        self.assertEqual(
            {item["reason"] for item in changes},
            {
                "drop_noncontract_fields",
                "feedback_string_to_array",
                "null_feedback_field_to_empty_array",
            },
        )

    def test_refine_normalizer_only_accepts_exact_frozen_answer_shape(self) -> None:
        normalized, changes = normalize_reasoning_refine_payload(
            {
                "answer_parts": "A",
                "reasoning": "定位监管条件，证据直接支持题干陈述成立。",
                "comment": "非契约字段",
            },
            frozen_answer_parts=["A"],
        )

        validate_reasoning_refine_schema(normalized)
        self.assertEqual(normalized["answer_parts"], ["A"])
        self.assertTrue(normalized["reasoning"].endswith("最终答案为A。"))
        self.assertEqual(
            {item["reason"] for item in changes},
            {
                "drop_noncontract_fields",
                "single_frozen_answer_string_to_array",
                "append_exact_frozen_answer_conclusion",
            },
        )

        changed, _ = normalize_reasoning_refine_payload(
            {
                "answer_parts": "B",
                "reasoning": "这段内容试图改变冻结答案。",
            },
            frozen_answer_parts=["A"],
        )
        self.assertEqual(changed["answer_parts"], "B")
        with self.assertRaisesRegex(
            ValueError,
            "ReasoningRefine schema violation",
        ):
            validate_reasoning_refine_schema(changed)


if __name__ == "__main__":
    unittest.main()
