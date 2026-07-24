from __future__ import annotations

import json
import unittest

from afa_agent.b_board.calculation_profile import (
    CALCULATION_FIRST_ATTEMPT_EVIDENCE_POLICY_VERSION,
    CALCULATION_PROFILE_SCHEMA,
    CALCULATION_THINKING_POLICY_VERSION,
    CalculationProfile,
    build_calculation_profile_messages,
    infer_calculation_first_attempt_evidence_policy,
    infer_calculation_thinking_policy,
    parse_calculation_profile,
)


class CalculationProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "task_types": ["ratio", "arithmetic"],
            "required_facts": [
                {
                    "subject": "甲公司",
                    "metric": "资产合计",
                    "period": "目标年度",
                    "unit": "亿元",
                    "role": "input",
                    "source_kind": "disclosed_metric",
                },
                {
                    "subject": "甲公司",
                    "metric": "负债合计",
                    "period": "目标年度",
                    "unit": "亿元",
                    "role": "input",
                    "source_kind": "disclosed_metric",
                },
            ],
            "operators": ["div", "sub"],
            "outputs": [{"kind": "number", "format": "decimal2"}],
            "complexity": "medium",
            "risk_checks": [
                "period_binding",
                "entity_binding",
                "unit_conversion",
            ],
        }

    def test_profile_request_excludes_qid_and_history(self) -> None:
        messages = build_calculation_profile_messages(
            domain="financial_reports",
            question="根据报告计算目标年度指标。",
            answer_format="calculation",
            answer_slots=1,
            answer_slot_templates=("0.00",),
        )

        user_payload = json.loads(messages[1]["content"])
        self.assertEqual(
            set(user_payload),
            {
                "domain",
                "question",
                "answer_format",
                "answer_slots",
                "answer_slot_templates",
            },
        )
        self.assertNotIn("qid", messages[1]["content"])
        self.assertNotIn("answer_parts", messages[1]["content"])
        self.assertNotIn("doc_ids", messages[1]["content"])

    def test_parses_valid_profile_and_builds_fact_queries(self) -> None:
        profile = parse_calculation_profile(
            self.payload,
            domain="financial_reports",
            expected_output_count=1,
        )

        self.assertIsInstance(profile, CalculationProfile)
        self.assertEqual(profile.domain, "financial_reports")
        self.assertEqual(profile.complexity, "medium")
        self.assertEqual(
            profile.retrieval_queries(),
            (
                "甲公司 资产合计 目标年度 亿元",
                "甲公司 负债合计 目标年度 亿元",
            ),
        )

    def test_normalizes_duplicate_enums_without_changing_fact_text(self) -> None:
        payload = dict(self.payload)
        payload["task_types"] = ["ratio", " ratio ", "arithmetic"]
        payload["operators"] = ["div", "div", "sub"]

        profile = parse_calculation_profile(
            payload,
            domain="financial_reports",
            expected_output_count=1,
        )

        self.assertEqual(profile.task_types, ("ratio", "arithmetic"))
        self.assertEqual(profile.operators, ("div", "sub"))

    def test_source_kind_adds_general_risk_without_question_matching(
        self,
    ) -> None:
        payload = dict(self.payload)
        payload["required_facts"] = [
            {
                **self.payload["required_facts"][0],
                "source_kind": "table_row",
            }
        ]

        profile = parse_calculation_profile(
            payload,
            domain="financial_reports",
            expected_output_count=1,
        )

        self.assertIn("table_row_binding", profile.risk_checks)

    def test_rejects_output_slot_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "output count does not match answer slots",
        ):
            parse_calculation_profile(
                self.payload,
                domain="financial_reports",
                expected_output_count=2,
            )

    def test_rejects_unknown_task_type(self) -> None:
        payload = dict(self.payload)
        payload["task_types"] = ["historical_question_special_case"]

        with self.assertRaisesRegex(
            ValueError,
            "CalculationProfile schema violation",
        ):
            parse_calculation_profile(
                payload,
                domain="financial_reports",
                expected_output_count=1,
            )

    def test_native_schema_is_strict(self) -> None:
        self.assertFalse(CALCULATION_PROFILE_SCHEMA["additionalProperties"])
        fact_schema = CALCULATION_PROFILE_SCHEMA["properties"][
            "required_facts"
        ]["items"]
        self.assertFalse(fact_schema["additionalProperties"])

    def test_dense_self_contained_single_slot_disables_first_thinking(
        self,
    ) -> None:
        policy = infer_calculation_thinking_policy(
            domain="research",
            question=(
                "总额100万元，其中60%属于甲类，甲类的35%来自渠道；"
                "人均消费2960元，已有24.01万人，计算剩余人数。"
            ),
            answer_slots=1,
        )

        self.assertEqual(policy.mode, "off")
        self.assertIsNone(policy.thinking_budget)
        self.assertEqual(
            policy.version,
            CALCULATION_THINKING_POLICY_VERSION,
        )

    def test_dense_self_contained_question_uses_question_only_first(
        self,
    ) -> None:
        policy = infer_calculation_first_attempt_evidence_policy(
            domain="research",
            question=(
                "总额100万元，其中60%属于甲类，甲类的35%来自渠道；"
                "人均消费2960元，已有24.01万人，计算剩余人数。"
            ),
            answer_slots=1,
        )

        self.assertEqual(policy.mode, "question_only_first_attempt")
        self.assertEqual(policy.max_non_question_hits, 0)
        self.assertEqual(
            policy.version,
            CALCULATION_FIRST_ATTEMPT_EVIDENCE_POLICY_VERSION,
        )

    def test_external_source_reference_keeps_progressive_evidence(
        self,
    ) -> None:
        policy = infer_calculation_first_attempt_evidence_policy(
            domain="research",
            question=(
                "根据材料给出的100、60%、35%和2960，计算目标值。"
            ),
            answer_slots=1,
        )

        self.assertEqual(policy.mode, "progressive_retrieval")
        self.assertIsNone(policy.max_non_question_hits)

    def test_runtime_semantic_constraint_keeps_provider_default(
        self,
    ) -> None:
        policy = infer_calculation_thinking_policy(
            domain="research",
            question=(
                "总量与2025年持平，单位强度从当年水平提升至56，"
                "求2026年需求增速。"
            ),
            answer_slots=1,
            semantic_constraint_types=("unchanged_aggregate_volume",),
        )

        self.assertEqual(policy.mode, "default")

    def test_single_period_explicit_formula_disables_first_thinking(
        self,
    ) -> None:
        policy = infer_calculation_thinking_policy(
            domain="financial_reports",
            question=(
                "查阅甲公司2025年报告，按“指标=数值甲÷数值乙”"
                "计算指标，并核对该结果与报告值之间的偏差。"
            ),
            answer_slots=2,
        )

        self.assertEqual(policy.mode, "off")

    def test_non_reconciliation_formula_keeps_provider_default(self) -> None:
        policy = infer_calculation_thinking_policy(
            domain="financial_reports",
            question=(
                "查阅甲公司2025年报告，按“指标甲=数值甲÷数值乙”"
                "计算后，再按“指标乙=指标甲÷数值丙”计算第二个指标。"
            ),
            answer_slots=2,
        )

        self.assertEqual(policy.mode, "default")

    def test_single_period_formula_ranking_uses_bounded_first_attempt(
        self,
    ) -> None:
        policy = infer_calculation_thinking_policy(
            domain="financial_reports",
            question=(
                "根据甲、乙、丙三家公司2025年报告，分别按"
                "“派生值=1÷(1-原始比率)”计算并排序，"
                "同时给出最高与最低之差。"
            ),
            answer_slots=2,
        )

        self.assertEqual(policy.mode, "budget")
        self.assertEqual(policy.thinking_budget, 4096)

    def test_cross_period_formula_keeps_provider_default(self) -> None:
        policy = infer_calculation_thinking_policy(
            domain="financial_reports",
            question=(
                "查阅甲公司2024年和2025年报告，按“比率=分子÷分母”"
                "计算两期变化。"
            ),
            answer_slots=2,
        )

        self.assertEqual(policy.mode, "default")


if __name__ == "__main__":
    unittest.main()
