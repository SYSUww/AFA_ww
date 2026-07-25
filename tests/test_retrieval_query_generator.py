from __future__ import annotations

from dataclasses import fields
import inspect
import re
import unittest

from afa_agent.models import Question
import afa_agent.retrieval_query as retrieval_query_module
from afa_agent.retrieval_query import (
    EVIDENCE_OBLIGATIONS_QUERY_PLAN_VERSION,
    QUERY_PLAN_VERSION,
    RetrievalRequest,
    generate_retrieval_plan,
)
from afa_agent.strategy import build_query_variants


class RetrievalQueryInterfaceTests(unittest.TestCase):
    def test_request_interface_cannot_receive_qid_or_frozen_answer(self) -> None:
        field_names = {field.name for field in fields(RetrievalRequest)}

        self.assertEqual(
            field_names,
            {
                "domain",
                "question",
                "option_text",
                "question_type",
                "answer_format",
                "document_hints",
            },
        )
        self.assertTrue(
            {
                "qid",
                "label",
                "expected_answer",
                "pred_answer",
                "answer_parts",
                "doc_id",
                "evidence_id",
            }.isdisjoint(field_names)
        )

    def test_coordinated_annual_report_subjects_become_separate_anchors(self) -> None:
        plan = generate_retrieval_plan(
            RetrievalRequest(
                domain="financial_reports",
                question=(
                    "根据甲科技、乙集团、丙银行和丁建筑 2025 年年度报告中的"
                    "现金分红数据，以下判断是否正确？"
                ),
                option_text="按每10股现金分红由高到低排序。",
                question_type="判断题",
                answer_format="tf",
            )
        )

        self.assertEqual(
            plan.slots.anchors[:4],
            ("甲科技", "乙集团", "丙银行", "丁建筑"),
        )

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            RetrievalRequest(domain="insurance", question=" ")

    def test_query_budget_must_be_positive(self) -> None:
        request = RetrievalRequest(domain="research", question="行业需求是否增长？")
        with self.assertRaisesRegex(ValueError, "max_queries must be positive"):
            generate_retrieval_plan(request, max_queries=0)

    def test_generator_source_contains_no_board_qid_literal(self) -> None:
        source = inspect.getsource(retrieval_query_module)

        self.assertIsNone(re.search(r"(?:fc|fin|ins|reg|res)_[ab]_\d{3}", source))


class RetrievalQueryPlanTests(unittest.TestCase):
    def test_obligation_plan_extracts_clean_multi_report_entities(self) -> None:
        request = RetrievalRequest(
            domain="financial_reports",
            question=(
                "根据宁德时代与美的集团 2024 年、2025 年年度报告中的"
                "营业收入和经营活动产生的现金流量净额计算差额。"
            ),
            question_type="计算题",
            answer_format="calculation",
        )

        plan = generate_retrieval_plan(
            request,
            plan_strategy=EVIDENCE_OBLIGATIONS_QUERY_PLAN_VERSION,
        )

        self.assertEqual(plan.slots.anchors, ("宁德时代", "美的集团"))
        serialized = "\n".join(plan.query_strings())
        self.assertNotIn("美的集团 2024 年", plan.slots.anchors)
        self.assertNotIn("宁德时代与美的集团", plan.slots.anchors)
        self.assertNotIn("：查阅", serialized)

    def test_obligation_plan_binds_insurance_product_and_synonyms(self) -> None:
        plan = generate_retrieval_plan(
            RetrievalRequest(
                domain="insurance",
                question="关于保单贷款或借款，下列说法正确的是？",
                option_text=(
                    "平安富鸿金生非按个人养老金制度投保时可申请保单贷款"
                ),
                answer_format="multi",
            ),
            plan_strategy=EVIDENCE_OBLIGATIONS_QUERY_PLAN_VERSION,
        )

        self.assertEqual(plan.slots.anchors, ("平安富鸿金生",))
        self.assertIn("保险单借款", plan.slots.topics)
        support = [
            item.query for item in plan.variants if item.channel == "support"
        ]
        self.assertTrue(all("平安富鸿金生" in query for query in support))

    def test_obligation_calculation_budget_covers_each_entity_first(self) -> None:
        plan = generate_retrieval_plan(
            RetrievalRequest(
                domain="financial_reports",
                question=(
                    "计算题：查阅甲公司、乙公司、丙公司和丁公司2025年"
                    "年度报告中的现金分红，统一换算并排序。"
                ),
                question_type="计算题",
                answer_format="calculation",
            ),
            max_queries=6,
            plan_strategy=EVIDENCE_OBLIGATIONS_QUERY_PLAN_VERSION,
        )

        coverage = [
            item.query for item in plan.variants if item.channel == "coverage"
        ]
        self.assertEqual(len(coverage), 4)
        for entity in ("甲公司", "乙公司", "丙公司", "丁公司"):
            self.assertTrue(any(entity in query for query in coverage))

    def test_obligation_plan_rejects_document_hints(self) -> None:
        with self.assertRaisesRegex(ValueError, "rejects document_hints"):
            generate_retrieval_plan(
                RetrievalRequest(
                    domain="financial_reports",
                    question="根据甲公司2025年年度报告核验营业收入。",
                    document_hints=("annual_company_2025_report",),
                ),
                plan_strategy=EVIDENCE_OBLIGATIONS_QUERY_PLAN_VERSION,
            )

    def test_financial_claim_has_support_broad_and_contrast_queries(self) -> None:
        request = RetrievalRequest(
            domain="financial_reports",
            question="根据美的集团2025年年度报告，下列判断是否正确？",
            option_text="2025年资产负债率为62.33%，且较上年增加",
            answer_format="tf",
        )

        plan = generate_retrieval_plan(request)
        by_channel = {variant.channel: variant.query for variant in plan.variants}
        support_queries = [variant.query for variant in plan.variants if variant.channel == "support"]

        self.assertEqual(plan.version, QUERY_PLAN_VERSION)
        self.assertIn("美的集团", plan.slots.anchors)
        self.assertIn("2025年", plan.slots.periods)
        self.assertIn("资产负债率", plan.slots.topics)
        self.assertTrue(any("62.33%" in query for query in support_queries))
        self.assertNotIn("62.33%", by_channel["broad"])
        self.assertIn("减少", by_channel["contrast"])

    def test_insurance_formula_slots_preserve_operator_semantics(self) -> None:
        request = RetrievalRequest(
            domain="insurance",
            question="分别核验各产品的身故保险金计算规则。",
            option_text=(
                "国寿增益宝中，40周岁的身故保险金取基本保险金额乘给付比例"
                "与个人账户价值的较大值"
            ),
            answer_format="multi",
        )

        plan = generate_retrieval_plan(request)

        self.assertIn("国寿增益宝", plan.slots.anchors)
        self.assertIn("身故保险金", plan.slots.topics)
        self.assertIn("给付比例", plan.slots.topics)
        self.assertIn("较大值", plan.slots.topics)
        self.assertTrue(any(item.channel == "contrast" for item in plan.variants))

    def test_negative_existence_claim_adds_scope_check(self) -> None:
        request = RetrievalRequest(
            domain="insurance",
            question="核验责任免除条款。",
            option_text="该产品未提及地震责任免除",
            answer_format="multi",
        )

        plan = generate_retrieval_plan(request)
        scope_queries = [item.query for item in plan.variants if item.channel == "scope_check"]

        self.assertTrue(plan.requires_scope_check)
        self.assertEqual(len(scope_queries), 1)
        self.assertIn("完整条款", scope_queries[0])
        self.assertIn("适用范围", scope_queries[0])
        self.assertIn("例外", scope_queries[0])

    def test_calculation_generates_per_entity_coverage_queries(self) -> None:
        request = RetrievalRequest(
            domain="financial_reports",
            question=(
                "查阅宁德时代股份有限公司、美的集团2025年年度报告，"
                "使用营业收入和经营活动产生的现金流量净额计算经营现金流率。"
            ),
            question_type="计算题",
            answer_format="calculation",
        )

        plan = generate_retrieval_plan(request)
        coverage_queries = [item.query for item in plan.variants if item.channel == "coverage"]

        self.assertEqual(set(plan.slots.anchors), {"宁德时代股份有限公司", "美的集团"})
        self.assertEqual(len(coverage_queries), 2)
        for query in coverage_queries:
            self.assertIn("2025年", query)
            self.assertIn("营业收入", query)
            self.assertIn("经营活动产生的现金流量净额", query)

    def test_regulatory_plan_preserves_deadline_modal_and_action(self) -> None:
        request = RetrievalRequest(
            domain="regulatory",
            question="关于支付机构调整收费标准，下列说法正确的是？",
            option_text="调整收费标准应当至少提前30个自然日持续公示",
            answer_format="multi",
        )

        plan = generate_retrieval_plan(request)

        self.assertIn("30个自然日", plan.slots.periods)
        self.assertIn("至少", plan.slots.scopes)
        self.assertIn("公示", plan.slots.relations)
        self.assertTrue(any(item.channel == "contrast" for item in plan.variants))


class StrategyAdapterTests(unittest.TestCase):
    @staticmethod
    def make_question(qid: str) -> Question:
        return Question(
            qid=qid,
            domain="financial_reports",
            split="B",
            question="根据美的集团2025年年度报告核验资产负债率。",
            options={"A": "资产负债率为61.17%"},
            answer_format="mcq",
            type="单选题",
            doc_ids=["annual_midea_2025_report"],
        )

    def test_semantic_generator_is_qid_invariant(self) -> None:
        settings = {
            "query_generator": "semantic_slots_v1",
            "semantic_query_max_variants": 8,
            "include_doc_id_hint": False,
        }

        first = build_query_variants(self.make_question("first_qid"), "A", "资产负债率为61.17%", settings)
        second = build_query_variants(
            self.make_question("completely_different_qid"),
            "A",
            "资产负债率为61.17%",
            settings,
        )

        self.assertEqual(first, second)
        self.assertFalse(any("first_qid" in query or "completely_different_qid" in query for query in first))

    def test_legacy_generator_remains_default(self) -> None:
        question = self.make_question("legacy")

        variants = build_query_variants(
            question,
            "A",
            "资产负债率为61.17%",
            {"query_mode": "question_option", "include_question_type": True},
        )

        self.assertEqual(
            variants[0],
            "根据美的集团2025年年度报告核验资产负债率。\n资产负债率为61.17%",
        )
        self.assertIn("单选题", variants[1])

    def test_unknown_query_generator_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported query_generator"):
            build_query_variants(
                self.make_question("unknown"),
                "A",
                "资产负债率为61.17%",
                {"query_generator": "typo"},
            )


if __name__ == "__main__":
    unittest.main()
