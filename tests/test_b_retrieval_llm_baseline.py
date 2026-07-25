from __future__ import annotations

import json
from pathlib import Path
import unittest

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.retrieval_llm_baseline import (
    _best_metric_evidence_id,
    _select_document_candidates,
    _reserved_evidence_ids,
    _required_metric_slots,
    _resolve_evidence_quota_strategy,
    build_answer_messages,
    build_answer_schema,
    build_frozen_answer_reasoning_messages,
    build_frozen_answer_reasoning_schema,
    build_reasoning_canonical_schema,
    build_retrieval_bundle,
    prepare_evidence_payload,
    validate_answer_payload,
    validate_answer_parts_shape,
    validate_answer_shape_payload,
    validate_frozen_answer_reasoning_payload,
    validate_joint_payload_with_format_recovery,
    validate_reasoning_canonical_payload,
)
from afa_agent.models import RetrievalHit


class RetrievalLLMBaselineTests(unittest.TestCase):
    @staticmethod
    def question(
        *,
        qid: str = "sentinel_qid_never_prompted",
        answer_format: str = "multi",
        question_type: str = "多选题",
        slots: int = 1,
        templates: tuple[str, ...] = ("AB",),
        options: dict[str, str] | None = None,
        question_text: str = "核验两个产品是否包含责任免除条款。",
    ) -> BQuestion:
        return BQuestion(
            qid=qid,
            domain="insurance",
            split="B",
            question=question_text,
            options=options
            if options is not None
            else {"A": "甲包含", "B": "乙不包含", "C": "两者均包含"},
            answer_format=answer_format,
            type=question_type,
            answer_slots=slots,
            answer_slot_templates=templates,
        )

    @staticmethod
    def evidence() -> list[dict[str, object]]:
        return [
            {
                "evidence_key": "E01",
                "source_key": "S01",
                "evidence_id": "very_long_doc::sec_1",
                "doc_id": "very_long_doc",
                "rank": 1,
                "title_path": ["保险条款", "责任免除"],
                "text": "甲产品和乙产品的责任免除原文。",
            }
        ]

    def test_model_messages_do_not_contain_qid_or_reference(self) -> None:
        question = self.question()
        messages = build_answer_messages(question, self.evidence())
        serialized = json.dumps(messages, ensure_ascii=False)

        self.assertNotIn(question.qid, serialized)
        self.assertNotIn("pseudo99", serialized)
        self.assertNotIn("官网答案", serialized)
        self.assertNotIn("very_long_doc::sec_1", serialized)
        self.assertIn("[E01|S01]", serialized)

    def test_retry_message_does_not_leak_qid(self) -> None:
        question = self.question()
        messages = build_answer_messages(
            question,
            self.evidence(),
            validation_error="answer_shape_error",
            previous_response='{"answer_parts":["A"]}',
        )
        self.assertNotIn(question.qid, json.dumps(messages, ensure_ascii=False))

    def test_retrieval_bundle_is_qid_invariant(self) -> None:
        first = build_retrieval_bundle(self.question(qid="first"))
        second = build_retrieval_bundle(self.question(qid="second"))

        self.assertEqual(first.primary_queries, second.primary_queries)
        self.assertEqual(first.supplemental_queries, second.supplemental_queries)

    def test_anchor_first_document_candidates_do_not_union_discovery(self) -> None:
        selected = _select_document_candidates(
            ["anchored_a", "anchored_b"],
            ["discovered_x", "anchored_a", "discovered_y"],
            max_doc_candidates=3,
            strategy="anchor_first",
        )

        self.assertEqual(selected, ["anchored_a", "anchored_b"])

    def test_anchor_first_falls_back_when_no_anchor_exists(self) -> None:
        selected = _select_document_candidates(
            [],
            ["discovered_x", "discovered_y"],
            max_doc_candidates=1,
            strategy="anchor_first",
        )

        self.assertEqual(selected, ["discovered_x"])

    def test_anchor_union_preserves_current_default_behavior(self) -> None:
        selected = _select_document_candidates(
            ["anchored_a"],
            ["discovered_x", "anchored_a", "discovered_y"],
            max_doc_candidates=3,
            strategy="anchor_union",
        )

        self.assertEqual(
            selected,
            ["anchored_a", "discovered_x", "discovered_y"],
        )

    def test_document_balanced_quota_reserves_two_items_per_document(
        self,
    ) -> None:
        reserved = _reserved_evidence_ids(
            {"ranked_ids": ["primary_1", "primary_2", "primary_3"]},
            document_rankings=[
                {"ranked_ids": ["doc_a_1", "doc_a_2", "doc_a_3"]},
                {"ranked_ids": ["doc_b_1", "doc_b_2", "doc_b_3"]},
            ],
            option_rankings=[
                {"ranked_ids": ["option_a_1", "option_a_2"]},
            ],
            final_top_k=10,
            strategy="document_balanced",
        )

        self.assertEqual(
            reserved,
            [
                "doc_a_1",
                "doc_a_2",
                "doc_b_1",
                "doc_b_2",
                "option_a_1",
            ],
        )
        self.assertNotIn("primary_1", reserved)

    def test_primary_guard_quota_preserves_current_reservation_order(
        self,
    ) -> None:
        reserved = _reserved_evidence_ids(
            {
                "ranked_ids": [
                    "primary_1",
                    "primary_2",
                    "primary_3",
                    "primary_4",
                    "primary_5",
                ]
            },
            document_rankings=[
                {"ranked_ids": ["doc_a_1", "doc_a_2"]},
                {"ranked_ids": ["doc_b_1", "doc_b_2"]},
            ],
            option_rankings=[{"ranked_ids": ["option_a_1"]}],
            final_top_k=10,
            strategy="primary_guard",
        )

        self.assertEqual(
            reserved,
            [
                "primary_1",
                "primary_2",
                "primary_3",
                "primary_4",
                "doc_a_1",
                "doc_b_1",
                "option_a_1",
            ],
        )

    def test_metric_slot_coverage_extracts_disclosed_operands_only(self) -> None:
        question = BQuestion(
            qid="generic_metric_case",
            domain="financial_reports",
            split="B",
            question=(
                "计算题：根据甲公司与乙公司2025年年度报告中的营业收入和"
                "经营活动产生的现金流量净额，计算经营现金流率并排序。"
            ),
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=2,
            answer_slot_templates=("甲>乙", "0.00"),
        )
        bundle = build_retrieval_bundle(question)

        slots = _required_metric_slots(question, bundle.plans)

        self.assertEqual(
            slots,
            ["营业收入", "经营活动产生的现金流量净额"],
        )
        self.assertNotIn("经营现金流", slots)

    def test_metric_slot_quota_reserves_each_entity_metric_hit(self) -> None:
        def metric_ranking(unit_id: str, metric: str) -> dict[str, object]:
            return {
                "metric_slot": metric,
                "ranked_ids": [unit_id],
                "ranked_hits": [
                    RetrievalHit(
                        unit_id=unit_id,
                        doc_id=unit_id.split("_metric_", 1)[0],
                        score=1.0,
                        title_path=[metric],
                        text=f"{metric} | 500 | 450",
                    )
                ],
            }

        reserved = _reserved_evidence_ids(
            {
                "ranked_ids": [
                    "primary_1",
                    "primary_2",
                    "primary_3",
                ]
            },
            document_rankings=[
                {"ranked_ids": ["generic_a"]},
                {"ranked_ids": ["generic_b"]},
            ],
            metric_slot_rankings=[
                metric_ranking("entity_a_metric_1", "营业收入"),
                metric_ranking("entity_a_metric_2", "经营现金流"),
                metric_ranking("entity_b_metric_1", "营业收入"),
                metric_ranking("entity_b_metric_2", "经营现金流"),
            ],
            option_rankings=[],
            final_top_k=10,
            strategy="metric_slot_coverage",
        )

        self.assertEqual(
            reserved,
            [
                "primary_1",
                "primary_2",
                "entity_a_metric_1",
                "entity_a_metric_2",
                "entity_b_metric_1",
                "entity_b_metric_2",
            ],
        )
        self.assertNotIn("generic_a", reserved)

    def test_metric_value_binding_rejects_threshold_and_header_rows(self) -> None:
        ranking = {
            "metric_slot": "资产负债率",
            "ranked_ids": ["header", "guarantee", "actual"],
            "ranked_hits": [
                RetrievalHit(
                    "header",
                    "report",
                    3.0,
                    ["资产负债率"],
                    "项目 | 2025年 | 2024年",
                ),
                RetrievalHit(
                    "guarantee",
                    "report",
                    2.0,
                    ["担保"],
                    "被担保对象资产负债率超过70%，担保比例为19.78%。",
                ),
                RetrievalHit(
                    "actual",
                    "report",
                    1.0,
                    ["主要会计数据"],
                    "资产负债率 | 61.94% | 60.21%",
                ),
            ],
        }

        self.assertEqual(_best_metric_evidence_id(ranking), "actual")

    def test_adaptive_quota_activates_only_for_multi_report_calculation(
        self,
    ) -> None:
        multi_report = BQuestion(
            qid="not_used_for_routing",
            domain="financial_reports",
            split="B",
            question="根据甲乙两份年报计算差额。",
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("0.00",),
        )

        self.assertEqual(
            _resolve_evidence_quota_strategy(
                multi_report,
                selected_doc_ids=["report_a", "report_b"],
                strategy="adaptive_multi_report_calculation",
            ),
            "document_balanced",
        )
        self.assertEqual(
            _resolve_evidence_quota_strategy(
                multi_report,
                selected_doc_ids=["report_a"],
                strategy="adaptive_multi_report_calculation",
            ),
            "primary_guard",
        )
        self.assertEqual(
            _resolve_evidence_quota_strategy(
                self.question(),
                selected_doc_ids=["contract_a", "contract_b"],
                strategy="adaptive_multi_report_calculation",
            ),
            "primary_guard",
        )

    def test_schema_only_contains_final_submission_fields(self) -> None:
        schema = build_answer_schema(self.question())

        self.assertEqual(
            list(schema["properties"]),
            ["reasoning", "answer_parts"],
        )
        self.assertEqual(
            set(schema["required"]),
            {"reasoning", "answer_parts"},
        )
        self.assertNotIn("option_assessments", json.dumps(schema))
        legal = schema["properties"]["answer_parts"]["items"]["enum"]
        self.assertIn("AC", legal)
        self.assertNotIn("A", legal)

    def test_reasoning_canonical_schema_has_one_model_owned_field(self) -> None:
        schema = build_reasoning_canonical_schema()

        self.assertEqual(set(schema["properties"]), {"reasoning"})
        self.assertEqual(schema["required"], ["reasoning"])
        self.assertFalse(schema["additionalProperties"])

    def test_reasoning_canonical_extracts_multi_answer_without_rewrite(self) -> None:
        question = self.question()
        reasoning = (
            "材料分别说明甲与丙满足责任免除条件，乙的适用范围不符合题设，"
            "因此选择满足条件的两个选项。结论：AC"
        )

        parsed = validate_reasoning_canonical_payload(
            question,
            {"reasoning": reasoning},
        )

        self.assertEqual(parsed["answer_parts"], ["AC"])
        self.assertEqual(parsed["reasoning"], reasoning)
        self.assertFalse(parsed["decision_trace"]["answer_modified"])
        self.assertFalse(parsed["decision_trace"]["reasoning_modified"])

    def test_reasoning_canonical_rejects_one_label_for_multi_question(self) -> None:
        with self.assertRaises(ValueError):
            validate_reasoning_canonical_payload(
                self.question(),
                {
                    "reasoning": (
                        "材料只支持第一项，其他选项均不满足条件，"
                        "所以最终仅选择第一项。结论：A"
                    )
                },
            )

    def test_reasoning_canonical_extracts_multiple_slots_exactly(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=2,
            templates=("999999.99", "999999.99%"),
            options={},
        )
        reasoning = (
            "由材料数值代入题设公式，第一项为1.23，第二项按要求保留"
            "两位小数并带百分号为4.56%。结论：1.23；4.56%"
        )

        parsed = validate_reasoning_canonical_payload(
            question,
            {"reasoning": reasoning},
        )

        self.assertEqual(parsed["answer_parts"], ["1.23", "4.56%"])
        self.assertEqual(parsed["reasoning"], reasoning)

    def test_reasoning_canonical_prompt_does_not_request_answer_parts(self) -> None:
        serialized = json.dumps(
            build_answer_messages(
                self.question(),
                self.evidence(),
                output_contract="reasoning_canonical",
            ),
            ensure_ascii=False,
        )

        self.assertNotIn("answer_parts", serialized)
        self.assertIn("唯一答案来源", serialized)

    def test_reasoning_canonical_retry_keeps_single_field_contract(self) -> None:
        serialized = json.dumps(
            build_answer_messages(
                self.question(),
                self.evidence(),
                output_contract="reasoning_canonical",
                validation_error="answer_shape_error",
                previous_response='{"reasoning":"证据不足。结论：A"}',
            ),
            ensure_ascii=False,
        )

        self.assertNotIn("answer_parts", serialized)
        self.assertNotIn("两个字段", serialized)
        self.assertIn("不得插入分号", serialized)

    def test_reasoning_canonical_rejects_hidden_normalization(self) -> None:
        for conclusion in ("AC。", " AC", "AC "):
            with self.subTest(conclusion=conclusion):
                with self.assertRaises(ValueError):
                    validate_reasoning_canonical_payload(
                        self.question(),
                        {
                            "reasoning": (
                                "材料支持第一项和第三项，第二项不符合适用范围，"
                                f"因此选择两项。结论：{conclusion}"
                            )
                        },
                    )

    def test_calculation_schema_binds_slot_count(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=2,
            templates=("0.00", "0.00"),
            options={},
        )
        schema = build_answer_schema(question)
        answer_parts = schema["properties"]["answer_parts"]

        self.assertEqual(answer_parts["minItems"], 2)
        self.assertEqual(answer_parts["maxItems"], 2)
        self.assertEqual(answer_parts["items"]["type"], "string")
        self.assertEqual(answer_parts["items"]["minLength"], 2)
        self.assertNotIn("pattern", answer_parts["items"])

    def test_calculation_punctuation_only_parts_are_rejected_locally(self) -> None:
        numeric_question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
        )

        for invalid in (">", "；", "||"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "punctuation-only"):
                    validate_answer_parts_shape(numeric_question, [invalid])

        valid_cases = (
            (
                self.question(
                    answer_format="calculation",
                    question_type="计算题",
                    slots=1,
                    templates=("999999.99",),
                    options={},
                    question_text="期满后次一工作日是哪一天？",
                ),
                "2026年3月27日",
            ),
            (
                self.question(
                    answer_format="calculation",
                    question_type="计算题",
                    slots=1,
                    templates=("999999.99%",),
                    options={},
                ),
                "12.34%",
            ),
            (
                self.question(
                    answer_format="calculation",
                    question_type="计算题",
                    slots=1,
                    templates=("公司名称>公司名称",),
                    options={},
                ),
                "甲公司>乙公司",
            ),
        )
        for question, valid in valid_cases:
            with self.subTest(valid=valid):
                self.assertEqual(
                    validate_answer_parts_shape(question, [valid]), [valid]
                )

    def test_prompt_is_modular_by_question_type(self) -> None:
        choice = json.dumps(
            build_answer_messages(self.question(), self.evidence()),
            ensure_ascii=False,
        )
        calculation_question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("0.00",),
            options={},
        )
        calculation = json.dumps(
            build_answer_messages(calculation_question, self.evidence()),
            ensure_ascii=False,
        )

        self.assertIn("多选题必须选择至少两个", choice)
        self.assertNotIn("必要公式、关键代入值", choice)
        self.assertIn("必要公式、关键代入值", calculation)
        self.assertNotIn("多选题必须选择至少两个", calculation)

    def test_calculation_prompt_derives_format_without_leaking_placeholder(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="计算最终金额约为多少亿元？",
        )
        serialized = json.dumps(
            build_answer_messages(question, self.evidence()),
            ensure_ascii=False,
        )

        self.assertIn("恰好保留2位小数", serialized)
        self.assertIn("不得带百分号", serialized)
        self.assertNotIn("999999.99", serialized)

    def test_mixed_ordering_slot_is_not_described_as_decimal(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=2,
            templates=("公司名称>公司名称", "999999.99"),
            options={},
            question_text=(
                "按指标从高到低排序，并计算第一名比第二名高多少个百分点，"
                "差值保留两位小数。"
            ),
        )
        serialized = json.dumps(
            build_answer_messages(question, self.evidence()),
            ensure_ascii=False,
        )

        self.assertIn("槽1：按题面要求填写非空名称", serialized)
        self.assertIn("半角大于号>", serialized)
        self.assertIn("槽2：数值必须恰好保留2位小数", serialized)

    def test_date_question_is_not_described_as_decimal(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="若第60日为2026年3月27日，期满后次一工作日是哪一天？",
        )
        serialized = json.dumps(
            build_answer_messages(question, self.evidence()),
            ensure_ascii=False,
        )

        self.assertIn("有效中文日期", serialized)
        self.assertNotIn("恰好保留2位小数", serialized)

    def test_valid_payload_is_preserved(self) -> None:
        reasoning = (
            "材料显示甲产品包含责任免除条款，乙产品的表述不满足题设条件，"
            "综合核验应选择第一项与第三项。结论：AC"
        )
        normalized = validate_answer_payload(
            self.question(),
            {"answer_parts": ["AC"], "reasoning": reasoning},
        )

        self.assertEqual(normalized["answer_parts"], ["AC"])
        self.assertEqual(normalized["reasoning"], reasoning)

    def test_multi_choice_single_letter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least two"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["A"],
                    "reasoning": "材料逐项核验后仅第一项成立，但该题要求多选。结论：A",
                },
            )

    def test_reasoning_without_explicit_conclusion_is_preserved(self) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="计算目标指标并保留两位小数。",
        )
        reasoning = (
            "将材料披露的分子与分母代入公式完成复核，最终答案为12.34。"
        )
        parsed = validate_answer_payload(
            question,
            {
                "answer_parts": ["12.34"],
                "reasoning": reasoning,
            },
        )

        self.assertEqual(parsed["answer_parts"], ["12.34"])
        self.assertEqual(parsed["reasoning"], reasoning)
        self.assertEqual(
            parsed["decision_trace"]["postprocessing_mode"],
            "reasoning_without_explicit_conclusion",
        )
        self.assertFalse(parsed["decision_trace"]["answer_modified"])
        self.assertFalse(parsed["decision_trace"]["reasoning_modified"])
        self.assertFalse(parsed["decision_trace"]["semantic_correction"])

    def test_reasoning_without_marker_rejects_conflicting_explicit_answer_cue(
        self,
    ) -> None:
        for reasoning in (
            "材料显示第一项和第三项成立，第二项与原文不符，因此最终答案为B。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此应该选择B。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此答案应为B。",
            "材料显示第一项和第三项成立，第二项与原文不符，答案应该是B。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此结论是B。",
            "材料显示第一项和第三项成立，第二项与原文不符，最终结果为B。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此只有B项正确。",
            "材料显示第一项和第三项成立，第二项与原文不符，最终应为B。",
            "材料显示第一项和第三项成立，第二项与原文不符，应选B项。",
            "材料显示第一项和第三项成立，第二项与原文不符，最终选择B选项。",
            "材料显示第一项和第三项成立，第二项与原文不符，应该选择选项B。",
            "材料显示第一项和第三项成立，第二项与原文不符，只有B一项正确。",
            "材料显示第一项和第三项成立，第二项与原文不符，B是正确的。",
            "材料显示第一项和第三项成立，第二项与原文不符，最终确定B。",
            "材料显示第一项和第三项成立，第二项与原文不符，证据中仅B可确认。",
        ):
            with self.subTest(reasoning=reasoning):
                with self.assertRaises(ValueError):
                    validate_answer_payload(
                        self.question(),
                        {
                            "answer_parts": ["AC"],
                            "reasoning": reasoning,
                        },
                    )

    def test_choice_reasoning_without_marker_or_answer_cue_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "matching answer cue"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["AC"],
                    "reasoning": (
                        "材料显示第一项和第三项成立，第二项与原文不符。"
                    ),
                },
            )

    def test_reasoning_without_marker_accepts_matching_explicit_answer_cue(
        self,
    ) -> None:
        for reasoning in (
            "材料显示第一项和第三项成立，第二项与原文不符，因此最终答案为A、C。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此应该选择A、C。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此答案应为A和C。",
            "材料显示第一项和第三项成立，第二项与原文不符，因此A、C项均正确。",
        ):
            with self.subTest(reasoning=reasoning):
                parsed = validate_answer_payload(
                    self.question(),
                    {
                        "answer_parts": ["AC"],
                        "reasoning": reasoning,
                    },
                )

                self.assertEqual(parsed["answer_parts"], ["AC"])
                self.assertEqual(parsed["reasoning"], reasoning)
                self.assertEqual(
                    parsed["decision_trace"]["postprocessing_mode"],
                    "reasoning_without_explicit_conclusion",
                )

    def test_reasoning_without_marker_rejects_unparseable_answer_tail(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "explicit answer cue"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["AC"],
                    "reasoning": (
                        "材料显示第一项和第三项成立，第二项与原文不符，"
                        "最终答案见上述分析。"
                    ),
                },
            )

    def test_reasoning_without_marker_rejects_negative_or_intermediate_cue(
        self,
    ) -> None:
        for reasoning in (
            "材料显示第一项和第三项成立，但错误答案为A、C。",
            "材料显示第一项和第三项成立，但排除的答案为A、C。",
            "材料显示第一项和第三项成立，中间结果为A、C。",
            "材料显示第一项和第三项成立，但不应该选择A、C。",
            "材料显示第一项和第三项成立，并非只有A、C项正确。",
        ):
            with self.subTest(reasoning=reasoning):
                with self.assertRaises(ValueError):
                    validate_answer_payload(
                        self.question(),
                        {
                            "answer_parts": ["AC"],
                            "reasoning": reasoning,
                        },
                    )

    def test_reasoning_without_marker_rejects_negative_prior_clause(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="计算目标指标并保留两位小数。",
        )
        for reasoning in (
            "材料核验后，按错误口径，计算可得12.34。",
            "材料核验后，该中间值不采用，结果为12.34。",
        ):
            with self.subTest(reasoning=reasoning):
                with self.assertRaises(ValueError):
                    validate_answer_payload(
                        question,
                        {
                            "answer_parts": ["12.34"],
                            "reasoning": reasoning,
                        },
                    )

    def test_reasoning_without_marker_does_not_treat_result_noun_as_answer_cue(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="计算目标指标并保留两位小数。",
        )
        for reasoning in (
            "材料中的计算结果显示同比保持增长，最终答案为12.34。",
            "逐项核验后，上述结果与材料披露一致，最终答案为12.34。",
            "核对主体、年份和指标后，所得结果支持前述判断，最终答案为12.34。",
        ):
            with self.subTest(reasoning=reasoning):
                parsed = validate_answer_payload(
                    question,
                    {
                        "answer_parts": ["12.34"],
                        "reasoning": reasoning,
                    },
                )
                self.assertEqual(parsed["reasoning"], reasoning)
                self.assertEqual(
                    parsed["decision_trace"]["postprocessing_mode"],
                    "reasoning_without_explicit_conclusion",
                )

    def test_numeric_reasoning_without_marker_rejects_conflicting_final_cue(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="计算目标指标并保留两位小数。",
        )
        for reasoning in (
            "根据材料中的数值代入公式并保留两位小数，计算可得56.78。",
            "根据材料中的数值代入公式并保留两位小数，最终应为56.78。",
            "材料披露12.34作为基数，复核后测算值为56.78。",
            "12.34只是中间值，实际应取56.78。",
            "原值为12.34，最后数值为56.78。",
            "材料列示12.34，综合核验后最终采用56.78。",
            "测算数值为112.34且过程完整，可以据此完成判断。",
            "材料列示12.34，中间计算结果为12.34。",
        ):
            with self.subTest(reasoning=reasoning):
                with self.assertRaises(ValueError):
                    validate_answer_payload(
                        question,
                        {
                            "answer_parts": ["12.34"],
                            "reasoning": reasoning,
                        },
                    )

    def test_reasoning_mismatched_conclusion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["AC"],
                    "reasoning": "材料显示第一项和第三项成立，第二项与原文不符。结论：BC",
                },
            )

    def test_multi_choice_conclusion_separators_are_equivalent_without_rewrite(
        self,
    ) -> None:
        for conclusion in ("A；C", "A、C", "A C", "A，C"):
            with self.subTest(conclusion=conclusion):
                reasoning = (
                    "材料显示第一项和第三项成立，第二项与原文不符。"
                    f"结论：{conclusion}"
                )
                parsed = validate_answer_payload(
                    self.question(),
                    {
                        "answer_parts": ["AC"],
                        "reasoning": reasoning,
                    },
                )

                self.assertEqual(parsed["answer_parts"], ["AC"])
                self.assertEqual(parsed["reasoning"], reasoning)
                self.assertEqual(
                    parsed["decision_trace"]["postprocessing_mode"],
                    "multi_choice_conclusion_separator_equivalence",
                )
                self.assertFalse(parsed["decision_trace"]["answer_modified"])
                self.assertFalse(parsed["decision_trace"]["reasoning_modified"])
                self.assertFalse(
                    parsed["decision_trace"]["semantic_correction"]
                )

    def test_multi_choice_conclusion_equivalence_rejects_unapproved_separators(
        self,
    ) -> None:
        for conclusion in ("A;C", "A,C", "A\nC", "A\tC"):
            with self.subTest(conclusion=conclusion):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    validate_answer_payload(
                        self.question(),
                        {
                            "answer_parts": ["AC"],
                            "reasoning": (
                                "材料显示第一项和第三项成立，第二项不成立。"
                                f"结论：{conclusion}"
                            ),
                        },
                    )

    def test_frozen_choice_conclusion_equivalence_requires_sorted_unique_labels(
        self,
    ) -> None:
        for answer, conclusion in (("CA", "C、A"), ("AA", "A、A")):
            with self.subTest(answer=answer):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    validate_frozen_answer_reasoning_payload(
                        [answer],
                        {
                            "reasoning": (
                                "材料逐项完成核验，并给出冻结标签。"
                                f"结论：{conclusion}"
                            )
                        },
                    )

    def test_multi_choice_conclusion_equivalence_rejects_changed_labels(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["AC"],
                    "reasoning": (
                        "材料显示第一项和第三项成立，第二项与原文不符。"
                        "结论：A；B"
                    ),
                },
            )

    def test_reasoning_conclusion_must_preserve_slot_boundaries_exactly(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=2,
            templates=("999999.99", "999999.99"),
            options={},
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_answer_payload(
                question,
                {
                    "answer_parts": ["12.00", "34.00"],
                    "reasoning": (
                        "材料给出两个输入并按顺序完成计算，得到两个最终数值。"
                        "结论：12.00，34.00"
                    ),
                },
            )

    def test_answer_shape_can_freeze_before_reasoning_only_retry(self) -> None:
        payload = {
            "answer_parts": ["AC"],
            "reasoning": "材料支持第一项与第三项，但末尾误写了其他字母。结论：BC",
        }

        frozen = validate_answer_shape_payload(self.question(), payload)

        self.assertEqual(frozen["answer_parts"], ["AC"])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_answer_payload(self.question(), payload)

    def test_frozen_answer_reasoning_prompt_and_validator_preserve_answer(self) -> None:
        question = self.question()
        messages = build_frozen_answer_reasoning_messages(
            question,
            self.evidence(),
            frozen_answer_parts=["AC"],
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        reasoning = (
            "材料说明甲产品包含责任免除条款，乙产品不满足适用范围，"
            "第三项同样得到原文支持。结论：AC"
        )

        parsed = validate_frozen_answer_reasoning_payload(
            ["AC"],
            {"reasoning": reasoning},
        )

        self.assertNotIn(question.qid, serialized)
        self.assertIn("可独立阅读", serialized)
        self.assertIn("不描述答案生成", serialized)
        self.assertIn("结论：AC", serialized)
        self.assertIn("不会由代码补写", serialized)
        self.assertNotIn("answer_parts", serialized)
        self.assertEqual(parsed["answer_parts"], ["AC"])
        self.assertEqual(parsed["reasoning"], reasoning)
        self.assertFalse(parsed["decision_trace"]["answer_modified"])

    def test_frozen_reasoning_schema_requires_final_reasoning_only(self) -> None:
        schema = build_frozen_answer_reasoning_schema(["AC"])

        self.assertEqual(set(schema["properties"]), {"reasoning"})
        self.assertEqual(schema["required"], ["reasoning"])

    def test_frozen_reasoning_rejects_separate_conclusion_field(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "only reasoning"):
            validate_frozen_answer_reasoning_payload(
                ["AC"],
                {
                    "reasoning": (
                        "材料明确说明甲和丙满足题设适用范围，乙的条件与原文不符。"
                    ),
                    "conclusion": "AC",
                },
            )

    def test_frozen_reasoning_still_requires_explicit_conclusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit 结论"):
            validate_frozen_answer_reasoning_payload(
                ["AC"],
                {
                    "reasoning": (
                        "材料显示第一项和第三项成立，第二项与原文不符，"
                        "因此最终答案为B。"
                    )
                },
            )

    def test_numeric_unit_is_deterministically_formatted_without_value_change(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=1,
            templates=("999999.99",),
            options={},
            question_text="两次公告之间间隔多少日？答案保留两位小数。",
        )

        parsed = validate_joint_payload_with_format_recovery(
            question,
            {
                "answer_parts": ["30日"],
                "reasoning": "规则要求此后每三十日公告一次。结论：30日",
            },
        )

        self.assertEqual(parsed["answer_parts"], ["30.00"])
        self.assertTrue(
            parsed["decision_trace"]["deterministic_format_normalization"]
        )
        self.assertTrue(parsed["reasoning"].endswith("结论：30.00"))

    def test_matching_answer_cue_is_preserved_from_same_qwen_response(self) -> None:
        reasoning = (
            "材料明确说明甲和丙满足题设范围，乙的条件与原文不符，"
            "因此最终答案为A、C。"
        )
        parsed = validate_joint_payload_with_format_recovery(
            self.question(),
            {
                "answer_parts": ["AC"],
                "reasoning": reasoning,
            },
        )

        self.assertEqual(parsed["answer_parts"], ["AC"])
        self.assertEqual(parsed["reasoning"], reasoning)
        self.assertEqual(
            parsed["decision_trace"]["postprocessing_mode"],
            "reasoning_without_explicit_conclusion",
        )
        self.assertFalse(parsed["decision_trace"]["answer_modified"])
        self.assertFalse(parsed["decision_trace"]["reasoning_modified"])
        self.assertFalse(parsed["decision_trace"]["semantic_correction"])

    def test_invalid_answer_field_recovers_from_same_response_conclusion(
        self,
    ) -> None:
        question = self.question(
            answer_format="calculation",
            question_type="计算题",
            slots=2,
            templates=("公司>公司", "999999.99"),
            options={},
            question_text="按指标排序并计算差额，差额保留两位小数。",
        )

        parsed = validate_joint_payload_with_format_recovery(
            question,
            {
                "answer_parts": [",", ",76.06"],
                "reasoning": (
                    "材料数值显示甲公司高于乙公司，差额为76.06。"
                    "结论：甲公司>乙公司；76.06"
                ),
            },
        )

        self.assertEqual(parsed["answer_parts"], ["甲公司>乙公司", "76.06"])
        self.assertEqual(
            parsed["decision_trace"]["answer_source"],
            "same_qwen_reasoning_conclusion",
        )
        self.assertFalse(parsed["decision_trace"]["semantic_correction"])

    def test_evidence_payload_has_stable_reversible_aliases_and_hashes(self) -> None:
        retrieval = {
            "final": {
                "hits": [
                    {
                        "unit_id": "doc::1",
                        "doc_id": "doc",
                        "metadata": {"unit_type": "paragraph"},
                        "title_path": ["标题"],
                        "text": "甲" * 100,
                        "merged_from": [
                            {
                                "unit_id": "doc::header",
                                "doc_id": "doc",
                                "sha256": "header-hash",
                            },
                            {
                                "unit_id": "doc::1",
                                "doc_id": "doc",
                                "sha256": "row-hash",
                            },
                        ],
                        "source_order": ["doc::header", "doc::1"],
                        "overlap_chars": 12,
                        "component_hashes": [
                            {"unit_id": "doc::header", "sha256": "header-hash"},
                            {"unit_id": "doc::1", "sha256": "row-hash"},
                        ],
                        "truncation_provenance": {
                            "applied": False,
                            "original_chars": 100,
                            "retained_chars": 100,
                        },
                    },
                    {
                        "unit_id": "doc::2",
                        "doc_id": "doc",
                        "metadata": {"unit_type": "paragraph"},
                        "title_path": ["标题"],
                        "text": "乙" * 100,
                    },
                ]
            }
        }

        evidence = prepare_evidence_payload(
            retrieval,
            max_hit_chars=60,
            max_total_chars=90,
        )

        self.assertEqual([item["evidence_key"] for item in evidence], ["E01", "E02"])
        self.assertEqual([item["source_key"] for item in evidence], ["S01", "S01"])
        self.assertEqual([item["evidence_id"] for item in evidence], ["doc::1", "doc::2"])
        self.assertEqual(sum(len(str(item["text"])) for item in evidence), 90)
        self.assertTrue(all(item["prompt_text_sha256"] for item in evidence))
        self.assertTrue(all(item["source_text_sha256"] for item in evidence))
        self.assertEqual(
            evidence[0]["source_order"],
            ["doc::header", "doc::1"],
        )
        self.assertEqual(evidence[0]["overlap_chars"], 12)
        self.assertEqual(len(evidence[0]["merged_from"]), 2)
        self.assertEqual(len(evidence[0]["component_hashes"]), 2)
        self.assertEqual(
            evidence[0]["compaction_truncation_provenance"]["applied"],
            False,
        )

    def test_generation_sources_do_not_load_accuracy_reference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                root / "src/afa_agent/b_board/retrieval_llm_baseline.py",
                root / "scripts/run_b_retrieval_llm_baseline.py",
            )
        )

        self.assertNotIn("pseudo99_from_official98", sources)
        self.assertNotIn("reference_answers.json", sources)
        self.assertNotIn("official_answer_locks.json", sources)


if __name__ == "__main__":
    unittest.main()
