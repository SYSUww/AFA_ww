from __future__ import annotations

import json
from pathlib import Path
import unittest

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.retrieval_llm_baseline import (
    build_answer_messages,
    build_answer_schema,
    build_retrieval_bundle,
    prepare_evidence_payload,
    validate_answer_payload,
)


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
    ) -> BQuestion:
        return BQuestion(
            qid=qid,
            domain="insurance",
            split="B",
            question="核验两个产品是否包含责任免除条款。",
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
        self.assertEqual(answer_parts["items"], {"type": "string", "minLength": 1})

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

    def test_reasoning_missing_conclusion_is_rejected_without_rewrite(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit 结论"):
            validate_answer_payload(
                self.question(),
                {
                    "answer_parts": ["AC"],
                    "reasoning": "材料显示第一项和第三项成立，第二项与原文不符。",
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
