from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from afa_agent.b_board import (
    SUBMISSION_COLUMNS,
    BAnswer,
    BQuestion,
    load_b_questions,
    read_b_question_file,
    validate_b_answer,
    validate_b_submission,
    write_b_submission,
)


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_B = ROOT / "upload_b"


def make_question(
    *,
    qid: str = "q1",
    answer_format: str = "mcq",
    slots: int = 1,
    slot_templates: tuple[str, ...] | None = None,
) -> BQuestion:
    raw_type = {
        "tf": "判断题",
        "mcq": "单选题",
        "multi": "多选题",
        "calculation": "计算题",
        "extraction": "抽取题",
    }[answer_format]
    options = {"A": "甲", "B": "乙", "C": "丙"} if answer_format in {"tf", "mcq", "multi"} else {}
    return BQuestion(
        qid=qid,
        domain="test",
        split="B",
        question="测试题",
        options=options,
        answer_format=answer_format,
        type=raw_type,
        answer_slots=slots,
        answer_slot_templates=slot_templates
        or tuple("A" if answer_format in {"tf", "mcq", "multi"} else "999999.99" for _ in range(slots)),
    )


class ActualBQuestionLoadingTests(unittest.TestCase):
    def test_explicit_question_precision_is_enforced_over_generic_slot(self) -> None:
        question = make_question(answer_format="calculation")
        question.question = "计算普通用户人数，保留一位小数。"

        validate_b_answer(question, BAnswer("q1", ("67.1",)))
        with self.assertRaisesRegex(ValueError, "requires exactly 1 decimal"):
            validate_b_answer(question, BAnswer("q1", ("67.10",)))

    def test_question_specific_no_percent_instruction_overrides_percent_template(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=2,
            slot_templates=("999999.99", "999999.99%"),
        )
        question.question = (
            "答案格式为‘权益乘数；近似资产收益率’，"
            "均保留两位小数，后者不带 %。"
        )

        validate_b_answer(question, BAnswer("q1", ("2.58", "7.65")))
        with self.assertRaisesRegex(ValueError, "without '%' suffix"):
            validate_b_answer(question, BAnswer("q1", ("2.58", "7.65%")))

    def test_scoped_positive_percent_instruction_only_applies_to_named_slot(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=2,
            slot_templates=("999999.99", "999999.99"),
        )
        question.question = "答案均保留两位小数，后者必须带%。"

        validate_b_answer(question, BAnswer("q1", ("2.58", "7.65%")))
        with self.assertRaisesRegex(ValueError, "answer_1"):
            validate_b_answer(question, BAnswer("q1", ("2.58%", "7.65%")))

    def test_answer_descriptors_apply_readme_percent_rule_per_slot(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=2,
            slot_templates=("999999.99", "999999.99"),
        )
        question.question = (
            "计算同比增幅和占比提高百分点。"
            "答案格式为“同比增幅；占比提高百分点”，均保留两位小数、不带单位。"
        )

        validate_b_answer(question, BAnswer("q1", ("40.05%", "10.10")))
        with self.assertRaisesRegex(ValueError, "answer_1"):
            validate_b_answer(question, BAnswer("q1", ("40.05", "10.10")))
        with self.assertRaisesRegex(ValueError, "answer_2"):
            validate_b_answer(question, BAnswer("q1", ("40.05%", "10.10%")))

    def test_percent_semantics_override_no_unit_but_not_explicit_no_percent(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=2,
            slot_templates=("999999.99", "999999.99"),
        )
        question.question = (
            "答案格式为“隐含营业收入；绝对相对偏差”，"
            "前者以百万元计，后者以百分数计，均不带单位。"
        )

        validate_b_answer(question, BAnswer("q1", ("1049321.98", "0.08%")))
        with self.assertRaisesRegex(ValueError, "answer_2"):
            validate_b_answer(question, BAnswer("q1", ("1049321.98", "0.08")))

    def test_single_slot_rate_uses_readme_percent_suffix(self) -> None:
        question = make_question(answer_format="calculation")
        question.question = "全年动力电池需求同比增速最接近多少？"

        validate_b_answer(question, BAnswer("q1", ("22.27%",)))
        with self.assertRaisesRegex(ValueError, "answer_1"):
            validate_b_answer(question, BAnswer("q1", ("22.27",)))

    def test_single_slot_percent_input_does_not_make_amount_answer_percent(self) -> None:
        question = make_question(answer_format="calculation")
        question.question = "已知投资收益率为10%。求对应收益金额，保留两位小数。"

        validate_b_answer(question, BAnswer("q1", ("100.00",)))

    def test_generic_explicit_no_percent_overrides_percent_template(self) -> None:
        question = make_question(
            answer_format="calculation",
            slot_templates=("999999.99%",),
        )
        question.question = "计算投资收益率，答案不带百分号，保留两位小数。"

        validate_b_answer(question, BAnswer("q1", ("7.65",)))
        with self.assertRaisesRegex(ValueError, "without '%' suffix"):
            validate_b_answer(question, BAnswer("q1", ("7.65%",)))

    def test_readme_percent_semantics_cover_ratio_and_margin_terms(self) -> None:
        for label in ("利润占比", "销售比例", "毛利率"):
            with self.subTest(label=label):
                question = make_question(answer_format="calculation")
                question.question = f"计算{label}是多少？"
                validate_b_answer(question, BAnswer("q1", ("12.34%",)))
                with self.assertRaisesRegex(ValueError, "'%' suffix"):
                    validate_b_answer(question, BAnswer("q1", ("12.34",)))

    def test_percentage_point_declines_remain_bare_by_answer_slot(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=3,
            slot_templates=("999999.99", "999999.99", "999999.99"),
        )
        question.question = (
            "分别计算净利率下降的百分点、现金流率下降的百分点，并计算两者差额。"
            "答案格式为“净利率降幅；现金流率降幅；两者差额”，均不带单位。"
        )

        validate_b_answer(question, BAnswer("q1", ("1.20", "2.30", "1.10")))

    def test_single_slot_percent_inputs_do_not_override_final_amount_question(self) -> None:
        question = make_question(answer_format="calculation")
        question.question = (
            "2025年订单增速超过100%，若2026年增速为30%，AI订单占比为80%，"
            "则2026年AI新签订单约为多少亿元？"
        )

        validate_b_answer(question, BAnswer("q1", ("18.72",)))

    def test_no_unit_instruction_does_not_turn_ordering_slot_into_numeric_slot(self) -> None:
        question = make_question(
            answer_format="calculation",
            slots=2,
            slot_templates=("公司>公司", "999999.99"),
        )
        question.question = "答案格式为公司>公司；差值，均不带单位，均保留两位小数。"

        validate_b_answer(question, BAnswer("q1", ("甲公司>乙公司", "1.20")))

    def test_loads_real_json_jsonl_and_bom_in_official_order(self) -> None:
        questions = load_b_questions(UPLOAD_B)

        self.assertEqual(len(questions), 100)
        self.assertEqual(questions[0].qid, "fc_b_001")
        self.assertEqual(questions[-1].qid, "res_b_020")
        self.assertEqual(Counter(item.domain for item in questions), {
            "financial_contracts": 20,
            "financial_reports": 20,
            "insurance": 20,
            "regulatory": 20,
            "research": 20,
        })
        self.assertEqual(Counter(item.answer_format for item in questions), {
            "multi": 66,
            "mcq": 7,
            "tf": 1,
            "calculation": 26,
        })
        self.assertEqual(Counter(item.answer_slots for item in questions), {1: 90, 2: 7, 3: 2, 4: 1})
        self.assertTrue(all(item.split == "B" and item.doc_ids == [] for item in questions))

    def test_reads_bom_json_array_and_jsonl_and_maps_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_dir = root / "question_b"
            question_dir.mkdir()
            json_path = question_dir / "one.json"
            json_path.write_text(
                "\ufeff" + json.dumps([{
                    "qid": "q1", "domain": "test", "split": "B", "question": "q",
                    "type": "抽取题", "options": {},
                }], ensure_ascii=False),
                encoding="utf-8",
            )
            jsonl_path = question_dir / "two.jsonl"
            jsonl_path.write_text(
                "\ufeff" + json.dumps({
                    "qid": "q2", "domain": "test", "split": "B", "question": "q",
                    "type": "单选题", "options": {"A": "是", "B": "否"},
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with (root / "submit.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(SUBMISSION_COLUMNS)
                writer.writerow(["summary", "", "", "", "", 0, 0, 0])
                writer.writerow(["q1", "文本", "", "", "", 0, 0, 0])
                writer.writerow(["q2", "A", "", "", "", 0, 0, 0])

            self.assertEqual(read_b_question_file(json_path)[0]["type"], "抽取题")
            self.assertEqual(read_b_question_file(jsonl_path)[0]["qid"], "q2")
            questions = load_b_questions(root)
            self.assertEqual([question.answer_format for question in questions], ["extraction", "mcq"])

    def test_template_qid_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_dir = root / "question_b"
            question_dir.mkdir()
            (question_dir / "q.json").write_text(json.dumps([{
                "qid": "q1", "domain": "test", "split": "B", "question": "q",
                "type": "单选题", "options": {"A": "是", "B": "否"},
            }]), encoding="utf-8")
            with (root / "submit.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(SUBMISSION_COLUMNS)
                writer.writerow(["summary", "", "", "", "", 0, 0, 0])
                writer.writerow(["q2", "A", "", "", "", 0, 0, 0])

            with self.assertRaisesRegex(ValueError, "qid mismatch"):
                load_b_questions(root)


class BAnswerContractTests(unittest.TestCase):
    def test_token_total_is_derived_and_explicit_mismatch_is_rejected(self) -> None:
        answer = BAnswer("q1", ["A"], prompt_tokens=7, completion_tokens=3)
        self.assertEqual(answer.answer_parts, ("A",))
        self.assertEqual(answer.total_tokens, 10)

        with self.assertRaisesRegex(ValueError, "does not equal"):
            BAnswer("q1", ("A",), 7, 3, 11)

    def test_choice_and_freeform_format_validation(self) -> None:
        validate_b_answer(make_question(answer_format="multi"), BAnswer("q1", ("AC",)))
        for invalid in ("A", "CA", "AA", "A,C", "aC"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_b_answer(make_question(answer_format="multi"), BAnswer("q1", (invalid,)))

        calculation = make_question(answer_format="calculation")
        for valid in ("12.34", "2026年4月1日"):
            with self.subTest(valid=valid):
                validate_b_answer(calculation, BAnswer("q1", (valid,)))
        for invalid in ("12.3", "12.345%", "2026年2月30日", "完整文本"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_b_answer(calculation, BAnswer("q1", (invalid,)))

        percent = make_question(
            answer_format="calculation", slot_templates=("999999.99%",)
        )
        validate_b_answer(percent, BAnswer("q1", ("12.34%",)))
        with self.assertRaises(ValueError):
            validate_b_answer(percent, BAnswer("q1", ("无法计算",)))

        ordering = make_question(
            answer_format="calculation", slot_templates=("公司名称>公司名称",)
        )
        validate_b_answer(ordering, BAnswer("q1", ("甲公司>乙公司",)))
        with self.assertRaises(ValueError):
            validate_b_answer(ordering, BAnswer("q1", ("甲 >乙",)))


class BSubmissionWriterTests(unittest.TestCase):
    def test_writes_official_columns_summary_first_tokens_and_empty_unused_slots(self) -> None:
        questions = [
            make_question(qid="q1", answer_format="mcq"),
            make_question(
                qid="q2",
                answer_format="calculation",
                slots=2,
                slot_templates=("999999.99", "999999.99%"),
            ),
        ]
        answers = [
            BAnswer("q1", ("B",), 10, 2, 12),
            BAnswer("q2", ("12.34", "56.78%"), 20, 3, 23),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "submit.csv"
            write_b_submission(destination, questions, answers)
            parsed = validate_b_submission(destination, questions)
            with destination.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(tuple(rows[0]), SUBMISSION_COLUMNS)
        self.assertEqual(rows[0], {
            "qid": "summary", "answer_1": "", "answer_2": "", "answer_3": "", "answer_4": "",
            "prompt_tokens": "30", "completion_tokens": "5", "total_tokens": "35",
        })
        self.assertEqual(rows[1]["answer_1"], "B")
        self.assertEqual(rows[1]["answer_2"], "")
        self.assertEqual(rows[2]["answer_1"], "12.34")
        self.assertEqual(rows[2]["answer_2"], "56.78%")
        self.assertEqual(rows[2]["answer_3"], "")
        self.assertEqual([answer.qid for answer in parsed], ["q1", "q2"])

    def test_rejects_missing_answers_and_wrong_slot_count(self) -> None:
        question = make_question(answer_format="calculation", slots=2)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "submit.csv"
            with self.assertRaisesRegex(ValueError, "expected 2 answer parts"):
                write_b_submission(destination, [question], [BAnswer("q1", ("12.34",))])
            with self.assertRaisesRegex(ValueError, "missing"):
                write_b_submission(destination, [question], [])

    def test_detects_tampered_summary_and_unused_slots(self) -> None:
        question = make_question(answer_format="mcq")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "submit.csv"
            write_b_submission(destination, [question], [BAnswer("q1", ("A",), 1, 2, 3)])
            text = destination.read_text(encoding="utf-8-sig")
            destination.write_text(text.replace("summary,,,,,1,2,3", "summary,,,,,1,2,4"), encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "total_tokens"):
                validate_b_submission(destination, [question])


if __name__ == "__main__":
    unittest.main()
