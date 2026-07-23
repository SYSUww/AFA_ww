from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.io import BAnswer, BQuestion
from scripts.build_b_reasoning_composite import (
    APP_COMPAT_SUBMISSION_COLUMNS,
    validate_app_compatible_submission,
    write_app_compatible_submission,
)


def make_question(qid: str, slots: int) -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="test",
        split="B",
        question="测试题",
        options={},
        answer_format="calculation",
        type="计算题",
        answer_slots=slots,
        answer_slot_templates=tuple("999999.99" for _ in range(slots)),
    )


class AppCompatibleCompositeExportTests(unittest.TestCase):
    def test_renames_answer_columns_without_changing_payload(self) -> None:
        questions = [make_question("q1", 1), make_question("q2", 2)]
        answers = [
            BAnswer("q1", ("12.34",), 10, 2, 12, "第一题的完整推理过程。"),
            BAnswer("q2", ("56.78", "90.12"), 20, 3, 23, "第二题的完整推理过程。"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "research_submit_compat.csv"
            write_app_compatible_submission(destination, questions, answers)
            validate_app_compatible_submission(destination, questions, answers)
            with destination.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = tuple(reader.fieldnames or ())
                rows = list(reader)

        self.assertEqual(columns, APP_COMPAT_SUBMISSION_COLUMNS)
        self.assertEqual(
            rows[0],
            {
                "qid": "summary",
                "answer_1": "",
                "answer_2": "",
                "answer_3": "",
                "answer_4": "",
                "prompt_tokens": "30",
                "completion_tokens": "5",
                "total_tokens": "35",
                "reasoning": "",
            },
        )
        self.assertEqual(rows[1]["answer_1"], "12.34")
        self.assertEqual(rows[1]["answer_2"], "")
        self.assertEqual(rows[1]["reasoning"], answers[0].reasoning)
        self.assertEqual(rows[2]["answer_1"], "56.78")
        self.assertEqual(rows[2]["answer_2"], "90.12")
        self.assertEqual(rows[2]["total_tokens"], "23")

    def test_validation_rejects_payload_drift(self) -> None:
        questions = [make_question("q1", 1)]
        answers = [BAnswer("q1", ("12.34",), 10, 2, 12, "完整推理过程。")]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "research_submit_compat.csv"
            write_app_compatible_submission(destination, questions, answers)
            text = destination.read_text(encoding="utf-8-sig")
            destination.write_text(text.replace("12.34", "99.99"), encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "answer fields drifted"):
                validate_app_compatible_submission(destination, questions, answers)


if __name__ == "__main__":
    unittest.main()
