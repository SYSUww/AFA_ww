from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.io import BQuestion, validate_b_submission
from afa_agent.b_board.reasoning_candidate import materialize_existing_reasoning_candidate


def question(qid: str) -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="test",
        split="B",
        question="which statements are correct?",
        options={"A": "one", "B": "two"},
        answer_format="multi",
        type="多选题",
        answer_slots=1,
        answer_slot_templates=("AB",),
    )


class ExistingReasoningCandidateTests(unittest.TestCase):
    def test_materializes_answers_and_tokens_without_claiming_submission_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_csv = root / "source.csv"
            with source_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "qid",
                        "answer_1",
                        "answer_2",
                        "answer_3",
                        "answer_4",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "qid": "summary",
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    }
                )
                writer.writerow(
                    {
                        "qid": "q1",
                        "answer_1": "AB",
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    }
                )
            answers_path = root / "answers.json"
            answers_path.write_text(
                json.dumps(
                    [
                        {
                            "qid": "q1",
                            "answer_parts": ["AB"],
                            "decision_summary": "先定位两个选项，再逐项核对事实，二者均得到支持，因此选择AB。",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            manifest = materialize_existing_reasoning_candidate(
                source_submission_path=source_csv,
                source_answers_path=answers_path,
                questions=[question("q1")],
                output_dir=root / "candidate",
            )

            self.assertFalse(manifest["submission_eligible"])
            self.assertEqual(manifest["answer_changes"], 0)
            self.assertEqual(manifest["token_usage"]["total_tokens"], 13)
            parsed = validate_b_submission(
                root / "candidate" / "research_submit.csv",
                [question("q1")],
            )
            self.assertEqual(parsed[0].answer_parts, ("AB",))
            self.assertIn("逐项核对", parsed[0].reasoning)


if __name__ == "__main__":
    unittest.main()
