from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.merge import assemble_answer_run
from afa_agent.io_utils import read_json, write_json


def question(qid: str, template: str = "999999.99") -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="test",
        split="B",
        question="q",
        options={},
        answer_format="calculation",
        type="计算题",
        answer_slots=1,
        answer_slot_templates=(template,),
    )


def artifact(qid: str, answer: str) -> dict:
    return {
        "qid": qid,
        "domain": "test",
        "answer_format": "calculation",
        "answer_slot_count": 1,
        "answer_parts": [answer],
        "used_evidence_ids": [f"{qid}:e"],
        "evidence_items": [{"unit_id": f"{qid}:e", "text": answer}],
        "decision_summary": "",
        "decision_trace": {},
        "calculation_trace": {"replay_verified": True},
        "token_usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "locator": {},
    }


class BBoardMergeTests(unittest.TestCase):
    def test_later_source_repairs_missing_qid_and_writes_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            repair = root / "repair"
            base.mkdir()
            repair.mkdir()
            write_json(base / "answers.json", [artifact("q1", "1.00")])
            write_json(repair / "answers.json", [artifact("q2", "2.00")])

            manifest = assemble_answer_run(
                questions=[question("q1"), question("q2")],
                source_run_dirs=[base, repair],
                output_dir=root / "merged",
            )

            self.assertTrue(manifest["submission_valid"])
            self.assertTrue((root / "merged" / "submit.csv").exists())
            self.assertEqual(len(read_json(root / "merged" / "answers.json")), 2)

    def test_invalid_baseline_is_evaluation_complete_but_not_submission_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_json(source / "answers.json", [artifact("q1", "无法计算")])

            manifest = assemble_answer_run(
                questions=[question("q1")],
                source_run_dirs=[source],
                output_dir=root / "merged",
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["submission_valid"])
            self.assertEqual(manifest["submission_validation_failures"][0]["qid"], "q1")
            self.assertFalse((root / "merged" / "submit.csv").exists())


if __name__ == "__main__":
    unittest.main()
