from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.io import BAnswer, BQuestion, validate_b_submission, write_b_submission
from afa_agent.b_board.reasoning_rewrite import run_reasoning_rewrite
from afa_agent.config import ModelConfig
from afa_agent.io_utils import write_json


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


class FakeRewriter:
    def __init__(self, reasoning: str, usage: dict[str, int]) -> None:
        self.reasoning = reasoning
        self.usage = usage

    def rewrite(self, payload):
        self.payload = payload
        return self.reasoning, self.usage


class ReasoningRewriteTests(unittest.TestCase):
    def test_rewrites_only_targets_preserves_answers_and_adds_raw_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [question("q1"), question("q2")]
            source = root / "source.csv"
            write_b_submission(
                source,
                questions,
                [
                    BAnswer("q1", ("AB",), 10, 2, reasoning="原摘要一已有足够长度用于研究基线。"),
                    BAnswer("q2", ("AB",), 20, 3, reasoning="原摘要二已有足够长度用于研究基线。"),
                ],
            )
            answers = root / "answers.json"
            write_json(
                answers,
                [
                    {"qid": "q1", "evidence_items": []},
                    {"qid": "q2", "evidence_items": []},
                ],
            )
            rewritten = "定位主体和条款；提取两项关键事实；逐项比较后两项均成立；因此冻结答案AB保持不变。"
            result = run_reasoning_rewrite(
                source_submission_path=source,
                source_answers_path=answers,
                questions=questions,
                target_qids=["q1"],
                model_config=ModelConfig("secret", "https://example.test/v1", "gpt-5.5"),
                output_dir=root / "output",
                workers=1,
                rewriter_factory=lambda: FakeRewriter(
                    rewritten,
                    {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
                ),
            )

            parsed = validate_b_submission(result.submission_path, questions)
            self.assertEqual(parsed[0].answer_parts, ("AB",))
            self.assertEqual(parsed[0].reasoning, rewritten)
            self.assertEqual(parsed[0].total_tokens, 50)
            self.assertEqual(parsed[1].reasoning, "原摘要二已有足够长度用于研究基线。")
            self.assertEqual(parsed[1].total_tokens, 23)
            self.assertEqual(result.manifest["answer_changes"], 0)
            self.assertFalse(result.manifest["submission_eligible"])

    def test_parse_failure_falls_back_but_accounts_for_the_call(self) -> None:
        class FailingRewriter:
            def rewrite(self, _payload):
                from afa_agent.b_board.reasoning_rewrite import ReasoningRewriteError

                raise ReasoningRewriteError(
                    "bad json",
                    token_usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [question("q1")]
            source = root / "source.csv"
            original = "原摘要已有足够长度并说明先定位事实再推导结论。"
            write_b_submission(source, questions, [BAnswer("q1", ("AB",), 10, 2, reasoning=original)])
            answers = root / "answers.json"
            write_json(answers, [{"qid": "q1", "evidence_items": []}])

            result = run_reasoning_rewrite(
                source_submission_path=source,
                source_answers_path=answers,
                questions=questions,
                target_qids=["q1"],
                model_config=ModelConfig("secret", "https://example.test/v1", "gpt-5.5"),
                output_dir=root / "output",
                workers=1,
                rewriter_factory=FailingRewriter,
            )

            parsed = validate_b_submission(result.submission_path, questions)
            self.assertEqual(parsed[0].reasoning, original)
            self.assertEqual(parsed[0].total_tokens, 21)
            self.assertEqual(result.manifest["rewrite_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
