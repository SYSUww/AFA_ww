from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Mapping

from afa_agent.b_board.io import BAnswer, BQuestion, write_b_submission
from afa_agent.b_board.reasoning_evaluation import (
    FixedReasoningEvaluator,
    ReasoningEvaluationFingerprintError,
    run_reasoning_evaluation,
)
from afa_agent.client import LLMResponse
from afa_agent.config import ModelConfig
from afa_agent.models import TokenUsage


def question(qid: str) -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="test",
        split="B",
        question=f"问题 {qid}",
        options={"A": "是", "B": "否"},
        answer_format="mcq",
        type="单选题",
        answer_slots=1,
        answer_slot_templates=("A",),
    )


class FakeReasoningController:
    def __init__(self, *, fail_text: str = "") -> None:
        self.calls: list[str] = []
        self.fail_text = fail_text
        self.lock = threading.Lock()

    def factory(self):
        controller = self

        class Evaluator:
            def evaluate(self, reasoning: str):
                with controller.lock:
                    controller.calls.append(reasoning)
                if reasoning == controller.fail_text:
                    raise RuntimeError(
                        "request to https://secret.example/v1 failed with Bearer sk-secret"
                    )
                return (
                    {"logical": 90, "completeness": 80, "clarity": 70},
                    {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                )

        return Evaluator()


class CapturingClient:
    def __init__(self) -> None:
        self.messages = []

    def chat_json(self, messages):
        self.messages = messages
        return LLMResponse(
            content='{"logical": 81.5, "completeness": 72, "clarity": 90}',
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            raw_payload={},
        )


class ReasoningEvaluationRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.questions = [question("q1"), question("q2")]
        self.submission = self.root / "research_submit.csv"
        self.long_reasoning = "定位到对应条款后提取关键事实，逐项比较适用条件与限制范围，因此结论具有完整依据。"
        self.short_reasoning = "理由太短"
        write_b_submission(
            self.submission,
            self.questions,
            [
                BAnswer("q1", ("A",), 10, 2, 12, self.long_reasoning),
                BAnswer("q2", ("B",), 20, 3, 23, self.short_reasoning),
            ],
        )
        self.model = ModelConfig(
            api_key="sk-secret",
            api_base="https://secret.example/v1",
            model_name="gpt-5.6",
            temperature=0.0,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, controller: FakeReasoningController, *, output_name: str = "evaluation"):
        return run_reasoning_evaluation(
            submission_path=self.submission,
            questions=self.questions,
            model_config=self.model,
            output_dir=self.root / output_name,
            workers=2,
            evaluator_factory=controller.factory,
            accuracy_score=97,
            accuracy_source="official_submission_007",
        )

    def test_short_reasoning_is_zero_without_model_call_and_scorecard_uses_submission_tokens(self) -> None:
        controller = FakeReasoningController()

        result = self.execute(controller)

        self.assertEqual(controller.calls, [self.long_reasoning])
        self.assertEqual(result.evaluations["q1"].reasoning_score, 80)
        self.assertEqual(result.evaluations["q2"].reasoning_score, 0)
        self.assertEqual(result.aggregate["reasoning_score"], 40)
        self.assertEqual(result.aggregate["status_counts"], {"below_minimum_length": 1, "scored": 1})
        self.assertEqual(result.manifest["submission_token_total"], 35)
        self.assertEqual(result.manifest["judge_token_usage"]["total_tokens"], 3)
        assert result.scorecard is not None
        self.assertEqual(result.scorecard["token_total"], 35)
        self.assertFalse(result.scorecard["judge_tokens_included_in_submission"])
        self.assertAlmostEqual(result.scorecard["total_score"], 48.5 + 12 + 0.0014)

    def test_judge_failure_is_persisted_as_zero_and_is_not_retried_on_resume(self) -> None:
        second_reasoning = "先定位比较对象，再核对材料中的数值和适用范围，最后依据完整推导得到结论。"
        write_b_submission(
            self.submission,
            self.questions,
            [
                BAnswer("q1", ("A",), 10, 2, 12, self.long_reasoning),
                BAnswer("q2", ("B",), 20, 3, 23, second_reasoning),
            ],
        )
        first = FakeReasoningController(fail_text=second_reasoning)

        first_result = self.execute(first)

        self.assertEqual(first_result.manifest["status"], "complete")
        self.assertEqual(first_result.manifest["failure_count"], 1)
        self.assertEqual(first_result.evaluations["q2"].status, "judge_error")
        failures = (self.root / "evaluation" / "reasoning_judge_failures.jsonl").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("sk-secret", failures)
        self.assertNotIn("secret.example", failures)

        resumed = FakeReasoningController()
        second_result = self.execute(resumed)

        self.assertEqual(resumed.calls, [])
        self.assertEqual(second_result.evaluations["q2"].reasoning_score, 0)

    def test_changed_submission_or_evaluator_identity_refuses_unsafe_resume(self) -> None:
        self.execute(FakeReasoningController())
        write_b_submission(
            self.submission,
            self.questions,
            [
                BAnswer("q1", ("A",), 10, 2, 12, self.long_reasoning + "新增内容"),
                BAnswer("q2", ("B",), 20, 3, 23, self.short_reasoning),
            ],
        )

        with self.assertRaisesRegex(ReasoningEvaluationFingerprintError, "mismatch"):
            self.execute(FakeReasoningController())

        changed_model = ModelConfig(
            api_key="x",
            api_base="https://other.example/v1",
            model_name="gpt-5.6",
            temperature=0.0,
        )
        with self.assertRaises(ReasoningEvaluationFingerprintError):
            run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=changed_model,
                output_dir=self.root / "evaluation",
                evaluator_factory=FakeReasoningController().factory,
                accuracy_score=97,
                accuracy_source="official_submission_007",
            )

    def test_model_and_temperature_are_fixed(self) -> None:
        wrong_model = ModelConfig(api_key="x", api_base="x", model_name="gpt-5.5")
        with self.assertRaisesRegex(ValueError, "gpt-5.6"):
            run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=wrong_model,
                output_dir=self.root / "wrong-model",
            )
        wrong_temperature = ModelConfig(
            api_key="x", api_base="x", model_name="gpt-5.6", temperature=0.1
        )
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=wrong_temperature,
                output_dir=self.root / "wrong-temperature",
            )


class FixedReasoningEvaluatorTests(unittest.TestCase):
    def test_model_message_contains_only_reasoning_and_accepts_float_scores(self) -> None:
        client = CapturingClient()
        evaluator = FixedReasoningEvaluator(client)  # type: ignore[arg-type]
        reasoning = "先定位数据，再完成计算，最后根据结果得出结论。"

        dimensions, usage = evaluator.evaluate(reasoning)

        self.assertEqual(client.messages[1], {"role": "user", "content": reasoning})
        joined = "\n".join(message["content"] for message in client.messages)
        for forbidden in ("qid", "answer", "evidence", "问题 q1"):
            self.assertNotIn(forbidden, joined)
        self.assertEqual(dimensions["logical"], 81.5)
        self.assertEqual(usage["total_tokens"], 12)

    def test_invalid_dimension_is_rejected_with_usage_attached(self) -> None:
        class InvalidClient(CapturingClient):
            def chat_json(self, messages):
                response = super().chat_json(messages)
                response.content = '{"logical": true, "completeness": 80, "clarity": 80}'
                return response

        with self.assertRaisesRegex(Exception, "invalid reasoning judge response") as raised:
            FixedReasoningEvaluator(InvalidClient()).evaluate("这是足够长的推理过程文本，用于验证非法评分字段会被拒绝。")  # type: ignore[arg-type]
        self.assertEqual(getattr(raised.exception, "token_usage")["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
