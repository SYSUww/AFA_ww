from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Mapping

from afa_agent.b_board.evaluation_run import (
    EvaluationRunFingerprintError,
    run_fixed_evaluation,
)
from afa_agent.b_board.evaluator import (
    CALCULATION_DIMENSION,
    CHOICE_DIMENSION,
    COMMON_DIMENSIONS,
    ConfidenceEvaluation,
    confidence_tier,
)
from afa_agent.b_board.io import BQuestion
from afa_agent.config import ModelConfig
from afa_agent.io_utils import read_json, write_json


def make_question(qid: str, *, calculation: bool = False) -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="regulatory" if not calculation else "financial_reports",
        split="B",
        question=f"问题 {qid}",
        options={} if calculation else {"A": "是", "B": "否"},
        answer_format="calculation" if calculation else "mcq",
        type="计算题" if calculation else "单选题",
        answer_slots=1,
        answer_slot_templates=("999999.99" if calculation else "A",),
    )


def make_artifact(question: BQuestion, answer: str) -> dict[str, Any]:
    evidence_id = f"{question.qid}:e1"
    return {
        "qid": question.qid,
        "answer_parts": [answer],
        "used_evidence_ids": [evidence_id],
        "evidence_items": [{"unit_id": evidence_id, "text": "支持证据"}],
        "decision_trace": {},
        "calculation_trace": {
            "replay_verified": True,
            "grounding_verified": True,
        } if question.answer_format == "calculation" else {},
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def make_evaluation(qid: str, score: int, verdict: str) -> ConfidenceEvaluation:
    dimensions: dict[str, int | None] = {key: score for key in COMMON_DIMENSIONS}
    dimensions[CHOICE_DIMENSION] = score
    dimensions[CALCULATION_DIMENSION] = score
    return ConfidenceEvaluation(
        qid=qid,
        dimensions=dimensions,
        confidence_score=score,
        tier=confidence_tier(score),
        verdict=verdict,
        blocking_reasons=(),
        low_confidence_reasons=(),
        suggested_improvements=(),
        hard_failures=(),
    )


class FakeEvaluationController:
    def __init__(self, failures: Mapping[str, int] | None = None) -> None:
        self.calls: list[str] = []
        self.failures = dict(failures or {})
        self.lock = threading.Lock()

    def factory(self):
        controller = self

        class Evaluator:
            def evaluate(self, subject):
                qid = str(subject["qid"])
                with controller.lock:
                    controller.calls.append(qid)
                    remaining = controller.failures.get(qid, 0)
                    if remaining:
                        controller.failures[qid] = remaining - 1
                        raise RuntimeError(
                            "request to http://secret.example/v1 failed with Bearer sk-secret"
                        )
                if qid.startswith(("wrong_", "irrelevant_", "missing_", "format_")):
                    result = make_evaluation(qid, 20, "unsupported")
                else:
                    result = make_evaluation(qid, 85, "supported")
                return result, {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}

        return Evaluator()


class FixedEvaluationRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.questions = [make_question("q1"), make_question("q2", calculation=True)]
        write_json(
            self.run_dir / "answers.json",
            [make_artifact(self.questions[0], "A"), make_artifact(self.questions[1], "30.00")],
        )
        self.model = ModelConfig(
            api_key="sk-secret",
            api_base="http://secret.example/v1",
            model_name="fixed-test-model",
            temperature=0.0,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def execute(self, controller: FakeEvaluationController, *, model: ModelConfig | None = None):
        return run_fixed_evaluation(
            run_dir=self.run_dir,
            questions=self.questions,
            model_config=model or self.model,
            workers=3,
            evaluator_factory=controller.factory,
        )

    def test_complete_run_seals_answers_uses_six_sentinels_and_returns_aggregate(self) -> None:
        controller = FakeEvaluationController()

        result = self.execute(controller)

        self.assertEqual(result.manifest["status"], "complete")
        self.assertEqual(result.manifest["expected_answer_count"], 2)
        self.assertEqual(result.manifest["expected_sentinel_count"], 6)
        self.assertEqual(len(controller.calls), 8)
        self.assertEqual(result.aggregate["question_count"], 2)
        self.assertEqual(result.aggregate["tiers"], {"high": 2})
        self.assertEqual(set(result.evaluations), {"q1", "q2"})
        self.assertTrue((self.run_dir / "evaluation" / "sealed_answers.json").exists())

        manifest_text = json.dumps(result.manifest, ensure_ascii=False)
        self.assertNotIn("sk-secret", manifest_text)
        self.assertNotIn("secret.example", manifest_text)
        identity = result.manifest["evaluator_identity"]
        self.assertIn("api_base_sha256", identity)
        self.assertNotIn("api_base", identity)
        self.assertTrue(result.manifest["sentinel_validation"]["passed"])

    def test_resume_only_retries_missing_evaluation_and_redacts_failure(self) -> None:
        first = FakeEvaluationController({"q2": 1})
        first_result = self.execute(first)
        self.assertEqual(first_result.manifest["status"], "invalid")
        self.assertEqual(first_result.manifest["failure_count"], 1)
        failures_text = (self.run_dir / "evaluation" / "failures.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("sk-secret", failures_text)
        self.assertNotIn("secret.example", failures_text)

        resumed = FakeEvaluationController()
        result = self.execute(resumed)

        self.assertEqual(resumed.calls, ["q2"])
        self.assertEqual(result.manifest["status"], "complete")
        self.assertTrue(result.manifest["resumed"])
        self.assertEqual(result.manifest["resumed_evaluation_count"], 7)

    def test_completed_run_is_idempotent_without_model_calls(self) -> None:
        self.execute(FakeEvaluationController())
        no_calls = FakeEvaluationController()

        result = self.execute(no_calls)

        self.assertEqual(no_calls.calls, [])
        self.assertEqual(result.manifest["status"], "complete")

    def test_changed_answers_or_url_identity_refuses_unsafe_resume(self) -> None:
        self.execute(FakeEvaluationController())
        answers = read_json(self.run_dir / "answers.json")
        answers[0]["answer_parts"] = ["B"]
        write_json(self.run_dir / "answers.json", answers)

        with self.assertRaisesRegex(EvaluationRunFingerprintError, "fingerprint mismatch"):
            self.execute(FakeEvaluationController())

        write_json(
            self.run_dir / "answers.json",
            [make_artifact(self.questions[0], "A"), make_artifact(self.questions[1], "30.00")],
        )
        changed_model = ModelConfig(
            api_key="different-key",
            api_base="https://other.example/v1",
            model_name=self.model.model_name,
            temperature=0.0,
        )
        with self.assertRaisesRegex(EvaluationRunFingerprintError, "identity changed"):
            self.execute(FakeEvaluationController(), model=changed_model)

    def test_nonzero_temperature_is_rejected_before_evaluation(self) -> None:
        model = ModelConfig(
            api_key="x",
            api_base="https://example.test/v1",
            model_name="judge",
            temperature=0.1,
        )
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            self.execute(FakeEvaluationController(), model=model)


if __name__ == "__main__":
    unittest.main()
