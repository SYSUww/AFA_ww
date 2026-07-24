from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

import requests

from afa_agent.b_board.io import BAnswer, BQuestion, write_b_submission
from afa_agent.b_board import reasoning_evaluation as reasoning_module
from afa_agent.b_board.reasoning_evaluation import (
    AUDITOR_SYSTEM_PROMPT,
    SCORER_SYSTEM_PROMPT,
    FixedReasoningEvaluator,
    ReasoningEvaluation,
    ReasoningJudgeOutcome,
    ReasoningEvaluationFingerprintError,
    apply_reasoning_hard_caps,
    audit_reasoning_corpus,
    build_reasoning_sentinels,
    reasoning_schema_fingerprint,
    run_reasoning_evaluation,
    validate_reasoning_sentinels,
)
from afa_agent.client import LLMResponse
from afa_agent.config import ModelConfig
from afa_agent.models import TokenUsage
from scripts.evaluate_b_board_reasoning import _select_questions, parse_args


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


class StagedReasoningController:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def factory(self):
        controller = self

        class Evaluator:
            def evaluate(self, reasoning: str):
                controller.calls.append(reasoning)
                return ReasoningJudgeOutcome(
                    rubric_scores={
                        "logical": 90.0,
                        "completeness": 80.0,
                        "clarity": 70.0,
                    },
                    auditor_scores={
                        "logical": 85.0,
                        "completeness": 75.0,
                        "clarity": 65.0,
                    },
                    violations=(),
                    final_dimensions={
                        "logical": 85.0,
                        "completeness": 75.0,
                        "clarity": 65.0,
                    },
                    hard_caps={},
                    total_usage={
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    },
                    calls=(
                        {
                            "stage": "rubric_scorer",
                            "token_usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 1,
                                "total_tokens": 5,
                            },
                            "raw_usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 1,
                                "total_tokens": 5,
                            },
                        },
                        {
                            "stage": "adversarial_auditor",
                            "token_usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 2,
                                "total_tokens": 7,
                            },
                            "raw_usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 2,
                                "total_tokens": 7,
                            },
                        },
                    ),
                )

        return Evaluator()


class CapturingClient:
    def __init__(self, responses=None) -> None:
        self.messages = []
        self.requests = []
        self.responses = list(
            responses
            or [
                (
                    '{"logical":81.5,"completeness":72,"clarity":90}',
                    TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                ),
                (
                    '{"logical":79,"completeness":88,"clarity":84,'
                    '"violations":[]}',
                    TokenUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
                ),
            ]
        )

    def chat_json(self, messages, **kwargs):
        self.messages = messages
        self.requests.append({"messages": messages, **kwargs})
        content, usage = self.responses.pop(0)
        return LLMResponse(
            content=content,
            token_usage=usage,
            raw_payload={"usage": usage.to_dict()},
            response_format_mode="native_json_schema_strict",
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

    def execute(
        self,
        controller: FakeReasoningController,
        *,
        output_name: str = "evaluation",
        workers: int = 2,
    ):
        return run_reasoning_evaluation(
            submission_path=self.submission,
            questions=self.questions,
            model_config=self.model,
            output_dir=self.root / output_name,
            workers=workers,
            evaluator_factory=controller.factory,
            evaluator_factory_identity="fake-reasoning-controller-v1",
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

    def test_second_stage_timeout_is_unobservable_and_resume_never_resends(
        self,
    ) -> None:
        class SecondStageTimeoutClient:
            def __init__(self, config: ModelConfig) -> None:
                self.config = config
                self.calls = 0

            def chat_json(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    error = requests.ReadTimeout(
                        "ambiguous second-stage timeout"
                    )
                    error.transport_attempt_count = 1
                    error.transport_rejections = ()
                    raise error
                usage = TokenUsage(10, 2, 12)
                return LLMResponse(
                    content=(
                        '{"logical":81,"completeness":72,"clarity":90}'
                    ),
                    token_usage=usage,
                    raw_payload={"usage": usage.to_dict()},
                    response_format_mode="native_json_schema_strict",
                )

        client = SecondStageTimeoutClient(self.model)
        output = self.root / "second-stage-timeout"
        with patch.object(
            reasoning_module,
            "OpenAICompatibleClient",
            return_value=client,
        ):
            first = run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=self.model,
                output_dir=output,
                workers=1,
            )

        self.assertEqual(client.calls, 2)
        self.assertEqual(first.evaluations["q1"].status, "judge_error")
        self.assertEqual(
            first.manifest["judge_token_usage"]["total_tokens"], 12
        )
        self.assertTrue(first.manifest["unobservable_usage_risk"])
        failure = json.loads(
            (output / "reasoning_checkpoints" / "q1.json").read_text(
                encoding="utf-8"
            )
        )["failure"]
        self.assertEqual(
            failure["unobserved_stages"], ["adversarial_auditor"]
        )
        self.assertTrue(failure["unobservable_usage_risk"])

        with patch.object(
            reasoning_module,
            "OpenAICompatibleClient",
            side_effect=AssertionError("resume must not resend"),
        ):
            resumed = run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=self.model,
                output_dir=output,
                workers=1,
            )
        self.assertTrue(resumed.manifest["unobservable_usage_risk"])

    def test_parallel_failures_are_canonically_sorted_and_resume(self) -> None:
        second_reasoning = (
            "定位第二份材料的适用条款，提取关键事实并逐项比较，"
            "最终依据比较结果形成明确结论。"
        )
        write_b_submission(
            self.submission,
            self.questions,
            [
                BAnswer("q1", ("A",), 10, 2, 12, self.long_reasoning),
                BAnswer("q2", ("B",), 20, 3, 23, second_reasoning),
            ],
        )
        q2_finished = threading.Event()

        class ReverseFailureController:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def factory(controller):
                class Evaluator:
                    def evaluate(self, reasoning: str):
                        controller.calls.append(reasoning)
                        if reasoning == self_outer.long_reasoning:
                            q2_finished.wait(timeout=1)
                            time.sleep(0.02)
                        else:
                            q2_finished.set()
                        raise RuntimeError("expected parallel failure")

                return Evaluator()

        self_outer = self
        controller = ReverseFailureController()
        first = self.execute(
            controller,  # type: ignore[arg-type]
            output_name="parallel-failures",
            workers=2,
        )
        self.assertEqual(first.manifest["failure_count"], 2)
        failure_rows = reasoning_module._read_jsonl(
            self.root
            / "parallel-failures"
            / "reasoning_judge_failures.jsonl"
        )
        self.assertEqual(
            [row["qid"] for row in failure_rows], ["q1", "q2"]
        )

        calls_before_resume = list(controller.calls)
        second = self.execute(
            controller,  # type: ignore[arg-type]
            output_name="parallel-failures",
            workers=2,
        )
        self.assertEqual(controller.calls, calls_before_resume)
        self.assertEqual(second.manifest["status"], "complete")

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
                evaluator_factory_identity="fake-reasoning-controller-v1",
                accuracy_score=97,
                accuracy_source="official_submission_007",
            )

    def test_partial_usage_mirror_tampering_refuses_resume(self) -> None:
        self.execute(FakeReasoningController())
        usage_path = (
            self.root
            / "evaluation"
            / "reasoning_judge_usage_partial.json"
        )
        payload = json.loads(usage_path.read_text(encoding="utf-8"))
        payload["q1"]["total_tokens"] = 300
        usage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ReasoningEvaluationFingerprintError,
            "mirror drifted",
        ):
            self.execute(FakeReasoningController())

    def test_checkpoint_binding_and_final_usage_tampering_refuse_resume(self) -> None:
        self.execute(FakeReasoningController(), output_name="binding")
        checkpoint = (
            self.root
            / "binding"
            / "reasoning_checkpoints"
            / "q1.json"
        )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        payload["sealed_reasoning_sha256"] = "0" * 64
        checkpoint.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ReasoningEvaluationFingerprintError,
            "sealed reasoning mismatch",
        ):
            self.execute(
                FakeReasoningController(),
                output_name="binding",
            )

        self.execute(FakeReasoningController(), output_name="final-usage")
        usage_path = self.root / "final-usage" / "reasoning_judge_usage.json"
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
        usage["total"]["total_tokens"] = 300
        usage_path.write_text(
            json.dumps(usage, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ReasoningEvaluationFingerprintError,
            "usage artifact changed",
        ):
            self.execute(
                FakeReasoningController(),
                output_name="final-usage",
            )

    def test_evaluation_output_lock_and_custom_factory_identity_are_required(
        self,
    ) -> None:
        output = self.root / "locked"
        with reasoning_module._exclusive_evaluation_lock(output):
            with self.assertRaisesRegex(
                ReasoningEvaluationFingerprintError,
                "already active",
            ):
                self.execute(
                    FakeReasoningController(),
                    output_name="locked",
                )
        with self.assertRaisesRegex(ValueError, "explicit stable identity"):
            run_reasoning_evaluation(
                submission_path=self.submission,
                questions=self.questions,
                model_config=self.model,
                output_dir=self.root / "missing-factory-identity",
                evaluator_factory=FakeReasoningController().factory,
            )

    def test_orphan_stage_is_charged_scored_zero_and_never_resent(self) -> None:
        output = self.root / "orphan-stage"
        call = {
            "stage": "rubric_scorer",
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
            "raw_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
            "transport_attempt_count": 1,
            "transport_rejections": [],
        }
        reasoning_module._record_reasoning_stage_intent(
            output,
            qid="q1",
            reasoning=self.long_reasoning,
            run_fingerprint="run-sha",
            run_instance_id="run-instance",
            stage="rubric_scorer",
        )
        reasoning_module._append_reasoning_stage_call(
            output,
            qid="q1",
            reasoning=self.long_reasoning,
            run_fingerprint="run-sha",
            run_instance_id="run-instance",
            call=call,
        )

        evaluations, usage, calls, failures = (
            reasoning_module._load_evaluation_checkpoints(
                output,
                {"q1"},
                require_call_trace=True,
                reasoning_by_qid={"q1": self.long_reasoning},
                run_fingerprint="run-sha",
                run_instance_id="run-instance",
                repair_partial_mirrors=True,
            )
        )

        self.assertEqual(evaluations["q1"].status, "judge_error")
        self.assertEqual(usage["q1"]["total_tokens"], 14)
        self.assertEqual(calls["q1"], [call])
        self.assertEqual(failures[0]["error_type"], "OrphanStageCheckpoint")
        self.assertTrue(
            (output / "reasoning_checkpoints" / "q1.json").exists()
        )
        with self.assertRaisesRegex(
            ReasoningEvaluationFingerprintError,
            "final checkpoint already exists",
        ):
            reasoning_module._append_reasoning_stage_call(
                output,
                qid="q1",
                reasoning=self.long_reasoning,
                run_fingerprint="run-sha",
                run_instance_id="run-instance",
                call={
                    **call,
                    "stage": "adversarial_auditor",
                },
            )

    def test_pre_call_intent_closes_post_response_pre_ledger_crash_window(
        self,
    ) -> None:
        output = self.root / "intent-crash"
        client = CapturingClient()
        evaluator = FixedReasoningEvaluator(
            client,  # type: ignore[arg-type]
            stage_intent=lambda stage: (
                reasoning_module._record_reasoning_stage_intent(
                    output,
                    qid="q1",
                    reasoning=self.long_reasoning,
                    run_fingerprint="run-sha",
                    run_instance_id="run-instance",
                    stage=stage,
                )
            ),
            stage_checkpoint=lambda call: (_ for _ in ()).throw(
                SystemExit("simulated process death")
            ),
        )

        with self.assertRaises(SystemExit):
            evaluator.evaluate(self.long_reasoning)

        evaluations, usage, calls, failures = (
            reasoning_module._load_evaluation_checkpoints(
                output,
                {"q1"},
                require_call_trace=True,
                reasoning_by_qid={"q1": self.long_reasoning},
                run_fingerprint="run-sha",
                run_instance_id="run-instance",
                repair_partial_mirrors=True,
            )
        )
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(evaluations["q1"].status, "judge_error")
        self.assertEqual(usage["q1"]["total_tokens"], 0)
        self.assertEqual(calls["q1"], [])
        self.assertTrue(failures[0]["unobservable_usage_risk"])
        self.assertEqual(failures[0]["error_type"], "UnobservableStageIntent")

    def test_copied_output_directory_cannot_reuse_another_run(self) -> None:
        self.execute(FakeReasoningController(), output_name="run-a")
        copied_submission = self.root / "copied-submit.csv"
        shutil.copyfile(self.submission, copied_submission)
        copied_output = self.root / "run-b"
        shutil.copytree(self.root / "run-a", copied_output)

        with self.assertRaisesRegex(
            ReasoningEvaluationFingerprintError,
            "submission path changed|output directory changed",
        ):
            run_reasoning_evaluation(
                submission_path=copied_submission,
                questions=self.questions,
                model_config=self.model,
                output_dir=copied_output,
                evaluator_factory=FakeReasoningController().factory,
                evaluator_factory_identity="fake-reasoning-controller-v1",
                accuracy_score=97,
                accuracy_source="official_submission_007",
            )

    def test_running_manifest_rebuilds_missing_partial_mirror_from_checkpoint(
        self,
    ) -> None:
        output = self.root / "mirror-crash"
        original_write_json = reasoning_module.write_json
        failed = False

        def crash_once(path, payload):
            nonlocal failed
            if Path(path).name == "reasoning_scores.json" and not failed:
                failed = True
                raise OSError("simulated mirror write crash")
            return original_write_json(path, payload)

        with patch.object(reasoning_module, "write_json", side_effect=crash_once):
            with self.assertRaisesRegex(OSError, "mirror write crash"):
                self.execute(
                    FakeReasoningController(),
                    output_name="mirror-crash",
                    workers=1,
                )
        self.assertTrue(
            (output / "reasoning_checkpoints" / "q1.json").exists()
        )
        self.assertFalse((output / "reasoning_scores.json").exists())

        resumed = FakeReasoningController()
        result = self.execute(
            resumed,
            output_name="mirror-crash",
            workers=1,
        )

        self.assertEqual(resumed.calls, [])
        self.assertEqual(result.manifest["status"], "complete")
        self.assertTrue((output / "reasoning_scores.json").exists())

    def test_evaluator_identity_covers_protocol_and_retry_configuration(self) -> None:
        native = ModelConfig(
            api_key="x",
            api_base="https://example.invalid/v1",
            model_name="gpt-5.6",
            temperature=0.0,
            structured_output_mode="native_json_schema_strict",
            read_timeout_seconds=120,
            max_retries=2,
        )
        local = ModelConfig(
            api_key="x",
            api_base="https://example.invalid/v1",
            model_name="gpt-5.6",
            temperature=0.0,
            structured_output_mode="json_object_local_schema",
            read_timeout_seconds=240,
            max_retries=0,
        )

        self.assertNotEqual(
            reasoning_module._evaluator_identity(native),
            reasoning_module._evaluator_identity(local),
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

    def test_dual_stage_usage_and_corpus_audit_are_shadow_only_artifacts(self) -> None:
        second_reasoning = (
            "定位另一段冻结摘要，提取其中两个关键事实并完成比较，"
            "根据比较结果形成明确结论。"
        )
        write_b_submission(
            self.submission,
            self.questions,
            [
                BAnswer("q1", ("A",), 10, 2, 12, self.long_reasoning),
                BAnswer("q2", ("B",), 20, 3, 23, second_reasoning),
            ],
        )
        controller = StagedReasoningController()

        result = self.execute(controller, output_name="staged")

        self.assertEqual(result.manifest["submission_token_total"], 35)
        self.assertEqual(result.manifest["judge_token_usage"]["total_tokens"], 24)
        self.assertFalse(result.manifest["judge_tokens_included_in_submission"])
        usage = json.loads(
            (self.root / "staged" / "reasoning_judge_usage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [call["stage"] for call in usage["calls_by_qid"]["q1"]],
            ["rubric_scorer", "adversarial_auditor"],
        )
        self.assertFalse(usage["included_in_submission_token_score"])
        corpus = json.loads(
            (self.root / "staged" / "reasoning_corpus_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(corpus["question_count"], 2)
        self.assertFalse(corpus["scores_mutated"])


class FixedReasoningEvaluatorTests(unittest.TestCase):
    def test_dual_stage_judge_sees_only_frozen_reasoning_and_uses_dimension_minimum(
        self,
    ) -> None:
        client = CapturingClient()
        evaluator = FixedReasoningEvaluator(client)  # type: ignore[arg-type]
        reasoning = "先定位数据，再完成计算，最后根据结果得出结论。"

        outcome = evaluator.evaluate(reasoning)

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            [request["messages"][1] for request in client.requests],
            [{"role": "user", "content": reasoning}] * 2,
        )
        self.assertEqual(client.requests[0]["messages"][0]["content"], SCORER_SYSTEM_PROMPT)
        self.assertEqual(client.requests[1]["messages"][0]["content"], AUDITOR_SYSTEM_PROMPT)
        self.assertTrue(
            all(
                request["response_format_mode"]
                if "response_format_mode" in request
                else request["response_schema"]["additionalProperties"] is False
                for request in client.requests
            )
        )
        self.assertEqual(
            outcome.final_dimensions,
            {"logical": 79.0, "completeness": 72.0, "clarity": 84.0},
        )
        self.assertEqual(outcome.total_usage["total_tokens"], 26)
        self.assertEqual([call["stage"] for call in outcome.calls], ["rubric_scorer", "adversarial_auditor"])

    def test_local_json_mode_keeps_the_same_strict_local_contract(self) -> None:
        client = CapturingClient()
        client.config = ModelConfig(
            api_key="x",
            api_base="https://example.invalid/v1",
            model_name="gpt-5.6",
            structured_output_mode="json_object_local_schema",
        )

        outcome = FixedReasoningEvaluator(client).evaluate(
            "先提取80与100两个数值，再计算差额20，最后明确得到增长结论。"
        )

        self.assertEqual(len(client.requests), 2)
        self.assertTrue(
            all("response_schema" not in request for request in client.requests)
        )
        self.assertEqual(outcome.final_dimensions["logical"], 79.0)

    def test_invalid_dimension_is_rejected_with_usage_attached(self) -> None:
        client = CapturingClient(
            responses=[
                (
                    '{"logical":80,"completeness":80,"clarity":80}',
                    TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                ),
                (
                    '{"logical":true,"completeness":80,"clarity":80,"violations":[]}',
                    TokenUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
                ),
            ]
        )

        with self.assertRaisesRegex(Exception, "invalid reasoning judge response") as raised:
            FixedReasoningEvaluator(client).evaluate(
                "这是足够长的推理过程文本，用于验证非法评分字段会被拒绝。"
            )  # type: ignore[arg-type]
        self.assertEqual(getattr(raised.exception, "token_usage")["total_tokens"], 26)
        self.assertEqual(
            [call["stage"] for call in getattr(raised.exception, "calls")],
            ["rubric_scorer", "adversarial_auditor"],
        )

    def test_auditor_violations_apply_the_fixed_dimension_caps(self) -> None:
        client = CapturingClient(
            responses=[
                (
                    '{"logical":94,"completeness":93,"clarity":92}',
                    TokenUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10),
                ),
                (
                    '{"logical":90,"completeness":91,"clarity":89,'
                    '"violations":["no_concrete_support","no_explicit_conclusion",'
                    '"machine_id_only"]}',
                    TokenUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
                ),
            ]
        )

        outcome = FixedReasoningEvaluator(client).evaluate(
            "根据E01和chunk_7进行核验，相关内容可以得到上述判断。"
        )  # type: ignore[arg-type]

        self.assertEqual(
            outcome.final_dimensions,
            {"logical": 90.0, "completeness": 39.0, "clarity": 59.0},
        )
        self.assertEqual(
            outcome.hard_caps,
            {"completeness": 39.0, "clarity": 59.0},
        )


class ReasoningHardCapTests(unittest.TestCase):
    def test_each_confirmed_violation_caps_only_the_required_dimensions(self) -> None:
        base = {"logical": 95, "completeness": 95, "clarity": 95}

        self.assertEqual(
            apply_reasoning_hard_caps(base, ["generic_or_repetition_only"])[0],
            {"logical": 29.0, "completeness": 29.0, "clarity": 29.0},
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["no_concrete_support"])[0]["completeness"],
            39.0,
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["no_reasoning_relation"])[0]["logical"],
            59.0,
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["no_explicit_conclusion"])[0]["completeness"],
            59.0,
        )
        self.assertEqual(
            apply_reasoning_hard_caps(
                base, ["internal_contradiction_or_arithmetic_error"]
            )[0]["logical"],
            29.0,
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["machine_id_only"])[0],
            {"logical": 95.0, "completeness": 59.0, "clarity": 59.0},
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["calculation_steps_missing"])[0][
                "completeness"
            ],
            59.0,
        )
        self.assertEqual(
            apply_reasoning_hard_caps(base, ["multi_labels_only"])[0]["completeness"],
            59.0,
        )

    def test_prompt_and_schema_hashes_are_fixed_and_distinct(self) -> None:
        from afa_agent.b_board.reasoning_evaluation import reasoning_prompt_fingerprint

        self.assertRegex(reasoning_prompt_fingerprint(), r"^[0-9a-f]{64}$")
        self.assertRegex(reasoning_schema_fingerprint(), r"^[0-9a-f]{64}$")
        self.assertNotEqual(reasoning_prompt_fingerprint(), reasoning_schema_fingerprint())


class ReasoningCalibrationTests(unittest.TestCase):
    def test_0_30_60_80_90_sentinels_must_be_monotonic(self) -> None:
        sentinels = build_reasoning_sentinels()
        self.assertEqual([item["target"] for item in sentinels], [0, 30, 60, 80, 90])
        evaluations = {
            item["sentinel_id"]: ReasoningEvaluation(
                qid=item["sentinel_id"],
                logical=float(item["target"]),
                completeness=float(item["target"]),
                clarity=float(item["target"]),
            )
            for item in sentinels
        }

        passed = validate_reasoning_sentinels(evaluations)
        self.assertTrue(passed["passed"])

        evaluations["sentinel_80"] = ReasoningEvaluation(
            qid="sentinel_80", logical=40, completeness=40, clarity=40
        )
        failed = validate_reasoning_sentinels(evaluations)
        self.assertFalse(failed["passed"])
        self.assertIn("sentinel_80", failed["failures"][0])

    def test_cross_corpus_template_audit_flags_clusters_without_applying_caps(self) -> None:
        corpus = {
            f"q{index:03d}": (
                f"主体为公司{index}。关键数值为{100 + index}，根据材料完成比较和计算。"
                f"因此可以得到对应结论{index % 4}。"
            )
            for index in range(1, 101)
        }
        corpus["q099"] = corpus["q098"]

        report = audit_reasoning_corpus(corpus)

        self.assertEqual(report["question_count"], 100)
        self.assertFalse(report["scores_mutated"])
        self.assertIn(["q098", "q099"], report["exact_duplicate_groups"])
        self.assertTrue(report["template_clusters"])
        self.assertIn("q099", report["flagged_qids"])
        self.assertEqual(report["suggested_reasoning_cap"], 29)


class ReasoningEvaluationCliTests(unittest.TestCase):
    def test_select_questions_preserves_requested_order_and_rejects_unknown(
        self,
    ) -> None:
        questions = [question("q1"), question("q2"), question("q3")]

        selected = _select_questions(questions, ["q3", "q1", "q3"])

        self.assertEqual([item.qid for item in selected], ["q3", "q1"])
        with self.assertRaisesRegex(ValueError, "unknown qids"):
            _select_questions(questions, ["q4"])

    def test_env_root_is_an_explicit_path_and_workers_remain_configurable(self) -> None:
        args = parse_args(
            [
                "--submission",
                "candidate.csv",
                "--output-dir",
                "judge-output",
                "--env-root",
                "/tmp/shared-config",
                "--workers",
                "9",
            ]
        )

        self.assertEqual(args.env_root, Path("/tmp/shared-config"))
        self.assertEqual(args.workers, 9)


if __name__ == "__main__":
    unittest.main()
