from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.joint_choice import validate_joint_choice_payload
from afa_agent.b_board.runner import (
    RUN_MODE_SUBMISSION,
    BAnswerArtifact,
    BBoardActualRunner,
    _answer_checkpoint_from_completed_artifact,
)
from afa_agent.client import LLMResponse
from afa_agent.models import RetrievalHit, TokenUsage
from scripts.build_b_board_overlay_candidate import (
    _reject_research_only_joint_artifacts,
)
from scripts.analyze_b_joint_choice import (
    _sha256 as analysis_sha256,
    _validate_source_lineage,
)
from scripts.run_b_joint_choice_research import _persist as persist_joint_research


def _multi_question() -> BQuestion:
    return BQuestion(
        qid="joint_q1",
        domain="regulatory",
        split="B",
        question="下列说法正确的是哪些？",
        options={
            "A": "说法A",
            "B": "说法B",
            "C": "说法C",
            "D": "说法D",
        },
        answer_format="multi",
        type="多选题",
        answer_slots=1,
        answer_slot_templates=("A",),
    )


def _payload() -> dict[str, object]:
    return {
        "option_assessments": [
            {
                "option": "A",
                "verdict": "support",
                "evidence_ids": ["u1"],
                "reason": "证据明确说明说法A成立。",
            },
            {
                "option": "B",
                "verdict": "refute",
                "evidence_ids": ["u2"],
                "reason": "证据中的适用范围与说法B不一致。",
            },
            {
                "option": "C",
                "verdict": "support",
                "evidence_ids": ["u3"],
                "reason": "证据直接支持说法C的完整条件。",
            },
            {
                "option": "D",
                "verdict": "refute",
                "evidence_ids": ["u4"],
                "reason": "证据明确排除了说法D描述的情形。",
            },
        ],
        "answer_parts": ["AC"],
        "reasoning": (
            "材料中的适用条件直接支持A项和C项；B项范围不符，D项所述"
            "情形被明确排除。因此选择A、C两项。最终答案为AC。"
        ),
    }


class JointChoicePayloadTests(unittest.TestCase):
    def test_joint_artifact_marks_grounding_and_freeze_as_unverified(
        self,
    ) -> None:
        class FakeMigration:
            @staticmethod
            def _effective_attempt_for_domain(attempt, _domain):
                return attempt

            @staticmethod
            def select_answer_doc_ids(_locator, _question, _attempt):
                return ["doc-1"]

        class FakeRetriever:
            @staticmethod
            def search(_doc_ids, query, **_kwargs):
                option = next(
                    label
                    for label in ("A", "B", "C", "D")
                    if f"待判断选项{label}" in query
                )
                return [
                    RetrievalHit(
                        unit_id=f"u{ord(option) - ord('A') + 1}",
                        doc_id="doc-1",
                        score=1.0,
                        title_path=["条款"],
                        text=f"选项{option}的核验材料。",
                    )
                ]

        class FakeClient:
            @staticmethod
            def chat_json(_messages, **_kwargs):
                return LLMResponse(
                    content=json.dumps(_payload(), ensure_ascii=False),
                    token_usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                    raw_payload={},
                    response_format_mode="native_json_schema_strict",
                )

        runner = object.__new__(BBoardActualRunner)
        runner._migration = FakeMigration()
        runner.attempt = object()
        runner.retrievers = {"regulatory": FakeRetriever()}
        runner.client = FakeClient()
        runner.config = SimpleNamespace(
            model=SimpleNamespace(
                model_name="qwen3.7-plus-2026-05-26"
            )
        )

        artifact = runner._answer_choice_joint(
            _multi_question(),
            {},
            thinking_budget=2048,
            per_option_top_k=1,
            max_evidence_items=4,
            evidence_char_limit=1200,
        )

        self.assertFalse(
            artifact.decision_trace["answer_stage"][
                "answer_parts_frozen"
            ]
        )
        self.assertFalse(
            artifact.decision_trace["reasoning_stage"][
                "answer_artifact_frozen"
            ]
        )
        self.assertFalse(
            artifact.decision_trace["submission_reasoning"][
                "local_evidence_gate_passed"
            ]
        )
        self.assertEqual(
            artifact.decision_trace["submission_reasoning"][
                "grounding_status"
            ],
            "model_cited_unverified",
        )

    def test_joint_research_layout_does_not_publish_answers_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)

            persist_joint_research(
                output_dir=output_dir,
                selected=[],
                artifacts_by_qid={},
                failures=[],
            )

            self.assertTrue((output_dir / "joint_outputs.json").is_file())
            self.assertFalse((output_dir / "answers.json").exists())
            self.assertFalse((output_dir / "submit.csv").exists())

    def test_analysis_rejects_drifted_source_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_run = Path(directory)
            answers_path = source_run / "answers.json"
            submission_path = source_run / "submit.csv"
            manifest_path = source_run / "run_manifest.json"
            answers_path.write_text("[]", encoding="utf-8")
            submission_path.write_text("qid,answer1\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"submission_path": str(submission_path)}),
                encoding="utf-8",
            )
            lineage = {
                "run_dir": str(source_run),
                "manifest_sha256": analysis_sha256(manifest_path),
                "answers_sha256": analysis_sha256(answers_path),
                "submission_sha256": analysis_sha256(submission_path),
            }

            verified = _validate_source_lineage(
                source_run=source_run,
                joint_manifest={"source_lineage": lineage},
            )
            self.assertEqual(verified, lineage)

            answers_path.write_text("[{}]", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "source lineage mismatch",
            ):
                _validate_source_lineage(
                    source_run=source_run,
                    joint_manifest={"source_lineage": lineage},
                )

    def test_submission_overlay_rejects_research_joint_artifact(self) -> None:
        artifact = BAnswerArtifact(
            qid="joint_q1",
            domain="regulatory",
            answer_format="multi",
            answer_slot_count=1,
            answer_parts=["AC"],
            used_evidence_ids=["u1"],
            evidence_items=[{"unit_id": "u1", "text": "证据"}],
            decision_summary="联合生成的研究推理。最终答案为AC。",
            decision_trace={"joint_choice_generation": {}},
            calculation_trace={},
            token_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            locator={},
        )

        with self.assertRaisesRegex(
            ValueError,
            "cannot be overlaid into a submission",
        ):
            _reject_research_only_joint_artifacts(
                [artifact],
                run_dir=Path("/tmp/research-joint"),
            )

    def test_joint_generation_is_research_only(self) -> None:
        runner = object.__new__(BBoardActualRunner)
        runner.run_mode = RUN_MODE_SUBMISSION

        with self.assertRaisesRegex(ValueError, "research-only"):
            runner.joint_choice_one(
                _multi_question(),
                {},
            )

    def test_joint_usage_has_one_owner_and_cannot_fake_frozen_checkpoint(
        self,
    ) -> None:
        call = {
            "call_index": 1,
            "call_id": "joint_q1:joint_choice:1",
            "model_name": "qwen3.7-plus-2026-05-26",
            "response_format_mode": "native_json_schema_strict",
            "accounting_owner": "answer",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        reasoning = (
            "材料中的适用条件直接支持A项和C项；B项范围不符，D项所述"
            "情形被明确排除。因此选择A、C两项。最终答案为AC。"
        )
        artifact = BAnswerArtifact(
            qid="joint_q1",
            domain="regulatory",
            answer_format="multi",
            answer_slot_count=1,
            answer_parts=["AC"],
            used_evidence_ids=["u1"],
            evidence_items=[{"unit_id": "u1", "text": "证据"}],
            reasoning_evidence_items=[
                {"unit_id": "u1", "text": "证据"}
            ],
            decision_summary=reasoning,
            decision_trace={
                "joint_choice_generation": {
                    "option_assessments": [],
                },
                "answer_stage": {
                    "status": "co_generated",
                    "answer_parts_frozen": False,
                    "post_response_answer_locked": True,
                    "decision_summary": reasoning,
                    "source_call_ids": ["joint_q1:joint_choice:1"],
                },
                "reasoning_stage": {
                    "status": "complete",
                    "answer_artifact_frozen": False,
                    "co_generated_with_answer": True,
                    "source_call_ids": ["joint_q1:joint_choice:1"],
                },
                "submission_reasoning": {
                    "answer_parts_preserved": False,
                    "answer_parts_consistent": True,
                    "source_call_ids": ["joint_q1:joint_choice:1"],
                },
                "answer_api_usage_ledger": {
                    "call_count": 1,
                    "calls": [call],
                },
                "reasoning_api_usage_ledger": {
                    "accounting_semantics": "incremental_only",
                    "call_count": 0,
                    "calls": [],
                    "source_call_ids": ["joint_q1:joint_choice:1"],
                },
                "api_usage_ledger": {
                    "accounting_semantics": "canonical_physical_calls",
                    "call_count": 1,
                    "calls": [call],
                },
            },
            calculation_trace={},
            token_usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            locator={},
        )

        self.assertEqual(
            artifact.decision_trace["answer_api_usage_ledger"][
                "call_count"
            ],
            1,
        )
        self.assertEqual(
            artifact.decision_trace["reasoning_api_usage_ledger"][
                "call_count"
            ],
            0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "no recoverable answer stage",
        ):
            _answer_checkpoint_from_completed_artifact(artifact)

    def test_accepts_grounded_answer_reasoning_payload(self) -> None:
        answer, assessments, reasoning = validate_joint_choice_payload(
            _payload(),
            question=_multi_question(),
            available_evidence_ids=["u1", "u2", "u3", "u4"],
        )

        self.assertEqual(answer, "AC")
        self.assertEqual(
            [item["option"] for item in assessments],
            ["A", "B", "C", "D"],
        )
        self.assertTrue(reasoning.endswith("最终答案为AC。"))

    def test_rejects_missing_option_assessment(self) -> None:
        payload = _payload()
        payload["option_assessments"] = payload["option_assessments"][:-1]

        with self.assertRaisesRegex(ValueError, "cover options exactly"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )

    def test_rejects_single_option_multi_answer(self) -> None:
        payload = _payload()
        payload["option_assessments"][2]["verdict"] = "refute"
        payload["answer_parts"] = ["A"]
        payload["reasoning"] = (
            "材料仅能支持A项，其他三个选项均与材料的适用条件冲突。"
            "最终答案为A。"
        )

        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )

    def test_rejects_answer_assessment_mismatch(self) -> None:
        payload = _payload()
        payload["answer_parts"] = ["AB"]
        payload["reasoning"] = (
            "材料支持A项和C项，但提交答案错误地写成了A项与B项。"
            "最终答案为AB。"
        )

        with self.assertRaisesRegex(ValueError, "all and only support"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )

    def test_rejects_unknown_evidence_id(self) -> None:
        payload = _payload()
        payload["option_assessments"][0]["evidence_ids"] = ["invented"]

        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )

    def test_rejects_insufficient_verdict_without_retry(self) -> None:
        payload = _payload()
        payload["option_assessments"][1]["verdict"] = "insufficient"
        payload["option_assessments"][1]["evidence_ids"] = []

        with self.assertRaisesRegex(ValueError, "remains insufficient"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )

    def test_rejects_reasoning_with_different_conclusion(self) -> None:
        payload = _payload()
        payload["reasoning"] = (
            "材料中的适用条件直接支持A项和C项，B项和D项均不成立。"
            "最终答案为AB。"
        )

        with self.assertRaisesRegex(ValueError, "must end with"):
            validate_joint_choice_payload(
                payload,
                question=_multi_question(),
                available_evidence_ids=["u1", "u2", "u3", "u4"],
            )


if __name__ == "__main__":
    unittest.main()
