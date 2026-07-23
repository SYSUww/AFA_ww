from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.merge import assemble_answer_run, hydrate_reasoning_patch
from afa_agent.b_board.runner import _answer_artifact_signature, _artifact_from_dict
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
        "decision_summary": "定位到题目对应的直接数值证据，按题目要求保留两位小数，因此得到该答案。",
        "decision_trace": {
            "api_usage_ledger": {
                "call_count": 1,
                "calls": [
                    {
                        "call_index": 1,
                        "model_name": "qwen3.5-plus",
                        "token_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    }
                ],
            }
        },
        "calculation_trace": {"replay_verified": True},
        "token_usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "locator": {},
    }


class BBoardMergeTests(unittest.TestCase):
    def test_reasoning_patch_hydrates_exact_frozen_answer_usage(self) -> None:
        answer = _artifact_from_dict(artifact("q1", "1.00"))
        answer.decision_trace["answer_api_usage_ledger"] = dict(
            answer.decision_trace["api_usage_ledger"]
        )
        patch = _artifact_from_dict(artifact("q1", "1.00"))
        patch.decision_summary = "证据先定位原始数值，再按题目要求保留两位小数，最终答案为1.00。"
        patch.reasoning_evidence_items = [{"unit_id": "q1:reasoning", "text": "1.00"}]
        patch.token_usage = {
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "total_tokens": 9,
        }
        patch.decision_trace = {
            "api_usage_ledger": {
                "call_count": 2,
                "calls": [
                    {
                        "call_index": 1,
                        "model_name": "qwen3.5-plus",
                        "token_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    },
                    {
                        "call_index": 2,
                        "model_name": "qwen3.7-plus-2026-05-26",
                        "token_usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                ],
            },
            "answer_api_usage_ledger": dict(
                answer.decision_trace["answer_api_usage_ledger"]
            ),
            "reasoning_api_usage_ledger": {
                "call_count": 1,
                "calls": [
                    {
                        "call_index": 1,
                        "model_name": "qwen3.7-plus-2026-05-26",
                        "token_usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                ],
            },
            "submission_reasoning": {
                "answer_parts_preserved": True,
                "grounding_status": "supported",
            },
            "reasoning_stage": {
                "status": "complete",
                "answer_artifact_frozen": True,
                "answer_artifact_sha256": _answer_artifact_signature(answer),
            },
        }

        hydrated = hydrate_reasoning_patch(
            answer_artifact=answer,
            reasoning_artifact=patch,
        )

        self.assertEqual(hydrated.answer_parts, ["1.00"])
        self.assertEqual(hydrated.decision_summary, patch.decision_summary)
        self.assertEqual(hydrated.token_usage["total_tokens"], 9)
        self.assertEqual(
            hydrated.decision_trace["api_usage_ledger"]["call_count"],
            2,
        )
        self.assertEqual(
            hydrated.decision_trace["reasoning_patch_lineage"]["answer_artifact_sha256"],
            _answer_artifact_signature(answer),
        )

        patch.decision_trace["reasoning_stage"]["answer_artifact_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "not sealed"):
            hydrate_reasoning_patch(
                answer_artifact=answer,
                reasoning_artifact=patch,
            )

    def test_reasoning_patch_accepts_explicit_zero_call_answer_checkpoint(self) -> None:
        answer = _artifact_from_dict(artifact("q1", "1.00"))
        answer.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        answer.decision_trace = {
            "answer_api_usage_ledger": {
                "call_count": 0,
                "calls": [],
            },
            "api_usage_ledger": {
                "call_count": 0,
                "calls": [],
            },
        }
        patch = _artifact_from_dict(artifact("q1", "1.00"))
        patch.decision_trace = {
            **patch.decision_trace,
            "reasoning_api_usage_ledger": dict(
                patch.decision_trace["api_usage_ledger"]
            ),
            "submission_reasoning": {
                "answer_parts_preserved": True,
                "grounding_status": "supported",
            },
            "reasoning_stage": {
                "status": "complete",
                "answer_artifact_frozen": True,
                "answer_artifact_sha256": _answer_artifact_signature(answer),
            },
        }

        hydrated = hydrate_reasoning_patch(
            answer_artifact=answer,
            reasoning_artifact=patch,
        )

        self.assertEqual(hydrated.token_usage["total_tokens"], 3)
        self.assertEqual(
            hydrated.decision_trace["api_usage_ledger"]["call_count"],
            1,
        )

    def test_later_source_repairs_missing_qid_and_writes_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            repair = root / "repair"
            base.mkdir()
            repair.mkdir()
            write_json(base / "run_manifest.json", {"model": {"model_name": "qwen3.5-plus"}})
            write_json(repair / "run_manifest.json", {"model": {"model_name": "qwen3.6"}})
            write_json(base / "answers.json", [artifact("q1", "1.00")])
            write_json(repair / "answers.json", [artifact("q2", "2.00")])

            manifest = assemble_answer_run(
                questions=[question("q1"), question("q2")],
                source_run_dirs=[base, repair],
                output_dir=root / "merged",
            )

            self.assertTrue(manifest["submission_valid"])
            self.assertTrue(manifest["submission_eligible"])
            self.assertEqual(
                manifest["generation_models"],
                ["qwen3.5-plus", "qwen3.6"],
            )
            self.assertTrue((root / "merged" / "submit.csv").exists())
            self.assertTrue((root / "merged" / "usage_ledger.jsonl").exists())
            self.assertEqual(
                manifest["generation_token_usage"],
                manifest["token_usage"],
            )
            self.assertEqual(len(read_json(root / "merged" / "answers.json")), 2)

    def test_invalid_baseline_is_evaluation_complete_but_not_submission_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_json(source / "run_manifest.json", {"model": {"model_name": "qwen3.5-plus"}})
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

    def test_disallowed_source_model_is_not_submission_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            write_json(source / "answers.json", [artifact("q1", "1.00")])
            write_json(source / "run_manifest.json", {"model": {"model_name": "gpt-5.5"}})

            manifest = assemble_answer_run(
                questions=[question("q1")],
                source_run_dirs=[source],
                output_dir=root / "merged",
            )

            self.assertFalse(manifest["submission_valid"])
            self.assertIn("allowed Qwen", manifest["submission_validation_failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
