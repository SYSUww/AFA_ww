from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.candidate_scorecard import (
    answer_parts_equivalent,
    evaluate_candidate_run,
)
from afa_agent.b_board.io import SUBMISSION_COLUMNS
from afa_agent.b_board.reasoning_evaluation import (
    PROMPT_VERSION as REASONING_PROMPT_VERSION,
)
from afa_agent.b_board.runner import SUBMISSION_REASONING_PROMPT_VERSION


class CandidateScorecardTests(unittest.TestCase):
    def test_full_compliant_match_emits_pseudo_accuracy_not_official_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, reference_manifest, locks, reasoning_dir = self._build_fixture(root)

            scorecard = evaluate_candidate_run(
                run_dir=run_dir,
                reference_manifest_path=reference_manifest,
                official_locks_path=locks,
                reasoning_evaluation_dir=reasoning_dir,
            )

        self.assertTrue(scorecard["compliance_passed"])
        self.assertEqual(scorecard["reference_equivalent_match_count"], 2)
        self.assertEqual(scorecard["accuracy_proxy_score"], 99.0)
        self.assertIsNone(scorecard["official_accuracy_score"])
        self.assertIsNone(scorecard["official_total_score"])
        self.assertAlmostEqual(scorecard["proxy_total_score"], 76.512)

    def test_mismatch_and_official_lock_regression_suppress_accuracy_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, reference_manifest, locks, _ = self._build_fixture(root)
            answers_path = run_dir / "answers.json"
            answers = json.loads(answers_path.read_text(encoding="utf-8"))
            answers[1]["answer_parts"] = ["AC"]
            answers_path.write_text(json.dumps(answers), encoding="utf-8")

            scorecard = evaluate_candidate_run(
                run_dir=run_dir,
                reference_manifest_path=reference_manifest,
                official_locks_path=locks,
            )

        self.assertFalse(scorecard["compliance_passed"])
        self.assertEqual(scorecard["reference_equivalent_match_count"], 1)
        self.assertEqual(scorecard["official_lock_regressions"][0]["qid"], "q2")
        self.assertIsNone(scorecard["accuracy_proxy_score"])
        self.assertIsNone(scorecard["proxy_total_score"])

    def test_missing_submit_csv_cannot_pass_compliance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, reference_manifest, locks, _ = self._build_fixture(root)
            (run_dir / "submit.csv").unlink()

            scorecard = evaluate_candidate_run(
                run_dir=run_dir,
                reference_manifest_path=reference_manifest,
                official_locks_path=locks,
            )

        self.assertFalse(scorecard["compliance_passed"])
        self.assertIn("submission_csv_is_missing", scorecard["compliance_failures"])

    def test_reasoning_rescue_evidence_does_not_mutate_answer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, reference_manifest, locks, reasoning_dir = self._build_fixture(root)
            answers_path = run_dir / "answers.json"
            answers = json.loads(answers_path.read_text(encoding="utf-8"))
            answers[0]["decision_trace"]["submission_reasoning"][
                "rescued_evidence_ids"
            ] = ["q1:r1"]
            answers[0]["reasoning_evidence_items"] = [
                *answers[0]["evidence_items"],
                {"unit_id": "q1:r1", "text": "仅供 reasoning 补强的证据"},
            ]
            answers_path.write_text(json.dumps(answers), encoding="utf-8")

            scorecard = evaluate_candidate_run(
                run_dir=run_dir,
                reference_manifest_path=reference_manifest,
                official_locks_path=locks,
                reasoning_evaluation_dir=reasoning_dir,
            )

        self.assertTrue(scorecard["compliance_passed"])
        self.assertEqual(
            [item["unit_id"] for item in answers[0]["evidence_items"]],
            ["q1:e1"],
        )

    def test_reasoning_score_must_be_sealed_to_exact_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, reference_manifest, locks, reasoning_dir = self._build_fixture(root)
            with (run_dir / "submit.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(
                ValueError, "evaluated against another submission"
            ):
                evaluate_candidate_run(
                    run_dir=run_dir,
                    reference_manifest_path=reference_manifest,
                    official_locks_path=locks,
                    reasoning_evaluation_dir=reasoning_dir,
                )

    def test_numeric_equivalence_preserves_units(self) -> None:
        self.assertTrue(answer_parts_equivalent(["67.10"], ["67.1"]))
        self.assertTrue(answer_parts_equivalent(["1,000.00%"], ["1000%"]))
        self.assertFalse(answer_parts_equivalent(["67.1%"], ["67.1"]))

    @staticmethod
    def _build_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
        run_dir = root / "run"
        reference_dir = root / "reference"
        run_dir.mkdir()
        reference_dir.mkdir()
        reference_path = reference_dir / "reference_answers.json"
        reference_rows = [
            {"qid": "q1", "answer_parts": ["67.1"]},
            {"qid": "q2", "answer_parts": ["BD"]},
        ]
        reference_path.write_text(json.dumps(reference_rows), encoding="utf-8")
        reference_manifest = reference_dir / "reference_manifest.json"
        reference_manifest.write_text(
            json.dumps(
                {
                    "label": "pseudo99_unsubmitted",
                    "officially_submitted": False,
                    "predicted_accuracy": 99,
                    "reference_answers": str(reference_path),
                    "reference_sha256": hashlib.sha256(
                        reference_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        locks = root / "locks.json"
        locks.write_text(
            json.dumps({"answers": {"q2": ["BD"]}}), encoding="utf-8"
        )

        answers = []
        ledger_rows = []
        total_prompt = 0
        total_completion = 0
        for qid, parts in (("q1", ["67.10"]), ("q2", ["BD"])):
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
            total_prompt += usage["prompt_tokens"]
            total_completion += usage["completion_tokens"]
            reasoning = (
                "定位证据并核对关键事实，数值67.10与题目口径一致，完成推导后最终答案为67.10。"
                if qid == "q1"
                else "定位相关条款并逐项核对，B和D均由证据直接支持，完成推导后最终答案为BD。"
            )
            answers.append(
                {
                    "qid": qid,
                    "answer_format": "extraction",
                    "answer_parts": parts,
                    "used_evidence_ids": [f"{qid}:e1"],
                    "evidence_items": [
                        {"unit_id": f"{qid}:e1", "text": "直接支持答案的证据"}
                    ],
                    "decision_summary": reasoning,
                    "decision_trace": {
                        "submission_reasoning": {
                            "prompt_version": SUBMISSION_REASONING_PROMPT_VERSION,
                            "grounding_status": "supported",
                            "answer_parts_preserved": True,
                            "model_name": "qwen3.7-plus",
                            "rescued_evidence_ids": [],
                        }
                    },
                    "token_usage": usage,
                }
            )
            ledger_rows.append(
                {
                    "qid": qid,
                    "status": "success",
                    "call_count": 1,
                    "calls": [
                        {
                            "model_name": "qwen3.7-plus",
                            "token_usage": usage,
                        }
                    ],
                    "token_usage": usage,
                }
            )
        (run_dir / "answers.json").write_text(
            json.dumps(answers), encoding="utf-8"
        )
        (run_dir / "usage_ledger.jsonl").write_text(
            "\n".join(json.dumps(row) for row in ledger_rows) + "\n",
            encoding="utf-8",
        )
        total = total_prompt + total_completion
        submission_path = (run_dir / "submit.csv").resolve()
        with submission_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "qid": "summary",
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total,
                }
            )
            for answer in answers:
                usage = answer["token_usage"]
                writer.writerow(
                    {
                        "qid": answer["qid"],
                        "answer1": answer["answer_parts"][0],
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "reasoning": answer["decision_summary"],
                    }
                )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "expected_question_count": 2,
                    "answered_question_count": 2,
                    "failed_qids": [],
                    "run_mode": "submission",
                    "submission_eligible": True,
                    "submission_path": str(submission_path),
                    "model": {"model_name": "qwen3.7-plus"},
                    "token_usage": {
                        "prompt_tokens": total_prompt,
                        "completion_tokens": total_completion,
                        "total_tokens": total,
                    },
                    "generation_token_usage": {
                        "prompt_tokens": total_prompt,
                        "completion_tokens": total_completion,
                        "total_tokens": total,
                    },
                }
            ),
            encoding="utf-8",
        )
        reasoning_dir = root / "reasoning"
        reasoning_dir.mkdir()
        sealed = [
            {"qid": answer["qid"], "reasoning": answer["decision_summary"]}
            for answer in answers
        ]
        sealed_path = (reasoning_dir / "sealed_reasoning.json").resolve()
        sealed_path.write_text(json.dumps(sealed), encoding="utf-8")
        score_rows = [
            {
                "qid": answer["qid"],
                "logical": 90,
                "completeness": 90,
                "clarity": 90,
                "reasoning_score": 90,
                "status": "scored",
            }
            for answer in answers
        ]
        (reasoning_dir / "reasoning_scores.json").write_text(
            json.dumps(score_rows), encoding="utf-8"
        )
        aggregate = {
            "question_count": 2,
            "reasoning_score": 90.0,
            "dimension_means": {
                "logical": 90.0,
                "completeness": 90.0,
                "clarity": 90.0,
            },
        }
        (reasoning_dir / "reasoning_aggregate.json").write_text(
            json.dumps(aggregate), encoding="utf-8"
        )
        identity = {
            "prompt_version": REASONING_PROMPT_VERSION,
            "schema_version": 1,
            "model_name": "gpt-5.6",
            "temperature": 0.0,
        }
        components = {
            "evaluator_identity": identity,
            "submission": {
                "sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
                "has_summary": True,
            },
            "sealed_reasoning": {
                "count": 2,
                "sha256": CandidateScorecardTests._payload_sha256(sealed),
            },
        }
        (reasoning_dir / "reasoning_evaluator_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "evaluator_identity": identity,
                    "fingerprint": {
                        "schema_version": 1,
                        "sha256": CandidateScorecardTests._payload_sha256(
                            components
                        ),
                        "components": components,
                    },
                    "sealed_reasoning_path": str(sealed_path),
                    "expected_reasoning_count": 2,
                    "evaluated_reasoning_count": 2,
                    "failure_count": 0,
                    "reasoning_aggregate": aggregate,
                }
            ),
            encoding="utf-8",
        )
        return run_dir, reference_manifest, locks, reasoning_dir

    @staticmethod
    def _payload_sha256(payload: object) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
