from __future__ import annotations

import unittest

from afa_agent.b_board.candidate_overlay import merge_candidate_artifacts
from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.runner import BAnswerArtifact


def _question(qid: str) -> BQuestion:
    return BQuestion(
        qid=qid,
        domain="financial_reports",
        split="B",
        question=f"{qid} question",
        options={},
        answer_format="calculation",
        type="计算题",
        answer_slots=1,
        answer_slot_templates=("999999.99",),
    )


def _artifact(qid: str, answer: str) -> BAnswerArtifact:
    return BAnswerArtifact(
        qid=qid,
        domain="financial_reports",
        answer_format="calculation",
        answer_slot_count=1,
        answer_parts=[answer],
        used_evidence_ids=[f"e:{qid}"],
        evidence_items=[{"unit_id": f"e:{qid}", "text": answer}],
        decision_summary=f"{qid} 的证据和计算支持最终答案为 {answer}。",
        decision_trace={},
        calculation_trace={},
        token_usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        locator={},
    )


class BBoardCandidateOverlayTests(unittest.TestCase):
    def test_overlay_replaces_only_declared_qids_and_preserves_question_order(
        self,
    ) -> None:
        questions = [_question("q1"), _question("q2"), _question("q3")]
        base = [_artifact("q1", "old1"), _artifact("q2", "old2"), _artifact("q3", "old3")]
        patch = [_artifact("q2", "new2")]

        merged, overlaid_qids = merge_candidate_artifacts(
            questions=questions,
            base_artifacts=base,
            overlay_artifact_groups=[patch],
        )

        self.assertEqual([item.qid for item in merged], ["q1", "q2", "q3"])
        self.assertEqual(
            [item.answer_parts for item in merged],
            [["old1"], ["new2"], ["old3"]],
        )
        self.assertEqual(overlaid_qids, ["q2"])

    def test_overlay_rejects_duplicate_patch_ownership(self) -> None:
        questions = [_question("q1"), _question("q2")]

        with self.assertRaisesRegex(ValueError, "duplicate overlay qid q2"):
            merge_candidate_artifacts(
                questions=questions,
                base_artifacts=[_artifact("q1", "old1"), _artifact("q2", "old2")],
                overlay_artifact_groups=[
                    [_artifact("q2", "new2-a")],
                    [_artifact("q2", "new2-b")],
                ],
            )

    def test_overlay_requires_complete_base_coverage(self) -> None:
        questions = [_question("q1"), _question("q2")]

        with self.assertRaisesRegex(ValueError, "base artifact coverage"):
            merge_candidate_artifacts(
                questions=questions,
                base_artifacts=[_artifact("q1", "old1")],
                overlay_artifact_groups=[],
            )


if __name__ == "__main__":
    unittest.main()
