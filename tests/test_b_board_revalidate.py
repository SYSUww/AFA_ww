from __future__ import annotations

import unittest

from afa_agent.b_board.calculation import CalculationPlanError
from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.revalidate import revalidate_calculation_artifact
from afa_agent.b_board.runner import BAnswerArtifact


class BBoardCalculationRevalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = BQuestion(
            qid="q1",
            domain="research",
            split="B",
            question="全年需求同比增速最接近多少？",
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("999999.99",),
        )
        self.artifact = BAnswerArtifact(
            qid="q1",
            domain="research",
            answer_format="calculation",
            answer_slot_count=1,
            answer_parts=["22.27"],
            used_evidence_ids=["e1"],
            evidence_items=[
                {
                    "unit_id": "e1",
                    "doc_id": "report",
                    "title_path": [],
                    "text": "同比增速为22.27%",
                    "score": 1.0,
                    "metadata": {},
                }
            ],
            decision_summary="incumbent",
            decision_trace={},
            calculation_trace={
                "schema_version": 2,
                "variables": [
                    {
                        "name": "growth",
                        "value": "22.27",
                        "value_type": "decimal",
                        "unit": "%",
                        "evidence_ids": ["e1"],
                    }
                ],
                "steps": [],
                "outputs": [
                    {
                        "source": {"ref": "growth"},
                        "format": "decimal2",
                        "value": "22.27",
                    }
                ],
            },
            token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            locator={},
        )

    def test_revalidation_rejects_contract_change_by_default(self) -> None:
        with self.assertRaisesRegex(CalculationPlanError, "changed the incumbent answer"):
            revalidate_calculation_artifact(
                question=self.question,
                artifact=self.artifact,
                index_units=[],
            )

    def test_revalidation_allows_only_guarded_format_migration(self) -> None:
        result = revalidate_calculation_artifact(
            question=self.question,
            artifact=self.artifact,
            index_units=[],
            allow_format_change=True,
        )

        self.assertEqual(result.answer_parts, ["22.27%"])
        self.assertFalse(result.decision_trace["format_forced"])
        self.assertTrue(result.decision_trace["format_migrated"])
        self.assertTrue(result.decision_trace["format_change_allowed"])
        self.assertEqual(result.token_usage["total_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
