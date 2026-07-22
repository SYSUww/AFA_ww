from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from afa_agent.b_board.experiment_journal import (
    ExperimentJournal,
    StaleHistoryReviewError,
)
from afa_agent.experiment_registry import ExperimentRegistry


def candidate(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "direction_id": "reasoning_structure",
        "pipeline_stage": "reasoning",
        "root_cause_cluster": "reasoning",
        "hypothesis": "use a complete four-step reasoning structure",
        "change_vector": {"template": "locate-facts-derive-conclude"},
        "target_qids": ["q1"],
    }
    payload.update(overrides)
    return payload


def result(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment_id": "reasoning-structure-a1",
        "status": "rejected",
        "approach": "rewrite reasoning without changing answers",
        "effect": "reasoning mean unchanged",
        "failure_analysis": "clarity improved but completeness regressed",
        "next_step": "add explicit evidence-to-conclusion bridge",
        "metrics": {"reasoning_score_delta": 0.0, "answer_changes": 0},
    }
    payload.update(overrides)
    return payload


class ExperimentJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.log_path = self.root / "wiki" / "loop.md"
        self.log_path.parent.mkdir(parents=True)
        self.log_path.write_text(
            "## prior-reasoning\n\nq1 reasoning_structure was rejected\n",
            encoding="utf-8",
        )
        self.registry = ExperimentRegistry(self.root / "experiments" / "registry.jsonl")
        self.journal = ExperimentJournal(
            registry=self.registry,
            markdown_log_path=self.log_path,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_review_reads_log_and_registry_before_attempt(self) -> None:
        review = self.journal.review(candidate())

        self.assertTrue(review.executable)
        self.assertEqual(review.history_decision.comparable_attempt_count, 0)
        self.assertEqual(review.related_log_sections, ("prior-reasoning",))
        self.assertGreater(review.log_snapshot.size, 0)

    def test_result_requires_method_effect_failure_next_and_metrics(self) -> None:
        review = self.journal.review(candidate())

        with self.assertRaisesRegex(ValueError, "failure_analysis"):
            self.journal.append_result(
                review=review,
                candidate=candidate(),
                result={key: value for key, value in result().items() if key != "failure_analysis"},
            )

    def test_history_change_forces_a_fresh_review(self) -> None:
        review = self.journal.review(candidate())
        self.log_path.write_text(
            self.log_path.read_text(encoding="utf-8") + "\n## concurrent\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(StaleHistoryReviewError, "review"):
            self.journal.append_result(
                review=review,
                candidate=candidate(),
                result=result(),
            )

    def test_success_appends_same_structured_result_to_registry_and_log(self) -> None:
        review = self.journal.review(candidate())

        stored = self.journal.append_result(
            review=review,
            candidate=candidate(),
            result=result(),
        )

        self.assertEqual(stored["approach"], result()["approach"])
        self.assertIn("history_review", stored)
        self.assertEqual(self.registry.read_all(), [stored])
        log = self.log_path.read_text(encoding="utf-8")
        self.assertIn("reasoning-structure-a1", log)
        self.assertIn("reasoning_score_delta", log)

    def test_three_comparable_results_make_review_non_executable(self) -> None:
        base = candidate()
        for index in range(3):
            self.registry.append(
                {
                    **base,
                    "experiment_id": f"old-{index}",
                    "status": "rejected",
                    "hypothesis": f"reasoning structure v{index}",
                    "change_vector": {"template": f"v{index}"},
                }
            )

        review = self.journal.review(
            candidate(
                hypothesis="reasoning structure v4",
                change_vector={"template": "v4"},
                material_delta={"template": "v4"},
            )
        )

        self.assertEqual(review.history_decision.comparable_attempt_count, 3)
        self.assertFalse(review.executable)


if __name__ == "__main__":
    unittest.main()
