from __future__ import annotations

import unittest

from afa_agent.b_board.total_score_promotion import (
    PromotionSnapshot,
    decide_total_score_promotion,
)


class TotalScorePromotionTests(unittest.TestCase):
    def test_promotes_when_weighted_total_improves_despite_component_regression(self) -> None:
        baseline = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=70.0,
            token_total=1_067_222,
            retry_count=72,
        )
        candidate = PromotionSnapshot(
            accuracy_score=73.0,
            reasoning_score=78.0,
            token_total=900_000,
            retry_count=20,
        )

        decision = decide_total_score_promotion(
            baseline=baseline,
            candidate=candidate,
        )

        self.assertTrue(decision.promote)
        self.assertGreater(decision.total_score_delta, 0)
        self.assertLess(decision.component_deltas["accuracy_score"], 0)

    def test_rejects_any_weighted_total_regression(self) -> None:
        baseline = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=80.0,
            token_total=900_000,
        )
        candidate = PromotionSnapshot(
            accuracy_score=75.0,
            reasoning_score=75.0,
            token_total=1_100_000,
        )

        decision = decide_total_score_promotion(
            baseline=baseline,
            candidate=candidate,
        )

        self.assertFalse(decision.promote)
        self.assertIn("weighted_total_regressed", decision.reasons)

    def test_equal_total_requires_an_operational_tie_breaker(self) -> None:
        baseline = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=80.0,
            token_total=900_000,
            retry_count=4,
            failure_count=0,
            audit_issue_count=0,
            stability_score=90.0,
        )
        unchanged = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=80.0,
            token_total=900_000,
            retry_count=4,
            failure_count=0,
            audit_issue_count=0,
            stability_score=90.0,
        )
        safer = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=80.0,
            token_total=900_000,
            retry_count=3,
            failure_count=0,
            audit_issue_count=0,
            stability_score=90.0,
        )

        unchanged_decision = decide_total_score_promotion(
            baseline=baseline,
            candidate=unchanged,
        )
        safer_decision = decide_total_score_promotion(
            baseline=baseline,
            candidate=safer,
        )

        self.assertFalse(unchanged_decision.promote)
        self.assertIn("no_total_or_operational_gain", unchanged_decision.reasons)
        self.assertTrue(safer_decision.promote)
        self.assertIn("retry_count", safer_decision.tie_breaker_improvements)

    def test_blocks_incomplete_or_unobservable_candidate_even_if_score_improves(self) -> None:
        baseline = PromotionSnapshot(
            accuracy_score=74.0,
            reasoning_score=70.0,
            token_total=1_067_222,
        )
        candidate = PromotionSnapshot(
            accuracy_score=90.0,
            reasoning_score=90.0,
            token_total=500_001,
            complete=False,
            unobservable_usage_risk=True,
        )

        decision = decide_total_score_promotion(
            baseline=baseline,
            candidate=candidate,
        )

        self.assertFalse(decision.promote)
        self.assertIn("candidate_incomplete", decision.reasons)
        self.assertIn("unobservable_usage_risk", decision.reasons)

    def test_rejects_negative_operational_counts_and_non_finite_epsilon(self) -> None:
        with self.assertRaisesRegex(ValueError, "retry_count"):
            PromotionSnapshot(
                accuracy_score=80,
                reasoning_score=80,
                token_total=500_000,
                retry_count=-1,
            )
        baseline = PromotionSnapshot(80, 80, 500_000)
        candidate = PromotionSnapshot(80, 80, 500_000, retry_count=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            decide_total_score_promotion(
                baseline=baseline,
                candidate=candidate,
                epsilon=float("inf"),
            )

    def test_rejects_string_booleans_and_out_of_range_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete"):
            PromotionSnapshot(
                accuracy_score=80,
                reasoning_score=80,
                token_total=500_000,
                complete="true",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "accuracy_score"):
            PromotionSnapshot(
                accuracy_score=101,
                reasoning_score=80,
                token_total=500_000,
            )


if __name__ == "__main__":
    unittest.main()
