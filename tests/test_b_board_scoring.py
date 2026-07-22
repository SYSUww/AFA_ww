from __future__ import annotations

import unittest

from afa_agent.b_board.scoring import (
    is_allowed_submission_model,
    require_allowed_submission_model,
    score_submission,
    token_efficiency_score,
)


class BBoardOfficialScoringTests(unittest.TestCase):
    def test_token_efficiency_boundaries(self) -> None:
        cases = {
            0: 0.0,
            100: 0.02,
            499_999: 99.9998,
            500_000: 100.0,
            5_000_000: 100.0,
            7_500_000: 50.0,
            10_000_000: 0.0,
            10_000_001: 0.0,
        }
        for token_total, expected in cases.items():
            with self.subTest(token_total=token_total):
                self.assertAlmostEqual(token_efficiency_score(token_total), expected)

    def test_composite_score_uses_official_weights_and_reasoning_mean(self) -> None:
        score = score_submission(
            accuracy_score=94,
            reasoning_scores=[80, 100],
            token_total=931_605,
        )

        self.assertEqual(score.reasoning_score, 90)
        self.assertEqual(score.token_efficiency_score, 100)
        self.assertAlmostEqual(score.total_score, 94.4)

    def test_empty_reasoning_scores_are_zero(self) -> None:
        score = score_submission(
            accuracy_score=94,
            reasoning_scores=[],
            token_total=931_605,
        )
        self.assertAlmostEqual(score.total_score, 76.4)

    def test_only_qwen35_and_qwen36_families_are_allowed(self) -> None:
        for model in ("qwen3.5-plus", "Qwen3.6", "dashscope/qwen-3_5-turbo"):
            with self.subTest(model=model):
                self.assertTrue(is_allowed_submission_model(model))
                require_allowed_submission_model(model)
        for model in ("qwen3-plus", "qwen2.5-max", "gpt-5.6", ""):
            with self.subTest(model=model):
                self.assertFalse(is_allowed_submission_model(model))
                with self.assertRaises(ValueError):
                    require_allowed_submission_model(model)


if __name__ == "__main__":
    unittest.main()
