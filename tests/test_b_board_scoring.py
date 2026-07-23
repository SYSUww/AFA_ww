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
            500_000: 90.0,
            931_605: 81.3679,
            5_000_000: 0.0,
            5_000_001: 99.99998,
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
        self.assertAlmostEqual(score.token_efficiency_score, 81.3679)
        self.assertAlmostEqual(score.total_score, 90.27358)

    def test_empty_reasoning_scores_are_zero(self) -> None:
        score = score_submission(
            accuracy_score=94,
            reasoning_scores=[],
            token_total=931_605,
        )
        self.assertAlmostEqual(score.total_score, 63.27358)

    def test_official_score_regression_anchors(self) -> None:
        expected = {
            98: 65.2736,
            96: 64.2736,
            97: 64.7736,
        }
        for accuracy, official_total in expected.items():
            with self.subTest(accuracy=accuracy):
                score = score_submission(
                    accuracy_score=accuracy,
                    reasoning_scores=[],
                    token_total=931_605,
                )
                self.assertEqual(round(score.total_score, 4), official_total)

    def test_only_qwen35_qwen36_and_qwen37_families_are_allowed(self) -> None:
        for model in (
            "qwen3.5-plus",
            "Qwen3.6",
            "dashscope/qwen-3_5-turbo",
            "qwen3.7-plus",
            "qwen3.7-plus-2026-05-26",
            "dashscope:qwen-3_7-max",
        ):
            with self.subTest(model=model):
                self.assertTrue(is_allowed_submission_model(model))
                require_allowed_submission_model(model)
        for model in (
            "qwen3-plus",
            "qwen2.5-max",
            "qwen3.8-plus",
            "qwen3.70-plus",
            "gpt-5.6",
            "deepseek-r1",
            "",
        ):
            with self.subTest(model=model):
                self.assertFalse(is_allowed_submission_model(model))
                with self.assertRaises(ValueError):
                    require_allowed_submission_model(model)


if __name__ == "__main__":
    unittest.main()
