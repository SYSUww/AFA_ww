from __future__ import annotations

import unittest

from scripts.run_b_board_migration_loop import AttemptConfig, select_answer_doc_ids


def make_attempt(**overrides: object) -> AttemptConfig:
    values = {
        "attempt_id": "test_attempt",
        "round_id": "test_round",
        "priority": "P0",
        "direction": "doc_selection",
        "variant_name": "alias_pruned",
        "hypothesis": "test",
        "answer_top_k": 3,
        "answer_doc_policy": "alias_pruned",
    }
    values.update(overrides)
    return AttemptConfig(**values)


class SelectAnswerDocIdsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate_row = {
            "candidate_doc_ids": ["ranked_1", "ranked_2", "generic_alias", "ranked_4"],
            "candidates": [
                {"doc_id": "ranked_1", "locator_reason": "bm25_profile_match"},
                {"doc_id": "ranked_2", "locator_reason": "entity=2025"},
                {"doc_id": "generic_alias", "locator_reason": "alias=深度报告"},
                {"doc_id": "ranked_4", "locator_reason": "bm25_profile_match"},
            ],
        }

    def test_research_keeps_locator_ranking_instead_of_generic_alias(self) -> None:
        selected = select_answer_doc_ids(self.candidate_row, {"domain": "research"}, make_attempt())
        self.assertEqual(selected, ["ranked_1", "ranked_2", "generic_alias"])

    def test_regulatory_keeps_locator_ranking_instead_of_generic_alias(self) -> None:
        selected = select_answer_doc_ids(self.candidate_row, {"domain": "regulatory"}, make_attempt())
        self.assertEqual(selected, ["ranked_1", "ranked_2", "generic_alias"])

    def test_insurance_still_uses_product_alias_shortlist(self) -> None:
        selected = select_answer_doc_ids(self.candidate_row, {"domain": "insurance"}, make_attempt())
        self.assertEqual(selected, ["generic_alias"])

    def test_expanded_topk_policy_is_unchanged(self) -> None:
        selected = select_answer_doc_ids(
            self.candidate_row,
            {"domain": "research"},
            make_attempt(answer_doc_policy="expanded_topk", answer_top_k=2),
        )
        self.assertEqual(selected, ["ranked_1", "ranked_2"])


if __name__ == "__main__":
    unittest.main()
