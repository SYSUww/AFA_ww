from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.evaluate_b_retrieval_decisive import (
    _index_catalog,
    document_metrics,
    evidence_pool_ids,
    full_pool_metrics,
    normalize_identifier,
    query_work,
    ranking_metrics,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "experiments/b_board_actual/proxy30_decisive_evidence_v1.json"
)


class RetrievalDecisiveEvaluatorTests(unittest.TestCase):
    def test_manifest_is_evaluator_only_and_answer_free(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

        validate_manifest(payload)

        self.assertEqual(len(payload["questions"]), 30)
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        for prohibited in (
            '"expected_answer"',
            '"proxy_answer"',
            '"official_answer"',
            '"answer_parts"',
        ):
            self.assertNotIn(prohibited, serialized)

    def test_manifest_is_not_referenced_by_production_source(self) -> None:
        forbidden_names = {
            MANIFEST.name,
            "b_board_proxy30_decisive_evidence_20260725",
            "evaluate_b_retrieval_decisive",
        }
        offenders: list[str] = []
        for path in (ROOT / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(name in text for name in forbidden_names):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])

    def test_normalize_identifier_preserves_semantic_source_kind(self) -> None:
        cases = (
            ("4::sec_143__dup2", "4::sec_143"),
            ("doc__dup2::第三十四条", "doc::第三十四条"),
            (
                "doc::supplemental::第三十四条",
                "doc::supplemental::第三十四条",
            ),
            (" １２::sec_１ ", "12::sec_1"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_identifier(raw), expected)

    def test_index_catalog_normalizes_ids_without_overwrite(self) -> None:
        payloads = {
            "regulatory": {
                "units": [
                    {"unit_id": "law::第三条", "doc_id": "law"},
                    {
                        "unit_id": "law__dup2::第三条",
                        "doc_id": "law__dup2",
                    },
                ]
            }
        }

        unit_ids, unit_docs = _index_catalog(payloads)

        self.assertEqual(unit_ids["regulatory"], {"law::第三条"})
        self.assertEqual(
            unit_docs["regulatory"]["law::第三条"],
            {"law"},
        )

    def test_fact_metrics_use_groups_not_flat_duplicate_ids(self) -> None:
        groups = [
            {
                "group_id": "revenue",
                "requirement": "any_of",
                "acceptable_unit_ids": [
                    "report::metric_1",
                    "report::metric_1__dup2",
                ],
            },
            {
                "group_id": "cashflow_series",
                "requirement": "all_of",
                "acceptable_unit_ids": [
                    "report::metric_2",
                    "report::metric_3",
                ],
            },
        ]

        partial = ranking_metrics(
            groups,
            ["noise", "report::metric_1__dup2", "report::metric_2"],
            reachable_group_ids={"revenue", "cashflow_series"},
        )
        complete = ranking_metrics(
            groups,
            [
                "report::metric_1",
                "report::metric_2",
                "report::metric_3",
            ],
            reachable_group_ids={"revenue", "cashflow_series"},
        )

        self.assertEqual(partial["any_recall_at_5"], 1.0)
        self.assertEqual(partial["fact_coverage_at_10"], 0.5)
        self.assertEqual(partial["fact_complete_at_10"], 0.0)
        self.assertAlmostEqual(partial["mrr_at_10"], 0.5)
        self.assertEqual(complete["fact_coverage_at_10"], 1.0)
        self.assertEqual(complete["fact_complete_at_10"], 1.0)

    def test_document_metrics_keep_locator_miss_in_denominator(self) -> None:
        groups = [
            {
                "group_id": "product_a",
                "requirement": "any_of",
                "acceptable_doc_ids": ["13"],
            },
            {
                "group_id": "product_b",
                "requirement": "any_of",
                "acceptable_doc_ids": ["14"],
            },
        ]

        self.assertEqual(
            document_metrics(groups, ["13"]),
            {
                "doc_any_recall": 1.0,
                "doc_coverage": 0.5,
                "doc_complete": 0.0,
            },
        )

    def test_evidence_pool_excludes_document_discovery(self) -> None:
        retrieval = {
            "document_discovery": {"ranked_ids": ["discovery_only"]},
            "primary": {"ranked_ids": ["primary"]},
            "supplemental": {"ranked_ids": ["supplemental", "primary"]},
            "document_rankings": [{"ranked_ids": ["document"]}],
            "metric_slot_rankings": [{"ranked_ids": ["metric"]}],
            "option_rankings": [{"ranked_ids": ["option"]}],
        }

        self.assertEqual(
            evidence_pool_ids(retrieval),
            ["primary", "supplemental", "document", "metric", "option"],
        )

    def test_pool_coverage_uses_the_entire_pool(self) -> None:
        groups = [
            {
                "group_id": "late_fact",
                "requirement": "any_of",
                "acceptable_unit_ids": ["fact_at_rank_11"],
            }
        ]
        pool = [f"noise_{index}" for index in range(10)]
        pool.append("fact_at_rank_11")

        self.assertEqual(
            full_pool_metrics(groups, pool),
            {
                "any_recall": 1.0,
                "fact_coverage": 1.0,
                "fact_complete": 1.0,
            },
        )

    def test_query_work_counts_repeated_search_executions(self) -> None:
        retrieval = {
            "bundle": {"anchor_queries": ["产品 条款"]},
            "document_discovery": {"queries": ["产品 条款"]},
            "primary": {"queries": ["条款 条件"]},
            "supplemental": {"queries": []},
            "document_rankings": [
                {"queries": ["责任免除"]},
                {"queries": ["责任免除"]},
            ],
            "metric_slot_rankings": [],
            "option_rankings": [],
        }

        metrics = query_work(retrieval)

        self.assertEqual(metrics["query_execution_count"], 5)
        self.assertEqual(metrics["unique_query_count"], 3)
        self.assertGreaterEqual(
            metrics["retrieval_term_count"],
            metrics["unique_retrieval_term_count"],
        )

    def test_manifest_validator_rejects_answer_keys(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["questions"][0]["expected_answer"] = "forbidden"

        with self.assertRaisesRegex(ValueError, "prohibited answer keys"):
            validate_manifest(payload)


if __name__ == "__main__":
    unittest.main()
