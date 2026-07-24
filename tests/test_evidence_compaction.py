from __future__ import annotations

import hashlib
import inspect
import re
import unittest

import afa_agent.b_board.evidence_compaction as evidence_compaction_module
from afa_agent.b_board.evidence_compaction import (
    compact_adjacent_evidence_hits,
    compact_retrieval_payload,
)
from afa_agent.b_board.evidence_rank_diagnostics import diagnose_historical_evidence_ranks
from afa_agent.models import RetrievalHit


def make_unit(
    unit_id: str,
    *,
    doc_id: str = "doc-a",
    title_path: list[str] | None = None,
    text: str,
    unit_type: str = "paragraph",
    parent_unit_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "doc_id": doc_id,
        "domain": "test",
        "unit_type": unit_type,
        "title_path": title_path or [doc_id, "同一章节"],
        "text": text,
        "page_refs": [],
        "parent_unit_id": parent_unit_id,
        "metadata": metadata or {},
    }


def make_hit(unit: dict[str, object], score: float) -> RetrievalHit:
    metadata = dict(unit.get("metadata", {}))
    metadata["unit_type"] = unit["unit_type"]
    return RetrievalHit(
        unit_id=str(unit["unit_id"]),
        doc_id=str(unit["doc_id"]),
        score=score,
        title_path=list(unit["title_path"]),
        text=str(unit["text"]),
        metadata=metadata,
    )


class AdjacentEvidenceCompactionTests(unittest.TestCase):
    def test_runtime_interface_has_no_question_or_historical_label_mapping(self) -> None:
        parameters = set(inspect.signature(compact_adjacent_evidence_hits).parameters)
        source = inspect.getsource(evidence_compaction_module)

        self.assertEqual(
            parameters,
            {
                "hits",
                "units",
                "top_k",
                "max_block_chars",
                "min_overlap_chars",
                "max_components_per_block",
            },
        )
        self.assertTrue(
            {
                "qid",
                "question_id",
                "expected_answer",
                "historical_evidence",
                "rule_label",
                "targeted_literal",
            }.isdisjoint(parameters)
        )
        self.assertIsNone(re.search(r"(?:fc|fin|ins|reg|res)_[ab]_\d{3}", source))

    def test_same_parent_chunks_merge_in_source_order_and_remove_exact_overlap(self) -> None:
        repeated = "经营活动产生的现金流量净额为100亿元。"
        units = [
            make_unit(
                "doc-a::sec-1::chunk-0",
                text=f"2025年营业收入为500亿元。{repeated}",
                parent_unit_id="doc-a::sec-1",
                metadata={"chunk_index": 0},
            ),
            make_unit(
                "doc-a::sec-1::chunk-1",
                text=f"{repeated}经营现金流率按二者相除计算。",
                parent_unit_id="doc-a::sec-1",
                metadata={"chunk_index": 1},
            ),
            make_unit("doc-a::sec-2", title_path=["doc-a", "其他章节"], text="不相关内容。"),
        ]
        hits = [make_hit(units[1], 9.0), make_hit(units[0], 8.5), make_hit(units[2], 8.0)]

        blocks = compact_adjacent_evidence_hits(hits, units=units, top_k=3)

        self.assertEqual(len(blocks), 2)
        merged = blocks[0]
        self.assertEqual(
            merged["source_order"],
            ["doc-a::sec-1::chunk-0", "doc-a::sec-1::chunk-1"],
        )
        self.assertEqual(merged["text"].count(repeated), 1)
        self.assertEqual(merged["overlap_chars"], len(repeated))
        self.assertEqual(
            [item["unit_id"] for item in merged["merged_from"]],
            merged["source_order"],
        )
        self.assertEqual(
            [item["sha256"] for item in merged["component_hashes"]],
            [
                hashlib.sha256(str(units[0]["text"]).encode("utf-8")).hexdigest(),
                hashlib.sha256(str(units[1]["text"]).encode("utf-8")).hexdigest(),
            ],
        )
        self.assertEqual(
            merged["text_sha256"],
            hashlib.sha256(merged["text"].encode("utf-8")).hexdigest(),
        )

    def test_header_and_table_row_merge_into_one_topk_slot(self) -> None:
        units = [
            make_unit(
                "doc-a::table-heading",
                title_path=["doc-a", "主要会计数据"],
                text="单位：亿元",
                unit_type="table_header",
            ),
            make_unit(
                "doc-a::metric-row",
                title_path=["doc-a", "主要会计数据", "营业收入"],
                text="| 项目 | 2025年 | 2024年 |\n| 营业收入 | 500 | 450 |",
                unit_type="metric_row",
            ),
            make_unit(
                "doc-b::standalone",
                doc_id="doc-b",
                title_path=["doc-b", "另一章节"],
                text="另一个高分命中。",
            ),
        ]
        hits = [make_hit(units[0], 10.0), make_hit(units[2], 9.5), make_hit(units[1], 9.0)]

        blocks = compact_adjacent_evidence_hits(hits, units=units, top_k=2)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["source_order"], ["doc-a::table-heading", "doc-a::metric-row"])
        self.assertIn("单位：亿元", blocks[0]["text"])
        self.assertIn("营业收入", blocks[0]["text"])
        self.assertEqual(blocks[1]["source_order"], ["doc-b::standalone"])

    def test_condition_clause_and_definition_exception_are_supported(self) -> None:
        units = [
            make_unit(
                "doc-a::condition",
                title_path=["doc-a", "身故保险金"],
                text="被保险人身故时，身故保险金按下列约定给付：",
                unit_type="clause_heading",
            ),
            make_unit(
                "doc-a::clause",
                title_path=["doc-a", "身故保险金", "给付条件"],
                text="取基本保险金额乘给付比例与个人账户价值的较大者。",
                unit_type="clause_block",
            ),
            make_unit(
                "doc-a::definition",
                title_path=["doc-a", "释义"],
                text="本合同所称意外伤害，是指外来的、突发的事件。",
                unit_type="definition",
            ),
            make_unit(
                "doc-a::exception",
                title_path=["doc-a", "释义", "除外情形"],
                text="但疾病导致的身体伤害不属于意外伤害。",
                unit_type="exception",
            ),
        ]
        hits = [make_hit(unit, 10.0 - index) for index, unit in enumerate(units)]

        blocks = compact_adjacent_evidence_hits(hits, units=units, top_k=4)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["source_order"], ["doc-a::condition", "doc-a::clause"])
        self.assertEqual(blocks[1]["source_order"], ["doc-a::definition", "doc-a::exception"])

    def test_unrelated_sibling_sections_and_different_documents_do_not_merge(self) -> None:
        units = [
            make_unit("doc-a::sec-1", title_path=["doc-a", "营业收入"], text="营业收入数据。"),
            make_unit("doc-a::sec-2", title_path=["doc-a", "员工情况"], text="员工人数数据。"),
            make_unit(
                "doc-b::sec-1",
                doc_id="doc-b",
                title_path=["doc-b", "员工情况"],
                text="另一文档的员工数据。",
            ),
        ]
        hits = [make_hit(unit, 10.0 - index) for index, unit in enumerate(units)]

        blocks = compact_adjacent_evidence_hits(hits, units=units, top_k=3)

        self.assertEqual(len(blocks), 3)
        self.assertEqual([block["source_order"] for block in blocks], [[unit["unit_id"]] for unit in units])
        self.assertEqual(blocks[0]["metadata"]["unit_type"], "paragraph")

    def test_exact_overlap_can_prove_compatibility_across_metric_sibling_titles(self) -> None:
        shared_row = "营业收入 | 500.00 | 450.00 | 11.11%"
        units = [
            make_unit(
                "doc-a::metric-revenue",
                title_path=["doc-a", "营业收入"],
                text=f"2025年 | 2024年 | 同比\n{shared_row}",
                unit_type="metric_row",
            ),
            make_unit(
                "doc-a::metric-profit",
                title_path=["doc-a", "归母净利润"],
                text=f"{shared_row}\n归母净利润 | 50.00 | 40.00 | 25.00%",
                unit_type="metric_row",
            ),
        ]

        blocks = compact_adjacent_evidence_hits(
            [make_hit(unit, 10.0 - index) for index, unit in enumerate(units)],
            units=units,
            top_k=2,
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"].count(shared_row), 1)
        self.assertEqual(blocks[0]["overlap_chars"], len(shared_row))

    def test_truncation_preserves_component_and_output_provenance(self) -> None:
        units = [
            make_unit(
                "doc-a::chunk-0",
                text="A" * 18,
                parent_unit_id="doc-a::parent",
                metadata={"chunk_index": 0},
            ),
            make_unit(
                "doc-a::chunk-1",
                text="B" * 18,
                parent_unit_id="doc-a::parent",
                metadata={"chunk_index": 1, "truncation_provenance": {"upstream": True}},
            ),
        ]

        blocks = compact_adjacent_evidence_hits(
            [make_hit(unit, 10.0 - index) for index, unit in enumerate(units)],
            units=units,
            top_k=1,
            max_block_chars=25,
        )

        self.assertEqual(len(blocks[0]["text"]), 25)
        self.assertEqual(
            blocks[0]["truncation_provenance"],
            {
                "applied": True,
                "strategy": "prefix_after_overlap_deduplication",
                "max_chars": 25,
                "original_chars": 37,
                "retained_chars": 25,
                "removed_chars": 12,
            },
        )
        self.assertEqual(
            blocks[0]["merged_from"][1]["input_truncation"],
            {"upstream": True},
        )

    def test_retrieval_payload_compaction_replaces_final_with_audited_blocks(self) -> None:
        units = [
            make_unit(
                "doc-a::chunk-0",
                text="标题：经营活动现金流量",
                parent_unit_id="doc-a::parent",
                metadata={"chunk_index": 0},
            ),
            make_unit(
                "doc-a::chunk-1",
                text="经营活动现金流量为100亿元。",
                parent_unit_id="doc-a::parent",
                metadata={"chunk_index": 1},
            ),
        ]
        retrieval = {
            "policy_version": "test",
            "final": {
                "queries": ["经营活动现金流量"],
                "hits": [
                    make_hit(units[0], 10).to_dict(),
                    make_hit(units[1], 9).to_dict(),
                ],
            },
        }

        compacted = compact_retrieval_payload(
            retrieval,
            units=units,
            top_k=1,
        )

        self.assertEqual(len(compacted["final"]["hits"]), 1)
        self.assertEqual(
            compacted["final"]["ranked_hits"][0]["unit_id"],
            compacted["final"]["hits"][0]["unit_id"],
        )
        self.assertEqual(compacted["compaction"]["input_hit_count"], 2)
        self.assertEqual(compacted["compaction"]["output_block_count"], 1)
        self.assertEqual(compacted["compaction"]["merged_block_count"], 1)
        self.assertEqual(len(compacted["compaction"]["input_hits"]), 2)
        self.assertTrue(
            all(item["text_sha256"] for item in compacted["compaction"]["input_hits"])
        )

    def test_transitive_chunk_chain_is_split_around_best_rank(self) -> None:
        units = [
            make_unit(
                f"doc-a::chunk-{index}",
                text=f"第{index}段" + ("甲" * 20),
                parent_unit_id="doc-a::parent",
                metadata={"chunk_index": index},
            )
            for index in range(4)
        ]
        hits = [
            make_hit(units[2], 10.0),
            make_hit(units[1], 9.0),
            make_hit(units[3], 8.0),
            make_hit(units[0], 7.0),
        ]

        blocks = compact_adjacent_evidence_hits(
            hits,
            units=units,
            top_k=4,
            max_components_per_block=2,
        )

        self.assertEqual(len(blocks), 3)
        self.assertTrue(
            all(len(block["merged_from"]) <= 2 for block in blocks)
        )
        self.assertEqual(
            blocks[0]["source_order"],
            ["doc-a::chunk-1", "doc-a::chunk-2"],
        )
        self.assertEqual(
            sorted(
                item["unit_id"]
                for block in blocks
                for item in block["merged_from"]
            ),
            sorted(str(unit["unit_id"]) for unit in units),
        )


class HistoricalEvidenceRankDiagnosticTests(unittest.TestCase):
    def test_metrics_filter_contaminated_positives_and_do_not_export_target_ids(self) -> None:
        ranked_first = [
            {"unit_id": f"doc-a::u-{index}", "doc_id": "doc-a"}
            for index in range(1, 13)
        ]
        cases = [
            {
                "case_id": "case-one",
                "ranked_hits": ranked_first,
                "historical_evidence": [
                    {"unit_id": "doc-a::u-12", "doc_id": "doc-a"},
                    {
                        "unit_id": "doc-a::u-1",
                        "doc_id": "doc-a",
                        "metadata": {"targeted_literal": "隐藏答案字符串"},
                    },
                ],
            },
            {
                "case_id": "case-two",
                "ranked_hits": [{"unit_id": "doc-b::u-1", "doc_id": "doc-b"}],
                "historical_evidence": [
                    {"unit_id": "doc-b::u-1", "doc_id": "doc-b"},
                    {
                        "unit_id": "doc-b::u-2",
                        "doc_id": "doc-b",
                        "metadata": {"rule_label": False},
                    },
                ],
            },
            {
                "case_id": "contaminated-only",
                "ranked_hits": [{"unit_id": "doc-c::u-1", "doc_id": "doc-c"}],
                "historical_evidence": [
                    {
                        "unit_id": "doc-c::u-1",
                        "doc_id": "doc-c",
                        "metadata": {"rule_label": True},
                    }
                ],
            },
        ]

        report = diagnose_historical_evidence_ranks(cases, k=10)

        self.assertTrue(report["offline_only"])
        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["evaluated_case_count"], 2)
        self.assertEqual(report["filtered_positive_count"], 3)
        self.assertAlmostEqual(report["mrr"], (1 / 12 + 1) / 2)
        self.assertEqual(report["recall_at_10"], 0.5)
        self.assertEqual(report["positive_recall_at_10"], 0.5)
        self.assertEqual(report["cases"][0]["gap"], "ranking_gap")
        self.assertEqual(report["cases"][0]["best_rank"], 12)
        self.assertEqual(report["cases"][2]["gap"], "no_eligible_positive")
        report_text = repr(report)
        self.assertNotIn("doc-a::u-12", report_text)
        self.assertNotIn("doc-b::u-1", report_text)
        self.assertNotIn("隐藏答案字符串", report_text)

    def test_missing_positive_distinguishes_document_and_chunk_recall_gaps(self) -> None:
        cases = [
            {
                "case_id": "same-doc",
                "ranked_hits": [{"unit_id": "doc-a::other", "doc_id": "doc-a"}],
                "historical_evidence": [{"unit_id": "doc-a::target", "doc_id": "doc-a"}],
            },
            {
                "case_id": "missing-doc",
                "ranked_hits": [{"unit_id": "doc-x::other", "doc_id": "doc-x"}],
                "historical_evidence": [{"unit_id": "doc-b::target", "doc_id": "doc-b"}],
            },
        ]

        report = diagnose_historical_evidence_ranks(cases, k=10)

        self.assertEqual(
            [case["gap"] for case in report["cases"]],
            ["chunk_recall_gap", "document_recall_gap"],
        )
        self.assertEqual(report["mrr"], 0.0)
        self.assertEqual(report["recall_at_10"], 0.0)

    def test_compacted_block_counts_its_component_as_a_hit_at_block_rank(self) -> None:
        cases = [
            {
                "case_id": "compacted",
                "ranked_hits": [
                    {
                        "unit_id": "merged::doc-a::digest",
                        "doc_id": "doc-a",
                        "merged_from": [
                            {"unit_id": "doc-a::header", "doc_id": "doc-a"},
                            {"unit_id": "doc-a::target", "doc_id": "doc-a"},
                        ],
                    }
                ],
                "historical_evidence": [
                    {"unit_id": "doc-a::target", "doc_id": "doc-a"},
                ],
            }
        ]

        report = diagnose_historical_evidence_ranks(cases, k=10)

        self.assertEqual(report["cases"][0]["best_rank"], 1)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["recall_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
