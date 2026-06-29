#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.io_utils import write_json
from afa_agent.models import Question
from afa_agent.text_utils import tokenize_zh


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]


DOMAIN_RETRIEVAL = {
    "financial_reports": {"top_k": 6, "boosts": {"metric_row": 1.8, "paragraph": 1.0}, "mode": "per_option"},
    "insurance": {"top_k": 4, "boosts": {"formula_block": 1.8, "clause_block": 1.1}, "mode": "whole_question"},
    "research": {"top_k": 7, "boosts": {"conclusion_block": 1.6, "paragraph": 1.0}, "mode": "per_option"},
    "financial_contracts": {"top_k": 7, "boosts": {"element_block": 1.8, "paragraph": 1.0}, "mode": "per_option"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(domain: str, split: str) -> list[Question]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    rows = read_json(Path(manifest["domains"][domain]["question_path"]))
    return [
        Question(
            qid=row["qid"],
            domain=row["domain"],
            split=row["split"],
            question=row["question"],
            options=row["options"],
            answer_format=row["answer_format"],
            type=row["type"],
            doc_ids=row.get("doc_ids", []),
        )
        for row in rows
        if row.get("split") == split
    ]


def make_retriever(domain: str, units: list[dict[str, Any]]):
    if domain == "regulatory":
        return RegulatoryRetriever(units)
    return GenericBM25Retriever(units)


def hit_payload(hit) -> dict[str, Any]:
    payload = hit.to_dict()
    payload["unit_type"] = payload.get("metadata", {}).get("unit_type") or payload.get("unit_type", "")
    return payload


def search_rows(domain: str, retriever, question: Question) -> list[dict[str, Any]]:
    if domain == "regulatory":
        rows = []
        ensure_per_doc = len(question.doc_ids) > 1
        for option, option_text in question.options.items():
            query = f"{question.question}\n{option_text}"
            hits = retriever.search(question.doc_ids, query, top_k=6, ensure_per_doc=ensure_per_doc)
            rows.append({"option": option, "query": query, "hits": [hit_payload(hit) for hit in hits]})
        return rows
    cfg = DOMAIN_RETRIEVAL[domain]
    ensure_per_doc = len(question.doc_ids) > 1
    if cfg["mode"] == "whole_question":
        query = f"{question.question}\n{json.dumps(question.options, ensure_ascii=False)}"
        hits = retriever.search(
            question.doc_ids,
            query,
            top_k=cfg["top_k"],
            unit_type_boosts=cfg["boosts"],
            ensure_per_doc=ensure_per_doc,
        )
        return [{"option": "", "query": query, "hits": [hit_payload(hit) for hit in hits]}]
    rows = []
    for option, option_text in question.options.items():
        query = f"{question.question}\n{option_text}"
        hits = retriever.search(
            question.doc_ids,
            query,
            top_k=cfg["top_k"],
            unit_type_boosts=cfg["boosts"],
            ensure_per_doc=ensure_per_doc,
        )
        rows.append({"option": option, "query": query, "hits": [hit_payload(hit) for hit in hits]})
    return rows


def audit_domain(domain: str, index_root: Path, split: str, limit: int) -> dict[str, Any]:
    index_payload = read_json(index_root / domain / "index.json")
    units = index_payload["units"]
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    retriever = make_retriever(domain, units)
    questions = load_questions(domain, split)
    if limit:
        questions = questions[:limit]
    total_rows = 0
    doc_hit_rows = 0
    top1_doc_hit_rows = 0
    no_hit_rows = []
    unit_type_counts: Counter[str] = Counter()
    query_token_counts = []
    multi_doc_coverages = []
    per_question = []
    traces = {}
    for question in questions:
        rows = search_rows(domain, retriever, question)
        traces[question.qid] = {"qid": question.qid, "rows": rows}
        q_doc_hit = 0
        q_hit_docs = set()
        for row in rows:
            total_rows += 1
            query_token_counts.append(len(tokenize_zh(row["query"])))
            hits = row["hits"]
            if not hits:
                no_hit_rows.append({"qid": question.qid, "option": row["option"]})
                continue
            hit_docs = {hit.get("doc_id") for hit in hits if hit.get("doc_id")}
            q_hit_docs.update(hit_docs)
            if set(question.doc_ids) & hit_docs:
                doc_hit_rows += 1
                q_doc_hit += 1
            if hits[0].get("doc_id") in set(question.doc_ids):
                top1_doc_hit_rows += 1
            for hit in hits:
                unit = unit_by_id.get(hit.get("unit_id"), {})
                unit_type_counts[unit.get("unit_type", hit.get("unit_type", ""))] += 1
        if len(question.doc_ids) > 1:
            multi_doc_coverages.append(len(q_hit_docs & set(question.doc_ids)) / len(set(question.doc_ids)))
        per_question.append(
            {
                "qid": question.qid,
                "answer_format": question.answer_format,
                "row_count": len(rows),
                "doc_hit_rate": round(q_doc_hit / max(1, len(rows)), 4),
                "question_docs": question.doc_ids,
                "retrieved_question_docs": sorted(q_hit_docs & set(question.doc_ids)),
            }
        )
    return {
        "domain": domain,
        "question_count": len(questions),
        "unit_count": len(units),
        "trace_rows": total_rows,
        "doc_hit_rate": round(doc_hit_rows / max(1, total_rows), 4),
        "top1_doc_hit_rate": round(top1_doc_hit_rows / max(1, total_rows), 4),
        "multi_doc_coverage_avg": round(statistics.mean(multi_doc_coverages), 4) if multi_doc_coverages else 1.0,
        "query_tokens_avg": round(statistics.mean(query_token_counts), 2) if query_token_counts else 0.0,
        "hit_unit_type_counts": dict(unit_type_counts.most_common()),
        "no_hit_rows": no_hit_rows[:100],
        "weak_questions": [row for row in per_question if row["doc_hit_rate"] < 1.0],
        "per_question": per_question,
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--index-root", default=str(ROOT / "artifacts" / "index"))
    parser.add_argument("--split", default="A")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "retrieval_loop_audit" / "latest"))
    args = parser.parse_args()

    domains = DOMAINS if args.domains == ["all"] else args.domains
    index_root = Path(args.index_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"index_root": str(index_root), "domains": {}}
    for domain in domains:
        result = audit_domain(domain, index_root, args.split, args.limit)
        results["domains"][domain] = {key: value for key, value in result.items() if key != "traces"}
        write_json(output_dir / f"{domain}_traces.json", result["traces"])
    write_json(output_dir / "summary.json", results)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
