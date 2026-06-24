#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.financial_reports.solver import FinancialReportsSolver
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.io_utils import ensure_dir, read_json, write_json
from afa_agent.models import Question
from afa_agent.text_utils import tokenize_zh


def hit_with_token_debug(hit, query_tokens: list[str]) -> dict:
    payload = hit.to_dict()
    hit_tokens = set(tokenize_zh(hit.text))
    matched_tokens = [token for token in query_tokens if token in hit_tokens]
    payload["matched_tokens"] = matched_tokens[:80]
    return payload


def load_questions(domain: str, split: str) -> list[Question]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    rows = read_json(Path(manifest["domains"][domain]["question_path"]))
    questions = []
    for row in rows:
        if row.get("split") != split:
            continue
        questions.append(
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
        )
    return questions


def regulatory_trace(question: Question, index_payload: dict) -> dict:
    retriever = RegulatoryRetriever(index_payload["units"])
    option_traces = []
    for option_key, option_text in question.options.items():
        query = f"{question.question}\n{option_text}"
        query_tokens = tokenize_zh(query)
        hits = retriever.search(question.doc_ids, query, top_k=6)
        option_traces.append(
            {
                "option": option_key,
                "query": query,
                "query_tokens": query_tokens,
                "top_k": 6,
                "doc_ids": question.doc_ids,
                "hits": [hit_with_token_debug(hit, query_tokens) for hit in hits],
            }
        )
    return {"qid": question.qid, "trace_mode": "per_option", "option_traces": option_traces}


def financial_reports_trace(question: Question, index_payload: dict) -> dict:
    retriever = GenericBM25Retriever(index_payload["units"])
    solver = FinancialReportsSolver(client=None, retriever=retriever, units=index_payload["units"], strategy="financial_reports")
    option_traces = []
    for option_key, option_text in question.options.items():
        query = f"{question.question}\n{option_text}"
        query_tokens = tokenize_zh(query)
        hits = retriever.search(
            question.doc_ids,
            query,
            top_k=6,
            unit_type_boosts={"metric_row": 1.8, "paragraph": 1.0},
            ensure_per_doc=len(question.doc_ids) > 1,
        )
        metric_key = solver._detect_metric_key(option_text)
        rule_label = None
        rule_reason = None
        if metric_key:
            rule_label, rule_reason, _ = solver._rule_evaluate(question, option_text)
        option_traces.append(
            {
                "option": option_key,
                "query": query,
                "query_tokens": query_tokens,
                "top_k": 6,
                "unit_type_boosts": {"metric_row": 1.8, "paragraph": 1.0},
                "ensure_per_doc": len(question.doc_ids) > 1,
                "doc_ids": question.doc_ids,
                "metric_key": metric_key,
                "rule_label": rule_label,
                "rule_reason": rule_reason,
                "hits": [hit_with_token_debug(hit, query_tokens) for hit in hits],
            }
        )
    return {"qid": question.qid, "trace_mode": "per_option", "option_traces": option_traces}


def insurance_trace(question: Question, index_payload: dict) -> dict:
    retriever = GenericBM25Retriever(index_payload["units"])
    query = f"{question.question}\n{json.dumps(question.options, ensure_ascii=False)}"
    query_tokens = tokenize_zh(query)
    hits = retriever.search(
        question.doc_ids,
        query,
        top_k=4,
        unit_type_boosts={"formula_block": 1.8, "clause_block": 1.1},
        ensure_per_doc=len(question.doc_ids) > 1,
    )
    return {
        "qid": question.qid,
        "trace_mode": "whole_question",
        "query": query,
        "query_tokens": query_tokens,
        "top_k": 4,
        "unit_type_boosts": {"formula_block": 1.8, "clause_block": 1.1},
        "ensure_per_doc": len(question.doc_ids) > 1,
        "doc_ids": question.doc_ids,
        "hits": [hit_with_token_debug(hit, query_tokens) for hit in hits],
    }


def research_trace(question: Question, index_payload: dict) -> dict:
    retriever = GenericBM25Retriever(index_payload["units"])
    option_traces = []
    for option_key, option_text in question.options.items():
        query = f"{question.question}\n{option_text}"
        query_tokens = tokenize_zh(query)
        hits = retriever.search(
            question.doc_ids,
            query,
            top_k=7,
            unit_type_boosts={"conclusion_block": 1.6, "paragraph": 1.0},
            ensure_per_doc=len(question.doc_ids) > 1,
        )
        option_traces.append(
            {
                "option": option_key,
                "query": query,
                "query_tokens": query_tokens,
                "top_k": 7,
                "unit_type_boosts": {"conclusion_block": 1.6, "paragraph": 1.0},
                "ensure_per_doc": len(question.doc_ids) > 1,
                "doc_ids": question.doc_ids,
                "hits": [hit_with_token_debug(hit, query_tokens) for hit in hits],
            }
        )
    return {"qid": question.qid, "trace_mode": "per_option", "option_traces": option_traces}


def contracts_trace(question: Question, index_payload: dict) -> dict:
    retriever = GenericBM25Retriever(index_payload["units"])
    option_traces = []
    for option_key, option_text in question.options.items():
        query = f"{question.question}\n{option_text}"
        query_tokens = tokenize_zh(query)
        hits = retriever.search(
            question.doc_ids,
            query,
            top_k=7,
            unit_type_boosts={"element_block": 1.8, "paragraph": 1.0},
            ensure_per_doc=len(question.doc_ids) > 1,
        )
        option_traces.append(
            {
                "option": option_key,
                "query": query,
                "query_tokens": query_tokens,
                "top_k": 7,
                "unit_type_boosts": {"element_block": 1.8, "paragraph": 1.0},
                "ensure_per_doc": len(question.doc_ids) > 1,
                "doc_ids": question.doc_ids,
                "hits": [hit_with_token_debug(hit, query_tokens) for hit in hits],
            }
        )
    return {"qid": question.qid, "trace_mode": "per_option", "option_traces": option_traces}


TRACE_BUILDERS = {
    "regulatory": regulatory_trace,
    "financial_reports": financial_reports_trace,
    "insurance": insurance_trace,
    "research": research_trace,
    "financial_contracts": contracts_trace,
}


def infer_domain_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    for domain in TRACE_BUILDERS:
        if name.startswith(domain + "_"):
            return domain
    raise ValueError(f"Could not infer domain from run dir: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--split", default="A")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    domain = args.domain or infer_domain_from_run_dir(run_dir)
    questions = load_questions(domain, args.split)
    index_path = ROOT / "artifacts" / "index" / domain / "index.json"
    index_payload = read_json(index_path)
    builder = TRACE_BUILDERS[domain]

    traces = {}
    for question in questions:
        traces[question.qid] = builder(question, index_payload)

    output_path = run_dir / "outputs" / "debug" / "bm25_retrievals.json"
    ensure_dir(output_path.parent)
    write_json(output_path, traces)
    print(output_path)


if __name__ == "__main__":
    main()
