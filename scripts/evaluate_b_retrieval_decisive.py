#!/usr/bin/env python3
"""Evaluator-only decisive-evidence replay for the B-board retrieval pipeline.

This module deliberately lives outside ``src/``.  It consumes a frozen
evaluator manifest after generation, calls the real answer-blind retrieval
entry point against the full corpus, and never reads proxy answers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import BQuestion, load_b_questions
from afa_agent.b_board.retrieval_llm_baseline import (
    DOCUMENT_CANDIDATE_STRATEGIES,
    EVIDENCE_QUOTA_STRATEGIES,
    QUERY_PLAN_STRATEGIES,
    load_domain_indexes,
    make_retriever,
    retrieve_question_evidence,
)
from afa_agent.io_utils import write_json
from afa_agent.text_utils import tokenize_zh


DEFAULT_SOURCE_ROOT = Path("/Users/abandon/Documents/AFA_ww")
DEFAULT_MANIFEST = (
    ROOT / "experiments/b_board_actual/proxy30_decisive_evidence_v1.json"
)
DEFAULT_INDEX_ROOT = (
    DEFAULT_SOURCE_ROOT / "artifacts/preprocessed_loop_candidates/index"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/b_board_actual/compliance_repair"
    / "proxy30_retrieval_decisive_baseline_v1.json"
)
_DUPLICATE_SUFFIX_RE = re.compile(r"__dup(?:[0-9]+)?(?=::|$)")
_PROHIBITED_MANIFEST_KEYS = {
    "answer",
    "answer_parts",
    "candidate_answer",
    "expected_answer",
    "official_answer",
    "proxy_answer",
    "reference_answer",
}


def normalize_identifier(value: str) -> str:
    """Canonicalize duplicate index copies without merging supplemental text."""

    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    return _DUPLICATE_SUFFIX_RE.sub("", normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "decisive_evidence_manifest_v1":
        raise ValueError("unsupported decisive evidence manifest schema")
    if payload.get("artifact_class") != "evaluator_only":
        raise ValueError("manifest must declare artifact_class=evaluator_only")
    if payload.get("production_load_policy") != "deny":
        raise ValueError("manifest must deny production loading")
    if payload.get("official_ground_truth") is not False:
        raise ValueError("manifest must not claim official ground truth")
    if payload.get("contains_expected_answers") is not False:
        raise ValueError("manifest must not contain expected answers")
    prohibited = sorted(
        {key.lower() for key in _walk_keys(payload)} & _PROHIBITED_MANIFEST_KEYS
    )
    if prohibited:
        raise ValueError(f"manifest contains prohibited answer keys: {prohibited}")
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != 30:
        raise ValueError("manifest must contain exactly 30 questions")
    qids: set[str] = set()
    for row in questions:
        if not isinstance(row, Mapping):
            raise ValueError("manifest question entries must be objects")
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in qids:
            raise ValueError(f"invalid or duplicate qid: {qid!r}")
        qids.add(qid)
        if not row.get("required_doc_groups"):
            raise ValueError(f"{qid}: required_doc_groups must not be empty")
        if not row.get("fact_groups"):
            raise ValueError(f"{qid}: fact_groups must not be empty")
        for group in [
            *row.get("required_doc_groups", []),
            *row.get("fact_groups", []),
        ]:
            requirement = group.get("requirement", "any_of")
            if requirement not in {"any_of", "all_of"}:
                raise ValueError(f"{qid}: invalid group requirement {requirement}")


def _group_hit(
    group: Mapping[str, Any],
    observed_ids: Sequence[str] | set[str],
    *,
    id_key: str,
) -> bool:
    acceptable = {
        normalize_identifier(item)
        for item in group.get(id_key, [])
        if str(item).strip()
    }
    observed = {normalize_identifier(item) for item in observed_ids}
    if not acceptable:
        return False
    if group.get("requirement", "any_of") == "all_of":
        return acceptable <= observed
    return bool(acceptable & observed)


def _first_group_rank(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[str],
    *,
    limit: int,
) -> int | None:
    normalized = [normalize_identifier(item) for item in ranked_ids[:limit]]
    acceptable = {
        normalize_identifier(unit_id)
        for group in groups
        for unit_id in group.get("acceptable_unit_ids", [])
    }
    return next(
        (rank for rank, unit_id in enumerate(normalized, start=1) if unit_id in acceptable),
        None,
    )


def ranking_metrics(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[str],
    *,
    reachable_group_ids: set[str] | None = None,
) -> dict[str, float]:
    """Return strict and reachable-only fact-group metrics."""

    def hits_at(limit: int) -> list[bool]:
        return [
            _group_hit(group, ranked_ids[:limit], id_key="acceptable_unit_ids")
            for group in groups
        ]

    hits5 = hits_at(5)
    hits10 = hits_at(10)
    first_rank = _first_group_rank(groups, ranked_ids, limit=10)
    reachable_indexes = [
        index
        for index, group in enumerate(groups)
        if reachable_group_ids is None
        or str(group.get("group_id", "")) in reachable_group_ids
    ]
    reachable_hits10 = [hits10[index] for index in reachable_indexes]
    return {
        "any_recall_at_5": float(any(hits5)),
        "any_recall_at_10": float(any(hits10)),
        "fact_coverage_at_5": sum(hits5) / len(groups) if groups else 0.0,
        "fact_coverage_at_10": sum(hits10) / len(groups) if groups else 0.0,
        "fact_complete_at_10": float(bool(groups) and all(hits10)),
        "reachable_fact_coverage_at_10": (
            sum(reachable_hits10) / len(reachable_hits10)
            if reachable_hits10
            else 0.0
        ),
        "mrr_at_10": 1.0 / first_rank if first_rank else 0.0,
    }


def full_pool_metrics(
    groups: Sequence[Mapping[str, Any]],
    pool_ids: Sequence[str],
) -> dict[str, float]:
    hits = [
        _group_hit(group, pool_ids, id_key="acceptable_unit_ids")
        for group in groups
    ]
    return {
        "any_recall": float(any(hits)),
        "fact_coverage": sum(hits) / len(hits) if hits else 0.0,
        "fact_complete": float(bool(hits) and all(hits)),
    }


def document_metrics(
    groups: Sequence[Mapping[str, Any]],
    selected_doc_ids: Sequence[str],
) -> dict[str, float]:
    hits = [
        _group_hit(group, selected_doc_ids, id_key="acceptable_doc_ids")
        for group in groups
    ]
    return {
        "doc_any_recall": float(any(hits)),
        "doc_coverage": sum(hits) / len(hits) if hits else 0.0,
        "doc_complete": float(bool(hits) and all(hits)),
    }


def _index_catalog(
    index_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    unit_ids: dict[str, set[str]] = {}
    unit_docs: dict[str, dict[str, set[str]]] = {}
    for domain, payload in index_payloads.items():
        domain_units: set[str] = set()
        domain_docs: defaultdict[str, set[str]] = defaultdict(set)
        for unit in payload.get("units", []):
            raw_unit_id = str(unit.get("unit_id", "")).strip()
            raw_doc_id = str(unit.get("doc_id", "")).strip()
            if not raw_unit_id:
                continue
            unit_id = normalize_identifier(raw_unit_id)
            domain_units.add(unit_id)
            if raw_doc_id:
                domain_docs[unit_id].add(normalize_identifier(raw_doc_id))
        unit_ids[domain] = domain_units
        unit_docs[domain] = dict(domain_docs)
    return unit_ids, unit_docs


def _reachable_group_ids(
    groups: Sequence[Mapping[str, Any]],
    available_unit_ids: set[str],
) -> set[str]:
    return {
        str(group["group_id"])
        for group in groups
        if _group_hit(
            group,
            available_unit_ids,
            id_key="acceptable_unit_ids",
        )
    }


def _ranking_ids(ranking: Mapping[str, Any]) -> list[str]:
    return [
        normalize_identifier(item)
        for item in ranking.get("ranked_ids", [])
        if str(item).strip()
    ]


def evidence_pool_ids(retrieval: Mapping[str, Any]) -> list[str]:
    """Union the answer-stage evidence pool before final Top-K blending."""

    ids: list[str] = []
    seen: set[str] = set()
    rankings: list[Mapping[str, Any]] = [
        retrieval.get("primary", {}),
        retrieval.get("supplemental", {}),
        *retrieval.get("document_rankings", []),
        *retrieval.get("metric_slot_rankings", []),
        *retrieval.get("option_rankings", []),
    ]
    for ranking in rankings:
        for unit_id in _ranking_ids(ranking):
            if unit_id in seen:
                continue
            seen.add(unit_id)
            ids.append(unit_id)
    return ids


def _all_ranking_hits(retrieval: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    hits: dict[str, Mapping[str, Any]] = {}
    rankings: list[Mapping[str, Any]] = [
        retrieval.get("document_discovery", {}),
        retrieval.get("primary", {}),
        retrieval.get("supplemental", {}),
        retrieval.get("final", {}),
        *retrieval.get("document_rankings", []),
        *retrieval.get("metric_slot_rankings", []),
        *retrieval.get("option_rankings", []),
    ]
    for ranking in rankings:
        for hit in ranking.get("hits", []):
            unit_id = normalize_identifier(str(hit.get("unit_id", "")))
            if unit_id and unit_id not in hits:
                hits[unit_id] = hit
    return hits


def _field_hit(
    field: Mapping[str, Any],
    group: Mapping[str, Any],
    observed_ids: set[str],
    hits_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    if field.get("source", "evidence") == "question":
        return True
    group_ids = {
        normalize_identifier(item)
        for item in group.get("acceptable_unit_ids", [])
    }
    texts = []
    for unit_id in observed_ids & group_ids:
        hit = hits_by_id.get(unit_id, {})
        texts.append(
            " ".join(
                [
                    *[str(item) for item in hit.get("title_path", [])],
                    str(hit.get("text", "")),
                ]
            )
        )
    aliases = [str(item) for item in field.get("aliases", []) if str(item)]
    if not texts or not aliases:
        return False
    normalized_text = unicodedata.normalize("NFKC", "\n".join(texts))
    match_mode = field.get("match", "alias")
    if match_mode == "regex":
        return any(re.search(pattern, normalized_text) for pattern in aliases)
    compact_text = re.sub(r"\s+", "", normalized_text)
    return any(
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", alias)) in compact_text
        for alias in aliases
    )


def field_metrics(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[str],
    hits_by_id: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, float | int]:
    observed = {
        normalize_identifier(item)
        for item in ranked_ids[:limit]
    }
    evidence_total = 0
    evidence_hit = 0
    question_total = 0
    for group in groups:
        for field in group.get("required_fields", []):
            if field.get("source", "evidence") == "question":
                question_total += 1
                continue
            evidence_total += 1
            evidence_hit += int(
                _field_hit(field, group, observed, hits_by_id)
            )
    return {
        f"field_coverage_at_{limit}": (
            evidence_hit / evidence_total if evidence_total else 0.0
        ),
        "evidence_field_hit_count": evidence_hit,
        "evidence_field_count": evidence_total,
        "question_field_count": question_total,
    }


def query_work(retrieval: Mapping[str, Any]) -> dict[str, int]:
    """Count local retrieval work; these are not model/API token metrics."""

    queries: list[str] = []
    rankings: list[Mapping[str, Any]] = [
        retrieval.get("document_discovery", {}),
        retrieval.get("primary", {}),
        retrieval.get("supplemental", {}),
        *retrieval.get("document_rankings", []),
        *retrieval.get("metric_slot_rankings", []),
        *retrieval.get("option_rankings", []),
    ]
    for ranking in rankings:
        queries.extend(
            str(query)
            for query in ranking.get("queries", [])
            if str(query).strip()
        )
    # Anchor queries are executed separately and are not represented by a
    # ranking payload.
    queries.extend(
        str(query)
        for query in retrieval.get("bundle", {}).get("anchor_queries", [])
        if str(query).strip()
    )
    unique_queries = list(dict.fromkeys(queries))
    return {
        "query_execution_count": len(queries),
        "unique_query_count": len(unique_queries),
        "retrieval_term_count": sum(
            len(tokenize_zh(query)) for query in queries
        ),
        "unique_retrieval_term_count": sum(
            len(tokenize_zh(query)) for query in unique_queries
        ),
    }


def _evaluate_one(
    row: Mapping[str, Any],
    *,
    question: BQuestion,
    retriever: Any,
    all_doc_ids: Sequence[str],
    available_unit_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    retrieval = retrieve_question_evidence(
        question,
        retriever=retriever,
        all_doc_ids=all_doc_ids,
        per_query_top_k=args.per_query_top_k,
        final_top_k=args.final_top_k,
        supplemental_weight=args.supplemental_weight,
        max_queries_per_option=args.max_queries_per_option,
        max_doc_candidates=args.max_doc_candidates,
        document_candidate_strategy=args.document_candidate_strategy,
        evidence_quota_strategy=args.evidence_quota_strategy,
        query_plan_strategy=args.query_plan_strategy,
    )
    fact_groups = list(row["fact_groups"])
    doc_groups = list(row["required_doc_groups"])
    reachable_group_ids = _reachable_group_ids(
        fact_groups,
        available_unit_ids,
    )
    selected_doc_ids = [
        normalize_identifier(item)
        for item in retrieval.get("selected_doc_ids", [])
    ]
    final_ids = _ranking_ids(retrieval["final"])
    pool_ids = evidence_pool_ids(retrieval)
    hits_by_id = _all_ranking_hits(retrieval)
    final_metrics = ranking_metrics(
        fact_groups,
        final_ids,
        reachable_group_ids=reachable_group_ids,
    )
    pool_metrics = full_pool_metrics(fact_groups, pool_ids)
    fact_group_status = []
    for group in fact_groups:
        group_id = str(group["group_id"])
        in_pool = _group_hit(
            group,
            pool_ids,
            id_key="acceptable_unit_ids",
        )
        in_top10 = _group_hit(
            group,
            final_ids[:10],
            id_key="acceptable_unit_ids",
        )
        fact_group_status.append(
            {
                "group_id": group_id,
                "corpus_reachable": group_id in reachable_group_ids,
                "in_evidence_pool": in_pool,
                "in_final_top10": in_top10,
                "pool_to_top10_loss": in_pool and not in_top10,
            }
        )
    return {
        "qid": question.qid,
        "domain": question.domain,
        "answer_format": question.answer_format,
        "policy_version": retrieval.get("policy_version"),
        "selected_doc_ids": selected_doc_ids,
        "corpus_reachable_fact_count": len(reachable_group_ids),
        "fact_group_count": len(fact_groups),
        "corpus_fact_reachability": (
            len(reachable_group_ids) / len(fact_groups)
            if fact_groups
            else 0.0
        ),
        "document": document_metrics(doc_groups, selected_doc_ids),
        "final": {
            **final_metrics,
            **field_metrics(
                fact_groups,
                final_ids,
                hits_by_id,
                limit=5,
            ),
            **field_metrics(
                fact_groups,
                final_ids,
                hits_by_id,
                limit=10,
            ),
            "ranked_ids": final_ids,
        },
        "pool": {
            **pool_metrics,
            "unit_count": len(pool_ids),
            "ranked_ids": pool_ids,
        },
        "pool_to_top10_loss_count": sum(
            item["pool_to_top10_loss"] for item in fact_group_status
        ),
        "fact_groups": fact_group_status,
        "negative_scope_group_count": len(
            row.get("negative_scope_groups", [])
        ),
        "known_unreachable": list(row.get("known_unreachable", [])),
        "query_work": query_work(retrieval),
        "retrieval_provenance": {
            "document_candidate_strategy": retrieval.get(
                "document_candidate_strategy"
            ),
            "evidence_quota_strategy": retrieval.get(
                "evidence_quota_strategy"
            ),
            "active_evidence_quota_strategy": retrieval.get(
                "active_evidence_quota_strategy"
            ),
            "anchored_doc_ids": [
                normalize_identifier(item)
                for item in retrieval.get("anchored_doc_ids", [])
            ],
            "document_discovery_ranked_ids": _ranking_ids(
                retrieval.get("document_discovery", {})
            ),
        },
    }


def _mean(rows: Sequence[Mapping[str, Any]], *path: str) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.mean(values) if values else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "question_count": len(rows),
        "corpus_fact_reachability": round(
            _mean(rows, "corpus_fact_reachability"), 6
        ),
        "doc_any_recall": round(
            _mean(rows, "document", "doc_any_recall"), 6
        ),
        "doc_coverage": round(
            _mean(rows, "document", "doc_coverage"), 6
        ),
        "doc_complete": round(
            _mean(rows, "document", "doc_complete"), 6
        ),
        "any_recall_at_5": round(
            _mean(rows, "final", "any_recall_at_5"), 6
        ),
        "any_recall_at_10": round(
            _mean(rows, "final", "any_recall_at_10"), 6
        ),
        "fact_coverage_at_5": round(
            _mean(rows, "final", "fact_coverage_at_5"), 6
        ),
        "fact_coverage_at_10": round(
            _mean(rows, "final", "fact_coverage_at_10"), 6
        ),
        "fact_complete_at_10": round(
            _mean(rows, "final", "fact_complete_at_10"), 6
        ),
        "field_coverage_at_5": round(
            _mean(rows, "final", "field_coverage_at_5"), 6
        ),
        "field_coverage_at_10": round(
            _mean(rows, "final", "field_coverage_at_10"), 6
        ),
        "mrr_at_10": round(_mean(rows, "final", "mrr_at_10"), 6),
        "pool_fact_coverage": round(
            _mean(rows, "pool", "fact_coverage"), 6
        ),
        "pool_to_top10_loss_count": sum(
            int(row["pool_to_top10_loss_count"]) for row in rows
        ),
        "query_execution_count": sum(
            int(row["query_work"]["query_execution_count"]) for row in rows
        ),
        "unique_query_count": sum(
            int(row["query_work"]["unique_query_count"]) for row in rows
        ),
        "retrieval_term_count": sum(
            int(row["query_work"]["retrieval_term_count"]) for row in rows
        ),
        "unique_retrieval_term_count": sum(
            int(row["query_work"]["unique_retrieval_term_count"])
            for row in rows
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.manifest)
    validate_manifest(manifest)
    questions = load_b_questions(
        args.question_root,
        args.submission_template,
    )
    questions_by_qid = {question.qid: question for question in questions}
    index_payloads = load_domain_indexes(args.index_root)
    retrievers = {
        domain: make_retriever(domain, payload["units"])
        for domain, payload in index_payloads.items()
    }
    doc_ids = {
        domain: sorted(
            {
                normalize_identifier(str(unit["doc_id"]))
                for unit in payload["units"]
                if unit.get("doc_id")
            }
        )
        for domain, payload in index_payloads.items()
    }
    unit_ids, _ = _index_catalog(index_payloads)
    manifest_rows = list(manifest["questions"])
    missing_questions = [
        str(row["qid"])
        for row in manifest_rows
        if str(row["qid"]) not in questions_by_qid
    ]
    if missing_questions:
        raise ValueError(f"manifest qids missing from corpus: {missing_questions}")
    per_question: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _evaluate_one,
                row,
                question=questions_by_qid[str(row["qid"])],
                retriever=retrievers[str(row["domain"])],
                all_doc_ids=doc_ids[str(row["domain"])],
                available_unit_ids=unit_ids[str(row["domain"])],
                args=args,
            ): str(row["qid"])
            for row in manifest_rows
        }
        for future in as_completed(futures):
            per_question.append(future.result())
    order = {
        str(row["qid"]): index for index, row in enumerate(manifest_rows)
    }
    per_question.sort(key=lambda row: order[str(row["qid"])])
    domains = sorted({str(row["domain"]) for row in per_question})
    formats = sorted({str(row["answer_format"]) for row in per_question})
    return {
        "schema_version": "b_retrieval_decisive_evaluation_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "score_type": "offline_decisive_evidence_retrieval_not_official_accuracy",
        "evaluator_only": True,
        "contains_expected_answers": False,
        "model_calls": 0,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "question_root": str(args.question_root),
        "index_root": str(args.index_root),
        "settings": {
            "per_query_top_k": args.per_query_top_k,
            "final_top_k": args.final_top_k,
            "supplemental_weight": args.supplemental_weight,
            "max_queries_per_option": args.max_queries_per_option,
            "max_doc_candidates": args.max_doc_candidates,
            "document_candidate_strategy": args.document_candidate_strategy,
            "evidence_quota_strategy": args.evidence_quota_strategy,
            "query_plan_strategy": args.query_plan_strategy,
            "workers": args.workers,
            "query_token_definition": (
                "sum of tokenize_zh tokens over actual retrieval query "
                "executions; not model tokens"
            ),
            "strict_denominator_policy": (
                "all manifest fact groups remain in strict metrics; index "
                "unreachable and locator-missed facts are not discarded"
            ),
        },
        "overall": summarize(per_question),
        "by_domain": {
            domain: summarize(
                [row for row in per_question if row["domain"] == domain]
            )
            for domain in domains
        },
        "by_answer_format": {
            answer_format: summarize(
                [
                    row
                    for row in per_question
                    if row["answer_format"] == answer_format
                ]
            )
            for answer_format in formats
        },
        "per_question": per_question,
        "limitations": [
            "The manifest is an offline evidence-discovery set, not official ground truth.",
            "Field coverage is deterministic alias matching and should be read with fact coverage.",
            "Negative absence claims require a separate full-section or full-document scope audit.",
            "Query work counts retrieval tokenizer tokens, not submitted Qwen tokens.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decisive fact retrieval for the frozen B-board proxy30 "
            "scope without loading proxy answers"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--question-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/question_b",
    )
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "upload_b/submit.csv",
    )
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-query-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--supplemental-weight", type=float, default=0.11)
    parser.add_argument("--max-queries-per-option", type=int, default=12)
    parser.add_argument("--max-doc-candidates", type=int, default=6)
    parser.add_argument(
        "--document-candidate-strategy",
        choices=DOCUMENT_CANDIDATE_STRATEGIES,
        default="anchor_union",
    )
    parser.add_argument(
        "--evidence-quota-strategy",
        choices=EVIDENCE_QUOTA_STRATEGIES,
        default="primary_guard",
    )
    parser.add_argument(
        "--query-plan-strategy",
        choices=QUERY_PLAN_STRATEGIES,
        default="semantic_slots_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_top_k < 10:
        raise ValueError("final_top_k must be at least 10 for @5/@10 metrics")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
