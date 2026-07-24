#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.runner import _calculation_semantic_query_terms
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.regulatory.facts import build_regulatory_query_variants
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.io_utils import write_json
from afa_agent.models import Question, RetrievalHit
from afa_agent.retrieval_query import RetrievalRequest, generate_retrieval_plan
from afa_agent.text_utils import tokenize_zh


ANSWER_FORMAT_BY_TYPE = {
    "多选题": "multi",
    "单选题": "mcq",
    "判断题": "tf",
    "计算题": "calculation",
}

CHOICE_SETTINGS = {
    "financial_contracts": {
        "top_k": 7,
        "unit_type_boosts": {"element_block": 1.8, "paragraph": 1.0},
    },
    "financial_reports": {
        "top_k": 7,
        "unit_type_boosts": {"metric_row": 1.8, "paragraph": 1.0},
    },
    "insurance": {
        "top_k": 6,
        "unit_type_boosts": {"formula_block": 1.8, "clause_block": 1.1},
    },
    "research": {
        "top_k": 7,
        "unit_type_boosts": {"conclusion_block": 1.6, "paragraph": 1.0},
    },
    "regulatory": {
        "top_k": 6,
        "unit_type_boosts": {},
    },
}

CALCULATION_SETTINGS = {
    "top_k": 10,
    "unit_type_boosts": {
        "metric_row": 1.8,
        "formula_block": 2.0,
        "clause_block": 1.5,
        "article": 1.3,
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_questions(question_root: Path) -> dict[str, Question]:
    questions: dict[str, Question] = {}
    for path in sorted(question_root.iterdir()):
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line.lstrip("\ufeff"))
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        elif path.suffix == ".json":
            rows = _read_json(path)
        else:
            continue
        for row in rows:
            answer_format = ANSWER_FORMAT_BY_TYPE.get(
                str(row.get("type", "")),
                str(row.get("answer_format", "")),
            )
            questions[str(row["qid"])] = Question(
                qid=str(row["qid"]),
                domain=str(row["domain"]),
                split=str(row.get("split", "B")),
                question=str(row["question"]),
                options={str(key): str(value) for key, value in (row.get("options") or {}).items()},
                answer_format=answer_format,
                type=str(row.get("type", "")),
                doc_ids=[],
            )
    return questions


def _normalize_unit_id(unit_id: str) -> str:
    return unit_id.replace("__dup2", "").replace("__dup", "")


def _make_retriever(domain: str, units: list[dict[str, Any]]):
    if domain == "regulatory":
        return RegulatoryRetriever(units)
    return GenericBM25Retriever(units)


def _search(
    retriever: Any,
    doc_ids: list[str],
    query: str,
    *,
    top_k: int,
    unit_type_boosts: dict[str, float],
) -> list[RetrievalHit]:
    kwargs = {
        "top_k": top_k,
        "ensure_per_doc": len(doc_ids) > 1,
        "expand_neighbors": True,
    }
    try:
        return retriever.search(
            doc_ids,
            query,
            unit_type_boosts=unit_type_boosts,
            **kwargs,
        )
    except TypeError:
        return retriever.search(doc_ids, query, **kwargs)


def _legacy_queries(question: Question) -> list[str]:
    if question.answer_format == "calculation":
        return [
            "\n".join(
                part
                for part in (
                    question.question,
                    question.type,
                    "数值 公式 单位 日期 条款 计算",
                    _calculation_semantic_query_terms(question.question),
                )
                if part
            )
        ]
    if question.domain == "regulatory":
        variants: list[str] = []
        items = (
            [("A", question.question)]
            if question.answer_format == "tf"
            else list(question.options.items())
        )
        for option_key, option_text in items:
            variants.extend(
                build_regulatory_query_variants(
                    question,
                    option_key,
                    option_text,
                )
            )
        return _dedupe(variants)
    items = (
        [("A", question.question)]
        if question.answer_format == "tf"
        else list(question.options.items())
    )
    return _dedupe(
        f"{question.question}\n{option_text}".strip()
        for _, option_text in items
    )


def _semantic_query_variants(
    question: Question,
    max_queries_per_option: int,
) -> list[tuple[str, str]]:
    if question.answer_format == "calculation":
        requests = [
            RetrievalRequest(
                domain=question.domain,
                question=question.question,
                question_type=question.type,
                answer_format=question.answer_format,
            )
        ]
    else:
        option_texts = (
            [question.question]
            if question.answer_format == "tf"
            else list(question.options.values())
        )
        requests = [
            RetrievalRequest(
                domain=question.domain,
                question=question.question,
                option_text=option_text,
                question_type=question.type,
                answer_format=question.answer_format,
            )
            for option_text in option_texts
        ]
    variants: list[tuple[str, str]] = []
    seen_queries: set[str] = set()
    for request in requests:
        for variant in generate_retrieval_plan(
            request,
            max_queries=max_queries_per_option,
        ).variants:
            query = variant.query.strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            variants.append((variant.channel, query))
    return variants


def _semantic_queries(question: Question, max_queries_per_option: int) -> list[str]:
    return [
        query
        for _, query in _semantic_query_variants(
            question,
            max_queries_per_option,
        )
    ]


def _supplemental_semantic_queries(
    question: Question,
    max_queries_per_option: int,
) -> list[str]:
    allowed_channels = (
        {"coverage"}
        if question.answer_format == "calculation"
        else {"support", "contrast", "scope_check"}
    )
    return _dedupe(
        query
        for channel, query in _semantic_query_variants(
            question,
            max_queries_per_option,
        )
        if channel in allowed_channels
    )


def _merge_gated_tail(
    legacy: dict[str, Any],
    supplemental: dict[str, Any],
    *,
    reserve_legacy: int,
    final_top_k: int,
) -> dict[str, Any]:
    if reserve_legacy < 0 or reserve_legacy > final_top_k:
        raise ValueError("reserve_legacy must be between zero and final_top_k")
    legacy_ids = list(legacy["ranked_ids"])
    supplemental_ids = list(supplemental["ranked_ids"])
    combined_ids = legacy_ids[:reserve_legacy]
    combined_ids.extend(
        unit_id
        for unit_id in supplemental_ids
        if unit_id not in combined_ids
    )
    combined_ids = combined_ids[:final_top_k]
    combined_ids.extend(
        unit_id
        for unit_id in legacy_ids
        if unit_id not in combined_ids
    )
    combined_ids = combined_ids[:final_top_k]
    hit_by_id = {
        _normalize_unit_id(hit.unit_id): hit
        for hit in [*legacy["ranked_hits"], *supplemental["ranked_hits"]]
    }
    combined_hits = [hit_by_id[unit_id] for unit_id in combined_ids]
    return {
        "query_count": legacy["query_count"] + supplemental["query_count"],
        "query_token_count": (
            legacy["query_token_count"] + supplemental["query_token_count"]
        ),
        "ranked_ids": combined_ids,
        "ranked_hits": combined_hits,
        "evidence_chars": sum(len(hit.text) for hit in combined_hits),
    }


def _merge_rank_blend(
    legacy: dict[str, Any],
    supplemental: dict[str, Any],
    *,
    supplemental_weight: float,
    final_top_k: int,
) -> dict[str, Any]:
    if supplemental_weight < 0.0:
        raise ValueError("supplemental_weight must not be negative")
    blended_scores: defaultdict[str, float] = defaultdict(float)
    for rank, unit_id in enumerate(legacy["ranked_ids"], start=1):
        blended_scores[unit_id] += 1.0 / rank
    for rank, unit_id in enumerate(supplemental["ranked_ids"], start=1):
        blended_scores[unit_id] += supplemental_weight / rank
    combined_ids = sorted(
        blended_scores,
        key=lambda unit_id: (-blended_scores[unit_id], unit_id),
    )[:final_top_k]
    hit_by_id = {
        _normalize_unit_id(hit.unit_id): hit
        for hit in [*legacy["ranked_hits"], *supplemental["ranked_hits"]]
    }
    combined_hits = [hit_by_id[unit_id] for unit_id in combined_ids]
    return {
        "query_count": legacy["query_count"] + supplemental["query_count"],
        "query_token_count": (
            legacy["query_token_count"] + supplemental["query_token_count"]
        ),
        "ranked_ids": combined_ids,
        "ranked_hits": combined_hits,
        "evidence_chars": sum(len(hit.text) for hit in combined_hits),
    }


def _rank_queries(
    retriever: Any,
    doc_ids: list[str],
    queries: list[str],
    *,
    per_query_top_k: int,
    final_top_k: int,
    unit_type_boosts: dict[str, float],
    rrf_k: int,
) -> dict[str, Any]:
    fused_scores: defaultdict[str, float] = defaultdict(float)
    hit_by_id: dict[str, RetrievalHit] = {}
    for query in queries:
        hits = _search(
            retriever,
            doc_ids,
            query,
            top_k=per_query_top_k,
            unit_type_boosts=unit_type_boosts,
        )
        seen_in_query: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            normalized = _normalize_unit_id(hit.unit_id)
            if normalized in seen_in_query:
                continue
            seen_in_query.add(normalized)
            fused_scores[normalized] += 1.0 / (rrf_k + rank)
            current = hit_by_id.get(normalized)
            if current is None or hit.score > current.score:
                hit_by_id[normalized] = hit
    ranked_ids = sorted(
        fused_scores,
        key=lambda unit_id: (-fused_scores[unit_id], unit_id),
    )[:final_top_k]
    ranked_hits = [hit_by_id[unit_id] for unit_id in ranked_ids]
    return {
        "query_count": len(queries),
        "query_token_count": sum(len(tokenize_zh(query)) for query in queries),
        "ranked_ids": ranked_ids,
        "ranked_hits": ranked_hits,
        "evidence_chars": sum(len(hit.text) for hit in ranked_hits),
    }


def _metrics(ranked_ids: list[str], gold_ids: set[str]) -> dict[str, float]:
    if not gold_ids:
        return {
            "any_recall_at_5": 0.0,
            "any_recall_at_10": 0.0,
            "coverage_at_5": 0.0,
            "coverage_at_10": 0.0,
            "mrr_at_10": 0.0,
        }
    top5 = set(ranked_ids[:5])
    top10 = set(ranked_ids[:10])
    first_rank = next(
        (rank for rank, unit_id in enumerate(ranked_ids[:10], start=1) if unit_id in gold_ids),
        None,
    )
    return {
        "any_recall_at_5": float(bool(top5 & gold_ids)),
        "any_recall_at_10": float(bool(top10 & gold_ids)),
        "coverage_at_5": len(top5 & gold_ids) / len(gold_ids),
        "coverage_at_10": len(top10 & gold_ids) / len(gold_ids),
        "mrr_at_10": 1.0 / first_rank if first_rank else 0.0,
    }


def _mean(rows: list[dict[str, Any]], path: tuple[str, str]) -> float:
    values = [float(row[path[0]][path[1]]) for row in rows]
    return statistics.mean(values) if values else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"evaluable_question_count": len(rows)}
    for method in ("legacy", "semantic"):
        summary[method] = {
            metric: round(_mean(rows, (method, metric)), 6)
            for metric in (
                "any_recall_at_5",
                "any_recall_at_10",
                "coverage_at_5",
                "coverage_at_10",
                "mrr_at_10",
                "query_count",
                "query_token_count",
                "evidence_chars",
            )
        }
    summary["delta_semantic_minus_legacy"] = {
        key: round(summary["semantic"][key] - summary["legacy"][key], 6)
        for key in summary["legacy"]
    }
    summary["recall_at_10_outcomes"] = dict(
        Counter(
            "win"
            if row["semantic"]["coverage_at_10"] > row["legacy"]["coverage_at_10"]
            else "loss"
            if row["semantic"]["coverage_at_10"] < row["legacy"]["coverage_at_10"]
            else "tie"
            for row in rows
        )
    )
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    questions = _load_questions(args.question_root)
    answer_rows = _read_json(args.answer_artifact)
    index_payloads = {
        domain: _read_json(args.index_root / domain / "index.json")
        for domain in CHOICE_SETTINGS
    }
    retrievers = {
        domain: _make_retriever(domain, payload.get("units", []))
        for domain, payload in index_payloads.items()
    }
    unit_ids_by_domain = {
        domain: {
            _normalize_unit_id(str(unit.get("unit_id", "")))
            for unit in payload.get("units", [])
            if unit.get("unit_id")
        }
        for domain, payload in index_payloads.items()
    }
    unit_doc_by_domain = {
        domain: {
            _normalize_unit_id(str(unit["unit_id"])): str(unit["doc_id"])
            for unit in payload.get("units", [])
            if unit.get("unit_id") and unit.get("doc_id")
        }
        for domain, payload in index_payloads.items()
    }
    per_question: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for answer_row in answer_rows:
        qid = str(answer_row["qid"])
        question = questions.get(qid)
        if question is None:
            skipped.append({"qid": qid, "reason": "missing_question"})
            continue
        selected_doc_ids = [
            str(item)
            for item in (answer_row.get("locator", {}).get("selected_doc_ids") or [])
            if item
        ]
        if not selected_doc_ids:
            skipped.append({"qid": qid, "reason": "missing_selected_doc_ids"})
            continue
        question.doc_ids = selected_doc_ids
        domain_units = unit_ids_by_domain[question.domain]
        unit_docs = unit_doc_by_domain[question.domain]
        raw_gold_ids = {
            _normalize_unit_id(str(unit_id))
            for unit_id in answer_row.get("used_evidence_ids", [])
            if unit_id and not str(unit_id).startswith("question:")
        }
        corpus_gold_ids = raw_gold_ids & domain_units
        eligible_gold_ids = {
            unit_id
            for unit_id in corpus_gold_ids
            if unit_docs.get(unit_id) in set(selected_doc_ids)
        }
        if not eligible_gold_ids:
            skipped.append(
                {
                    "qid": qid,
                    "reason": "no_gold_in_selected_corpus",
                    "raw_gold_count": len(raw_gold_ids),
                    "corpus_gold_count": len(corpus_gold_ids),
                }
            )
            continue
        settings = (
            CALCULATION_SETTINGS
            if question.answer_format == "calculation"
            else CHOICE_SETTINGS[question.domain]
        )
        legacy_queries = _legacy_queries(question)
        legacy_ranked = _rank_queries(
            retrievers[question.domain],
            selected_doc_ids,
            legacy_queries,
            per_query_top_k=settings["top_k"],
            final_top_k=args.final_top_k,
            unit_type_boosts=settings["unit_type_boosts"],
            rrf_k=args.rrf_k,
        )
        if args.fusion_mode == "all_channels_rrf":
            semantic_queries = _semantic_queries(
                question,
                args.max_queries_per_option,
            )
            semantic_ranked = _rank_queries(
                retrievers[question.domain],
                selected_doc_ids,
                semantic_queries,
                per_query_top_k=settings["top_k"],
                final_top_k=args.final_top_k,
                unit_type_boosts=settings["unit_type_boosts"],
                rrf_k=args.rrf_k,
            )
            semantic_queries_used = semantic_queries
            supplemental_queries: list[str] = []
        else:
            supplemental_queries = _supplemental_semantic_queries(
                question,
                args.max_queries_per_option,
            )
            supplemental_ranked = _rank_queries(
                retrievers[question.domain],
                selected_doc_ids,
                supplemental_queries,
                per_query_top_k=settings["top_k"],
                final_top_k=max(args.final_top_k * 3, args.final_top_k),
                unit_type_boosts=settings["unit_type_boosts"],
                rrf_k=args.rrf_k,
            )
            if args.fusion_mode == "gated_tail_v1":
                semantic_ranked = _merge_gated_tail(
                    legacy_ranked,
                    supplemental_ranked,
                    reserve_legacy=args.reserve_legacy,
                    final_top_k=args.final_top_k,
                )
            else:
                semantic_ranked = _merge_rank_blend(
                    legacy_ranked,
                    supplemental_ranked,
                    supplemental_weight=args.supplemental_weight,
                    final_top_k=args.final_top_k,
                )
            semantic_queries_used = [*legacy_queries, *supplemental_queries]
        ranked = {
            "legacy": legacy_ranked,
            "semantic": semantic_ranked,
        }
        row: dict[str, Any] = {
            "qid": qid,
            "domain": question.domain,
            "answer_format": question.answer_format,
            "selected_doc_ids": selected_doc_ids,
            "raw_gold_count": len(raw_gold_ids),
            "eligible_gold_ids": sorted(eligible_gold_ids),
            "unreachable_gold_count": len(raw_gold_ids - eligible_gold_ids),
        }
        for method, payload in ranked.items():
            row[method] = {
                **_metrics(payload["ranked_ids"], eligible_gold_ids),
                "query_count": payload["query_count"],
                "query_token_count": payload["query_token_count"],
                "evidence_chars": payload["evidence_chars"],
                "ranked_ids": payload["ranked_ids"],
                "queries": (
                    legacy_queries
                    if method == "legacy"
                    else semantic_queries_used
                ),
            }
            if method == "semantic" and args.fusion_mode != "all_channels_rrf":
                row[method]["supplemental_queries"] = supplemental_queries
        per_question.append(row)
    by_domain = {
        domain: _summary([row for row in per_question if row["domain"] == domain])
        for domain in CHOICE_SETTINGS
    }
    by_format = {
        answer_format: _summary(
            [row for row in per_question if row["answer_format"] == answer_format]
        )
        for answer_format in sorted({row["answer_format"] for row in per_question})
    }
    return {
        "experiment_id": {
            "all_channels_rrf": "semantic_slots_v1_first_pass_retrieval_proxy_a1",
            "gated_tail_v1": "semantic_slots_v1_gated_tail_retrieval_proxy_a2",
            "rank_blend_v1": "semantic_slots_v1_rank_blend_retrieval_proxy_a3",
        }[args.fusion_mode],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "score_type": "offline_a13_evidence_overlap_proxy_not_official",
        "generator_uses_qid": False,
        "model_calls": 0,
        "question_root": str(args.question_root),
        "answer_artifact": str(args.answer_artifact),
        "index_root": str(args.index_root),
        "settings": {
            "max_queries_per_option": args.max_queries_per_option,
            "final_top_k": args.final_top_k,
            "rrf_k": args.rrf_k,
            "fusion_mode": args.fusion_mode,
            "reserve_legacy": args.reserve_legacy,
            "supplemental_weight": args.supplemental_weight,
            "gold_policy": "A13 used_evidence_ids present in index and selected locator docs",
            "comparison": (
                "legacy first-pass queries vs semantic multi-channel queries"
                if args.fusion_mode == "all_channels_rrf"
                else (
                    "legacy first-pass ranking vs legacy primary with answer-blind "
                    "semantic supplemental queries conservatively blended into the tail"
                )
            ),
            "supplemental_channel_policy": (
                None
                if args.fusion_mode == "all_channels_rrf"
                else {
                    "calculation": ["coverage"],
                    "choice": ["support", "contrast", "scope_check"],
                }
            ),
        },
        "limitations": [
            "A13 used_evidence_ids contain supporting and noisy evidence; they are not official gold labels.",
            "Evidence outside selected locator documents is excluded as unreachable by this answer-stage test.",
            "This test measures first-pass retrieval only; it does not run evidence-gate rescue or Qwen.",
            "Query count and query tokens measure retrieval work, not submitted model-token usage.",
        ],
        "overall": _summary(per_question),
        "by_domain": by_domain,
        "by_answer_format": by_format,
        "skipped": skipped,
        "per_question": per_question,
    }


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question-root",
        type=Path,
        default=Path("/Users/abandon/Documents/AFA_ww/upload_b/question_b"),
    )
    parser.add_argument(
        "--answer-artifact",
        type=Path,
        default=Path(
            "/Users/abandon/Documents/AFA_ww/artifacts/b_board_actual/"
            "qwen37_integrated_full100_candidate_a13_reasoning_provenance/answers.json"
        ),
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(
            "/Users/abandon/Documents/AFA_ww/artifacts/preprocessed_loop_candidates/index"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "retrieval_query_generator"
        / "semantic_slots_v1_first_pass_a1.json",
    )
    parser.add_argument("--max-queries-per-option", type=int, default=12)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--fusion-mode",
        choices=("all_channels_rrf", "gated_tail_v1", "rank_blend_v1"),
        default="all_channels_rrf",
    )
    parser.add_argument("--reserve-legacy", type=int, default=8)
    parser.add_argument("--supplemental-weight", type=float, default=0.11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
