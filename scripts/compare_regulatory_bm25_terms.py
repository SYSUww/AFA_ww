#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.bm25 import BM25Index
from afa_agent.text_utils import DOMAIN_TERMS, KEEP_SINGLE_CHARS, STOPWORD_SINGLE_CHARS, normalize_whitespace


@dataclass
class SearchHit:
    rank: int
    unit_id: str
    doc_id: str
    unit_type: str
    score: float
    title_path: list[str]
    matched_tokens: list[str]
    text_preview: str


def _require_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba is required for BM25 comparison") from exc
    return jieba


def load_terms(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Term file not found: {path}")
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def make_tokenizer(extra_terms: list[str]):
    jieba = _require_jieba()
    tokenizer = jieba.Tokenizer()
    for term in DOMAIN_TERMS:
        tokenizer.add_word(term, freq=200000)
    for term in extra_terms:
        tokenizer.add_word(term, freq=300000)

    def keep_token(token: str) -> bool:
        if not token.strip():
            return False
        if re.fullmatch(r"\W+", token, flags=re.UNICODE):
            return False
        if len(token) == 1 and token not in KEEP_SINGLE_CHARS and token in STOPWORD_SINGLE_CHARS:
            return False
        return True

    def append_unique(tokens: list[str], seen: set[str], token: str) -> None:
        cleaned = token.strip().lower()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        tokens.append(cleaned)

    def tokenize(text: str) -> list[str]:
        normalized = normalize_whitespace(text).lower()
        tokens: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"[a-z0-9_.%]+", normalized):
            append_unique(tokens, seen, match.group(0))
        for match in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|亿元|万元|元|年|月|日|股|倍|个工作日|个月)?", normalized):
            raw = match.group(0).strip()
            append_unique(tokens, seen, raw)
            if "," in raw:
                append_unique(tokens, seen, raw.replace(",", ""))
        for match in re.finditer(r"第[一二三四五六七八九十百零〇两\d]+[章节条款项]", normalized):
            append_unique(tokens, seen, match.group(0))
        for term in DOMAIN_TERMS:
            if term.lower() in normalized:
                append_unique(tokens, seen, term)
        for term in extra_terms:
            if term.lower() in normalized:
                append_unique(tokens, seen, term)
        for token in tokenizer.lcut(normalized, cut_all=False):
            token = token.strip()
            if keep_token(token):
                append_unique(tokens, seen, token)
        for segment in re.findall(r"[\u4e00-\u9fff]{1,}", normalized):
            if len(segment) == 1:
                if keep_token(segment):
                    append_unique(tokens, seen, segment)
                continue
            for item in (segment[i : i + 2] for i in range(len(segment) - 1)):
                append_unique(tokens, seen, item)
        return tokens

    return tokenize


def unit_text(unit: dict[str, Any]) -> str:
    article_no = unit.get("metadata", {}).get("article_no", "")
    title_path = " ".join(unit.get("title_path") or [])
    return f"{title_path}\n{article_no}\n{unit.get('text', '')}".strip()


def score_units(
    *,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    tokenize,
    doc_ids: list[str],
    query: str,
    top_k: int,
) -> tuple[list[str], list[SearchHit]]:
    query_tokens = tokenize(query)
    doc_filter = set(doc_ids)
    scored = []
    for idx, unit in enumerate(units):
        if doc_filter and unit["doc_id"] not in doc_filter:
            continue
        score = bm25.score(query_tokens, idx)
        if score > 0:
            scored.append((idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    hits = []
    for rank, (idx, score) in enumerate(scored[:top_k], start=1):
        unit = units[idx]
        matched_tokens = [token for token in query_tokens if token in set(doc_tokens[idx])]
        text = re.sub(r"\s+", " ", unit.get("text", "")).strip()
        hits.append(
            SearchHit(
                rank=rank,
                unit_id=unit["unit_id"],
                doc_id=unit["doc_id"],
                unit_type=unit.get("unit_type", ""),
                score=round(score, 4),
                title_path=unit.get("title_path") or [],
                matched_tokens=matched_tokens[:60],
                text_preview=text[:220],
            )
        )
    return query_tokens, hits


def compare_option(
    *,
    question: dict[str, Any],
    option_key: str,
    option_text: str,
    units: list[dict[str, Any]],
    baseline: dict[str, Any],
    experiment: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    query = f"{question['question']}\n{option_text}"
    baseline_tokens, baseline_hits = score_units(
        units=units,
        bm25=baseline["bm25"],
        doc_tokens=baseline["doc_tokens"],
        tokenize=baseline["tokenize"],
        doc_ids=question.get("doc_ids", []),
        query=query,
        top_k=top_k,
    )
    experiment_tokens, experiment_hits = score_units(
        units=units,
        bm25=experiment["bm25"],
        doc_tokens=experiment["doc_tokens"],
        tokenize=experiment["tokenize"],
        doc_ids=question.get("doc_ids", []),
        query=query,
        top_k=top_k,
    )
    baseline_ids = [hit.unit_id for hit in baseline_hits]
    experiment_ids = [hit.unit_id for hit in experiment_hits]
    added_tokens = [token for token in experiment_tokens if token not in set(baseline_tokens)]
    removed_tokens = [token for token in baseline_tokens if token not in set(experiment_tokens)]
    return {
        "option": option_key,
        "query": query,
        "baseline_query_tokens": baseline_tokens,
        "experiment_query_tokens": experiment_tokens,
        "added_query_tokens": added_tokens,
        "removed_query_tokens": removed_tokens,
        "baseline_hits": [asdict(hit) for hit in baseline_hits],
        "experiment_hits": [asdict(hit) for hit in experiment_hits],
        "top1_changed": bool(baseline_ids and experiment_ids and baseline_ids[0] != experiment_ids[0]),
        "topk_overlap": len(set(baseline_ids) & set(experiment_ids)),
        "baseline_top_ids": baseline_ids,
        "experiment_top_ids": experiment_ids,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    option_count = 0
    top1_changed = 0
    overlap_total = 0
    token_added_options = 0
    changed_examples = []
    for question in traces:
        for option in question["options"]:
            option_count += 1
            top1_changed += int(option["top1_changed"])
            overlap_total += option["topk_overlap"]
            token_added_options += int(bool(option["added_query_tokens"]))
            if option["top1_changed"] and len(changed_examples) < 12:
                changed_examples.append(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "added_query_tokens": option["added_query_tokens"],
                        "baseline_top1": option["baseline_top_ids"][0] if option["baseline_top_ids"] else "",
                        "experiment_top1": option["experiment_top_ids"][0] if option["experiment_top_ids"] else "",
                    }
                )
    return {
        "question_count": len(traces),
        "option_count": option_count,
        "top1_changed_options": top1_changed,
        "options_with_added_query_tokens": token_added_options,
        "average_topk_overlap": round(overlap_total / option_count, 3) if option_count else 0,
        "changed_examples": changed_examples,
    }


def write_summary_csv(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "option",
                "top1_changed",
                "topk_overlap",
                "added_query_tokens",
                "removed_query_tokens",
                "baseline_top1",
                "experiment_top1",
                "baseline_score",
                "experiment_score",
                "baseline_preview",
                "experiment_preview",
            ],
        )
        writer.writeheader()
        for question in traces:
            for option in question["options"]:
                baseline_top = option["baseline_hits"][0] if option["baseline_hits"] else {}
                experiment_top = option["experiment_hits"][0] if option["experiment_hits"] else {}
                writer.writerow(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "top1_changed": option["top1_changed"],
                        "topk_overlap": option["topk_overlap"],
                        "added_query_tokens": " ".join(option["added_query_tokens"]),
                        "removed_query_tokens": " ".join(option["removed_query_tokens"]),
                        "baseline_top1": baseline_top.get("unit_id", ""),
                        "experiment_top1": experiment_top.get("unit_id", ""),
                        "baseline_score": baseline_top.get("score", ""),
                        "experiment_score": experiment_top.get("score", ""),
                        "baseline_preview": baseline_top.get("text_preview", ""),
                        "experiment_preview": experiment_top.get("text_preview", ""),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regulatory BM25 retrieval before/after selected terms.")
    parser.add_argument("--index-path", type=Path, default=ROOT / "artifacts/index/regulatory/index.json")
    parser.add_argument("--questions-path", type=Path, default=ROOT / "test/regulatory_questions.json")
    parser.add_argument(
        "--terms-path",
        type=Path,
        default=ROOT / "artifacts/preprocessed/regulatory/terms/regulatory_terms_selected.txt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/bm25_ab_compare.json")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/bm25_ab_compare_summary.csv")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    index_payload = json.loads(args.index_path.read_text(encoding="utf-8"))
    questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    units = index_payload["units"]
    selected_terms = load_terms(args.terms_path)

    baseline_tokenize = make_tokenizer([])
    experiment_tokenize = make_tokenizer(selected_terms)
    baseline_doc_tokens = [baseline_tokenize(unit_text(unit)) for unit in units]
    experiment_doc_tokens = [experiment_tokenize(unit_text(unit)) for unit in units]
    baseline = {
        "tokenize": baseline_tokenize,
        "doc_tokens": baseline_doc_tokens,
        "bm25": BM25Index(baseline_doc_tokens),
    }
    experiment = {
        "tokenize": experiment_tokenize,
        "doc_tokens": experiment_doc_tokens,
        "bm25": BM25Index(experiment_doc_tokens),
    }

    traces = []
    for question in questions:
        options = []
        for option_key, option_text in question.get("options", {}).items():
            options.append(
                compare_option(
                    question=question,
                    option_key=option_key,
                    option_text=option_text,
                    units=units,
                    baseline=baseline,
                    experiment=experiment,
                    top_k=args.top_k,
                )
            )
        traces.append(
            {
                "qid": question["qid"],
                "question": question["question"],
                "doc_ids": question.get("doc_ids", []),
                "options": options,
            }
        )

    payload = {
        "index_path": str(args.index_path.resolve()),
        "questions_path": str(args.questions_path.resolve()),
        "terms_path": str(args.terms_path.resolve()),
        "selected_term_count": len(selected_terms),
        "top_k": args.top_k,
        "summary": summarize(traces),
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(args.summary_csv, traces)
    print(args.output)
    print(args.summary_csv)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
