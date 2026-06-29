#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.bm25 import BM25Index
from afa_agent.text_utils import DOMAIN_TERMS, KEEP_SINGLE_CHARS, STOPWORD_SINGLE_CHARS, normalize_whitespace
from compare_regulatory_bm25_terms import SearchHit, load_terms, make_tokenizer, unit_text


QUERY_STOPWORDS = {
    "关于",
    "结合",
    "依据",
    "根据",
    "按照",
    "下列",
    "以下",
    "说法",
    "表述",
    "描述",
    "哪项",
    "正确",
    "准确",
    "符合",
    "相关",
    "要求",
    "现行",
    "判断题",
    "有",
    "中",
}

KEEP_LEGAL_OPERATORS = {
    "应当",
    "不得",
    "可以",
    "无需",
    "需要",
    "至少",
    "提前",
    "终止",
    "废止",
    "施行",
    "提交",
    "报告",
    "披露",
    "核实",
    "保存",
    "审议",
    "批准",
    "处罚",
    "扣减",
    "禁入",
}

TEMPLATE_RE = re.compile(
    r"判断题[:：]|"
    r"^(关于|结合|依据|根据|按照)|"
    r"下列|以下|说法|表述|描述|哪项|正确|准确|符合"
)


def _require_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba is required for query cleaning") from exc
    return jieba


def normalize_query_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = TEMPLATE_RE.sub(" ", text)
    text = re.sub(r"[？?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_clean_query_tokenizer(extra_terms: list[str]):
    jieba = _require_jieba()
    tokenizer = jieba.Tokenizer()
    all_terms = list(dict.fromkeys(list(DOMAIN_TERMS) + extra_terms))
    for term in all_terms:
        tokenizer.add_word(term, freq=300000)

    def append(tokens: list[str], seen: set[str], token: str) -> None:
        token = token.strip().lower()
        if not token or token in seen:
            return
        if token in QUERY_STOPWORDS:
            return
        if len(token) == 1 and token not in KEEP_SINGLE_CHARS and token not in {"对", "错"}:
            return
        if re.fullmatch(r"\W+", token, flags=re.UNICODE):
            return
        seen.add(token)
        tokens.append(token)

    def tokenize(text: str) -> list[str]:
        normalized = normalize_query_text(text).lower()
        tokens: list[str] = []
        seen: set[str] = set()

        for match in re.finditer(r"[a-z0-9_.%]+", normalized):
            append(tokens, seen, match.group(0))
        for match in re.finditer(
            r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|亿元|万元|元|年|月|日|股|倍|个工作日|自然日|个月|美元)?",
            normalized,
        ):
            raw = re.sub(r"\s+", "", match.group(0))
            append(tokens, seen, raw)
            if "," in raw:
                append(tokens, seen, raw.replace(",", ""))
        for match in re.finditer(r"第[一二三四五六七八九十百零〇两\d]+[章节条款项]", normalized):
            append(tokens, seen, match.group(0))
        for term in all_terms:
            if term.lower() in normalized:
                append(tokens, seen, term)

        for token in tokenizer.lcut(normalized, cut_all=False):
            cleaned = token.strip()
            if not cleaned:
                continue
            if cleaned in KEEP_LEGAL_OPERATORS or len(cleaned) >= 2:
                append(tokens, seen, cleaned)

        return tokens

    return tokenize


def make_soft_clean_query_tokenizer(raw_tokenize):
    def is_noisy_bigram(token: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fff]{2}", token):
            return False
        if token in KEEP_LEGAL_OPERATORS:
            return False
        return token in {
            "下列",
            "以下",
            "说法",
            "表述",
            "描述",
            "哪项",
            "正确",
            "准确",
            "符合",
            "相关",
            "要求",
            "现行",
        }

    def tokenize(text: str) -> list[str]:
        tokens = []
        seen = set()
        for token in raw_tokenize(text):
            if token in QUERY_STOPWORDS:
                continue
            if is_noisy_bigram(token):
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens

    return tokenize


def score_tokens(
    *,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    query_tokens: list[str],
    doc_ids: list[str],
    top_k: int,
) -> list[SearchHit]:
    scored = []
    doc_filter = set(doc_ids)
    for idx, unit in enumerate(units):
        if doc_filter and unit["doc_id"] not in doc_filter:
            continue
        score = bm25.score(query_tokens, idx)
        if score <= 0:
            continue
        if unit.get("unit_type") == "preamble":
            score *= 0.55
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
    return hits


def score_blended(
    *,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    raw_tokens: list[str],
    clean_tokens: list[str],
    doc_ids: list[str],
    top_k: int,
    clean_weight: float,
) -> list[SearchHit]:
    scored = []
    doc_filter = set(doc_ids)
    for idx, unit in enumerate(units):
        if doc_filter and unit["doc_id"] not in doc_filter:
            continue
        score = bm25.score(raw_tokens, idx) + bm25.score(clean_tokens, idx) * clean_weight
        if score <= 0:
            continue
        if unit.get("unit_type") == "preamble":
            score *= 0.55
        scored.append((idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    query_tokens = list(dict.fromkeys(raw_tokens + clean_tokens))
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
    return hits


def compare_option(
    *,
    question: dict[str, Any],
    option_key: str,
    option_text: str,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    raw_tokenize,
    clean_tokenize,
    top_k: int,
    clean_weight: float,
) -> dict[str, Any]:
    query = f"{question['question']}\n{option_text}"
    raw_tokens = raw_tokenize(query)
    clean_tokens = clean_tokenize(query)
    raw_hits = score_tokens(
        units=units,
        bm25=bm25,
        doc_tokens=doc_tokens,
        query_tokens=raw_tokens,
        doc_ids=question.get("doc_ids", []),
        top_k=top_k,
    )
    clean_hits = score_blended(
        units=units,
        bm25=bm25,
        doc_tokens=doc_tokens,
        raw_tokens=raw_tokens,
        clean_tokens=clean_tokens,
        doc_ids=question.get("doc_ids", []),
        top_k=top_k,
        clean_weight=clean_weight,
    )
    raw_ids = [hit.unit_id for hit in raw_hits]
    clean_ids = [hit.unit_id for hit in clean_hits]
    return {
        "option": option_key,
        "query": query,
        "raw_query_tokens": raw_tokens,
        "clean_query_tokens": clean_tokens,
        "removed_query_tokens": [token for token in raw_tokens if token not in set(clean_tokens)],
        "added_query_tokens": [token for token in clean_tokens if token not in set(raw_tokens)],
        "raw_hits": [asdict(hit) for hit in raw_hits],
        "clean_hits": [asdict(hit) for hit in clean_hits],
        "top1_changed": bool(raw_ids and clean_ids and raw_ids[0] != clean_ids[0]),
        "topk_overlap": len(set(raw_ids) & set(clean_ids)),
        "raw_top_ids": raw_ids,
        "clean_top_ids": clean_ids,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    option_count = 0
    top1_changed = 0
    overlap_total = 0
    removed_total = 0
    changed_examples = []
    for question in traces:
        for option in question["options"]:
            option_count += 1
            top1_changed += int(option["top1_changed"])
            overlap_total += option["topk_overlap"]
            removed_total += len(option["removed_query_tokens"])
            if option["top1_changed"] and len(changed_examples) < 14:
                changed_examples.append(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "raw_top1": option["raw_top_ids"][0] if option["raw_top_ids"] else "",
                        "clean_top1": option["clean_top_ids"][0] if option["clean_top_ids"] else "",
                        "removed": option["removed_query_tokens"][:20],
                    }
                )
    return {
        "question_count": len(traces),
        "option_count": option_count,
        "top1_changed_options": top1_changed,
        "average_topk_overlap": round(overlap_total / option_count, 3) if option_count else 0,
        "average_removed_tokens": round(removed_total / option_count, 3) if option_count else 0,
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
                "removed_query_tokens",
                "added_query_tokens",
                "raw_top1",
                "clean_top1",
                "raw_score",
                "clean_score",
                "raw_preview",
                "clean_preview",
            ],
        )
        writer.writeheader()
        for question in traces:
            for option in question["options"]:
                raw_top = option["raw_hits"][0] if option["raw_hits"] else {}
                clean_top = option["clean_hits"][0] if option["clean_hits"] else {}
                writer.writerow(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "top1_changed": option["top1_changed"],
                        "topk_overlap": option["topk_overlap"],
                        "removed_query_tokens": " ".join(option["removed_query_tokens"]),
                        "added_query_tokens": " ".join(option["added_query_tokens"]),
                        "raw_top1": raw_top.get("unit_id", ""),
                        "clean_top1": clean_top.get("unit_id", ""),
                        "raw_score": raw_top.get("score", ""),
                        "clean_score": clean_top.get("score", ""),
                        "raw_preview": raw_top.get("text_preview", ""),
                        "clean_preview": clean_top.get("text_preview", ""),
                    }
                )


def pill(text: str, cls: str = "") -> str:
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def hit_html(hit: dict[str, Any], peer_ids: set[str], peer_top: str) -> str:
    unit_id = hit.get("unit_id", "")
    cls = "hit changed"
    if unit_id == peer_top:
        cls = "hit same-top"
    elif unit_id in peer_ids:
        cls = "hit overlap"
    matches = "".join(pill(str(token), "match") for token in hit.get("matched_tokens", [])[:30])
    return f"""
      <article class="{cls}">
        <header><b>#{hit.get('rank')} {hit.get('score')}</b><code>{html.escape(unit_id)}</code></header>
        <div class="doc">{html.escape(hit.get('doc_id', ''))}</div>
        <p>{html.escape(hit.get('text_preview', ''))}</p>
        <div>{matches}</div>
      </article>
    """


def build_html_report(path: Path, payload: dict[str, Any], case_limit: int = 18) -> None:
    cases = []
    for question in payload["traces"]:
        for option in question["options"]:
            if option["top1_changed"] or option["topk_overlap"] < payload["top_k"]:
                cases.append((question, option))
    cases = cases[:case_limit]
    sections = []
    for question, option in cases:
        raw_ids = set(option["raw_top_ids"])
        clean_ids = set(option["clean_top_ids"])
        raw_top = option["raw_top_ids"][0] if option["raw_top_ids"] else ""
        clean_top = option["clean_top_ids"][0] if option["clean_top_ids"] else ""
        removed = set(option["removed_query_tokens"])
        raw_tokens = "".join(pill(token, "removed" if token in removed else "") for token in option["raw_query_tokens"])
        clean_tokens = "".join(pill(token, "kept") for token in option["clean_query_tokens"])
        sections.append(
            f"""
            <section class="case">
              <div class="head">
                <div>
                  <h2>{html.escape(question['qid'])} / {html.escape(option['option'])}</h2>
                  <p>{html.escape(question['question'])}</p>
                  <p class="opt">{html.escape(option['query'].splitlines()[-1])}</p>
                </div>
                <dl>
                  <div><dt>top1 changed</dt><dd>{option['top1_changed']}</dd></div>
                  <div><dt>overlap</dt><dd>{option['topk_overlap']}</dd></div>
                </dl>
              </div>
              <div class="tokens">
                <div><h3>Raw Query Tokens</h3>{raw_tokens}</div>
                <div><h3>Clean Query Tokens</h3>{clean_tokens}</div>
              </div>
              <div class="cols">
                <div><h3>Raw Query</h3>{''.join(hit_html(h, clean_ids, clean_top) for h in option['raw_hits'])}</div>
                <div><h3>Clean Query</h3>{''.join(hit_html(h, raw_ids, raw_top) for h in option['clean_hits'])}</div>
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regulatory Query Cleaner Diff</title>
  <style>
    :root {{ --ink:#17202f; --muted:#667085; --line:#d9e0ea; --bg:#f6f7f9; --same:#eaf2ff; --warn:#fff2bd; --removed:#ffe1dd; --kept:#dff5e5; --accent:#176b87; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
    header.page {{ position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid var(--line); padding:24px 34px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    header.page p {{ margin:0; color:var(--muted); }}
    main {{ padding:20px 34px 42px; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)); gap:10px; margin-bottom:18px; }}
    .metric {{ background:#fff; border:1px solid var(--line); border-radius:7px; padding:10px 12px; }}
    .metric b {{ display:block; font-size:22px; }}
    .metric span {{ color:var(--muted); font-size:12px; }}
    .case {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:22px; }}
    .head {{ display:grid; grid-template-columns:1fr 240px; gap:16px; padding:16px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0 0 8px; font-size:18px; }}
    p {{ line-height:1.6; margin:0 0 8px; }}
    .opt {{ color:var(--accent); font-weight:700; }}
    dl {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:0; }}
    dl div {{ border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfe; }}
    dt {{ color:var(--muted); font-size:12px; }}
    dd {{ margin:4px 0 0; font-weight:700; }}
    .tokens {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:14px 16px; border-bottom:1px solid var(--line); }}
    h3 {{ margin:0 0 8px; font-size:15px; }}
    .pill {{ display:inline-block; border:1px solid var(--line); border-radius:5px; padding:2px 6px; margin:2px; font-size:12px; background:#fff; }}
    .removed {{ background:var(--removed); text-decoration:line-through; }}
    .kept {{ background:var(--kept); }}
    .match {{ background:#eef6f8; }}
    .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px; }}
    .hit {{ border:1px solid var(--line); border-left-width:5px; border-radius:6px; padding:10px 12px; margin-bottom:10px; }}
    .hit.same-top {{ border-left-color:#7aa7e8; background:#f8fbff; }}
    .hit.overlap {{ border-left-color:var(--same); }}
    .hit.changed {{ border-left-color:var(--warn); background:#fffdf3; }}
    .hit header {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    code {{ font-size:12px; color:#475467; word-break:break-all; }}
    .doc {{ color:var(--muted); font-size:12px; margin-top:6px; word-break:break-all; }}
    .hit p {{ font-size:13px; }}
    @media (max-width:1000px) {{ header.page,main {{ padding-left:16px; padding-right:16px; }} .summary,.head,.tokens,.cols {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header class="page">
    <h1>Regulatory Query Cleaner Diff</h1>
    <p>Selected-term raw query vs stopword-cleaned query without query fusion.</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><b>{payload['summary']['question_count']}</b><span>questions</span></div>
      <div class="metric"><b>{payload['summary']['option_count']}</b><span>options</span></div>
      <div class="metric"><b>{payload['summary']['top1_changed_options']}</b><span>top1 changed</span></div>
      <div class="metric"><b>{payload['summary']['average_topk_overlap']}</b><span>avg top-k overlap</span></div>
    </section>
    {''.join(sections)}
  </main>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regulatory BM25 raw query vs stopword-cleaned query.")
    parser.add_argument("--index-path", type=Path, default=ROOT / "artifacts/index/regulatory/index.json")
    parser.add_argument("--questions-path", type=Path, default=ROOT / "test/regulatory_questions.json")
    parser.add_argument("--terms-path", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/regulatory_terms_selected.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_cleaner_compare.json")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_cleaner_compare_summary.csv")
    parser.add_argument("--html", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_cleaner_diff_cases.html")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--clean-weight", type=float, default=0.25)
    args = parser.parse_args()

    index_payload = json.loads(args.index_path.read_text(encoding="utf-8"))
    questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
    units = index_payload["units"]
    selected_terms = load_terms(args.terms_path)
    raw_tokenize = make_tokenizer(selected_terms)
    clean_tokenize = make_soft_clean_query_tokenizer(raw_tokenize)
    doc_tokens = [raw_tokenize(unit_text(unit)) for unit in units]
    bm25 = BM25Index(doc_tokens)

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
                    bm25=bm25,
                    doc_tokens=doc_tokens,
                    raw_tokenize=raw_tokenize,
                    clean_tokenize=clean_tokenize,
                    top_k=args.top_k,
                    clean_weight=args.clean_weight,
                )
            )
        traces.append({"qid": question["qid"], "question": question["question"], "doc_ids": question.get("doc_ids", []), "options": options})

    payload = {
        "index_path": str(args.index_path.resolve()),
        "questions_path": str(args.questions_path.resolve()),
        "terms_path": str(args.terms_path.resolve()),
        "top_k": args.top_k,
        "clean_weight": args.clean_weight,
        "summary": summarize(traces),
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(args.summary_csv, traces)
    build_html_report(args.html, payload)
    print(args.output)
    print(args.summary_csv)
    print(args.html)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
