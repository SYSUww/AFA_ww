#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.bm25 import BM25Index
from compare_regulatory_bm25_terms import SearchHit, load_terms, make_tokenizer, unit_text


TEMPLATE_PATTERNS = [
    r"^判断题[:：]",
    r"^(关于|结合|依据|根据|按照)",
    r"下列(说法|哪项表述|描述)?(中)?(正确的有|正确的是|符合监管要求|符合规定的有哪些|准确的有)?[？?]?$",
    r"以下(说法)?(正确的有|正确的是)?[？?]?$",
    r"相关(监管)?规定",
    r"现行规定",
    r"说法(中)?",
    r"描述",
    r"表述",
    r"准确",
    r"正确",
    r"符合",
    r"要求",
]

GENERIC_TOKENS = {
    "关于",
    "结合",
    "依据",
    "根据",
    "按照",
    "相关",
    "监管",
    "规定",
    "说法",
    "正确",
    "符合",
    "要求",
    "下列",
    "以下",
    "哪项",
    "表述",
    "描述",
    "准确",
    "判断题",
    "现行",
}

IMPORTANT_PHRASES = [
    "受益所有人",
    "客户尽职调查",
    "客户身份资料",
    "交易记录保存",
    "可疑交易报告",
    "大额交易报告",
    "反洗钱",
    "数据安全",
    "核心数据",
    "重要数据",
    "风险评估报告",
    "银行卡清算机构",
    "非银行支付机构",
    "高级管理人员",
    "董事会",
    "股东会",
    "股东大会",
    "定期报告",
    "年度报告",
    "半年度报告",
    "现金分红",
    "利润分配",
    "信息披露",
    "上市公司信息披露",
    "行政处罚",
    "处罚时效",
    "市场禁入",
    "证券公司分类评价",
    "分类评价",
    "分类监管",
    "违法违规",
    "重大违法违规",
    "重大资产重组",
    "关联交易",
    "直接负责的主管人员",
    "注册会计师",
    "会计师事务所",
    "审计报告",
    "审计委员会",
    "审议通过",
    "不得披露",
    "提交差异报告",
    "差异报告",
    "终止业务关系",
    "核实身份",
    "报告时限",
    "施行日期",
    "生效时点",
]

ACTION_PATTERNS = [
    r"(应当|不得|无需|需要|可以|可能|必须|原则上|至少|同时|仅需|无需)[^，。；;、]{2,28}",
    r"[^，。；;、]{2,16}(报告|披露|提交|核实|保存|终止|审议|批准|公示|处罚|扣减|禁入|施行|废止)",
]

NUMBER_PATTERN = re.compile(
    r"(?:\d+|[一二三四五六七八九十百千万]+)\s*(?:个工作日|自然日|个月|年|日|万元|美元|元)"
    r"|20\d{2}\s*年\s*\d+\s*月\s*\d+\s*日"
)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    for pattern in TEMPLATE_PATTERNS:
        text = re.sub(pattern, "", text)
    text = text.replace("？", "").replace("?", "")
    return text


def titles_from_text(text: str) -> list[str]:
    titles = []
    seen = set()
    for title in re.findall(r"《([^》]+)》", text):
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def short_doc_hint(doc_id: str) -> str:
    match = re.search(r"（([^（）]+)）", doc_id)
    if match:
        return match.group(1)
    return doc_id


def phrases_in_text(text: str, selected_terms: list[str]) -> list[str]:
    phrases = []
    seen = set()
    for phrase in IMPORTANT_PHRASES + selected_terms:
        if len(phrase) < 3:
            continue
        if phrase in text and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    phrases.sort(key=lambda item: (-len(item), item))
    return phrases[:24]


def extract_fact_chunks(text: str) -> list[str]:
    chunks = []
    seen = set()
    for pattern in ACTION_PATTERNS:
        for match in re.finditer(pattern, text):
            chunk = match.group(0).strip("，。；;、 ")
            if re.match(r"^\d", chunk):
                continue
            if len(chunk) < 3 or chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
    for match in NUMBER_PATTERN.finditer(text):
        chunk = match.group(0).replace(" ", "")
        if chunk not in seen:
            seen.add(chunk)
            chunks.append(chunk)
    return chunks[:18]


def filtered_tokens(text: str, tokenize) -> list[str]:
    tokens = []
    for token in tokenize(text):
        if token in GENERIC_TOKENS:
            continue
        if len(token) == 1 and not re.fullmatch(r"\d", token):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", token) and token in {"下列", "以下", "相关", "规定", "说法"}:
            continue
        tokens.append(token)
    return tokens


def build_queries(question: dict[str, Any], option_text: str, tokenize, selected_terms: list[str]) -> dict[str, Any]:
    q_text = question["question"]
    clean_q = normalize_text(q_text)
    if question.get("answer_format") == "tf" or option_text.strip() in {"正确", "错误"}:
        clean_option = clean_q
    else:
        clean_option = normalize_text(option_text)
    titles = titles_from_text(q_text) + titles_from_text(option_text)
    if not titles:
        titles = [short_doc_hint(doc_id) for doc_id in question.get("doc_ids", [])]
    titles = list(dict.fromkeys(titles))
    phrase_source = clean_q + " " + clean_option
    concepts = phrases_in_text(phrase_source, selected_terms)
    facts = extract_fact_chunks(clean_option)
    fact_tokens = filtered_tokens(clean_option, tokenize)
    concept_query = " ".join(concepts)
    fact_query = " ".join(facts + fact_tokens[:40])
    title_query = " ".join(titles)
    compact_query = " ".join([concept_query, fact_query]).strip()
    return {
        "title_query": title_query,
        "concept_query": concept_query,
        "fact_query": fact_query,
        "compact_query": compact_query,
        "facts": facts,
        "concepts": concepts,
        "titles": titles,
    }


def apply_unit_weight(unit: dict[str, Any], score: float, query: str) -> float:
    unit_type = unit.get("unit_type", "")
    if unit_type == "preamble":
        score *= 0.55
    if unit_type == "article_chunk":
        score *= 1.03
    if query and len(query) < 18 and unit_type == "preamble":
        score *= 0.75
    return score


def score_query_tokens(
    *,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    doc_ids: list[str],
    query_tokens: list[str],
    top_k: int,
) -> list[SearchHit]:
    doc_filter = set(doc_ids)
    scored = []
    for idx, unit in enumerate(units):
        if doc_filter and unit["doc_id"] not in doc_filter:
            continue
        score = bm25.score(query_tokens, idx)
        score = apply_unit_weight(unit, score, " ".join(query_tokens))
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
    return hits


def score_builder(
    *,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    tokenize,
    doc_ids: list[str],
    queries: dict[str, Any],
    top_k: int,
) -> tuple[list[str], list[SearchHit]]:
    weighted_queries = [
        (queries["fact_query"], 1.45),
        (queries["concept_query"], 0.85),
        (queries["compact_query"], 0.35),
        (queries["title_query"], 0.08),
    ]
    combined: dict[int, float] = Counter()
    query_tokens_all = []
    doc_filter = set(doc_ids)
    for query, weight in weighted_queries:
        if not query.strip():
            continue
        tokens = filtered_tokens(query, tokenize)
        query_tokens_all.extend(tokens)
        for idx, unit in enumerate(units):
            if doc_filter and unit["doc_id"] not in doc_filter:
                continue
            score = apply_unit_weight(unit, bm25.score(tokens, idx), query)
            if score > 0:
                combined[idx] += score * weight
    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    unique_tokens = list(dict.fromkeys(query_tokens_all))
    hits = []
    for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
        unit = units[idx]
        matched_tokens = [token for token in unique_tokens if token in set(doc_tokens[idx])]
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
    return unique_tokens, hits


def compare_option(
    *,
    question: dict[str, Any],
    option_key: str,
    option_text: str,
    units: list[dict[str, Any]],
    bm25: BM25Index,
    doc_tokens: list[list[str]],
    tokenize,
    selected_terms: list[str],
    top_k: int,
) -> dict[str, Any]:
    raw_query = f"{question['question']}\n{option_text}"
    baseline_tokens = tokenize(raw_query)
    baseline_hits = score_query_tokens(
        units=units,
        bm25=bm25,
        doc_tokens=doc_tokens,
        doc_ids=question.get("doc_ids", []),
        query_tokens=baseline_tokens,
        top_k=top_k,
    )
    queries = build_queries(question, option_text, tokenize, selected_terms)
    builder_tokens, builder_hits = score_builder(
        units=units,
        bm25=bm25,
        doc_tokens=doc_tokens,
        tokenize=tokenize,
        doc_ids=question.get("doc_ids", []),
        queries=queries,
        top_k=top_k,
    )
    baseline_ids = [hit.unit_id for hit in baseline_hits]
    builder_ids = [hit.unit_id for hit in builder_hits]
    return {
        "option": option_key,
        "raw_query": raw_query,
        "query_parts": queries,
        "baseline_query_tokens": baseline_tokens,
        "builder_query_tokens": builder_tokens,
        "removed_query_tokens": [token for token in baseline_tokens if token not in set(builder_tokens)],
        "added_query_tokens": [token for token in builder_tokens if token not in set(baseline_tokens)],
        "baseline_hits": [asdict(hit) for hit in baseline_hits],
        "builder_hits": [asdict(hit) for hit in builder_hits],
        "top1_changed": bool(baseline_ids and builder_ids and baseline_ids[0] != builder_ids[0]),
        "topk_overlap": len(set(baseline_ids) & set(builder_ids)),
        "baseline_top_ids": baseline_ids,
        "builder_top_ids": builder_ids,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    option_count = 0
    top1_changed = 0
    overlap_total = 0
    changed_examples = []
    for question in traces:
        for option in question["options"]:
            option_count += 1
            top1_changed += int(option["top1_changed"])
            overlap_total += option["topk_overlap"]
            if option["top1_changed"] and len(changed_examples) < 16:
                changed_examples.append(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "baseline_top1": option["baseline_top_ids"][0] if option["baseline_top_ids"] else "",
                        "builder_top1": option["builder_top_ids"][0] if option["builder_top_ids"] else "",
                        "concepts": option["query_parts"]["concepts"],
                        "facts": option["query_parts"]["facts"],
                    }
                )
    return {
        "question_count": len(traces),
        "option_count": option_count,
        "top1_changed_options": top1_changed,
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
                "titles",
                "concepts",
                "facts",
                "baseline_top1",
                "builder_top1",
                "baseline_score",
                "builder_score",
                "baseline_preview",
                "builder_preview",
            ],
        )
        writer.writeheader()
        for question in traces:
            for option in question["options"]:
                baseline_top = option["baseline_hits"][0] if option["baseline_hits"] else {}
                builder_top = option["builder_hits"][0] if option["builder_hits"] else {}
                parts = option["query_parts"]
                writer.writerow(
                    {
                        "qid": question["qid"],
                        "option": option["option"],
                        "top1_changed": option["top1_changed"],
                        "topk_overlap": option["topk_overlap"],
                        "titles": " | ".join(parts["titles"]),
                        "concepts": " | ".join(parts["concepts"]),
                        "facts": " | ".join(parts["facts"]),
                        "baseline_top1": baseline_top.get("unit_id", ""),
                        "builder_top1": builder_top.get("unit_id", ""),
                        "baseline_score": baseline_top.get("score", ""),
                        "builder_score": builder_top.get("score", ""),
                        "baseline_preview": baseline_top.get("text_preview", ""),
                        "builder_preview": builder_top.get("text_preview", ""),
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
        parts = option["query_parts"]
        base_ids = set(option["baseline_top_ids"])
        builder_ids = set(option["builder_top_ids"])
        base_top = option["baseline_top_ids"][0] if option["baseline_top_ids"] else ""
        builder_top = option["builder_top_ids"][0] if option["builder_top_ids"] else ""
        section = f"""
        <section class="case">
          <div class="head">
            <div>
              <h2>{html.escape(question['qid'])} / {html.escape(option['option'])}</h2>
              <p>{html.escape(question['question'])}</p>
              <p class="opt">{html.escape(option['raw_query'].splitlines()[-1])}</p>
            </div>
            <dl>
              <div><dt>top1 changed</dt><dd>{option['top1_changed']}</dd></div>
              <div><dt>overlap</dt><dd>{option['topk_overlap']}</dd></div>
            </dl>
          </div>
          <div class="parts">
            <div><h3>Titles</h3>{''.join(pill(x, 'title') for x in parts['titles'])}</div>
            <div><h3>Concepts</h3>{''.join(pill(x, 'concept') for x in parts['concepts'])}</div>
            <div><h3>Facts</h3>{''.join(pill(x, 'fact') for x in parts['facts'])}</div>
          </div>
          <div class="cols">
            <div><h3>Baseline</h3>{''.join(hit_html(h, builder_ids, builder_top) for h in option['baseline_hits'])}</div>
            <div><h3>Query Builder</h3>{''.join(hit_html(h, base_ids, base_top) for h in option['builder_hits'])}</div>
          </div>
        </section>
        """
        sections.append(section)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regulatory Query Builder Diff</title>
  <style>
    :root {{ --ink:#17202f; --muted:#667085; --line:#d9e0ea; --bg:#f6f7f9; --add:#dff5e5; --warn:#fff2bd; --same:#eaf2ff; --accent:#176b87; }}
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
    .parts {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; padding:14px 16px; border-bottom:1px solid var(--line); }}
    h3 {{ margin:0 0 8px; font-size:15px; }}
    .pill {{ display:inline-block; border:1px solid var(--line); border-radius:5px; padding:2px 6px; margin:2px; font-size:12px; background:#fff; }}
    .title {{ background:#edf4ff; }}
    .concept {{ background:var(--add); font-weight:700; }}
    .fact {{ background:#fff7d6; }}
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
    @media (max-width:1000px) {{ header.page,main {{ padding-left:16px; padding-right:16px; }} .summary,.head,.parts,.cols {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header class="page">
    <h1>Regulatory Query Builder Diff</h1>
    <p>Baseline raw question+option vs cleaned title/concept/fact query fusion.</p>
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
    parser = argparse.ArgumentParser(description="Compare baseline regulatory BM25 with a structured query builder.")
    parser.add_argument("--index-path", type=Path, default=ROOT / "artifacts/index/regulatory/index.json")
    parser.add_argument("--questions-path", type=Path, default=ROOT / "test/regulatory_questions.json")
    parser.add_argument("--terms-path", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/regulatory_terms_selected.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_builder_compare.json")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_builder_compare_summary.csv")
    parser.add_argument("--html", type=Path, default=ROOT / "artifacts/preprocessed/regulatory/terms/query_builder_diff_cases.html")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    index_payload = json.loads(args.index_path.read_text(encoding="utf-8"))
    questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    units = index_payload["units"]
    selected_terms = load_terms(args.terms_path)
    tokenize = make_tokenizer(selected_terms)
    doc_tokens = [tokenize(unit_text(unit)) for unit in units]
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
                    tokenize=tokenize,
                    selected_terms=selected_terms,
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
        "top_k": args.top_k,
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
