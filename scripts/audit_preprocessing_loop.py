#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.text_utils import tokenize_zh


DOMAINS = [
    "regulatory",
    "financial_reports",
    "insurance",
    "research",
    "financial_contracts",
]

DOMAIN_MARKERS = {
    "regulatory": [
        "第",
        "条",
        "办法",
        "规定",
        "决定",
        "行政处罚",
        "市场禁入",
        "施行",
        "违法",
    ],
    "financial_reports": [
        "营业收入",
        "净利润",
        "现金流量",
        "研发投入",
        "资产总计",
        "同比",
        "分红",
        "报告期",
    ],
    "insurance": [
        "保险责任",
        "身故保险金",
        "现金价值",
        "账户价值",
        "基本保额",
        "已交保费",
        "退保",
        "年金",
    ],
    "research": [
        "预计",
        "同比",
        "市场规模",
        "渗透率",
        "增速",
        "结论",
        "投资建议",
        "风险提示",
    ],
    "financial_contracts": [
        "发行人",
        "发行金额",
        "主体信用评级",
        "债项信用评级",
        "受托管理人",
        "主承销商",
        "回售",
        "赎回",
        "违约",
    ],
}

EXPECTED_UNIT_TYPES = {
    "regulatory": {"article", "article_chunk", "preamble", "penalty_decision", "penalty_fact", "penalty_basis", "penalty_decision_item"},
    "financial_reports": {"paragraph", "metric_row"},
    "insurance": {"clause_block", "formula_block"},
    "research": {"paragraph", "conclusion_block"},
    "financial_contracts": {"paragraph", "element_block"},
}

NOISE_PATTERNS = {
    "page_marker": re.compile(r"\[PAGE\s+\d+\]"),
    "markdown_image": re.compile(r"!\[[^\]]*]\([^)]+\)"),
    "html_tag": re.compile(r"<[^>]{1,80}>"),
    "table_pipe": re.compile(r"^\s*\|.+\|\s*$"),
    "toc_line": re.compile(r"^\s*(目录|目\s+录|contents)\s*$", re.I),
    "nav_line": re.compile(r"(首页|当前位置|English|移动端|微博|微信)"),
    "broken_number": re.compile(r"\d\s*\n\s*\d"),
}


@dataclass
class TextFile:
    doc_id: str
    path: Path
    text: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return float(ordered[index])


def short_hash(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def doc_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".md"):
        return name[:-3]
    if name.endswith(".html"):
        return name[:-5]
    if name.endswith(".txt"):
        return name[:-4]
    return path.stem


def collect_text_files(domain: str, extracted_root: Path) -> list[TextFile]:
    root = extracted_root / domain
    if not root.exists():
        return []
    suffixes = {".md", ".html", ".txt"}
    files: list[TextFile] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        if path.name.endswith(".clean_meta.json"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        files.append(TextFile(doc_id_from_path(path), path, text))
    return files


def summarize_lengths(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "avg": 0, "p90": 0, "p99": 0, "max": 0}
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "avg": round(statistics.mean(values), 2),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def line_shape_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            counts["blank"] += 1
        elif stripped.startswith("#"):
            counts["heading_md"] += 1
        elif re.match(r"^第[一二三四五六七八九十百千万零〇两\d]+[章节条]", stripped):
            counts["legal_heading"] += 1
        elif re.match(r"^\d+(\.\d+){0,4}\s+", stripped):
            counts["numbered_heading"] += 1
        elif stripped.startswith("|") and stripped.endswith("|"):
            counts["table_row"] += 1
        elif re.search(r"\d", stripped) and any(unit in stripped for unit in ["元", "万元", "%", "亿元", "年", "月", "日"]):
            counts["numeric_fact"] += 1
        else:
            counts["plain"] += 1
    return counts


def source_profile(files: list[TextFile], extracted_root: Path) -> dict[str, Any]:
    char_lengths = [len(item.text) for item in files]
    line_lengths = [len(line.strip()) for item in files for line in item.text.splitlines() if line.strip()]
    suffix_counts = Counter(item.path.suffix.lower() for item in files)
    subdir_counts = Counter(str(item.path.parent.relative_to(extracted_root)) for item in files)
    shape_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    duplicate_hashes: Counter[str] = Counter()
    samples = []
    for item in files:
        shape_counts.update(line_shape_counts(item.text))
        normalized = "\n".join(line.strip() for line in item.text.splitlines() if line.strip())
        duplicate_hashes[short_hash(normalized[:5000])] += 1
        for name, pattern in NOISE_PATTERNS.items():
            hits = pattern.findall(item.text)
            if hits:
                marker_counts[name] += len(hits)
        for keyword in DOMAIN_MARKERS.get(item.path.parts[-2] if len(item.path.parts) >= 2 else "", []):
            if keyword in item.text:
                keyword_counts[keyword] += 1
        if len(samples) < 3:
            samples.append(
                {
                    "doc_id": item.doc_id,
                    "path": str(item.path.relative_to(ROOT)),
                    "chars": len(item.text),
                    "first_lines": [line.strip() for line in item.text.splitlines() if line.strip()][:8],
                }
            )
    duplicate_groups = sum(1 for count in duplicate_hashes.values() if count > 1)
    return {
        "file_count": len(files),
        "suffix_counts": dict(suffix_counts),
        "subdir_counts": dict(subdir_counts),
        "char_lengths": summarize_lengths(char_lengths),
        "non_empty_line_lengths": summarize_lengths(line_lengths),
        "line_shapes": dict(shape_counts.most_common()),
        "noise_markers": dict(marker_counts.most_common()),
        "duplicate_prefix_groups": duplicate_groups,
        "samples": samples,
    }


def domain_source_profile(domain: str, files: list[TextFile], extracted_root: Path) -> dict[str, Any]:
    profile = source_profile(files, extracted_root)
    keyword_counts: Counter[str] = Counter()
    for item in files:
        for keyword in DOMAIN_MARKERS[domain]:
            if keyword in item.text:
                keyword_counts[keyword] += 1
    profile["domain_marker_doc_hits"] = dict(keyword_counts.most_common())
    return profile


def load_manifest_domain(manifest: dict[str, Any], domain: str) -> dict[str, Any]:
    return manifest["domains"].get(domain, {})


def manifest_profile(manifest_domain: dict[str, Any], source_files: list[TextFile]) -> dict[str, Any]:
    documents = manifest_domain.get("documents", {})
    referenced = set(manifest_domain.get("referenced_doc_ids", []))
    source_doc_ids = {item.doc_id for item in source_files}
    manifest_doc_ids = set(documents.keys())
    return {
        "manifest_documents": len(documents),
        "referenced_documents": len(referenced),
        "source_files": len(source_files),
        "referenced_missing_from_extracted_cleaned": sorted(referenced - source_doc_ids)[:50],
        "extracted_cleaned_not_in_manifest": sorted(source_doc_ids - manifest_doc_ids)[:50],
        "source_type_counts": dict(Counter(record.get("source_type", "") for record in documents.values())),
    }


def parsed_profile(domain: str, parsed_path: Path) -> dict[str, Any]:
    if not parsed_path.exists():
        return {"exists": False}
    parsed = read_json(parsed_path)
    documents = parsed.get("documents", [])
    units = parsed.get("units", [])
    unit_lengths = [len(unit.get("text", "")) for unit in units]
    empty_units = [unit.get("unit_id") for unit in units if not unit.get("text", "").strip()]
    unit_type_counts = Counter(unit.get("unit_type", "") for unit in units)
    by_doc: Counter[str] = Counter(unit.get("doc_id", "") for unit in units)
    duplicate_ids = [unit_id for unit_id, count in Counter(unit.get("unit_id", "") for unit in units).items() if count > 1]
    expected = EXPECTED_UNIT_TYPES.get(domain, set())
    unexpected = sorted(set(unit_type_counts) - expected)
    marker_hits: Counter[str] = Counter()
    for unit in units:
        text = unit.get("text", "")
        for name, pattern in NOISE_PATTERNS.items():
            if pattern.search(text):
                marker_hits[name] += 1
    return {
        "exists": True,
        "document_count": len(documents),
        "unit_count": len(units),
        "unit_type_counts": dict(unit_type_counts.most_common()),
        "unit_length_chars": summarize_lengths(unit_lengths),
        "empty_units": empty_units[:30],
        "duplicate_unit_ids": duplicate_ids[:30],
        "unexpected_unit_types": unexpected,
        "docs_with_no_units": sorted(set(doc.get("doc_id") for doc in documents) - set(by_doc))[:50],
        "units_per_doc": summarize_lengths(list(by_doc.values())),
        "noise_unit_hits": dict(marker_hits.most_common()),
    }


def question_profile(domain: str, manifest_domain: dict[str, Any]) -> dict[str, Any]:
    question_path = Path(manifest_domain.get("question_path", ""))
    if not question_path.exists():
        return {"exists": False}
    rows = read_json(question_path)
    a_rows = [row for row in rows if row.get("split") == "A"]
    option_counts = [len(row.get("options", {})) for row in a_rows]
    answer_formats = Counter(row.get("answer_format", "") for row in a_rows)
    doc_counts = [len(row.get("doc_ids", [])) for row in a_rows]
    query_tokens = []
    domain_marker_question_hits: Counter[str] = Counter()
    for row in a_rows:
        query = row.get("question", "") + "\n" + json.dumps(row.get("options", {}), ensure_ascii=False)
        tokens = tokenize_zh(query)
        query_tokens.append(len(tokens))
        for keyword in DOMAIN_MARKERS[domain]:
            if keyword in query:
                domain_marker_question_hits[keyword] += 1
    return {
        "exists": True,
        "question_count": len(rows),
        "group_a_count": len(a_rows),
        "answer_format_counts": dict(answer_formats),
        "option_counts": summarize_lengths(option_counts),
        "doc_ids_per_question": summarize_lengths(doc_counts),
        "query_token_counts": summarize_lengths(query_tokens),
        "domain_marker_question_hits": dict(domain_marker_question_hits.most_common()),
    }


def retrieval_profile(
    domain: str,
    index_path: Path,
    manifest_domain: dict[str, Any],
    *,
    mode: str,
    max_questions: int,
) -> dict[str, Any]:
    if mode == "off":
        return {"exists": False, "reason": "disabled"}
    if not index_path.exists():
        return {"exists": False}
    index_payload = read_json(index_path)
    units = index_payload.get("units", [])
    question_path = Path(manifest_domain.get("question_path", ""))
    if not question_path.exists():
        return {"exists": False, "reason": "missing question_path"}
    try:
        from export_bm25_retrievals import TRACE_BUILDERS, load_questions
    except Exception as exc:
        return {"exists": False, "reason": f"trace import failed: {exc}"}
    builder = TRACE_BUILDERS[domain]
    questions = load_questions(domain, "A")
    if mode == "sample":
        questions = questions[:max_questions]
    unit_by_id = {unit.get("unit_id"): unit for unit in units}
    unit_type_hits: Counter[str] = Counter()
    doc_hit_rows = 0
    total_rows = 0
    no_hit_rows = []
    multi_doc_coverages = []
    per_question = []
    for question in questions:
        trace = builder(question, index_payload)
        rows = trace.get("option_traces") or [trace]
        q_doc_hits = 0
        q_total = 0
        q_hit_docs = set()
        for row in rows:
            hits = row.get("hits", [])
            total_rows += 1
            q_total += 1
            if not hits:
                no_hit_rows.append({"qid": question.qid, "option": row.get("option", "")})
                continue
            top_docs = {hit.get("doc_id") for hit in hits if hit.get("doc_id")}
            q_hit_docs.update(top_docs)
            if set(question.doc_ids) & top_docs:
                doc_hit_rows += 1
                q_doc_hits += 1
            for hit in hits:
                unit_type = hit.get("unit_type") or unit_by_id.get(hit.get("unit_id"), {}).get("unit_type", "")
                unit_type_hits[unit_type] += 1
        if len(question.doc_ids) > 1:
            multi_doc_coverages.append(len(q_hit_docs & set(question.doc_ids)) / max(1, len(set(question.doc_ids))))
        per_question.append(
            {
                "qid": question.qid,
                "answer_format": question.answer_format,
                "trace_rows": q_total,
                "doc_hit_rate": round(q_doc_hits / max(1, q_total), 4),
                "question_docs": question.doc_ids,
                "retrieved_question_docs": sorted(q_hit_docs & set(question.doc_ids)),
            }
        )
    preferred = EXPECTED_UNIT_TYPES.get(domain, set())
    preferred_hits = sum(count for unit_type, count in unit_type_hits.items() if unit_type in preferred)
    return {
        "exists": True,
        "mode": mode,
        "question_count": len(questions),
        "trace_rows": total_rows,
        "doc_hit_rate": round(doc_hit_rows / max(1, total_rows), 4),
        "multi_doc_coverage_avg": round(statistics.mean(multi_doc_coverages), 4) if multi_doc_coverages else 1.0,
        "preferred_unit_hit_rate": round(preferred_hits / max(1, sum(unit_type_hits.values())), 4),
        "hit_unit_type_counts": dict(unit_type_hits.most_common()),
        "no_hit_rows": no_hit_rows[:50],
        "weak_questions": [row for row in per_question if row["doc_hit_rate"] < 1.0][:30],
        "per_question": per_question,
    }


def recommendations(domain: str, profile: dict[str, Any]) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    source = profile["source"]
    parsed = profile["parsed"]
    retrieval = profile["retrieval"]
    noise = source.get("noise_markers", {})
    parsed_noise = parsed.get("noise_unit_hits", {}) if parsed.get("exists") else {}
    if noise.get("markdown_image") or parsed_noise.get("markdown_image"):
        recs.append({"stage": "preprocess", "priority": "high", "action": "删除图片说明/Markdown image 行，避免图注干扰检索。"})
    if noise.get("html_tag") or parsed_noise.get("html_tag"):
        recs.append({"stage": "preprocess", "priority": "high", "action": "对 HTML 残留标签做渲染文本抽取，禁止整页 body 噪声进入正文。"})
    if noise.get("broken_number"):
        recs.append({"stage": "preprocess", "priority": "medium", "action": "合并数字、日期、金额、百分比之间的异常换行。"})
    if domain == "financial_reports" and source.get("line_shapes", {}).get("table_row", 0) > 0:
        recs.append({"stage": "preprocess", "priority": "high", "action": "把 Markdown 表格转为“指标-年份-数值-单位”结构块，保留表头。"})
    if domain == "insurance":
        recs.append({"stage": "preprocess", "priority": "medium", "action": "识别条款层级、责任/免责/现金价值公式；公式块应保留前后条件和适用年龄/期间。"})
    if domain == "research":
        recs.append({"stage": "preprocess", "priority": "medium", "action": "保留核心观点、预测假设、风险提示和图表附近数值，减少免责声明/目录噪声。"})
    if domain == "financial_contracts":
        recs.append({"stage": "preprocess", "priority": "medium", "action": "抽发行要素表、信用评级、回售赎回和违约条款为结构化 element_block。"})
    if domain == "regulatory":
        recs.append({"stage": "preprocess", "priority": "high", "action": "md/html/txt 分源处理：法规按条，处罚决定按事实/依据/决定，HTML 用正文选择器抽取。"})
    if parsed.get("unit_length_chars", {}).get("p90", 0) > 1200:
        recs.append({"stage": "segmentation", "priority": "medium", "action": "p90 unit 偏长，增加标题/条款/表格边界切分，避免证据块过宽。"})
    if retrieval.get("exists") and retrieval.get("doc_hit_rate", 1.0) < 0.95:
        recs.append({"stage": "retrieval", "priority": "high", "action": "doc_hit_rate 偏低，回看预处理是否丢标题/关键词，并调整 query_builder 与 unit_type boost。"})
    if retrieval.get("exists") and retrieval.get("multi_doc_coverage_avg", 1.0) < 0.8:
        recs.append({"stage": "retrieval", "priority": "high", "action": "跨文档题覆盖不足，检索阶段需要 ensure_per_doc 或分文档召回后合并。"})
    if not recs:
        recs.append({"stage": "preprocess", "priority": "low", "action": "当前画像未发现明显预处理阻塞点，下一轮可从错题证据回溯。"})
    return recs


def audit_domain(domain: str, args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    extracted_root = Path(args.extracted_root).resolve()
    source_files = collect_text_files(domain, extracted_root)
    manifest_domain = load_manifest_domain(manifest, domain)
    parsed_path = Path(args.parsed_root).resolve() / domain / "parsed.json"
    index_path = Path(args.index_root).resolve() / domain / "index.json"
    profile = {
        "domain": domain,
        "source": domain_source_profile(domain, source_files, extracted_root),
        "manifest": manifest_profile(manifest_domain, source_files),
        "questions": question_profile(domain, manifest_domain),
        "parsed": parsed_profile(domain, parsed_path),
        "retrieval": retrieval_profile(
            domain,
            index_path,
            manifest_domain,
            mode=args.retrieval_mode,
            max_questions=args.retrieval_sample_size,
        ),
    }
    profile["recommendations"] = recommendations(domain, profile)
    return profile


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Preprocessing Loop Audit",
        "",
        "This report treats `artifacts/extracted_cleaned` as the PDF-parsed source layer and audits each domain before downstream chunk/retrieval/prompt tuning.",
        "",
    ]
    for domain, profile in results["domains"].items():
        source = profile["source"]
        parsed = profile["parsed"]
        retrieval = profile["retrieval"]
        lines.extend(
            [
                f"## {domain}",
                "",
                f"- source files: {source['file_count']} / suffixes: {source['suffix_counts']}",
                f"- source char length p50/p90/max: {source['char_lengths']['p50']:.0f} / {source['char_lengths']['p90']:.0f} / {source['char_lengths']['max']:.0f}",
                f"- line shapes: {dict(list(source['line_shapes'].items())[:6])}",
                f"- source noise markers: {source['noise_markers']}",
            ]
        )
        if parsed.get("exists"):
            lines.extend(
                [
                    f"- parsed docs/units: {parsed['document_count']} / {parsed['unit_count']}",
                    f"- unit types: {parsed['unit_type_counts']}",
                    f"- unit length p50/p90/max: {parsed['unit_length_chars']['p50']:.0f} / {parsed['unit_length_chars']['p90']:.0f} / {parsed['unit_length_chars']['max']:.0f}",
                    f"- parsed noise hits: {parsed['noise_unit_hits']}",
                ]
            )
        else:
            lines.append("- parsed: missing")
        if retrieval.get("exists"):
            lines.extend(
                [
                    f"- retrieval doc hit rate: {retrieval['doc_hit_rate']}",
                    f"- multi-doc coverage avg: {retrieval['multi_doc_coverage_avg']}",
                    f"- preferred unit hit rate: {retrieval['preferred_unit_hit_rate']}",
                    f"- weak questions: {[row['qid'] for row in retrieval['weak_questions'][:8]]}",
                ]
            )
        else:
            lines.append(f"- retrieval: missing ({retrieval.get('reason', '')})")
        lines.append("- loop recommendations:")
        for item in profile["recommendations"]:
            lines.append(f"  - [{item['priority']}] {item['stage']}: {item['action']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--extracted-root", default=str(ROOT / "artifacts" / "extracted_cleaned"))
    parser.add_argument("--manifest-path", default=str(ROOT / "artifacts" / "manifest" / "dataset_manifest.json"))
    parser.add_argument("--parsed-root", default=str(ROOT / "artifacts" / "parsed"))
    parser.add_argument("--index-root", default=str(ROOT / "artifacts" / "index"))
    parser.add_argument("--retrieval-mode", choices=["off", "sample", "full"], default="off")
    parser.add_argument("--retrieval-sample-size", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "preprocessing_loop_audit" / "latest"))
    args = parser.parse_args()

    domains = DOMAINS if args.domains == ["all"] else args.domains
    manifest = read_json(Path(args.manifest_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "inputs": {
            "extracted_root": args.extracted_root,
            "manifest_path": args.manifest_path,
            "parsed_root": args.parsed_root,
            "index_root": args.index_root,
        },
        "domains": {},
    }
    for domain in domains:
        results["domains"][domain] = audit_domain(domain, args, manifest)

    write_json(output_dir / "audit.json", results)
    (output_dir / "report.md").write_text(render_markdown(results), encoding="utf-8")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
