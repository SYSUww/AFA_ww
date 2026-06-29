#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.text_utils import normalize_whitespace


DOMAINS = [
    "regulatory",
    "financial_reports",
    "insurance",
    "research",
    "financial_contracts",
]

IMAGE_SUMMARY_WORDS = {
    "bar",
    "line",
    "pie",
    "flowchart",
    "stacked bar chart",
    "figure",
    "image",
    "chart",
}

NAV_LINE_RE = re.compile(
    r"^(首页|当前位置|English|移动端|微博|微信|返回顶部|机构概况|新闻发布|政务信息|办事服务|互动交流|统计信息|专题专栏|主动公开目录|按主题查看|无障碍|>)$",
    re.I,
)
TAG_RE = re.compile(r"<[^>]{1,120}>")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_files(domain_root: Path) -> list[Path]:
    suffixes = {".md", ".txt", ".html", ".htm"}
    files = []
    for path in sorted(domain_root.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        if path.name.endswith(".clean_meta.json"):
            continue
        files.append(path)
    return files


def doc_id_from_path(path: Path) -> str:
    suffix = path.suffix
    return path.name[: -len(suffix)] if suffix else path.stem


def convert_tables(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "html.parser")
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows))
    return str(soup)


def strip_details_wrappers(text: str) -> str:
    text = re.sub(r"</?details[^>]*>", "\n", text, flags=re.I)

    def replace_summary(match: re.Match[str]) -> str:
        label = BeautifulSoup(match.group(1), "html.parser").get_text(" ", strip=True)
        if label.strip().lower() in IMAGE_SUMMARY_WORDS:
            return "\n"
        return f"\n{label}\n" if label else "\n"

    return re.sub(r"<summary[^>]*>(.*?)</summary>", replace_summary, text, flags=re.I | re.S)


def drop_mermaid_blocks(text: str) -> str:
    return re.sub(r"```mermaid\s+.*?```", "\n", text, flags=re.I | re.S)


def extract_html_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "form", "header", "footer", "nav", "img"]):
        tag.decompose()
    selectors = [".detail-news", ".TRS_Editor", "#zoom", ".content", "article"]
    for selector in selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        text = node.get_text("\n", strip=True)
        if len(text) >= 80:
            return text
    return soup.get_text("\n", strip=True)


def remove_residual_tags(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript", "form", "header", "footer", "img"]):
        tag.decompose()
    return soup.get_text("\n")


def repair_numeric_breaks(text: str) -> str:
    text = re.sub(r"〔\s*\n?\s*(\d{4})\s*\n?\s*〕\s*\n?\s*(\d+)\s*\n?\s*号", r"〔\1〕\2号", text)
    text = re.sub(r"(?<=\d)\s*\n\s*(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s*\n\s*(?=(年|月|日|号|元|万元|亿元|%|％|股|份|倍|个|百分点))", "", text)
    text = re.sub(r"([年月日])\s*\n\s*(?=\d)", r"\1", text)
    text = re.sub(r"(人民币|美元|港元|金额|余额|收入|利润|保费|保额)\s*\n\s*(?=\d)", r"\1", text)
    text = re.sub(r"([（(《])\s*\n\s*", r"\1", text)
    text = re.sub(r"\s*\n\s*([,，。；;：:、）)》])", r"\1", text)
    text = re.sub(r"([,，、:：])\s*\n\s*", r"\1", text)
    return text


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if NAV_LINE_RE.match(line):
            continue
        if line.startswith("![") or line.lower().startswith("<img"):
            continue
        if line in {"|", "||", "---"}:
            continue
        lines.append(line)
    return lines


def normalize_extracted_text(raw: str, *, source_suffix: str) -> tuple[str, dict[str, int]]:
    before = {
        "html_tag_count": len(TAG_RE.findall(raw)),
        "markdown_image_count": raw.count("!["),
        "broken_number_count": len(re.findall(r"\d\s*\n\s*\d", raw)),
    }
    text = raw
    if source_suffix.lower() in {".html", ".htm"}:
        text = extract_html_main_text(text)
    else:
        text = drop_mermaid_blocks(text)
        text = strip_details_wrappers(text)
        text = convert_tables(text)
        text = remove_residual_tags(text)
    text = repair_numeric_breaks(text)
    lines = clean_lines(text)
    normalized = normalize_whitespace("\n".join(lines))
    after = {
        "html_tag_count": len(TAG_RE.findall(normalized)),
        "markdown_image_count": normalized.count("!["),
        "broken_number_count": len(re.findall(r"\d\s*\n\s*\d", normalized)),
    }
    stats = {f"before_{key}": value for key, value in before.items()}
    stats.update({f"after_{key}": value for key, value in after.items()})
    stats["before_chars"] = len(raw)
    stats["after_chars"] = len(normalized)
    return normalized, stats


def preprocess_domain(domain: str, extracted_root: Path, output_root: Path) -> dict[str, Any]:
    domain_root = extracted_root / domain
    output_dir = output_root / domain
    text_dir = output_dir / "cleaned_text"
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    stats_counter: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    files = collect_files(domain_root)
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned, stats = normalize_extracted_text(raw, source_suffix=path.suffix)
        suffix_counts[path.suffix.lower()] += 1
        stats_counter.update(stats)
        rel = path.relative_to(domain_root)
        doc_id = doc_id_from_path(path)
        out_name = rel.as_posix().replace("/", "__") + ".txt"
        out_path = text_dir / out_name
        out_path.write_text(cleaned, encoding="utf-8")
        documents.append(
            {
                "doc_id": doc_id,
                "domain": domain,
                "source_path": str(path),
                "cleaned_path": str(out_path),
                "source_suffix": path.suffix.lower(),
                "source_relpath": str(rel),
                "before_chars": stats["before_chars"],
                "after_chars": stats["after_chars"],
                "before_html_tag_count": stats["before_html_tag_count"],
                "after_html_tag_count": stats["after_html_tag_count"],
                "before_broken_number_count": stats["before_broken_number_count"],
                "after_broken_number_count": stats["after_broken_number_count"],
            }
        )
    summary = {
        "domain": domain,
        "document_count": len(documents),
        "suffix_counts": dict(suffix_counts),
        "totals": dict(stats_counter),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "documents.json", documents)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--extracted-root", default=str(ROOT / "artifacts" / "extracted_cleaned"))
    parser.add_argument("--output-root", default=str(ROOT / "artifacts" / "preprocessed_loop"))
    args = parser.parse_args()

    domains = DOMAINS if args.domains == ["all"] else args.domains
    extracted_root = Path(args.extracted_root)
    output_root = Path(args.output_root)
    summaries = {}
    for domain in domains:
        summaries[domain] = preprocess_domain(domain, extracted_root, output_root)
    write_json(output_root / "summary.json", summaries)
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
