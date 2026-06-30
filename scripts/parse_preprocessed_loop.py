#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.common import (
    clean_lines,
    detect_title,
    extract_page_refs,
    looks_like_heading,
    make_units_from_sections,
    split_text_into_sections,
)
from afa_agent.domains.financial_contracts.plugin import CONTRACT_KEYWORDS, FinancialContractsPlugin
from afa_agent.domains.financial_reports.plugin import FINANCIAL_METRICS, FinancialReportsPlugin
from afa_agent.domains.insurance.plugin import INSURANCE_KEYWORDS
from afa_agent.domains.regulatory.parser import split_regulatory_units
from afa_agent.domains.research.plugin import RESEARCH_KEYWORDS
from afa_agent.models import Document


DOMAINS = [
    "regulatory",
    "financial_reports",
    "insurance",
    "research",
    "financial_contracts",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_preprocessed_documents(domain: str, preprocessed_root: Path) -> list[dict[str, Any]]:
    path = preprocessed_root / domain / "documents.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def make_document(domain: str, row: dict[str, Any], text: str) -> Document:
    title = detect_title(text, row["doc_id"])
    return Document(
        doc_id=row["doc_id"],
        domain=domain,
        title=title,
        source_type="preprocessed_text",
        source_path=row["cleaned_path"],
        metadata={
            "original_source_path": row.get("source_path", ""),
            "source_suffix": row.get("source_suffix", ""),
            "source_relpath": row.get("source_relpath", ""),
            "before_chars": row.get("before_chars", 0),
            "after_chars": row.get("after_chars", len(text)),
        },
    )


def parse_regulatory(doc: Document, text: str) -> list[dict[str, Any]]:
    article_units = split_regulatory_units(doc, text, max_chars=1200)
    has_articles = any(unit.unit_type in {"article", "article_chunk"} for unit in article_units)
    if has_articles:
        return [unit.to_dict() for unit in article_units]
    sections = split_text_into_sections(
        text,
        doc.title,
        "penalty",
        unit_type="penalty_decision",
        max_chars=900,
    )
    if not sections and text.strip():
        sections = [
            {
                "section_id": "penalty_1",
                "title_path": [doc.title],
                "text": text.strip(),
                "unit_type": "penalty_decision",
                "page_refs": [],
                "max_chars": 900,
            }
        ]
    return [unit.to_dict() for unit in make_units_from_sections(doc, sections)]


def parse_financial_reports(doc: Document, text: str) -> list[dict[str, Any]]:
    sections = split_text_into_sections(text, doc.title, "sec", unit_type="paragraph", max_chars=1000)
    metric_sections = FinancialReportsPlugin()._extract_metric_sections(text, doc.title, doc.doc_id, max_chars=600)
    return [unit.to_dict() for unit in make_units_from_sections(doc, sections + metric_sections)]


def parse_insurance(doc: Document, text: str) -> list[dict[str, Any]]:
    sections = split_insurance_clause_sections(text, doc.title)
    special_sections = []
    for section in sections:
        if any(keyword in section["text"] for keyword in INSURANCE_KEYWORDS):
            section_copy = dict(section)
            section_copy["unit_type"] = (
                "formula_block"
                if any(keyword in section["text"] for keyword in ["账户价值", "已交保费", "基本保额", "现金价值"])
                else "clause_block"
            )
            special_sections.append(section_copy)
    return [unit.to_dict() for unit in make_units_from_sections(doc, sections + special_sections)]


def split_insurance_clause_sections(text: str, title: str) -> list[dict[str, Any]]:
    lines = clean_lines(text.splitlines())
    sections: list[dict[str, Any]] = []
    current_title = title
    buffer: list[str] = []
    section_index = 1
    page_refs: list[int] = []

    def flush() -> None:
        nonlocal buffer, section_index, page_refs
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                {
                    "section_id": f"sec_{section_index}",
                    "title_path": [title, current_title] if current_title != title else [title],
                    "text": body,
                    "unit_type": "clause_block",
                    "page_refs": sorted(set(page_refs)),
                    "max_chars": 900,
                }
            )
            section_index += 1
        buffer = []
        page_refs = []

    for line in lines:
        if line.startswith("[PAGE "):
            page_refs.extend(extract_page_refs(line))
            continue
        if looks_like_heading(line):
            flush()
            current_title = line[:120]
            buffer.append(line)
            continue
        buffer.append(line)
    flush()
    return sections


def parse_research(doc: Document, text: str) -> list[dict[str, Any]]:
    sections = split_text_into_sections(text, doc.title, "sec", unit_type="paragraph", max_chars=1000)
    highlights = []
    for section in sections:
        if any(keyword in section["text"] for keyword in RESEARCH_KEYWORDS):
            section_copy = dict(section)
            section_copy["unit_type"] = "conclusion_block"
            highlights.append(section_copy)
    return [unit.to_dict() for unit in make_units_from_sections(doc, sections + highlights)]


def parse_financial_contracts(doc: Document, text: str) -> list[dict[str, Any]]:
    sections = split_text_into_sections(text, doc.title, "sec", unit_type="paragraph", max_chars=900)
    element_sections = FinancialContractsPlugin()._extract_element_sections(text, doc.title, max_chars=700)
    table_element_sections = extract_key_value_element_sections(text, doc.title)
    return [unit.to_dict() for unit in make_units_from_sections(doc, sections + table_element_sections + element_sections)]


def extract_key_value_element_sections(text: str, title: str) -> list[dict[str, Any]]:
    sections = []
    section_index = 1
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if "|" not in line:
            continue
        if not any(keyword in line for keyword in CONTRACT_KEYWORDS):
            continue
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) < 2:
            continue
        matched = next(keyword for keyword in CONTRACT_KEYWORDS if keyword in line)
        sections.append(
            {
                "section_id": f"table_element_{section_index}",
                "title_path": [title, matched],
                "text": " | ".join(cells),
                "unit_type": "element_block",
                "metadata": {"element_name": matched, "source": "table"},
                "max_chars": 700,
            }
        )
        section_index += 1
    return sections


PARSERS = {
    "regulatory": parse_regulatory,
    "financial_reports": parse_financial_reports,
    "insurance": parse_insurance,
    "research": parse_research,
    "financial_contracts": parse_financial_contracts,
}


def normalize_domain_doc_id(domain: str, row: dict[str, Any], manifest_domain: dict[str, Any]) -> str:
    doc_id = row["doc_id"]
    if domain != "regulatory":
        return doc_id
    source_relpath = row.get("source_relpath", "")
    if source_relpath.startswith("html/") or source_relpath.startswith("txt/"):
        return doc_id
    if doc_id.endswith("_extracted") and doc_id[:-10] in manifest_domain.get("documents", {}):
        return doc_id[:-10]
    return doc_id


def parse_domain(domain: str, preprocessed_root: Path, output_path: Path, manifest: dict[str, Any]) -> None:
    rows = load_preprocessed_documents(domain, preprocessed_root)
    manifest_domain = manifest["domains"].get(domain, {})
    docs_payload = []
    units_payload = []
    seen_doc_ids: set[str] = set()
    for row in rows:
        cleaned_path = Path(row["cleaned_path"])
        if not cleaned_path.exists():
            continue
        text = cleaned_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        row = dict(row)
        row["doc_id"] = normalize_domain_doc_id(domain, row, manifest_domain)
        duplicate_suffix = 2
        base_doc_id = row["doc_id"]
        while row["doc_id"] in seen_doc_ids:
            row["doc_id"] = f"{base_doc_id}__dup{duplicate_suffix}"
            duplicate_suffix += 1
        seen_doc_ids.add(row["doc_id"])
        doc = make_document(domain, row, text)
        docs_payload.append(doc.to_dict())
        units_payload.extend(PARSERS[domain](doc, text))
    units_payload = finalize_units(units_payload)
    write_json(output_path, {"documents": docs_payload, "units": units_payload})


def max_chars_for_unit(unit: dict[str, Any]) -> int:
    unit_type = unit.get("unit_type", "")
    if unit_type in {"metric_row", "element_block"}:
        return 700
    if unit_type in {"penalty_decision", "formula_block", "clause_block"}:
        return 900
    if unit_type in {"article", "article_chunk", "preamble"}:
        return 1200
    return 1000


def split_text_hard(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[。；;！？!?])|\n", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(part[i : i + max_chars].strip() for i in range(0, len(part), max_chars))
            continue
        if len(buffer) + len(part) + 1 > max_chars and buffer:
            chunks.append(buffer.strip())
            buffer = part
        else:
            buffer = f"{buffer}\n{part}".strip() if buffer else part
    if buffer.strip():
        chunks.append(buffer.strip())
    return [chunk for chunk in chunks if chunk]


def finalize_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for unit in units:
        text = unit.get("text", "").strip()
        if not text:
            continue
        max_chars = max_chars_for_unit(unit)
        chunks = split_text_hard(text, max_chars)
        base_id = unit["unit_id"]
        for index, chunk in enumerate(chunks, start=1):
            new_unit = dict(unit)
            new_unit["text"] = chunk
            if len(chunks) > 1:
                new_unit["parent_unit_id"] = unit.get("parent_unit_id") or base_id
                new_unit["unit_id"] = f"{base_id}::part_{index}"
            candidate_id = new_unit["unit_id"]
            suffix = 2
            while candidate_id in seen_ids:
                candidate_id = f"{new_unit['unit_id']}__dup{suffix}"
                suffix += 1
            new_unit["unit_id"] = candidate_id
            seen_ids.add(candidate_id)
            finalized.append(new_unit)
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--preprocessed-root", default=str(ROOT / "artifacts" / "preprocessed_loop"))
    parser.add_argument("--manifest-path", default=str(ROOT / "artifacts" / "manifest" / "dataset_manifest.json"))
    parser.add_argument("--output-root", default=str(ROOT / "artifacts" / "preprocessed_loop_candidates" / "parsed"))
    args = parser.parse_args()

    domains = DOMAINS if args.domains == ["all"] else args.domains
    manifest = read_json(Path(args.manifest_path))
    for domain in domains:
        output_path = Path(args.output_root) / domain / "parsed.json"
        parse_domain(domain, Path(args.preprocessed_root), output_path, manifest)
        print(output_path)


if __name__ == "__main__":
    main()
