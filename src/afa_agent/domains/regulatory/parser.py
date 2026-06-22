from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pypdf import PdfReader

from afa_agent.models import Document, EvidenceUnit
from afa_agent.text_utils import ARTICLE_RE, CHAPTER_RE, SECTION_RE, clean_lines, normalize_whitespace


def extract_text_from_html(path: Path) -> tuple[str, dict[str, Any]]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = ""
    meta_title = soup.find("meta", attrs={"name": "ArticleTitle"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    lines = clean_lines(soup.get_text("\n").splitlines())
    return "\n".join(lines), {"title": title}


def extract_text_from_txt(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return normalize_whitespace(text), {}


def extract_text_from_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[PAGE {page_index}]\n{text.strip()}")
    return normalize_whitespace("\n\n".join(pages)), {"page_count": len(reader.pages)}


def detect_title(text: str, fallback: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback
    quoted = re.search(r"《([^》]{2,80})》", "\n".join(lines[:10]))
    if quoted:
        return quoted.group(1)
    for line in lines[:5]:
        if "办法" in line or "法" in line or "指引" in line or "准则" in line:
            candidate = line.strip("《》 ")
            if len(candidate) <= 80:
                return candidate
    return lines[0][:80]


def split_regulatory_units(doc: Document, text: str) -> list[EvidenceUnit]:
    lines = clean_lines(text.splitlines())
    units: list[EvidenceUnit] = []
    chapter = ""
    section = ""
    current_article_no = ""
    current_article_title = ""
    current_buffer: list[str] = []
    preamble: list[str] = []

    def flush_article() -> None:
        nonlocal current_article_no, current_article_title, current_buffer
        if not current_article_no:
            return
        article_text = "\n".join(current_buffer).strip()
        if not article_text:
            return
        title_path = [part for part in [doc.title, chapter, section, current_article_no] if part]
        unit_id = f"{doc.doc_id}::{current_article_no}"
        full_text = f"{current_article_no} {current_article_title}".strip()
        if article_text:
            full_text = f"{full_text}\n{article_text}".strip()
        units.extend(split_long_unit(doc, unit_id, title_path, current_article_no, full_text))
        current_buffer = []

    for line in lines:
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter = " ".join(part for part in chapter_match.groups() if part).strip()
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            section = " ".join(part for part in section_match.groups() if part).strip()
            continue
        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            current_article_no = article_match.group(1)
            current_article_title = article_match.group(2).strip()
            current_buffer = []
            continue
        if current_article_no:
            current_buffer.append(line)
        else:
            preamble.append(line)
    flush_article()
    if preamble:
        units.insert(
            0,
            EvidenceUnit(
                unit_id=f"{doc.doc_id}::preamble",
                doc_id=doc.doc_id,
                domain=doc.domain,
                unit_type="preamble",
                title_path=[doc.title, "preamble"],
                text="\n".join(preamble).strip(),
                page_refs=[],
                metadata={},
            ),
        )
    return units


def split_long_unit(
    doc: Document,
    unit_id: str,
    title_path: list[str],
    article_no: str,
    text: str,
    max_chars: int = 1200,
) -> list[EvidenceUnit]:
    if len(text) <= max_chars:
        return [
            EvidenceUnit(
                unit_id=unit_id,
                doc_id=doc.doc_id,
                domain=doc.domain,
                unit_type="article",
                title_path=title_path,
                text=text,
                page_refs=[],
                metadata={"article_no": article_no},
            )
        ]
    chunks: list[EvidenceUnit] = []
    parts = re.split(r"(?<=。)", text)
    buffer = ""
    part_index = 1
    for part in parts:
        if len(buffer) + len(part) > max_chars and buffer:
            chunks.append(
                EvidenceUnit(
                    unit_id=f"{unit_id}::chunk_{part_index}",
                    doc_id=doc.doc_id,
                    domain=doc.domain,
                    unit_type="article_chunk",
                    title_path=title_path,
                    text=buffer.strip(),
                    page_refs=[],
                    parent_unit_id=unit_id,
                    metadata={"article_no": article_no, "chunk_index": part_index},
                )
            )
            part_index += 1
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        chunks.append(
            EvidenceUnit(
                unit_id=f"{unit_id}::chunk_{part_index}",
                doc_id=doc.doc_id,
                domain=doc.domain,
                unit_type="article_chunk",
                title_path=title_path,
                text=buffer.strip(),
                page_refs=[],
                parent_unit_id=unit_id,
                metadata={"article_no": article_no, "chunk_index": part_index},
            )
        )
    return chunks
