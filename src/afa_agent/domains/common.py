from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader

from afa_agent.models import Document, EvidenceUnit
from afa_agent.text_utils import clean_lines, normalize_whitespace


HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+){0,3}\s+.+$"),
    re.compile(r"^第[一二三四五六七八九十百零〇两\d]+[章节条]\s*.*$"),
    re.compile(r"^[一二三四五六七八九十]+[、.].+$"),
]


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


def extract_text_from_markdown(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return normalize_whitespace(text), {}


def extract_text_from_pdf(path: Path, options: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    options = options or {}
    backend = options.get("pdf_backend", "pypdf")
    if backend == "mineru":
        mineru_result = _extract_text_from_pdf_via_mineru(path, options)
        if mineru_result is not None:
            return mineru_result
        mineru_error = getattr(_extract_text_from_pdf_via_mineru, "_last_error", "")
    else:
        mineru_error = ""
    reader = PdfReader(str(path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = normalize_whitespace(text)
        if cleaned:
            if options.get("drop_short_lines"):
                cleaned = "\n".join(line for line in cleaned.splitlines() if len(line.strip()) >= 3)
            if options.get("keep_page_markers", True):
                pages.append(f"[PAGE {page_index}]\n{cleaned}")
            else:
                pages.append(cleaned)
    return "\n\n".join(pages), {
        "page_count": len(reader.pages),
        "pdf_backend": "pypdf",
        "requested_pdf_backend": backend,
        "pdf_backend_fallback_reason": mineru_error,
    }


def _extract_text_from_pdf_via_mineru(path: Path, options: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    command_factory = _resolve_mineru_command(path, options)
    if not command_factory:
        return None
    with tempfile.TemporaryDirectory(prefix="afa_mineru_") as temp_dir:
        output_dir = Path(temp_dir)
        env = os.environ.copy()
        model_source = options.get("mineru_model_source", "")
        if model_source:
            env["MINERU_MODEL_SOURCE"] = str(model_source)
        try:
            proc = subprocess.run(command_factory(output_dir), check=True, capture_output=True, text=True, env=env)
        except Exception as exc:
            setattr(_extract_text_from_pdf_via_mineru, "_last_error", str(exc))
            return None
        text = _collect_mineru_output_text(output_dir)
        if not text.strip():
            setattr(
                _extract_text_from_pdf_via_mineru,
                "_last_error",
                f"mineru returned code {proc.returncode} but no markdown/txt output was collected",
            )
            return None
        if options.get("drop_short_lines"):
            text = "\n".join(line for line in text.splitlines() if len(line.strip()) >= 3)
        normalized = normalize_whitespace(text)
        if options.get("keep_page_markers", True):
            normalized = f"[PAGE 1]\n{normalized}"
        setattr(_extract_text_from_pdf_via_mineru, "_last_error", "")
        return normalized, {"page_count": 0, "pdf_backend": "mineru"}


def _resolve_mineru_command(path: Path, options: dict[str, Any]):
    mineru_bin = shutil.which("mineru")
    if mineru_bin:
        return lambda output_dir: _build_mineru_args(mineru_bin, path, output_dir, options)
    magic_pdf_bin = shutil.which("magic-pdf")
    if magic_pdf_bin:
        return lambda output_dir: [magic_pdf_bin, "-p", str(path), "-o", str(output_dir)]
    return None


def _build_mineru_args(binary: str, path: Path, output_dir: Path, options: dict[str, Any]) -> list[str]:
    args = [binary, "--path", str(path), "--output", str(output_dir)]
    backend = options.get("mineru_backend")
    method = options.get("mineru_method")
    lang = options.get("mineru_lang")
    if backend:
        args.extend(["--backend", str(backend)])
    if method:
        args.extend(["--method", str(method)])
    if lang:
        args.extend(["--lang", str(lang)])
    return args


def _collect_mineru_output_text(output_dir: Path) -> str:
    candidates = sorted(output_dir.rglob("*.md")) + sorted(output_dir.rglob("*.markdown")) + sorted(output_dir.rglob("*.txt"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return ""


def load_text_by_source(path: Path, source_type: str, options: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    normalized = source_type.lower()
    if normalized == "txt":
        return extract_text_from_txt(path)
    if normalized in {"md", "markdown"}:
        return extract_text_from_markdown(path)
    if normalized == "html":
        return extract_text_from_html(path)
    if normalized == "pdf":
        return extract_text_from_pdf(path, options=options)
    raise ValueError(f"Unsupported source_type: {source_type}")


def detect_title(text: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback
    quoted = re.search(r"《([^》]{2,100})》", "\n".join(lines[:10]))
    if quoted:
        return quoted.group(1)
    for line in lines[:10]:
        if any(keyword in line for keyword in ["报告", "办法", "合同", "条款", "说明书", "研究", "保险", "债券"]):
            return line[:120]
    return lines[0][:120]


def extract_page_refs(text: str) -> list[int]:
    refs = []
    for match in re.finditer(r"\[PAGE (\d+)\]", text):
        refs.append(int(match.group(1)))
    return refs


def split_long_text(text: str, max_chars: int = 1200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[。；！？\n])", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if len(buffer) + len(part) > max_chars and buffer:
            chunks.append(buffer.strip())
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks or [text]


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in HEADING_PATTERNS)


def make_units_from_sections(
    doc: Document,
    sections: list[dict[str, Any]],
    default_unit_type: str = "paragraph",
) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for index, section in enumerate(sections, start=1):
        raw_text = section["text"].strip()
        if not raw_text:
            continue
        chunks = split_long_text(raw_text, max_chars=section.get("max_chars", 1200))
        for chunk_index, chunk in enumerate(chunks, start=1):
            unit_id = f"{doc.doc_id}::{section['section_id']}"
            if len(chunks) > 1:
                unit_id = f"{unit_id}::chunk_{chunk_index}"
            units.append(
                EvidenceUnit(
                    unit_id=unit_id,
                    doc_id=doc.doc_id,
                    domain=doc.domain,
                    unit_type=section.get("unit_type", default_unit_type),
                    title_path=section.get("title_path", [doc.title]),
                    text=chunk,
                    page_refs=section.get("page_refs", []),
                    parent_unit_id=section.get("parent_unit_id"),
                    metadata=section.get("metadata", {}),
                )
            )
    return units


def split_text_into_sections(
    text: str,
    title: str,
    section_prefix: str,
    unit_type: str = "paragraph",
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
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
                    "section_id": f"{section_prefix}_{section_index}",
                    "title_path": [title, current_title] if current_title != title else [title],
                    "text": body,
                    "unit_type": unit_type,
                    "page_refs": sorted(set(page_refs)),
                    "max_chars": max_chars,
                }
            )
            section_index += 1
        buffer = []
        page_refs = []

    for line in lines:
        if line.startswith("[PAGE "):
            page_refs.extend(extract_page_refs(line))
            continue
        if looks_like_heading(line) and buffer:
            flush()
            current_title = line[:120]
            continue
        if looks_like_heading(line):
            current_title = line[:120]
            continue
        buffer.append(line)
    flush()
    return sections


def iter_keyword_windows(lines: Iterable[str], keywords: list[str], radius: int = 1) -> list[str]:
    line_list = [line for line in lines if line.strip()]
    windows: list[str] = []
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for index, line in enumerate(line_list):
        lowered = line.lower()
        if not any(keyword in lowered for keyword in lowered_keywords):
            continue
        start = max(0, index - radius)
        end = min(len(line_list), index + radius + 1)
        windows.append("\n".join(line_list[start:end]))
    return windows
