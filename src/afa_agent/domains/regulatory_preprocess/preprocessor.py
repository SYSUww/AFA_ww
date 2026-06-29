from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


CHINESE_NUM = "一二三四五六七八九十百零〇两\\d"
ARTICLE_RE = re.compile(rf"^(第[{CHINESE_NUM}]+条)\s*(.*)$")
CHAPTER_RE = re.compile(rf"^(第[{CHINESE_NUM}]+章)\s*(.*)$")
SECTION_RE = re.compile(rf"^(第[{CHINESE_NUM}]+节)\s*(.*)$")
NUMBERED_ITEM_RE = re.compile(rf"^([{CHINESE_NUM}]+)[、.．]\s*(.*)$")
HTML_BODY_SELECTORS = [".detail-news", ".TRS_Editor", "#zoom", ".content", "article"]


@dataclass(slots=True)
class PreprocessStats:
    input_chars: int = 0
    output_chars: int = 0
    input_lines: int = 0
    output_lines: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "input_lines": self.input_lines,
            "output_lines": self.output_lines,
            "dropped": dict(sorted(self.dropped.items())),
        }


@dataclass(slots=True)
class RegulatoryDocument:
    doc_id: str
    source_path: str
    source_bucket: str
    doc_type: str
    title: str
    doc_no: str
    agency: str
    publish_date: str
    effective_date: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegulatoryUnit:
    unit_id: str
    doc_id: str
    unit_type: str
    title_path: list[str]
    text: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocess_regulatory_corpus(input_root: Path, output_root: Path) -> dict[str, Any]:
    files = _iter_regulatory_input_files(input_root)
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    file_stats: dict[str, Any] = {}
    totals = Counter()
    doc_types = Counter()
    source_buckets = Counter()

    for path in files:
        document, doc_units, stats = preprocess_regulatory_file(path, input_root)
        documents.append(document.to_dict())
        units.extend(unit.to_dict() for unit in doc_units)
        key = str(path.relative_to(input_root))
        file_stats[key] = stats.to_dict() | {
            "doc_id": document.doc_id,
            "doc_type": document.doc_type,
            "unit_count": len(doc_units),
        }
        totals.update(stats.dropped)
        doc_types[document.doc_type] += 1
        source_buckets[document.source_bucket] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "documents.json", documents)
    _write_json(output_root / "units.json", units)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "document_count": len(documents),
        "unit_count": len(units),
        "source_buckets": dict(sorted(source_buckets.items())),
        "doc_types": dict(sorted(doc_types.items())),
        "dropped": dict(sorted(totals.items())),
        "unit_lengths": _unit_length_summary(units),
        "files": file_stats,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def preprocess_regulatory_file(path: Path, input_root: Path) -> tuple[RegulatoryDocument, list[RegulatoryUnit], PreprocessStats]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    source_bucket = _infer_source_bucket(path, input_root)
    doc_id = path.stem
    metadata_hint: dict[str, str] = {}
    if path.suffix.lower() in {".html", ".htm"}:
        body_text, metadata_hint, html_dropped = extract_html_visible_text(raw_text)
        cleaned, stats = clean_regulatory_text(body_text, source_bucket)
        stats.input_chars = len(raw_text)
        stats.input_lines = len(raw_text.splitlines())
        stats.dropped.update(html_dropped)
    else:
        cleaned, stats = clean_regulatory_text(raw_text, source_bucket)
    metadata = extract_document_metadata(cleaned, source_bucket, doc_id)
    for key, value in metadata_hint.items():
        if value:
            metadata[key] = value
    doc_type = classify_doc_type(cleaned, source_bucket, metadata["title"], doc_id)
    document = RegulatoryDocument(
        doc_id=doc_id,
        source_path=str(path),
        source_bucket=source_bucket,
        doc_type=doc_type,
        title=metadata["title"],
        doc_no=metadata["doc_no"],
        agency=metadata["agency"],
        publish_date=metadata["publish_date"],
        effective_date=metadata["effective_date"],
    )
    units = split_units(document, cleaned)
    stats.output_chars = sum(len(unit.text) for unit in units)
    stats.output_lines = sum(len(unit.text.splitlines()) for unit in units)
    return document, units, stats


def _iter_regulatory_input_files(input_root: Path) -> list[Path]:
    suffixes = {".html", ".htm"} if input_root.name == "html" else {".md", ".html", ".htm"}
    candidates = [path for path in input_root.glob("*") if path.suffix.lower() in suffixes]
    candidates.extend(path for path in input_root.glob("*/*") if path.suffix.lower() in suffixes)
    return sorted(set(candidates))


def _infer_source_bucket(path: Path, input_root: Path) -> str:
    relative = path.relative_to(input_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".md":
        return input_root.name if input_root.name in {"attachments", "html", "txt"} else "md"
    return suffix.lstrip(".") or "unknown"


def extract_html_visible_text(html: str) -> tuple[str, dict[str, str], Counter[str]]:
    dropped: Counter[str] = Counter()
    soup = BeautifulSoup(html, "html.parser")
    metadata = {
        "title": _meta_content(soup, "ArticleTitle"),
        "publish_date": _normalize_meta_date(_meta_content(soup, "PubDate")),
    }
    for tag_name in ["script", "style", "noscript", "form", "header", "footer"]:
        tags = soup.find_all(tag_name)
        dropped[f"html_{tag_name}_tags"] += len(tags)
        for tag in tags:
            tag.decompose()
    for tag in soup.find_all("img"):
        dropped["html_img_tags"] += 1
        tag.decompose()

    content = None
    selected = ""
    for selector in HTML_BODY_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and len(candidate.get_text("", strip=True)) >= 50:
            content = candidate
            selected = selector
            break
    if content is None:
        content = soup.body or soup
        selected = "body"
    dropped[f"html_body_selector:{selected}"] += 1
    return content.get_text("\n", strip=True), metadata, dropped


def clean_regulatory_text(text: str, source_bucket: str) -> tuple[str, PreprocessStats]:
    stats = PreprocessStats(input_chars=len(text), input_lines=len(text.splitlines()))
    text = text.replace("\ufeff", "")
    stats.dropped["chunk_comment"] += len(re.findall(r"<!--\s*Chunk:", text))
    text = re.sub(r"<!--\s*Chunk:.*?-->\s*", "", text)
    text = _strip_image_details(text, stats)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = _convert_html_tables(text, source_bucket, stats)

    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if not line.strip():
            if previous_blank:
                stats.dropped["extra_blank"] += 1
                continue
            previous_blank = True
            lines.append("")
            continue
        previous_blank = False
        if source_bucket != "html" and re.fullmatch(r"[-—_ ]*\d{1,4}[-—_ ]*", line.strip()):
            stats.dropped["page_footer"] += 1
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    cleaned = _repair_title_line_breaks(cleaned)
    cleaned = _repair_spacing(cleaned)
    if source_bucket == "html":
        cleaned = _repair_html_rendered_breaks(cleaned)
    cleaned = _strip_residual_html_tags(cleaned, stats)
    return cleaned, stats


def normalize_line(line: str) -> str:
    line = line.replace("\u3000", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.rstrip()


def extract_document_metadata(text: str, source_bucket: str, fallback: str) -> dict[str, str]:
    title = _extract_title(text, fallback)
    doc_no = _extract_doc_no(text, fallback)
    agency = _extract_agency(text, title, fallback)
    publish_date = _extract_publish_date(text)
    effective_date = _extract_effective_date(text)
    return {
        "title": title,
        "doc_no": doc_no,
        "agency": agency,
        "publish_date": publish_date,
        "effective_date": effective_date,
    }


def classify_doc_type(text: str, source_bucket: str, title: str, doc_id: str) -> str:
    sample = "\n".join(text.splitlines()[:80])
    if "行政处罚决定书" in title or "行政处罚决定书" in sample:
        return "penalty_decision"
    if "市场禁入决定书" in title or "市场禁入决定书" in sample:
        return "penalty_decision"
    if "修改" in title or re.search(r"将《.+?》.*修改为", sample):
        return "amendment"
    if "接口规范" in title or "数据接口" in title or "接口规范" in sample:
        return "interface_spec"
    if "内容与格式准则" in title or "编报规则" in title or "格式准则" in sample:
        return "format_guideline"
    if source_bucket == "txt" or ARTICLE_RE.search(text) or "第一章" in sample:
        return "law_text"
    return "unknown"


def split_units(document: RegulatoryDocument, text: str) -> list[RegulatoryUnit]:
    if document.doc_type == "penalty_decision":
        return _split_penalty_units(document, text)
    if document.doc_type == "amendment":
        return _split_numbered_units(document, text, "amendment_item")
    return _split_article_units(document, text)


def _split_article_units(document: RegulatoryDocument, text: str) -> list[RegulatoryUnit]:
    lines = [line for line in text.splitlines() if line.strip()]
    units: list[RegulatoryUnit] = []
    chapter = ""
    section = ""
    current_article = ""
    current_title = ""
    buffer: list[str] = []
    preamble: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_article, current_title
        if not current_article:
            return
        unit_text = "\n".join([f"{current_article} {current_title}".strip(), *buffer]).strip()
        _append_split_unit(units, document, current_article, "article", [document.title, chapter, section, current_article], unit_text)
        buffer = []
        current_article = ""
        current_title = ""

    for line in lines:
        heading = _markdown_heading_text(line)
        if heading:
            if not document.title or document.title == document.doc_id:
                document.title = heading
            if not current_article:
                preamble.append(heading)
            continue
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            flush()
            chapter = " ".join(part for part in chapter_match.groups() if part).strip()
            section = ""
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            flush()
            section = " ".join(part for part in section_match.groups() if part).strip()
            continue
        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush()
            current_article = article_match.group(1)
            current_title = article_match.group(2).strip()
            buffer = []
            continue
        if current_article:
            buffer.append(line)
        else:
            preamble.append(line)
    flush()

    if preamble:
        preamble_text = "\n".join(preamble).strip()
        _append_split_unit(units, document, "preamble", "preamble", [document.title, "preamble"], preamble_text)
    if not units and text.strip():
        _append_split_unit(units, document, "body", "body", [document.title], text.strip())
    return units


def _split_penalty_units(document: RegulatoryDocument, text: str) -> list[RegulatoryUnit]:
    markers = [
        ("penalty_fact", re.compile(r"^(经查明|一、|违法事实|内幕信息情况|信息披露违法)", re.M)),
        ("penalty_argument", re.compile(r"^(在听证过程中|当事人.*提出|.*申辩意见)", re.M)),
        ("penalty_reasoning", re.compile(r"^(经复核|我会认为)", re.M)),
        ("penalty_decision", re.compile(r"^(根据当事人违法行为|我会决定|依据.*我会决定)", re.M)),
        ("remedy", re.compile(r"^(上述当事人应|当事人如果对本处罚决定不服|复议和诉讼期间)", re.M)),
    ]
    lines = [line for line in text.splitlines() if line.strip()]
    sections: list[tuple[str, list[str]]] = [("preamble", [])]
    current_type = "preamble"
    for line in lines:
        matched_type = ""
        for unit_type, pattern in markers:
            if pattern.search(line):
                matched_type = unit_type
                break
        if matched_type and matched_type != current_type:
            current_type = matched_type
            sections.append((current_type, []))
        sections[-1][1].append(line)

    units: list[RegulatoryUnit] = []
    counts = Counter()
    for unit_type, section_lines in sections:
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        counts[unit_type] += 1
        suffix = unit_type if counts[unit_type] == 1 else f"{unit_type}_{counts[unit_type]}"
        _append_split_unit(units, document, suffix, unit_type, [document.title, unit_type], body)
    return units


def _split_numbered_units(document: RegulatoryDocument, text: str, unit_type: str) -> list[RegulatoryUnit]:
    units: list[RegulatoryUnit] = []
    heading_path = [document.title]
    current_id = ""
    buffer: list[str] = []
    preamble: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_id
        if not current_id:
            return
        body = "\n".join(buffer).strip()
        if body:
            _append_split_unit(units, document, current_id, unit_type, heading_path + [current_id], body)
        buffer = []

    for line in [line for line in text.splitlines() if line.strip()]:
        heading = _markdown_heading_text(line)
        if heading:
            heading_path[:] = [document.title, heading]
            if not current_id:
                preamble.append(heading)
            continue
        match = NUMBERED_ITEM_RE.match(line)
        if match:
            flush()
            current_id = match.group(1)
            buffer = [line]
            continue
        if current_id:
            buffer.append(line)
        else:
            preamble.append(line)
    flush()

    if preamble:
        _append_split_unit(units, document, "preamble", "preamble", [document.title, "preamble"], "\n".join(preamble).strip())
    if not units and text.strip():
        _append_split_unit(units, document, "body", "body", [document.title], text.strip())
    return units


def _append_split_unit(
    units: list[RegulatoryUnit],
    document: RegulatoryDocument,
    suffix: str,
    unit_type: str,
    title_path: list[str],
    text: str,
    max_chars: int = 1800,
) -> None:
    clean_title_path = []
    for item in title_path:
        if item and (not clean_title_path or clean_title_path[-1] != item):
            clean_title_path.append(item)
    chunks = _split_long_text(text, max_chars=max_chars)
    for idx, chunk in enumerate(chunks, start=1):
        unit_suffix = suffix if len(chunks) == 1 else f"{suffix}::chunk_{idx}"
        unit_id = _unique_unit_id(units, f"{document.doc_id}::{unit_suffix}")
        units.append(
            RegulatoryUnit(
                unit_id=unit_id,
                doc_id=document.doc_id,
                unit_type=unit_type if len(chunks) == 1 else f"{unit_type}_chunk",
                title_path=clean_title_path,
                text=chunk,
                source_path=document.source_path,
            )
        )


def _unique_unit_id(units: list[RegulatoryUnit], preferred: str) -> str:
    existing = {unit.unit_id for unit in units}
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}::dup_{index}" in existing:
        index += 1
    return f"{preferred}::dup_{index}"


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces = re.split(r"(?<=。|；|;|\n)", text)
    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) > max_chars and buffer.strip():
            chunks.append(buffer.strip())
            buffer = piece
        else:
            buffer += piece
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks or [text[:max_chars]]


def _strip_image_details(text: str, stats: PreprocessStats) -> str:
    def replace(match: re.Match[str]) -> str:
        block = match.group(0)
        summary_match = re.search(r"<summary>(.*?)</summary>", block, flags=re.I | re.S)
        summary = summary_match.group(1).strip().lower() if summary_match else ""
        body = re.sub(r"</?details>|<summary>.*?</summary>", "", block, flags=re.I | re.S).strip()
        has_structured_content = bool(re.search(r"[\u4e00-\u9fff\d|]", body))
        if summary in {"natural_image", "text_image", "image"} and not has_structured_content:
            stats.dropped[f"details:{summary or 'unknown'}"] += 1
            return ""
        if summary in {"natural_image", "text_image"}:
            stats.dropped[f"details:{summary}"] += 1
            return ""
        return body

    return re.sub(r"<details\b.*?</details>", replace, text, flags=re.I | re.S)


def _convert_html_tables(text: str, source_bucket: str, stats: PreprocessStats) -> str:
    def convert(match: re.Match[str]) -> str:
        table = match.group(0)
        soup = BeautifulSoup(table, "html.parser")
        rows: list[str] = []
        for tr in soup.find_all("tr"):
            raw_cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            cells = raw_cells if source_bucket == "html" else [cell for cell in raw_cells if cell]
            if not cells:
                continue
            if source_bucket == "html":
                for idx in range(0, len(cells) - 1, 2):
                    key = cells[idx]
                    value = cells[idx + 1]
                    if key in {"名 称", "文 号", "发布机构", "发文日期"} and value:
                        rows.append(f"{key} | {value}")
                continue
            rows.append(" | ".join(cells))
        stats.dropped["html_table_tags"] += table.count("<")
        return "\n".join(rows)

    return re.sub(r"<table\b.*?</table>", convert, text, flags=re.I | re.S)


def _repair_title_line_breaks(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = re.sub(r"\s+", "", match.group(1))
        return f"《{inner}》"

    return re.sub(r"《([^》]{2,120})》", replace, text, flags=re.S)


def _repair_spacing(text: str) -> str:
    text = re.sub(r"〔\s*(\d{4})\s*〕\s*(\d+)\s*号", r"〔\1〕\2号", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=年|月|日|%|％|号|万元|亿元|元|分)", "", text)
    text = re.sub(r"(?<=年)\s+(?=\d{1,2}月)", "", text)
    text = re.sub(r"(?<=月)\s+(?=\d{1,2}日)", "", text)
    text = re.sub(r"(?<=月)\s+(?=至)", "", text)
    text = re.sub(r"(?<=第)\s+(?=[一二三四五六七八九十百零〇两\d])", "", text)
    text = re.sub(r"(?<=[一二三四五六七八九十百零〇两\d])\s+(?=条|章|节)", "", text)
    text = re.sub(r"([一二三四五六七八九十百零〇两])\s+([一二三四五六七八九十百零〇两])(?=[条章节])", r"\1\2", text)
    return text


def _repair_html_rendered_breaks(text: str) -> str:
    text = re.sub(r"(?<=[（(《])\s+(?=[^）)》\n]{1,20}[）)》])", "", text)
    text = re.sub(r"(?<=[，,、:：;；])\s+(?=[\u4e00-\u9fffA-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[,，。；;、）)])", "", text)
    text = re.sub(r"(?<=\d)\s+(?=,|，|、|至|起|月|年|日)", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=所[)）])", "", text)
    return text


def _meta_content(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    if not tag:
        return ""
    value = tag.get("content")
    return value.strip() if isinstance(value, str) else ""


def _normalize_meta_date(value: str) -> str:
    match = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", value)
    if not match:
        return value[:10] if re.match(r"20\d{2}-\d{2}-\d{2}", value) else ""
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _strip_residual_html_tags(text: str, stats: PreprocessStats) -> str:
    matches = re.findall(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s+[^<>]*)?>", text)
    if matches:
        stats.dropped["residual_html_tags"] += len(matches)
        text = re.sub(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s+[^<>]*)?>", "", text)
    return text


def _extract_title(text: str, fallback: str) -> str:
    lines = [line.strip("# ").strip() for line in text.splitlines() if line.strip()]
    first_lines = "\n".join(lines[:12])
    metadata_title = re.search(r"^名\s*称\s*\|\s*(.+)$", first_lines, flags=re.M)
    if metadata_title:
        title = metadata_title.group(1).strip()
        quoted = re.search(r"《([^》]{2,100})》", title)
        return quoted.group(1).strip() if quoted else title[:100]
    for raw_line in text.splitlines()[:20]:
        heading = _markdown_heading_text(raw_line)
        if heading:
            if re.fullmatch(r"附件\s*\d*[：:]?", heading):
                continue
            quoted = re.search(r"《([^》]{2,100})》", heading)
            return quoted.group(1).strip() if quoted else heading[:100]
    quoted = re.search(r"《([^》]{2,100})》", first_lines)
    if quoted:
        return quoted.group(1).strip()
    for line in lines[:8]:
        if any(word in line for word in ["办法", "规定", "准则", "指引", "细则", "决定书", "条例", "规则"]):
            return line[:100]
    return lines[0][:100] if lines else fallback


def _extract_doc_no(text: str, fallback: str) -> str:
    patterns = [
        r"(中国人民银行令〔\d{4}〕第\d+号)",
        r"(证监会令[〔【\[]?第?\d+号[〕】\]]?)",
        r"(〔\s*\d{4}\s*〕\s*\d+\s*号)",
    ]
    sample = f"{fallback}\n" + "\n".join(text.splitlines()[:20])
    for pattern in patterns:
        match = re.search(pattern, sample)
        if match:
            return re.sub(r"\s+", "", match.group(1))
    return ""


def _extract_agency(text: str, title: str, fallback: str) -> str:
    strong_source = f"{fallback}\n{title}"
    agencies = [
        "中国证券监督管理委员会",
        "中国证监会",
        "中国人民银行",
        "国家金融监督管理总局",
        "国务院",
    ]
    for agency in agencies:
        if agency in strong_source:
            return agency
    sample = "\n".join(text.splitlines()[:12])
    for agency in agencies:
        if agency in sample or agency in title:
            return agency
    if "证监会" in sample:
        return "证监会"
    return ""


def _extract_publish_date(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()[:30] if line.strip()]
    dates: list[str] = []
    for line in lines:
        match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", line)
        if match:
            dates.append(_date(match))
    return dates[-1] if dates else ""


def _extract_effective_date(text: str) -> str:
    sample = "\n".join(text.splitlines()[:40])
    match = re.search(r"自(20\d{2})年(\d{1,2})月(\d{1,2})日起施行", sample)
    if match:
        return _date(match)
    if "自发布之日起施行" in sample:
        return "自发布之日起施行"
    return ""


def _date(match: re.Match[str]) -> str:
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _markdown_heading_text(line: str) -> str:
    match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
    return match.group(1).strip() if match else ""


def _unit_length_summary(units: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(len(unit.get("text", "")) for unit in units)
    if not lengths:
        return {"min": 0, "median": 0, "max": 0, "over_1800": 0}
    return {
        "min": lengths[0],
        "median": lengths[len(lengths) // 2],
        "max": lengths[-1],
        "over_1800": sum(1 for length in lengths if length > 1800),
    }


def _write_json(path: Path, payload: Any) -> None:
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
