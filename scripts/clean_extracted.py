#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "artifacts" / "extracted"
OUTPUT_ROOT = ROOT / "artifacts" / "extracted_cleaned"
RAW_REGULATORY_ROOT = ROOT / "public_dataset_upload" / "raw" / "regulatory"

STRUCTURED_DETAIL_SUMMARIES = {
    "bar chart",
    "bar-line hybrid chart",
    "flowchart",
    "line chart",
    "pie chart",
    "stacked bar chart",
    "table",
}
DROP_DETAIL_SUMMARIES = {"natural_image", "text_image"}

IMPORTANT_KEYWORDS = {
    "financial_reports": [
        "营业收入",
        "净利润",
        "归母",
        "经营活动",
        "现金流",
        "研发",
        "分红",
        "派息",
        "同比",
        "资产负债",
    ],
    "insurance": [
        "保险责任",
        "责任免除",
        "现金价值",
        "账户价值",
        "已交保费",
        "基本保额",
        "领取",
        "退保",
        "身故保险金",
        "保险金",
    ],
    "research": [
        "投资要点",
        "投资建议",
        "风险提示",
        "同比",
        "市场规模",
        "渗透率",
        "增速",
        "数据来源",
        "预计",
    ],
    "financial_contracts": [
        "发行人",
        "发行规模",
        "债券期限",
        "票面利率",
        "评级",
        "受托管理人",
        "承销商",
        "回售",
        "赎回",
        "违约",
        "募集资金",
        "偿债",
    ],
}


@dataclass
class CleanStats:
    input_lines: int = 0
    output_lines: int = 0
    input_chars: int = 0
    output_chars: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "input_lines": self.input_lines,
            "output_lines": self.output_lines,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "line_reduction": self.input_lines - self.output_lines,
            "char_reduction": self.input_chars - self.output_chars,
            "dropped": dict(sorted(self.dropped.items())),
        }


def infer_domain(path: Path) -> str:
    try:
        relative = path.relative_to(INPUT_ROOT)
        return relative.parts[0] if relative.parts else ""
    except ValueError:
        pass
    try:
        path.relative_to(RAW_REGULATORY_ROOT)
        return "regulatory"
    except ValueError:
        return ""


def normalize_line(line: str) -> str:
    line = line.replace("\u3000", " ").replace("\ufeff", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.rstrip()


def is_page_or_footer_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.fullmatch(r"[-—_ ]*\d{1,4}[-—_ ]*", stripped):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
        return True
    if re.fullmatch(r"[|｜]?\s*\d{1,4}\s*[|｜]\s*[\u4e00-\u9fffA-Za-z0-9（）() ]{2,40}", stripped):
        return True
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9（）() ]{2,40}\s*[|｜]\s*\d{1,4}", stripped):
        return True
    return False


def is_toc_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "目录" in stripped and len(stripped) <= 8:
        return False
    if re.search(r"\.{2,}\s*\d{1,4}$", stripped):
        return True
    if re.search(r"[·.。．]{2,}\s*$", stripped):
        return True
    if re.match(r"^(第[一二三四五六七八九十百零〇两\d]+[章节]|[一二三四五六七八九十]+[、.])", stripped) and re.search(r"\s+\d{1,4}$", stripped):
        return True
    return False


def is_low_value_ocr(line: str, domain: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(keyword in stripped for keyword in IMPORTANT_KEYWORDS.get(domain, [])):
        return False
    if re.search(r"\d", stripped) and re.search(r"[%亿元万年月日.AAA]", stripped):
        return False
    if re.fullmatch(r"[#>\-—_\\/*|｜`~·.。:：,，;；!！?？（）()\[\]【】 ]+", stripped):
        return True
    if len(stripped) <= 2 and not re.search(r"[\d一二三四五六七八九十]", stripped):
        return True
    if re.fullmatch(r"[A-Za-z ]{2,35}", stripped):
        allowed = {"AAA", "AA", "A", "MSCI", "ESG", "CICC"}
        compact = stripped.replace(" ", "")
        return compact not in allowed
    return False


def should_drop_repeated_short(line: str, repeated_short: set[str], domain: str) -> bool:
    stripped = line.strip()
    if stripped not in repeated_short:
        return False
    if any(keyword in stripped for keyword in IMPORTANT_KEYWORDS.get(domain, [])):
        return False
    if re.search(r"\d", stripped):
        return False
    if stripped.startswith("#"):
        return False
    return True


def repeated_short_lines(lines: list[str]) -> set[str]:
    counts = Counter(line.strip() for line in lines if 2 <= len(line.strip()) <= 50)
    return {line for line, count in counts.items() if count >= 4}


def clean_markdown(text: str, domain: str) -> tuple[str, CleanStats]:
    stats = CleanStats(input_chars=len(text))
    raw_lines = text.splitlines()
    stats.input_lines = len(raw_lines)
    repeated_short = repeated_short_lines(raw_lines)
    output: list[str] = []
    in_details = False
    detail_summary = ""
    detail_buffer: list[str] = []

    def flush_detail() -> None:
        nonlocal detail_buffer, detail_summary, in_details
        summary = detail_summary.strip().lower()
        keep = summary in STRUCTURED_DETAIL_SUMMARIES or any("|" in line for line in detail_buffer)
        if keep:
            output.extend(detail_buffer)
        else:
            stats.dropped[f"details:{summary or 'unknown'}"] += len(detail_buffer)
        detail_buffer = []
        detail_summary = ""
        in_details = False

    previous_blank = False
    previous_heading = ""
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        stripped = line.strip()

        if stripped.startswith("<details"):
            in_details = True
            detail_buffer = [line]
            detail_summary = ""
            continue
        if in_details:
            detail_buffer.append(line)
            summary_match = re.search(r"<summary>(.*?)</summary>", stripped, flags=re.IGNORECASE)
            if summary_match:
                detail_summary = summary_match.group(1)
            if stripped.startswith("</details"):
                flush_detail()
            continue

        if re.match(r"<!--\s*Chunk:", stripped):
            stats.dropped["chunk_comment"] += 1
            continue
        if re.fullmatch(r"!\[[^\]]*\]\([^)]*\)", stripped):
            stats.dropped["image_link"] += 1
            continue
        if re.fullmatch(r"#{1,6}\s*", stripped):
            stats.dropped["empty_heading"] += 1
            continue
        if stripped.startswith("#") and stripped == previous_heading:
            stats.dropped["duplicate_heading"] += 1
            continue
        if is_toc_noise(stripped):
            stats.dropped["toc_line"] += 1
            continue
        if is_page_or_footer_noise(stripped):
            stats.dropped["page_footer"] += 1
            continue
        if should_drop_repeated_short(stripped, repeated_short, domain):
            stats.dropped["repeated_short"] += 1
            continue
        if is_low_value_ocr(stripped, domain):
            stats.dropped["low_value_ocr"] += 1
            continue

        is_blank = not stripped
        if is_blank and previous_blank:
            stats.dropped["extra_blank"] += 1
            continue
        output.append(line)
        previous_blank = is_blank
        if stripped.startswith("#"):
            previous_heading = stripped

    if in_details:
        flush_detail()

    cleaned = "\n".join(output).strip() + "\n"
    stats.output_lines = len(cleaned.splitlines())
    stats.output_chars = len(cleaned)
    return cleaned, stats


def clean_html(text: str) -> tuple[str, CleanStats]:
    stats = CleanStats(input_lines=len(text.splitlines()), input_chars=len(text))
    cleaned = re.sub(r"<script\b.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style\b.*?</style>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<noscript\b.*?</noscript>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    stats.dropped["html_script_style_blocks"] = len(re.findall(r"<(?:script|style|noscript)\b", text, flags=re.IGNORECASE))
    stats.output_lines = len(cleaned.splitlines())
    stats.output_chars = len(cleaned)
    return cleaned, stats


def clean_text_file(path: Path) -> tuple[str, CleanStats]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    domain = infer_domain(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return clean_markdown(text, domain)
    if suffix == ".html":
        return clean_html(text)
    stats = CleanStats(input_lines=len(text.splitlines()), output_lines=len(text.splitlines()), input_chars=len(text), output_chars=len(text))
    return text, stats


def output_path_for(path: Path) -> Path:
    try:
        return OUTPUT_ROOT / path.relative_to(INPUT_ROOT)
    except ValueError:
        relative = path.relative_to(RAW_REGULATORY_ROOT)
        return OUTPUT_ROOT / "regulatory" / relative


def iter_input_files(domain: str = "") -> list[Path]:
    roots = [INPUT_ROOT / domain] if domain else [INPUT_ROOT]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    if domain in {"", "regulatory"}:
        for subdir in ["txt", "html"]:
            raw_root = RAW_REGULATORY_ROOT / subdir
            if raw_root.exists():
                files.extend(path for path in sorted(raw_root.rglob("*")) if path.is_file() and path.suffix.lower() in {".txt", ".html"})
    return files


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = iter_input_files(args.domain)
    if not files:
        raise FileNotFoundError(f"No input files found under {INPUT_ROOT / args.domain if args.domain else INPUT_ROOT}")

    summary: dict[str, object] = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "domain": args.domain or "all",
        "file_count": 0,
        "total_input_chars": 0,
        "total_output_chars": 0,
        "total_input_lines": 0,
        "total_output_lines": 0,
        "dropped": {},
        "files": {},
    }
    total_dropped: Counter[str] = Counter()

    for source in files:
        target = output_path_for(source)
        if target.exists() and not args.force:
            continue
        cleaned, stats = clean_text_file(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(cleaned, encoding="utf-8")
        meta_path = target.with_suffix(target.suffix + ".clean_meta.json")
        write_json(
            meta_path,
            {
                "source_path": str(source),
                "output_path": str(target),
                "domain": infer_domain(source),
                **stats.to_dict(),
            },
        )
        try:
            file_key = str(source.relative_to(INPUT_ROOT))
        except ValueError:
            file_key = f"raw/regulatory/{source.relative_to(RAW_REGULATORY_ROOT)}"
        summary["files"][file_key] = stats.to_dict()  # type: ignore[index]
        summary["file_count"] = int(summary["file_count"]) + 1
        summary["total_input_chars"] = int(summary["total_input_chars"]) + stats.input_chars
        summary["total_output_chars"] = int(summary["total_output_chars"]) + stats.output_chars
        summary["total_input_lines"] = int(summary["total_input_lines"]) + stats.input_lines
        summary["total_output_lines"] = int(summary["total_output_lines"]) + stats.output_lines
        total_dropped.update(stats.dropped)

    summary["dropped"] = dict(sorted(total_dropped.items()))
    write_json(OUTPUT_ROOT / "clean_summary.json", summary)
    print(OUTPUT_ROOT / "clean_summary.json")


if __name__ == "__main__":
    main()
