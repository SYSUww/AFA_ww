from __future__ import annotations

import re
from typing import Iterable


ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+条)\s*(.*)$")
CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+章)\s*(.*)$")
SECTION_RE = re.compile(r"^(第[一二三四五六七八九十百零〇两\d]+节)\s*(.*)$")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned = []
    for line in lines:
        normalized = normalize_whitespace(line)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def tokenize_zh(text: str) -> list[str]:
    normalized = normalize_whitespace(text).lower()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9_.%]+", normalized):
        tokens.append(match.group(0))
    for segment in re.findall(r"[\u4e00-\u9fff]{1,}", normalized):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment)
        tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    for match in re.finditer(r"第[一二三四五六七八九十百零〇两\d]+[章节条]", normalized):
        tokens.append(match.group(0))
    return tokens
