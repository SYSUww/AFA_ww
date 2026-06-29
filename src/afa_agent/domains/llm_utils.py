from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.models import RetrievalHit, TokenUsage


def format_hits(
    hits: list[RetrievalHit],
    max_items: int = 6,
    max_chars: int | None = None,
    focus_terms: list[str] | None = None,
) -> str:
    lines = []
    for idx, hit in enumerate(hits[:max_items], start=1):
        title = " > ".join(hit.title_path)
        text = truncate_text(hit.text, max_chars=max_chars, focus_terms=focus_terms)
        lines.append(f"[{idx}] {hit.doc_id} | {title}\n{text}")
    return "\n\n".join(lines)


def truncate_text(text: str, *, max_chars: int | None = None, focus_terms: list[str] | None = None) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    focus_terms = [term for term in focus_terms or [] if term]
    center = -1
    for term in focus_terms:
        position = text.find(term)
        if position >= 0:
            center = position
            break
    if center >= 0:
        start = max(0, center - max_chars // 3)
    else:
        start = 0
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "...[truncated]\n" if start > 0 else ""
    suffix = "\n...[truncated]" if end < len(text) else ""
    return f"{prefix}{text[start:end].rstrip()}{suffix}"


def ask_option_judgment(
    client: OpenAICompatibleClient,
    system_prompt: str,
    question_text: str,
    answer_format: str,
    option_key: str,
    option_text: str,
    evidence_text: str,
    extra_context: str = "",
) -> tuple[dict[str, Any], TokenUsage]:
    user_parts = [
        f"题目：{question_text}",
        f"题型：{answer_format}",
        f"选项 {option_key}：{option_text}",
    ]
    if extra_context.strip():
        user_parts.append(extra_context.strip())
    user_parts.append(f"证据：\n{evidence_text}")
    user_parts.append('请输出 JSON，格式为 {"label": true/false, "reasoning_summary": "...", "used_evidence_ids": [1,2]}。')
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
    total_usage = TokenUsage()
    last_error: Exception | None = None
    for _ in range(2):
        response = client.chat_json(messages)
        total_usage.add(response.token_usage)
        try:
            return extract_json_object(response.content), total_usage
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "\n\n".join(user_parts)
                    + '\n\n上一次输出不是合法 JSON。请只输出一个 JSON 对象，例如 {"label": true, "reasoning_summary": "...", "used_evidence_ids": [1]}。',
                },
            ]
    assert last_error is not None
    raise last_error


def ask_answer_fallback(
    client: OpenAICompatibleClient,
    system_prompt: str,
    question_text: str,
    option_payloads: list[dict[str, Any]],
    answer_format: str,
    allowed_options: list[str],
) -> tuple[str, TokenUsage]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"题目：{question_text}\n题型：{answer_format}\n"
                f"选项摘要：{json.dumps(option_payloads, ensure_ascii=False)}\n"
                '输出格式：{"answer": "A"} 或 {"answer": "AC"}'
            ),
        },
    ]
    total_usage = TokenUsage()
    for _ in range(2):
        response = client.chat_json(messages)
        total_usage.add(response.token_usage)
        try:
            parsed = extract_json_object(response.content)
            raw_answer = str(parsed.get("answer", "")).strip().upper()
            cleaned = "".join(ch for ch in sorted(set(raw_answer)) if ch in allowed_options)
            if cleaned:
                return cleaned, total_usage
        except Exception:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + '\n\n上一次输出不是合法 JSON。请只输出一个 JSON 对象，例如 {"answer": "A"}。',
                },
            ]
    fallback = allowed_options[0] if answer_format in {"mcq", "multi"} else "B"
    return fallback, total_usage


NUMBER_RE = re.compile(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)(?![\d.])")


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers
