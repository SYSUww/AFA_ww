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


def finalize_answer(
    answer: str,
    *,
    answer_format: str,
    allowed_options: list[str],
    option_labels: dict[str, bool],
    option_payloads: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Normalize final answer while recording when format constraints force a choice."""
    metadata: dict[str, Any] = {
        "raw_answer": answer,
        "answer_format": answer_format,
        "format_forced": False,
        "forced_options": [],
        "no_supported_fallback": False,
        "invalid_model_answer": False,
    }
    allowed = [option.upper() for option in allowed_options]
    if answer_format == "tf":
        letters = [ch for ch in answer.upper() if ch in {"A", "B"}]
        if letters:
            return letters[0], metadata
        metadata["invalid_model_answer"] = True
        return ("A" if option_labels.get("A", False) else "B"), metadata

    cleaned = "".join(ch for ch in sorted(set(answer.upper())) if ch in allowed)
    if answer and not cleaned:
        metadata["invalid_model_answer"] = True
    supported = sorted(option for option, label in option_labels.items() if label and option.upper() in allowed)

    if answer_format == "mcq":
        if len(supported) == 1:
            return supported[0], metadata
        if cleaned[:1] in allowed:
            chosen = cleaned[:1]
        else:
            ranked = _rank_option_payloads(option_payloads or [], allowed)
            chosen = ranked[0] if ranked else (allowed[0] if allowed else "")
            metadata["invalid_model_answer"] = True
        if chosen and not option_labels.get(chosen, False):
            metadata["no_supported_fallback"] = not supported
            metadata["format_forced"] = True
            metadata["forced_options"] = [chosen]
        return chosen, metadata

    if answer_format != "multi":
        return cleaned, metadata

    selected = set(cleaned)
    selected.update(supported)
    if len(selected) < 2:
        for option in _rank_option_payloads(option_payloads or [], allowed):
            if option in selected:
                continue
            selected.add(option)
            if not option_labels.get(option, False):
                metadata["format_forced"] = True
                metadata["forced_options"].append(option)
            if len(selected) >= 2:
                break
    if len(selected) < 2:
        for option in allowed:
            if option in selected:
                continue
            selected.add(option)
            metadata["format_forced"] = True
            metadata["forced_options"].append(option)
            if len(selected) >= 2:
                break
    if not supported and selected:
        metadata["no_supported_fallback"] = True
    return "".join(sorted(selected)), metadata


def collect_evidence_items(
    option_payloads: list[dict[str, Any]],
    *,
    doc_ids: list[str],
    max_per_supported_option: int = 3,
    max_total: int = 16,
) -> list[dict[str, Any]]:
    """Build final evidence with selected-option support plus per-document coverage."""
    evidence_items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def item_key(item: dict[str, Any]) -> str:
        unit_id = str(item.get("unit_id", ""))
        if unit_id:
            return unit_id.replace("__dup2", "").replace("__dup", "")
        return f"{item.get('doc_id', '')}:{str(item.get('text', ''))[:80]}"

    def add(items: list[dict[str, Any]], limit: int | None = None) -> None:
        added = 0
        for item in items:
            if len(evidence_items) >= max_total:
                return
            key = item_key(item)
            if key in seen:
                continue
            seen.add(key)
            evidence_items.append(item)
            added += 1
            if limit is not None and added >= limit:
                return

    for payload in option_payloads:
        if payload.get("label"):
            add(list(payload.get("evidence_items", []))[:max_per_supported_option])

    covered_docs = {str(item.get("doc_id", "")) for item in evidence_items if item.get("doc_id")}
    for doc_id in doc_ids:
        if doc_id in covered_docs:
            continue
        found = None
        for payload in option_payloads:
            for item in payload.get("evidence_items", []):
                if str(item.get("doc_id", "")) == str(doc_id):
                    found = item
                    break
            if found:
                break
        if found:
            add([found], limit=1)
            covered_docs.add(str(doc_id))

    if not evidence_items and option_payloads:
        add(list(option_payloads[0].get("evidence_items", []))[:max_per_supported_option])

    return evidence_items


def _rank_option_payloads(option_payloads: list[dict[str, Any]], allowed_options: list[str]) -> list[str]:
    order = {option: idx for idx, option in enumerate(allowed_options)}

    def score(payload: dict[str, Any]) -> tuple[float, ...]:
        option = str(payload.get("option", "")).upper()
        label_score = 1.0 if bool(payload.get("label", False)) else 0.0
        verdict = str(payload.get("verdict", "")).lower()
        refuted = bool(payload.get("is_clearly_refuted", False)) or verdict == "refute"
        gate_status = str(payload.get("gate_status", "")).lower()
        gate_score = {"pass": 1.0, "partial": 0.5, "": 0.25, "fail": 0.0}.get(gate_status, 0.25)
        try:
            support_score = float(payload.get("support_score", 0.0))
        except (TypeError, ValueError):
            support_score = 0.0
        evidence_score = 1.0 if payload.get("evidence_items") else 0.0
        return (
            0.0 if refuted else 1.0,
            label_score,
            support_score,
            gate_score,
            evidence_score,
            -float(order.get(option, len(order))),
        )

    ranked = sorted(
        [payload for payload in option_payloads if str(payload.get("option", "")).upper() in order],
        key=score,
        reverse=True,
    )
    seen = set()
    options = []
    for payload in ranked:
        option = str(payload.get("option", "")).upper()
        if option not in seen:
            seen.add(option)
            options.append(option)
    options.extend(option for option in allowed_options if option not in seen)
    return options


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
