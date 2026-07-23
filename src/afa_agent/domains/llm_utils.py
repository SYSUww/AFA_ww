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
    user_parts.append(
        '请输出 JSON，格式为 {"label": true/false, "confidence": 0.0-1.0, '
        '"confidence_reason": "...", "reasoning_summary": "...", "used_evidence_ids": [1,2]}。'
        "label 表示该选项是否应按题干要求被选中，而不是该陈述是否为真；若题干问错误项，陈述错误时 label 应为 true。"
        "confidence 表示仅基于给定证据判断该选项 label 是否可靠的置信度；证据缺关键指标、条款、公式或文档时必须降低。"
        f"reasoning_summary 最后必须明确写“选项{option_key}应选”或“选项{option_key}不应选”，且必须与 label 一致。"
    )
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
            parsed = extract_json_object(response.content)
            parsed["label"] = _parse_option_label(parsed.get("label"))
            reasoning = str(parsed.get("reasoning_summary", ""))
            verdict = _terminal_option_verdict(reasoning, option_key, question_text)
            if verdict is not None and verdict != parsed["label"]:
                parsed["label_before_reconciliation"] = parsed["label"]
                parsed["label"] = verdict
                parsed["label_reconciliation"] = {
                    "strategy": "explicit_terminal_option_verdict",
                    "option": option_key.upper(),
                    "verdict": verdict,
                }
            if parsed["label"] and _reasoning_says_insufficient(reasoning):
                parsed.setdefault("label_before_reconciliation", parsed["label"])
                parsed["label"] = False
                parsed["label_coerced_reason"] = "insufficient_evidence_reasoning"
            return parsed, total_usage
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "\n\n".join(user_parts)
                    + '\n\n上一次输出不符合约定。请只输出一个 JSON 对象；label 必须是 JSON 布尔值 true 或 false，'
                    '例如 {"label": true, "reasoning_summary": "...", "used_evidence_ids": [1]}。',
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
    last_error: Exception | None = None
    for _ in range(2):
        response = client.chat_json(messages)
        total_usage.add(response.token_usage)
        try:
            parsed = extract_json_object(response.content)
            raw_value = parsed.get("answer")
            if not isinstance(raw_value, str):
                raise ValueError("Fallback answer must be a string")
            raw_answer = raw_value.strip().upper()
            cleaned = "".join(ch for ch in sorted(set(raw_answer)) if ch in allowed_options)
            if cleaned:
                return cleaned, total_usage
            raise ValueError("Fallback answer does not contain an allowed option")
        except Exception as exc:
            last_error = exc
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + '\n\n上一次输出不是合法 JSON。请只输出一个 JSON 对象，例如 {"answer": "A"}。',
                },
            ]
    assert last_error is not None
    raise ValueError("Fallback model returned no valid answer after 2 attempts") from last_error


def _parse_option_label(value: Any) -> bool:
    """Accept only JSON booleans, plus numeric JSON 0/1 for provider compatibility."""
    if isinstance(value, bool):
        return value
    if type(value) is int and value in {0, 1}:
        return value == 1
    raise ValueError(f"Option judgment label must be boolean or integer 0/1, got {type(value).__name__}")


def parse_confidence(payload: dict[str, Any], default: float = 0.5) -> float:
    value = payload.get("confidence")
    if value is None:
        value = payload.get("support_score")
    try:
        score = float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        score = default
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _reasoning_says_insufficient(reasoning: str) -> bool:
    compact = re.sub(r"\s+", "", reasoning)
    if not compact:
        return False
    patterns = [
        "证据不足",
        "无法确认",
        "无法判断",
        "无法证实",
        "无法根据现有证据",
        "不能根据现有证据",
        "没有直接证据支持",
        "证据中未提及",
        "未提供",
        "未见任何",
        "未见直接",
        "未包含",
    ]
    return any(pattern in compact for pattern in patterns)


def _terminal_option_verdict(
    reasoning: str,
    option_key: str,
    question_text: str = "",
) -> bool | None:
    """Return an explicit terminal verdict for the current isolated option.

    The matcher is deliberately conservative: it only trusts a conclusion near
    the end of the reasoning, requires an option/verdict phrase, and rejects a
    verdict that names another option. Ambiguous prose is left to the model
    retry path instead of being guessed locally.
    """

    compact = re.sub(r"\s+", "", reasoning).strip()
    if not compact:
        return None
    terminal = compact[-180:].rstrip("。！？；;,.，")
    key = re.escape(option_key.strip().upper())
    option_ref = rf"(?:(?:该|此)?选项(?:{key}|(?![A-D])))"
    subject = option_ref
    gap = r"[^。！？；;]{0,64}"
    selection_object = r"(?:选|选择|选中|选取)"
    generic_direct_negative = re.compile(
        rf"(?:不应该|不应当|不应|不需要|无需|不可以|不能|不可)(?:被)?{selection_object}$"
    )
    generic_direct_positive = re.compile(
        rf"(?<![不无])(?:应该|应当|应|需要|需|可以|可)(?:被)?{selection_object}$"
    )
    if generic_direct_negative.search(terminal):
        return False
    if generic_direct_positive.search(terminal):
        return True

    direct_negative = re.compile(
        rf"{subject}{gap}(?:不应该|不应当|不应|不需要|无需|不可以|不能|不可)(?:被)?{selection_object}$"
    )
    direct_positive = re.compile(
        rf"{subject}{gap}(?<![不无])(?:应该|应当|应|需要|需|可以|可)(?:被)?{selection_object}$"
    )
    direct_negative_match = direct_negative.search(terminal)
    direct_positive_match = direct_positive.search(terminal)
    if direct_negative_match and direct_positive_match:
        return None
    if direct_negative_match:
        return False
    if direct_positive_match:
        return True

    negated_error = re.compile(
        rf"{subject}{gap}(?:并非|不是|并无)(?:错误|不正确|不成立)$"
    )
    negated_error_match = negated_error.search(terminal)
    if negated_error_match:
        content_is_true = True
        return not content_is_true if _question_requests_incorrect_option(question_text) else content_is_true

    content_negative = re.compile(
        rf"{subject}{gap}(?:错误|不正确|不成立|不符合|不一致|不吻合)$"
    )
    content_positive = re.compile(
        rf"{subject}{gap}(?:正确|成立|符合|一致|吻合)$"
    )
    negative_match = content_negative.search(terminal)
    positive_match = content_positive.search(terminal)
    if negative_match and positive_match:
        if negative_match.start() == positive_match.start():
            return None
        content_is_true = positive_match.start() > negative_match.start()
    elif negative_match:
        content_is_true = False
    elif positive_match:
        content_is_true = True
    else:
        return None
    if _question_requests_incorrect_option(question_text):
        return not content_is_true
    return content_is_true


def _question_requests_incorrect_option(question_text: str) -> bool:
    compact = re.sub(r"\s+", "", question_text)
    negative_cues = (
        "错误的是",
        "错误的有",
        "错误的选项",
        "哪些错误",
        "不正确的是",
        "不正确的有",
        "不正确的选项",
        "说法错误",
        "表述错误",
        "判断错误",
        "不符合的是",
        "不符合题意",
        "不属于的是",
        "不包括的是",
    )
    return any(cue in compact for cue in negative_cues)


def finalize_answer(
    answer: str,
    *,
    answer_format: str,
    allowed_options: list[str],
    option_labels: dict[str, bool],
    option_payloads: list[dict[str, Any]] | None = None,
    answer_policy_settings: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Normalize final answer while recording when format constraints force a choice."""
    answer_policy_settings = answer_policy_settings or {}
    metadata: dict[str, Any] = {
        "raw_answer": answer,
        "answer_format": answer_format,
        "format_forced": False,
        "forced_options": [],
        "no_supported_fallback": False,
        "invalid_model_answer": False,
    }
    allowed = [option.upper() for option in allowed_options]
    supported = sorted(option for option, label in option_labels.items() if label and option.upper() in allowed)

    def finish(final_answer: str) -> tuple[str, dict[str, Any]]:
        if answer_policy_settings.get("enabled", False):
            observation = _answer_policy_observation(
                final_answer=final_answer,
                answer_format=answer_format,
                allowed_options=allowed,
                supported_options=supported,
                option_payloads=option_payloads or [],
                metadata=metadata,
            )
            metadata["answer_policy"] = observation
            mode = str(answer_policy_settings.get("mode", "observe"))
            supported_only_formats = {
                str(item)
                for item in answer_policy_settings.get("supported_only_formats", [])
            }
            if (
                mode == "supported_only"
                and answer_policy_settings.get("allow_supported_only_output", False)
                and (not supported_only_formats or answer_format in supported_only_formats)
            ):
                candidate = observation.get("supported_only_answer", "")
                if observation.get("is_supported_only_format_compliant", False):
                    metadata["answer_policy"]["applied_mode"] = mode
                    metadata["answer_policy"]["pre_policy_answer"] = final_answer
                    return candidate, metadata
        return final_answer, metadata

    if answer_format == "tf":
        letters = [ch for ch in answer.upper() if ch in {"A", "B"}]
        if letters:
            return finish(letters[0])
        metadata["invalid_model_answer"] = True
        return finish("A" if option_labels.get("A", False) else "B")

    cleaned = "".join(ch for ch in sorted(set(answer.upper())) if ch in allowed)
    if answer and not cleaned:
        metadata["invalid_model_answer"] = True

    if answer_format == "mcq":
        if len(supported) == 1:
            return finish(supported[0])
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
        return finish(chosen)

    if answer_format != "multi":
        return finish(cleaned)

    selected = set(supported) if len(supported) >= 2 else set(cleaned)
    selected.update(supported)
    if len(selected) < 2:
        fallback_order = (
            allowed
            if answer_policy_settings.get("forced_multi_fallback_order", "ranked") == "allowed"
            else _rank_option_payloads(option_payloads or [], allowed)
        )
        for option in fallback_order:
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
    return finish("".join(sorted(selected)))


def _answer_policy_observation(
    *,
    final_answer: str,
    answer_format: str,
    allowed_options: list[str],
    supported_options: list[str],
    option_payloads: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    selected = [option for option in allowed_options if option in set(final_answer.upper())]
    false_selected = [option for option in selected if option not in supported_options]
    ranked = _rank_option_payloads(option_payloads, allowed_options)
    min_required = 1 if answer_format in {"tf", "mcq"} else (2 if answer_format == "multi" else 0)
    supported_only = "".join(supported_options)
    warnings: list[str] = []
    if not supported_options and answer_format in {"mcq", "multi"}:
        warnings.append("no_supported_option")
    if answer_format == "multi" and len(supported_options) == 1:
        warnings.append("single_supported_multi")
    if false_selected:
        warnings.append("selected_false_option")
    if metadata.get("format_forced"):
        warnings.append("format_forced")
    return {
        "mode": "observe",
        "min_required_options": min_required,
        "final_answer": final_answer,
        "selected_options": selected,
        "supported_options": supported_options,
        "supported_count": len(supported_options),
        "supported_only_answer": supported_only,
        "is_supported_only_format_compliant": len(supported_options) >= min_required if min_required else True,
        "false_selected_options": false_selected,
        "forced_options": list(metadata.get("forced_options", [])),
        "ranked_fallback_options": ranked,
        "warnings": sorted(set(warnings)),
    }


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
