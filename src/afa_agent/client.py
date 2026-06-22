from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from .config import ModelConfig
from .models import TokenUsage


@dataclass(slots=True)
class LLMResponse:
    content: str
    token_usage: TokenUsage
    raw_payload: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat_json(self, messages: list[dict[str, str]]) -> LLMResponse:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json()
        content = parsed["choices"][0]["message"]["content"]
        usage_payload = parsed.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_payload.get("prompt_tokens", 0),
            completion_tokens=usage_payload.get("completion_tokens", 0),
            total_tokens=usage_payload.get("total_tokens", 0),
        )
        return LLMResponse(content=content, token_usage=usage, raw_payload=parsed)


def extract_json_object(text: str) -> dict[str, Any]:
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = code_block.group(1) if code_block else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        repaired = _regex_fallback_payload(candidate)
        if repaired:
            return repaired
        raise ValueError(f"Could not locate JSON object in: {text}")
    payload = candidate[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = _regex_fallback_payload(payload)
        if repaired:
            return repaired
        raise


def _regex_fallback_payload(text: str) -> dict[str, Any] | None:
    answer_match = re.search(r'"answer"\s*:\s*"([A-D]+)"', text, re.I)
    label_match = re.search(r'"label"\s*:\s*(true|false)', text, re.I)
    reasoning_match = re.search(r'"reasoning_summary"\s*:\s*"([^"]*)"', text, re.S)
    evidence_match = re.search(r'"used_evidence_ids"\s*:\s*\[([^\]]*)\]', text, re.S)
    if not any([answer_match, label_match, reasoning_match, evidence_match]):
        loose_answer = re.search(r"\b([A-D]{1,4})\b", text)
        if loose_answer:
            return {"answer": loose_answer.group(1)}
        return None
    payload: dict[str, Any] = {}
    if answer_match:
        payload["answer"] = answer_match.group(1).upper()
    if label_match:
        payload["label"] = label_match.group(1).lower() == "true"
    if reasoning_match:
        payload["reasoning_summary"] = reasoning_match.group(1).replace('\\"', '"').strip()
    if evidence_match:
        ids = [part.strip() for part in evidence_match.group(1).split(",") if part.strip().isdigit()]
        payload["used_evidence_ids"] = [int(item) for item in ids]
    return payload if payload else None
