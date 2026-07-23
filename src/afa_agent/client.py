from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import requests

from .config import ModelConfig
from .models import TokenUsage


@dataclass(slots=True)
class LLMResponse:
    content: str
    token_usage: TokenUsage
    raw_payload: dict[str, Any]
    response_format_mode: str = "json_object_local_schema"


@dataclass(slots=True)
class LLMUsageLedger:
    calls: list[dict[str, Any]]

    def record(
        self,
        model_name: str,
        usage: TokenUsage,
        *,
        response_format_mode: str = "json_object_local_schema",
    ) -> None:
        self.calls.append(
            {
                "call_index": len(self.calls) + 1,
                "model_name": model_name,
                "response_format_mode": response_format_mode,
                "token_usage": usage.to_dict(),
            }
        )

    def total(self) -> dict[str, int]:
        return {
            field_name: sum(int(call["token_usage"][field_name]) for call in self.calls)
            for field_name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }


_ACTIVE_USAGE_LEDGER: ContextVar[LLMUsageLedger | None] = ContextVar(
    "afa_active_llm_usage_ledger",
    default=None,
)


@contextmanager
def capture_llm_usage() -> Iterator[LLMUsageLedger]:
    """Capture raw usage for every successful API response in the current context."""

    ledger = LLMUsageLedger(calls=[])
    token = _ACTIVE_USAGE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_USAGE_LEDGER.reset(token)


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: Mapping[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMResponse:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        if response_schema is None:
            response_format = {"type": "json_object"}
            response_format_mode = "json_object_local_schema"
        else:
            normalized_schema_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                str(schema_name).strip(),
            ).strip("_")
            if not normalized_schema_name:
                raise ValueError("schema_name must contain an alphanumeric character")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": normalized_schema_name,
                    "strict": True,
                    "schema": dict(response_schema),
                },
            }
            response_format_mode = "native_json_schema_strict"
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": response_format,
        }
        last_error: Exception | None = None
        max_attempts = max(1, self.config.max_retries + 1)
        timeout = (
            max(1, self.config.connect_timeout_seconds),
            max(1, self.config.read_timeout_seconds or self.config.timeout_seconds),
        )
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload),
                    timeout=timeout,
                )
                response.raise_for_status()
                parsed = response.json()
                break
            except requests.ReadTimeout:
                # A timed-out generation may still complete server-side. Sending
                # the same request again would create an unobservable duplicate
                # whose raw usage cannot be declared in the submission ledger.
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    raise
                time.sleep(max(0.0, self.config.retry_backoff_seconds) * (attempt + 1))
        else:
            assert last_error is not None
            raise last_error
        content = parsed["choices"][0]["message"]["content"]
        usage_payload = parsed.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_payload.get("prompt_tokens", 0),
            completion_tokens=usage_payload.get("completion_tokens", 0),
            total_tokens=usage_payload.get("total_tokens", 0),
        )
        _validate_raw_usage(usage)
        ledger = _ACTIVE_USAGE_LEDGER.get()
        if ledger is not None:
            ledger.record(
                self.config.model_name,
                usage,
                response_format_mode=response_format_mode,
            )
        return LLMResponse(
            content=content,
            token_usage=usage,
            raw_payload=parsed,
            response_format_mode=response_format_mode,
        )


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


def _validate_raw_usage(usage: TokenUsage) -> None:
    values = (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("model API usage fields must be non-negative integers")
    if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
        raise ValueError("model API total_tokens must equal prompt_tokens + completion_tokens")
