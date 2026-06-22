from __future__ import annotations

import json
from typing import Any

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.models import AnswerResult, Question, TokenUsage


class RegulatorySolver:
    def __init__(self, client: OpenAICompatibleClient, retriever):
        self.client = client
        self.retriever = retriever

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []

        for option_key, option_text in question.options.items():
            hits = self.retriever.search(question.doc_ids, f"{question.question}\n{option_text}", top_k=6)
            payload = self._judge_option(question, option_key, option_text, hits)
            total_usage.add(payload["token_usage"])
            option_labels[option_key] = payload["label"]
            option_payloads.append(
                {
                    "option": option_key,
                    "label": payload["label"],
                    "reasoning_summary": payload["reasoning_summary"],
                    "evidence_items": [hit.to_dict() for hit in hits],
                }
            )
            reasoning_chunks.append(f"{option_key}: {payload['reasoning_summary']}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        if question.answer_format == "mcq" and (len([k for k, v in option_labels.items() if v]) != 1):
            pred_answer = self._fallback_single_choice(question, option_payloads, total_usage)
        if question.answer_format == "multi" and not pred_answer:
            pred_answer = self._fallback_multi_choice(question, option_payloads, total_usage)

        if question.answer_format == "tf":
            pred_answer = "A" if option_labels.get("A", False) else "B"

        evidence_items = []
        for item in option_payloads:
            if item["label"]:
                evidence_items.extend(item["evidence_items"][:3])
        if not evidence_items and option_payloads:
            evidence_items.extend(option_payloads[0]["evidence_items"][:3])

        return AnswerResult(
            qid=question.qid,
            domain=question.domain,
            question_type=question.answer_format,
            pred_answer=pred_answer,
            option_labels=option_labels,
            evidence_items=evidence_items,
            reasoning_summary=" | ".join(reasoning_chunks),
            token_usage=total_usage,
            debug_meta={"doc_ids": question.doc_ids, "type": question.type},
        )

    def _judge_option(self, question: Question, option_key: str, option_text: str, hits) -> dict[str, Any]:
        evidence_lines = []
        for idx, hit in enumerate(hits, start=1):
            title_path = " > ".join(hit.title_path)
            evidence_lines.append(f"[{idx}] {hit.doc_id} | {title_path}\n{hit.text}")
        messages = [
            {
                "role": "system",
                "content": (
                    "你是金融法规问答助手。你只能依据给定证据判断选项真伪，"
                    "不能使用常识补全。输出必须是 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.question}\n"
                    f"题型：{question.answer_format}\n"
                    f"选项 {option_key}：{option_text}\n\n"
                    f"证据：\n{'\n\n'.join(evidence_lines)}\n\n"
                    '请输出 JSON，格式为 {"label": true/false, "reasoning_summary": "...", "used_evidence_ids": [1,2]}。'
                ),
            },
        ]
        response = self.client.chat_json(messages)
        parsed = extract_json_object(response.content)
        return {
            "label": bool(parsed.get("label", False)),
            "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip(),
            "token_usage": response.token_usage,
        }

    def _fallback_single_choice(
        self,
        question: Question,
        option_payloads: list[dict[str, Any]],
        total_usage: TokenUsage,
    ) -> str:
        summary = json.dumps(option_payloads, ensure_ascii=False)
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "你是单选题裁决器。根据各选项判断摘要选出唯一最可能正确的字母，只输出 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        f"题目：{question.question}\n"
                        f"选项摘要：{summary}\n"
                        '输出格式：{"answer": "A"}'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        answer = str(parsed.get("answer", "")).strip().upper()
        return answer[:1] if answer[:1] in question.options else "A"

    def _fallback_multi_choice(
        self,
        question: Question,
        option_payloads: list[dict[str, Any]],
        total_usage: TokenUsage,
    ) -> str:
        summary = json.dumps(option_payloads, ensure_ascii=False)
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "你是多选题复核器。根据各选项判断摘要挑出所有正确选项，只输出 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        f"题目：{question.question}\n"
                        f"选项摘要：{summary}\n"
                        '输出格式：{"answer": "AC"}'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        answer = "".join(sorted(set(str(parsed.get("answer", "")).strip().upper())))
        cleaned = "".join(ch for ch in answer if ch in question.options)
        if cleaned:
            return cleaned
        return sorted(question.options.keys())[0]

    @staticmethod
    def _compose_answer(answer_format: str, option_labels: dict[str, bool]) -> str:
        if answer_format == "tf":
            return "A" if option_labels.get("A", False) else "B"
        selected = [option for option, label in option_labels.items() if label]
        selected.sort()
        if answer_format == "mcq":
            return selected[0] if selected else "A"
        return "".join(selected)
