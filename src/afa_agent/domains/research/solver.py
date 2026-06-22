from __future__ import annotations

from typing import Any

from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, format_hits
from afa_agent.models import AnswerResult, Question, TokenUsage


class ResearchSolver:
    def __init__(self, client, retriever):
        self.client = client
        self.retriever = retriever

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []

        for option_key, option_text in question.options.items():
            hits = self.retriever.search(
                question.doc_ids,
                f"{question.question}\n{option_text}",
                top_k=7,
                unit_type_boosts={"conclusion_block": 1.6, "paragraph": 1.0},
                ensure_per_doc=len(question.doc_ids) > 1,
            )
            parsed, usage = ask_option_judgment(
                self.client,
                "你是行业研报问答助手。你需要核对行业结论、数据、市场规模与趋势判断。请区分原文明示信息和推断信息，只能依据证据作答。",
                question.question,
                question.answer_format,
                option_key,
                option_text,
                format_hits(hits, max_items=7),
                "如果题目涉及多个研报，请优先对齐每份研报中的对应结论或数据，不要让单一文档覆盖另一份文档。",
            )
            total_usage.add(usage)
            label = bool(parsed.get("label", False))
            reasoning = str(parsed.get("reasoning_summary", "")).strip()
            option_labels[option_key] = label
            option_payloads.append(
                {
                    "option": option_key,
                    "label": label,
                    "reasoning_summary": reasoning,
                    "evidence_items": [hit.to_dict() for hit in hits],
                }
            )
            reasoning_chunks.append(f"{option_key}: {reasoning}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        if question.answer_format == "mcq" and len([k for k, v in option_labels.items() if v]) != 1:
            answer, usage = ask_answer_fallback(
                self.client,
                "你是研报单选题裁决器。根据各选项与证据摘要，选出唯一最可能正确的选项，只输出 JSON。",
                question.question,
                option_payloads,
                question.answer_format,
                list(question.options.keys()),
            )
            total_usage.add(usage)
            pred_answer = answer[:1]
        elif question.answer_format == "multi" and not pred_answer:
            answer, usage = ask_answer_fallback(
                self.client,
                "你是研报多选题复核器。根据各选项与证据摘要，选出所有正确选项，只输出 JSON。",
                question.question,
                option_payloads,
                question.answer_format,
                list(question.options.keys()),
            )
            total_usage.add(usage)
            pred_answer = answer
        elif question.answer_format == "tf":
            pred_answer = "A" if option_labels.get("A", False) else "B"

        evidence_items = []
        for payload in option_payloads:
            if payload["label"]:
                evidence_items.extend(payload["evidence_items"][:3])
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

    @staticmethod
    def _compose_answer(answer_format: str, option_labels: dict[str, bool]) -> str:
        if answer_format == "tf":
            return "A" if option_labels.get("A", False) else "B"
        selected = sorted([option for option, label in option_labels.items() if label])
        if answer_format == "mcq":
            return selected[0] if selected else "A"
        return "".join(selected)
