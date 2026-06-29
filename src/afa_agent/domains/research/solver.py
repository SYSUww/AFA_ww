from __future__ import annotations

from typing import Any

from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, format_hits
from afa_agent.models import AnswerResult, Question, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


class ResearchSolver:
    def __init__(self, client, retriever, strategy: str):
        self.client = client
        self.retriever = retriever
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.answering_settings = get_stage_settings(strategy, "answering")

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        option_debug: list[dict[str, Any]] = []
        query_variants_all: list[str] = []

        for option_key, option_text in question.options.items():
            query_variants = build_query_variants(question, option_key, option_text, self.retrieval_settings)
            query_variants_all.extend(query_variants)
            hits = self.retriever.search(
                question.doc_ids,
                query_variants[0],
                top_k=self.retrieval_settings.get("top_k", 7),
                unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"conclusion_block": 1.6, "paragraph": 1.0}),
                ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(question.doc_ids) > 1),
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            parsed, usage = ask_option_judgment(
                self.client,
                self._system_prompt(),
                question.question,
                question.answer_format,
                option_key,
                option_text,
                format_hits(hits, max_items=self.answering_settings.get("max_hits", 7)),
                self._extra_context(),
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
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 7)),
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
                "你是研报多选题复核器。根据各选项与证据摘要，选出所有正确选项；答案必须至少包含两个选项字母，只输出 JSON。",
                question.question,
                option_payloads,
                question.answer_format,
                list(question.options.keys()),
            )
            total_usage.add(usage)
            pred_answer = answer
        if question.answer_format == "multi":
            pred_answer = self._ensure_multi_minimum(pred_answer, question, option_labels)
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
            debug_meta={
                "doc_ids": question.doc_ids,
                "type": question.type,
                "prompt_template_id": self.answering_settings.get("prompt_template_id", "default"),
                "query_variants": query_variants_all,
                "retrieval_topk": [hit for item in option_debug for hit in item["retrieval_topk"]][: self.retrieval_settings.get("top_k", 7)],
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": [],
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
            },
        )

    def _system_prompt(self) -> str:
        prompt_id = self.answering_settings.get("prompt_template_id", "default")
        if prompt_id == "evidence_strict":
            return "你是行业研报问答助手。你必须区分原文明示信息和推断信息，严格依据证据核对行业结论、数据与趋势判断，只能输出 JSON。"
        if prompt_id == "compact":
            return "你是行业研报问答助手。请基于最关键证据快速判断选项真伪，只输出 JSON。"
        return "你是行业研报问答助手。你需要核对行业结论、数据、市场规模与趋势判断。请区分原文明示信息和推断信息，只能依据证据作答。"

    def _extra_context(self) -> str:
        extra = self.answering_settings.get("extra_context", "").strip()
        base = "如果题目涉及多个研报，请优先对齐每份研报中的对应结论或数据，不要让单一文档覆盖另一份文档。"
        return f"{base}\n{extra}".strip()

    @staticmethod
    def _compose_answer(answer_format: str, option_labels: dict[str, bool]) -> str:
        if answer_format == "tf":
            return "A" if option_labels.get("A", False) else "B"
        selected = sorted([option for option, label in option_labels.items() if label])
        if answer_format == "mcq":
            return selected[0] if selected else "A"
        return "".join(selected)

    @staticmethod
    def _ensure_multi_minimum(answer: str, question: Question, option_labels: dict[str, bool]) -> str:
        selected = {ch for ch in answer.upper() if ch in question.options}
        for option, label in sorted(option_labels.items()):
            if label:
                selected.add(option)
            if len(selected) >= 2:
                break
        for option in sorted(question.options):
            selected.add(option)
            if len(selected) >= 2:
                break
        return "".join(sorted(selected))
