from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import extract_json_object
from afa_agent.domains.llm_utils import format_hits
from afa_agent.models import AnswerResult, Question, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


class InsuranceSolver:
    def __init__(self, client, retriever, strategy: str):
        self.client = client
        self.retriever = retriever
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.answering_settings = get_stage_settings(strategy, "answering")

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        query_variants = build_query_variants(
            question,
            "ALL",
            json.dumps(question.options, ensure_ascii=False),
            self.retrieval_settings,
        )
        hits = self.retriever.search(
            question.doc_ids,
            query_variants[0],
            top_k=self.retrieval_settings.get("top_k", 4),
            unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"formula_block": 1.8, "clause_block": 1.1}),
            ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(question.doc_ids) > 1),
            expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
        )
        options_text = "\n".join([f"{key}: {value}" for key, value in question.options.items()])
        calc_hint = "这可能是计算/比较题，请优先核对公式、适用条件、现金价值、账户价值、已交保费、基本保额。" if any(
            keyword in (question.type + question.question) for keyword in ["计算", "推理", "比较", "多少"]
        ) else ""
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": (
                        f"题目：{question.question}\n题型：{question.answer_format}\n选项：\n{options_text}\n\n"
                        f"{calc_hint}\n{self.answering_settings.get('extra_context', '').strip()}\n\n证据：\n{format_hits(hits, max_items=self.answering_settings.get('max_hits', 4))}\n\n"
                        '请输出 JSON，格式为 {"answer": "A", "reasoning_summary": "..."} 或 {"answer": "AC", "reasoning_summary": "..."}。'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        pred_answer = str(parsed.get("answer", "")).strip().upper()
        pred_answer = "".join(ch for ch in pred_answer if ch in question.options)
        if question.answer_format == "mcq":
            pred_answer = pred_answer[:1] if pred_answer[:1] in question.options else "A"
        elif question.answer_format == "multi":
            pred_answer = self._ensure_multi_minimum(pred_answer, question)
        reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
        option_labels = {option: option in pred_answer for option in question.options}
        evidence_items = [hit.to_dict() for hit in hits[:4]]

        return AnswerResult(
            qid=question.qid,
            domain=question.domain,
            question_type=question.answer_format,
            pred_answer=pred_answer,
            option_labels=option_labels,
            evidence_items=evidence_items,
            reasoning_summary=reasoning_summary,
            token_usage=total_usage,
            debug_meta={
                "doc_ids": question.doc_ids,
                "type": question.type,
                "prompt_template_id": self.answering_settings.get("prompt_template_id", "default"),
                "query_variants": query_variants,
                "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 4)),
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": [],
                "consistency_answers": [pred_answer],
            },
        )

    def _system_prompt(self) -> str:
        prompt_id = self.answering_settings.get("prompt_template_id", "default")
        if prompt_id == "evidence_strict":
            return "你是保险条款问答助手。你必须严格依据证据核对触发条件、给付规则与公式，不得使用常识补全，输出必须是 JSON。"
        if prompt_id == "compact":
            return "你是保险条款问答助手。请用最关键的证据快速判断最终答案，输出必须是 JSON。"
        return "你是保险条款问答助手。请根据给定证据直接判断最终答案，只能依据证据，输出必须是 JSON。"

    @staticmethod
    def _ensure_multi_minimum(answer: str, question: Question) -> str:
        selected = {ch for ch in answer.upper() if ch in question.options}
        for option in sorted(question.options):
            selected.add(option)
            if len(selected) >= 2:
                break
        return "".join(sorted(selected))
