from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import extract_json_object
from afa_agent.domains.llm_utils import format_hits
from afa_agent.models import AnswerResult, Question, TokenUsage


class InsuranceSolver:
    def __init__(self, client, retriever):
        self.client = client
        self.retriever = retriever

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        hits = self.retriever.search(
            question.doc_ids,
            f"{question.question}\n{json.dumps(question.options, ensure_ascii=False)}",
            top_k=4,
            unit_type_boosts={"formula_block": 1.8, "clause_block": 1.1},
            ensure_per_doc=len(question.doc_ids) > 1,
        )
        options_text = "\n".join([f"{key}: {value}" for key, value in question.options.items()])
        calc_hint = "这可能是计算/比较题，请优先核对公式、适用条件、现金价值、账户价值、已交保费、基本保额。" if any(
            keyword in (question.type + question.question) for keyword in ["计算", "推理", "比较", "多少"]
        ) else ""
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": "你是保险条款问答助手。请根据给定证据直接判断最终答案，只能依据证据，输出必须是 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        f"题目：{question.question}\n题型：{question.answer_format}\n选项：\n{options_text}\n\n"
                        f"{calc_hint}\n\n证据：\n{format_hits(hits, max_items=4)}\n\n"
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
            pred_answer = "".join(sorted(set(pred_answer))) or "A"
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
            debug_meta={"doc_ids": question.doc_ids, "type": question.type},
        )
