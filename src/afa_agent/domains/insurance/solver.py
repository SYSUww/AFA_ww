from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import extract_json_object
from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, format_hits
from afa_agent.evidence_gate import answer_consistency_issues, evaluate_evidence, gate_enabled, rescue_evidence
from afa_agent.models import AnswerResult, Question, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


class InsuranceSolver:
    def __init__(self, client, retriever, strategy: str):
        self.client = client
        self.retriever = retriever
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.answering_settings = get_stage_settings(strategy, "answering")
        self.gate_settings = get_stage_settings(strategy, "evidence_gate")

    def solve(self, question: Question) -> AnswerResult:
        if gate_enabled(self.gate_settings):
            return self._solve_with_gate(question)

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

    def _solve_with_gate(self, question: Question) -> AnswerResult:
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
                top_k=self.retrieval_settings.get("top_k", 4),
                unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"formula_block": 1.8, "clause_block": 1.1}),
                ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(question.doc_ids) > 1),
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            initial_gate = evaluate_evidence(question, option_key, option_text, hits, question.domain, self.gate_settings)
            rescue_result = rescue_evidence(
                question=question,
                option_key=option_key,
                option_text=option_text,
                domain=question.domain,
                retriever=self.retriever,
                initial_hits=hits,
                retrieval_settings=self.retrieval_settings,
                gate_settings=self.gate_settings,
                initial_gate=initial_gate,
            )
            hits = rescue_result.hits
            gate_debug = rescue_result.to_dict()
            calc_hint = "这可能是计算/比较题，请优先核对公式、适用条件、现金价值、账户价值、已交保费、基本保额。" if any(
                keyword in (question.type + question.question + option_text) for keyword in ["计算", "推理", "比较", "多少", "排序"]
            ) else ""
            parsed, usage = ask_option_judgment(
                self.client,
                self._system_prompt(),
                question.question,
                question.answer_format,
                option_key,
                option_text,
                format_hits(hits, max_items=self.answering_settings.get("max_hits", 6)),
                f"{calc_hint}\n{self.answering_settings.get('extra_context', '').strip()}".strip(),
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
                    "gate_status": gate_debug.get("final_gate", {}).get("status", ""),
                    "gate_reasons": gate_debug.get("final_gate", {}).get("reasons", []),
                }
            )
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 4)),
                    "evidence_gate": gate_debug,
                }
            )
            reasoning_chunks.append(f"{option_key}: {reasoning}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        if question.answer_format == "mcq" and len([k for k, value in option_labels.items() if value]) != 1:
            answer, usage = ask_answer_fallback(
                self.client,
                "你是保险单选题裁决器。根据各选项证据摘要选出唯一最可能正确的字母，只输出 JSON。",
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
                "你是保险多选题复核器。根据各选项证据摘要选出所有正确选项；答案必须至少包含两个选项字母，只输出 JSON。",
                question.question,
                option_payloads,
                question.answer_format,
                list(question.options.keys()),
            )
            total_usage.add(usage)
            pred_answer = answer
        if question.answer_format == "multi":
            pred_answer = self._ensure_multi_minimum(pred_answer, question)
        elif question.answer_format == "tf":
            pred_answer = "A" if option_labels.get("A", False) else "B"

        consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)
        if consistency_issues and self.gate_settings.get("final_consistency_retry", True):
            answer, usage = ask_answer_fallback(
                self.client,
                "你是保险答案一致性复核器。只能选择 label=true 且 evidence gate 未失败的选项；必须核对产品、条件和公式，只输出 JSON。",
                question.question,
                option_payloads,
                question.answer_format,
                list(question.options.keys()),
            )
            total_usage.add(usage)
            pred_answer = answer[:1] if question.answer_format == "mcq" else answer
            if question.answer_format == "multi":
                pred_answer = self._ensure_multi_minimum(pred_answer, question)
        consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)

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
                "retrieval_topk": [hit for item in option_debug for hit in item["retrieval_topk"]][: self.retrieval_settings.get("top_k", 4)],
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": [],
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
                "final_consistency_check": {"issues": consistency_issues},
            },
        )

    @staticmethod
    def _compose_answer(answer_format: str, option_labels: dict[str, bool]) -> str:
        if answer_format == "tf":
            return "A" if option_labels.get("A", False) else "B"
        selected = sorted([option for option, label in option_labels.items() if label])
        if answer_format == "mcq":
            return selected[0] if selected else "A"
        return "".join(selected)

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
