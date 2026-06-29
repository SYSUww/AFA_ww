from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import extract_json_object
from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, format_hits, truncate_text
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
                        f"{calc_hint}\n{self.answering_settings.get('extra_context', '').strip()}\n\n证据：\n{self._format_prompt_hits(hits, default_max_items=4)}\n\n"
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
        if self._should_use_single_call_mcq(question):
            return self._solve_formula_mcq_with_gate(question)

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
                self._format_prompt_hits(hits, default_max_items=6),
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
                    "evidence_items": self._compact_evidence_items(hits),
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

    def _solve_formula_mcq_with_gate(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        query_variants_all: list[str] = []
        option_debug: list[dict[str, Any]] = []
        all_hits: list[Any] = []

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
            all_hits.extend(hits)
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 4)),
                    "evidence_gate": rescue_result.to_dict(),
                }
            )

        combined_hits = self._dedupe_hits(all_hits)
        evidence_text = self._format_prompt_hits(combined_hits, default_max_items=6)
        options_text = "\n".join([f"{key}: {value}" for key, value in question.options.items()])
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
                        "这是保险条款单选题。请先依据证据逐项核对选项，再选择唯一正确选项。"
                        "如涉及计算/排序，必须核对公式、适用条件、现金价值、账户价值、已交保费、基本保险金额、免赔额、赔付比例、赔偿限额等要素；不要因为某个选项排在前面就默认选择。\n\n"
                        f"{self.answering_settings.get('extra_context', '').strip()}\n\n证据：\n{evidence_text}\n\n"
                        '请只输出 JSON，格式为 {"answer": "A", "reasoning_summary": "不超过120字"}。'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        pred_answer = str(parsed.get("answer", "")).strip().upper()
        pred_answer = "".join(ch for ch in pred_answer if ch in question.options)
        pred_answer = pred_answer[:1] if pred_answer[:1] in question.options else "A"
        option_labels = {option: option == pred_answer for option in question.options}
        reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
        evidence_items = self._compact_evidence_items(combined_hits)

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
                "query_variants": query_variants_all,
                "retrieval_topk": serialize_hits(combined_hits, limit=self.retrieval_settings.get("top_k", 4)),
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": [],
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
                "final_consistency_check": {"issues": []},
                "single_call_mcq": True,
            },
        )

    @staticmethod
    def _should_use_single_call_mcq(question: Question) -> bool:
        return question.answer_format == "mcq"

    @staticmethod
    def _dedupe_hits(hits: list[Any]) -> list[Any]:
        deduped = []
        seen = set()
        for hit in hits:
            unit_id = str(getattr(hit, "unit_id", ""))
            key = unit_id.replace("__dup2", "").replace("__dup", "") or f"{getattr(hit, 'doc_id', '')}:{getattr(hit, 'text', '')[:80]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    def _format_prompt_hits(self, hits: list[Any], default_max_items: int) -> str:
        max_items = int(self.answering_settings.get("max_hits", default_max_items))
        max_chars = self.answering_settings.get("max_hit_chars")
        max_chars = int(max_chars) if max_chars else None
        focus_terms = self._focus_terms("")
        prompt_hits = self._select_prompt_hits(hits, focus_terms, max_items)
        return format_hits(prompt_hits, max_items=len(prompt_hits), max_chars=max_chars, focus_terms=focus_terms)

    def _compact_evidence_items(self, hits: list[Any]) -> list[dict[str, Any]]:
        max_items = int(self.answering_settings.get("max_hits", 6))
        max_chars = int(self.answering_settings.get("max_hit_chars", 0) or 0)
        focus_terms = self._focus_terms("")
        prompt_hits = self._select_prompt_hits(hits, focus_terms, max_items)
        rows = []
        for hit in prompt_hits:
            row = hit.to_dict()
            text = str(row.get("text", ""))
            row["text"] = truncate_text(text, max_chars=max_chars, focus_terms=focus_terms)
            rows.append(row)
        return rows

    @staticmethod
    def _focus_terms(option_text: str) -> list[str]:
        base_terms = [
            "较大者",
            "较大值",
            "下列两者",
            "给付比例",
            "基本保额",
            "基本保险金额",
            "身故保险金额",
            "身故保险金",
            "所交保险费",
            "已交保费",
            "免赔额",
            "赔付比例",
            "赔偿限额",
            "保险金额",
            "医疗费用",
            "报销",
            "赔偿",
            "等待期",
            "责任免除",
            "特定药品",
            "保险责任",
            "账户价值",
            "保单账户价值",
            "个人账户价值",
            "现金价值",
            "养老年金",
            "保险金",
            "给付",
            "比例",
            "身故",
        ]
        products = re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚)[\u4e00-\u9fa5A-Za-z0-9]{2,18}", option_text)
        seen = set()
        terms = []
        for term in [*products, *base_terms]:
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    @staticmethod
    def _rank_prompt_hits(hits: list[Any], focus_terms: list[str]) -> list[Any]:
        def score(hit: Any) -> tuple[int, float]:
            text = getattr(hit, "text", "")
            title = " ".join(getattr(hit, "title_path", []))
            haystack = f"{title}\n{text}"
            term_score = sum(1 for term in focus_terms if term in haystack)
            term_score += 3 if "身故保险金" in text else 0
            term_score += 2 if "较大者" in text or "较大值" in text or "下列两者" in text or "给付比例" in text else 0
            term_score += 1 if "现金价值" in text and ("账户价值" in text or "基本保险金额" in text) else 0
            return term_score, float(getattr(hit, "score", 0.0))

        return sorted(hits, key=score, reverse=True)

    @classmethod
    def _select_prompt_hits(cls, hits: list[Any], focus_terms: list[str], max_items: int) -> list[Any]:
        ranked = cls._rank_prompt_hits(hits, focus_terms)
        selected: list[Any] = []
        seen_units: set[str] = set()
        covered_docs: set[str] = set()

        def base_unit_id(hit: Any) -> str:
            return str(getattr(hit, "unit_id", "")).replace("__dup2", "").replace("__dup", "")

        def add(hit: Any) -> bool:
            unit_id = base_unit_id(hit)
            if unit_id and unit_id in seen_units:
                return False
            selected.append(hit)
            if unit_id:
                seen_units.add(unit_id)
            doc_id = str(getattr(hit, "doc_id", ""))
            if doc_id:
                covered_docs.add(doc_id)
            return True

        for hit in ranked:
            if len(selected) >= max_items:
                return selected
            doc_id = str(getattr(hit, "doc_id", ""))
            if doc_id and doc_id not in covered_docs:
                add(hit)
        for hit in ranked:
            if len(selected) >= max_items:
                break
            add(hit)
        return selected

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
            return "你是保险条款问答助手。你必须严格依据证据核对触发条件、给付规则与公式，不得使用常识补全，输出必须是 JSON；reasoning_summary 不超过 80 字，不要复述大段条款。"
        if prompt_id == "compact":
            return "你是保险条款问答助手。请用最关键的证据快速判断最终答案，输出必须是 JSON；reasoning_summary 不超过 80 字。"
        return "你是保险条款问答助手。请根据给定证据直接判断最终答案，只能依据证据，输出必须是 JSON；reasoning_summary 不超过 80 字。"

    @staticmethod
    def _ensure_multi_minimum(answer: str, question: Question) -> str:
        selected = {ch for ch in answer.upper() if ch in question.options}
        for option in sorted(question.options):
            selected.add(option)
            if len(selected) >= 2:
                break
        return "".join(sorted(selected))
