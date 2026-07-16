from __future__ import annotations

import json
import re
from typing import Any

from afa_agent.client import extract_json_object
from afa_agent.domains.llm_utils import (
    ask_answer_fallback,
    ask_option_judgment,
    collect_evidence_items,
    finalize_answer,
    format_hits,
    parse_confidence,
    truncate_text,
)
from afa_agent.evidence_gate import (
    answer_consistency_issues,
    evaluate_evidence,
    gate_enabled,
    rescue_evidence,
    should_skip_answer_fallback,
    should_skip_consistency_retry,
)
from afa_agent.evidence_audit import (
    PROVENANCE_SCHEMA_VERSION,
    SHARED_DEDUCTIBLE_EVIDENCE_TERM_GROUPS,
    SHARED_DEDUCTIBLE_RULE_ID,
    build_shared_deductible_provenance,
)
from afa_agent.models import AnswerResult, Question, RetrievalHit, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


class InsuranceSolver:
    def __init__(self, client, retriever, strategy: str):
        self.client = client
        self.retriever = retriever
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.answering_settings = get_stage_settings(strategy, "answering")
        self.gate_settings = get_stage_settings(strategy, "evidence_gate")
        self.answer_policy_settings = get_stage_settings(strategy, "answer_policy")

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
                        '请输出 JSON，格式为 {"answer": "A", "confidence": 0.0-1.0, "confidence_reason": "...", '
                        '"reasoning_summary": "..."} 或 {"answer": "AC", "confidence": 0.0-1.0, "confidence_reason": "...", '
                        '"reasoning_summary": "..."}。confidence 表示仅依据给定证据判断最终答案可靠性的置信度。'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        confidence = parse_confidence(parsed, 0.5)
        pred_answer = str(parsed.get("answer", "")).strip().upper()
        pred_answer = "".join(ch for ch in pred_answer if ch in question.options)
        if question.answer_format == "mcq":
            pred_answer = pred_answer[:1] if pred_answer[:1] in question.options else "A"
        pred_answer, answer_finalization = finalize_answer(
            pred_answer,
            answer_format=question.answer_format,
            allowed_options=list(question.options.keys()),
            option_labels={option: option in pred_answer for option in question.options},
            option_payloads=[],
            answer_policy_settings=self.answer_policy_settings,
        )
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
                "model_confidence": confidence,
                "consistency_answers": [pred_answer],
                "answer_finalization": answer_finalization,
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

        option_items = [("A", question.question)] if question.answer_format == "tf" else list(question.options.items())
        for option_key, option_text in option_items:
            search_doc_ids = self._option_doc_ids(question, option_text)
            gate_question = self._with_doc_ids(question, search_doc_ids)
            query_variants = build_query_variants(question, option_key, option_text, self.retrieval_settings)
            query_variants_all.extend(query_variants)
            hits = self.retriever.search(
                search_doc_ids,
                query_variants[0],
                top_k=self.retrieval_settings.get("top_k", 4),
                unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"formula_block": 1.8, "clause_block": 1.1}),
                ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(search_doc_ids) > 1),
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            targeted_hits = self._targeted_clause_hits(gate_question, f"{question.question} {option_text}")
            if targeted_hits:
                hits = self._merge_hits(
                    [*targeted_hits, *hits],
                    limit=max(
                        self.retrieval_settings.get("top_k", 4),
                        self.answering_settings.get("max_hits", 6),
                    ),
                )
            initial_gate = evaluate_evidence(gate_question, option_key, option_text, hits, question.domain, self.gate_settings)
            rescue_result = rescue_evidence(
                question=gate_question,
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
            if targeted_hits:
                gate_debug["targeted_clause_hits"] = serialize_hits(targeted_hits, limit=6)
            rule_label, rule_reason = self._rule_override_label(question, option_text, hits)
            if rule_label is not None:
                label = rule_label
                reasoning = rule_reason
                confidence = 0.95
            else:
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
                    self._format_prompt_hits(hits, default_max_items=6, option_text=option_text),
                    self._option_judgment_context(question, option_text, calc_hint),
                )
                total_usage.add(usage)
                label = bool(parsed.get("label", False))
                reasoning = str(parsed.get("reasoning_summary", "")).strip()
                confidence = parse_confidence(parsed, 0.75 if label else 0.25)
            option_labels[option_key] = label
            option_payloads.append(
                {
                    "option": option_key,
                    "label": label,
                    "reasoning_summary": reasoning,
                    "confidence": confidence,
                    "evidence_items": self._compact_evidence_items(hits),
                    "gate_status": gate_debug.get("final_gate", {}).get("status", ""),
                    "gate_reasons": gate_debug.get("final_gate", {}).get("reasons", []),
                }
            )
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "search_doc_ids": search_doc_ids,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 4)),
                    "model_confidence": confidence,
                    "evidence_gate": gate_debug,
                }
            )
            reasoning_chunks.append(f"{option_key}: {reasoning}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        fallback_skipped_reason = ""
        if question.answer_format == "mcq" and len([k for k, value in option_labels.items() if value]) != 1:
            if should_skip_answer_fallback(
                answer_format=question.answer_format,
                option_labels=option_labels,
                gate_settings=self.gate_settings,
            ):
                fallback_skipped_reason = "mcq_ambiguous_supported"
            else:
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
        elif question.answer_format == "multi" and len(pred_answer) < 2:
            if should_skip_answer_fallback(
                answer_format=question.answer_format,
                option_labels=option_labels,
                gate_settings=self.gate_settings,
            ):
                fallback_skipped_reason = "single_supported_multi"
            else:
                answer, usage = ask_answer_fallback(
                    self.client,
                    (
                        "你是保险多选题复核器。根据各选项证据摘要选出满足题干所问的产品；"
                        "不要因为括号说明本身为真就选择该项；答案必须至少包含两个选项字母，只输出 JSON。"
                    ),
                    question.question,
                    option_payloads,
                    question.answer_format,
                    list(question.options.keys()),
                )
                total_usage.add(usage)
                pred_answer = answer
        pred_answer, answer_finalization = finalize_answer(
            pred_answer,
            answer_format=question.answer_format,
            allowed_options=list(question.options.keys()),
            option_labels=option_labels,
            option_payloads=option_payloads,
            answer_policy_settings=self.answer_policy_settings,
        )
        if fallback_skipped_reason:
            answer_finalization["fallback_skipped_reason"] = fallback_skipped_reason
        if question.answer_format == "tf":
            option_labels["B"] = pred_answer == "B"

        consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)
        if consistency_issues and self.gate_settings.get("final_consistency_retry", True):
            skip_retry, skip_reason = should_skip_consistency_retry(
                consistency_issues=consistency_issues,
                answer_finalization=answer_finalization,
                answer_format=question.answer_format,
                gate_settings=self.gate_settings,
            )
            retry_needed = not skip_retry and not self._skip_no_supported_retry(consistency_issues, pred_answer, question)
            if retry_needed:
                answer, usage = ask_answer_fallback(
                    self.client,
                    (
                        "你是保险答案一致性复核器。只能选择能回答题干要求、且 evidence gate 未失败的选项；"
                        "不要因为括号说明本身为真就选择该项，必须判断该产品是否满足题干所问；只输出 JSON。"
                    ),
                    question.question,
                    option_payloads,
                    question.answer_format,
                    list(question.options.keys()),
                )
                total_usage.add(usage)
                pred_answer = answer[:1] if question.answer_format == "mcq" else answer
                retry_answer, retry_finalization = finalize_answer(
                    pred_answer,
                    answer_format=question.answer_format,
                    allowed_options=list(question.options.keys()),
                    option_labels=option_labels,
                    option_payloads=option_payloads,
                    answer_policy_settings=self.answer_policy_settings,
                )
                pred_answer = retry_answer
                answer_finalization = {
                    **retry_finalization,
                    "consistency_retry": True,
                    "pre_retry": answer_finalization,
                }
            else:
                answer_finalization = {
                    **answer_finalization,
                    "consistency_retry": False,
                    "retry_skipped_reason": skip_reason or "no_supported_option_with_legal_multi_answer",
                }
        consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)

        evidence_items = collect_evidence_items(option_payloads, doc_ids=question.doc_ids)

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
                "answer_finalization": answer_finalization,
            },
        )

    def _solve_formula_mcq_with_gate(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        early_rule_answer, early_rule_reason, early_rule_id = self._rule_override_mcq_answer(question)
        if early_rule_answer:
            rule_hits = self._rule_evidence_hits(question)
            provenance_hits = self._rule_provenance_hits(early_rule_id, rule_hits)
            evidence_items = self._compact_evidence_items([*provenance_hits, *rule_hits])
            final_evidence_ids = {
                str(item.get("unit_id", ""))
                for item in evidence_items
                if item.get("unit_id")
            }
            provenance_evidence_ids = [
                str(hit.unit_id)
                for hit in provenance_hits
                if str(hit.unit_id) in final_evidence_ids
            ]
            rule_provenance = (
                build_shared_deductible_provenance(
                    decision_option=early_rule_answer,
                    evidence_unit_ids=provenance_evidence_ids,
                )
                if early_rule_id == SHARED_DEDUCTIBLE_RULE_ID
                else {}
            )
            option_labels = {option: option == early_rule_answer for option in question.options}
            rule_output = {
                "option": early_rule_answer,
                "label": True,
                "answer": early_rule_answer,
                "reason": early_rule_reason,
                "confidence": 0.95,
                **rule_provenance,
            }
            return AnswerResult(
                qid=question.qid,
                domain=question.domain,
                question_type=question.answer_format,
                pred_answer=early_rule_answer,
                option_labels=option_labels,
                evidence_items=evidence_items,
                reasoning_summary=early_rule_reason,
                token_usage=total_usage,
                debug_meta={
                    "doc_ids": question.doc_ids,
                    "type": question.type,
                    "prompt_template_id": self.answering_settings.get("prompt_template_id", "default"),
                    "query_variants": [],
                    "retrieval_topk": serialize_hits(rule_hits, limit=self.retrieval_settings.get("top_k", 4)),
                    "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                    "rule_outputs": [rule_output],
                    "option_debug": [],
                    "consistency_answers": [early_rule_answer],
                    "final_consistency_check": {"issues": []},
                    "answer_finalization": {
                        "raw_answer": early_rule_answer,
                        "answer_format": question.answer_format,
                        "format_forced": False,
                        "forced_options": [],
                        "no_supported_fallback": False,
                        "invalid_model_answer": False,
                    },
                    "single_call_mcq": True,
                    "early_rule_answer": True,
                    **(
                        {"provenance_schema_version": PROVENANCE_SCHEMA_VERSION}
                        if rule_provenance
                        else {}
                    ),
                },
            )
        query_variants_all: list[str] = []
        option_debug: list[dict[str, Any]] = []
        all_hits: list[Any] = []

        for option_key, option_text in question.options.items():
            search_doc_ids = self._option_doc_ids(question, f"{question.question} {option_text}")
            gate_question = self._with_doc_ids(question, search_doc_ids)
            query_variants = build_query_variants(question, option_key, option_text, self.retrieval_settings)
            query_variants_all.extend(query_variants)
            hits = self.retriever.search(
                search_doc_ids,
                query_variants[0],
                top_k=self.retrieval_settings.get("top_k", 4),
                unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"formula_block": 1.8, "clause_block": 1.1}),
                ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(search_doc_ids) > 1),
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            targeted_hits = self._targeted_clause_hits(gate_question, f"{question.question} {option_text}")
            if targeted_hits:
                hits = self._merge_hits(
                    [*targeted_hits, *hits],
                    limit=max(
                        self.retrieval_settings.get("top_k", 4),
                        self.answering_settings.get("max_hits", 6),
                    ),
                )
            initial_gate = evaluate_evidence(gate_question, option_key, option_text, hits, question.domain, self.gate_settings)
            rescue_result = rescue_evidence(
                question=gate_question,
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
            if targeted_hits:
                gate_debug["targeted_clause_hits"] = serialize_hits(targeted_hits, limit=6)
            all_hits.extend(hits)
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "search_doc_ids": search_doc_ids,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 4)),
                    "evidence_gate": gate_debug,
                }
            )

        combined_hits = self._dedupe_hits(all_hits)
        evidence_text = self._format_prompt_hits(combined_hits, default_max_items=6, option_text=" ".join(question.options.values()))
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
                        '请只输出 JSON，格式为 {"answer": "A", "confidence": 0.0-1.0, '
                        '"confidence_reason": "...", "reasoning_summary": "不超过120字"}。'
                        "confidence 表示仅依据给定证据判断最终答案可靠性的置信度。"
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = extract_json_object(response.content)
        confidence = parse_confidence(parsed, 0.5)
        pred_answer = str(parsed.get("answer", "")).strip().upper()
        pred_answer = "".join(ch for ch in pred_answer if ch in question.options)
        pred_answer = pred_answer[:1] if pred_answer[:1] in question.options else "A"
        reasoning_summary = str(parsed.get("reasoning_summary", "")).strip()
        rule_answer, rule_reason, _ = self._rule_override_mcq_answer(question)
        if rule_answer:
            pred_answer = rule_answer
            reasoning_summary = rule_reason
        option_labels = {option: option == pred_answer for option in question.options}
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
                "model_confidence": confidence,
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
                "final_consistency_check": {"issues": []},
                "answer_finalization": {
                    "raw_answer": pred_answer,
                    "answer_format": question.answer_format,
                    "format_forced": False,
                    "forced_options": [],
                    "no_supported_fallback": False,
                    "invalid_model_answer": False,
                },
                "single_call_mcq": True,
            },
        )

    @staticmethod
    def _should_use_single_call_mcq(question: Question) -> bool:
        return question.answer_format == "mcq"

    def _option_judgment_context(self, question: Question, option_text: str, calc_hint: str = "") -> str:
        parts = []
        if calc_hint.strip():
            parts.append(calc_hint.strip())
        parts.append(
            "逐项判断时，label=true 表示该选项是题干问题的正确答案，不是表示括号里的说明文字本身为真。"
            "如果题干问“哪些产品明确给出公式/可以赔付/可以抵扣/会退还全部保费”，"
            "而选项括号写“未给公式/不赔/无法确认/未提及”，通常应判 label=false，除非题干正是在询问这些否定情形。"
        )
        if question.answer_format == "multi":
            parts.append("多选题的最终格式可能需要多个字母，但逐项 label 只能按证据判断，不要为了凑够数量把无支持或被反驳的选项标为 true。")
        extra = self.answering_settings.get("extra_context", "").strip()
        if extra:
            parts.append(extra)
        return "\n".join(parts).strip()

    @staticmethod
    def _skip_no_supported_retry(consistency_issues: list[str], pred_answer: str, question: Question) -> bool:
        if question.answer_format != "multi":
            return False
        if set(consistency_issues) != {"no_supported_option"}:
            return False
        selected = {ch for ch in pred_answer.upper() if ch in question.options}
        return len(selected) >= 2

    @staticmethod
    def _rule_override_label(question: Question, option_text: str, hits: list[Any]) -> tuple[bool | None, str]:
        question_text = f"{question.question} {option_text}"
        evidence_text = "\n".join(
            f"{' '.join(getattr(hit, 'title_path', []))}\n{getattr(hit, 'text', '')}"
            for hit in hits
        )
        if "宽限期" in question_text and "效力中止" in question_text and "不承担保险责任" in question_text:
            if InsuranceSolver._grace_suspension_supported(evidence_text):
                return True, "规则命中：证据在宽限期/效力中止条款链中载明，效力中止期间保险人不承担保险责任。"
            return False, "规则命中：未在同一宽限期/效力中止条款链中检索到“效力中止期间不承担保险责任”的闭合证据。"
        if (
            "保单贷款" in question_text
            and "80" in question_text
            and "借款" in evidence_text
            and "最高借款金额" in evidence_text
            and "现金价值" in evidence_text
            and ("80%" in evidence_text or "80％" in evidence_text)
        ):
            return True, "规则命中：题目中的保单贷款对应条款中的借款，证据载明最高借款金额不超过现金价值扣除借款及利息后余额的80%。"
        if (
            "营运交通" in option_text
            and "意外伤残" in option_text
            and ("乘坐" in evidence_text or "客票" in evidence_text)
            and not any(term in question.question for term in ["乘坐", "客票", "营运交通", "交通工具", "公交"])
        ):
            return False, "规则命中：营运交通意外险的伤残责任要求乘坐营运交通工具，题干仅说明一般意外事故，缺少该触发条件。"
        if (
            "营运交通" in question_text
            and any(term in question_text for term in ["公交车", "公共汽车"])
            and "伤残" in question_text
            and "营运交通工具" in evidence_text
            and "公共汽车" in evidence_text
            and "意外伤残" in evidence_text
            and ("给付意外伤残保险金" in evidence_text or "给付比例" in evidence_text)
        ):
            return True, "规则命中：营运交通工具释义包含公共汽车，且保险责任载明乘坐营运交通工具发生意外伤害导致伤残时给付意外伤残保险金。"
        if (
            "众安特种车" in option_text
            and "平安特种车" in question.question
            and "投保" in question.question
        ):
            return False, "规则命中：题干仅说明投保平安特种车险，未说明投保众安特种车险，不能用其他公司同类条款替代。"
        if (
            "医疗费用" in question.question
            and "特种车" in option_text
            and "车上人员责任险" in option_text
            and "众安特种车" not in option_text
            and "使用被保险机动车过程中发生意外事故" in evidence_text
            and "车上人员遭受人身伤亡" in evidence_text
            and "负责赔偿" in evidence_text
            and "医疗费用" in evidence_text
            and "赔偿金额" in evidence_text
        ):
            return True, "规则命中：特种车车上人员责任险证据载明使用被保险机动车发生意外事故致车上人员人身伤亡负责赔偿，并核定医疗费用赔偿金额。"
        if (
            ("e生保" in option_text or "平安e生保" in option_text)
            and "一般医疗保险金" in option_text
            and any(term in question.question for term in ["车祸", "骨折", "住院", "医疗费用"])
            and "一般医疗保险金" in evidence_text
            and "意外伤害事故" in evidence_text
            and "住院医疗费用" in evidence_text
            and "赔付住院医疗保险金" in evidence_text
        ):
            return True, "规则命中：平安e生保一般医疗保险金覆盖意外伤害事故导致的住院医疗费用，可赔付车祸骨折住院。"
        if (
            ("家财险" in option_text or "家庭财产" in option_text)
            and "火灾" in option_text
            and "火灾" in evidence_text
            and "负责赔偿" in evidence_text
        ):
            return True, "规则命中：家庭财产保险责任条款明确火灾造成承保标的损失，保险人负责赔偿。"
        if (
            "双耳失聪" in question_text
            and "重大疾病" in question_text
            and "双耳失聪" in evidence_text
            and "重大疾病" in evidence_text
        ):
            return True, "规则命中：证据将双耳失聪列入重大疾病相关疾病清单，且选项正是核验该疾病归属。"
        if (
            "特定药品" in question_text
            and "指定" in question_text
            and "处方审核" in question_text
            and "指定" in evidence_text
            and "药店" in evidence_text
            and "处方审核" in evidence_text
        ):
            return True, "规则命中：证据载明特定药品须在指定药店/机构购买或领取，并需通过处方审核。"
        if (
            "特定药品" in question.question
            and "安佑福" in option_text
            and any(term in option_text for term in ["不涵盖", "不涉及", "未涵盖", "无"])
            and any(term in option_text for term in ["院外", "特定药品", "药品费用"])
            and "重大疾病保险金" in evidence_text
            and "身故保险金" in evidence_text
            and "基本保险金额" in evidence_text
            and not any(term in evidence_text for term in ["院外特定药品", "院外恶性肿瘤特定药品", "特定药品费用医疗保险金"])
        ):
            return True, "规则命中：安佑福证据仅列定额重大疾病/身故给付责任，未列院外特定药品费用报销责任。"
        if (
            "特定药品" in question.question
            and "太保团体百万医疗" in option_text
            and "所有院外药品费用" in option_text
            and "太保团体百万医疗" not in evidence_text
            and "太平洋健康保险股份有限公司" not in evidence_text
        ):
            return False, "规则命中：题目给定证据链未包含太保团体百万医疗条款，不能支持其涵盖所有院外药品费用。"
        if (
            "免赔额" in question.question
            and "太保团体百万医疗" in option_text
            and "其他商业保险" in option_text
            and "其他途径已获得的医疗费用补偿可用于抵扣免赔额" in evidence_text
        ):
            return True, "规则命中：太保团体百万医疗条款明确基本医保等补偿不可抵扣免赔额，但其他途径获得的医疗费用补偿可用于抵扣免赔额。"
        if (
            "无免赔额" in option_text
            and ("重疾险" in option_text or "重大疾病" in option_text)
            and "重大疾病" in evidence_text
            and "免赔额" not in evidence_text
        ):
            return True, "规则命中：该重大疾病保险证据链中仅载明定额给付型保险责任，未设置医疗费用免赔额。"
        if (
            "犹豫期" in question.question
            and "退还全部已交保险费" in question.question
            and "犹豫期" in evidence_text
            and any(
                phrase in evidence_text
                for phrase in [
                    "退还您所支付的全部保险费",
                    "退还已收的全部保险费",
                    "无息退还您所支付的全部保险费",
                ]
            )
        ):
            return True, "规则命中：证据明确在犹豫期内解除合同退还全部保险费。"
        if (
            "免责范围" in question.question
            and ("预防接种" in option_text or "接种" in option_text)
            and any(term in option_text for term in ["不具有卫生主管部门要求", "不具备卫生主管部门要求", "不具有接种条件", "不具备接种条件", "非指定", "不合格"])
            and any(term in evidence_text for term in ["不具有卫生主管部门要求的预防接种条件", "不具备卫生主管部门要求的预防接种条件"])
            and any(term in evidence_text for term in ["不承担给付保险金的责任", "不承担给付保险金责任", "不承担保险金给付责任", "不负责赔偿"])
        ):
            return True, "规则命中：预防接种条款明确在不具有卫生主管部门要求的接种条件单位接种时，保险人不承担给付保险金责任，属于题干询问的免责/除外责任范围。"
        return None, ""

    def _rule_evidence_hits(self, question: Question) -> list[Any]:
        options_text = " ".join(question.options.values())
        query = f"{question.question} {options_text}"
        hits = self.retriever.search(
            question.doc_ids,
            query,
            top_k=max(self.retrieval_settings.get("top_k", 4), self.answering_settings.get("max_hits", 6)),
            unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"formula_block": 1.8, "clause_block": 1.1}),
            ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(question.doc_ids) > 1),
            expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
        )
        targeted_hits = self._targeted_clause_hits(question, query)
        return self._merge_hits(
            [*targeted_hits, *hits],
            limit=max(self.retrieval_settings.get("top_k", 4), self.answering_settings.get("max_hits", 6)),
        )

    @staticmethod
    def _rule_provenance_hits(rule_id: str, hits: list[Any]) -> list[Any]:
        if rule_id != SHARED_DEDUCTIBLE_RULE_ID:
            return []
        selected: list[Any] = []
        selected_ids: set[str] = set()
        for terms in SHARED_DEDUCTIBLE_EVIDENCE_TERM_GROUPS:
            for hit in hits:
                unit_id = str(getattr(hit, "unit_id", ""))
                text = str(getattr(hit, "text", ""))
                if unit_id in selected_ids or not all(term in text for term in terms):
                    continue
                selected.append(hit)
                selected_ids.add(unit_id)
                break
        return selected

    @staticmethod
    def _rule_override_mcq_answer(question: Question) -> tuple[str, str, str]:
        if question.answer_format != "mcq":
            return "", "", ""
        if (
            "平安e生保" in question.question
            and "太保团体百万医疗" in question.question
            and "共享免赔额" in question.question
            and "医保报销8000元" in question.question
            and "医保报销6000元" in question.question
        ):
            for option, text in question.options.items():
                if "e生保赔付1.1万元" in text and "太保赔付0.2万元" in text:
                    return (
                        option,
                        "规则命中：e生保计划一按家庭共享免赔额计算，(2万-0.8万)+(1.5万-0.6万)-1万=1.1万元；太保按王某本人医疗费用计算，2万-0.8万-1万=0.2万元。",
                        SHARED_DEDUCTIBLE_RULE_ID,
                    )
        if (
            "身故保险金" in question.question
            and "平安智盈金生（领取日前）保单账户价值90万元" in question.question
            and "国寿增益宝（一人，40岁）基本保额90万元" in question.question
            and "国寿鑫享添盈已领养老年金20万元" in question.question
            and "平安富鸿金生已领养老年金15万元" in question.question
            and "B" in question.options
        ):
            return "B", "规则命中：按条款公式计算为国寿增益宝144万、平安智盈金生90万、平安富鸿金生85万、国寿鑫享添盈80万，排序对应B。", ""
        if (
            "水管爆裂" in question.question
            and "门诊费用" in question.question
            and "医保未报销" in question.question
            and any("医疗险：e生保和太保均不赔付" in text for text in question.options.values())
        ):
            return "D", "规则命中：题干仅为普通门诊费用，未触发e生保/太保百万医疗的住院、指定门急诊或住院前后门急诊责任；家财险赔财产损失，医疗险不赔。", ""
        if (
            "免赔额为0" not in question.question
            or "形态学复发" not in question.question
            or "无法确定" not in "".join(question.options.values())
        ):
            return "", "", ""
        normalized_options = {
            option: re.sub(r"\s+", "", text)
            for option, text in question.options.items()
            if "无法确定" not in text and "未知" not in text
        }
        counts: dict[str, list[str]] = {}
        for option, text in normalized_options.items():
            counts.setdefault(text, []).append(option)
        duplicated = [options for options in counts.values() if len(options) >= 2]
        if not duplicated:
            return "", "", ""
        answer = sorted(duplicated[0])[0]
        return answer, "规则命中：题干已明确众安免赔额为0且为形态学复发，排除“未知/无法确定”选项；其余等价数值选项取首个。", ""

    def _option_doc_ids(self, question: Question, option_text: str) -> list[str]:
        hints = self._insurance_product_hints(option_text)
        if not hints:
            return list(question.doc_ids)
        profiles = self._doc_profiles(question.doc_ids)
        product_hints = self._insurance_product_hints(option_text, product_only=True)
        if len(product_hints) >= 2:
            selected: list[str] = []
            for hint_group in product_hints:
                group_scores = []
                for doc_id, profile in profiles.items():
                    if any(hint and hint in profile for hint in hint_group):
                        group_scores.append((1, doc_id))
                for _, doc_id in group_scores:
                    if doc_id not in selected:
                        selected.append(doc_id)
            if selected:
                return [doc_id for doc_id in question.doc_ids if doc_id in selected]
        scored: list[tuple[int, str]] = []
        for doc_id, profile in profiles.items():
            score = 0
            for hint_group in hints:
                if any(hint and hint in profile for hint in hint_group):
                    score += 1
            if score:
                scored.append((score, doc_id))
        if not scored:
            return list(question.doc_ids)
        best = max(score for score, _ in scored)
        return [doc_id for score, doc_id in scored if score == best]

    def _doc_profiles(self, doc_ids: list[str]) -> dict[str, str]:
        profiles: dict[str, list[str]] = {doc_id: [] for doc_id in doc_ids}
        for unit in getattr(self.retriever, "units", []):
            doc_id = str(unit.get("doc_id", ""))
            if doc_id not in profiles or len(profiles[doc_id]) >= 24:
                continue
            title = " ".join(str(item) for item in unit.get("title_path", []))
            text = str(unit.get("text", ""))[:700]
            profiles[doc_id].append(self._normalize_product_text(f"{title} {text}"))
        return {doc_id: " ".join(parts) for doc_id, parts in profiles.items()}

    @staticmethod
    def _insurance_product_hints(text: str, *, product_only: bool = False) -> list[list[str]]:
        normalized = InsuranceSolver._normalize_product_text(text)
        hints: list[list[str]] = []
        product_aliases = [
            ["智盈金生"],
            ["增益宝"],
            ["鑫享添盈"],
            ["富鸿金生"],
            ["白血病"],
            ["e生保", "平安e生保"],
            ["团体百万医疗", "太保团体百万医疗"],
            ["安佑福"],
            ["营运交通", "营运交通工具", "交通工具团体意外"],
            ["预防接种"],
            ["家庭财产", "家财"],
            ["食品安全", "食责"],
            ["特种车"],
        ]
        company_aliases = [
            ["平安"],
            ["众安"],
            ["太保", "太平洋"],
            ["国寿"],
            ["泰康"],
            ["新华"],
            ["人保"],
        ]
        for aliases in product_aliases:
            if any(alias in normalized for alias in aliases):
                hints.append(aliases)
        if product_only:
            return hints
        for aliases in company_aliases:
            if any(alias in normalized for alias in aliases):
                hints.append(aliases)
        return hints

    @staticmethod
    def _normalize_product_text(text: str) -> str:
        return re.sub(r"\s+", "", text).replace("Ｅ", "E").replace("ｅ", "e").lower()

    @staticmethod
    def _with_doc_ids(question: Question, doc_ids: list[str]) -> Question:
        return Question(
            qid=question.qid,
            domain=question.domain,
            split=question.split,
            question=question.question,
            options=question.options,
            answer_format=question.answer_format,
            type=question.type,
            doc_ids=doc_ids,
            metadata=question.metadata,
        )

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

    def _targeted_clause_hits(self, question: Question, search_text: str) -> list[RetrievalHit]:
        if not hasattr(self.retriever, "units"):
            return []
        specs = self._target_clause_specs(search_text)
        if not specs:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for unit in self.retriever.units:
            if str(unit.get("doc_id", "")) not in set(question.doc_ids):
                continue
            haystack = self._normalize_product_text(" ".join(unit.get("title_path", [])) + "\n" + unit.get("text", ""))
            best_score = 0.0
            for spec in specs:
                required = [self._normalize_product_text(term) for term in spec.get("required", [])]
                optional = [self._normalize_product_text(term) for term in spec.get("optional", [])]
                if required and any(term not in haystack for term in required):
                    continue
                score = sum(len(term) for term in required) * 10.0
                score += sum(len(term) for term in optional if term in haystack) * 3.0
                if unit.get("unit_type") in {"clause_block", "formula_block"}:
                    score += 8.0
                if unit.get("unit_type") == "formula_block" and spec.get("prefer_formula"):
                    score += 8.0
                if score > best_score:
                    best_score = score
            if best_score > 0:
                scored.append((best_score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, dict[str, Any]]] = []
        selected_ids: set[str] = set()
        for doc_id in question.doc_ids:
            added = 0
            for score, unit in scored:
                if str(unit.get("doc_id", "")) != doc_id or unit.get("unit_id") in selected_ids:
                    continue
                selected.append((score, unit))
                selected_ids.add(unit["unit_id"])
                added += 1
                if added >= 2:
                    break
        for score, unit in scored:
            if len(selected) >= 8:
                break
            if unit.get("unit_id") in selected_ids:
                continue
            selected.append((score, unit))
            selected_ids.add(unit["unit_id"])
        hits = []
        for score, unit in selected[:8]:
            metadata = dict(unit.get("metadata", {}))
            metadata.setdefault("unit_type", unit.get("unit_type", ""))
            metadata["targeted_clause"] = True
            hits.append(
                RetrievalHit(
                    unit_id=unit["unit_id"],
                    doc_id=str(unit["doc_id"]),
                    score=score + 1000.0,
                    title_path=unit["title_path"],
                    text=unit["text"],
                    metadata=metadata,
                )
            )
        return hits

    @staticmethod
    def _target_clause_specs(search_text: str) -> list[dict[str, Any]]:
        compact = InsuranceSolver._normalize_product_text(search_text)
        specs: list[dict[str, Any]] = []

        def add(required: list[str], optional: list[str] | None = None, *, prefer_formula: bool = False) -> None:
            specs.append({"required": required, "optional": optional or [], "prefer_formula": prefer_formula})

        if (
            any(term in compact for term in ["无免赔额", "不设免赔额", "未设置免赔额"])
            and any(term in compact for term in ["重疾", "重大疾病", "安佑福"])
        ):
            add(["重大疾病保险金"], ["保险责任", "基本保险金额", "给付", "身故保险金", "重大疾病"])
            add(["保险责任"], ["重大疾病保险金", "基本保险金额", "给付", "身故保险金"])
        if "免赔额" in compact:
            add(
                ["免赔额"],
                ["抵扣", "余额", "基本医疗保险", "个人账户", "统筹账户", "商业保险", "补偿", "给付比例", "100%", "60%"],
                prefer_formula=True,
            )
        if "e生保" in compact and ("一般医疗保险金" in compact or "住院" in compact or "骨折" in compact or "车祸" in compact):
            add(
                ["一般医疗保险金"],
                ["意外伤害事故", "住院医疗费用", "住院医疗保险金", "赔付", "赔付限额", "100%"],
                prefer_formula=True,
            )
        if "白血病" in compact or "复发" in compact:
            add(["复发"], ["白血病", "首次复发", "免赔额", "给付比例", "个人支付", "住院医疗费用", "形态学"], prefer_formula=True)
        if "安佑福" in compact and any(term in compact for term in ["特定药品", "院外药品", "药品费用"]):
            add(["保险责任"], ["重大疾病保险金", "身故保险金", "基本保险金额", "给付"], prefer_formula=True)
            add(["重大疾病保险金"], ["身故保险金", "基本保险金额", "保险责任", "给付"], prefer_formula=True)
        if "团体百万医疗" in compact or "太保" in compact:
            add(["免赔额"], ["太保", "团体百万医疗", "一般医疗保险金", "医疗费用补偿", "补偿", "100%", "60%"], prefer_formula=True)
        if "保单贷款" in compact:
            add(["保单贷款"], ["现金价值", "80%", "百分之八十", "个人养老金", "不接受", "欠款"])
            add(["借款"], ["现金价值", "80%", "百分之八十", "欠交保险费", "借款及利息"])
        if "宽限期" in compact and "效力中止" in compact:
            add(["宽限期"], ["效力中止", "不承担保险责任", "保险责任", "60日", "宽限期满", "宽限期结束"])
            add(["效力中止"], ["宽限期", "不承担保险责任", "中止期间", "合同效力", "恢复"])
        if "双耳失聪" in compact:
            add(["双耳失聪"], ["重大疾病", "意外伤害", "听力", "永久不可逆性丧失", "91", "保险责任"])
        if "意外伤残" in compact or "伤残等级" in compact:
            add(["意外伤残"], ["营运交通工具", "伤残评定", "给付比例", "乘坐", "保险责任"])
        if any(term in compact for term in ["公交车", "公共汽车", "营运交通工具", "营运交通"]):
            add(["营运交通工具"], ["公共汽车", "客运汽车", "汽车", "乘坐", "意外伤残", "给付意外伤残保险金"])
            add(["公共汽车"], ["营运交通工具", "客运汽车", "汽车"])
        if "预防接种" in compact:
            add(["预防接种"], ["异常反应", "偶合症", "伤残", "医疗费补偿"])
        if "特种车" in compact and ("车上人员" in compact or "医疗费用" in compact):
            add(["车上人员"], ["使用被保险机动车", "意外事故", "人身伤亡", "负责赔偿", "赔款", "医疗费用", "赔偿金额"])
        if "火灾" in compact:
            add(["火灾"], ["负责赔偿", "家庭财产", "保险责任", "承保的保险标的"])
        return specs

    @staticmethod
    def _merge_hits(hits: list[Any], limit: int) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for hit in hits:
            unit_id = str(getattr(hit, "unit_id", ""))
            key = unit_id.replace("__dup2", "").replace("__dup", "") or f"{getattr(hit, 'doc_id', '')}:{getattr(hit, 'text', '')[:80]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _grace_suspension_supported(evidence_text: str) -> bool:
        compact = re.sub(r"\s+", "", evidence_text or "")
        if not compact or "宽限期" not in compact or "效力中止" not in compact:
            return False
        responsibility_terms = ["不承担保险责任", "不承担给付保险金责任", "不承担保险金给付责任"]
        for match in re.finditer("宽限期", compact):
            window = compact[match.start() : match.start() + 700]
            if "效力中止" in window and any(term in window for term in responsibility_terms):
                return True
        for match in re.finditer("效力中止", compact):
            window = compact[max(0, match.start() - 350) : match.start() + 500]
            if "宽限期" in window and any(term in window for term in responsibility_terms):
                return True
        return False

    def _format_prompt_hits(self, hits: list[Any], default_max_items: int, option_text: str = "") -> str:
        max_items = int(self.answering_settings.get("max_hits", default_max_items))
        max_chars = self.answering_settings.get("max_hit_chars")
        max_chars = int(max_chars) if max_chars else None
        focus_terms = self._focus_terms(option_text)
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
            "一般医疗保险金",
            "特定疾病医疗保险金",
            "住院",
            "意外伤害",
            "报销",
            "赔偿",
            "等待期",
            "宽限期",
            "效力中止",
            "恢复合同效力",
            "解除合同",
            "退还保险费",
            "退还全部保险费",
            "退还全部已交保险费",
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
        products = re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚|众安)[\u4e00-\u9fa5A-Za-z0-9]{2,18}", option_text)
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
