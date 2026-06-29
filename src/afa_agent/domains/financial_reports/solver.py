from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from afa_agent.domains.llm_utils import ask_answer_fallback, ask_option_judgment, format_hits
from afa_agent.evidence_gate import answer_consistency_issues, evaluate_evidence, gate_enabled, rescue_evidence
from afa_agent.models import AnswerResult, Question, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


METRIC_ALIASES = {
    "研发占比": ["研发投入占营业收入的比例"],
    "现金分红": ["每10股派", "现金分红", "末期股息"],
    "营业收入": ["营业收入", "营业总收入", "营业额"],
    "归母净利润": ["归属于上市公司股东的净利润", "归母净利润", "母公司拥有人应占溢利"],
    "经营现金流": ["经营活动产生的现金流量净额"],
    "研发投入": ["研发投入"],
}


class FinancialReportsSolver:
    def __init__(self, client, retriever, units: list[dict[str, Any]], strategy: str):
        self.client = client
        self.retriever = retriever
        self.units = units
        self.metric_index = self._build_metric_index(units)
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.rule_settings = get_stage_settings(strategy, "rule_layer")
        self.answering_settings = get_stage_settings(strategy, "answering")
        self.gate_settings = get_stage_settings(strategy, "evidence_gate")

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        option_debug: list[dict[str, Any]] = []
        rule_outputs: list[dict[str, Any]] = []
        query_variants_all: list[str] = []

        for option_key, option_text in question.options.items():
            query_variants = build_query_variants(question, option_key, option_text, self.retrieval_settings)
            query_variants_all.extend(query_variants)
            rule_label, rule_reason, rule_evidence = self._rule_evaluate(question, option_text)
            hits = self.retriever.search(
                question.doc_ids,
                query_variants[0],
                top_k=self.retrieval_settings.get("top_k", 6),
                unit_type_boosts=self.retrieval_settings.get("unit_type_boosts", {"metric_row": 1.8, "paragraph": 1.0}),
                ensure_per_doc=self.retrieval_settings.get("ensure_per_doc", len(question.doc_ids) > 1),
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            gate_debug: dict[str, Any] = {}
            if gate_enabled(self.gate_settings):
                initial_gate = evaluate_evidence(
                    question,
                    option_key,
                    option_text,
                    [*rule_evidence, *hits],
                    question.domain,
                    self.gate_settings,
                )
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
            if self.rule_settings.get("enabled", True) and rule_label is not None:
                label = rule_label
                reasoning = rule_reason
                evidence_items = rule_evidence + [hit.to_dict() for hit in hits[:2]]
                rule_outputs.append(
                    {
                        "option": option_key,
                        "label": label,
                        "answer": option_key if label else "",
                        "reason": rule_reason,
                        "confidence": 0.95,
                    }
                )
            else:
                parsed, usage = ask_option_judgment(
                    self.client,
                    self._system_prompt(),
                    question.question,
                    question.answer_format,
                    option_key,
                    option_text,
                    format_hits(hits, max_items=self.answering_settings.get("max_hits", 6)),
                    self._extra_context(),
                )
                total_usage.add(usage)
                label = bool(parsed.get("label", False))
                reasoning = str(parsed.get("reasoning_summary", "")).strip()
                evidence_items = [hit.to_dict() for hit in hits]
            option_labels[option_key] = label
            option_payloads.append(
                {
                    "option": option_key,
                    "label": label,
                    "reasoning_summary": reasoning,
                    "evidence_items": evidence_items[:6],
                    "gate_status": gate_debug.get("final_gate", {}).get("status", ""),
                    "gate_reasons": gate_debug.get("final_gate", {}).get("reasons", []),
                }
            )
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "retrieval_topk": serialize_hits(hits, limit=self.retrieval_settings.get("top_k", 6)),
                    "used_rule": bool(self.rule_settings.get("enabled", True) and rule_label is not None),
                    "evidence_gate": gate_debug,
                }
            )
            reasoning_chunks.append(f"{option_key}: {reasoning}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        if question.answer_format == "mcq" and len([k for k, v in option_labels.items() if v]) != 1:
            answer, usage = ask_answer_fallback(
                self.client,
                "你是财报单选题裁决器。根据各选项判断摘要，选出唯一最可能正确的字母，只输出 JSON。",
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
                "你是财报多选题复核器。根据各选项判断摘要，挑出所有正确选项；答案必须至少包含两个选项字母，只输出 JSON。",
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
        consistency_issues = []
        if gate_enabled(self.gate_settings):
            consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)
            if consistency_issues and self.gate_settings.get("final_consistency_retry", True):
                answer, usage = ask_answer_fallback(
                    self.client,
                    "你是财报答案一致性复核器。只能选择 label=true 且 evidence gate 未失败的选项；必须核对年份、指标和正负方向，只输出 JSON。",
                    question.question,
                    option_payloads,
                    question.answer_format,
                    list(question.options.keys()),
                )
                total_usage.add(usage)
                pred_answer = answer[:1] if question.answer_format == "mcq" else answer
                if question.answer_format == "multi":
                    pred_answer = self._ensure_multi_minimum(pred_answer, question, option_labels)
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
                "retrieval_topk": [hit for item in option_debug for hit in item["retrieval_topk"]][: self.retrieval_settings.get("top_k", 6)],
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": rule_outputs,
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
                "final_consistency_check": {"issues": consistency_issues},
            },
        )

    def _system_prompt(self) -> str:
        prompt_id = self.answering_settings.get("prompt_template_id", "default")
        if prompt_id == "evidence_strict":
            return "你是财报问答助手。必须逐条核对财务指标、年份和比较关系，只能依据证据作答，禁止常识补全，输出必须是 JSON。"
        if prompt_id == "compact":
            return "你是财报问答助手。请在有限证据内快速判断选项真伪，只输出 JSON。"
        return "你是财报问答助手。优先依据财务指标、年份和比较关系判断选项真伪。只能依据证据作答，输出必须是 JSON。"

    def _extra_context(self) -> str:
        extra = self.answering_settings.get("extra_context", "").strip()
        base = "请优先核对年份、同比、现金分红、研发占比等财务指标，不要只做模糊语义判断。"
        return f"{base}\n{extra}".strip()

    def _build_metric_index(self, units: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        metric_index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for unit in units:
            if unit.get("unit_type") != "metric_row":
                continue
            meta = unit.get("metadata", {})
            metric_name = meta.get("metric_name")
            if not metric_name:
                continue
            metric_key = self._normalize_metric(metric_name)
            metric_index[unit["doc_id"]][metric_key].append(unit)
        return metric_index

    def _rule_evaluate(self, question: Question, option_text: str):
        metric_key = self._detect_metric_key(option_text)
        if not metric_key or len(question.doc_ids) < 2:
            return None, "", []
        doc_metrics = [self.metric_index.get(doc_id, {}).get(metric_key, []) for doc_id in question.doc_ids[:2]]
        if not all(doc_metrics):
            return None, "", []
        values = []
        evidence = []
        for doc_units in doc_metrics:
            best_unit = self._choose_best_unit(doc_units)
            parsed_value = self._extract_best_number(best_unit["text"], metric_key)
            if parsed_value is None:
                return None, "", []
            values.append(parsed_value)
            evidence.append(
                {
                    "unit_id": best_unit["unit_id"],
                    "doc_id": best_unit["doc_id"],
                    "score": 999.0,
                    "title_path": best_unit["title_path"],
                    "text": best_unit["text"],
                    "metadata": best_unit.get("metadata", {}),
                }
            )
        label = None
        if any(keyword in option_text for keyword in ["增长", "高于", "优于", "提升"]):
            label = values[1] > values[0]
        elif any(keyword in option_text for keyword in ["下降", "低于", "减少", "下滑"]):
            label = values[1] < values[0]
        if label is None:
            return None, "", []
        reason = f"规则比较到 {metric_key} 在两份报告中的候选值分别为 {values[0]} 和 {values[1]}，据此判断选项为 {'正确' if label else '错误'}。"
        return label, reason, evidence

    def _detect_metric_key(self, text: str) -> str | None:
        for metric_key, aliases in METRIC_ALIASES.items():
            if any(alias in text for alias in aliases):
                return metric_key
        return None

    def _normalize_metric(self, metric_name: str) -> str:
        for metric_key, aliases in METRIC_ALIASES.items():
            if any(alias in metric_name for alias in aliases):
                return metric_key
        return metric_name

    @staticmethod
    def _choose_best_unit(units: list[dict[str, Any]]) -> dict[str, Any]:
        def score(unit: dict[str, Any]) -> tuple[int, int]:
            text = unit.get("text", "")
            quality = 0
            if "同比" in text:
                quality += 3
            if "（元）" in text or "元）" in text:
                quality += 2
            if "占营业收入比例" in text:
                quality += 2
            if "\n" not in text:
                quality += 1
            return quality, len(text)

        return max(units, key=score)

    @staticmethod
    def _extract_best_number(text: str, metric_key: str) -> float | None:
        raw_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
        if not raw_numbers:
            return None
        values = [float(item.replace(",", "")) for item in raw_numbers]
        if metric_key == "研发占比":
            percents = [value for value in values if value <= 100]
            return percents[-1] if percents else values[-1]
        return max(values)

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
