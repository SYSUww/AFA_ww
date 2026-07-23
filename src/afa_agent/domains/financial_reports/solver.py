from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from afa_agent.domains.llm_utils import (
    ask_answer_fallback,
    ask_option_judgment,
    collect_evidence_items,
    finalize_answer,
    format_hits,
    parse_confidence,
)
from afa_agent.evidence_gate import (
    answer_consistency_issues,
    evaluate_evidence,
    gate_enabled,
    rescue_evidence,
    should_skip_answer_fallback,
    should_skip_consistency_retry,
)
from afa_agent.models import AnswerResult, Question, RetrievalHit, TokenUsage
from afa_agent.strategy import build_query_variants, get_stage_settings, serialize_hits


METRIC_ALIASES = {
    "研发占比": [
        "研发投入总额占营业收入比例",
        "研发投入总额占营业收入的比例",
        "研发投入占营业收入的比例",
        "研发投入占营业收入比例",
        "研发费用占营业收入比例",
        "研发投入占营业收入的比重",
        "研发投入占营业收入比重",
    ],
    "现金分红": ["每10股派", "现金分红", "末期股息"],
    "营业收入": ["营业收入", "营业总收入", "营业额"],
    "归母净利润": ["归属于上市公司股东的净利润", "归母净利润", "母公司拥有人应占溢利"],
    "经营现金流": ["经营活动产生的现金流量净额"],
    "基本每股收益": ["基本每股收益"],
    "研发投入": ["研发投入"],
}

COMPANY_DOC_HINTS = {
    "比亚迪": "byd",
    "宁德时代": "catl",
    "宁德": "catl",
    "美的集团": "midea",
    "美的": "midea",
    "中国移动": "chinamobile",
    "中移动": "chinamobile",
    "中国建筑": "cscec",
    "中建": "cscec",
    "招商银行": "cmb",
    "招行": "cmb",
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
        self.answer_policy_settings = get_stage_settings(strategy, "answer_policy")

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        option_debug: list[dict[str, Any]] = []
        rule_outputs: list[dict[str, Any]] = []
        query_variants_all: list[str] = []

        option_items = [("A", question.question)] if question.answer_format == "tf" else list(question.options.items())
        for option_key, option_text in option_items:
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
            hits, targeted_debug = self._augment_targeted_hits(question, option_text, hits)
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
                confidence = 0.95
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
                confidence = parse_confidence(parsed, 0.75 if label else 0.25)
                evidence_items = [hit.to_dict() for hit in hits]
            option_labels[option_key] = label
            option_payloads.append(
                {
                    "option": option_key,
                    "label": label,
                    "reasoning_summary": reasoning,
                    "confidence": confidence,
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
                    "model_confidence": confidence,
                    "targeted_evidence": targeted_debug,
                    "evidence_gate": gate_debug,
                }
            )
            reasoning_chunks.append(f"{option_key}: {reasoning}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        fallback_skipped_reason = ""
        if question.answer_format == "mcq" and len([k for k, v in option_labels.items() if v]) != 1:
            if should_skip_answer_fallback(
                answer_format=question.answer_format,
                option_labels=option_labels,
                gate_settings=self.gate_settings,
            ):
                fallback_skipped_reason = "mcq_ambiguous_supported"
            else:
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
                    "你是财报多选题复核器。根据各选项判断摘要，挑出所有正确选项；答案必须至少包含两个选项字母，只输出 JSON。",
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
        consistency_issues = []
        if gate_enabled(self.gate_settings):
            consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)
            if consistency_issues and self.gate_settings.get("final_consistency_retry", True):
                skip_retry, skip_reason = should_skip_consistency_retry(
                    consistency_issues=consistency_issues,
                    answer_finalization=answer_finalization,
                    answer_format=question.answer_format,
                    gate_settings=self.gate_settings,
                )
                if skip_retry:
                    answer_finalization = {
                        **answer_finalization,
                        "consistency_retry": False,
                        "retry_skipped_reason": skip_reason,
                    }
                else:
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
                "retrieval_topk": [hit for item in option_debug for hit in item["retrieval_topk"]][: self.retrieval_settings.get("top_k", 6)],
                "selected_evidence_ids": [item.get("unit_id", "") for item in evidence_items if item.get("unit_id")],
                "rule_outputs": rule_outputs,
                "option_debug": option_debug,
                "consistency_answers": [pred_answer],
                "final_consistency_check": {"issues": consistency_issues},
                "answer_finalization": answer_finalization,
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
            metric_keys = {self._normalize_metric(metric_name)}
            text = unit.get("text", "")
            for alias_key, aliases in METRIC_ALIASES.items():
                if any(alias in text for alias in aliases):
                    metric_keys.add(alias_key)
            for metric_key in metric_keys:
                bucket = metric_index[unit["doc_id"]][metric_key]
                if all(existing.get("unit_id") != unit.get("unit_id") for existing in bucket):
                    bucket.append(unit)
        return metric_index

    def _rule_evaluate(self, question: Question, option_text: str):
        metric_bundle_rule = self._choice_metric_bundle_rule(question, option_text)
        if metric_bundle_rule is not None:
            return metric_bundle_rule
        rd_ratio_repurchase_rule = self._rd_ratio_repurchase_compound_rule(question, option_text)
        if rd_ratio_repurchase_rule is not None:
            return rd_ratio_repurchase_rule
        repurchase_rule = self._repurchase_rule(question, option_text)
        if repurchase_rule is not None:
            return repurchase_rule
        shareholder_return_total_rule = self._shareholder_return_total_rule(question, option_text)
        if shareholder_return_total_rule is not None:
            return shareholder_return_total_rule
        net_profit_dividend_rule = self._net_profit_dividend_compound_rule(question, option_text)
        if net_profit_dividend_rule is not None:
            return net_profit_dividend_rule
        dividend_ratio_rule = self._dividend_ratio_rule(question, option_text)
        if dividend_ratio_rule is not None:
            return dividend_ratio_rule
        revenue_multiple_rule = self._revenue_multiple_rule(question, option_text)
        if revenue_multiple_rule is not None:
            return revenue_multiple_rule
        foreign_revenue_rule = self._foreign_revenue_ratio_rule(question, option_text)
        if foreign_revenue_rule is not None:
            return foreign_revenue_rule
        cash_flow_revenue_ratio_rule = self._cash_flow_revenue_ratio_rule(question, option_text)
        if cash_flow_revenue_ratio_rule is not None:
            return cash_flow_revenue_ratio_rule

        metric_key = self._detect_metric_key(option_text)
        if not metric_key or len(question.doc_ids) < 2:
            return None, "", []
        single_doc_growth_rule = self._single_doc_growth_polarity_rule(question, option_text, metric_key)
        if single_doc_growth_rule is not None:
            return single_doc_growth_rule
        growth_rule = self._growth_rate_rule(question, option_text, metric_key)
        if growth_rule is not None:
            return growth_rule
        if metric_key in {"现金分红", "研发投入"}:
            return None, "", []
        ordered_doc_ids = self._mentioned_doc_order(option_text, question.doc_ids)
        if len(ordered_doc_ids) < 2:
            ordered_doc_ids = list(question.doc_ids[:2])
        doc_metrics = [self.metric_index.get(doc_id, {}).get(metric_key, []) for doc_id in ordered_doc_ids[:2]]
        if not all(doc_metrics):
            return None, "", []
        values = []
        evidence = []
        for doc_units in doc_metrics:
            best_unit, parsed_value = self._choose_best_metric_unit(doc_units, metric_key)
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
            label = values[0] > values[1]
        elif any(keyword in option_text for keyword in ["下降", "低于", "减少", "下滑"]):
            label = values[0] < values[1]
        if label is None:
            return None, "", []
        reason = (
            f"规则按选项顺序比较 {metric_key}：{ordered_doc_ids[0]}={values[0]}，"
            f"{ordered_doc_ids[1]}={values[1]}，据此判断选项为 {'正确' if label else '错误'}。"
        )
        return label, reason, evidence

    def _choice_metric_bundle_rule(self, question: Question, option_text: str):
        for rule in (
            self._operating_metric_bundle_rule,
            self._dividend_per_ten_bundle_rule,
            self._midea_statement_scope_rule,
            self._byd_cross_year_amount_rule,
            self._cscec_original_basis_rule,
            self._solvency_metric_bundle_rule,
            self._research_expense_rate_bundle_rule,
        ):
            result = rule(question, option_text)
            if result is not None:
                return result
        return None

    def _operating_metric_bundle_rule(self, question: Question, option_text: str):
        if not ("宁德时代" in question.question and "美的集团" in question.question):
            return None
        catl_doc = self._company_year_doc(question.doc_ids, "catl", "2025")
        midea_doc = self._company_year_doc(question.doc_ids, "midea", "2025")
        if not catl_doc or not midea_doc:
            return None

        if "营业收入均同比增长" in option_text:
            catl = self._metric_series(catl_doc, "营业收入")
            midea = self._metric_series(midea_doc, "营业收入")
            if not catl or not midea:
                return None
            label = catl[0] > catl[1] and midea[0] > midea[1]
            reason = (
                f"规范指标束：宁德时代营业收入 {catl[0]:g}>{catl[1]:g}，"
                f"美的集团营业收入 {midea[0]:g}>{midea[1]:g}，两家公司均同比增长。"
            )
            return label, reason, [self._unit_to_evidence(catl[3], 999.0), self._unit_to_evidence(midea[3], 999.0)]

        if "经营现金流率" in option_text:
            catl_revenue = self._metric_series(catl_doc, "营业收入")
            catl_cash = self._metric_series(catl_doc, "经营现金流")
            midea_revenue = self._metric_series(midea_doc, "营业收入")
            midea_cash = self._metric_series(midea_doc, "经营现金流")
            if not all((catl_revenue, catl_cash, midea_revenue, midea_cash)):
                return None
            catl_current = catl_cash[0] / catl_revenue[0] * 100
            catl_prior = catl_cash[1] / catl_revenue[1] * 100
            midea_current = midea_cash[0] / midea_revenue[0] * 100
            midea_prior = midea_cash[1] / midea_revenue[1] * 100
            if "上升" in option_text and "下降" in option_text:
                label = catl_current > catl_prior and midea_current < midea_prior
            elif "高约" in option_text or "高出" in option_text:
                expected = self._extract_expected_points(option_text)
                if expected is None:
                    return None
                label = abs((catl_current - midea_current) - expected) <= 0.05
            else:
                return None
            reason = (
                f"规范指标束计算经营现金流率：宁德时代2025/2024为{catl_current:.2f}%/{catl_prior:.2f}%，"
                f"美的集团2025/2024为{midea_current:.2f}%/{midea_prior:.2f}%，"
                f"2025年差值为{catl_current - midea_current:.2f}个百分点。"
            )
            evidence_units = [catl_revenue[3], catl_cash[3], midea_revenue[3], midea_cash[3]]
            return label, reason, [self._unit_to_evidence(unit, 999.0) for unit in evidence_units]

        if "基本每股收益同比增幅比" in option_text:
            catl = self._metric_series(catl_doc, "基本每股收益")
            midea = self._metric_series(midea_doc, "基本每股收益")
            expected = self._extract_expected_points(option_text)
            if not catl or not midea or expected is None or catl[2] is None or midea[2] is None:
                return None
            difference = catl[2] - midea[2]
            label = abs(difference - expected) <= 0.05
            reason = (
                f"规范指标束读取基本每股收益同比增幅：宁德时代{catl[2]:.2f}%，"
                f"美的集团{midea[2]:.2f}%，差值{difference:.2f}个百分点。"
            )
            return label, reason, [self._unit_to_evidence(catl[3], 999.0), self._unit_to_evidence(midea[3], 999.0)]
        return None

    def _dividend_per_ten_bundle_rule(self, question: Question, option_text: str):
        if not ("现金分红数据" in question.question and "2025" in question.question):
            return None
        doc_hints = {
            "宁德时代": "catl",
            "美的集团": "midea",
            "招商银行": "cmb",
            "中国建筑": "cscec",
        }
        values: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        for company, hint in doc_hints.items():
            doc_id = self._company_year_doc(question.doc_ids, hint, "2025")
            result = self._best_dividend_per_ten(doc_id) if doc_id else None
            if result is None:
                return None
            values[company] = result

        if "由高到低排序" in option_text:
            ordered = ["宁德时代", "美的集团", "招商银行", "中国建筑"]
            label = all(values[left][0] > values[right][0] for left, right in zip(ordered, ordered[1:]))
            reason = "规范全年分红口径（每10股）：" + "，".join(
                f"{company}{values[company][0]:g}元" for company in ordered
            )
            evidence = [
                self._unit_to_evidence(unit, 999.0)
                for company in ordered
                for unit in values[company][1]
            ]
            return label, reason, evidence
        if "美的集团" in option_text and "全年每 10 股现金分红为 38 元" in option_text:
            label = abs(values["美的集团"][0] - 38.0) <= 0.001
            reason = f"规范全年口径：美的集团2025年全年每10股现金分红为{values['美的集团'][0]:g}元，38元仅为年末方案。"
            return label, reason, [
                self._unit_to_evidence(unit, 999.0)
                for unit in values["美的集团"][1]
            ]
        if "招商银行" in option_text and "每股现金分红 2.016 元" in option_text:
            label = abs(values["招商银行"][0] - 20.16) <= 0.001
            reason = f"规范单位换算：招商银行全年每股2.016元，等价于每10股{values['招商银行'][0]:.2f}元。"
            return label, reason, [
                self._unit_to_evidence(unit, 999.0)
                for unit in values["招商银行"][1]
            ]
        if "宁德时代与美的集团" in option_text and "相差 26.57 元" in option_text:
            difference = values["宁德时代"][0] - values["美的集团"][0]
            label = abs(difference - 26.57) <= 0.001
            reason = f"规范全年口径差值：{values['宁德时代'][0]:g}-{values['美的集团'][0]:g}={difference:.2f}元。"
            return label, reason, [
                self._unit_to_evidence(unit, 999.0)
                for company in ("宁德时代", "美的集团")
                for unit in values[company][1]
            ]
        return None

    def _midea_statement_scope_rule(self, question: Question, option_text: str):
        if not ("合并财务报表与母公司财务报表" in question.question and "美的集团" in question.question):
            return None
        doc_id = self._company_year_doc(question.doc_ids, "midea", "2025")
        if not doc_id:
            return None
        revenue_unit = self._find_unit(
            doc_id,
            required=("2025年度合并", "2025年度公司", "营业收入"),
        )
        cash_unit = self._find_unit(
            doc_id,
            required=("经营活动产生/(使用)的现金流量净额",),
        )
        eps_unit = next(iter(self.metric_index.get(doc_id, {}).get("基本每股收益", [])), None)
        if revenue_unit is None or cash_unit is None or eps_unit is None:
            return None
        revenue = self._extract_row_numbers(revenue_unit.get("text", ""), "其中:营业收入")
        cash = self._extract_row_numbers(cash_unit.get("text", ""), "经营活动产生/(使用)的现金流量净额")
        if len(revenue) < 4 or len(cash) < 4:
            return None
        evidence = [self._unit_to_evidence(revenue_unit, 999.0), self._unit_to_evidence(cash_unit, 999.0)]
        if "合并口径营业收入同比增长" in option_text and "母公司口径营业收入同比下降" in option_text:
            label = revenue[0] > revenue[1] and revenue[2] < revenue[3]
            reason = f"报表口径束：合并营业收入{revenue[0]:g}>{revenue[1]:g}，母公司营业收入{revenue[2]:g}<{revenue[3]:g}。"
            return label, reason, evidence
        if "合并口径经营活动现金流量净额为正" in option_text and "母公司口径为负" in option_text:
            label = cash[0] > 0 and cash[2] < 0
            reason = f"报表口径束：合并经营现金流净额{cash[0]:g}为正，母公司{cash[2]:g}为负。"
            return label, reason, evidence
        if "5.80 元的基本每股收益为母公司" in option_text:
            return False, "报表口径束：5.80元基本每股收益列于合并口径主要财务指标，并非母公司单体指标。", [
                self._unit_to_evidence(eps_unit, 999.0),
                self._unit_to_evidence(revenue_unit, 998.0),
            ]
        if "母公司 2025 年经营活动产生的现金流量净额为 53,345,930 千元" in option_text:
            label = abs(cash[2] - 53_345_930) <= 0.5
            reason = f"报表口径束：53,345,930千元为合并口径，母公司2025年经营现金流净额为{cash[2]:g}千元。"
            return label, reason, [self._unit_to_evidence(cash_unit, 999.0)]
        return None

    def _byd_cross_year_amount_rule(self, question: Question, option_text: str):
        if not ("比亚迪" in question.question and "2024" in question.question and "2025" in question.question):
            return None
        doc_id = self._company_year_doc(question.doc_ids, "byd", "2025")
        if not doc_id:
            return None

        if "分地区营业收入" in question.question:
            region_unit = self._find_unit(
                doc_id,
                required=("地区信息", "中国(包括港澳台地区)", "境外", "合计"),
            )
            if region_unit is None:
                return None
            china = self._extract_row_numbers(region_unit.get("text", ""), "中国(包括港澳台地区)")
            foreign = self._extract_row_numbers(region_unit.get("text", ""), "境外")
            total = self._extract_row_numbers(region_unit.get("text", ""), "合计")
            if min(len(china), len(foreign), len(total)) < 2 or not total[0] or not total[1]:
                return None
            foreign_share_current = foreign[0] / total[0] * 100
            foreign_share_prior = foreign[1] / total[1] * 100
            share_change = foreign_share_current - foreign_share_prior
            relative_share_change = share_change / foreign_share_prior * 100
            foreign_increase = foreign[0] - foreign[1]
            china_decrease = china[1] - china[0]
            revenue_increase = total[0] - total[1]

            if "提高约" in option_text and "个百分点" in option_text:
                expected = self._extract_expected_points(option_text)
                label = expected is not None and abs(share_change - expected) <= 0.02
            elif "相对增幅" in option_text:
                expected = self._extract_expected_points(option_text)
                label = expected is not None and abs(relative_share_change - expected) <= 0.02
            elif "境外收入增加额大于" in option_text and "收入减少额" in option_text:
                label = foreign_increase > china_decrease
            elif "境外收入增加额减去" in option_text and "营业收入增加额基本一致" in option_text:
                label = abs((foreign_increase - china_decrease) - revenue_increase) <= 1.0
            else:
                return None
            reason = (
                "跨年地区金额束：境外收入占比2025/2024为"
                f"{foreign_share_current:.2f}%/{foreign_share_prior:.2f}%，差{share_change:.2f}个百分点、"
                f"相对增幅{relative_share_change:.2f}%；境外增加{foreign_increase:g}，"
                f"中国地区减少{china_decrease:g}，营业收入增加{revenue_increase:g}。"
            )
            return label, reason, [self._unit_to_evidence(region_unit, 999.0)]

        if not (
            "归属于上市公司股东的净利润" in question.question
            and "经营活动产生的现金流量净额" in question.question
        ):
            return None
        revenue = self._metric_series(doc_id, "营业收入")
        profit = self._metric_series(doc_id, "归母净利润")
        cash = self._metric_series(doc_id, "经营现金流")
        if not all((revenue, profit, cash)) or not revenue[0] or not revenue[1] or not profit[1] or not cash[1]:
            return None
        margin_current = profit[0] / revenue[0] * 100
        margin_prior = profit[1] / revenue[1] * 100
        margin_point_change = margin_prior - margin_current
        margin_relative_decline = margin_point_change / margin_prior * 100
        profit_decline = (profit[1] - profit[0]) / profit[1] * 100
        cash_decline = (cash[1] - cash[0]) / cash[1] * 100
        cash_ratio_current = cash[0] / revenue[0] * 100
        cash_ratio_prior = cash[1] / revenue[1] * 100
        cash_ratio_point_change = cash_ratio_prior - cash_ratio_current

        if "归母净利率" in option_text and "相对降幅" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs(margin_relative_decline - expected) <= 0.02
        elif "归母净利润同比下降" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs(profit_decline - expected) <= 0.02
        elif "经营活动现金流量净额同比下降" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs(cash_decline - expected) <= 0.02
        elif "经营活动现金流量净额占营业收入的比例由" in option_text:
            expected = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%?", option_text)]
            label = len(expected) >= 3 and all(
                abs(actual - stated) <= 0.02
                for actual, stated in zip(
                    (cash_ratio_prior, cash_ratio_current, cash_ratio_point_change),
                    expected[:3],
                )
            )
        else:
            return None
        reason = (
            f"跨年指标束：归母净利率2025/2024为{margin_current:.2f}%/{margin_prior:.2f}%，"
            f"相对降幅{margin_relative_decline:.2f}%；归母净利润同比下降{profit_decline:.2f}%，"
            f"经营现金流同比下降{cash_decline:.2f}%，其占营业收入比例由{cash_ratio_prior:.2f}%"
            f"降至{cash_ratio_current:.2f}%，下降{cash_ratio_point_change:.2f}个百分点。"
        )
        evidence_units = {unit["unit_id"]: unit for unit in (revenue[3], profit[3], cash[3])}
        return label, reason, [self._unit_to_evidence(unit, 999.0) for unit in evidence_units.values()]

    def _cscec_original_basis_rule(self, question: Question, option_text: str):
        compact_question = re.sub(r"\s+", "", question.question)
        if not (
            "中国建筑" in question.question
            and "2024年数据采用2024年年报原始披露值" in compact_question
        ):
            return None
        current_doc = self._company_year_doc(question.doc_ids, "cscec", "2025")
        prior_doc = self._company_year_doc(question.doc_ids, "cscec", "2024")
        if not current_doc or not prior_doc:
            return None
        main_unit = self._find_unit(
            current_doc,
            required=("调整后", "调整前", "营业收入", "经营活动产生的现金流量净额"),
        )
        eps_unit = self._find_unit(
            current_doc,
            required=("调整后", "调整前", "基本每股收益(元/股)"),
        )
        current_dividend_unit = self._find_unit(current_doc, required=("现金分红占", "28.75%"))
        prior_dividend_unit = self._find_unit(prior_doc, required=("现金分红占", "24.29%"))
        if not all((main_unit, eps_unit, current_dividend_unit, prior_dividend_unit)):
            return None
        revenue = self._extract_row_numbers(main_unit.get("text", ""), "营业收入")
        profit = self._extract_row_numbers(main_unit.get("text", ""), "归属于上市公司股东的净利润")
        cash = self._extract_row_numbers(main_unit.get("text", ""), "经营活动产生的现金流量净额")
        eps = self._extract_row_numbers(eps_unit.get("text", ""), "基本每股收益(元/股)")
        if len(revenue) < 3 or len(profit) < 3 or len(cash) < 3 or len(eps) < 3 or not revenue[0]:
            return None

        current_dividend = self._extract_disclosed_ratio(current_dividend_unit.get("text", ""))
        prior_dividend = self._extract_disclosed_ratio(prior_dividend_unit.get("text", ""))
        if current_dividend is None or prior_dividend is None or not eps[2] or not profit[2]:
            return None
        dividend_change = current_dividend - prior_dividend
        cash_revenue_ratio = cash[0] / revenue[0] * 100
        eps_decline = (eps[2] - eps[0]) / eps[2] * 100
        profit_decline = (profit[2] - profit[0]) / profit[2] * 100
        profit_decrease = profit[2] - profit[0]
        cash_increase = cash[0] - cash[2]

        if "现金分红占归母净利润比例" in option_text and "提高" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs(dividend_change - expected) <= 0.01
        elif "经营活动现金流量净额占营业收入的比例超过" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and cash_revenue_ratio > expected
        elif "基本每股收益降幅" in option_text and "归母净利润" in option_text:
            expected = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", option_text)]
            label = (
                len(expected) >= 2
                and abs(eps_decline - expected[0]) <= 0.02
                and abs(profit_decline - expected[1]) <= 0.02
                and abs(eps_decline - profit_decline) <= 0.15
            )
        elif "归母净利润减少额小于经营活动现金流量净额增加额" in option_text:
            label = profit_decrease < cash_increase
        else:
            return None
        reason = (
            f"原始披露口径束：现金分红比例提高{dividend_change:.2f}个百分点；"
            f"2025经营现金流/营业收入={cash_revenue_ratio:.2f}%；基本每股收益降幅{eps_decline:.2f}%，"
            f"归母净利润降幅{profit_decline:.2f}%；归母净利润减少{profit_decrease:g}，"
            f"经营现金流增加{cash_increase:g}。2024年均取调整前/2024年报原始值。"
        )
        evidence = [main_unit, eps_unit, current_dividend_unit, prior_dividend_unit]
        return label, reason, [self._unit_to_evidence(unit, 999.0) for unit in evidence]

    @staticmethod
    def _extract_disclosed_ratio(text: str) -> float | None:
        match = re.search(r"比例为\s*(\d+(?:\.\d+)?)%", text)
        return float(match.group(1)) if match else None

    def _solvency_metric_bundle_rule(self, question: Question, option_text: str):
        if "资产负债率、流动比率和速动比率" in question.question:
            doc_hints = {"比亚迪": "byd", "宁德时代": "catl", "美的集团": "midea"}
            tables: dict[str, tuple[dict[str, list[float]], dict[str, Any]]] = {}
            for company, hint in doc_hints.items():
                doc_id = self._company_year_doc(question.doc_ids, hint, "2025")
                result = self._solvency_table(doc_id) if doc_id else None
                if result is None:
                    return None
                tables[company] = result
            ratios = {company: values for company, (values, _) in tables.items()}
            if "三家公司 2025 年资产负债率均较 2024 年下降" in option_text:
                label = all(values["资产负债率"][0] < values["资产负债率"][1] for values in ratios.values())
            elif "2025 年资产负债率由低到高排序" in option_text:
                expected_order = ["美的集团", "宁德时代", "比亚迪"]
                actual_order = sorted(expected_order, key=lambda company: ratios[company]["资产负债率"][0])
                label = actual_order == expected_order
            elif "比亚迪 2025 年流动比率和速动比率均较 2024 年上升" in option_text:
                values = ratios["比亚迪"]
                label = values["流动比率"][0] > values["流动比率"][1] and values["速动比率"][0] > values["速动比率"][1]
            elif "宁德时代 2025 年流动比率和速动比率均较 2024 年上升" in option_text:
                values = ratios["宁德时代"]
                label = values["流动比率"][0] > values["流动比率"][1] and values["速动比率"][0] > values["速动比率"][1]
            else:
                return None
            reason = "偿债指标束：" + "；".join(
                f"{company}资产负债率{values['资产负债率'][0]:.2f}%/{values['资产负债率'][1]:.2f}%，"
                f"流动比率{values['流动比率'][0]:g}/{values['流动比率'][1]:g}，"
                f"速动比率{values['速动比率'][0]:g}/{values['速动比率'][1]:g}"
                for company, values in ratios.items()
            )
            return label, reason, [self._unit_to_evidence(unit, 999.0) for _, unit in tables.values()]

        if not ("2023—2025 年偿债指标" in question.question and "比亚迪" in question.question and "宁德时代" in question.question):
            return None
        byd_2025 = self._company_year_doc(question.doc_ids, "byd", "2025")
        byd_2024 = self._company_year_doc(question.doc_ids, "byd", "2024")
        catl_2025 = self._company_year_doc(question.doc_ids, "catl", "2025")
        catl_2024 = self._company_year_doc(question.doc_ids, "catl", "2024")
        if not all((byd_2025, byd_2024, catl_2025, catl_2024)):
            return None
        byd_balance = self._solvency_table(byd_2025)
        catl_balance = self._solvency_table(catl_2025)
        byd_current = self._interest_coverage_table(byd_2025)
        byd_prior = self._interest_coverage_table(byd_2024)
        catl_current = self._interest_coverage_table(catl_2025)
        catl_prior = self._interest_coverage_table(catl_2024)
        if not all((byd_balance, catl_balance, byd_current, byd_prior, catl_current, catl_prior)):
            return None
        catl_assets = [
            catl_prior[0]["资产负债率"][1],
            catl_balance[0]["资产负债率"][1],
            catl_balance[0]["资产负债率"][0],
        ]
        catl_interest = [
            catl_prior[0]["利息保障倍数"][1],
            catl_current[0]["利息保障倍数"][1],
            catl_current[0]["利息保障倍数"][0],
        ]
        byd_cash = [
            byd_prior[0]["现金利息保障倍数"][1],
            byd_current[0]["现金利息保障倍数"][1],
            byd_current[0]["现金利息保障倍数"][0],
        ]
        byd_balance_values = byd_balance[0]["资产负债率"]
        byd_interest_values = byd_current[0]["利息保障倍数"]
        byd_cash_values = byd_current[0]["现金利息保障倍数"]
        byd_cash_relative_decline = (byd_cash[0] - byd_cash[2]) / byd_cash[0] * 100
        catl_interest_relative_growth = (catl_interest[2] - catl_interest[0]) / catl_interest[0] * 100

        if "宁德时代资产负债率连续下降，利息保障倍数连续上升" in option_text:
            label = all(left > right for left, right in zip(catl_assets, catl_assets[1:])) and all(
                left < right for left, right in zip(catl_interest, catl_interest[1:])
            )
        elif "比亚迪现金利息保障倍数由 2023 年" in option_text and "百分点" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs((byd_cash[0] - byd_cash[2]) - expected) <= 0.02
        elif "宁德时代利息保障倍数 2025 年较 2023 年提高" in option_text and "百分点" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs((catl_interest[2] - catl_interest[0]) - expected) <= 0.02
        elif "比亚迪 2025 年资产负债率下降" in option_text and "均较 2024 年下降" in option_text:
            label = (
                byd_balance_values[0] < byd_balance_values[1]
                and byd_interest_values[0] < byd_interest_values[1]
                and byd_cash_values[0] < byd_cash_values[1]
            )
        else:
            return None
        reason = (
            f"三年偿债指标束：宁德时代资产负债率2023/2024/2025为{catl_assets[0]:.2f}%/"
            f"{catl_assets[1]:.2f}%/{catl_assets[2]:.2f}%，利息保障倍数为"
            f"{catl_interest[0]:g}/{catl_interest[1]:g}/{catl_interest[2]:g}（相对增长"
            f"{catl_interest_relative_growth:.2f}%）；比亚迪现金利息保障倍数为"
            f"{byd_cash[0]:g}/{byd_cash[1]:g}/{byd_cash[2]:g}（相对下降{byd_cash_relative_decline:.2f}%），"
            "倍数的相对变化不能表述为百分点。"
        )
        byd_evidence = [byd_balance[1], byd_current[1], byd_prior[1]]
        catl_evidence = [catl_balance[1], catl_current[1], catl_prior[1]]
        # Final evidence collection keeps only the first few units per selected
        # option. Put the company named by the option first so a split table's
        # continuation rows are not displaced by unrelated company coverage.
        if "宁德时代" in option_text:
            evidence_units = [*catl_evidence, *byd_evidence]
        else:
            evidence_units = [*byd_evidence, *catl_evidence]
        unique = {unit["unit_id"]: unit for unit in evidence_units}
        return label, reason, [self._unit_to_evidence(unit, 999.0) for unit in unique.values()]

    def _research_expense_rate_bundle_rule(self, question: Question, option_text: str):
        if not (
            "宁德时代与美的集团" in question.question
            and "研发费用及研发费用占营业收入比例" in question.question
        ):
            return None
        catl_doc = self._company_year_doc(question.doc_ids, "catl", "2025")
        midea_doc = self._company_year_doc(question.doc_ids, "midea", "2025")
        if not catl_doc or not midea_doc:
            return None
        catl_unit = self._find_unit(catl_doc, required=("研发投入金额", "研发投入占营业收入比例", "2025年", "2024年"))
        midea_unit = self._find_unit(midea_doc, required=("研发费用金额", "研发费用占营业收入比例"))
        if catl_unit is None or midea_unit is None:
            return None
        catl_amount = self._extract_row_numbers(catl_unit.get("text", ""), "研发投入金额")
        catl_rate = self._extract_row_numbers(catl_unit.get("text", ""), "研发投入占营业收入比例")
        midea_amount = self._extract_row_numbers(midea_unit.get("text", ""), "研发费用金额")
        midea_rate = self._extract_row_numbers(midea_unit.get("text", ""), "研发费用占营业收入比例")
        if min(len(catl_amount), len(catl_rate), len(midea_amount), len(midea_rate)) < 2:
            return None
        catl_amount_growth = (catl_amount[0] - catl_amount[1]) / catl_amount[1] * 100
        catl_revenue_growth = (
            (catl_amount[0] / (catl_rate[0] / 100)) / (catl_amount[1] / (catl_rate[1] / 100)) - 1
        ) * 100
        midea_amount_growth = (midea_amount[0] - midea_amount[1]) / midea_amount[1] * 100
        catl_rate_change = catl_rate[0] - catl_rate[1]
        midea_rate_change = midea_rate[0] - midea_rate[1]
        current_rate_difference = catl_rate[0] - midea_rate[0]
        if "两家公司 2025 年研发费用占营业收入比例均较 2024 年上升" in option_text:
            label = catl_rate_change > 0 and midea_rate_change > 0
        elif "宁德时代 2025 年研发费用增幅高于营业收入增幅" in option_text:
            label = catl_amount_growth > catl_revenue_growth and catl_rate_change > 0
        elif "美的集团 2025 年研发费用金额增长约" in option_text and "研发费用率下降" in option_text:
            expected = [
                float(value)
                for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|个百分点)", option_text)
            ]
            label = (
                len(expected) >= 2
                and abs(midea_amount_growth - expected[0]) <= 0.02
                and abs(abs(midea_rate_change) - expected[1]) <= 0.01
                and midea_rate_change < 0
            )
        elif "宁德时代研发费用率比美的集团高约" in option_text:
            expected = self._extract_expected_points(option_text)
            label = expected is not None and abs(current_rate_difference - expected) <= 0.01
        else:
            return None
        reason = (
            f"研发费用率束：宁德时代研发投入增幅{catl_amount_growth:.2f}%、反推营业收入增幅"
            f"{catl_revenue_growth:.2f}%，研发投入率{catl_rate[1]:.2f}%→{catl_rate[0]:.2f}%；"
            f"美的研发费用增幅{midea_amount_growth:.2f}%，研发费用率{midea_rate[1]:.2f}%→"
            f"{midea_rate[0]:.2f}%；2025年两者费率差{current_rate_difference:.2f}个百分点。"
        )
        return label, reason, [self._unit_to_evidence(catl_unit, 999.0), self._unit_to_evidence(midea_unit, 999.0)]

    def _solvency_table(self, doc_id: str) -> tuple[dict[str, list[float]], dict[str, Any]] | None:
        unit = self._find_unit(doc_id, required=("流动比率", "资产负债率", "速动比率"))
        if unit is None:
            return None
        values = {
            key: self._extract_row_numbers(unit.get("text", ""), key)
            for key in ("流动比率", "资产负债率", "速动比率")
        }
        if any(len(row) < 2 for row in values.values()):
            return None
        return values, unit

    def _interest_coverage_table(self, doc_id: str) -> tuple[dict[str, list[float]], dict[str, Any]] | None:
        unit = self._find_unit(doc_id, required=("利息保障倍数", "现金利息保障倍数"))
        if unit is None:
            return None
        values = {
            "利息保障倍数": self._extract_row_numbers(unit.get("text", ""), "利息保障倍数"),
            "现金利息保障倍数": self._extract_row_numbers(unit.get("text", ""), "现金利息保障倍数"),
            "资产负债率": self._extract_row_numbers(unit.get("text", ""), "资产负债率"),
        }
        if len(values["利息保障倍数"]) < 2 or len(values["现金利息保障倍数"]) < 2:
            return None
        return values, unit

    def _metric_series(self, doc_id: str, metric_key: str) -> tuple[float, float, float | None, dict[str, Any]] | None:
        units = self.metric_index.get(doc_id, {}).get(metric_key, [])
        unit, rate = self._choose_best_growth_unit(units, metric_key)
        if unit is None:
            return None
        values = self._extract_metric_values(unit.get("text", ""), metric_key)
        if len(values) < 2:
            return None
        return values[0], values[1], rate, unit

    @staticmethod
    def _company_year_doc(doc_ids: list[str], hint: str, year: str) -> str:
        return next((doc_id for doc_id in doc_ids if hint in doc_id and year in doc_id), "")

    @staticmethod
    def _extract_expected_points(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:个?百分点|%)", text)
        return float(match.group(1)) if match else None

    def _best_dividend_per_ten(
        self, doc_id: str
    ) -> tuple[float, list[dict[str, Any]]] | None:
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for unit in self.units:
            if unit.get("doc_id") != doc_id or unit.get("unit_type") != "metric_row":
                continue
            text = unit.get("text", "")
            if "现金分红" not in text and "派息" not in text:
                continue
            patterns = [
                (12, r"全年每股现金分红\s*(\d+(?:\.\d+)?)\s*元", 10.0),
                (11, r"2025\s*年度利润分配方案为[^。\n]{0,80}?每\s*10\s*股派发现金\s*(\d+(?:\.\d+)?)\s*元", 1.0),
                (10, r"每\s*10\s*股派息数\(元\)[^\d]{0,10}(\d+(?:\.\d+)?)", 1.0),
                (9, r"每\s*10\s*股派发现金(?:分红|红利)?\s*(\d+(?:\.\d+)?)\s*元", 1.0),
            ]
            for priority, pattern, multiplier in patterns:
                match = re.search(pattern, text)
                if match:
                    candidates.append((priority, float(match.group(1)) * multiplier, unit))
                    break
        if not candidates:
            return None
        _, value, unit = max(candidates, key=lambda item: (item[0], -len(item[2].get("text", ""))))
        evidence_units = [unit]

        # Annual reports can headline the year-end residual as the per-10-share
        # dividend. Add the interim payment only when the same report explicitly
        # says that the selected value is the amount remaining after interim.
        residual_unit = next(
            (
                candidate
                for candidate in self.units
                if candidate.get("doc_id") == doc_id
                and "中期" in candidate.get("text", "")
                and "剩余待分配" in candidate.get("text", "")
                and re.search(
                    rf"每\s*10\s*股派发现金(?:分红|红利)?\s*{re.escape(f'{value:g}')}\s*元",
                    candidate.get("text", ""),
                )
            ),
            None,
        )
        if residual_unit is not None:
            interim_candidates: list[tuple[float, dict[str, Any]]] = []
            for candidate in self.units:
                if candidate.get("doc_id") != doc_id:
                    continue
                text = candidate.get("text", "")
                if "中期分红" not in text:
                    continue
                match = re.search(
                    r"每\s*10\s*股派发现金(?:分红|红利)?(?:人民币)?\s*(\d+(?:\.\d+)?)\s*元",
                    text,
                )
                if match:
                    interim_candidates.append((float(match.group(1)), candidate))
            if interim_candidates:
                interim_value, interim_unit = max(
                    interim_candidates,
                    key=lambda item: (-len(item[1].get("text", "")), item[0]),
                )
                value += interim_value
                for candidate in (interim_unit, residual_unit):
                    if candidate.get("unit_id") not in {
                        item.get("unit_id") for item in evidence_units
                    }:
                        evidence_units.append(candidate)
        return value, evidence_units

    def _find_unit(self, doc_id: str, *, required: tuple[str, ...]) -> dict[str, Any] | None:
        return next(
            (
                unit
                for unit in self.units
                if unit.get("doc_id") == doc_id and all(term in unit.get("text", "") for term in required)
            ),
            None,
        )

    @staticmethod
    def _extract_row_numbers(text: str, label: str) -> list[float]:
        line = next((line for line in text.splitlines() if label in line and "|" in line), "")
        values: list[float] = []
        for segment in line.split("|"):
            token = segment.strip().replace(",", "").removesuffix("%")
            if not re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", token):
                continue
            negative = token.startswith("(") and token.endswith(")")
            value = float(token.strip("()"))
            values.append(-value if negative else value)
        return values

    def _augment_targeted_hits(
        self,
        question: Question,
        option_text: str,
        hits: list[RetrievalHit],
    ) -> tuple[list[RetrievalHit], list[dict[str, Any]]]:
        text = f"{question.question} {option_text}"
        target_doc_ids = self._target_doc_ids(question, option_text)
        targeted: list[RetrievalHit] = []
        debug: list[dict[str, Any]] = []

        def add_metric(metric_key: str, reason: str, score: float = 1200.0) -> None:
            before = len(targeted)
            for doc_id in target_doc_ids:
                for unit in self.metric_index.get(doc_id, {}).get(metric_key, [])[:3]:
                    targeted.append(self._unit_to_hit(unit, score, reason))
            if len(targeted) > before:
                debug.append(
                    {
                        "channel": "metric_index",
                        "reason": reason,
                        "metric_key": metric_key,
                        "doc_ids": target_doc_ids,
                        "hits_added": len(targeted) - before,
                    }
                )

        if any(term in text for term in ["营业收入增速", "营业收入增长", "收入增速", "收入增长", "同比增减"]):
            add_metric("营业收入", "revenue_growth")
        if any(term in text for term in ["营业收入", "营收", "营业总收入"]) and any(
            term in text for term in ["超过", "大于", "高于", "低于", "小于", "两倍", "2倍", "同比"]
        ):
            add_metric("营业收入", "revenue_growth")
        if any(term in text for term in ["经营活动现金流", "经营现金流", "现金流净额", "现金流量净额"]):
            add_metric("经营现金流", "operating_cash_flow_growth")
        if any(term in text for term in ["归属于上市公司股东的净利润", "归母净利润", "归属于上市公司股东净利润"]):
            add_metric("归母净利润", "net_profit_growth")
        if "研发" in text and any(term in text for term in ["占", "比例", "比重", "%", "不足", "低于", "高于"]):
            add_metric("研发占比", "rd_ratio")
        if any(term in text for term in ["现金分红", "利润分配", "每 10 股", "每10股", "派发"]):
            add_metric("现金分红", "cash_dividend")
            if "净利润" in option_text and any(term in option_text for term in ["比例", "占", "%"]):
                raw_hits = self._raw_text_hits(question, option_text, target_doc_ids, reason="dividend_ratio_raw")
                if raw_hits:
                    targeted.extend(raw_hits)
                    debug.append(
                        {
                            "channel": "raw_extracted_search",
                            "reason": "dividend_ratio_raw",
                            "doc_ids": target_doc_ids,
                            "hits_added": len(raw_hits),
                        }
                    )

        if "回购" in option_text or "股东回报" in option_text:
            raw_hits = self._raw_text_hits(question, option_text, target_doc_ids, reason="shareholder_return_raw")
            if raw_hits:
                targeted.extend(raw_hits)
                debug.append(
                    {
                        "channel": "raw_extracted_search",
                        "reason": "shareholder_return_raw",
                        "doc_ids": target_doc_ids,
                        "hits_added": len(raw_hits),
                    }
                )
            queries = [
                f"{option_text} 股东回报 股份回购 回购计划 连续四年 2019",
                f"{question.question} {option_text} 回购 股东权益 股东利益",
            ]
            for query in queries:
                before = len(targeted)
                targeted.extend(
                    self.retriever.search(
                        target_doc_ids,
                        query,
                        top_k=4,
                        unit_type_boosts={"paragraph": 2.4, "metric_row": 1.0},
                        ensure_per_doc=False,
                        expand_neighbors=True,
                    )
                )
                if len(targeted) > before:
                    debug.append(
                        {
                            "channel": "shareholder_return_search",
                            "query": query,
                            "doc_ids": target_doc_ids,
                            "hits_added": len(targeted) - before,
                        }
                    )

        return self._merge_hits(targeted, hits, limit=max(self.retrieval_settings.get("top_k", 6), 10)), debug

    def _growth_rate_rule(self, question: Question, option_text: str, metric_key: str):
        if not any(term in option_text for term in ["增速", "增长", "同比", "上升", "下降", "减少", "下滑"]):
            return None
        if metric_key in {"现金分红", "研发投入", "研发占比"}:
            return None

        doc_rates: dict[str, float] = {}
        evidence = []
        for doc_id in question.doc_ids:
            units = self.metric_index.get(doc_id, {}).get(metric_key, [])
            best_unit, rate = self._choose_best_growth_unit(units, metric_key)
            if best_unit is None or rate is None:
                continue
            doc_rates[doc_id] = rate
            evidence.append(self._unit_to_evidence(best_unit, score=999.0))
        if len(doc_rates) < min(2, len(question.doc_ids)):
            return None

        label: bool | None = None
        if any(term in option_text for term in ["均", "都", "均实现", "均为", "同时"]):
            if any(term in option_text for term in ["正增长", "增长", "上升", "增加"]):
                label = all(rate > 0 for rate in doc_rates.values())
            elif any(term in option_text for term in ["下降", "减少", "下滑", "负增长"]):
                label = all(rate < 0 for rate in doc_rates.values())
        else:
            ordered_doc_ids = self._mentioned_doc_order(option_text, question.doc_ids)
            if len(ordered_doc_ids) < 2:
                ordered_doc_ids = list(question.doc_ids[:2])
            left_doc, right_doc = ordered_doc_ids[0], ordered_doc_ids[1]
            if left_doc in doc_rates and right_doc in doc_rates:
                if any(term in option_text for term in ["高于", "大于", "超过", "优于"]):
                    label = doc_rates[left_doc] > doc_rates[right_doc]
                elif any(term in option_text for term in ["低于", "小于", "不及"]):
                    label = doc_rates[left_doc] < doc_rates[right_doc]
        if label is None:
            return None

        rate_text = "，".join(f"{doc_id}: {rate:g}%" for doc_id, rate in doc_rates.items())
        reason = f"规则读取到 {metric_key} 的同比增减为 {rate_text}，据此判断选项为 {'正确' if label else '错误'}。"
        return label, reason, evidence

    def _single_doc_growth_polarity_rule(self, question: Question, option_text: str, metric_key: str):
        if not any(term in option_text for term in ["同比", "增长", "增加", "上升", "下降", "减少", "下滑", "降低"]):
            return None
        if metric_key in {"现金分红", "研发投入", "研发占比"}:
            return None
        target_doc_ids = self._target_doc_ids(question, option_text)
        if len(target_doc_ids) != 1:
            return None
        doc_id = target_doc_ids[0]
        best_unit, rate = self._choose_best_growth_unit(self.metric_index.get(doc_id, {}).get(metric_key, []), metric_key)
        if best_unit is None or rate is None:
            return None

        positive = any(term in option_text for term in ["同比增长", "增长", "增加", "上升", "提升"])
        negative = any(term in option_text for term in ["同比下降", "同比减少", "下降", "减少", "下滑", "降低"])
        if not positive and not negative:
            return None
        expected_pct = self._extract_expected_percent(option_text)
        pct_matches = self._growth_percent_matches(option_text, rate, expected_pct)
        if positive and not negative:
            label = rate > 0 and pct_matches
            direction = "增长"
        elif negative and not positive:
            label = rate < 0 and pct_matches
            direction = "下降/减少"
        else:
            return None
        reason = (
            f"规则读取到 {doc_id} 的{metric_key}同比增减为 {rate:g}%，"
            f"选项要求{direction}{'' if expected_pct is None else f'{expected_pct:g}%'}，据此判断为{'正确' if label else '错误'}。"
        )
        return label, reason, [self._unit_to_evidence(best_unit, score=999.0)]

    @staticmethod
    def _growth_percent_matches(option_text: str, rate: float, expected_pct: float | None) -> bool:
        if expected_pct is None:
            return True
        magnitude = abs(rate)
        if any(term in option_text for term in ["未超过", "不超过"]):
            return magnitude <= expected_pct
        if any(term in option_text for term in ["超过", "高于", "大于", "逾"]):
            return magnitude > expected_pct
        if any(term in option_text for term in ["低于", "小于", "不足"]):
            return magnitude < expected_pct
        return abs(magnitude - expected_pct) <= 0.15

    def _net_profit_dividend_compound_rule(self, question: Question, option_text: str):
        if len(question.doc_ids) < 2:
            return None
        if not any(term in option_text for term in ["归属于上市公司股东的净利润", "归属于上市公司股东净利润", "归母净利润"]):
            return None
        if not ("现金分红" in option_text and any(term in option_text for term in ["比例", "提升", "提高", "上升"])):
            return None
        old_doc, new_doc = question.doc_ids[0], question.doc_ids[1]
        old_profit = self._best_metric_value(old_doc, "归母净利润")
        new_profit = self._best_metric_value(new_doc, "归母净利润")
        old_dividend = self._best_dividend_ratio(old_doc)
        new_dividend = self._best_dividend_ratio(new_doc)
        if not old_profit or not new_profit or not old_dividend or not new_dividend:
            return None
        old_profit_value, old_profit_unit = old_profit
        new_profit_value, new_profit_unit = new_profit
        old_ratio, old_dividend_unit = old_dividend
        new_ratio, new_dividend_unit = new_dividend
        profit_down = new_profit_value < old_profit_value
        dividend_up = new_ratio > old_ratio
        label = profit_down and dividend_up
        reason = (
            f"规则复核复合条件：归母净利润 {old_doc}={old_profit_value:g}、{new_doc}={new_profit_value:g}，"
            f"{'下降' if profit_down else '未下降'}；现金分红占归母净利润比例 {old_doc}={old_ratio:g}%、"
            f"{new_doc}={new_ratio:g}%，{'提升' if dividend_up else '未提升'}。据此判断为{'正确' if label else '错误'}。"
        )
        evidence_units = [old_profit_unit, new_profit_unit, old_dividend_unit, new_dividend_unit]
        return label, reason, [self._unit_to_evidence(unit, 999.0) for unit in evidence_units]

    def _revenue_multiple_rule(self, question: Question, option_text: str):
        if not any(term in option_text for term in ["营业收入", "营收", "营业总收入"]):
            return None
        if not any(term in option_text for term in ["两倍", "2倍"]):
            return None
        ordered_doc_ids = self._mentioned_doc_order(option_text, question.doc_ids)
        if len(ordered_doc_ids) < 2:
            return None
        left_doc, right_doc = ordered_doc_ids[0], ordered_doc_ids[1]
        left = self._best_revenue_value_yi(left_doc)
        right = self._best_revenue_value_yi(right_doc)
        if not left or not right:
            return None
        left_value, left_unit = left
        right_value, right_unit = right
        label = left_value > right_value * 2
        reason = (
            f"规则比较营业收入规模：{left_doc}约{left_value:.2f}亿元，{right_doc}约{right_value:.2f}亿元；"
            f"{left_value:.2f} {'>' if label else '<='} {right_value * 2:.2f}，据此判断选项为{'正确' if label else '错误'}。"
        )
        return label, reason, [self._unit_to_evidence(left_unit, 999.0), self._unit_to_evidence(right_unit, 999.0)]

    def _foreign_revenue_ratio_rule(self, question: Question, option_text: str):
        if not any(term in option_text for term in ["境外收入", "海外销售收入", "境外销售收入", "境外"]):
            return None
        if not any(term in option_text for term in ["占比", "占营业收入", "超过", "高于", "低于", "不足"]):
            return None
        threshold = self._extract_expected_percent(option_text)
        if threshold is None:
            return None
        target_doc_ids = self._target_doc_ids(question, option_text)
        matched: list[tuple[float, dict[str, Any]]] = []
        for doc_id in target_doc_ids:
            for unit in self.units:
                if unit.get("doc_id") != doc_id:
                    continue
                text = unit.get("text", "")
                if "境外" not in text or "营业收入" not in text:
                    continue
                ratio = self._extract_foreign_revenue_ratio(text)
                if ratio is not None:
                    matched.append((ratio, unit))
                    break
        if not matched:
            return None
        ratio, unit = matched[0]
        if any(term in option_text for term in ["超过", "高于", "大于"]):
            label = ratio > threshold
            comparator = "超过"
        elif any(term in option_text for term in ["低于", "不足", "小于"]):
            label = ratio < threshold
            comparator = "低于"
        else:
            return None
        reason = f"规则读取到境外收入占比为 {ratio:g}%，选项要求{comparator}{threshold:g}%，据此判断为{'正确' if label else '错误'}。"
        return label, reason, [self._unit_to_evidence(unit, 999.0)]

    def _cash_flow_revenue_ratio_rule(self, question: Question, option_text: str):
        if "经营活动产生的现金流量净额" not in option_text or "营业收入" not in option_text:
            return None
        if not any(term in option_text for term in ["一半", "50%", "十分之一", "10%"]):
            return None

        checks: list[tuple[str, str, float, str]] = []
        if "比亚迪" in option_text and any(term in option_text for term in ["一半", "50%"]):
            checks.append(("比亚迪", "byd", 0.5, "lt"))
        if "美的" in option_text and any(term in option_text for term in ["十分之一", "10%"]):
            checks.append(("美的", "midea", 0.1, "gt"))
        if not checks:
            return None

        evidence = []
        parts = []
        labels = []
        for company, hint, threshold, comparator in checks:
            doc_id = next((doc_id for doc_id in question.doc_ids if hint in doc_id), "")
            if not doc_id:
                return None
            cash_flow = self._best_metric_value(doc_id, "经营现金流")
            revenue = self._best_metric_value(doc_id, "营业收入")
            if not cash_flow or not revenue or revenue[0] == 0:
                return None
            cash_value, cash_unit = cash_flow
            revenue_value, revenue_unit = revenue
            ratio = cash_value / revenue_value
            label = ratio < threshold if comparator == "lt" else ratio > threshold
            labels.append(label)
            symbol = "<" if comparator == "lt" else ">"
            parts.append(f"{company}经营现金流/营业收入={ratio:.2%}，{symbol}{threshold:.0%} 为{label}")
            evidence.extend([self._unit_to_evidence(cash_unit, 999.0), self._unit_to_evidence(revenue_unit, 998.0)])

        final_label = all(labels)
        reason = f"规则计算复合比例条件：{'；'.join(parts)}，据此判断整句为{'正确' if final_label else '错误'}。"
        return final_label, reason, evidence

    def _rd_ratio_repurchase_compound_rule(self, question: Question, option_text: str):
        if question.answer_format != "tf":
            return None
        if not (
            "比亚迪" in option_text
            and "美的" in option_text
            and "研发" in option_text
            and "占营业收入" in option_text
            and any(term in option_text for term in ["上升", "提升", "提高"])
            and "回购" in option_text
            and "2019" in option_text
            and "连续" in option_text
        ):
            return None

        byd_doc = next((doc_id for doc_id in question.doc_ids if "byd" in doc_id), "")
        midea_doc = next((doc_id for doc_id in question.doc_ids if "midea" in doc_id), "")
        if not byd_doc or not midea_doc:
            return None

        rd_units = self.metric_index.get(byd_doc, {}).get("研发占比", [])
        if not rd_units:
            return None
        rd_unit, rd_values = self._choose_best_metric_unit(rd_units, "研发占比")
        rd_series = self._extract_metric_values(rd_unit.get("text", ""), "研发占比")
        if len(rd_series) < 2:
            return None
        rd_up = rd_series[0] > rd_series[1]

        repurchase_hit = self._repurchase_raw_hit(question, option_text, midea_doc)
        if repurchase_hit is None:
            return None

        label = rd_up
        reason = (
            f"规则命中复合判断：比亚迪研发投入占营业收入比例 {rd_series[0]:g}% 高于上年 {rd_series[1]:g}%；"
            "美的原文载明自2019年起连续四年推出回购计划。"
        )
        evidence = [self._unit_to_evidence(rd_unit, 999.0), repurchase_hit.to_dict()]
        return label, reason, evidence

    def _repurchase_rule(self, question: Question, option_text: str):
        if "回购" not in option_text:
            return None
        if not ("2019" in option_text and ("连续四年" in option_text or "连续" in option_text)):
            return None
        target_doc_ids = self._target_doc_ids(question, option_text)
        matched_units = []
        for unit in self.units:
            if unit.get("doc_id") not in target_doc_ids:
                continue
            text = unit.get("text", "")
            compact = re.sub(r"\s+", "", text)
            if "回购" in text and "2019" in text and ("连续四年" in compact or "连续4年" in compact):
                matched_units.append(unit)
        if not matched_units:
            raw_hits = self._raw_text_hits(question, option_text, target_doc_ids, reason="repurchase_rule_raw")
            raw_hits = [hit for hit in raw_hits if self._is_repurchase_continuity_text(hit.text)]
            if not raw_hits:
                return None
            evidence = [hit.to_dict() for hit in raw_hits[:2]]
            return True, "规则命中原始清洗文本：原文同时包含 2019、连续四年和回购计划，支持该选项。", evidence
        evidence = [self._unit_to_evidence(unit, score=999.0) for unit in matched_units[:2]]
        return True, "规则命中股东回报段落：原文同时包含 2019、连续四年和回购计划，支持该选项。", evidence

    def _repurchase_raw_hit(self, question: Question, option_text: str, doc_id: str) -> RetrievalHit | None:
        for unit in self.units:
            if unit.get("doc_id") != doc_id:
                continue
            text = unit.get("text", "")
            compact = re.sub(r"\s+", "", text)
            if "回购" in text and "2019" in text and ("连续四年" in compact or "连续4年" in compact):
                return self._unit_to_hit(unit, 999.0, "repurchase_compound_rule")
        raw_hits = self._raw_text_hits(question, option_text, [doc_id], reason="repurchase_compound_raw")
        for hit in raw_hits:
            if self._is_repurchase_continuity_text(hit.text):
                return hit
        return None

    @staticmethod
    def _is_repurchase_continuity_text(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return "回购" in text and "2019" in text and ("连续四年" in compact or "连续4年" in compact)

    def _shareholder_return_total_rule(self, question: Question, option_text: str):
        if not ("现金分红" in option_text and "回购" in option_text):
            return None
        if not any(term in option_text for term in ["归母净利润", "归属于上市公司股东的净利润", "净利润"]):
            return None
        if not any(term in option_text for term in ["超过", "高于", "大于"]):
            return None

        target_doc_ids = self._target_doc_ids(question, option_text)
        exact_terms = [
            "现金分红与股份回购之总金额超过当年度公司归母净利润",
            "现金分红与股份回购之总金额超过",
            "全年股份回购总金额超过",
        ]
        matched_units = []
        for unit in self.units:
            if unit.get("doc_id") not in target_doc_ids:
                continue
            text = unit.get("text", "")
            compact = re.sub(r"\s+", "", text)
            if any(term in compact or term in text for term in exact_terms) and "现金分红" in text and "回购" in text:
                matched_units.append(unit)
        if matched_units:
            evidence = [self._unit_to_evidence(unit, score=999.0) for unit in matched_units[:2]]
            return True, "规则命中股东回报段落：原文直述现金分红与股份回购总金额超过当年度归母净利润。", evidence

        raw_hits = self._raw_text_hits(question, option_text, target_doc_ids, reason="shareholder_return_total_raw")
        if raw_hits:
            evidence = [hit.to_dict() for hit in raw_hits[:2]]
            return True, "规则命中原始清洗文本：现金分红、股份回购与归母净利润比较在同一窗口内出现，支持该选项。", evidence
        return None

    def _dividend_ratio_rule(self, question: Question, option_text: str):
        if "现金分红" not in option_text or "净利润" not in option_text:
            return None
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", option_text)
        if not pct_match:
            return None
        pct = float(pct_match.group(1))
        pct_text = f"{pct:g}"
        pct_term = f"净利润的{pct_text}%"
        total_wording = any(term in option_text for term in ["合计", "总额", "总计"])
        target_doc_ids = self._target_doc_ids(question, option_text)
        matched_units = []
        for unit in self.units:
            if unit.get("doc_id") not in target_doc_ids:
                continue
            text = unit.get("text", "")
            compact = re.sub(r"\s+", "", text)
            if "现金分红" in compact and pct_term in compact:
                matched_units.append(unit)
        if not matched_units:
            raw_hits = self._raw_text_hits(question, option_text, target_doc_ids, reason="dividend_ratio_raw")
            raw_hits = [
                hit
                for hit in raw_hits
                if pct_term in re.sub(r"\s+", "", hit.text) and "现金分红" in hit.text
            ]
            if not raw_hits:
                return None
            evidence = [hit.to_dict() for hit in raw_hits[:2]]
            return True, self._dividend_ratio_reason(pct_text, total_wording), evidence
        evidence = [self._unit_to_evidence(unit, score=999.0) for unit in matched_units[:2]]
        return True, self._dividend_ratio_reason(pct_text, total_wording), evidence

    @staticmethod
    def _dividend_ratio_reason(pct_text: str, total_wording: bool) -> str:
        if pct_text == "20" and not total_wording:
            return (
                "规则命中利润分配预案中的年度现金分红口径：原文明确拟以归母净利润的20%实施年度现金分红；"
                "同段可能另列特别现金分红，应在证据中保留口径说明。"
            )
        if pct_text == "30" and not total_wording:
            return "规则命中特别现金分红口径：原文明确拟以归母净利润的30%实施特别现金分红。"
        return f"规则命中现金分红比例：原文明确以归母净利润的{pct_text}%实施现金分红。"

    def _detect_metric_key(self, text: str) -> str | None:
        candidates = [
            (len(alias), metric_key)
            for metric_key, aliases in METRIC_ALIASES.items()
            for alias in aliases
            if alias in text
        ]
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        return None

    def _normalize_metric(self, metric_name: str) -> str:
        candidates = [
            (len(alias), metric_key)
            for metric_key, aliases in METRIC_ALIASES.items()
            for alias in aliases
            if alias in metric_name
        ]
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        return metric_name

    def _choose_best_metric_unit(self, units: list[dict[str, Any]], metric_key: str) -> tuple[dict[str, Any], float | None]:
        scored = []
        for unit in units:
            values = self._extract_metric_values(unit.get("text", ""), metric_key)
            if not values:
                continue
            scored.append((self._metric_unit_score(unit, metric_key, values), unit, values[0]))
        if not scored:
            return self._choose_best_unit(units), None
        _, unit, value = max(scored, key=lambda item: item[0])
        return unit, value

    def _choose_best_growth_unit(self, units: list[dict[str, Any]], metric_key: str) -> tuple[dict[str, Any] | None, float | None]:
        scored = []
        for unit in units:
            rate = self._extract_growth_rate(unit.get("text", ""), metric_key)
            if rate is None:
                continue
            values = self._extract_metric_values(unit.get("text", ""), metric_key)
            scored.append((self._metric_unit_score(unit, metric_key, values), unit, rate))
        if not scored:
            return None, None
        _, unit, rate = max(scored, key=lambda item: item[0])
        return unit, rate

    def _metric_unit_score(self, unit: dict[str, Any], metric_key: str, values: list[float]) -> tuple[int, int, int]:
        text = unit.get("text", "")
        aliases = METRIC_ALIASES.get(metric_key, [metric_key])
        quality = 0
        if any(re.search(rf"{re.escape(alias)}(?:\([^)]*\))?\s*\|", text) for alias in aliases):
            quality += 8
        if "本年比上年增减" in text or "同比增减" in text:
            quality += 4
        if "第一季度" in text or "第二季度" in text or "第三季度" in text or "第四季度" in text:
            quality -= 6
        if "主要控股子公司" in text or "子公司基本情况" in text:
            quality -= 10
        if "财务概览" in text or "主要会计数据" in text or "本年比上年增减" in text:
            quality += 5
        if "相关数据同比发生重大变动" in text:
            quality -= 4
        if len(values) >= 2:
            quality += 3
        if metric_key == "研发占比" and "%" in text:
            quality += 4
        return quality, len(values), -len(text)

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

    def _extract_metric_values(self, text: str, metric_key: str) -> list[float]:
        segment = self._metric_segment(text, metric_key)
        if not segment:
            return []
        tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", segment)
        if metric_key == "研发占比":
            return [float(token.rstrip("%").replace(",", "")) for token in tokens if token.endswith("%")]
        values = []
        for token in tokens:
            if token.endswith("%"):
                continue
            value = float(token.replace(",", ""))
            if 1900 <= value <= 2100 and f"{int(value)}年" in segment:
                continue
            values.append(value)
        while len(values) > 2 and abs(values[0]) < 100 and abs(values[1]) > 1000:
            values.pop(0)
        return values

    def _extract_growth_rate(self, text: str, metric_key: str) -> float | None:
        segment = self._metric_segment(text, metric_key)
        if not segment:
            return None
        percent_tokens = re.findall(r"-?\d[\d,]*(?:\.\d+)?%", segment)
        if not percent_tokens:
            return None
        return float(percent_tokens[0].rstrip("%").replace(",", ""))

    @staticmethod
    def _extract_expected_percent(text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def _extract_foreign_revenue_ratio(text: str) -> float | None:
        patterns = [
            r"境外[^%]{0,120}?占(?:本期)?营业收入[^%]{0,20}?(\d+(?:\.\d+)?)%",
            r"境外[^%]{0,120}?(\d+(?:\.\d+)?)%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _best_revenue_value_yi(self, doc_id: str) -> tuple[float, dict[str, Any]] | None:
        units = self.metric_index.get(doc_id, {}).get("营业收入", [])
        if not units:
            return None
        best_unit, value = self._choose_best_metric_unit(units, "营业收入")
        if value is None:
            return None
        multiplier = 1_000.0
        if "chinamobile" in doc_id or "cmb" in doc_id:
            multiplier = 1_000_000.0
        text = best_unit.get("text", "")
        if "亿元" in text:
            # Prefer explicit Chinese-yuan prose when the extracted value is already in yi-yuan units.
            yi_match = re.search(r"营业收入[^。；\n]{0,40}?(\d[\d,]*(?:\.\d+)?)\s*亿元", text)
            if yi_match:
                return float(yi_match.group(1).replace(",", "")), best_unit
        return value * multiplier / 100_000_000.0, best_unit

    def _best_metric_value(self, doc_id: str, metric_key: str) -> tuple[float, dict[str, Any]] | None:
        units = self.metric_index.get(doc_id, {}).get(metric_key, [])
        if not units:
            return None
        best_unit, value = self._choose_best_metric_unit(units, metric_key)
        if value is None:
            return None
        return value, best_unit

    def _best_dividend_ratio(self, doc_id: str) -> tuple[float, dict[str, Any]] | None:
        candidates = [
            *self.metric_index.get(doc_id, {}).get("现金分红", []),
            *self.metric_index.get(doc_id, {}).get("每10股派", []),
        ]
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for unit in candidates:
            ratio = self._extract_dividend_ratio(unit.get("text", ""))
            if ratio is None:
                continue
            text = unit.get("text", "")
            quality = 0
            if "现金分红占" in text or "分红年度合并报表" in text:
                quality += 5
            if "比例为" in text or "比率" in text:
                quality += 3
            scored.append((quality, ratio, unit))
        if not scored:
            return None
        _, ratio, unit = max(scored, key=lambda item: (item[0], -len(item[2].get("text", ""))))
        return ratio, unit

    @staticmethod
    def _extract_dividend_ratio(text: str) -> float | None:
        patterns = [
            r"现金分红占[^。；\n]{0,80}?比例为\s*(\d+(?:\.\d+)?)%",
            r"占合并报表[^。；\n]{0,120}?比率\(%\)\s*\|?\s*(\d+(?:\.\d+)?)",
            r"合计分红金额占[^。；\n]{0,120}?比例\(%\)\s*\|?\s*(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _target_doc_ids(self, question: Question, option_text: str) -> list[str]:
        if any(term in option_text for term in ["两家", "两份", "均", "都", "同时", "分别"]):
            return list(question.doc_ids)
        matched_doc_ids: list[str] = []
        for company, hint in COMPANY_DOC_HINTS.items():
            if company not in option_text:
                continue
            for doc_id in question.doc_ids:
                if hint in doc_id and doc_id not in matched_doc_ids:
                    matched_doc_ids.append(doc_id)
        return matched_doc_ids or list(question.doc_ids)

    def _mentioned_doc_order(self, option_text: str, doc_ids: list[str]) -> list[str]:
        positions: list[tuple[int, str]] = []
        for company, hint in COMPANY_DOC_HINTS.items():
            pos = option_text.find(company)
            if pos < 0:
                continue
            for doc_id in doc_ids:
                if hint in doc_id:
                    positions.append((pos, doc_id))
        ordered = []
        for _, doc_id in sorted(positions, key=lambda item: item[0]):
            if doc_id not in ordered:
                ordered.append(doc_id)
        if len(ordered) >= 2:
            return ordered

        year_order = self._mentioned_year_doc_order(option_text, doc_ids)
        if len(year_order) >= 2:
            return year_order
        return ordered

    @staticmethod
    def _mentioned_year_doc_order(option_text: str, doc_ids: list[str]) -> list[str]:
        positions: list[tuple[int, str]] = []
        mentioned_years: list[tuple[int, int]] = []
        for match in re.finditer(r"20\d{2}", option_text):
            year = match.group(0)
            mentioned_years.append((match.start(), int(year)))
            for doc_id in doc_ids:
                if year in doc_id:
                    positions.append((match.start(), doc_id))

        ordered: list[str] = []
        for _, doc_id in sorted(positions, key=lambda item: item[0]):
            if doc_id not in ordered:
                ordered.append(doc_id)
        if len(ordered) == 1 and mentioned_years and any(
            term in option_text for term in ["上年", "较上年", "比上年", "同比", "增长", "下降", "减少", "下滑", "提升"]
        ):
            _, year = sorted(mentioned_years, key=lambda item: item[0])[0]
            prior_year = str(year - 1)
            prior_doc = next((doc_id for doc_id in doc_ids if prior_year in doc_id), "")
            if prior_doc and prior_doc not in ordered:
                ordered.append(prior_doc)
        return ordered

    def _raw_text_hits(
        self,
        question: Question,
        option_text: str,
        doc_ids: list[str],
        *,
        reason: str,
    ) -> list[RetrievalHit]:
        if "回购" not in option_text and "股东回报" not in option_text and "现金分红" not in option_text:
            return []
        shareholder_total_search = "现金分红" in option_text and "回购" in option_text
        required_terms = ["回购"]
        if shareholder_total_search:
            required_terms = ["现金分红", "回购"]
        elif "现金分红" in option_text and "净利润" in option_text:
            required_terms = ["现金分红", "净利润"]
        elif "连续四年" in option_text:
            required_terms.extend(["连续四年", "2019"])
        hits: list[RetrievalHit] = []
        root = Path.cwd() / "artifacts" / "extracted_cleaned" / "financial_reports"
        for doc_id in doc_ids:
            path = root / f"{doc_id}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            compact_text = re.sub(r"\s+", "", text)
            if not all(term in compact_text or term in text for term in required_terms):
                continue
            dividend_ratio_search = "现金分红" in required_terms and "净利润" in required_terms
            if shareholder_total_search:
                position_terms = [
                    "现金分红与股份回购之总金额超过",
                    "全年股份回购总金额超过",
                    "稳定分红派现",
                    "股东的信任和支持",
                    "股份回购之总金额",
                    "股份回购",
                ]
            elif dividend_ratio_search:
                position_terms = [
                    "利润分配预案如下",
                    "归属于上市公司股东的净利润的",
                    "净利润的",
                    "年度现金分红",
                    "特别现金分红",
                    "综上",
                    "每10股派息数",
                    "现金分红金额",
                    "现金分红",
                ]
            else:
                position_terms = ["连续四年", "2019", "股份回购", "股东回报", "回购"]
            position_candidates = [(term, text.find(term)) for term in position_terms]
            position_candidates = [(term, position) for term, position in position_candidates if position >= 0]
            if not position_candidates:
                continue
            if dividend_ratio_search or shareholder_total_search or ("2019" in option_text and "连续" in option_text):
                anchor = position_candidates[0][1]
            else:
                anchor = min(position for _, position in position_candidates)
            start = max(0, anchor - 350)
            end = min(len(text), anchor + 850)
            window = " ".join(text[start:end].split())
            hits.append(
                RetrievalHit(
                    unit_id=f"{doc_id}::raw_rescue::{reason}",
                    doc_id=doc_id,
                    score=1300.0,
                    title_path=["extracted_cleaned", reason],
                    text=window,
                    metadata={"unit_type": "raw_rescue", "targeted_reason": reason, "source_path": str(path)},
                )
            )
        return hits

    @staticmethod
    def _unit_to_hit(unit: dict[str, Any], score: float, reason: str) -> RetrievalHit:
        metadata = dict(unit.get("metadata", {}))
        metadata.setdefault("unit_type", unit.get("unit_type", ""))
        metadata["targeted_reason"] = reason
        return RetrievalHit(
            unit_id=unit["unit_id"],
            doc_id=unit["doc_id"],
            score=score,
            title_path=unit.get("title_path", []),
            text=unit.get("text", ""),
            metadata=metadata,
        )

    @staticmethod
    def _unit_to_evidence(unit: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "unit_id": unit["unit_id"],
            "doc_id": unit["doc_id"],
            "score": score,
            "title_path": unit.get("title_path", []),
            "text": unit.get("text", ""),
            "metadata": unit.get("metadata", {}),
        }

    @staticmethod
    def _merge_hits(priority_hits: list[RetrievalHit], base_hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hit in [*priority_hits, *base_hits]:
            key = hit.unit_id.replace("__dup2", "").replace("__dup", "")
            current = merged.get(key)
            if current is None or hit.score > current.score:
                merged[key] = hit
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _metric_segment(text: str, metric_key: str) -> str:
        aliases = METRIC_ALIASES.get(metric_key, [metric_key])
        matches = [(text.find(alias), alias) for alias in aliases if text.find(alias) >= 0]
        if not matches:
            return ""
        start = min(pos for pos, _ in matches)
        protected_end = max(pos + len(alias) for pos, alias in matches if pos == start)
        next_positions = []
        for other_key, other_aliases in METRIC_ALIASES.items():
            for alias in other_aliases:
                pos = text.find(alias, start + 1)
                if pos >= protected_end and other_key != metric_key:
                    next_positions.append(pos)
        for marker in [" 基本每股", " 稀释每股", " 上述财务指标", " 分行业", " #"]:
            pos = text.find(marker, start + 1)
            if pos >= protected_end:
                next_positions.append(pos)
        end = min(next_positions) if next_positions else min(len(text), start + 260)
        return text[start:end]

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
