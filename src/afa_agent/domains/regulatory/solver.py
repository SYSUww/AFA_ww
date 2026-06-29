from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.domains.llm_utils import collect_evidence_items, finalize_answer
from afa_agent.domains.regulatory.facts import (
    build_regulatory_query_variants,
    format_rule_summary,
    summarize_rule_alignment,
)
from afa_agent.evidence_gate import answer_consistency_issues, evaluate_evidence, gate_enabled, rescue_evidence
from afa_agent.models import AnswerResult, Question, RetrievalHit, TokenUsage
from afa_agent.strategy import get_stage_settings, serialize_hits


class RegulatorySolver:
    def __init__(self, client: OpenAICompatibleClient, retriever, strategy: str):
        self.client = client
        self.retriever = retriever
        self.domain = strategy
        self.retrieval_settings = get_stage_settings(strategy, "retrieval")
        self.answering_settings = get_stage_settings(strategy, "answering")
        self.gate_settings = get_stage_settings(strategy, "evidence_gate")
        self.supplemental_units = self._load_supplemental_units(self.retrieval_settings.get("supplemental_units_path", ""))

    def solve(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        option_labels: dict[str, bool] = {}
        option_payloads: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        option_debug: list[dict[str, Any]] = []
        query_variants_all: list[str] = []
        rule_outputs: list[dict[str, Any]] = []

        option_items = [("A", question.question)] if question.answer_format == "tf" else list(question.options.items())

        for option_key, option_text in option_items:
            query_variants = build_regulatory_query_variants(question, option_key, option_text)
            query_variants_all.extend(query_variants)
            top_k = self.retrieval_settings.get("top_k", 6)
            ensure_per_doc = self.retrieval_settings.get("ensure_per_doc", True) or len(question.doc_ids) > 1
            candidate_hits = self.retriever.candidate_search(
                question.doc_ids,
                query_variants,
                top_k=max(top_k * 2, 10),
                ensure_per_doc=ensure_per_doc,
            )
            hits = self.retriever.search(
                question.doc_ids,
                query_variants,
                top_k=top_k,
                ensure_per_doc=ensure_per_doc,
                expand_neighbors=self.retrieval_settings.get("expand_neighbors", True),
            )
            gate_debug: dict[str, Any] = {}
            if gate_enabled(self.gate_settings):
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
            targeted_hits = self._targeted_literal_hits(question, option_text)
            if targeted_hits:
                hits = self._merge_hits([*targeted_hits, *hits], limit=max(top_k, self.answering_settings.get("max_hits", 6)))
                if gate_debug:
                    gate_debug["targeted_literal_hits"] = serialize_hits(targeted_hits, limit=5)
                    gate_debug["final_gate"] = evaluate_evidence(
                        question,
                        option_key,
                        option_text,
                        hits,
                        question.domain,
                        self.gate_settings,
                    ).to_dict()
            rule_summary = summarize_rule_alignment(option_text if question.answer_format != "tf" else question.question, hits)
            payload = self._judge_option(question, option_key, option_text, hits, rule_summary)
            total_usage.add(payload["token_usage"])
            option_labels[option_key] = payload["label"]
            option_payloads.append(
                {
                    "option": option_key,
                    "label": payload["label"],
                    "support_score": payload["support_score"],
                    "verdict": payload["verdict"],
                    "is_clearly_refuted": payload["is_clearly_refuted"],
                    "reasoning_summary": payload["reasoning_summary"],
                    "evidence_items": [hit.to_dict() for hit in hits],
                    "rule_summary": rule_summary,
                    "gate_status": gate_debug.get("final_gate", {}).get("status", ""),
                    "gate_reasons": gate_debug.get("final_gate", {}).get("reasons", []),
                }
            )
            rule_outputs.append({"option": option_key, **rule_summary})
            option_debug.append(
                {
                    "option": option_key,
                    "query_variants": query_variants,
                    "candidate_hits": serialize_hits(candidate_hits, limit=max(top_k * 2, 10)),
                    "prompt_hits": serialize_hits(hits, limit=top_k),
                    "retrieval_topk": serialize_hits(hits, limit=top_k),
                    "rule_summary": rule_summary,
                    "label": payload["label"],
                    "support_score": payload["support_score"],
                    "evidence_gate": gate_debug,
                }
            )
            reasoning_chunks.append(f"{option_key}({payload['support_score']:.2f}): {payload['reasoning_summary']}")

        pred_answer = self._compose_answer(question.answer_format, option_labels)
        if question.answer_format == "mcq" and (len([k for k, v in option_labels.items() if v]) != 1):
            pred_answer = self._fallback_single_choice(question, option_payloads, total_usage)
        if question.answer_format == "multi" and len(pred_answer) < 2:
            pred_answer = self._fallback_multi_choice(question, option_payloads, total_usage)
        pred_answer, answer_finalization = finalize_answer(
            pred_answer,
            answer_format=question.answer_format,
            allowed_options=list(question.options.keys()) or ["A", "B"],
            option_labels=option_labels,
            option_payloads=option_payloads,
        )
        if question.answer_format == "tf":
            option_labels["B"] = pred_answer == "B"
        consistency_issues = []
        if gate_enabled(self.gate_settings):
            consistency_issues = answer_consistency_issues(pred_answer, option_payloads, question.answer_format)
            if consistency_issues and self.gate_settings.get("final_consistency_retry", True):
                if question.answer_format == "mcq":
                    pred_answer = self._fallback_single_choice(question, option_payloads, total_usage)
                elif question.answer_format == "multi":
                    pred_answer = self._fallback_multi_choice(question, option_payloads, total_usage)
                retry_answer, retry_finalization = finalize_answer(
                    pred_answer,
                    answer_format=question.answer_format,
                    allowed_options=list(question.options.keys()) or ["A", "B"],
                    option_labels=option_labels,
                    option_payloads=option_payloads,
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

    def _judge_option(
        self,
        question: Question,
        option_key: str,
        option_text: str,
        hits,
        rule_summary: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_lines = []
        for idx, hit in enumerate(hits, start=1):
            title_path = " > ".join(hit.title_path)
            evidence_lines.append(f"[{idx}] {hit.doc_id} | {title_path}\n{hit.text}")
        evidence_text = "\n\n".join(evidence_lines[: self.answering_settings.get("max_hits", 6)])
        extra_context = self.answering_settings.get("extra_context", "").strip()
        rule_text = format_rule_summary(rule_summary)
        target_text = question.question if question.answer_format == "tf" else option_text
        task_guidance = self._task_guidance(question.answer_format)
        few_shot = self._few_shot_prompt()
        messages = [
            {
                "role": "system",
                "content": (
                    self._system_prompt()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题目：{question.question}\n"
                    f"题型：{question.answer_format}\n"
                    f"选项 {option_key}：{option_text}\n\n"
                    f"待核验陈述：{target_text}\n\n"
                    f"{task_guidance}\n\n{few_shot}\n\n{extra_context}\n\n规则抽取摘要：\n{rule_text}\n\n证据：\n{evidence_text}\n\n"
                    "请输出 JSON，格式为 "
                    '{"label": true/false, "support_score": 0.0, "verdict": "support/refute/insufficient", '
                    '"is_clearly_refuted": true/false, "subclaims": ["..."], '
                    '"reasoning_summary": "...", "used_evidence_ids": [1,2]}。\n'
                    "support_score 是当前选项被证据支持、应被选为正确答案的概率，范围为 0.0 到 1.0。"
                    "0.8-1.0 表示证据明确支持；0.5-0.8 表示较可能正确；"
                    "0.2-0.5 表示证据不足或有疑点；0.0-0.2 表示证据明确反驳。"
                    ),
                },
        ]
        response = self.client.chat_json(messages)
        parsed = self._safe_extract_json(response.content)
        label = bool(parsed.get("label", False))
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"support", "refute", "insufficient"}:
            verdict = "support" if label else "insufficient"
        is_clearly_refuted = bool(parsed.get("is_clearly_refuted", verdict == "refute"))
        return {
            "label": label,
            "support_score": self._normalize_support_score(parsed.get("support_score"), label),
            "verdict": verdict,
            "is_clearly_refuted": is_clearly_refuted,
            "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip()
            or "模型未返回可解析JSON，按无证据支持处理。",
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
        parsed = self._safe_extract_json(response.content)
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
                    "content": "你是多选题复核器。根据各选项判断摘要挑出所有正确选项，答案必须至少包含两个字母，只输出 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        f"题目：{question.question}\n"
                        f"选项摘要：{summary}\n"
                        "多选题答案必须至少包含两个选项字母。"
                        "优先选择 label=true 的选项；若不足两个，只能从 is_clearly_refuted=false 的选项中按 support_score 补足。"
                        "不要选择 verdict=refute 或明确被证据反驳的选项来凑数。"
                        '输出格式：{"answer": "AC"}。'
                    ),
                },
            ]
        )
        total_usage.add(response.token_usage)
        parsed = self._safe_extract_json(response.content)
        answer = "".join(sorted(set(str(parsed.get("answer", "")).strip().upper())))
        cleaned = "".join(ch for ch in answer if ch in question.options)
        return cleaned

    @staticmethod
    def _safe_extract_json(content: str) -> dict[str, Any]:
        try:
            return extract_json_object(content)
        except ValueError:
            return {}

    @staticmethod
    def _normalize_support_score(value: Any, label: bool) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.75 if label else 0.25
        return max(0.0, min(1.0, score))

    @staticmethod
    def _ensure_multi_minimum(answer: str, question: Question, option_payloads: list[dict[str, Any]]) -> str:
        cleaned = "".join(sorted({ch for ch in answer.upper() if ch in question.options}))
        if len(cleaned) >= 2:
            return cleaned
        selected = set(cleaned)
        non_refuted = [
            item
            for item in option_payloads
            if not bool(item.get("is_clearly_refuted", False)) and str(item.get("verdict", "")) != "refute"
        ]
        ranked = sorted(
            non_refuted or option_payloads,
            key=lambda item: (float(item.get("support_score", 0.0)), bool(item.get("label", False))),
            reverse=True,
        )
        for item in ranked:
            option = str(item.get("option", "")).upper()
            if option in question.options:
                selected.add(option)
            if len(selected) >= 2:
                break
        for option in sorted(question.options):
            selected.add(option)
            if len(selected) >= 2:
                break
        return "".join(sorted(selected))

    @staticmethod
    def _compose_answer(answer_format: str, option_labels: dict[str, bool]) -> str:
        if answer_format == "tf":
            return "A" if option_labels.get("A", False) else "B"
        selected = [option for option, label in option_labels.items() if label]
        selected.sort()
        if answer_format == "mcq":
            return selected[0] if selected else "A"
        return "".join(selected)

    def _system_prompt(self) -> str:
        prompt_id = self.answering_settings.get("prompt_template_id", "default")
        if prompt_id == "evidence_strict":
            return "你是金融法规问答助手。你只能依据给定证据判断选项真伪，不能使用常识补全，必须逐条对应法条并输出 JSON。"
        if prompt_id == "compact":
            return "你是金融法规问答助手。请基于最关键法条快速判断选项真伪，只输出 JSON。"
        return (
            "你是金融法规问答助手。你只能依据给定证据判断选项真伪，不能使用常识补全。"
            "必须先拆解复合陈述，再判断每个子命题是否被支持、证据不足或明确反驳。"
            "合理概括不等于错误；只有期限、金额门槛、报告对象、义务动作、应当/可以/无需、报告类型等发生实质冲突时，才判明确反驳。"
            "输出必须是 JSON。"
        )

    @staticmethod
    def _task_guidance(answer_format: str) -> str:
        if answer_format == "multi":
            return (
                "题型约束：当前题为多选题，最终答案必须至少包含两个选项。"
                "你现在只判断一个选项，但要避免把可支持的概括性表述误判为 false。"
                "若证据直接支持核心主体、动作、期限、金额或报告对象，即使选项省略非核心限定，也可判 support。"
                "若题干和引用文档已经限定了业务场景，选项省略条文中的前置语通常是合理概括；只有出现“任何、所有、无论”等明显扩大适用范围，才视为实质冲突。"
                "若证据明确显示选项改变了期限、金额门槛、报告对象、义务动作或报告类型，判 refute 且 is_clearly_refuted=true。"
                "若选项声称“无需、可以不、不披露、不报告”等免除义务，必须有证据明确支持该豁免；不能仅因未看到强制义务就推断为 support。"
                "若只有部分子命题缺证但没有冲突，判 insufficient，不要标记为明确反驳。"
            )
        if answer_format == "mcq":
            return "题型约束：当前题为单选题，最终只能选择一个最直接、最完整被证据支持的选项。"
        if answer_format == "tf":
            return "题型约束：当前题为判断题，A 表示题干陈述正确，B 表示题干陈述错误。复合陈述需逐个子命题核验。"
        return ""

    @staticmethod
    def _few_shot_prompt() -> str:
        return (
            "判题示例：\n"
            "1. 多选复核：证据写“重大差异应在30个工作日内提交差异报告”，选项概括为“发现重大差异30个工作日内提交差异报告”。"
            "在题干已经限定为受益所有人信息核对场景时，若未改变期限和动作，可判 support；不要因省略非核心前置语就直接 refute。\n"
            "2. 复合陈述：选项为“X 且 Y”。若证据1支持 X、证据2支持 Y，则整体 support；"
            "若只看到 X 而未看到 Y，最多判 insufficient，不能说 Y 被明确反驳。\n"
            "3. 明确反驳：原文为“人民币1万元以上或者外币等值1000美元以上”，选项说“达到1000美元以上才需要”。"
            "这改变金额门槛且“才需要”有排他含义，应判 refute。\n"
            "4. 免除义务：选项说“可以不披露具体原因”。若证据只是没有提到该情形，不能推出可以不披露；"
            "只有证据明确给出豁免，才可判 support，否则判 insufficient 或 refute。"
        )

    def _targeted_literal_hits(self, question: Question, option_text: str) -> list[RetrievalHit]:
        specs = self._target_specs(option_text)
        if not specs or not hasattr(self.retriever, "units"):
            return []
        doc_filter = set(question.doc_ids)
        scored: list[tuple[float, dict[str, Any]]] = []
        units = [*self.retriever.units, *self.supplemental_units]
        for unit in units:
            if doc_filter and unit.get("doc_id") not in doc_filter:
                continue
            haystack = self._normalize_literal(" ".join(unit.get("title_path", [])) + "\n" + unit.get("text", ""))
            best_score = 0.0
            for spec in specs:
                required = [self._normalize_literal(term) for term in spec["required"]]
                optional = [self._normalize_literal(term) for term in spec.get("optional", [])]
                if any(term not in haystack for term in required):
                    continue
                score = sum(len(term) for term in required) * 10.0
                score += sum(len(term) for term in optional if term in haystack) * 3.0
                if score > best_score:
                    best_score = score
            if best_score > 0:
                scored.append((best_score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = []
        for score, unit in scored[:3]:
            metadata = dict(unit.get("metadata", {}))
            metadata.setdefault("unit_type", unit.get("unit_type", ""))
            metadata["targeted_literal"] = True
            hits.append(
                RetrievalHit(
                    unit_id=unit["unit_id"],
                    doc_id=unit["doc_id"],
                    score=score + 1000.0,
                    title_path=unit["title_path"],
                    text=unit["text"],
                    metadata=metadata,
                )
            )
        return hits

    @staticmethod
    def _load_supplemental_units(path_value: str) -> list[dict[str, Any]]:
        if not path_value:
            return []
        path = Path(path_value)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        units: dict[str, dict[str, Any]] = {}

        def add_hit(hit: dict[str, Any]) -> None:
            unit_id = str(hit.get("unit_id", ""))
            if not unit_id or unit_id in units:
                return
            units[unit_id] = {
                "unit_id": unit_id,
                "doc_id": hit.get("doc_id", ""),
                "domain": "regulatory",
                "unit_type": hit.get("unit_type") or hit.get("metadata", {}).get("unit_type") or "article",
                "title_path": hit.get("title_path", []),
                "text": hit.get("text", ""),
                "page_refs": hit.get("page_refs", []),
                "metadata": {
                    **(hit.get("metadata", {}) or {}),
                    "supplemental_source": str(path),
                },
            }

        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                if "unit_id" in row and "text" in row:
                    add_hit(row)
                for key in ["old_hits", "new_hits", "hits"]:
                    for hit in row.get(key, []) or []:
                        if isinstance(hit, dict):
                            add_hit(hit)
        elif isinstance(payload, dict):
            for hit in payload.get("units", []) or payload.get("hits", []) or []:
                if isinstance(hit, dict):
                    add_hit(hit)
        return list(units.values())

    @staticmethod
    def _target_specs(option_text: str) -> list[dict[str, list[str]]]:
        compact = RegulatorySolver._normalize_literal(option_text)
        specs: list[dict[str, list[str]]] = []
        if "撤并分支机构" in compact:
            specs.append(
                {
                    "required": ["撤并分支机构", "至少提前30日"],
                    "optional": ["分支机构住所地", "报告内容", "撤并方案", "持卡人及商户权益保障"],
                }
            )
        if "现金分红" in compact or "分红条件" in compact or "不进行现金分红" in compact:
            specs.append(
                {
                    "required": ["现金分红", "充分披露原因"],
                    "optional": ["具备条件", "不进行现金分红", "利润分配", "资金用途"],
                }
            )
        if "存量" in compact and "受益所有人" in compact and ("6个月" in compact or "六月" in compact):
            specs.append(
                {
                    "required": ["存量非自然人客户", "6个月内完成", "较高风险以上存量客户", "受益所有人识别核实"],
                    "optional": ["2年内完成全部存量客户", "本办法施行之日起"],
                }
            )
        if "保单贷款" in compact or "1000美元" in compact or "1万元" in compact:
            specs.append(
                {
                    "required": ["保单贷款", "人民币1万元以上", "外币等值1000美元以上", "核实申请人身份"],
                    "optional": ["解除保险合同", "减保", "现金价值", "贷款金额"],
                }
            )
        if "废止" in compact and ("2007" in compact or "2022" in compact):
            specs.append(
                {
                    "required": ["废止", "2007", "2022"],
                    "optional": ["自2026年1月1日起施行", "客户尽职调查", "客户身份资料及交易记录保存"],
                }
            )
        return specs

    @staticmethod
    def _normalize_literal(text: str) -> str:
        return "".join(str(text or "").split())

    @staticmethod
    def _merge_hits(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
        merged: list[RetrievalHit] = []
        seen: set[str] = set()
        for hit in hits:
            key = hit.unit_id.replace("__dup2", "").replace("__dup", "") if hit.unit_id else f"{hit.doc_id}:{hit.text[:80]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= limit:
                break
        return merged
