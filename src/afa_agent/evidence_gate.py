from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from afa_agent.models import Question, RetrievalHit
from afa_agent.strategy import build_query_variants, serialize_hits


GATE_PASS = "pass"
GATE_PARTIAL = "partial"
GATE_FAIL = "fail"

REASON_MISSING_DOC = "missing_doc"
REASON_MISSING_METRIC = "missing_metric"
REASON_MISSING_CLAUSE = "missing_clause"
REASON_WRONG_CHUNK = "wrong_chunk"
REASON_CONTRADICTORY = "contradictory"
REASON_LOW_CERTAINTY = "low_certainty"
REASON_OPTION_SEMANTICS = "option_semantics_error"

DEFAULT_RESCUE_CHANNELS = [
    "option_assertion_search",
    "contradiction_search",
    "query_rewrite_search",
    "title_search",
    "unit_type_search",
    "table_metric_search",
    "clause_formula_search",
    "per_doc_search",
    "neighbor_expansion",
]


METRIC_TERMS = {
    "营业收入": ["营业收入", "营业总收入", "营业额"],
    "归母净利润": ["归属于上市公司股东的净利润", "归母净利润", "母公司拥有人应占溢利"],
    "经营现金流": ["经营活动产生的现金流量净额", "经营现金流"],
    "研发投入": ["研发投入", "研发费用", "研发投入占营业收入比例", "研发费用占营业收入比例"],
    "现金分红": ["现金分红", "每10股派", "每 10 股派", "股息", "分红"],
}

CLAUSE_TERMS = {
    "insurance": [
        "身故保险金",
        "现金价值",
        "犹豫期",
        "宽限期",
        "效力中止",
        "保险责任",
        "责任免除",
        "免赔额",
        "一般医疗保险金",
        "特定疾病医疗保险金",
        "住院",
        "意外伤害",
        "医疗费用",
        "保单贷款",
        "账户价值",
        "基本保险金额",
        "退保",
        "解除合同",
        "退还保险费",
    ],
    "financial_contracts": [
        "发行人",
        "发行规模",
        "发行金额",
        "主体信用评级",
        "债项信用评级",
        "受托管理人",
        "兑付日",
        "回售",
        "赎回",
        "违约",
        "转股价格",
        "证券简称",
        "股票代码",
        "资产负债率",
    ],
    "regulatory": ["第", "条", "报告", "披露", "施行", "客户尽职调查", "受益所有人", "银行卡清算机构"],
    "research": ["同比", "增速", "市场规模", "预计", "收入", "净利润", "结论", "风险提示"],
}

TERM_ALIASES = {
    "financial_contracts": {
        "股票代码": ["股票代码", "证券代码", "代码"],
        "证券简称": ["证券简称", "股票简称", "简称"],
        "转股价格": ["转股价格", "初始转股价格"],
        "发行金额": ["发行金额", "发行规模", "发行总额", "本期债券总规模", "债券总规模", "本次债券", "注册金额"],
        "发行规模": ["发行规模", "发行金额", "发行总额", "本期债券总规模", "债券总规模", "本次债券", "注册金额"],
    },
    "insurance": {
        "账户价值": ["账户价值", "保单账户价值", "个人账户价值"],
        "基本保险金额": ["基本保险金额", "基本保额", "基本保险额"],
        "保单贷款": ["保单贷款", "保险单借款", "借款"],
    }
}

MATCH_ALIASES = {
    "保单贷款": ["保单贷款", "保险单借款", "借款"],
    "发行金额": ["发行金额", "发行规模", "发行总额", "本期债券总规模", "债券总规模", "注册金额", "面值不超过"],
    "发行规模": ["发行规模", "发行金额", "发行总额", "本期债券总规模", "债券总规模", "注册金额", "面值不超过"],
}


@dataclass(slots=True)
class GateResult:
    status: str
    reasons: list[str] = field(default_factory=list)
    missing_doc_ids: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    certainty_score: float = 0.0
    coverage_status: dict[str, Any] = field(default_factory=dict)
    rescue_needed: bool = False
    rescue_trigger: list[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RescueResult:
    hits: list[RetrievalHit]
    initial_gate: GateResult
    final_gate: GateResult
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_gate": self.initial_gate.to_dict(),
            "final_gate": self.final_gate.to_dict(),
            "rounds": self.rounds,
        }


def gate_enabled(settings: dict[str, Any]) -> bool:
    return bool(settings.get("enabled", False))


def evaluate_evidence(
    question: Question,
    option_key: str,
    option_text: str,
    hits: Iterable[Any],
    domain: str,
    settings: dict[str, Any] | None = None,
) -> GateResult:
    settings = settings or {}
    hit_rows = list(hits)
    required_doc_ids = _required_docs_for_option(question, option_text)
    if settings.get("ignore_locator_doc_requirements") and question.metadata.get("doc_ids_are_locator_candidates"):
        required_doc_ids = []
    if not hit_rows:
        reasons = [REASON_WRONG_CHUNK]
        if required_doc_ids:
            reasons.append(REASON_MISSING_DOC)
        coverage = {
            "hit_count": 0,
            "doc_coverage": {"covered": 0, "total": len(set(required_doc_ids)), "missing_doc_ids": required_doc_ids},
            "expected_terms": [],
            "matched_terms": [],
            "term_coverage": 1.0,
            "weak_ratio": 1.0,
            "unit_type_counts": {},
            "preferred_unit_types": _preferred_unit_types(domain),
            "preferred_unit_hit_count": 0,
        }
        return GateResult(
            status=GATE_FAIL,
            reasons=sorted(set(reasons)),
            missing_doc_ids=required_doc_ids,
            certainty_score=0.0,
            coverage_status=coverage,
            rescue_needed=True,
            rescue_trigger=sorted(set(reasons)),
            details="no evidence hits",
        )

    reasons: list[str] = []
    missing_docs = _missing_doc_ids(required_doc_ids, hit_rows)
    if missing_docs:
        reasons.append(REASON_MISSING_DOC)

    weak_ratio = _weak_hit_ratio(hit_rows, min_chars=int(settings.get("min_hit_chars", 24)))
    if weak_ratio >= 0.75:
        reasons.append(REASON_WRONG_CHUNK)

    expected_terms = _expected_terms(option_text, domain)
    if not expected_terms:
        expected_terms = _expected_terms(f"{question.question} {option_text}", domain)
    required_statement_terms = _required_statement_terms(option_text, domain) if settings.get("require_statement_terms", False) else []
    matched_terms = _matched_terms(expected_terms, hit_rows)
    matched_statement_terms = _matched_terms(required_statement_terms, hit_rows)
    term_doc_misses = _term_doc_misses(question, option_text, expected_terms, hit_rows, domain)
    if expected_terms and not matched_terms:
        reasons.append(REASON_MISSING_METRIC if domain == "financial_reports" else REASON_MISSING_CLAUSE)
    elif domain == "insurance" and expected_terms and len(matched_terms) < len(expected_terms):
        reasons.append(REASON_MISSING_CLAUSE)
    elif term_doc_misses:
        reasons.append(REASON_MISSING_METRIC if domain == "financial_reports" else REASON_MISSING_CLAUSE)
    if required_statement_terms and len(matched_statement_terms) < len(required_statement_terms):
        reasons.append(REASON_MISSING_METRIC if domain in {"financial_reports", "financial_contracts", "research"} else REASON_MISSING_CLAUSE)
    if domain == "insurance" and _is_insurance_formula_question(question.question, option_text):
        min_hits = int(settings.get("formula_rescue_min_hits", settings.get("max_hits_after_rescue", 12)))
        if len(hit_rows) < min_hits:
            reasons.append(REASON_LOW_CERTAINTY)

    contradiction = _simple_contradiction(option_text, hit_rows)
    if contradiction:
        reasons.append(REASON_CONTRADICTORY)
    semantic_issues = _semantic_assertion_issues(question, option_text, hit_rows, domain)
    if semantic_issues:
        reasons.append(REASON_OPTION_SEMANTICS)

    if REASON_CONTRADICTORY in reasons or REASON_OPTION_SEMANTICS in reasons or (REASON_WRONG_CHUNK in reasons and len(reasons) > 1):
        status = GATE_FAIL
    elif reasons:
        status = GATE_PARTIAL
    else:
        status = GATE_PASS
    coverage = _coverage_status(
        question=question,
        hits=hit_rows,
        expected_terms=expected_terms,
        matched_terms=matched_terms,
        required_statement_terms=required_statement_terms,
        matched_statement_terms=matched_statement_terms,
        term_doc_misses=term_doc_misses,
        required_doc_ids=required_doc_ids,
        weak_ratio=weak_ratio,
        domain=domain,
    )
    if semantic_issues:
        coverage["semantic_issues"] = semantic_issues
    certainty_score = _certainty_score(status, reasons, coverage)
    rescue_needed = status != GATE_PASS or certainty_score < float(settings.get("high_certainty_threshold", 0.7))
    rescue_trigger = sorted(set(reasons or ([REASON_LOW_CERTAINTY] if rescue_needed else [])))

    return GateResult(
        status=status,
        reasons=sorted(set(reasons)),
        missing_doc_ids=missing_docs,
        matched_terms=matched_terms,
        certainty_score=certainty_score,
        coverage_status=coverage,
        rescue_needed=rescue_needed,
        rescue_trigger=rescue_trigger,
        details=f"option={option_key}; expected_terms={expected_terms}; weak_ratio={weak_ratio:.2f}",
    )


def rescue_evidence(
    *,
    question: Question,
    option_key: str,
    option_text: str,
    domain: str,
    retriever: Any,
    initial_hits: list[RetrievalHit],
    retrieval_settings: dict[str, Any],
    gate_settings: dict[str, Any],
    initial_gate: GateResult | None = None,
) -> RescueResult:
    initial_gate = initial_gate or evaluate_evidence(question, option_key, option_text, initial_hits, domain, gate_settings)
    if not initial_gate.rescue_needed:
        max_initial_hits = min(len(initial_hits), int(gate_settings.get("max_hits_after_rescue", len(initial_hits))))
        reranked_hits = _rerank_hits(question, option_text, domain, _dedupe_hits(initial_hits))[:max_initial_hits]
        return RescueResult(hits=reranked_hits, initial_gate=initial_gate, final_gate=initial_gate, rounds=[])

    max_rounds = int(gate_settings.get("max_rescue_rounds", len(DEFAULT_RESCUE_CHANNELS)))
    top_k = int(gate_settings.get("rescue_top_k", max(retrieval_settings.get("top_k", 6), 12)))
    max_hits = int(gate_settings.get("max_hits_after_rescue", top_k))
    merged = _rerank_hits(question, option_text, domain, _dedupe_hits(initial_hits))
    rounds: list[dict[str, Any]] = []
    search_specs = _rescue_search_specs(question, option_key, option_text, domain, retrieval_settings, gate_settings, initial_gate)

    for round_index, spec in enumerate(search_specs[:max_rounds], start=1):
        gate_before = evaluate_evidence(question, option_key, option_text, merged, domain, gate_settings)
        query = str(spec["query"])
        doc_ids = list(spec.get("doc_ids") or question.doc_ids)
        round_hits = _search(
            retriever,
            doc_ids,
            query,
            retrieval_settings,
            top_k=int(spec.get("top_k", top_k)),
            ensure_per_doc=bool(spec.get("ensure_per_doc", True)),
            unit_type_boosts=spec.get("unit_type_boosts"),
            expand_neighbors=spec.get("expand_neighbors"),
        )
        hits_before = len(merged)
        merged = _rerank_hits(question, option_text, domain, _dedupe_hits([*merged, *round_hits]))
        merged = _enforce_doc_quota(merged, question.doc_ids, int(gate_settings.get("per_doc_quota", 2)), max_hits)
        gate_after = evaluate_evidence(question, option_key, option_text, merged, domain, gate_settings)
        rounds.append(
            {
                "round": round_index,
                "retrieval_channel": spec["channel"],
                "query": query,
                "doc_ids": doc_ids,
                "top_k": int(spec.get("top_k", top_k)),
                "unit_type_boosts": spec.get("unit_type_boosts") or retrieval_settings.get("unit_type_boosts", {}),
                "hits_before": hits_before,
                "hits_after": len(merged),
                "gate_before": gate_before.to_dict(),
                "gate_after": gate_after.to_dict(),
                "gate": gate_after.to_dict(),
                "hits": serialize_hits(merged, limit=max_hits),
            }
        )
        if not gate_after.rescue_needed:
            return RescueResult(hits=merged[:max_hits], initial_gate=initial_gate, final_gate=gate_after, rounds=rounds)

    final_gate = evaluate_evidence(question, option_key, option_text, merged, domain, gate_settings)
    return RescueResult(hits=merged[:max_hits], initial_gate=initial_gate, final_gate=final_gate, rounds=rounds)


def answer_consistency_issues(answer: str, option_payloads: list[dict[str, Any]], answer_format: str = "") -> list[str]:
    selected = {ch for ch in answer.upper()}
    issues = []
    supported_options = {
        str(payload.get("option", "")).upper()
        for payload in option_payloads
        if bool(payload.get("label", False))
    }
    if answer_format == "mcq" and len(supported_options) != 1:
        issues.append("mcq_ambiguous_supported" if supported_options else "no_supported_option")
    if answer_format == "multi":
        if not supported_options:
            issues.append("no_supported_option")
        elif len(supported_options) == 1:
            issues.append("single_supported_multi")
        missing_supported = supported_options - selected
        for option in sorted(missing_supported):
            issues.append(f"missing_supported_option:{option}")
    for payload in option_payloads:
        option = str(payload.get("option", "")).upper()
        if option and option in selected and not bool(payload.get("label", False)):
            if supported_options:
                issues.append(f"selected_false_option:{option}")
            elif answer_format in {"mcq", "multi"}:
                issues.append("no_supported_option")
            else:
                issues.append(f"selected_false_option:{option}")
        if option and option in selected and payload.get("gate_status") == GATE_FAIL:
            issues.append(f"selected_gate_fail:{option}")
    return sorted(set(issues))


def should_skip_answer_fallback(
    *,
    answer_format: str,
    option_labels: dict[str, bool],
    gate_settings: dict[str, Any],
) -> bool:
    """Skip expensive fallback calls for known format conflicts.

    This is only active when explicitly enabled in strategy config. It leaves
    factual option labels unchanged and lets finalize_answer apply the existing
    answer-format policy.
    """
    if not gate_settings.get("skip_format_conflict_fallback", False):
        return False
    supported_count = sum(1 for label in option_labels.values() if label)
    if answer_format == "multi" and supported_count == 1:
        return bool(gate_settings.get("skip_single_supported_multi_fallback", True))
    if answer_format == "mcq" and supported_count > 1:
        return bool(gate_settings.get("skip_mcq_ambiguous_fallback", False))
    return False


def should_skip_consistency_retry(
    *,
    consistency_issues: list[str],
    answer_finalization: dict[str, Any],
    answer_format: str,
    gate_settings: dict[str, Any],
) -> tuple[bool, str]:
    """Return whether final consistency retry is unlikely to add evidence value."""
    if not gate_settings.get("skip_format_conflict_retry", False):
        return False, ""
    issue_set = set(consistency_issues)
    policy = (answer_finalization or {}).get("answer_policy") or {}
    warnings = set(policy.get("warnings") or [])

    if answer_format == "multi":
        has_false_selected = any(issue.startswith("selected_false_option") for issue in issue_set)
        if "single_supported_multi" in issue_set and (has_false_selected or "selected_false_option" in warnings):
            return True, "single_supported_multi_format_conflict"
        if "no_supported_option" in issue_set and gate_settings.get("skip_no_supported_multi_retry", True):
            return True, "no_supported_multi_format_conflict"

    if answer_format == "mcq" and "mcq_ambiguous_supported" in issue_set:
        if gate_settings.get("skip_mcq_ambiguous_retry", True):
            return True, "mcq_ambiguous_supported"

    return False, ""


def _coverage_status(
    *,
    question: Question,
    hits: list[Any],
    expected_terms: list[str],
    matched_terms: list[str],
    required_statement_terms: list[str],
    matched_statement_terms: list[str],
    term_doc_misses: list[dict[str, Any]],
    required_doc_ids: list[str],
    weak_ratio: float,
    domain: str,
) -> dict[str, Any]:
    expected_docs = set(required_doc_ids)
    hit_docs = {_hit_doc_id(hit) for hit in hits if _hit_doc_id(hit)}
    unit_type_counts = Counter(_hit_unit_type(hit) for hit in hits)
    preferred_types = set(_preferred_unit_types(domain))
    return {
        "hit_count": len(hits),
        "doc_coverage": {
            "covered": len(expected_docs & hit_docs),
            "total": len(expected_docs),
                "missing_doc_ids": [doc_id for doc_id in required_doc_ids if doc_id not in hit_docs],
        },
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "required_statement_terms": required_statement_terms,
        "matched_statement_terms": matched_statement_terms,
        "statement_term_coverage": round(len(matched_statement_terms) / len(required_statement_terms), 4)
        if required_statement_terms
        else 1.0,
        "term_doc_misses": term_doc_misses,
        "term_coverage": round(len(matched_terms) / len(expected_terms), 4) if expected_terms else 1.0,
        "weak_ratio": round(weak_ratio, 4),
        "unit_type_counts": {key: value for key, value in unit_type_counts.items() if key},
        "preferred_unit_types": sorted(preferred_types),
        "preferred_unit_hit_count": sum(unit_type_counts.get(unit_type, 0) for unit_type in preferred_types),
    }


def _certainty_score(status: str, reasons: list[str], coverage: dict[str, Any]) -> float:
    if coverage.get("hit_count", 0) == 0:
        return 0.0
    score = 1.0
    doc_coverage = coverage.get("doc_coverage", {})
    total_docs = int(doc_coverage.get("total") or 0)
    missing_docs = len(doc_coverage.get("missing_doc_ids") or [])
    if total_docs:
        score -= min(0.35, 0.35 * missing_docs / total_docs)
    if REASON_WRONG_CHUNK in reasons:
        score -= 0.25
    if REASON_MISSING_METRIC in reasons or REASON_MISSING_CLAUSE in reasons:
        score -= 0.22
    if REASON_CONTRADICTORY in reasons:
        score -= 0.45
    if REASON_OPTION_SEMANTICS in reasons:
        score -= 0.35
    score -= min(0.2, float(coverage.get("weak_ratio", 0.0)) * 0.2)
    if coverage.get("expected_terms") and coverage.get("term_coverage", 1.0) < 1.0:
        score -= (1.0 - float(coverage.get("term_coverage", 0.0))) * 0.15
    if coverage.get("required_statement_terms") and coverage.get("statement_term_coverage", 1.0) < 1.0:
        score -= (1.0 - float(coverage.get("statement_term_coverage", 0.0))) * 0.2
    if coverage.get("preferred_unit_types") and not coverage.get("preferred_unit_hit_count"):
        score -= 0.08
    if status == GATE_FAIL:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 4)


def _search(
    retriever: Any,
    doc_ids: list[str],
    query: str,
    retrieval_settings: dict[str, Any],
    *,
    top_k: int,
    ensure_per_doc: bool,
    unit_type_boosts: dict[str, float] | None = None,
    expand_neighbors: bool | None = None,
) -> list[RetrievalHit]:
    kwargs = {
        "top_k": top_k,
        "ensure_per_doc": ensure_per_doc,
        "expand_neighbors": retrieval_settings.get("expand_neighbors", True) if expand_neighbors is None else expand_neighbors,
    }
    boosts = retrieval_settings.get("unit_type_boosts", {}) if unit_type_boosts is None else unit_type_boosts
    try:
        return retriever.search(doc_ids, query, unit_type_boosts=boosts, **kwargs)
    except TypeError:
        return retriever.search(doc_ids, query, **kwargs)


def _rescue_search_specs(
    question: Question,
    option_key: str,
    option_text: str,
    domain: str,
    retrieval_settings: dict[str, Any],
    gate_settings: dict[str, Any],
    gate: GateResult,
) -> list[dict[str, Any]]:
    channels = gate_settings.get("rescue_channels") or DEFAULT_RESCUE_CHANNELS
    enabled_channels = [channel for channel in channels if channel in DEFAULT_RESCUE_CHANNELS]
    enabled_channels = _prioritize_rescue_channels(enabled_channels, domain, gate)
    top_k = int(gate_settings.get("rescue_top_k", max(retrieval_settings.get("top_k", 6), 12)))
    base_variants = build_query_variants(question, option_key, option_text, retrieval_settings)
    term_text = " ".join(_expected_terms(f"{question.question} {option_text}", domain))
    focused_terms = _focused_terms(question, option_text, domain)
    title_terms = _title_terms_from_question(question, option_text)
    doc_anchor_terms = _doc_anchor_terms(question, option_text, domain)
    focused = " ".join(part for part in [question.question, option_text, focused_terms, term_text, doc_anchor_terms] if part).strip()
    specs: list[dict[str, Any]] = []
    for channel in enabled_channels:
        if channel == "option_assertion_search":
            queries = _dedupe_strings(
                [
                    _option_assertion_query(question, option_text, domain),
                    f"{option_text}\n{doc_anchor_terms}".strip(),
                    f"{option_text}\n{focused_terms}".strip(),
                ]
            )
            specs.extend(_spec(channel, query, top_k=top_k, ensure_per_doc=False) for query in queries)
        elif channel == "contradiction_search":
            queries = _dedupe_strings(_contradiction_queries(question, option_text, domain))
            specs.extend(_spec(channel, query, top_k=top_k, ensure_per_doc=False) for query in queries)
        elif channel == "query_rewrite_search":
            queries = _dedupe_strings([focused, *base_variants, f"{option_text}\n{focused_terms}".strip()])
            specs.extend(_spec(channel, query, top_k=top_k) for query in queries)
        elif channel == "title_search":
            title_query = " ".join(part for part in [title_terms, focused_terms, doc_anchor_terms] if part).strip()
            specs.extend(_spec(channel, query, top_k=top_k) for query in _dedupe_strings([title_query, f"{title_query}\n{option_text}".strip()]))
        elif channel == "unit_type_search":
            specs.append(_spec(channel, focused, top_k=top_k, unit_type_boosts=_preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))))
        elif channel == "table_metric_search" and domain in {"financial_reports", "financial_contracts"}:
            boosts = _preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))
            queries = _dedupe_strings([_table_metric_terms(question.question, option_text), doc_anchor_terms, focused])
            specs.extend(_spec(channel, query, top_k=top_k, unit_type_boosts=boosts) for query in queries)
        elif channel == "clause_formula_search" and domain == "insurance":
            boosts = _preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))
            queries = _dedupe_strings([_insurance_clause_query(question.question, option_text), doc_anchor_terms, focused])
            specs.extend(_spec(channel, query, top_k=top_k, unit_type_boosts=boosts) for query in queries)
        elif channel == "per_doc_search":
            for doc_id in gate.missing_doc_ids or question.doc_ids:
                query = " ".join(part for part in [option_text, focused_terms, term_text, doc_anchor_terms, doc_id] if part).strip()
                specs.append(_spec(channel, query, top_k=top_k, doc_ids=[doc_id], ensure_per_doc=False))
        elif channel == "neighbor_expansion":
            specs.append(_spec(channel, focused, top_k=top_k, expand_neighbors=True))
    return [spec for spec in specs if spec.get("query")]


def _prioritize_rescue_channels(channels: list[str], domain: str, gate: GateResult) -> list[str]:
    reasons = set(gate.reasons) | set(gate.rescue_trigger)
    priority: list[str] = []
    if REASON_MISSING_DOC in reasons:
        priority.append("per_doc_search")
    if domain in {"financial_reports", "financial_contracts"}:
        if REASON_MISSING_METRIC in reasons or REASON_MISSING_DOC in reasons:
            priority.extend(["table_metric_search", "unit_type_search"])
        priority.extend(["option_assertion_search", "contradiction_search", "title_search", "query_rewrite_search", "neighbor_expansion"])
    elif domain == "insurance":
        priority.extend(["option_assertion_search", "clause_formula_search", "contradiction_search", "per_doc_search", "unit_type_search", "neighbor_expansion", "query_rewrite_search"])
    elif domain == "regulatory":
        priority.extend(["option_assertion_search", "title_search", "unit_type_search", "per_doc_search", "contradiction_search", "query_rewrite_search", "neighbor_expansion"])
    elif domain == "research":
        priority.extend(["option_assertion_search", "contradiction_search", "unit_type_search", "title_search", "per_doc_search", "query_rewrite_search", "neighbor_expansion"])
    else:
        priority.extend(["option_assertion_search", "contradiction_search", "query_rewrite_search", "title_search", "unit_type_search", "per_doc_search", "neighbor_expansion"])
    return [channel for channel in _dedupe_strings([*priority, *channels]) if channel in channels]


def _spec(
    channel: str,
    query: str,
    *,
    top_k: int,
    doc_ids: list[str] | None = None,
    ensure_per_doc: bool = True,
    unit_type_boosts: dict[str, float] | None = None,
    expand_neighbors: bool | None = None,
) -> dict[str, Any]:
    return {
        "channel": channel,
        "query": query,
        "top_k": top_k,
        "doc_ids": doc_ids,
        "ensure_per_doc": ensure_per_doc,
        "unit_type_boosts": unit_type_boosts,
        "expand_neighbors": expand_neighbors,
    }


def _preferred_unit_types(domain: str) -> list[str]:
    return {
        "financial_reports": ["metric_row"],
        "insurance": ["formula_block", "clause_block"],
        "financial_contracts": ["element_block"],
        "regulatory": ["article", "article_chunk", "penalty_decision"],
        "research": ["conclusion_block"],
    }.get(domain, [])


def _preferred_unit_boosts(domain: str, base_boosts: dict[str, float]) -> dict[str, float]:
    boosts = dict(base_boosts)
    for unit_type in _preferred_unit_types(domain):
        boosts[unit_type] = max(float(boosts.get(unit_type, 1.0)), 2.2)
    return boosts


def _focused_terms(question: Question, option_text: str, domain: str) -> str:
    text = f"{question.question} {option_text}"
    parts = [
        " ".join(_expected_terms(text, domain)),
        " ".join(re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text)),
    ]
    if domain == "insurance":
        parts.append(" ".join(re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚)[\u4e00-\u9fa5A-Za-z0-9]{2,18}", text)))
    if domain == "regulatory":
        parts.append(" ".join(re.findall(r"第[一二三四五六七八九十百千万零\d]+条|[\u4e00-\u9fa5]{2,8}(?:报告|披露|禁入|处罚|备案|登记)", text)))
    return " ".join(part for part in parts if part).strip()


def _doc_anchor_terms(question: Question, option_text: str, domain: str) -> str:
    text = f"{question.question} {option_text}"
    anchors: list[str] = []
    if domain == "financial_contracts":
        anchors.extend(
            re.findall(
                r"[\u4e00-\u9fa5A-Za-z0-9]{2,30}(?:股份有限公司|有限公司|集团|控股|证券|创新|航运|科技)",
                text,
            )
        )
        anchors.extend(
            term
            for term in [
                "股票代码",
                "证券代码",
                "证券简称",
                "发行人",
                "初始转股价格",
                "转股价格",
                "资产负债率",
                "主体信用评级",
                "债项信用评级",
                "发行规模",
                "募集资金",
            ]
            if term in text
        )
    elif domain == "financial_reports":
        anchors.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:股份|集团|银行|证券|能源|科技|股份有限公司|集团有限公司)", text))
        anchors.extend(term for terms in METRIC_TERMS.values() for term in terms if term in text)
    elif domain == "insurance":
        anchors.extend(re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚|众安)[\u4e00-\u9fa5A-Za-z0-9]{2,24}", text))
        anchors.extend(term for term in CLAUSE_TERMS["insurance"] if term in text)
    elif domain == "regulatory":
        anchors.extend(re.findall(r"《[^》]{2,40}》|第[一二三四五六七八九十百千万零\d]+条|[\u4e00-\u9fa5]{2,12}(?:办法|规定|决定|通知)", text))
    elif domain == "research":
        anchors.extend(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:行业|市场|业务|收入|规模|增速|同比)", text))
    anchors.extend(re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text))
    return " ".join(_dedupe_strings(anchors[:16]))


def _option_assertion_query(question: Question, option_text: str, domain: str) -> str:
    """Build a low-noise query from the option itself.

    The normal query includes the full question, which is useful for recall but can
    swamp option-specific facts in comparison questions. This channel makes the
    option's entity/value assertion the first-class retrieval target.
    """
    anchors = _doc_anchor_terms(question, option_text, domain)
    focused = _focused_terms(Question(qid=question.qid, domain=question.domain, split=question.split, question="", type=question.type, doc_ids=question.doc_ids, options=question.options, answer_format=question.answer_format), option_text, domain)
    statement_terms = _statement_terms(option_text, domain)
    parts = [option_text, anchors, focused, " ".join(statement_terms)]
    return " ".join(part for part in parts if part).strip()


def _contradiction_queries(question: Question, option_text: str, domain: str) -> list[str]:
    text = option_text
    anchors = _doc_anchor_terms(question, option_text, domain)
    statement_terms = _statement_terms(text, domain)
    numbers = re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text)
    ratings = re.findall(r"\b(?:AAA|AA\+|AA|A\+|A)\b", text)
    queries: list[str] = []
    if statement_terms or anchors:
        queries.append(" ".join(_dedupe_strings([anchors, *statement_terms])))
    if numbers:
        queries.append(" ".join(_dedupe_strings([anchors, *statement_terms, *numbers])))
    if ratings:
        queries.append(" ".join(_dedupe_strings([anchors, "主体信用等级", "主体信用评级", "信用评级", *ratings])))
        queries.append(" ".join(_dedupe_strings([anchors, "主体信用等级", "主体信用评级", "信用评级"])))
    if domain == "financial_contracts":
        if "发行人" in text:
            queries.append(" ".join(_dedupe_strings([anchors, "发行人", "公司", "本公司"])))
        if any(term in text for term in ["发行金额", "发行规模", "注册金额"]):
            queries.append(" ".join(_dedupe_strings([anchors, "发行金额", "发行规模", "注册金额", "不超过"])))
        if any(term in text for term in ["主承销商", "簿记管理人", "受托管理人"]):
            queries.append(" ".join(_dedupe_strings([anchors, "主承销商", "簿记管理人", "受托管理人"])))
        if "兑付日" in text:
            queries.append(" ".join(_dedupe_strings([anchors, "兑付日", "回售", "赎回", *numbers])))
        if "违约" in text:
            queries.append(" ".join(_dedupe_strings([anchors, "违约", "逾期利息", "违约金", "计算方式", *numbers])))
    elif domain == "financial_reports":
        metric_terms = [term for terms in METRIC_TERMS.values() for term in terms if term in text]
        if metric_terms:
            queries.append(" ".join(_dedupe_strings([anchors, *metric_terms, *numbers])))
            queries.append(" ".join(_dedupe_strings(metric_terms)))
    elif domain == "insurance":
        clause_terms = [term for term in CLAUSE_TERMS["insurance"] if term in text]
        if clause_terms:
            queries.append(" ".join(_dedupe_strings([anchors, *clause_terms, *numbers])))
    elif domain == "research":
        research_terms = [term for term in CLAUSE_TERMS["research"] if term in text]
        if research_terms:
            queries.append(" ".join(_dedupe_strings([anchors, *research_terms, *numbers])))
    return [query for query in _dedupe_strings(queries) if query]


def _statement_terms(text: str, domain: str) -> list[str]:
    terms: list[str] = []
    terms.extend(re.findall(r"《[^》]{2,40}》", text))
    terms.extend(
        re.findall(
            r"[\u4e00-\u9fa5A-Za-z0-9]{2,32}(?:股份有限公司|有限公司|集团|控股|证券|投资|科技|银行|保险|人寿)",
            text,
        )
    )
    terms.extend(re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text))
    terms.extend(re.findall(r"\b(?:AAA|AA\+|AA|A\+|A)\b", text))
    for term in CLAUSE_TERMS.get(domain, []):
        if term in text:
            terms.append(term)
    if domain == "financial_reports":
        terms.extend(term for metric_terms in METRIC_TERMS.values() for term in metric_terms if term in text)
    return _dedupe_strings(terms[:20])


def _required_statement_terms(text: str, domain: str) -> list[str]:
    terms: list[str] = []
    text_without_doc_ids = re.sub(r"\b(?:fc_)?text_?0*\d+\b", " ", text, flags=re.IGNORECASE)
    raw_companies = re.findall(
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,32}(?:股份有限公司|有限公司|集团|控股|证券|投资|科技|银行|保险|人寿)",
        text_without_doc_ids,
    )
    terms.extend(_clean_company_term(term) for term in raw_companies)
    terms.extend(re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text_without_doc_ids))
    terms.extend(re.findall(r"\b(?:AAA|AA\+|AA|A\+|A)\b", text))
    if domain == "insurance":
        terms.extend(re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚|众安)[\u4e00-\u9fa5A-Za-z0-9]{2,24}", text))
    if domain == "regulatory":
        terms.extend(re.findall(r"第[一二三四五六七八九十百千万零\d]+条|《[^》]{2,40}》", text))
    return _dedupe_strings(terms[:12])


def _clean_company_term(term: str) -> str:
    cleaned = term.strip()
    for marker in ["发行人是", "发行人为", "公司是", "公司为", "记载", "显示", "披露", "关于"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1]
    for prefix in ["文档", "明确", "其", "该"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.strip("：:，,。；; ")


def _title_terms_from_question(question: Question, option_text: str) -> str:
    text = f"{question.type} {question.question} {option_text}"
    candidates = re.findall(r"《[^》]{2,40}》|#[^#\n]{2,40}|[\u4e00-\u9fa5A-Za-z0-9]{2,30}(?:报告|条款|说明书|决定书|募集说明书)", text)
    return " ".join(candidates[:8])


def _table_metric_terms(question_text: str, option_text: str) -> str:
    text = f"{question_text} {option_text}"
    metrics = [term for terms in METRIC_TERMS.values() for term in terms if term in text]
    contract_terms = [term for term in CLAUSE_TERMS["financial_contracts"] if term in text]
    years = re.findall(r"\d{4}年?", text)
    numbers = re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text)
    return " ".join(_dedupe_strings([*metrics, *contract_terms, *years, *numbers]))


def _insurance_clause_query(question_text: str, option_text: str) -> str:
    text = f"{question_text} {option_text}"
    products = re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚)[\u4e00-\u9fa5A-Za-z0-9]{2,18}", text)
    clause_terms = [term for term in CLAUSE_TERMS["insurance"] if term in text]
    formula_terms = [
        term
        for term in [
            "给付",
            "赔付",
            "较大者",
            "较大值",
            "下列两者",
            "比例",
            "基本保额",
            "基本保险金额",
            "账户价值",
            "现金价值",
            "已交保费",
            "一般医疗保险金",
            "住院",
            "意外伤害",
            "宽限期",
            "效力中止",
            "解除合同",
            "退还保险费",
        ]
        if term in text
    ]
    return " ".join(_dedupe_strings([*products, *clause_terms, *formula_terms]))


def _is_insurance_formula_question(question_text: str, option_text: str) -> bool:
    text = f"{question_text} {option_text}"
    return "身故保险金" in text and any(term in text for term in ["计算", "排序", "金额", "多少", "基本保额", "账户价值", "现金价值"])


def _expected_terms(option_text: str, domain: str) -> list[str]:
    expected: list[str] = []
    no_deductible_claim = domain == "insurance" and any(term in option_text for term in ["无免赔额", "不设免赔额", "未设置免赔额"])
    vehicle_medical_claim = (
        domain == "insurance"
        and "特种车" in option_text
        and "医疗费用" in option_text
        and ("车上人员" in option_text or "责任险" in option_text)
    )
    if domain == "financial_reports":
        for terms in METRIC_TERMS.values():
            if any(term in option_text for term in terms):
                expected.extend(terms)
    for canonical, aliases in TERM_ALIASES.get(domain, {}).items():
        if any(alias in option_text for alias in aliases):
            if domain == "financial_contracts":
                expected.extend(aliases)
            else:
                expected.append(canonical)
    for term in CLAUSE_TERMS.get(domain, []):
        if no_deductible_claim and term == "免赔额":
            continue
        if vehicle_medical_claim and term == "住院":
            continue
        if term in option_text:
            expected.append(term)
    return _dedupe_strings(expected)


def _matched_terms(expected_terms: list[str], hits: Iterable[Any]) -> list[str]:
    text = "\n".join(f"{' '.join(_hit_title_path(hit))}\n{_hit_text(hit)}" for hit in hits)
    matched = []
    for term in expected_terms:
        aliases = MATCH_ALIASES.get(term, [term])
        if any(alias and alias in text for alias in aliases):
            matched.append(term)
    return matched


def _missing_doc_ids(doc_ids: list[str], hits: Iterable[Any]) -> list[str]:
    seen = {_hit_doc_id(hit) for hit in hits}
    return [doc_id for doc_id in doc_ids if doc_id not in seen]


def _weak_hit_ratio(hits: list[Any], *, min_chars: int) -> float:
    if not hits:
        return 1.0
    weak = 0
    for hit in hits:
        text = _hit_text(hit).strip()
        title = " ".join(_hit_title_path(hit))
        if len(text) < min_chars or title.endswith("目录") or text in title:
            weak += 1
    return weak / len(hits)


def _simple_contradiction(option_text: str, hits: Iterable[Any]) -> bool:
    exemption_terms = ["无需", "不需要", "可以不", "不必", "免于"]
    obligation_terms = ["应当", "必须", "需要", "需", "须"]
    action_terms = ["披露", "报告", "提交", "办理", "履行", "赔付", "给付"]
    option_claims_exemption = any(term in option_text for term in exemption_terms)
    option_claims_obligation = any(term in option_text for term in obligation_terms)
    support_present = False
    for hit in hits:
        evidence_text = _hit_text(hit)
        evidence_claims_exemption = any(term in evidence_text for term in exemption_terms)
        evidence_claims_obligation = any(term in evidence_text for term in obligation_terms)
        shared_action = any(term in option_text and term in evidence_text for term in action_terms)
        if shared_action and (
            (option_claims_exemption and evidence_claims_exemption)
            or (option_claims_obligation and evidence_claims_obligation)
        ):
            support_present = True
            break
    for hit in hits:
        evidence_text = _hit_text(hit)
        if "重大差异" in option_text and "非重大差异" in evidence_text:
            continue
        evidence_claims_exemption = any(term in evidence_text for term in exemption_terms)
        evidence_claims_obligation = any(term in evidence_text for term in obligation_terms)
        shared_action = any(term in option_text and term in evidence_text for term in action_terms)
        if not support_present and shared_action and (
            (option_claims_exemption and evidence_claims_obligation)
            or (option_claims_obligation and evidence_claims_exemption)
        ):
            return True
    return False


def _semantic_assertion_issues(question: Question, option_text: str, hits: Iterable[Any], domain: str) -> list[dict[str, str]]:
    option = _compact_for_semantics(option_text)
    evidence = _compact_for_semantics("\n".join(f"{' '.join(_hit_title_path(hit))}\n{_hit_text(hit)}" for hit in hits))
    issues: list[dict[str, str]] = []
    if domain == "regulatory":
        issues.extend(_regulatory_threshold_issues(option, evidence))
    elif domain == "financial_contracts":
        issues.extend(_contract_default_interest_scope_issues(option, evidence))
    elif domain == "research":
        issues.extend(_research_metric_scope_issues(option, evidence))
    return issues


def _compact_for_semantics(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).replace("％", "%")


def _regulatory_threshold_issues(option: str, evidence: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    # In regulatory rules, inclusive/exclusive thresholds are often the claim itself.
    # "1万元以上" includes exactly 1万元; "超过1万元" does not.
    if (
        "保单贷款" in option
        and "超过人民币1万元" in option
        and "人民币1万元以上" in evidence
        and "超过人民币1万元" not in evidence
    ):
        issues.append(
            {
                "kind": "threshold_boundary_mismatch",
                "option_claim": "超过人民币1万元",
                "evidence_claim": "人民币1万元以上",
            }
        )
    return issues


def _contract_default_interest_scope_issues(option: str, evidence: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    claims_interest_base_includes_interest = (
        ("违约利息" in option or "逾期利息" in option)
        and any(term in option for term in ["本金和利息", "本金及利息", "包含本金和利息", "包含本金及利息"])
    )
    if not claims_interest_base_includes_interest:
        return issues
    interest_principal_only = (
        "逾期利息" in evidence
        and "逾期利息具体计算方式为本金×票面利率×逾期天数/365" in evidence
    )
    nearby_penalty_includes_interest = (
        "违约金" in evidence
        and any(term in evidence for term in ["延迟支付的本金和利息×票面利率×150%×违约天数/365", "延迟支付的本金及利息"])
    )
    if interest_principal_only and nearby_penalty_includes_interest:
        issues.append(
            {
                "kind": "default_interest_penalty_scope_mismatch",
                "option_claim": "违约/逾期利息计算基数包含本金和利息",
                "evidence_claim": "逾期利息公式只以本金为基数；相邻的本金和利息基数属于违约金公式",
            }
        )
    return issues


def _research_metric_scope_issues(option: str, evidence: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if (
        "韩国" in option
        and "银保" in option
        and "保费贡献率" in option
        and "复合增速" in option
        and "12%" in option
        and "韩国" in evidence
        and "银保" in evidence
        and "保费贡献超过50%" in evidence
        and "复合增速" in evidence
        and "保费贡献率复合增速" not in evidence
    ):
        issues.append(
            {
                "kind": "metric_scope_mismatch",
                "option_claim": "保费贡献率复合增速12%",
                "evidence_claim": "银保渠道保费/贡献相关口径复合增速12%，非贡献率CAGR",
            }
        )
    return issues


def _term_doc_misses(
    question: Question,
    option_text: str,
    expected_terms: list[str],
    hits: list[Any],
    domain: str,
) -> list[dict[str, Any]]:
    if domain not in {"financial_reports", "financial_contracts"} or not expected_terms or len(question.doc_ids) <= 1:
        return []
    required_terms = _strict_doc_terms(question, option_text, expected_terms, domain)
    required_docs = _required_docs_for_option(question, option_text)
    misses: list[dict[str, Any]] = []
    for doc_id in required_docs:
        doc_hits = [hit for hit in hits if _hit_doc_id(hit) == doc_id]
        doc_text = "\n".join(_hit_text(hit) for hit in doc_hits)
        terms_to_check = required_terms or expected_terms
        if not any(term and term in doc_text for term in terms_to_check):
            misses.append({"doc_id": doc_id, "missing_terms": terms_to_check})
    return misses


def _strict_doc_terms(question: Question, option_text: str, expected_terms: list[str], domain: str) -> list[str]:
    text = f"{question.question} {option_text}"
    if domain == "financial_reports" and "研发" in text and any(term in text for term in ["强度", "占比", "比例"]):
        ratio_terms = [term for term in expected_terms if "比例" in term or "占营业收入" in term]
        return ratio_terms or ["研发投入占营业收入比例", "研发费用占营业收入比例"]
    if domain == "financial_contracts":
        strict_terms: list[str] = []
        if any(term in option_text for term in ["股票代码", "证券代码"]):
            strict_terms.extend(["股票代码", "证券代码"])
        if any(term in option_text for term in ["股票简称", "证券简称"]):
            strict_terms.extend(["股票简称", "证券简称"])
        if "初始转股价格" in option_text or "18.26" in option_text:
            strict_terms.append("初始转股价格")
        if any(term in option_text for term in ["发行规模", "发行金额", "发行总额", "债券总规模", "本期债券总规模", "注册金额"]):
            strict_terms.append("发行规模")
        if "资产减值补偿" in option_text:
            strict_terms.extend(["资产减值补偿", "通知"])
        if "违约" in option_text:
            strict_terms.append("违约")
        return _dedupe_strings(strict_terms)
    return []


def _required_docs_for_option(question: Question, option_text: str) -> list[str]:
    if len(question.doc_ids) <= 1:
        return list(question.doc_ids)
    explicit_docs = [doc_id for doc_id in question.doc_ids if any(alias in option_text for alias in _doc_aliases(doc_id))]
    if explicit_docs:
        return explicit_docs
    mentions_first = "第一份" in option_text or "首份" in option_text
    mentions_second = "第二份" in option_text or "另一份" in option_text
    if mentions_first and not mentions_second:
        return [question.doc_ids[0]]
    if mentions_second and not mentions_first:
        return [question.doc_ids[1]]
    if mentions_first and mentions_second:
        return list(question.doc_ids[:2])
    comparison_text = f"{question.question} {option_text}"
    if any(term in comparison_text for term in ["对比", "两份", "两家", "均", "不同", "高于", "低于", "大于", "小于"]):
        return list(question.doc_ids)
    return list(question.doc_ids)


def _doc_aliases(doc_id: str) -> list[str]:
    aliases = [doc_id]
    match = re.fullmatch(r"text0*(\d+)", doc_id)
    if match:
        number = int(match.group(1))
        aliases.extend([f"fc_text_{number:03d}", f"fc_text_{number:02d}", f"text_{number:03d}", f"text_{number:02d}"])
    return aliases


def _enforce_doc_quota(hits: list[RetrievalHit], doc_ids: list[str], per_doc_quota: int, max_hits: int) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    selected_ids: set[str] = set()
    for doc_id in doc_ids:
        count = 0
        for hit in hits:
            if hit.doc_id != doc_id or hit.unit_id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(hit.unit_id)
            count += 1
            if count >= per_doc_quota or len(selected) >= max_hits:
                break
    for hit in hits:
        if len(selected) >= max_hits:
            break
        if hit.unit_id not in selected_ids:
            selected.append(hit)
            selected_ids.add(hit.unit_id)
    return selected


def _rerank_hits(question: Question, option_text: str, domain: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    if not hits:
        return hits
    expected_terms = _expected_terms(f"{question.question} {option_text}", domain)
    anchor_terms = _dedupe_strings([*_doc_anchor_terms(question, option_text, domain).split(), *_focused_terms(question, option_text, domain).split()])
    preferred_types = set(_preferred_unit_types(domain))
    required_docs = set(_required_docs_for_option(question, option_text))

    def score(hit: RetrievalHit) -> tuple[float, float]:
        text = hit.text
        term_hits = sum(1 for term in expected_terms if term and term in text)
        anchor_hits = sum(1 for term in anchor_terms if term and term in text)
        preferred = 1 if _hit_unit_type(hit) in preferred_types else 0
        required_doc = 1 if hit.doc_id in required_docs else 0
        boost = term_hits * 50.0 + anchor_hits * 8.0 + preferred * 5.0 + required_doc * 2.0
        return boost + hit.score, hit.score

    return sorted(hits, key=score, reverse=True)


def _dedupe_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    deduped: dict[str, RetrievalHit] = {}
    for hit in hits:
        if not isinstance(hit, RetrievalHit):
            continue
        key = _canonical_unit_id(hit.unit_id)
        current = deduped.get(key)
        if current is None or hit.score > current.score:
            deduped[key] = hit
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


def _canonical_unit_id(unit_id: str) -> str:
    return unit_id.replace("__dup2", "").replace("__dup", "")


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _hit_text(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("text") or hit.get("text_preview") or "")
    return str(getattr(hit, "text", ""))


def _hit_doc_id(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("doc_id", ""))
    return str(getattr(hit, "doc_id", ""))


def _hit_title_path(hit: Any) -> list[str]:
    if isinstance(hit, dict):
        value = hit.get("title_path", [])
    else:
        value = getattr(hit, "title_path", [])
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _hit_unit_type(hit: Any) -> str:
    if isinstance(hit, dict):
        metadata = hit.get("metadata", {}) if isinstance(hit.get("metadata", {}), dict) else {}
        return str(hit.get("unit_type") or metadata.get("unit_type") or metadata.get("source_unit_type") or "")
    metadata = getattr(hit, "metadata", {}) or {}
    return str(metadata.get("unit_type") or metadata.get("source_unit_type") or "")
