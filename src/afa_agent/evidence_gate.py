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

DEFAULT_RESCUE_CHANNELS = [
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
        "保险责任",
        "责任免除",
        "免赔额",
        "保单贷款",
        "账户价值",
        "基本保额",
        "退保",
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
    if not hit_rows:
        reasons = [REASON_WRONG_CHUNK]
        if question.doc_ids:
            reasons.append(REASON_MISSING_DOC)
        coverage = {
            "hit_count": 0,
            "doc_coverage": {"covered": 0, "total": len(set(question.doc_ids)), "missing_doc_ids": question.doc_ids},
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
            missing_doc_ids=question.doc_ids,
            certainty_score=0.0,
            coverage_status=coverage,
            rescue_needed=True,
            rescue_trigger=sorted(set(reasons)),
            details="no evidence hits",
        )

    reasons: list[str] = []
    missing_docs = _missing_doc_ids(question.doc_ids, hit_rows)
    if missing_docs:
        reasons.append(REASON_MISSING_DOC)

    weak_ratio = _weak_hit_ratio(hit_rows, min_chars=int(settings.get("min_hit_chars", 24)))
    if weak_ratio >= 0.75:
        reasons.append(REASON_WRONG_CHUNK)

    expected_terms = _expected_terms(option_text, domain)
    matched_terms = _matched_terms(expected_terms, hit_rows)
    if expected_terms and not matched_terms:
        reasons.append(REASON_MISSING_METRIC if domain == "financial_reports" else REASON_MISSING_CLAUSE)

    contradiction = _simple_contradiction(option_text, hit_rows)
    if contradiction:
        reasons.append(REASON_CONTRADICTORY)

    if REASON_CONTRADICTORY in reasons or (REASON_WRONG_CHUNK in reasons and len(reasons) > 1):
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
        weak_ratio=weak_ratio,
        domain=domain,
    )
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
        return RescueResult(hits=initial_hits, initial_gate=initial_gate, final_gate=initial_gate, rounds=[])

    max_rounds = int(gate_settings.get("max_rescue_rounds", len(DEFAULT_RESCUE_CHANNELS)))
    top_k = int(gate_settings.get("rescue_top_k", max(retrieval_settings.get("top_k", 6), 12)))
    max_hits = int(gate_settings.get("max_hits_after_rescue", top_k))
    merged = _dedupe_hits(initial_hits)
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
        merged = _dedupe_hits([*merged, *round_hits])
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


def _coverage_status(
    *,
    question: Question,
    hits: list[Any],
    expected_terms: list[str],
    matched_terms: list[str],
    weak_ratio: float,
    domain: str,
) -> dict[str, Any]:
    expected_docs = set(question.doc_ids)
    hit_docs = {_hit_doc_id(hit) for hit in hits if _hit_doc_id(hit)}
    unit_type_counts = Counter(_hit_unit_type(hit) for hit in hits)
    preferred_types = set(_preferred_unit_types(domain))
    return {
        "hit_count": len(hits),
        "doc_coverage": {
            "covered": len(expected_docs & hit_docs),
            "total": len(expected_docs),
            "missing_doc_ids": [doc_id for doc_id in question.doc_ids if doc_id not in hit_docs],
        },
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
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
    score -= min(0.2, float(coverage.get("weak_ratio", 0.0)) * 0.2)
    if coverage.get("expected_terms") and coverage.get("term_coverage", 1.0) < 1.0:
        score -= (1.0 - float(coverage.get("term_coverage", 0.0))) * 0.15
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
    top_k = int(gate_settings.get("rescue_top_k", max(retrieval_settings.get("top_k", 6), 12)))
    base_variants = build_query_variants(question, option_key, option_text, retrieval_settings)
    term_text = " ".join(_expected_terms(option_text, domain))
    focused_terms = _focused_terms(question, option_text, domain)
    focused = " ".join(part for part in [question.question, option_text, focused_terms, term_text] if part).strip()
    specs: list[dict[str, Any]] = []
    for channel in enabled_channels:
        if channel == "query_rewrite_search":
            queries = _dedupe_strings([focused, *base_variants, f"{option_text}\n{focused_terms}".strip()])
            specs.extend(_spec(channel, query, top_k=top_k) for query in queries)
        elif channel == "title_search":
            title_query = " ".join(part for part in [_title_terms_from_question(question, option_text), focused_terms] if part).strip()
            specs.extend(_spec(channel, query, top_k=top_k) for query in _dedupe_strings([title_query, f"{title_query}\n{option_text}".strip()]))
        elif channel == "unit_type_search":
            specs.append(_spec(channel, focused, top_k=top_k, unit_type_boosts=_preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))))
        elif channel == "table_metric_search" and domain in {"financial_reports", "financial_contracts"}:
            boosts = _preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))
            queries = _dedupe_strings([_table_metric_terms(question.question, option_text), focused])
            specs.extend(_spec(channel, query, top_k=top_k, unit_type_boosts=boosts) for query in queries)
        elif channel == "clause_formula_search" and domain == "insurance":
            boosts = _preferred_unit_boosts(domain, retrieval_settings.get("unit_type_boosts", {}))
            queries = _dedupe_strings([_insurance_clause_query(question.question, option_text), focused])
            specs.extend(_spec(channel, query, top_k=top_k, unit_type_boosts=boosts) for query in queries)
        elif channel == "per_doc_search":
            for doc_id in gate.missing_doc_ids or question.doc_ids:
                query = " ".join(part for part in [option_text, focused_terms, term_text, doc_id] if part).strip()
                specs.append(_spec(channel, query, top_k=top_k, doc_ids=[doc_id], ensure_per_doc=False))
        elif channel == "neighbor_expansion":
            specs.append(_spec(channel, focused, top_k=top_k, expand_neighbors=True))
    return [spec for spec in specs if spec.get("query")]


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
        " ".join(_expected_terms(option_text, domain)),
        " ".join(re.findall(r"\d{4}年?|\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:万|亿)?元", text)),
    ]
    if domain == "insurance":
        parts.append(" ".join(re.findall(r"(?:平安|国寿|太保|泰康|新华|人保|友邦|招商信诺|中信保诚)[\u4e00-\u9fa5A-Za-z0-9]{2,18}", text)))
    if domain == "regulatory":
        parts.append(" ".join(re.findall(r"第[一二三四五六七八九十百千万零\d]+条|[\u4e00-\u9fa5]{2,8}(?:报告|披露|禁入|处罚|备案|登记)", text)))
    return " ".join(part for part in parts if part).strip()


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
    formula_terms = [term for term in ["给付", "赔付", "较大者", "比例", "基本保额", "账户价值", "现金价值", "已交保费"] if term in text]
    return " ".join(_dedupe_strings([*products, *clause_terms, *formula_terms]))


def _expected_terms(option_text: str, domain: str) -> list[str]:
    expected: list[str] = []
    if domain == "financial_reports":
        for terms in METRIC_TERMS.values():
            if any(term in option_text for term in terms):
                expected.extend(terms)
    for term in CLAUSE_TERMS.get(domain, []):
        if term in option_text:
            expected.append(term)
    return _dedupe_strings(expected)


def _matched_terms(expected_terms: list[str], hits: Iterable[Any]) -> list[str]:
    text = "\n".join(_hit_text(hit) for hit in hits)
    return [term for term in expected_terms if term and term in text]


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
    evidence_text = "\n".join(_hit_text(hit) for hit in hits)
    exemption_terms = ["无需", "不需要", "可以不", "不必", "免于"]
    obligation_terms = ["应当", "必须", "需要", "需", "须"]
    action_terms = ["披露", "报告", "提交", "办理", "履行", "赔付", "给付"]
    option_claims_exemption = any(term in option_text for term in exemption_terms)
    option_claims_obligation = any(term in option_text for term in obligation_terms)
    evidence_claims_exemption = any(term in evidence_text for term in exemption_terms)
    evidence_claims_obligation = any(term in evidence_text for term in obligation_terms)
    shared_action = any(term in option_text and term in evidence_text for term in action_terms)
    return shared_action and (
        (option_claims_exemption and evidence_claims_obligation)
        or (option_claims_obligation and evidence_claims_exemption)
    )


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


def _dedupe_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    deduped: dict[str, RetrievalHit] = {}
    for hit in hits:
        if not isinstance(hit, RetrievalHit):
            continue
        current = deduped.get(hit.unit_id)
        if current is None or hit.score > current.score:
            deduped[hit.unit_id] = hit
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


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
