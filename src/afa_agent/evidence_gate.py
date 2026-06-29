from __future__ import annotations

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
        return GateResult(status=GATE_FAIL, reasons=sorted(set(reasons)), missing_doc_ids=question.doc_ids, details="no evidence hits")

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

    return GateResult(
        status=status,
        reasons=sorted(set(reasons)),
        missing_doc_ids=missing_docs,
        matched_terms=matched_terms,
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
    if initial_gate.status == GATE_PASS:
        return RescueResult(hits=initial_hits, initial_gate=initial_gate, final_gate=initial_gate, rounds=[])

    max_rounds = int(gate_settings.get("max_rescue_rounds", 2))
    top_k = int(gate_settings.get("rescue_top_k", max(retrieval_settings.get("top_k", 6), 12)))
    max_hits = int(gate_settings.get("max_hits_after_rescue", top_k))
    merged = _dedupe_hits(initial_hits)
    rounds: list[dict[str, Any]] = []
    query_groups = _rescue_query_groups(question, option_key, option_text, domain, retrieval_settings, initial_gate)

    for round_index, queries in enumerate(query_groups[:max_rounds], start=1):
        round_hits: list[RetrievalHit] = []
        for query in queries:
            round_hits.extend(
                _search(
                    retriever,
                    question.doc_ids,
                    query,
                    retrieval_settings,
                    top_k=top_k,
                    ensure_per_doc=True,
                )
            )
        merged = _dedupe_hits([*merged, *round_hits])
        merged = _enforce_doc_quota(merged, question.doc_ids, int(gate_settings.get("per_doc_quota", 2)), max_hits)
        gate = evaluate_evidence(question, option_key, option_text, merged, domain, gate_settings)
        rounds.append(
            {
                "round": round_index,
                "queries": queries,
                "gate": gate.to_dict(),
                "hits": serialize_hits(merged, limit=max_hits),
            }
        )
        if gate.status == GATE_PASS:
            return RescueResult(hits=merged[:max_hits], initial_gate=initial_gate, final_gate=gate, rounds=rounds)

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


def _search(
    retriever: Any,
    doc_ids: list[str],
    query: str,
    retrieval_settings: dict[str, Any],
    *,
    top_k: int,
    ensure_per_doc: bool,
) -> list[RetrievalHit]:
    kwargs = {
        "top_k": top_k,
        "ensure_per_doc": ensure_per_doc,
        "expand_neighbors": retrieval_settings.get("expand_neighbors", True),
    }
    unit_type_boosts = retrieval_settings.get("unit_type_boosts", {})
    try:
        return retriever.search(doc_ids, query, unit_type_boosts=unit_type_boosts, **kwargs)
    except TypeError:
        return retriever.search(doc_ids, query, **kwargs)


def _rescue_query_groups(
    question: Question,
    option_key: str,
    option_text: str,
    domain: str,
    retrieval_settings: dict[str, Any],
    gate: GateResult,
) -> list[list[str]]:
    base_variants = build_query_variants(question, option_key, option_text, retrieval_settings)
    term_text = " ".join(_expected_terms(option_text, domain))
    doc_hint = " ".join(question.doc_ids)
    focused = " ".join(part for part in [question.question, option_text, term_text] if part).strip()
    doc_focused = " ".join(part for part in [option_text, term_text, doc_hint] if part).strip()
    missing_doc_queries = [f"{option_text}\n{doc_id}\n{term_text}".strip() for doc_id in gate.missing_doc_ids]
    groups = [
        _dedupe_strings([focused, *base_variants, doc_focused]),
        _dedupe_strings([*missing_doc_queries, f"{question.question}\n{doc_hint}\n{term_text}".strip()]),
    ]
    return [group for group in groups if group]


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
