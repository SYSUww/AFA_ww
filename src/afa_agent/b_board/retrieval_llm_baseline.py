from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from afa_agent.b_board.io import (
    BAnswer,
    BQuestion,
    infer_percent_suffix_requirement,
    infer_requested_decimal_places,
    validate_b_answer,
)
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.models import RetrievalHit
from afa_agent.retrieval_query import (
    RetrievalPlan,
    RetrievalRequest,
    generate_retrieval_plan,
)


PIPELINE_VERSION = "b_retrieval_llm_answer_blind_v4"
PROMPT_VERSION = "b_retrieval_llm_modular_final_submission_v8"
RETRIEVAL_POLICY_VERSION = (
    "semantic_slots_entity_coverage_option_quota_doc_union_primary_guard_v5"
)
ANCHOR_FIRST_RETRIEVAL_POLICY_VERSION = (
    "semantic_slots_entity_coverage_option_quota_anchor_first_v3"
)
DOCUMENT_CANDIDATE_STRATEGIES = ("anchor_union", "anchor_first")
EVIDENCE_QUOTA_STRATEGIES = (
    "primary_guard",
    "document_balanced",
    "adaptive_multi_report_calculation",
    "metric_slot_coverage",
)
ANCHOR_FIRST_DOCUMENT_BALANCED_POLICY_VERSION = (
    "semantic_slots_entity_coverage_option_quota_v3"
)
ADAPTIVE_MULTI_REPORT_POLICY_VERSION = (
    "semantic_slots_anchor_first_adaptive_multi_report_quota_v6"
)
METRIC_SLOT_COVERAGE_POLICY_VERSION = (
    "semantic_slots_anchor_first_entity_metric_coverage_v7"
)

UNIT_TYPE_BOOSTS: dict[str, dict[str, float]] = {
    "financial_contracts": {"element_block": 1.8, "paragraph": 1.0},
    "financial_reports": {"metric_row": 1.8, "paragraph": 1.0},
    "insurance": {"formula_block": 1.8, "clause_block": 1.1},
    "research": {"conclusion_block": 1.6, "paragraph": 1.0},
    "regulatory": {},
}


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    anchor_queries: tuple[str, ...]
    document_queries: tuple[str, ...]
    primary_queries: tuple[str, ...]
    supplemental_queries: tuple[str, ...]
    option_queries: tuple[tuple[str, ...], ...]
    plans: tuple[RetrievalPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_queries": list(self.anchor_queries),
            "document_queries": list(self.document_queries),
            "primary_queries": list(self.primary_queries),
            "supplemental_queries": list(self.supplemental_queries),
            "option_queries": [list(items) for items in self.option_queries],
            "plans": [plan.to_dict() for plan in self.plans],
        }


def load_domain_indexes(index_root: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for domain in UNIT_TYPE_BOOSTS:
        path = index_root / domain / "index.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload.get("units")
        if not isinstance(units, list) or not units:
            raise ValueError(f"{path}: index must contain a non-empty units array")
        payloads[domain] = payload
    return payloads


def make_retriever(domain: str, units: list[dict[str, Any]]) -> Any:
    if domain == "regulatory":
        return RegulatoryRetriever(units)
    return GenericBM25Retriever(units)


def build_retrieval_bundle(
    question: BQuestion,
    *,
    max_queries_per_option: int = 12,
) -> RetrievalBundle:
    requests = _retrieval_requests(question)
    plans = tuple(
        generate_retrieval_plan(request, max_queries=max_queries_per_option)
        for request in requests
    )
    primary = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel == "primary"
    )
    allowed_supplemental = (
        {"coverage"}
        if question.answer_format == "calculation"
        else {"support", "contrast", "scope_check"}
    )
    supplemental = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel in allowed_supplemental
    )
    document_queries = _dedupe(
        variant.query
        for plan in plans
        for variant in plan.variants
        if variant.channel in {"primary", "broad"}
    )
    anchor_queries = _document_anchor_queries(question, plans)
    option_queries = tuple(_focused_plan_queries(plan) for plan in plans)
    return RetrievalBundle(
        anchor_queries=tuple(anchor_queries),
        document_queries=tuple(document_queries),
        primary_queries=tuple(primary),
        supplemental_queries=tuple(supplemental),
        option_queries=option_queries,
        plans=plans,
    )


def retrieve_question_evidence(
    question: BQuestion,
    *,
    retriever: Any,
    all_doc_ids: Sequence[str],
    per_query_top_k: int = 10,
    final_top_k: int = 10,
    supplemental_weight: float = 0.11,
    max_queries_per_option: int = 12,
    max_doc_candidates: int = 6,
    document_candidate_strategy: str = "anchor_union",
    evidence_quota_strategy: str = "primary_guard",
) -> dict[str, Any]:
    if not all_doc_ids:
        raise ValueError(f"{question.domain}: corpus has no document ids")
    bundle = build_retrieval_bundle(
        question,
        max_queries_per_option=max_queries_per_option,
    )
    boosts = UNIT_TYPE_BOOSTS.get(question.domain, {})
    discovery = _rank_queries(
        retriever,
        list(all_doc_ids),
        list(bundle.document_queries),
        per_query_top_k=max(per_query_top_k, 20),
        final_top_k=max(final_top_k * 5, 50),
        unit_type_boosts=boosts,
        expand_neighbors=False,
    )
    anchored_doc_ids = _anchor_doc_candidates(
        retriever,
        list(all_doc_ids),
        list(bundle.anchor_queries),
        per_query_top_k=max(per_query_top_k, 10),
        unit_type_boosts=boosts,
    )
    selected_doc_ids = _select_document_candidates(
        anchored_doc_ids,
        _ranked_doc_ids(discovery),
        max_doc_candidates=max_doc_candidates,
        strategy=document_candidate_strategy,
    )
    if not selected_doc_ids:
        raise ValueError("document discovery produced no candidates")
    primary = _rank_queries(
        retriever,
        selected_doc_ids,
        list(bundle.primary_queries),
        per_query_top_k=per_query_top_k,
        final_top_k=final_top_k,
        unit_type_boosts=boosts,
        expand_neighbors=True,
    )
    supplemental = _rank_queries(
        retriever,
        selected_doc_ids,
        list(bundle.supplemental_queries),
        per_query_top_k=per_query_top_k,
        final_top_k=max(final_top_k * 3, final_top_k),
        unit_type_boosts=boosts,
        expand_neighbors=True,
    )
    document_evidence_query = _document_evidence_query(bundle.plans)
    document_rankings = [
        _rank_queries(
            retriever,
            [doc_id],
            [document_evidence_query],
            per_query_top_k=per_query_top_k,
            final_top_k=max(2, min(final_top_k, 4)),
            unit_type_boosts=boosts,
            expand_neighbors=True,
        )
        for doc_id in selected_doc_ids
        if document_evidence_query
    ]
    required_metric_slots = (
        _required_metric_slots(question, bundle.plans)
        if (
            evidence_quota_strategy == "metric_slot_coverage"
            and len(_dedupe(selected_doc_ids)) > 1
        )
        else []
    )
    metric_slot_rankings = _build_metric_slot_rankings(
        retriever,
        selected_doc_ids=selected_doc_ids,
        metrics=required_metric_slots,
        period=_first_period(bundle.plans),
        per_query_top_k=per_query_top_k,
        final_top_k=final_top_k,
        unit_type_boosts=boosts,
    )
    option_rankings = [
        _rank_queries(
            retriever,
            selected_doc_ids,
            list(queries),
            per_query_top_k=per_query_top_k,
            final_top_k=max(3, min(final_top_k, 5)),
            unit_type_boosts=boosts,
            expand_neighbors=True,
        )
        for queries in bundle.option_queries
        if queries
    ]
    active_evidence_quota_strategy = _resolve_evidence_quota_strategy(
        question,
        selected_doc_ids=selected_doc_ids,
        strategy=evidence_quota_strategy,
    )
    blended = _blend_rankings(
        primary,
        supplemental,
        document_rankings=document_rankings,
        metric_slot_rankings=metric_slot_rankings,
        option_rankings=option_rankings,
        supplemental_weight=supplemental_weight,
        final_top_k=final_top_k,
        evidence_quota_strategy=active_evidence_quota_strategy,
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "policy_version": retrieval_policy_version(
            document_candidate_strategy,
            evidence_quota_strategy,
        ),
        "document_candidate_strategy": document_candidate_strategy,
        "evidence_quota_strategy": evidence_quota_strategy,
        "active_evidence_quota_strategy": active_evidence_quota_strategy,
        "corpus_doc_count": len(set(all_doc_ids)),
        "selected_doc_ids": selected_doc_ids,
        "anchored_doc_ids": anchored_doc_ids,
        "bundle": bundle.to_dict(),
        "document_discovery": _ranking_to_dict(discovery),
        "primary": _ranking_to_dict(primary),
        "supplemental": _ranking_to_dict(supplemental),
        "document_rankings": [
            _ranking_to_dict(ranking) for ranking in document_rankings
        ],
        "required_metric_slots": required_metric_slots,
        "metric_slot_rankings": [
            _ranking_to_dict(ranking) for ranking in metric_slot_rankings
        ],
        "option_rankings": [
            _ranking_to_dict(ranking) for ranking in option_rankings
        ],
        "final": _ranking_to_dict(blended),
    }


def retrieval_policy_version(
    document_candidate_strategy: str,
    evidence_quota_strategy: str = "primary_guard",
) -> str:
    if evidence_quota_strategy not in EVIDENCE_QUOTA_STRATEGIES:
        raise ValueError(
            "evidence_quota_strategy must be one of "
            f"{EVIDENCE_QUOTA_STRATEGIES}"
        )
    if (
        evidence_quota_strategy
        in {
            "adaptive_multi_report_calculation",
            "metric_slot_coverage",
        }
        and document_candidate_strategy != "anchor_first"
    ):
        raise ValueError(
            f"{evidence_quota_strategy} requires anchor_first documents"
        )
    if (
        document_candidate_strategy == "anchor_union"
        and evidence_quota_strategy == "primary_guard"
    ):
        return RETRIEVAL_POLICY_VERSION
    if (
        document_candidate_strategy == "anchor_first"
        and evidence_quota_strategy == "primary_guard"
    ):
        return ANCHOR_FIRST_RETRIEVAL_POLICY_VERSION
    if (
        document_candidate_strategy == "anchor_first"
        and evidence_quota_strategy == "document_balanced"
    ):
        return ANCHOR_FIRST_DOCUMENT_BALANCED_POLICY_VERSION
    if (
        document_candidate_strategy == "anchor_first"
        and evidence_quota_strategy == "adaptive_multi_report_calculation"
    ):
        return ADAPTIVE_MULTI_REPORT_POLICY_VERSION
    if (
        document_candidate_strategy == "anchor_first"
        and evidence_quota_strategy == "metric_slot_coverage"
    ):
        return METRIC_SLOT_COVERAGE_POLICY_VERSION
    if document_candidate_strategy == "anchor_union":
        return (
            "semantic_slots_entity_coverage_option_quota_doc_union_v4"
        )
    raise ValueError(
        "document_candidate_strategy must be one of "
        f"{DOCUMENT_CANDIDATE_STRATEGIES}"
    )


def _resolve_evidence_quota_strategy(
    question: BQuestion,
    *,
    selected_doc_ids: Sequence[str],
    strategy: str,
) -> str:
    if strategy not in {
        "adaptive_multi_report_calculation",
        "metric_slot_coverage",
    }:
        if strategy not in {"primary_guard", "document_balanced"}:
            raise ValueError(
                f"evidence quota strategy must be one of {EVIDENCE_QUOTA_STRATEGIES}"
            )
        return strategy
    active = (
        question.domain == "financial_reports"
        and question.answer_format == "calculation"
        and len(_dedupe(selected_doc_ids)) > 1
    )
    if not active:
        return "primary_guard"
    return (
        "document_balanced"
        if strategy == "adaptive_multi_report_calculation"
        else "metric_slot_coverage"
    )


def _select_document_candidates(
    anchored_doc_ids: Sequence[str],
    discovered_doc_ids: Sequence[str],
    *,
    max_doc_candidates: int,
    strategy: str,
) -> list[str]:
    if max_doc_candidates < 1:
        raise ValueError("max_doc_candidates must be positive")
    retrieval_policy_version(strategy)
    anchors = _dedupe(anchored_doc_ids)
    discovered = _dedupe(discovered_doc_ids)
    if strategy == "anchor_first" and anchors:
        return anchors[:max_doc_candidates]
    return _dedupe([*anchors, *discovered])[:max_doc_candidates]


def _document_anchor_queries(
    question: BQuestion,
    plans: Sequence[RetrievalPlan],
) -> list[str]:
    if not plans:
        return []
    anchors = list(plans[0].slots.anchors)
    periods = list(plans[0].slots.periods)
    document_marker = next(
        (
            marker
            for marker in (
                "年度报告",
                "年报",
                "募集说明书",
                "报告书",
                "保险条款",
                "管理办法",
                "管理规定",
            )
            if marker in question.question
        ),
        "",
    )
    period = periods[0] if periods else ""
    return _dedupe(
        " ".join(part for part in (anchor, period, document_marker) if part)
        for anchor in anchors
    )


def _focused_plan_queries(plan: RetrievalPlan) -> tuple[str, ...]:
    slots = plan.slots
    if slots.atoms:
        return tuple(_dedupe(slots.atoms))
    slot_query = " ".join(
        [
            *slots.articles,
            *slots.periods,
            *slots.topics,
            *slots.quantities,
            *slots.relations,
            *slots.scopes,
            *slots.exceptions,
        ]
    )
    return tuple(_dedupe([slot_query]))


def _document_evidence_query(plans: Sequence[RetrievalPlan]) -> str:
    if not plans:
        return ""
    slots = plans[0].slots
    return " ".join(
        _dedupe(
            [
                *slots.articles,
                *slots.periods,
                *slots.topics,
                *slots.quantities,
                *slots.relations,
                *slots.scopes,
                *slots.exceptions,
            ]
        )
    )


def _first_period(plans: Sequence[RetrievalPlan]) -> str:
    if not plans or not plans[0].slots.periods:
        return ""
    return str(plans[0].slots.periods[0])


def _required_metric_slots(
    question: BQuestion,
    plans: Sequence[RetrievalPlan],
) -> list[str]:
    """Extract disclosed operands before the question's calculation clause."""

    if (
        question.domain != "financial_reports"
        or question.answer_format != "calculation"
        or not plans
    ):
        return []
    body = question.question.split("：", 1)[-1]
    boundaries = [
        position
        for marker in ("计算", "换算", "排序", "比较")
        if (position := body.find(marker)) >= 0
    ]
    prefix = body[: min(boundaries)] if boundaries else body
    non_metric_topics = {
        "年度报告",
        "报告期",
        "原始金额",
        "数据",
        "统一",
        "公司",
        "两家",
    }
    return _dedupe(
        topic
        for topic in plans[0].slots.topics
        if topic not in non_metric_topics
        and topic in prefix
        and re.search(
            r"(?:收入|利润|现金流量净额|现金流量|现金流|分红|"
            r"资产负债率|收益率|"
            r"每股收益|金额|费用|成本|余额|价值|销量|规模|占比|"
            r"渗透率|保费|保额)$",
            topic,
        )
    )


def _build_metric_slot_rankings(
    retriever: Any,
    *,
    selected_doc_ids: Sequence[str],
    metrics: Sequence[str],
    period: str,
    per_query_top_k: int,
    final_top_k: int,
    unit_type_boosts: Mapping[str, float],
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for doc_id in selected_doc_ids:
        for metric in metrics:
            ranking = _rank_queries(
                retriever,
                [str(doc_id)],
                [" ".join(part for part in (period, metric) if part)],
                per_query_top_k=per_query_top_k,
                final_top_k=max(2, min(final_top_k, 4)),
                unit_type_boosts=unit_type_boosts,
                expand_neighbors=True,
            )
            ranking["metric_slot"] = str(metric)
            ranking["metric_doc_id"] = str(doc_id)
            rankings.append(ranking)
    return rankings


def _retrieval_requests(question: BQuestion) -> list[RetrievalRequest]:
    if question.answer_format in {"calculation", "extraction"}:
        return [
            RetrievalRequest(
                domain=question.domain,
                question=question.question,
                question_type=question.type,
                answer_format=question.answer_format,
            )
        ]
    return [
        RetrievalRequest(
            domain=question.domain,
            question=question.question,
            option_text=option_text,
            question_type=question.type,
            answer_format=question.answer_format,
        )
        for option_text in question.options.values()
    ]


def _rank_queries(
    retriever: Any,
    doc_ids: list[str],
    queries: list[str],
    *,
    per_query_top_k: int,
    final_top_k: int,
    unit_type_boosts: Mapping[str, float],
    expand_neighbors: bool,
) -> dict[str, Any]:
    scores: defaultdict[str, float] = defaultdict(float)
    hit_by_id: dict[str, RetrievalHit] = {}
    for query in queries:
        kwargs = {
            "top_k": per_query_top_k,
            "ensure_per_doc": False,
            "expand_neighbors": expand_neighbors,
        }
        try:
            hits = retriever.search(
                doc_ids,
                query,
                unit_type_boosts=dict(unit_type_boosts),
                **kwargs,
            )
        except TypeError:
            hits = retriever.search(doc_ids, query, **kwargs)
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            unit_id = _normalize_unit_id(hit.unit_id)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            scores[unit_id] += 1.0 / (60 + rank)
            current = hit_by_id.get(unit_id)
            if current is None or hit.score > current.score:
                hit_by_id[unit_id] = hit
    ranked_ids = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
    ranked_ids = ranked_ids[:final_top_k]
    return {
        "queries": list(queries),
        "ranked_ids": ranked_ids,
        "ranked_hits": [hit_by_id[unit_id] for unit_id in ranked_ids],
    }


def _ranked_doc_ids(ranking: Mapping[str, Any]) -> list[str]:
    return _dedupe(hit.doc_id for hit in ranking["ranked_hits"])


def _anchor_doc_candidates(
    retriever: Any,
    doc_ids: list[str],
    anchor_queries: list[str],
    *,
    per_query_top_k: int,
    unit_type_boosts: Mapping[str, float],
) -> list[str]:
    selected: list[str] = []
    for query in anchor_queries:
        kwargs = {
            "top_k": per_query_top_k,
            "ensure_per_doc": False,
            "expand_neighbors": False,
        }
        try:
            hits = retriever.search(
                doc_ids,
                query,
                unit_type_boosts=dict(unit_type_boosts),
                **kwargs,
            )
        except TypeError:
            hits = retriever.search(doc_ids, query, **kwargs)
        if hits:
            selected.append(str(hits[0].doc_id))
    return _dedupe(selected)


def _blend_rankings(
    primary: Mapping[str, Any],
    supplemental: Mapping[str, Any],
    *,
    document_rankings: Sequence[Mapping[str, Any]],
    metric_slot_rankings: Sequence[Mapping[str, Any]],
    option_rankings: Sequence[Mapping[str, Any]],
    supplemental_weight: float,
    final_top_k: int,
    evidence_quota_strategy: str = "primary_guard",
) -> dict[str, Any]:
    if supplemental_weight < 0:
        raise ValueError("supplemental_weight must not be negative")
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, unit_id in enumerate(primary["ranked_ids"], start=1):
        scores[str(unit_id)] += 1.0 / rank
    for rank, unit_id in enumerate(supplemental["ranked_ids"], start=1):
        scores[str(unit_id)] += supplemental_weight / rank
    hit_by_id = {
        _normalize_unit_id(hit.unit_id): hit
        for hit in [
            *primary["ranked_hits"],
            *supplemental["ranked_hits"],
            *(
                hit
                for ranking in document_rankings
                for hit in ranking["ranked_hits"]
            ),
            *(
                hit
                for ranking in metric_slot_rankings
                for hit in ranking["ranked_hits"]
            ),
            *(
                hit
                for ranking in option_rankings
                for hit in ranking["ranked_hits"]
            ),
        ]
    }
    reserved_ids = _reserved_evidence_ids(
        primary,
        document_rankings=document_rankings,
        metric_slot_rankings=metric_slot_rankings,
        option_rankings=option_rankings,
        final_top_k=final_top_k,
        strategy=evidence_quota_strategy,
    )
    for ranking in document_rankings:
        for rank, unit_id in enumerate(ranking["ranked_ids"], start=1):
            scores[str(unit_id)] += 0.5 / rank
    for ranking in metric_slot_rankings:
        selected_metric_id = _best_metric_evidence_id(ranking)
        if selected_metric_id:
            scores[selected_metric_id] += 0.75
    for ranking in option_rankings:
        for rank, unit_id in enumerate(ranking["ranked_ids"], start=1):
            scores[str(unit_id)] += 0.5 / rank
    ranked_ids = [
        *reserved_ids[:final_top_k],
        *(
            unit_id
            for unit_id in sorted(
                scores,
                key=lambda unit_id: (-scores[unit_id], unit_id),
            )
            if unit_id not in reserved_ids
        ),
    ][:final_top_k]
    return {
        "queries": [
            *primary["queries"],
            *supplemental["queries"],
            *(
                query
                for ranking in document_rankings
                for query in ranking["queries"]
            ),
            *(
                query
                for ranking in metric_slot_rankings
                for query in ranking["queries"]
            ),
            *(
                query
                for ranking in option_rankings
                for query in ranking["queries"]
            ),
        ],
        "ranked_ids": ranked_ids,
        "ranked_hits": [hit_by_id[unit_id] for unit_id in ranked_ids],
    }


def _reserved_evidence_ids(
    primary: Mapping[str, Any],
    *,
    document_rankings: Sequence[Mapping[str, Any]],
    option_rankings: Sequence[Mapping[str, Any]],
    final_top_k: int,
    strategy: str,
    metric_slot_rankings: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    if strategy not in EVIDENCE_QUOTA_STRATEGIES:
        raise ValueError(
            f"evidence quota strategy must be one of {EVIDENCE_QUOTA_STRATEGIES}"
        )
    document_quota = 2 if strategy == "document_balanced" else 1
    primary_guard = (
        []
        if strategy == "document_balanced"
        else list(
            primary[
                "ranked_ids"
            ][
                : min(
                    2 if strategy == "metric_slot_coverage" else 4,
                    final_top_k,
                )
            ]
        )
    )
    reserved_document_ids = (
        []
        if strategy == "metric_slot_coverage"
        else [
            unit_id
            for ranking in document_rankings
            for unit_id in ranking["ranked_ids"][:document_quota]
        ]
    )
    return _dedupe(
        [
            *primary_guard,
            *reserved_document_ids,
            *(
                selected
                for ranking in metric_slot_rankings
                if (selected := _best_metric_evidence_id(ranking))
            ),
            *(
                unit_id
                for ranking in option_rankings
                for unit_id in ranking["ranked_ids"][:1]
            ),
        ]
    )


def _best_metric_evidence_id(ranking: Mapping[str, Any]) -> str:
    """Require the target metric and its disclosed value in one local window."""

    metric = str(ranking.get("metric_slot", "")).strip()
    if not metric:
        return ""
    hits = list(ranking.get("ranked_hits") or [])
    for hit in hits:
        text = re.sub(r"\s+", " ", str(getattr(hit, "text", "")))
        metric_positions = [
            match.start()
            for match in re.finditer(re.escape(metric), text)
        ]
        if any(
            _metric_value_window_is_disclosed(
                text[max(0, position - 24) : position + len(metric) + 80],
                metric,
            )
            for position in metric_positions
        ):
            return _normalize_unit_id(hit.unit_id)
    return ""


def _metric_value_window_is_disclosed(window: str, metric: str) -> bool:
    if re.search(
        r"(?:被担保|担保对象|超过|高于|低于|不低于|不超过|阈值)",
        window,
    ):
        return False
    after_metric = window.split(metric, 1)[-1]
    return bool(
        re.search(
            r"(?:为|是|达|[:：|]|\s{1,3}).{0,24}"
            r"(?<!\d)(?!20\d{2}\b)"
            r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|％)?",
            after_metric,
        )
    )


def _ranking_to_dict(ranking: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queries": list(ranking["queries"]),
        "ranked_ids": list(ranking["ranked_ids"]),
        "hits": [
            {
                **hit.to_dict(),
                "unit_id": _normalize_unit_id(hit.unit_id),
            }
            for hit in ranking["ranked_hits"]
        ],
    }


def prepare_evidence_payload(
    retrieval: Mapping[str, Any],
    *,
    max_hit_chars: int = 1800,
    max_total_chars: int = 12000,
) -> list[dict[str, Any]]:
    if max_hit_chars < 1 or max_total_chars < 1:
        raise ValueError("evidence character limits must be positive")
    evidence: list[dict[str, Any]] = []
    used_chars = 0
    source_keys: dict[str, str] = {}
    for rank, hit in enumerate(retrieval["final"]["hits"], start=1):
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break
        raw_text = str(hit.get("text", "")).strip()
        text = raw_text[: min(max_hit_chars, remaining)]
        if not text:
            continue
        doc_id = str(hit["doc_id"])
        source_key = source_keys.setdefault(doc_id, f"S{len(source_keys) + 1:02d}")
        evidence.append(
            {
                "evidence_key": f"E{len(evidence) + 1:02d}",
                "source_key": source_key,
                "evidence_id": str(hit["unit_id"]),
                "doc_id": doc_id,
                "rank": rank,
                "unit_type": str((hit.get("metadata") or {}).get("unit_type", "")),
                "title_path": [str(item) for item in hit.get("title_path", [])],
                "text": text,
                "prompt_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "source_text_sha256": hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest(),
                "truncated": len(text) < len(raw_text),
                "merged_from": list(hit.get("merged_from", [])),
                "source_order": list(hit.get("source_order", [])),
                "overlap_chars": int(hit.get("overlap_chars", 0)),
                "component_hashes": list(hit.get("component_hashes", [])),
                "compaction_truncation_provenance": dict(
                    hit.get("truncation_provenance", {})
                ),
            }
        )
        used_chars += len(text)
    if not evidence:
        raise ValueError("retrieval produced no usable evidence")
    return evidence


def build_answer_schema(
    question: BQuestion,
) -> dict[str, Any]:
    option_labels = list(question.options)
    if question.answer_format in {"tf", "mcq"}:
        answer_item_schema: dict[str, Any] = {
            "type": "string",
            "enum": option_labels,
        }
    elif question.answer_format == "multi":
        legal_answers = [
            "".join(items)
            for size in range(2, len(option_labels) + 1)
            for items in combinations(option_labels, size)
        ]
        answer_item_schema = {
            "type": "string",
            "enum": legal_answers,
        }
    else:
        answer_item_schema = {
            "type": "string",
            "minLength": 2,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Let the model complete its evidence-backed audit summary before
            # committing the final structured answer.
            "reasoning": {"type": "string", "minLength": 20},
            "answer_parts": {
                "type": "array",
                "items": answer_item_schema,
                "minItems": question.answer_slots,
                "maxItems": question.answer_slots,
            },
        },
        "required": ["reasoning", "answer_parts"],
    }


def build_reasoning_canonical_schema() -> dict[str, Any]:
    """Schema whose unchanged reasoning conclusion is the sole answer source."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string", "minLength": 20},
        },
        "required": ["reasoning"],
    }


def build_frozen_answer_reasoning_schema(
    frozen_answer_parts: Sequence[str],
) -> dict[str, Any]:
    """Schema for a reasoning-only retry whose answer is already frozen."""

    conclusion = "；".join(str(item) for item in frozen_answer_parts)
    if not conclusion:
        raise ValueError("frozen answer must not be empty")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string", "minLength": 20},
        },
        "required": ["reasoning"],
    }


def build_answer_messages(
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
    *,
    validation_error: str = "",
    previous_response: str = "",
    output_contract: str = "joint",
) -> list[dict[str, str]]:
    if output_contract not in {"joint", "reasoning_canonical"}:
        raise ValueError(f"unsupported output contract: {output_contract}")
    options_text = (
        "\n".join(f"{key}. {value}" for key, value in question.options.items())
        if question.options
        else "（无选项）"
    )
    source_lines: list[str] = []
    seen_sources: set[str] = set()
    for item in evidence:
        source_key = str(item["source_key"])
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_lines.append(f"[{source_key}] {item['doc_id']}")
    evidence_text = "\n\n".join(
        (
            f"[{item['evidence_key']}|{item['source_key']}] "
            f"title={_prompt_title(item.get('title_path', []))}\n{item['text']}"
        )
        for item in evidence
    )
    format_instruction = _format_instruction(question)
    slot_text = (
        f"提交槽数量：{question.answer_slots}\n"
        if question.answer_format not in {"tf", "mcq", "multi"}
        else ""
    )
    output_request = (
        "请一次性输出符合JSON Schema的最终reasoning。reasoning末尾的"
        "“结论：...”是唯一答案来源，代码只做原样抽取，不补写或改写答案。"
        if output_contract == "reasoning_canonical"
        else "请一次性输出符合JSON Schema、可直接提交的最终answer_parts和最终reasoning。"
    )
    messages = [
        {
            "role": "system",
            "content": _system_prompt(
                question,
                output_contract=output_contract,
            ),
        },
        {
            "role": "user",
            "content": (
                f"领域：{question.domain}\n"
                f"题型：{question.type}（{question.answer_format}）\n"
                f"题目：{question.question}\n"
                f"选项：\n{options_text}\n\n"
                f"{slot_text}"
                f"格式要求：{format_instruction}\n\n"
                f"来源：\n{chr(10).join(source_lines)}\n\n"
                f"检索证据：\n{evidence_text}\n\n"
                f"{output_request}"
                "reasoning不得只引用E01等编号，必须复述支撑答案的关键事实。"
                "输出后不会由代码或其他模型补写、改写或总结。"
            ),
        },
    ]
    if validation_error:
        messages.extend(
            [
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": _retry_instruction(
                        validation_error,
                        question=question,
                        output_contract=output_contract,
                    ),
                },
            ]
        )
    serialized = json.dumps(messages, ensure_ascii=False)
    if question.qid in serialized:
        raise ValueError("qid leaked into model messages")
    return messages


def build_frozen_answer_reasoning_messages(
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
    *,
    frozen_answer_parts: Sequence[str],
) -> list[dict[str, str]]:
    required_conclusion = "；".join(
        str(item) for item in frozen_answer_parts
    )
    if not required_conclusion:
        raise ValueError("frozen answer must not be empty")
    options_text = (
        "\n".join(f"{key}. {value}" for key, value in question.options.items())
        if question.options
        else "（无选项）"
    )
    source_lines: list[str] = []
    seen_sources: set[str] = set()
    for item in evidence:
        source_key = str(item["source_key"])
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        source_lines.append(f"[{source_key}] {item['doc_id']}")
    evidence_text = "\n\n".join(
        (
            f"[{item['evidence_key']}|{item['source_key']}] "
            f"title={_prompt_title(item.get('title_path', []))}\n{item['text']}"
        )
        for item in evidence
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是金融材料问答模型，生成可直接提交、可独立阅读的最终说明。"
                "只返回JSON Schema规定的reasoning字段。reasoning写成"
                "简洁、自包含、可审计的证据摘要：先明确主体、年份或适用范围，"
                "再复述检索证据明示的关键事实和必要计算，最后自然收束。只写给定"
                "证据中的事实，不补充外部常识或未经检索的数值；证据有限时，客观写明"
                "缺少的指标或口径。正文只讨论材料与计算，不描述答案生成、任务执行或"
                "系统约束。必须在reasoning末尾写出指定的最终结论。不得只引用E01等"
                "编号，不得输出答案数组、隐藏思维过程或其他字段。输出后不会由代码"
                "补写、拼接或改写。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"领域：{question.domain}\n"
                f"题型：{question.type}（{question.answer_format}）\n"
                f"题目：{question.question}\n"
                f"选项：\n{options_text}\n\n"
                f"reasoning末尾必须原样写“结论：{required_conclusion}”。\n\n"
                f"来源：\n{chr(10).join(source_lines)}\n\n"
                f"检索证据：\n{evidence_text}\n\n"
                "请返回仅含reasoning字段的JSON对象。"
            ),
        },
    ]
    serialized = json.dumps(messages, ensure_ascii=False)
    if question.qid in serialized:
        raise ValueError("qid leaked into frozen-answer reasoning messages")
    return messages


def validate_reasoning_canonical_payload(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if set(payload) != {"reasoning"}:
        raise ValueError("reasoning-canonical response must contain only reasoning")
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str) or len(reasoning.strip()) < 20:
        raise ValueError("reasoning must contain at least 20 characters")
    marker_index = max(reasoning.rfind("结论："), reasoning.rfind("结论:"))
    if marker_index < 0:
        raise ValueError("reasoning must end with an explicit 结论")
    conclusion = reasoning[marker_index + 3 :]
    if not conclusion:
        raise ValueError("reasoning conclusion must not be empty")
    if conclusion != conclusion.strip():
        raise ValueError("reasoning conclusion must not contain outer whitespace")
    if re.search(r"[。.!！]$", conclusion):
        raise ValueError("reasoning conclusion must not contain terminal punctuation")
    raw_parts = (
        conclusion.split("；")
        if question.answer_slots > 1
        else [conclusion]
    )
    parts = tuple(part.strip() for part in raw_parts)
    if (
        len(parts) != question.answer_slots
        or any(not part for part in parts)
    ):
        raise ValueError("reasoning conclusion slot count mismatch")
    validate_b_answer(
        question,
        BAnswer(
            qid=question.qid,
            answer_parts=parts,
            reasoning=reasoning,
        ),
    )
    return {
        "answer_parts": list(parts),
        "reasoning": reasoning,
        "decision_trace": {
            "output_contract": "reasoning_canonical_v1",
            "answer_source": "unchanged_model_reasoning_conclusion",
            "answer_modified": False,
            "reasoning_modified": False,
        },
    }


def validate_answer_payload(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = validate_answer_shape_payload(question, payload)
    separator_equivalence = _validate_reasoning_conclusion(
        normalized["answer_parts"],
        normalized["reasoning"],
    )
    if separator_equivalence:
        normalized["decision_trace"] = {
            "postprocessing_mode": (
                "multi_choice_conclusion_separator_equivalence"
            ),
            "answer_source": "qwen_answer_parts",
            "answer_modified": False,
            "reasoning_modified": False,
            "semantic_correction": False,
        }
    return normalized


def validate_joint_payload_with_format_recovery(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate joint output with deterministic, answer-blind format recovery.

    A shape-valid answer remains authoritative. Recovery from the same Qwen
    response's explicit conclusion is allowed only when the answer field itself
    is structurally invalid. Calculation normalization changes representation
    only (units, percent glyph, separators, and requested decimal precision).
    """

    candidate, normalized = _normalize_calculation_payload(question, payload)
    try:
        shaped = validate_answer_shape_payload(question, candidate)
    except (ValueError, TypeError, KeyError):
        candidate = _recover_answer_field_from_reasoning(question, candidate)
        candidate, normalized_after_recovery = _normalize_calculation_payload(
            question,
            candidate,
        )
        normalized = normalized or normalized_after_recovery
        validated = validate_answer_payload(question, candidate)
        validated["decision_trace"] = {
            "postprocessing_mode": "same_response_contract_recovery",
            "answer_source": "same_qwen_reasoning_conclusion",
            "answer_field_recovered": True,
            "deterministic_format_normalization": normalized,
            "semantic_correction": False,
        }
        return validated

    marker_index = max(
        shaped["reasoning"].rfind("结论："),
        shaped["reasoning"].rfind("结论:"),
    )
    if marker_index < 0:
        shaped["reasoning"] = (
            shaped["reasoning"].rstrip()
            + "\n结论："
            + "；".join(shaped["answer_parts"])
        )
        shaped["decision_trace"] = {
            "postprocessing_mode": "same_response_conclusion_assembly",
            "answer_source": "qwen_answer_parts",
            "reasoning_source": "same_qwen_response",
            "answer_field_recovered": False,
            "deterministic_format_normalization": normalized,
            "semantic_correction": False,
        }
        return shaped
    separator_equivalence = _validate_reasoning_conclusion(
        shaped["answer_parts"], shaped["reasoning"]
    )
    if separator_equivalence:
        shaped["decision_trace"] = {
            "postprocessing_mode": (
                "multi_choice_conclusion_separator_equivalence"
            ),
            "answer_source": "qwen_answer_parts",
            "answer_modified": False,
            "reasoning_modified": False,
            "semantic_correction": False,
        }
    if normalized:
        shaped["decision_trace"] = {
            "postprocessing_mode": "deterministic_format_normalization",
            "answer_source": "qwen_answer_parts",
            "answer_field_recovered": False,
            "deterministic_format_normalization": True,
            "semantic_correction": False,
        }
    return shaped


def validate_answer_shape_payload(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"answer_parts", "reasoning"}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing or extra:
        raise ValueError(f"response fields mismatch: missing={missing}, extra={extra}")

    raw_parts = payload["answer_parts"]
    parts = tuple(validate_answer_parts_shape(question, raw_parts))

    if not isinstance(payload["reasoning"], str):
        raise ValueError("reasoning must be a string")
    reasoning = payload["reasoning"]
    if len(reasoning.strip()) < 20:
        raise ValueError("reasoning must contain at least 20 characters")
    answer = BAnswer(
        qid=question.qid,
        answer_parts=parts,
        reasoning=reasoning,
    )
    validate_b_answer(question, answer)
    return {
        "answer_parts": list(parts),
        "reasoning": reasoning,
    }


def validate_answer_parts_shape(
    question: BQuestion,
    raw_parts: Any,
) -> list[str]:
    if not isinstance(raw_parts, list):
        raise ValueError("answer_parts must be an array")
    parts = tuple(str(item).strip() for item in raw_parts)
    if any(not part for part in parts):
        raise ValueError("answer_parts must contain non-empty strings")
    if question.answer_format not in {"tf", "mcq", "multi"} and any(
        re.search(r"[0-9A-Za-z\u4e00-\u9fff]", part) is None for part in parts
    ):
        raise ValueError(
            "calculation/extraction answer_parts cannot be punctuation-only"
        )
    validate_b_answer(
        question,
        BAnswer(
            qid=question.qid,
            answer_parts=parts,
        ),
    )
    return list(parts)


def validate_frozen_answer_reasoning_payload(
    frozen_answer_parts: Sequence[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if set(payload) != {"reasoning"}:
        raise ValueError(
            "frozen-answer reasoning response must contain only reasoning"
        )
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str) or len(reasoning.strip()) < 20:
        raise ValueError("reasoning must contain at least 20 characters")
    separator_equivalence = _validate_reasoning_conclusion(
        frozen_answer_parts, reasoning
    )
    result = {
        "answer_parts": [str(item) for item in frozen_answer_parts],
        "reasoning": reasoning,
        "decision_trace": {
            "answer_stage": "frozen_from_initial_qwen_response",
            "reasoning_stage": "reasoning_only_retry",
            "postprocessing_mode": (
                "multi_choice_conclusion_separator_equivalence"
                if separator_equivalence
                else "none"
            ),
            "reasoning_assembled_from_model_fields": False,
            "answer_modified": False,
            "reasoning_modified": False,
            "semantic_correction": False,
        },
    }
    return result


def _recover_answer_field_from_reasoning(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError("answer field recovery requires model reasoning")
    marker_index = max(reasoning.rfind("结论："), reasoning.rfind("结论:"))
    if marker_index < 0:
        raise ValueError("answer field recovery requires an explicit conclusion")
    conclusion = re.sub(
        r"[。.!！]+$",
        "",
        reasoning[marker_index + 3 :].strip(),
    )
    if not conclusion:
        raise ValueError("answer field recovery conclusion must not be empty")
    parts = (
        conclusion.split("；")
        if question.answer_slots > 1
        else [conclusion]
    )
    if len(parts) != question.answer_slots or any(not part.strip() for part in parts):
        raise ValueError("answer field recovery slot count mismatch")
    return {
        "answer_parts": [part.strip() for part in parts],
        "reasoning": reasoning,
    }


def _normalize_calculation_payload(
    question: BQuestion,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    candidate = dict(payload)
    if question.answer_format not in {"calculation", "extraction"}:
        return candidate, False
    raw_parts = candidate.get("answer_parts")
    reasoning = candidate.get("reasoning")
    if (
        not isinstance(raw_parts, list)
        or len(raw_parts) != question.answer_slots
        or not all(isinstance(item, str) for item in raw_parts)
        or not isinstance(reasoning, str)
    ):
        return candidate, False
    normalized_parts = [
        _normalize_numeric_slot(question, index, part)
        for index, part in enumerate(raw_parts, start=1)
    ]
    if normalized_parts == raw_parts:
        return candidate, False
    marker_index = max(reasoning.rfind("结论："), reasoning.rfind("结论:"))
    if marker_index < 0:
        return candidate, False
    raw_conclusion = re.sub(
        r"[。.!！]+$",
        "",
        reasoning[marker_index + 3 :].strip(),
    )
    expected_raw = "；".join(raw_parts)
    if _contract_text(raw_conclusion) != _contract_text(expected_raw):
        return candidate, False
    candidate["answer_parts"] = normalized_parts
    candidate["reasoning"] = (
        reasoning[: marker_index + 3] + "；".join(normalized_parts)
    )
    return candidate, True


def _normalize_numeric_slot(
    question: BQuestion,
    slot_index: int,
    value: str,
) -> str:
    template = str(question.answer_slot_templates[slot_index - 1])
    if ">" in template or re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", value.strip()):
        return value
    template_match = re.fullmatch(r"-?(\d+)\.(\d+)(%)?", template)
    if template_match is None:
        return value
    numeric_match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)\s*"
        r"(%|％|个百分点|个?百分点|日|天|倍|元|万元|亿元|万|亿)?\s*",
        value,
    )
    if numeric_match is None:
        return value
    try:
        numeric = Decimal(numeric_match.group(1).replace(",", ""))
    except InvalidOperation:
        return value
    decimal_places = infer_requested_decimal_places(question.question)
    if decimal_places is None:
        decimal_places = len(template_match.group(2))
    quantum = Decimal(1).scaleb(-decimal_places)
    rounded = numeric.quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    percent_requirement = infer_percent_suffix_requirement(
        question.question,
        slot_index=slot_index,
        slot_count=question.answer_slots,
    )
    if percent_requirement is None:
        percent_requirement = bool(template_match.group(3))
    suffix = "%" if percent_requirement else ""
    return f"{rounded:.{decimal_places}f}{suffix}"


def _contract_text(value: str) -> str:
    return re.sub(r"[\s、,，;；|]+", "", str(value)).upper()


def public_run_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def public_run_config_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash immutable run configuration while excluding display-only time."""

    return public_run_fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key != "created_at"
        }
    )


def is_verified_calculation_path(
    calculation_mode: str,
    answer_format: str,
) -> bool:
    return (
        calculation_mode == "verified"
        and answer_format == "calculation"
    )


def suspicious_generation_prompt_literals(tree: ast.AST) -> list[str]:
    """Find answer-like literals in code that can construct system prompts."""

    prompt_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            names = [
                target.id for target in targets
                if isinstance(target, ast.Name)
            ]
            if any("PROMPT" in name.upper() for name in names):
                prompt_nodes.append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lowered = node.name.lower()
            if "prompt" in lowered or "message" in lowered:
                prompt_nodes.append(node)
        elif isinstance(node, ast.Dict):
            pairs = {
                key.value: value
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }
            role = pairs.get("role")
            content = pairs.get("content")
            if (
                isinstance(role, ast.Constant)
                and role.value == "system"
                and content is not None
            ):
                prompt_nodes.append(content)

    strings: list[str] = []
    for prompt_node in prompt_nodes:
        strings.extend(
            str(value.value)
            for value in ast.walk(prompt_node)
            if isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        )

    numeric_pattern = re.compile(
        r"(?<![A-Za-z_])(?:20\d{2}|\d+\.\d+(?:[%％])?)"
    )
    subject_suffix_pattern = re.compile(
        r"[\u4e00-\u9fffA-Za-z0-9]{2,16}"
        r"(?:集团|证券|银行|保险|股份有限公司|集团有限公司)"
    )
    answer_cue_pattern = re.compile(
        r"[\u4e00-\u9fff]{2,16}"
        r"(?=应选|答案为|正确答案|满足条件|不满足条件)"
    )
    generic_markers = (
        "多个公司",
        "各公司",
        "目标公司",
        "上市公司",
        "保险公司",
        "证券公司",
        "集团公司",
        "公司、产品",
        "公司或产品",
        "公司名称",
    )
    generic_exact_subjects = {"最终", "标准", "正确", "冻结"}
    findings: set[str] = set()
    for value in strings:
        findings.update(numeric_pattern.findall(value))
        for subject in [
            *subject_suffix_pattern.findall(value),
            *answer_cue_pattern.findall(value),
        ]:
            if (
                subject not in generic_exact_subjects
                and not any(marker in subject for marker in generic_markers)
            ):
                findings.add(subject)
    return sorted(findings)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _system_prompt(
    question: BQuestion,
    *,
    output_contract: str = "joint",
) -> str:
    if output_contract not in {"joint", "reasoning_canonical"}:
        raise ValueError(f"unsupported output contract: {output_contract}")
    output_fields = (
        "reasoning，禁止附加字段。reasoning末尾的“结论：...”是唯一答案来源，"
        "代码只会原样抽取该结论到提交槽，不会补写、纠错或改写。"
        if output_contract == "reasoning_canonical"
        else "answer_parts与reasoning，禁止附加字段。"
    )
    conclusion_contract = (
        "reasoning中的最终结论必须直接给出最终提交槽：单槽写“结论：<槽1>”，"
        "多槽按顺序写“结论：<槽1>；<槽2>”；不得用“正确/错误”、单位或解释"
        "替代、增补槽内文本。"
        if output_contract == "reasoning_canonical"
        else "reasoning中的最终结论必须机械复制answer_parts：单槽写“结论：<槽1>”，"
        "多槽按顺序写“结论：<槽1>；<槽2>”；不得用“正确/错误”、单位或解释"
        "替代、增补槽内文本。"
    )
    common = (
        "你是金融材料问答模型。只能依据给定题目与检索证据作答，不得使用题号、"
        "历史答案、隐藏标签或外部知识。输出将直接提交：只返回JSON Schema规定的"
        f"{output_fields}reasoning也是最终提交文本，会在"
        "看不到题目、证据和答案的情况下被单独评审，因此必须是简洁、自包含、可审计"
        "的结论摘要：明确主体、年份或适用范围，复述关键事实，说明必要的判断或计算，"
        "并以“结论：...”结束；不得只引用E01等证据编号，也不要输出冗长的隐藏思维过程。"
        f"{conclusion_contract}"
    )
    if question.answer_format in {"tf", "mcq", "multi"}:
        return (
            common
            + "这是选择类问题。应在内部逐项核验主体、时间、指标、范围、否定词和例外条件，"
            "但不要输出逐项布尔标签或中间结构。判断题仅选A或B；单选题仅选一个字母；"
            "多选题必须选择至少两个不同字母并按升序连接。"
        )
    if question.answer_format == "calculation":
        return (
            common
            + "这是计算题。先核对主体、年份、指标口径、单位和取值范围；再写出必要公式、"
            "关键代入值与舍入结果。特别检查汇总值是否已包含分项，避免重复计算，并正确处理"
            "百分比、最大最小值、差额、累计值及排序。严格按槽位格式模板保留小数位、百分号"
            "及分隔符，末尾结论复制格式化后的槽位。"
        )
    return (
        common
        + "这是信息抽取题。核对主体、年份、指标、单位和输出顺序，直接复述证据支持的结果，"
        "严格按题目要求返回全部槽位。"
    )


def _prompt_title(title_path: Sequence[Any]) -> str:
    parts = [str(item).strip() for item in title_path if str(item).strip()]
    if not parts:
        return "无"
    return " / ".join(part[:120] for part in parts[-2:])


def _validate_reasoning_conclusion(
    answer_parts: Sequence[str],
    reasoning: str,
) -> bool:
    marker_index = max(reasoning.rfind("结论："), reasoning.rfind("结论:"))
    if marker_index < 0:
        raise ValueError("reasoning must end with an explicit 结论")
    conclusion = reasoning[marker_index + 3 :]
    if not conclusion:
        raise ValueError("reasoning conclusion must not be empty")
    expected = "；".join(str(part) for part in answer_parts)
    if conclusion == expected:
        return False
    if (
        len(answer_parts) == 1
        and re.fullmatch(r"[A-D]{2,4}", expected)
        and expected == "".join(sorted(set(expected)))
    ):
        compact_conclusion = re.sub(r"[；、， ]+", "", conclusion)
        if (
            compact_conclusion == expected
            and re.fullmatch(r"[A-D](?:[；、， ]+[A-D])+", conclusion)
        ):
            return True
    raise ValueError("reasoning conclusion does not exactly match answer_parts")


def _format_instruction(question: BQuestion) -> str:
    if question.answer_format == "tf":
        return "判断题只填一个字母：A=正确，B=错误。"
    if question.answer_format == "mcq":
        return "单选题只填一个选项字母。"
    if question.answer_format == "multi":
        return "多选题在一个槽内填写至少两个不同字母，按字母升序连接。"
    decimal_places = infer_requested_decimal_places(question.question)
    date_output = bool(
        re.search(
            r"(哪一天|何时|日期|最晚.{0,12}(?:开始|公示|决定)|"
            r"次一工作日)",
            question.question,
        )
    )
    slot_rules: list[str] = []
    for slot_index in range(1, question.answer_slots + 1):
        requirements: list[str] = []
        template = str(question.answer_slot_templates[slot_index - 1])
        if ">" in template:
            requirements.append(
                "按题面要求填写非空名称，并用半角大于号>连接，不得填写数值"
            )
            slot_rules.append(f"槽{slot_index}：" + "，".join(requirements))
            continue
        if date_output and question.answer_slots == 1:
            requirements.append(
                "填写有效中文日期，格式为YYYY年M月D日，不得填写小数"
            )
            slot_rules.append(f"槽{slot_index}：" + "，".join(requirements))
            continue
        template_match = re.fullmatch(r"-?\d+\.(\d+)(%)?", template)
        slot_decimal_places = (
            decimal_places
            if decimal_places is not None
            else len(template_match.group(1))
            if template_match
            else None
        )
        if slot_decimal_places is not None:
            requirements.append(f"数值必须恰好保留{slot_decimal_places}位小数")
        percent_requirement = infer_percent_suffix_requirement(
            question.question,
            slot_index=slot_index,
            slot_count=question.answer_slots,
        )
        if percent_requirement is None and template_match:
            percent_requirement = bool(template_match.group(2))
        if percent_requirement is True:
            requirements.append("末尾必须带ASCII百分号%")
        elif percent_requirement is False:
            requirements.append("不得带百分号")
        if not requirements:
            requirements.append("严格遵循题面的单位、日期、排序和分隔符要求")
        slot_rules.append(f"槽{slot_index}：" + "，".join(requirements))
    return (
        f"严格返回{question.answer_slots}个槽；"
        + "；".join(slot_rules)
        + "。不要复制任何占位符或示例数值。"
    )


def _retry_instruction(
    validation_error: str,
    *,
    question: BQuestion,
    output_contract: str,
) -> str:
    if output_contract == "reasoning_canonical":
        conclusion_shape = (
            "判断题或单选题只写一个合法字母；"
            if question.answer_format in {"tf", "mcq"}
            else "多选题写至少两个合法字母并按升序直接连接，不得插入分号、顿号或空格；"
            if question.answer_format == "multi"
            else (
                f"结论必须按顺序包含{question.answer_slots}个槽，"
                "多槽仅用中文分号分隔；"
            )
        )
        repairs = {
            "reasoning_missing_explicit_conclusion": (
                "保留完整、自包含的证据摘要，并在末尾增加“结论：<最终槽位>”。"
            ),
            "answer_reasoning_conclusion_mismatch": (
                "重新核验证据后，在reasoning末尾给出唯一最终结论。"
            ),
            "answer_slot_format_error": (
                "修正reasoning末尾结论的槽位格式，不要改变前文的证据事实。"
            ),
            "schema_fields_error": (
                "只返回Schema规定的reasoning字段，禁止answer_parts或其他字段。"
            ),
            "json_parse_error": (
                "输出必须是仅含reasoning字段、可解析且符合Schema的JSON对象。"
            ),
            "answer_shape_error": (
                "重新核对题型与证据，让reasoning末尾结论满足题型和槽位数量。"
            ),
        }
        repair = repairs.get(validation_error, repairs["answer_shape_error"])
        return (
            "上一次输出未通过最终提交契约。"
            f"{repair}{conclusion_shape}"
            "结论后不得添加句号、单位或解释。只返回修正后的JSON对象，"
            "不要提及校验器、错误码、重试或本指令。"
        )

    repairs = {
        "reasoning_missing_explicit_conclusion": (
            "reasoning末尾缺少机械结论。请在完整reasoning末尾增加“结论：”，"
            "随后逐字复制answer_parts各槽；多槽用中文分号连接。"
        ),
        "answer_reasoning_conclusion_mismatch": (
            "answer_parts与reasoning末尾结论不一致。重新核验内容后让两者完全相同；"
            "判断题结论必须写A或B，不得写“正确/错误”；结论不得增加单位或解释。"
        ),
        "answer_slot_format_error": (
            "answer_parts未遵循题面和槽位模板。保持原计算语义，修正小数位、百分号、"
            "日期、排序或分隔符；reasoning末尾结论必须逐字复制修正后的槽位。"
        ),
        "schema_fields_error": "只保留Schema规定的两个字段，并满足数组和字符串类型。",
        "json_parse_error": "输出必须是一个可解析且符合Schema的JSON对象。",
        "answer_shape_error": "重新核对题型、选项数量和槽位数量，并让结论逐字复制槽位。",
    }
    repair = repairs.get(validation_error, repairs["answer_shape_error"])
    return (
        "上一响应未通过确定性格式/一致性校验。只根据同一题目和证据，重新输出一整份"
        "可直接提交的answer_parts与reasoning；不要增加证据中不存在的事实。"
        f"\n修复要求：{repair}\n错误码：{validation_error}"
    )


def _normalize_unit_id(unit_id: str) -> str:
    return str(unit_id).replace("__dup2", "").replace("__dup", "")


def _dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
