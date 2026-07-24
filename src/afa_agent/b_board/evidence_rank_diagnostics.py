from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from afa_agent.models import RetrievalHit


_CONTAMINATED_KEYS = frozenset({"rule_label", "targeted_literal"})
_DUPLICATE_SUFFIX_RE = re.compile(r"__dup(?:\d+)?$")


def diagnose_historical_evidence_ranks(
    cases: Sequence[Mapping[str, Any]],
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate historical evidence as offline positives, never runtime routing.

    The returned report intentionally omits positive document and chunk
    identifiers.  Historical identifiers are used transiently to compute rank
    statistics and cannot be consumed as an online question-to-evidence map.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    case_reports: list[dict[str, Any]] = []
    reciprocal_rank_sum = 0.0
    hit_at_k_count = 0
    evaluated_case_count = 0
    eligible_positive_count = 0
    matched_positive_at_k_count = 0
    filtered_positive_count = 0

    for ordinal, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"case-{ordinal}"))
        ranked_hits = _mapping_sequence(case.get("ranked_hits", []))
        historical_evidence = _mapping_sequence(case.get("historical_evidence", []))
        eligible_positives: list[Mapping[str, Any]] = []
        filtered_for_case = 0
        for positive in historical_evidence:
            if _contains_contamination_marker(positive):
                filtered_for_case += 1
                continue
            if _evidence_key(positive) is not None:
                eligible_positives.append(positive)

        filtered_positive_count += filtered_for_case
        eligible_keys = {
            key
            for positive in eligible_positives
            if (key := _evidence_key(positive)) is not None
        }
        target_docs = {
            str(positive.get("doc_id", "")).strip()
            for positive in eligible_positives
            if str(positive.get("doc_id", "")).strip()
        }
        ranked_key_sets = [_evidence_keys(hit) for hit in ranked_hits]
        matching_ranks = [
            rank
            for rank, keys in enumerate(ranked_key_sets, start=1)
            if keys & eligible_keys
        ]
        best_rank = min(matching_ranks) if matching_ranks else None
        matched_at_k = len(
            eligible_keys.intersection(
                set().union(*ranked_key_sets[:k]) if ranked_key_sets[:k] else set()
            )
        )

        if eligible_keys:
            evaluated_case_count += 1
            eligible_positive_count += len(eligible_keys)
            matched_positive_at_k_count += matched_at_k
            reciprocal_rank = 1.0 / best_rank if best_rank is not None else 0.0
            reciprocal_rank_sum += reciprocal_rank
            hit_at_k = best_rank is not None and best_rank <= k
            hit_at_k_count += int(hit_at_k)
            gap = _classify_gap(
                best_rank=best_rank,
                k=k,
                target_docs=target_docs,
                ranked_hits=ranked_hits,
            )
            positive_recall_at_k = matched_at_k / len(eligible_keys)
        else:
            reciprocal_rank = None
            hit_at_k = None
            gap = "no_eligible_positive"
            positive_recall_at_k = None

        case_reports.append(
            {
                "case_id": case_id,
                "eligible_positive_count": len(eligible_keys),
                "filtered_positive_count": filtered_for_case,
                "best_rank": best_rank,
                "reciprocal_rank": reciprocal_rank,
                "hit_at_k": hit_at_k,
                f"positive_recall_at_{k}": positive_recall_at_k,
                "gap": gap,
            }
        )

    return {
        "offline_only": True,
        "k": k,
        "case_count": len(cases),
        "evaluated_case_count": evaluated_case_count,
        "eligible_positive_count": eligible_positive_count,
        "filtered_positive_count": filtered_positive_count,
        "mrr": (
            reciprocal_rank_sum / evaluated_case_count
            if evaluated_case_count
            else 0.0
        ),
        f"recall_at_{k}": (
            hit_at_k_count / evaluated_case_count
            if evaluated_case_count
            else 0.0
        ),
        f"positive_recall_at_{k}": (
            matched_positive_at_k_count / eligible_positive_count
            if eligible_positive_count
            else 0.0
        ),
        "cases": case_reports,
    }


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, RetrievalHit):
            result.append(item.to_dict())
        elif isinstance(item, Mapping):
            result.append(item)
    return result


def _contains_contamination_marker(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, child in value.items():
        if str(key) in _CONTAMINATED_KEYS:
            return True
        if isinstance(child, Mapping) and _contains_contamination_marker(child):
            return True
        if (
            isinstance(child, Sequence)
            and not isinstance(child, (str, bytes))
            and any(_contains_contamination_marker(item) for item in child)
        ):
            return True
    return False


def _evidence_key(evidence: Mapping[str, Any]) -> tuple[str, str] | None:
    doc_id = str(evidence.get("doc_id", "")).strip()
    unit_id = str(evidence.get("unit_id", "")).strip()
    if not doc_id or not unit_id:
        return None
    return doc_id, _DUPLICATE_SUFFIX_RE.sub("", unit_id)


def _evidence_keys(evidence: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if key := _evidence_key(evidence):
        keys.add(key)
    for component in _mapping_sequence(evidence.get("merged_from", [])):
        if key := _evidence_key(component):
            keys.add(key)
    return keys


def _classify_gap(
    *,
    best_rank: int | None,
    k: int,
    target_docs: set[str],
    ranked_hits: Sequence[Mapping[str, Any]],
) -> str:
    if best_rank is not None:
        return "retrieved_at_k" if best_rank <= k else "ranking_gap"
    retrieved_docs = {
        str(hit.get("doc_id", "")).strip()
        for hit in ranked_hits
        if str(hit.get("doc_id", "")).strip()
    }
    if target_docs & retrieved_docs:
        return "chunk_recall_gap"
    return "document_recall_gap"
