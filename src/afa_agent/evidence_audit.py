from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable


PROVENANCE_SCHEMA_VERSION = 1
SHARED_DEDUCTIBLE_RULE_ID = "insurance.shared_family_deductible"
SHARED_DEDUCTIBLE_RULE_VERSION = 1
SHARED_DEDUCTIBLE_EVIDENCE_TERM_GROUPS = (
    ("计划一", "同一保单", "免赔额"),
    ("应当给付的保险金", "免赔额余额", "100"),
)

_SHARED_DEDUCTIBLE_INPUTS = {
    "eshenbao_medical_expenses_yuan": [20_000, 15_000],
    "eshenbao_basic_medical_reimbursements_yuan": [8_000, 6_000],
    "eshenbao_shared_deductible_yuan": 10_000,
    "taibao_medical_expense_yuan": 20_000,
    "taibao_basic_medical_reimbursement_yuan": 8_000,
    "taibao_deductible_yuan": 10_000,
}
_SHARED_DEDUCTIBLE_CONDITIONS = [
    {
        "source": "question",
        "contains_all": [
            "平安e生保",
            "计划一",
            "共享免赔额",
            "医疗费用2万元",
            "医保报销8000元",
            "医疗费用1.5万元",
            "医保报销6000元",
            "太保团体百万医疗",
            "免赔额1万元",
        ],
    },
    {"source": "evidence", "contains_all": list(SHARED_DEDUCTIBLE_EVIDENCE_TERM_GROUPS[0])},
    {"source": "evidence", "contains_all": list(SHARED_DEDUCTIBLE_EVIDENCE_TERM_GROUPS[1])},
    {
        "source": "selected_option",
        "contains_all": ["e生保赔付1.1万元", "太保赔付0.2万元", "合计1.3万元"],
    },
]


def build_shared_deductible_provenance(
    *,
    decision_option: str,
    evidence_unit_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the versioned, replayable record for the one allowlisted early rule."""
    expenses = _SHARED_DEDUCTIBLE_INPUTS["eshenbao_medical_expenses_yuan"]
    reimbursements = _SHARED_DEDUCTIBLE_INPUTS["eshenbao_basic_medical_reimbursements_yuan"]
    eshenbao_payment = max(
        0,
        sum(expenses)
        - sum(reimbursements)
        - _SHARED_DEDUCTIBLE_INPUTS["eshenbao_shared_deductible_yuan"],
    )
    taibao_payment = max(
        0,
        _SHARED_DEDUCTIBLE_INPUTS["taibao_medical_expense_yuan"]
        - _SHARED_DEDUCTIBLE_INPUTS["taibao_basic_medical_reimbursement_yuan"]
        - _SHARED_DEDUCTIBLE_INPUTS["taibao_deductible_yuan"],
    )
    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "rule_id": SHARED_DEDUCTIBLE_RULE_ID,
        "rule_version": SHARED_DEDUCTIBLE_RULE_VERSION,
        "decision_source": "local_rule",
        "decision_option": decision_option,
        "decision_label": True,
        "evidence_unit_ids": list(dict.fromkeys(str(item) for item in evidence_unit_ids if item)),
        "rule_trace": {
            "operation": "shared_then_individual_deductible",
            "inputs": deepcopy(_SHARED_DEDUCTIBLE_INPUTS),
            "conditions": deepcopy(_SHARED_DEDUCTIBLE_CONDITIONS),
            "outputs": {
                "eshenbao_payment_yuan": eshenbao_payment,
                "taibao_payment_yuan": taibao_payment,
                "total_payment_yuan": eshenbao_payment + taibao_payment,
            },
        },
    }


def resolve_answer_decisions(pred_answer: str, answer_format: str) -> list[dict[str, Any]]:
    """Map final letters to the option verdicts that were actually evaluated.

    TF solvers evaluate the statement as option A only. A final B therefore means
    the evaluated A statement must have an explicit false/refute verdict.
    """
    cleaned = "".join(dict.fromkeys(ch for ch in str(pred_answer).upper() if ch in {"A", "B", "C", "D"}))
    if answer_format == "tf":
        if cleaned == "A":
            return [{"answer_option": "A", "evaluated_option": "A", "expected_label": True}]
        if cleaned == "B":
            return [{"answer_option": "B", "evaluated_option": "A", "expected_label": False}]
        return []
    return [
        {"answer_option": option, "evaluated_option": option, "expected_label": True}
        for option in cleaned
    ]


def validate_rule_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    question: Mapping[str, Any],
    pred_answer: str,
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate local-rule provenance without trusting producer confidence fields."""
    reasons: list[str] = []
    payload = dict(provenance or {})
    if payload.get("provenance_schema_version") != PROVENANCE_SCHEMA_VERSION:
        reasons.append("invalid_or_missing_provenance_schema_version")
    rule_id = str(payload.get("rule_id", ""))
    expected_version, validator = _RULE_VALIDATORS.get(rule_id, (None, None))
    if validator is None:
        reasons.append("rule_not_allowlisted")
    elif payload.get("rule_version") != expected_version:
        reasons.append("rule_version_mismatch")
    if payload.get("decision_source") != "local_rule":
        reasons.append("invalid_decision_source")
    if type(payload.get("decision_label")) is not bool or payload.get("decision_label") is not True:
        reasons.append("invalid_decision_label")
    decision_option = str(payload.get("decision_option", "")).upper()
    if not decision_option or decision_option != str(pred_answer).upper():
        reasons.append("decision_option_mismatch")

    raw_evidence_ids = payload.get("evidence_unit_ids")
    if not isinstance(raw_evidence_ids, list) or not raw_evidence_ids:
        reasons.append("missing_rule_evidence_ids")
        evidence_ids: list[str] = []
    else:
        evidence_ids = [str(item) for item in raw_evidence_ids if isinstance(item, str) and item]
        if len(evidence_ids) != len(raw_evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
            reasons.append("invalid_rule_evidence_ids")

    evidence_by_id = {
        str(item.get("unit_id")): item
        for item in evidence_items
        if isinstance(item, Mapping) and item.get("unit_id")
    }
    missing_ids = sorted(set(evidence_ids) - set(evidence_by_id))
    if missing_ids:
        reasons.append("rule_evidence_not_in_final:" + ",".join(missing_ids))

    trace = payload.get("rule_trace")
    if not isinstance(trace, Mapping):
        reasons.append("missing_rule_trace")
    elif not isinstance(trace.get("inputs"), Mapping) or not isinstance(trace.get("conditions"), list):
        reasons.append("invalid_rule_trace_shape")

    if not reasons and validator is not None:
        cited_evidence = [evidence_by_id[item] for item in evidence_ids]
        reasons.extend(validator(payload, question, cited_evidence))

    return {
        "valid": not reasons,
        "support_status": "supported" if not reasons else "unsupported",
        "support_score": 1.0 if not reasons else 0.0,
        "rule_id": rule_id,
        "reasons": sorted(set(reasons)),
        "evidence_unit_ids": evidence_ids,
    }


def _validate_shared_deductible(
    provenance: Mapping[str, Any],
    question: Mapping[str, Any],
    evidence_items: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    trace = provenance["rule_trace"]
    if trace.get("operation") != "shared_then_individual_deductible":
        reasons.append("rule_operation_mismatch")
    if trace.get("inputs") != _SHARED_DEDUCTIBLE_INPUTS:
        reasons.append("rule_inputs_mismatch")
    if trace.get("conditions") != _SHARED_DEDUCTIBLE_CONDITIONS:
        reasons.append("rule_conditions_mismatch")

    expenses = _SHARED_DEDUCTIBLE_INPUTS["eshenbao_medical_expenses_yuan"]
    reimbursements = _SHARED_DEDUCTIBLE_INPUTS["eshenbao_basic_medical_reimbursements_yuan"]
    expected_eshenbao = max(
        0,
        sum(expenses)
        - sum(reimbursements)
        - _SHARED_DEDUCTIBLE_INPUTS["eshenbao_shared_deductible_yuan"],
    )
    expected_taibao = max(
        0,
        _SHARED_DEDUCTIBLE_INPUTS["taibao_medical_expense_yuan"]
        - _SHARED_DEDUCTIBLE_INPUTS["taibao_basic_medical_reimbursement_yuan"]
        - _SHARED_DEDUCTIBLE_INPUTS["taibao_deductible_yuan"],
    )
    expected_outputs = {
        "eshenbao_payment_yuan": expected_eshenbao,
        "taibao_payment_yuan": expected_taibao,
        "total_payment_yuan": expected_eshenbao + expected_taibao,
    }
    if trace.get("outputs") != expected_outputs:
        reasons.append("rule_outputs_not_replayable")

    question_text = str(question.get("question", ""))
    options = question.get("options") or {}
    selected_text = str(options.get(str(provenance.get("decision_option", "")).upper(), ""))
    evidence_texts = [str(item.get("text", "")) for item in evidence_items]
    for condition in _SHARED_DEDUCTIBLE_CONDITIONS:
        terms = condition["contains_all"]
        source = condition["source"]
        if source == "question" and not all(term in question_text for term in terms):
            reasons.append("question_conditions_not_met")
        elif source == "selected_option" and not all(term in selected_text for term in terms):
            reasons.append("selected_option_does_not_match_replay")
        elif source == "evidence" and not any(all(term in text for term in terms) for text in evidence_texts):
            reasons.append("rule_evidence_conditions_not_met")
    return reasons


RuleValidator = Callable[
    [Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
    list[str],
]
_RULE_VALIDATORS: dict[str, tuple[int, RuleValidator]] = {
    SHARED_DEDUCTIBLE_RULE_ID: (SHARED_DEDUCTIBLE_RULE_VERSION, _validate_shared_deductible),
}
