from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from afa_agent.client import extract_json_object
from afa_agent.b_board.io import (
    infer_percent_suffix_requirement,
    infer_requested_decimal_places,
    validate_freeform_slot,
)


PROMPT_VERSION = "b_answer_error_judge_v5_multi_min_two"
INDEPENDENT_PROMPT_VERSION = "b_independent_solve_v4_multi_min_two"
BLIND_PROMPT_VERSION = "b_blind_pair_v4_multi_min_two"
SCHEMA_VERSION = 2
HARD_GATE_VERSION = "b_hard_gate_v6_choice_contract"

COMMON_DIMENSIONS = (
    "document_relevance",
    "evidence_sufficiency",
    "citation_alignment",
    "answer_entailment",
    "format_compliance",
    "internal_consistency",
    "overall_confidence",
)
CHOICE_DIMENSION = "alternative_exclusion"
CALCULATION_DIMENSION = "calculation_reproducibility"

_TIER_ORDER = {"blocked": 0, "low": 1, "medium": 2, "high": 3}
_ALLOWED_VERDICTS = {"supported", "partially_supported", "unsupported", "contradicted"}
_ALLOWED_ANSWER_VERDICTS = {"likely_correct", "uncertain", "likely_wrong"}
_ALLOWED_INDEPENDENT_STATUSES = {"resolved", "insufficient_evidence"}
_ALLOWED_OPTION_ASSESSMENTS = {"supported", "contradicted", "insufficient"}
_ALLOWED_ERROR_TYPES = {
    "wrong_document",
    "wrong_subject",
    "wrong_year_or_date",
    "wrong_scope_or_definition",
    "wrong_unit",
    "wrong_formula",
    "wrong_calculation",
    "wrong_or_missing_option",
    "citation_mismatch",
    "format_error",
    "insufficient_evidence",
    "other",
}
_FORBIDDEN_SUBJECT_KEYS = {
    "branch",
    "branch_name",
    "round",
    "round_id",
    "candidate_id",
    "experiment_id",
    "generator_confidence",
    "model_confidence",
    "previous_score",
    "historical_score",
    "generation_prompt",
}


INDEPENDENT_SYSTEM_PROMPT = f"""你是金融长文问答的独立解题员。你看不到现有答案，也不得猜测现有答案。
只根据题目、选项和给定证据独立求解；不得补充外部事实。逐项区分 supported、contradicted、insufficient。
严格检查主体、文件、年份或日期、定义口径、单位、方向、公式和多选完整性。计算题必须重算，不能照抄证据中的结论。
题型为多选题时，最终答案必须包含至少两个不同选项字母；即使只有一个选项证据充分，也不得输出单字母多选答案，应将 status 标为 insufficient_evidence 并给出最合理的合法候选。
答案格式严格按“题目明确要求 > README通用规则 > 提交模板占位”裁决。“不带单位”不等于“不带%”；只有题目明确写“不带%”或“不带百分号”才禁止%。题目未明确禁止时，百分数答案按README必须带%并保留两位小数；“提高若干个百分点”的数值不加%。
证据不足时仍按题目要求给出最佳候选答案，但 status 必须为 insufficient_evidence，并明确缺少什么证据。
只输出一个 JSON 对象，不输出 Markdown。prompt_version={INDEPENDENT_PROMPT_VERSION}, schema_version={SCHEMA_VERSION}。
JSON 字段：schema_version, prompt_version, status, answer_parts, used_evidence_ids,
option_assessments, confidence, solution_summary, missing_evidence。
answer_parts 必须是字符串数组，槽位数量和题目要求一致；used_evidence_ids 只能引用给定 unit_id。
status 只能是 resolved 或 insufficient_evidence；confidence 为 0 到 100 的整数；
选择题答案使用连续大写字母且不加分隔符，例如 ["ACD"]；
option_assessments 的值只能是 supported、contradicted、insufficient。"""

JUDGE_SYSTEM_PROMPT = f"""你是金融长文问答的错题发现审计员。目标是识别封存答案是否可能错误，而不是替它补理由。
你会收到一份在看不到封存答案时产生的独立解题结果，以及封存答案、证据引用和计算轨迹。
比较二者时只使用给定材料，不补充外部事实。独立答案不同只是风险信号，不自动等于封存答案错误；必须依据证据裁决。
严格区分相关性与蕴含，检查错文件、错主体、错年份或日期、错口径、错单位、错公式、错计算、漏选或错选、引用不符和格式错误。
格式裁决必须遵守“题目明确要求 > README通用规则 > 提交模板占位”。“不带单位”不禁止%；只有明确的“不带%”或“不带百分号”才禁止%。题目未明确禁止时，百分数答案按README必须带%并保留两位小数；百分点变化值不加%。
选择题逐项核对选中项与未选项；计算题复核变量、公式、单位和重放结果。证据缺失应标记 uncertain，不得伪造确定结论。
题型为多选题时，封存答案、独立答案及非空 correction_candidate_parts 都必须包含至少两个不同选项字母；单字母候选属于非法答案，不能据此判定封存答案错误。
只输出一个 JSON 对象，不输出 Markdown。prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}。
JSON 字段：schema_version, prompt_version, document_relevance, evidence_sufficiency,
citation_alignment, answer_entailment, alternative_exclusion, calculation_reproducibility,
format_compliance, internal_consistency, overall_confidence, verdict, blocking_reasons,
low_confidence_reasons, suggested_improvements, answer_verdict, error_likelihood,
suspected_error_types, suspected_error_reasons, correction_candidate_parts。
所有适用分数为 0 到 100 的整数；不适用的 alternative_exclusion 或 calculation_reproducibility 填 null。
verdict 只能是 supported、partially_supported、unsupported、contradicted；answer_verdict 只能是
likely_correct、uncertain、likely_wrong；error_likelihood 为封存答案错误的概率式风险分 0 到 100。
suspected_error_types 只能使用：wrong_document、wrong_subject、wrong_year_or_date、
wrong_scope_or_definition、wrong_unit、wrong_formula、wrong_calculation、wrong_or_missing_option、
citation_mismatch、format_error、insufficient_evidence、other；correction_candidate_parts 必须是字符串数组，
没有证据充分的修正答案时输出空数组。"""

BLIND_SYSTEM_PROMPT = f"""你是金融长文问答的盲审裁判。A/B 的来源和新旧身份已隐藏。
只根据各自封存的答案、证据、引用和可重放计算轨迹，选择更受证据支持且更可验证的一方。
出现主体、年份、单位、公式、方向或格式错误时必须拒绝；不得使用外部知识。
题型为多选题时，A/B 答案均必须至少包含两个不同选项字母；单字母多选答案必须拒绝。
格式按“题目明确要求 > README通用规则 > 提交模板占位”；“不带单位”不等于“不带%”，题目未明确禁止时百分数必须带%并保留两位小数，百分点数值不加%。
只输出 JSON：{{"prompt_version":"{BLIND_PROMPT_VERSION}","winner":"A|B|tie","confidence":0-100,"reason":"..."}}。"""


class JudgeClient(Protocol):
    def chat_json(self, messages: list[dict[str, str]]) -> Any: ...


@dataclass(frozen=True, slots=True)
class IndependentSolve:
    status: str
    answer_parts: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    option_assessments: dict[str, str]
    confidence: int
    solution_summary: str
    missing_evidence: tuple[str, ...]
    prompt_version: str = INDEPENDENT_PROMPT_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer_parts": list(self.answer_parts),
            "used_evidence_ids": list(self.used_evidence_ids),
            "option_assessments": self.option_assessments,
            "confidence": self.confidence,
            "solution_summary": self.solution_summary,
            "missing_evidence": list(self.missing_evidence),
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceEvaluation:
    qid: str
    dimensions: dict[str, int | None]
    confidence_score: int
    tier: str
    verdict: str
    blocking_reasons: tuple[str, ...]
    low_confidence_reasons: tuple[str, ...]
    suggested_improvements: tuple[str, ...]
    hard_failures: tuple[str, ...]
    independent_status: str = ""
    independent_answer_parts: tuple[str, ...] = ()
    independent_used_evidence_ids: tuple[str, ...] = ()
    independent_option_assessments: dict[str, str] = field(default_factory=dict)
    independent_confidence: int | None = None
    independent_solution_summary: str = ""
    independent_missing_evidence: tuple[str, ...] = ()
    answer_match: bool | None = None
    answer_verdict: str = ""
    error_likelihood: int | None = None
    suspected_error: bool = False
    suspected_error_types: tuple[str, ...] = ()
    suspected_error_reasons: tuple[str, ...] = ()
    correction_candidate_parts: tuple[str, ...] = ()
    prompt_version: str = PROMPT_VERSION
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "dimensions": self.dimensions,
            "confidence_score": self.confidence_score,
            "tier": self.tier,
            "verdict": self.verdict,
            "blocking_reasons": list(self.blocking_reasons),
            "low_confidence_reasons": list(self.low_confidence_reasons),
            "suggested_improvements": list(self.suggested_improvements),
            "hard_failures": list(self.hard_failures),
            "independent_status": self.independent_status,
            "independent_answer_parts": list(self.independent_answer_parts),
            "independent_used_evidence_ids": list(self.independent_used_evidence_ids),
            "independent_option_assessments": self.independent_option_assessments,
            "independent_confidence": self.independent_confidence,
            "independent_solution_summary": self.independent_solution_summary,
            "independent_missing_evidence": list(self.independent_missing_evidence),
            "answer_match": self.answer_match,
            "answer_verdict": self.answer_verdict,
            "error_likelihood": self.error_likelihood,
            "suspected_error": self.suspected_error,
            "suspected_error_types": list(self.suspected_error_types),
            "suspected_error_reasons": list(self.suspected_error_reasons),
            "correction_candidate_parts": list(self.correction_candidate_parts),
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class BlindPair:
    public_payload: dict[str, Any]
    candidate_label: str
    incumbent_label: str


@dataclass(frozen=True, slots=True)
class BlindPairEvaluation:
    winner: str
    confidence: int
    reason: str
    prompt_version: str = BLIND_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "confidence": self.confidence,
            "reason": self.reason,
            "prompt_version": self.prompt_version,
        }


def prompt_fingerprint() -> str:
    payload = (
        f"{PROMPT_VERSION}\n{SCHEMA_VERSION}\n{HARD_GATE_VERSION}\n"
        f"{INDEPENDENT_PROMPT_VERSION}\n{INDEPENDENT_SYSTEM_PROMPT}\n{JUDGE_SYSTEM_PROMPT}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def blind_prompt_fingerprint() -> str:
    payload = f"{BLIND_PROMPT_VERSION}\n{BLIND_SYSTEM_PROMPT}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_subject(subject: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only inference artifacts and recursively reject optimizer metadata."""

    forbidden = sorted(str(key) for key in subject if str(key).lower() in _FORBIDDEN_SUBJECT_KEYS)
    if forbidden:
        raise ValueError("Evaluator subject contains optimizer metadata: " + ", ".join(forbidden))
    allowed = {
        "qid",
        "domain",
        "type",
        "answer_format",
        "question",
        "options",
        "answer_slot_count",
        "answer_slot_templates",
        "answer_parts",
        "used_evidence_ids",
        "evidence_items",
        "decision_trace",
        "calculation_trace",
        "token_usage",
    }
    return {key: subject[key] for key in allowed if key in subject}


def build_independent_messages(subject: Mapping[str, Any]) -> list[dict[str, str]]:
    clean = sanitize_subject(subject)
    independent_input = {
        key: clean[key]
        for key in (
            "qid",
            "domain",
            "type",
            "answer_format",
            "question",
            "options",
            "answer_slot_count",
            "answer_slot_templates",
            "evidence_items",
        )
        if key in clean
    }
    return [
        {"role": "system", "content": INDEPENDENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "请在不知道任何现有答案的前提下独立求解：\n"
            + json.dumps(independent_input, ensure_ascii=False, sort_keys=True),
        },
    ]


def build_evaluation_messages(
    subject: Mapping[str, Any], independent: IndependentSolve
) -> list[dict[str, str]]:
    clean = sanitize_subject(subject)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "请比较独立解题结果与封存答案并进行错题审计：\n"
            + json.dumps(
                {"independent_solve": independent.to_dict(), "sealed_subject": clean},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def detect_hard_failures(subject: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    parts = subject.get("answer_parts")
    if not isinstance(parts, list) or not parts or any(not str(item).strip() for item in parts):
        failures.append("empty_or_invalid_answer_parts")
    expected_slots = int(subject.get("answer_slot_count") or len(parts or []) or 0)
    if not isinstance(parts, list) or len(parts) != expected_slots:
        failures.append("answer_slot_count_mismatch")
    templates = subject.get("answer_slot_templates")
    question_type = str(subject.get("answer_format") or subject.get("type") or "")
    question_text = str(subject.get("question") or "")
    if isinstance(parts, list):
        try:
            _validate_choice_answer_parts(
                tuple(str(item) for item in parts),
                question_type=question_type,
                field_name="sealed answer_parts",
            )
        except ValueError:
            failures.append("invalid_choice_answer_contract")
    if question_type in {"calculation", "freeform", "extraction", "计算题", "抽取题"}:
        if not isinstance(templates, list) or len(templates) != expected_slots:
            failures.append("answer_slot_templates_missing_or_mismatched")
        elif isinstance(parts, list) and len(parts) == len(templates):
            for index, (part, template) in enumerate(zip(parts, templates), start=1):
                try:
                    validate_freeform_slot(
                        str(part),
                        str(template),
                        numeric_decimal_places=infer_requested_decimal_places(question_text),
                        percent_suffix=infer_percent_suffix_requirement(
                            question_text,
                            slot_index=index,
                            slot_count=expected_slots,
                        ),
                    )
                except ValueError:
                    failures.append(f"invalid_answer_slot:{index}")

    evidence_items = subject.get("evidence_items")
    used_ids = subject.get("used_evidence_ids")
    if not isinstance(evidence_items, list) or not evidence_items:
        failures.append("empty_final_evidence")
    if not isinstance(used_ids, list) or not used_ids:
        failures.append("missing_used_evidence_ids")
    else:
        available = {
            str(item.get("unit_id"))
            for item in (evidence_items or [])
            if isinstance(item, Mapping) and item.get("unit_id") is not None
        }
        missing = sorted(str(item) for item in used_ids if str(item) not in available)
        if missing:
            failures.append("used_evidence_not_in_final:" + ",".join(missing))

    token_usage = subject.get("token_usage") or {}
    try:
        prompt = int(token_usage.get("prompt_tokens", 0))
        completion = int(token_usage.get("completion_tokens", 0))
        total = int(token_usage.get("total_tokens", 0))
        if min(prompt, completion, total) < 0 or prompt + completion != total:
            failures.append("invalid_token_usage")
    except (AttributeError, TypeError, ValueError):
        failures.append("invalid_token_usage")

    trace = subject.get("decision_trace") or {}
    if isinstance(trace, Mapping):
        for flag in ("format_forced", "invalid_model_answer", "no_supported_fallback", "default_answer"):
            if trace.get(flag):
                failures.append(flag)

    if question_type in {"calculation", "freeform", "计算题"}:
        calculation_trace = subject.get("calculation_trace")
        if not isinstance(calculation_trace, Mapping) or not calculation_trace.get("replay_verified"):
            failures.append("calculation_not_replay_verified")
        if not isinstance(calculation_trace, Mapping) or not calculation_trace.get(
            "grounding_verified"
        ):
            failures.append("calculation_not_grounding_verified")
    return sorted(set(failures))


def parse_evaluation_payload(
    *,
    qid: str,
    question_type: str,
    payload: Mapping[str, Any],
    sealed_answer_parts: Sequence[str] = (),
    independent: IndependentSolve | None = None,
    hard_failures: Sequence[str] = (),
) -> ConfidenceEvaluation:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Evaluator schema_version mismatch")
    if payload.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Evaluator prompt_version mismatch")

    dimensions: dict[str, int | None] = {}
    for key in (*COMMON_DIMENSIONS, CHOICE_DIMENSION, CALCULATION_DIMENSION):
        value = payload.get(key)
        if value is None and key in {CHOICE_DIMENSION, CALCULATION_DIMENSION}:
            dimensions[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise ValueError(f"Invalid evaluator dimension {key}: {value!r}")
        dimensions[key] = int(round(float(value)))

    verdict = str(payload.get("verdict", ""))
    if verdict not in _ALLOWED_VERDICTS:
        raise ValueError(f"Invalid evaluator verdict: {verdict!r}")
    answer_verdict = str(payload.get("answer_verdict", ""))
    if answer_verdict not in _ALLOWED_ANSWER_VERDICTS:
        raise ValueError(f"Invalid evaluator answer_verdict: {answer_verdict!r}")
    error_likelihood = _bounded_integer(payload.get("error_likelihood"), "error_likelihood")
    suspected_error_types = _enum_tuple(
        payload.get("suspected_error_types"),
        allowed=_ALLOWED_ERROR_TYPES,
        field_name="suspected_error_types",
    )
    suspected_error_reasons = _string_tuple(payload.get("suspected_error_reasons"))
    correction_candidate_parts = _answer_parts_tuple(
        payload.get("correction_candidate_parts"),
        field_name="correction_candidate_parts",
        allow_empty=True,
    )
    if correction_candidate_parts:
        _validate_choice_answer_parts(
            correction_candidate_parts,
            question_type=question_type,
            field_name="correction_candidate_parts",
        )

    relevant = [int(dimensions[key]) for key in COMMON_DIMENSIONS]
    if question_type in {"calculation", "freeform", "计算题"}:
        extra = dimensions[CALCULATION_DIMENSION]
    else:
        extra = dimensions[CHOICE_DIMENSION]
    if extra is None and hard_failures:
        # A judge may mark reproducibility/exclusion as inapplicable when the
        # deterministic gate has already proved the answer structurally invalid.
        # Preserve complete coverage conservatively instead of dropping the qid.
        dimensions[
            CALCULATION_DIMENSION
            if question_type in {"calculation", "freeform", "计算题"}
            else CHOICE_DIMENSION
        ] = 0
        extra = 0
    if extra is None:
        raise ValueError("Evaluator omitted the applicable decision dimension")
    relevant.append(int(extra))

    combined_hard_failures = tuple(sorted(set(str(item) for item in hard_failures if str(item))))
    score = min(relevant)
    if combined_hard_failures or verdict == "contradicted":
        score = 0
    if answer_verdict == "likely_wrong":
        score = min(score, 39)
    elif answer_verdict == "uncertain":
        score = min(score, 59)
    if independent is not None and independent.status == "insufficient_evidence":
        score = min(score, 59)
    tier = confidence_tier(score)
    independent_answer_parts = independent.answer_parts if independent is not None else ()
    answer_match = (
        _normalized_answer_parts(question_type, independent_answer_parts)
        == _normalized_answer_parts(question_type, sealed_answer_parts)
        if independent is not None
        else None
    )
    low_confidence_reasons = list(_string_tuple(payload.get("low_confidence_reasons")))
    if answer_verdict != "likely_correct":
        low_confidence_reasons.extend(suspected_error_reasons)
    return ConfidenceEvaluation(
        qid=qid,
        dimensions=dimensions,
        confidence_score=score,
        tier=tier,
        verdict=verdict,
        blocking_reasons=_string_tuple(payload.get("blocking_reasons")),
        low_confidence_reasons=tuple(dict.fromkeys(low_confidence_reasons)),
        suggested_improvements=_string_tuple(payload.get("suggested_improvements")),
        hard_failures=combined_hard_failures,
        independent_status=independent.status if independent is not None else "",
        independent_answer_parts=independent_answer_parts,
        independent_used_evidence_ids=(
            independent.used_evidence_ids if independent is not None else ()
        ),
        independent_option_assessments=(
            independent.option_assessments if independent is not None else {}
        ),
        independent_confidence=independent.confidence if independent is not None else None,
        independent_solution_summary=(
            independent.solution_summary if independent is not None else ""
        ),
        independent_missing_evidence=(
            independent.missing_evidence if independent is not None else ()
        ),
        answer_match=answer_match,
        answer_verdict=answer_verdict,
        error_likelihood=error_likelihood,
        suspected_error=answer_verdict == "likely_wrong",
        suspected_error_types=suspected_error_types,
        suspected_error_reasons=suspected_error_reasons,
        correction_candidate_parts=correction_candidate_parts,
    )


def parse_independent_payload(
    payload: Mapping[str, Any], *, subject: Mapping[str, Any]
) -> IndependentSolve:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Independent evaluator schema_version mismatch")
    if payload.get("prompt_version") != INDEPENDENT_PROMPT_VERSION:
        raise ValueError("Independent evaluator prompt_version mismatch")
    raw_status = str(payload.get("status", ""))
    status = {
        "supported": "resolved",
        "uncertain": "insufficient_evidence",
        "insufficient": "insufficient_evidence",
    }.get(raw_status, raw_status)
    if status not in _ALLOWED_INDEPENDENT_STATUSES:
        raise ValueError(f"Invalid independent status: {status!r}")
    answer_parts = _answer_parts_tuple(payload.get("answer_parts"), field_name="answer_parts")
    expected_slots = int(subject.get("answer_slot_count") or 0)
    if len(answer_parts) != expected_slots:
        raise ValueError(
            f"Independent answer slot count mismatch: expected {expected_slots}, got {len(answer_parts)}"
        )
    _validate_choice_answer_parts(
        answer_parts,
        question_type=str(subject.get("answer_format") or subject.get("type") or ""),
        field_name="independent answer_parts",
    )
    evidence_items = subject.get("evidence_items") or []
    available_evidence_ids = {
        str(item.get("unit_id"))
        for item in evidence_items
        if isinstance(item, Mapping) and item.get("unit_id") is not None
    }
    used_evidence_ids = _string_tuple(payload.get("used_evidence_ids"))
    unknown_ids = sorted(set(used_evidence_ids) - available_evidence_ids)
    if unknown_ids:
        raise ValueError(
            "Independent evaluator cited unknown evidence ids: " + ", ".join(unknown_ids)
        )
    raw_assessments = payload.get("option_assessments")
    if not isinstance(raw_assessments, Mapping):
        raise ValueError("Independent option_assessments must be an object")
    option_assessments: dict[str, str] = {}
    for key, value in raw_assessments.items():
        assessment = str(value)
        if assessment not in _ALLOWED_OPTION_ASSESSMENTS:
            raise ValueError(f"Invalid option assessment for {key}: {assessment!r}")
        option_assessments[str(key)] = assessment
    expected_options = set(str(key) for key in (subject.get("options") or {}))
    if expected_options and set(option_assessments) != expected_options:
        raise ValueError(
            "Independent option assessment coverage mismatch: "
            f"expected={sorted(expected_options)}, got={sorted(option_assessments)}"
        )
    return IndependentSolve(
        status=status,
        answer_parts=answer_parts,
        used_evidence_ids=used_evidence_ids,
        option_assessments=option_assessments,
        confidence=_bounded_integer(payload.get("confidence"), "confidence"),
        solution_summary=str(payload.get("solution_summary", "")).strip(),
        missing_evidence=_string_tuple(payload.get("missing_evidence")),
    )


def confidence_tier(score: int) -> str:
    if score < 40:
        return "blocked"
    if score < 60:
        return "low"
    if score < 80:
        return "medium"
    return "high"


class FixedConfidenceEvaluator:
    def __init__(self, client: JudgeClient):
        self.client = client

    def evaluate(self, subject: Mapping[str, Any]) -> tuple[ConfidenceEvaluation, dict[str, int]]:
        clean = sanitize_subject(subject)
        independent_response = self.client.chat_json(build_independent_messages(clean))
        independent = parse_independent_payload(
            extract_json_object(independent_response.content),
            subject=clean,
        )
        response = self.client.chat_json(build_evaluation_messages(clean, independent))
        payload = extract_json_object(response.content)
        question_type = str(clean.get("answer_format") or clean.get("type") or "")
        evaluation = parse_evaluation_payload(
            qid=str(clean.get("qid", "")),
            question_type=question_type,
            payload=payload,
            sealed_answer_parts=tuple(str(item) for item in clean.get("answer_parts", [])),
            independent=independent,
            hard_failures=detect_hard_failures(clean),
        )
        first_usage = independent_response.token_usage.to_dict()
        second_usage = response.token_usage.to_dict()
        usage = {
            key: int(first_usage.get(key, 0)) + int(second_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return evaluation, usage


class FixedBlindPairEvaluator:
    def __init__(self, client: JudgeClient):
        self.client = client

    def evaluate(self, pair: BlindPair) -> tuple[BlindPairEvaluation, dict[str, int]]:
        response = self.client.chat_json(
            [
                {"role": "system", "content": BLIND_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(pair.public_payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        payload = extract_json_object(response.content)
        if payload.get("prompt_version") != BLIND_PROMPT_VERSION:
            raise ValueError("Blind evaluator prompt_version mismatch")
        winner = str(payload.get("winner", ""))
        if winner not in {"A", "B", "tie"}:
            raise ValueError(f"Invalid blind winner: {winner!r}")
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Blind evaluator confidence must be numeric")
        score = int(round(float(confidence)))
        if not 0 <= score <= 100:
            raise ValueError("Blind evaluator confidence is outside 0..100")
        result = BlindPairEvaluation(
            winner=winner,
            confidence=score,
            reason=str(payload.get("reason", "")).strip(),
        )
        usage = response.token_usage.to_dict()
        return result, {key: int(value) for key, value in usage.items()}


def build_blind_pair(
    *,
    qid: str,
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    salt: str,
) -> BlindPair:
    seed_text = f"{qid}:{salt}:{_stable_json(incumbent)}:{_stable_json(candidate)}"
    rng = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16))
    candidate_first = bool(rng.getrandbits(1))
    if candidate_first:
        answers = {"A": sanitize_subject(candidate), "B": sanitize_subject(incumbent)}
        candidate_label, incumbent_label = "A", "B"
    else:
        answers = {"A": sanitize_subject(incumbent), "B": sanitize_subject(candidate)}
        candidate_label, incumbent_label = "B", "A"
    public = {"qid": qid, "answers": answers, "instruction": "比较A/B，选择证据支持更充分且可验证的答案。"}
    return BlindPair(public_payload=public, candidate_label=candidate_label, incumbent_label=incumbent_label)


def decide_candidate_promotion(
    *,
    incumbent: ConfidenceEvaluation,
    candidate: ConfidenceEvaluation,
    answer_changed: bool,
    blind_winner: str | None = None,
    candidate_blind_label: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if candidate.hard_failures:
        reasons.append("candidate_hard_failure")
    if candidate.verdict == "contradicted":
        reasons.append("candidate_contradicted")
    tier_improved = _TIER_ORDER[candidate.tier] > _TIER_ORDER[incumbent.tier]
    score_improved = candidate.confidence_score >= incumbent.confidence_score + 5
    if not (tier_improved or score_improved):
        reasons.append("insufficient_confidence_gain")

    for key, incumbent_value in incumbent.dimensions.items():
        candidate_value = candidate.dimensions.get(key)
        if incumbent_value is None or candidate_value is None:
            continue
        if candidate_value < incumbent_value - 5:
            reasons.append(f"dimension_regressed:{key}")

    if answer_changed:
        if not tier_improved:
            reasons.append("changed_answer_without_tier_gain")
        if not blind_winner or blind_winner != candidate_blind_label:
            reasons.append("blind_pair_did_not_prefer_candidate")
    return {
        "promote": not reasons,
        "reasons": sorted(set(reasons)),
        "score_delta": candidate.confidence_score - incumbent.confidence_score,
        "tier_before": incumbent.tier,
        "tier_after": candidate.tier,
    }


def build_calibration_sentinels() -> list[dict[str, Any]]:
    return [
        {"sentinel_id": "wrong_year", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
        {"sentinel_id": "wrong_unit", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
        {"sentinel_id": "wrong_arithmetic", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
        {"sentinel_id": "irrelevant_evidence", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
        {"sentinel_id": "missing_citation", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
        {"sentinel_id": "format_only", "expected_max_score": 39, "expected_verdicts": ["unsupported", "contradicted"]},
    ]


def build_calibration_subjects() -> list[dict[str, Any]]:
    """Known-bad, structurally valid cases used to detect a permissive judge."""

    def base(sentinel_id: str, *, question: str, answer: str, evidence: str) -> dict[str, Any]:
        unit_id = f"sentinel:{sentinel_id}:e1"
        return {
            "qid": sentinel_id,
            "domain": "calibration",
            "type": "单选题",
            "answer_format": "mcq",
            "question": question,
            "options": {"A": answer, "B": "其他"},
            "answer_slot_count": 1,
            "answer_slot_templates": ["A"],
            "answer_parts": ["A"],
            "used_evidence_ids": [unit_id],
            "evidence_items": [{"unit_id": unit_id, "doc_id": "sentinel", "text": evidence}],
            "decision_trace": {},
            "calculation_trace": {},
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    subjects = [
        base(
            "wrong_year",
            question="根据2025年报告，营业收入是否为100亿元？",
            answer="是",
            evidence="2024年营业收入为100亿元。",
        ),
        base(
            "wrong_unit",
            question="该金额是否为100亿元？",
            answer="是",
            evidence="该金额为100万元。",
        ),
        base(
            "irrelevant_evidence",
            question="资产负债率是否为50%？",
            answer="是",
            evidence="公司注册地址位于北京市。",
        ),
        base(
            "missing_citation",
            question="净利润是否同比增长20%？",
            answer="是",
            evidence="本段没有净利润、同比增速或相关数值。",
        ),
        base(
            "format_only",
            question="证据是否明确支持选项A？",
            answer="选项A",
            evidence="条款明确否定选项A，并支持选项B。",
        ),
    ]
    wrong_arithmetic = base(
        "wrong_arithmetic",
        question="已知10与20，二者之和是多少？",
        answer="31.00",
        evidence="输入值分别为10和20。",
    )
    wrong_arithmetic.update(
        {
            "type": "计算题",
            "answer_format": "calculation",
            "options": {},
            "answer_parts": ["31.00"],
            "answer_slot_templates": ["999999.99"],
            "calculation_trace": {
                "variables": [{"name": "a", "value": "10"}, {"name": "b", "value": "20"}],
                "steps": [{"id": "sum", "op": "add", "result": "31"}],
                "replay_verified": True,
                "grounding_verified": True,
            },
        }
    )
    subjects.insert(2, wrong_arithmetic)
    return subjects


def validate_calibration_sentinels(
    evaluations: Mapping[str, ConfidenceEvaluation],
) -> dict[str, Any]:
    failures: list[str] = []
    for sentinel in build_calibration_sentinels():
        sentinel_id = sentinel["sentinel_id"]
        evaluation = evaluations.get(sentinel_id)
        if evaluation is None:
            failures.append(f"missing:{sentinel_id}")
            continue
        if evaluation.confidence_score > sentinel["expected_max_score"]:
            failures.append(f"score_too_high:{sentinel_id}")
        if evaluation.verdict not in sentinel["expected_verdicts"]:
            failures.append(f"verdict_not_rejected:{sentinel_id}")
    return {"passed": not failures, "failures": failures, "sentinel_count": len(build_calibration_sentinels())}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _answer_parts_tuple(
    value: Any, *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise ValueError(f"{field_name} must be {qualifier}")
    parts = tuple(str(item).strip() for item in value)
    if any(not item for item in parts):
        raise ValueError(f"{field_name} contains an empty answer slot")
    return parts


def _validate_choice_answer_parts(
    parts: Sequence[str], *, question_type: str, field_name: str
) -> None:
    if question_type not in {"multi", "多选题", "多选"}:
        return
    if len(parts) != 1:
        raise ValueError(f"{field_name}: multi-choice answer must use one slot")
    answer = str(parts[0])
    if not 2 <= len(answer) <= 4 or not answer.isalpha() or answer != answer.upper():
        raise ValueError(
            f"{field_name}: multi-choice answer must contain at least two uppercase letters"
        )
    if len(set(answer)) != len(answer) or answer != "".join(sorted(answer)):
        raise ValueError(
            f"{field_name}: multi-choice answer letters must be unique and sorted"
        )


def _bounded_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = int(round(float(value)))
    if not 0 <= result <= 100:
        raise ValueError(f"{field_name} is outside 0..100")
    return result


def _enum_tuple(value: Any, *, allowed: set[str], field_name: str) -> tuple[str, ...]:
    items = _string_tuple(value)
    invalid = sorted(set(items) - allowed)
    if invalid:
        raise ValueError(f"Invalid {field_name}: {invalid}")
    return items


def _normalized_answer_parts(question_type: str, parts: Sequence[str]) -> tuple[str, ...]:
    normalized = [str(item).strip() for item in parts]
    if question_type not in {"calculation", "freeform", "extraction", "计算题", "抽取题"}:
        normalized = ["".join(sorted(set(char for char in item.upper() if char in "ABCD"))) for item in normalized]
    return tuple(normalized)


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(sanitize_subject(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
