from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from afa_agent.client import extract_json_object
from afa_agent.b_board.io import validate_freeform_slot


PROMPT_VERSION = "b_confidence_judge_v1"
BLIND_PROMPT_VERSION = "b_blind_pair_v1"
SCHEMA_VERSION = 1
HARD_GATE_VERSION = "b_hard_gate_v2"

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


JUDGE_SYSTEM_PROMPT = f"""你是金融长文问答的独立证据审计员。你只评估答案是否被给定证据支持，不补充外部事实。
严格区分相关性与蕴含：出现相同关键词不等于支持答案。检查主体、年份、口径、单位、方向、公式和答案格式。
选择题必须检查选中项以及最强竞争项；计算题必须检查变量、公式、单位和重放结果。
只输出一个 JSON 对象，不输出 Markdown。prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}。
JSON 字段：schema_version, prompt_version, document_relevance, evidence_sufficiency,
citation_alignment, answer_entailment, alternative_exclusion, calculation_reproducibility,
format_compliance, internal_consistency, overall_confidence, verdict, blocking_reasons,
low_confidence_reasons, suggested_improvements。
所有适用分数为 0 到 100 的整数；不适用的 alternative_exclusion 或 calculation_reproducibility 填 null。
verdict 只能是 supported、partially_supported、unsupported、contradicted。"""

BLIND_SYSTEM_PROMPT = f"""你是金融长文问答的盲审裁判。A/B 的来源和新旧身份已隐藏。
只根据各自封存的答案、证据、引用和可重放计算轨迹，选择更受证据支持且更可验证的一方。
出现主体、年份、单位、公式、方向或格式错误时必须拒绝；不得使用外部知识。
只输出 JSON：{{"prompt_version":"{BLIND_PROMPT_VERSION}","winner":"A|B|tie","confidence":0-100,"reason":"..."}}。"""


class JudgeClient(Protocol):
    def chat_json(self, messages: list[dict[str, str]]) -> Any: ...


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
        f"{JUDGE_SYSTEM_PROMPT}"
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


def build_evaluation_messages(subject: Mapping[str, Any]) -> list[dict[str, str]]:
    clean = sanitize_subject(subject)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "请审计以下封存答案。只根据给定证据判断：\n"
            + json.dumps(clean, ensure_ascii=False, sort_keys=True),
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
    if question_type in {"calculation", "freeform", "extraction", "计算题", "抽取题"}:
        if not isinstance(templates, list) or len(templates) != expected_slots:
            failures.append("answer_slot_templates_missing_or_mismatched")
        elif isinstance(parts, list) and len(parts) == len(templates):
            for index, (part, template) in enumerate(zip(parts, templates), start=1):
                try:
                    validate_freeform_slot(str(part), str(template))
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
    tier = confidence_tier(score)
    return ConfidenceEvaluation(
        qid=qid,
        dimensions=dimensions,
        confidence_score=score,
        tier=tier,
        verdict=verdict,
        blocking_reasons=_string_tuple(payload.get("blocking_reasons")),
        low_confidence_reasons=_string_tuple(payload.get("low_confidence_reasons")),
        suggested_improvements=_string_tuple(payload.get("suggested_improvements")),
        hard_failures=combined_hard_failures,
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
        response = self.client.chat_json(build_evaluation_messages(clean))
        payload = extract_json_object(response.content)
        question_type = str(clean.get("answer_format") or clean.get("type") or "")
        evaluation = parse_evaluation_payload(
            qid=str(clean.get("qid", "")),
            question_type=question_type,
            payload=payload,
            hard_failures=detect_hard_failures(clean),
        )
        usage = response.token_usage.to_dict()
        return evaluation, {key: int(value) for key, value in usage.items()}


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


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(sanitize_subject(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
