from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from afa_agent.b_board.calculation import CalculationExecutor
from afa_agent.b_board.calculation_schema import (
    CALCULATION_PLAN_SCHEMA,
    CALCULATION_PLAN_SCHEMA_VERSION,
    validate_calculation_plan_schema,
)
from afa_agent.b_board.io import (
    BQuestion,
    infer_percent_suffix_requirement,
    infer_requested_decimal_places,
)
from afa_agent.b_board.reasoning_schema import (
    required_frozen_answer_conclusion,
    validate_model_generated_frozen_answer_conclusion,
)
from afa_agent.b_board.runner import (
    CALCULATION_SYSTEM_PROMPT,
    _normalize_calculation_numeric_literals,
    _normalize_calculation_plan_structure,
    _validate_calculation_plan_has_required_inputs,
    _validate_calculation_result_semantics,
)
from afa_agent.client import TRANSPORT_RETRY_POLICY_VERSION
from afa_agent.io_utils import write_json as write_json_atomic
from afa_agent.models import TokenUsage


FROZEN_CALCULATION_CHECKPOINT_VERSION = "retrieval_llm_frozen_calculation_v1"
CALCULATION_REASONING_SCHEMA_VERSION = "verified_calculation_reasoning_only_v1"
MAX_VERIFIED_REASONING_CALLS = 2
CALCULATION_REASONING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "string",
            "minLength": 20,
        }
    },
    "required": ["reasoning"],
}


class VerifiedCalculationStageError(RuntimeError):
    """An auditable verified-calculation failure with its observed usage."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        error_code: str,
        retry_route: str,
        calls: Sequence[Mapping[str, Any]],
        unobservable_usage_risk: bool,
        decision_trace: Mapping[str, Any] | None = None,
        frozen_checkpoint_path: Path | None = None,
        failed_transport_attempt_count: int = 0,
        failed_transport_rejections: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.error_code = error_code
        self.retry_route = retry_route
        self.calls = [dict(call) for call in calls]
        self.token_usage = _sum_call_usage(self.calls)
        self.unobservable_usage_risk = unobservable_usage_risk
        self.decision_trace = dict(decision_trace or {})
        self.frozen_checkpoint_path = (
            str(frozen_checkpoint_path) if frozen_checkpoint_path else ""
        )
        self.failed_transport_attempt_count = failed_transport_attempt_count
        self.failed_transport_rejections = [
            dict(item) for item in failed_transport_rejections
        ]


def run_verified_calculation(
    question: BQuestion,
    *,
    evidence: Sequence[Mapping[str, Any]],
    client: Any,
    run_dir: Path,
    thinking_budget: int,
) -> dict[str, Any]:
    """Run and audit a Qwen plan -> Decimal freeze -> reasoning-only pipeline."""

    plan_evidence = [_plan_evidence_item(item) for item in evidence]
    evidence_text_by_id = {
        "QUESTION": question.question,
        **{
            str(item["evidence_key"]): _evidence_text(item)
            for item in evidence
        },
    }
    checkpoint_path = run_dir / "frozen_answers" / f"{question.qid}.json"
    raw_calls_path = run_dir / "raw_calls" / f"{question.qid}.json"
    run_binding = _run_instance_binding(run_dir)
    if checkpoint_path.exists():
        try:
            checkpoint = _load_frozen_checkpoint(
                checkpoint_path,
                question=question,
                evidence=evidence,
                expected_model_name=client.config.model_name,
                expected_run_binding=run_binding,
            )
            calls = _load_observed_calls(
                raw_calls_path,
                fallback=checkpoint.get("calls", []),
                expected_qid=question.qid,
                expected_evidence_alias_map=_evidence_alias_map(evidence),
            )
        except Exception as exc:
            raise VerifiedCalculationStageError(
                str(exc),
                stage="answer",
                error_code="frozen_checkpoint_invalid",
                retry_route="stop_checkpoint_integrity_failure",
                calls=[],
                unobservable_usage_risk=(
                    isinstance(exc, RuntimeError)
                    and "unobservable" in str(exc)
                ),
                decision_trace={
                    "checkpoint_validation": "failed_before_provider_call"
                },
                frozen_checkpoint_path=checkpoint_path,
            ) from exc
        plan = dict(checkpoint["calculation_plan"])
        calculation_trace = dict(checkpoint["calculation_trace"])
        frozen_answer_parts = [
            str(item) for item in checkpoint["answer_parts"]
        ]
        decision_trace = dict(checkpoint["decision_trace"])
        decision_trace["answer_stage"] = {
            **dict(decision_trace.get("answer_stage") or {}),
            "resumed_from_frozen_checkpoint": True,
        }
    else:
        if raw_calls_path.exists():
            try:
                calls = _load_observed_calls(
                    raw_calls_path,
                    fallback=[],
                    expected_qid=question.qid,
                    expected_evidence_alias_map=_evidence_alias_map(evidence),
                )
            except Exception as exc:
                raise VerifiedCalculationStageError(
                    str(exc),
                    stage="answer",
                    error_code="orphan_raw_checkpoint_invalid",
                    retry_route="stop_checkpoint_integrity_failure",
                    calls=[],
                    unobservable_usage_risk=True,
                    decision_trace={
                        "raw_checkpoint_validation": "failed_before_provider_call"
                    },
                ) from exc
            raise VerifiedCalculationStageError(
                "observed calculation-plan call exists without a frozen answer; "
                "automatic resend is forbidden",
                stage="answer",
                error_code="orphan_raw_checkpoint",
                retry_route="stop_without_provider_call",
                calls=calls,
                unobservable_usage_risk=False,
                decision_trace={
                    "raw_checkpoint_validation": "preserved_without_resend"
                },
            )
        calls = []
        plan_messages = _plan_messages(question, plan_evidence)
        try:
            _record_verified_call_intent(
                run_dir,
                qid=question.qid,
                call_index=1,
                purpose="calculation_plan",
                evidence=evidence,
            )
            plan_response = client.chat_json(
                plan_messages,
                response_schema=CALCULATION_PLAN_SCHEMA,
                schema_name=CALCULATION_PLAN_SCHEMA_VERSION,
                extra_body=_thinking_body(thinking_budget),
            )
        except Exception as exc:
            raise _provider_stage_error(exc, stage="answer", calls=calls) from exc
        calls.append(
            _call_record(
                client=client,
                response=plan_response,
                messages=plan_messages,
                schema=CALCULATION_PLAN_SCHEMA,
                purpose="calculation_plan",
            )
        )
        _write_raw_calls_checkpoint(run_dir, question.qid, calls, evidence)

        plan_normalizations: list[dict[str, Any]] = []
        plan: dict[str, Any] | None = None
        try:
            raw_plan = json.loads(plan_response.content)
            if not isinstance(raw_plan, dict):
                raise ValueError("calculation plan response must be a JSON object")
            plan, numeric_normalizations = (
                _normalize_calculation_numeric_literals(raw_plan)
            )
            plan, structure_normalizations = (
                _normalize_calculation_plan_structure(
                    plan,
                    evidence_text_by_id=evidence_text_by_id,
                )
            )
            plan_normalizations = [
                *numeric_normalizations,
                *structure_normalizations,
            ]
            validate_calculation_plan_schema(plan)
            _validate_calculation_plan_has_required_inputs(plan)
            result = CalculationExecutor().execute(
                plan,
                expected_slots=question.answer_slots,
                evidence_text_by_id=evidence_text_by_id,
                expected_slot_templates=question.answer_slot_templates,
                expected_numeric_decimal_places=infer_requested_decimal_places(
                    question.question
                ),
                expected_percent_suffixes=tuple(
                    infer_percent_suffix_requirement(
                        question.question,
                        slot_index=index,
                        slot_count=question.answer_slots,
                    )
                    for index in range(1, question.answer_slots + 1)
                ),
            )
            _validate_calculation_result_semantics(question, result.trace)
        except Exception as exc:
            error_code, retry_route = _classify_answer_stage_error(exc)
            raise VerifiedCalculationStageError(
                str(exc),
                stage="answer",
                error_code=error_code,
                retry_route=retry_route,
                calls=calls,
                unobservable_usage_risk=False,
                decision_trace={
                    "calculation_plan_normalizations": plan_normalizations,
                    "failed_plan": plan,
                },
            ) from exc
        assert plan is not None
        frozen_answer_parts = list(result.answer_parts)
        calculation_trace = result.trace
        decision_trace = {
            "source": "retrieval_llm_verified_calculation",
            "calculation_plan_schema_version": CALCULATION_PLAN_SCHEMA_VERSION,
            "calculation_plan_normalizations": plan_normalizations,
            "answer_stage": {
                "status": "frozen",
                "answer_parts_frozen": True,
                "resumed_from_frozen_checkpoint": False,
                "model_name": client.config.model_name,
                "response_format_mode": plan_response.response_format_mode,
                "call_indexes": [1],
                "token_usage": _sum_call_usage(calls),
            },
        }
        checkpoint = {
            "checkpoint_version": FROZEN_CALCULATION_CHECKPOINT_VERSION,
            "qid": question.qid,
            **run_binding,
            "question_sha256": _sha256_text(question.question),
            "evidence_sha256": _evidence_sha256(evidence),
            "answer_parts": frozen_answer_parts,
            "decision_summary": str(plan.get("decision_summary", "")).strip(),
            "calculation_plan": plan,
            "calculation_trace": calculation_trace,
            "decision_trace": decision_trace,
            "token_usage": _sum_call_usage(calls),
            "calls": list(calls),
            "evidence_alias_map": _evidence_alias_map(evidence),
        }
        _write_json(checkpoint_path, checkpoint)

    required_conclusion = required_frozen_answer_conclusion(frozen_answer_parts)
    prior_reasoning_call_count = sum(
        call.get("purpose") == "verified_calculation_reasoning"
        for call in calls
    )
    if prior_reasoning_call_count >= MAX_VERIFIED_REASONING_CALLS:
        raise VerifiedCalculationStageError(
            "verified calculation reasoning retry budget is exhausted",
            stage="reasoning",
            error_code="reasoning_retry_budget_exhausted",
            retry_route="stop_reasoning_retry_budget_exhausted",
            calls=calls,
            unobservable_usage_risk=False,
            decision_trace=decision_trace,
            frozen_checkpoint_path=checkpoint_path,
        )
    reasoning_messages = _reasoning_messages(
        question,
        checkpoint=checkpoint,
        evidence=plan_evidence,
        required_conclusion=required_conclusion,
    )
    try:
        _record_verified_call_intent(
            run_dir,
            qid=question.qid,
            call_index=len(calls) + 1,
            purpose="verified_calculation_reasoning",
            evidence=evidence,
        )
        reasoning_response = client.chat_json(
            reasoning_messages,
            response_schema=CALCULATION_REASONING_SCHEMA,
            schema_name=CALCULATION_REASONING_SCHEMA_VERSION,
            extra_body=_thinking_body(thinking_budget),
        )
    except Exception as exc:
        raise _provider_stage_error(
            exc,
            stage="reasoning",
            calls=calls,
            decision_trace=decision_trace,
            frozen_checkpoint_path=checkpoint_path,
        ) from exc
    calls.append(
        _call_record(
            client=client,
            response=reasoning_response,
            messages=reasoning_messages,
            schema=CALCULATION_REASONING_SCHEMA,
            purpose="verified_calculation_reasoning",
        )
    )
    _write_raw_calls_checkpoint(run_dir, question.qid, calls, evidence)
    try:
        reasoning_payload = json.loads(reasoning_response.content)
        reasoning = _validate_reasoning_only_payload(
            reasoning_payload,
            frozen_answer_parts=frozen_answer_parts,
        )
    except Exception as exc:
        raise VerifiedCalculationStageError(
            str(exc),
            stage="reasoning",
            error_code="reasoning_contract_error",
            retry_route="resume_reasoning_from_frozen_checkpoint",
            calls=calls,
            unobservable_usage_risk=False,
            decision_trace=decision_trace,
            frozen_checkpoint_path=checkpoint_path,
        ) from exc

    reasoning_calls = [
        call
        for call in calls
        if call.get("purpose") == "verified_calculation_reasoning"
    ]
    decision_trace["reasoning_stage"] = {
        "status": "complete",
        "schema_version": CALCULATION_REASONING_SCHEMA_VERSION,
        "output_fields": ["reasoning"],
        "answer_parts_preserved_from_checkpoint": True,
        "model_name": client.config.model_name,
        "response_format_mode": reasoning_response.response_format_mode,
        "call_indexes": [
            int(call["call_index"]) for call in reasoning_calls
        ],
        "token_usage": _sum_call_usage(reasoning_calls),
    }
    return {
        "answer_parts": frozen_answer_parts,
        "reasoning": reasoning,
        "calls": calls,
        "token_usage": _sum_call_usage(calls),
        "calculation_plan": plan,
        "calculation_trace": calculation_trace,
        "decision_trace": decision_trace,
        "frozen_answer_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "normalization": {
            "answer_modified": False,
            "answer_transformations": [],
            "reasoning_modified": False,
        },
    }


def _provider_stage_error(
    exc: Exception,
    *,
    stage: str,
    calls: Sequence[Mapping[str, Any]],
    decision_trace: Mapping[str, Any] | None = None,
    frozen_checkpoint_path: Path | None = None,
) -> VerifiedCalculationStageError:
    explicit_429 = _is_explicit_429(exc)
    attempt_count = getattr(exc, "transport_attempt_count", 1)
    rejections = [
        dict(item)
        for item in getattr(exc, "transport_rejections", ())
        if isinstance(item, Mapping)
    ]
    if explicit_429 and not rejections and attempt_count == 1:
        rejections = [
            {
                "attempt_index": 1,
                "status_code": 429,
                "pre_generation_rejection": True,
                "token_usage_observed": False,
            }
        ]
    fully_observed_rejections = bool(
        explicit_429
        and not isinstance(attempt_count, bool)
        and isinstance(attempt_count, int)
        and attempt_count >= 1
        and attempt_count == len(rejections)
        and all(
            item.get("attempt_index") == index
            and item.get("status_code") == 429
            and item.get("pre_generation_rejection") is True
            and item.get("token_usage_observed") is False
            for index, item in enumerate(rejections, start=1)
        )
    )
    return VerifiedCalculationStageError(
        str(exc),
        stage=stage,
        error_code=f"{stage}_transport_error",
        retry_route=(
            "transport_client_explicit_429_only"
            if explicit_429
            else "do_not_retry_unobservable_generation"
        ),
        calls=calls,
        unobservable_usage_risk=not fully_observed_rejections,
        decision_trace={
            **dict(decision_trace or {}),
            "transport_retry_policy_version": TRANSPORT_RETRY_POLICY_VERSION,
        },
        frozen_checkpoint_path=frozen_checkpoint_path,
        failed_transport_attempt_count=(
            int(attempt_count)
            if isinstance(attempt_count, int)
            and not isinstance(attempt_count, bool)
            else 0
        ),
        failed_transport_rejections=rejections,
    )


def _is_explicit_429(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return (
        isinstance(exc, requests.HTTPError)
        and response is not None
        and int(getattr(response, "status_code", 0)) == 429
    )


def _classify_answer_stage_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    if isinstance(exc, json.JSONDecodeError) or "schema violation" in message:
        return (
            "calculation_plan_schema_error",
            "local_structure_repair_exhausted",
        )
    if any(
        marker in message
        for marker in (
            "not grounded",
            "evidence_ids",
            "cited evidence",
            "unit source",
        )
    ):
        return (
            "calculation_grounding_error",
            "stop_missing_grounding_no_automatic_retrieval",
        )
    if any(
        marker in message
        for marker in (
            "operand",
            "direction",
            "must derive",
            "must use",
            "period mismatch",
        )
    ):
        return (
            "calculation_formula_ambiguity",
            "stop_formula_ambiguity_no_automatic_resend",
        )
    return (
        "calculation_decimal_replay_error",
        "stop_verified_answer_stage",
    )


def _plan_messages(
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": CALCULATION_SYSTEM_PROMPT.replace(
                "question:<qid>",
                "QUESTION",
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question.question,
                    "answer_format": question.answer_format,
                    "answer_slot_count": question.answer_slots,
                    "question_evidence_id": "QUESTION",
                    "evidence": list(evidence),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _reasoning_messages(
    question: BQuestion,
    *,
    checkpoint: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    required_conclusion: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是金融计算题最终推理摘要生成器。答案已经由证据grounding与Decimal"
                "执行器验证并冻结。你不得重算、修改、补写或再次输出答案字段；只输出"
                "JSON Schema允许的reasoning字符串。摘要必须自包含地说明主体、期间、"
                "原始输入、单位、公式和关键计算，并以给定required_conclusion_text"
                "逐字结束。不得引用题号、历史答案或外部知识。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question.question,
                    "frozen_answer_parts": checkpoint["answer_parts"],
                    "required_conclusion_text": required_conclusion,
                    "verified_solution_summary": checkpoint["decision_summary"],
                    "verified_calculation_trace": checkpoint["calculation_trace"],
                    "evidence": list(evidence),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _validate_reasoning_only_payload(
    payload: Any,
    *,
    frozen_answer_parts: Sequence[str],
) -> str:
    if not isinstance(payload, Mapping) or set(payload) != {"reasoning"}:
        raise ValueError("reasoning response must contain only the reasoning field")
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str) or len(reasoning.strip()) < 20:
        raise ValueError("reasoning must contain at least 20 characters")
    validate_model_generated_frozen_answer_conclusion(
        reasoning,
        frozen_answer_parts=frozen_answer_parts,
        contract_name="VerifiedCalculationReasoning",
    )
    return reasoning


def validate_verified_frozen_answer_reasoning_payload(
    frozen_answer_parts: Sequence[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct a verified answer without applying the direct-call contract."""

    reasoning = _validate_reasoning_only_payload(
        payload,
        frozen_answer_parts=frozen_answer_parts,
    )
    return {
        "answer_parts": [str(item) for item in frozen_answer_parts],
        "reasoning": reasoning,
        "decision_trace": {
            "answer_stage": "frozen_from_verified_decimal_replay",
            "reasoning_stage": "verified_calculation_reasoning",
            "postprocessing_mode": "none",
            "answer_modified": False,
            "reasoning_modified": False,
        },
    }


def validate_verified_calculation_checkpoint(
    path: Path,
    *,
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
    expected_model_name: str,
    expected_run_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Revalidate the sealed plan, evidence binding and Decimal replay."""

    return _load_frozen_checkpoint(
        path,
        question=question,
        evidence=evidence,
        expected_model_name=expected_model_name,
        expected_run_binding=expected_run_binding,
    )


def _plan_evidence_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": str(item["evidence_key"]),
        "source_id": str(item["source_key"]),
        "title_path": [str(value) for value in item.get("title_path", [])],
        "text": _evidence_text(item),
        "raw_text_sha256": str(item.get("prompt_text_sha256", "")),
    }


def _evidence_text(item: Mapping[str, Any]) -> str:
    titles = [
        str(value).strip()
        for value in item.get("title_path", [])
        if str(value).strip()
    ]
    body = str(item.get("text", ""))
    if not titles:
        return body
    return f"[原始标题/表头] {' / '.join(titles)}\n[原始正文]\n{body}"


def _call_record(
    *,
    client: Any,
    response: Any,
    messages: list[dict[str, str]],
    schema: Mapping[str, Any],
    purpose: str,
) -> dict[str, Any]:
    return {
        "call_index": 0,  # assigned by the caller below
        "purpose": purpose,
        "model_name": client.config.model_name,
        "response_format_mode": response.response_format_mode,
        "messages": messages,
        "response_schema": dict(schema),
        "raw_response": response.raw_payload,
        "content": response.content,
        "token_usage": response.token_usage.to_dict(),
        "transport_attempt_count": int(
            getattr(response, "transport_attempt_count", 1)
        ),
        "transport_rejections": list(
            getattr(response, "transport_rejections", ())
        ),
    }


def _sum_call_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    usage = TokenUsage()
    indexes = [int(call.get("call_index", -1)) for call in calls]
    if indexes and indexes != list(
        range(indexes[0], indexes[0] + len(indexes))
    ):
        raise ValueError("observed call indexes are not contiguous")
    if indexes and indexes[0] < 1:
        raise ValueError("observed call indexes must be positive")
    for call in calls:
        call_index = int(call["call_index"])
        if call_index < 1:
            raise ValueError("observed call indexes are not contiguous")
        raw = call["token_usage"]
        current = _strict_usage(
            raw,
            label=f"observed call {call_index} token usage",
        )
        provider = _strict_usage(
            (call.get("raw_response") or {}).get("usage") or {},
            label=f"observed call {call_index} provider usage",
        )
        if current.to_dict() != provider.to_dict():
            raise ValueError("observed call usage differs from provider raw usage")
        if (
            "transport_attempt_count" not in call
            or "transport_rejections" not in call
        ):
            raise ValueError("observed call transport audit is missing")
        attempt_count = call["transport_attempt_count"]
        rejections = call["transport_rejections"]
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count < 1
            or not isinstance(rejections, list)
            or attempt_count != len(rejections) + 1
            or any(
                not isinstance(item, Mapping)
                or item.get("attempt_index") != rejection_index
                or item.get("status_code") != 429
                or item.get("pre_generation_rejection") is not True
                or item.get("token_usage_observed") is not False
                for rejection_index, item in enumerate(
                    rejections, start=1
                )
            )
        ):
            raise ValueError("observed call transport audit is invalid")
        usage.add(current)
    return usage.to_dict()


def _write_raw_calls_checkpoint(
    run_dir: Path,
    qid: str,
    calls: list[dict[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    for index, call in enumerate(calls, start=1):
        call["call_index"] = index
    path = run_dir / "raw_calls" / f"{qid}.json"
    payload = {
        "qid": qid,
        **_run_instance_binding(run_dir),
        "evidence_alias_map": _evidence_alias_map(evidence),
        "calls": calls,
        "checkpoint_only": True,
        "ledger_state": "checkpoint",
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("existing raw call ledger must be an object")
        for key in (
            "qid",
            "run_fingerprint",
            "run_instance_id",
            "evidence_alias_map",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"raw call ledger {key} changed")
        existing_calls = existing.get("calls")
        if not isinstance(existing_calls, list):
            raise ValueError("existing raw call ledger calls must be an array")
        if len(calls) < len(existing_calls):
            raise ValueError("raw call ledger cannot lose observed calls")
        if calls[: len(existing_calls)] != existing_calls:
            raise ValueError("raw call ledger can only append observed calls")
    _write_json(path, payload)


def _record_verified_call_intent(
    run_dir: Path,
    *,
    qid: str,
    call_index: int,
    purpose: str,
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    path = run_dir / "call_intents" / f"{qid}.{call_index}.json"
    payload = {
        "qid": qid,
        "call_index": call_index,
        "purpose": purpose,
        **_run_instance_binding(run_dir),
        "evidence_alias_map_sha256": _sha256_text(
            json.dumps(
                _evidence_alias_map(evidence),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "state": "provider_call_started",
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("verified call intent changed")
        raise ValueError("verified call intent already exists")
    write_json_atomic(path, payload)


def _validate_verified_call_intents(
    run_dir: Path,
    *,
    qid: str,
    evidence_alias_map: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
) -> None:
    intent_dir = run_dir / "call_intents"
    if not intent_dir.exists():
        return
    alias_sha256 = _sha256_text(
        json.dumps(
            list(evidence_alias_map),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    intents: dict[int, dict[str, Any]] = {}
    expected_binding = _run_instance_binding(run_dir)
    for path in sorted(intent_dir.glob(f"{qid}.*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        call_index = payload.get("call_index") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or payload.get("qid") != qid
            or path.name != f"{qid}.{call_index}.json"
            or call_index in intents
            or payload.get("evidence_alias_map_sha256") != alias_sha256
            or payload.get("state") != "provider_call_started"
            or any(
                payload.get(key) != value
                for key, value in expected_binding.items()
            )
        ):
            raise ValueError("verified call intent binding is invalid")
        intents[call_index] = payload
    for call in calls:
        call_index = int(call.get("call_index", -1))
        intent = intents.get(call_index)
        if intent is None or intent.get("purpose") != call.get("purpose"):
            raise ValueError("observed verified call has no matching intent")
    unmatched = sorted(
        set(intents)
        - {int(call.get("call_index", -1)) for call in calls}
    )
    if unmatched:
        raise RuntimeError(
            "unobservable verified provider attempt exists for call indexes "
            + ",".join(str(item) for item in unmatched)
        )


def _evidence_alias_map(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "alias": str(item["evidence_key"]),
            "source_alias": str(item["source_key"]),
            "source_evidence_id": str(item["evidence_id"]),
            "doc_id": str(item["doc_id"]),
            "retrieval_rank": int(item["rank"]),
            "unit_type": str(item["unit_type"]),
            "title_path": list(item["title_path"]),
            "prompt_text_sha256": str(item["prompt_text_sha256"]),
            "source_text_sha256": str(item["source_text_sha256"]),
            "prompt_text_chars": len(str(item["text"])),
            "truncated": bool(item["truncated"]),
            "merged_from": list(item.get("merged_from", [])),
            "source_order": list(item.get("source_order", [])),
            "overlap_chars": int(item.get("overlap_chars", 0)),
            "component_hashes": list(item.get("component_hashes", [])),
            "compaction_truncation_provenance": dict(
                item.get("compaction_truncation_provenance", {})
            ),
        }
        for item in evidence
    ]


def _load_frozen_checkpoint(
    path: Path,
    *,
    question: BQuestion,
    evidence: Sequence[Mapping[str, Any]],
    expected_model_name: str,
    expected_run_binding: Mapping[str, str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen calculation checkpoint must be an object")
    if payload.get("checkpoint_version") != FROZEN_CALCULATION_CHECKPOINT_VERSION:
        raise ValueError("frozen calculation checkpoint version mismatch")
    if payload.get("qid") != question.qid:
        raise ValueError("frozen calculation checkpoint qid mismatch")
    if expected_run_binding and any(
        payload.get(key) != value
        for key, value in expected_run_binding.items()
    ):
        raise ValueError("frozen calculation checkpoint run instance mismatch")
    if payload.get("question_sha256") != _sha256_text(question.question):
        raise ValueError("frozen calculation checkpoint question mismatch")
    if payload.get("evidence_sha256") != _evidence_sha256(evidence):
        raise ValueError("frozen calculation checkpoint evidence mismatch")
    answer_parts = payload.get("answer_parts")
    if (
        not isinstance(answer_parts, list)
        or len(answer_parts) != question.answer_slots
        or any(not isinstance(item, str) or not item for item in answer_parts)
    ):
        raise ValueError("frozen calculation checkpoint answer shape mismatch")
    trace = payload.get("calculation_trace")
    if not isinstance(trace, dict) or not (
        trace.get("grounding_verified") and trace.get("replay_verified")
    ):
        raise ValueError("frozen calculation checkpoint is not verified")
    decision_trace = payload.get("decision_trace")
    answer_stage = (
        decision_trace.get("answer_stage")
        if isinstance(decision_trace, dict)
        else None
    )
    if (
        not isinstance(answer_stage, dict)
        or not answer_stage.get("answer_parts_frozen")
        or answer_stage.get("model_name") != expected_model_name
    ):
        raise ValueError("frozen calculation checkpoint answer contract mismatch")
    checkpoint_calls = payload.get("calls")
    if not isinstance(checkpoint_calls, list):
        raise ValueError("frozen calculation checkpoint calls are missing")
    if _sum_call_usage(checkpoint_calls) != payload.get("token_usage"):
        raise ValueError("frozen calculation checkpoint usage mismatch")
    plan = payload.get("calculation_plan", {})
    validate_calculation_plan_schema(plan)
    _validate_calculation_plan_has_required_inputs(plan)
    evidence_text_by_id = {
        "QUESTION": question.question,
        **{
            str(item["evidence_key"]): _evidence_text(item)
            for item in evidence
        },
    }
    replay = CalculationExecutor().execute(
        plan,
        expected_slots=question.answer_slots,
        evidence_text_by_id=evidence_text_by_id,
        expected_slot_templates=question.answer_slot_templates,
        expected_numeric_decimal_places=infer_requested_decimal_places(
            question.question
        ),
        expected_percent_suffixes=tuple(
            infer_percent_suffix_requirement(
                question.question,
                slot_index=index,
                slot_count=question.answer_slots,
            )
            for index in range(1, question.answer_slots + 1)
        ),
    )
    _validate_calculation_result_semantics(question, replay.trace)
    if list(replay.answer_parts) != answer_parts:
        raise ValueError("frozen calculation checkpoint answer replay mismatch")
    if replay.trace != trace:
        raise ValueError("frozen calculation checkpoint trace replay mismatch")
    return payload


def _load_observed_calls(
    path: Path,
    *,
    fallback: Any,
    expected_qid: str,
    expected_evidence_alias_map: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fallback_calls = (
        [dict(call) for call in fallback]
        if isinstance(fallback, list)
        else None
    )
    raw_calls = fallback
    if path.exists():
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise ValueError("raw call checkpoint must be an object")
        if raw_payload.get("qid") != expected_qid:
            raise ValueError("raw call checkpoint qid mismatch")
        expected_binding = _run_instance_binding(path.parents[1])
        if expected_binding and any(
            raw_payload.get(key) != value
            for key, value in expected_binding.items()
        ):
            raise ValueError("raw call checkpoint run instance mismatch")
        if raw_payload.get("evidence_alias_map") != list(
            expected_evidence_alias_map
        ):
            raise ValueError("raw call checkpoint evidence fingerprint mismatch")
        raw_calls = raw_payload.get("calls")
    if not isinstance(raw_calls, list):
        raise ValueError("raw observed calls must be an array")
    calls = [dict(call) for call in raw_calls]
    _sum_call_usage(calls)
    if [int(call.get("call_index", -1)) for call in calls] != list(
        range(1, len(calls) + 1)
    ):
        raise ValueError("raw observed calls must start at one")
    if fallback_calls is None:
        raise ValueError("frozen checkpoint fallback calls must be an array")
    if calls[: len(fallback_calls)] != fallback_calls:
        raise ValueError("raw observed calls do not preserve frozen answer calls")
    _validate_verified_call_intents(
        path.parents[1],
        qid=expected_qid,
        evidence_alias_map=expected_evidence_alias_map,
        calls=calls,
    )
    return calls


def _strict_usage(payload: Any, *, label: str) -> TokenUsage:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{label} {key} must be a non-negative integer"
            )
        values[key] = value
    if values["total_tokens"] != (
        values["prompt_tokens"] + values["completion_tokens"]
    ):
        raise ValueError(f"{label} total_tokens is inconsistent")
    return TokenUsage(**values)


def _run_instance_binding(run_dir: Path) -> dict[str, str]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run config must be an object")
    fingerprint = payload.get("fingerprint")
    run_instance_id = payload.get("run_instance_id")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or not isinstance(run_instance_id, str)
        or not run_instance_id
    ):
        raise ValueError("run config binding is invalid")
    return {
        "run_fingerprint": fingerprint,
        "run_instance_id": run_instance_id,
    }


def _evidence_sha256(evidence: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "evidence_key": str(item["evidence_key"]),
            "evidence_id": str(item["evidence_id"]),
            "prompt_text_sha256": str(item["prompt_text_sha256"]),
            "text": str(item["text"]),
        }
        for item in evidence
    ]
    return _sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _thinking_body(thinking_budget: int) -> dict[str, Any]:
    return {
        "enable_thinking": thinking_budget > 0,
        **(
            {"thinking_budget": thinking_budget}
            if thinking_budget > 0
            else {}
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
