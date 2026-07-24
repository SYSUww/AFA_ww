from __future__ import annotations

import csv
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
from uuid import uuid4

from afa_agent.b_board.io import BQuestion, validate_b_submission
from afa_agent.b_board.scoring import score_submission
from afa_agent.client import TRANSPORT_RETRY_POLICY_VERSION, OpenAICompatibleClient
from afa_agent.config import (
    STRUCTURED_OUTPUT_LOCAL,
    STRUCTURED_OUTPUT_NATIVE,
    ModelConfig,
)
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


PROMPT_VERSION = "b_reasoning_shadow_judge_v2_dual_stage"
SCHEMA_VERSION = 2
FINGERPRINT_SCHEMA_VERSION = 3
MIN_REASONING_NON_WHITESPACE = 20
REASONING_DIMENSIONS = ("logical", "completeness", "clarity")
REASONING_JUDGE_MODEL = "gpt-5.6"

HARD_CAPS: dict[str, dict[str, float]] = {
    "generic_or_repetition_only": {
        "logical": 29.0,
        "completeness": 29.0,
        "clarity": 29.0,
    },
    "no_concrete_support": {"completeness": 39.0},
    "no_reasoning_relation": {"logical": 59.0},
    "no_explicit_conclusion": {"completeness": 59.0},
    "internal_contradiction_or_arithmetic_error": {"logical": 29.0},
    "machine_id_only": {"completeness": 59.0, "clarity": 59.0},
    "calculation_steps_missing": {"completeness": 59.0},
    "multi_labels_only": {"completeness": 59.0},
}
HARD_CAP_VIOLATIONS = tuple(HARD_CAPS)

SCORER_SYSTEM_PROMPT = f"""你是推理摘要写作质量的 Rubric Scorer。你只能评估用户消息中的冻结文本本身，不使用任何外部上下文，也不猜测缺失信息。
分别给出三个 0 到 100 的分数：
logical：推理步骤之间是否存在清晰因果关系，整体链条是否自洽。
completeness：文本是否呈现定位、提取、推导和结论等完整分析过程。
clarity：文本是否条理清晰、结构化、表达准确。
评分尺度：80 分以上表示逻辑清晰、论证完整、表达专业；60-79 表示有明确分析步骤和推导；30-59 表示有部分分析但不完整；0-29 表示空泛、模板化或无实质内容。
不要评判无法核验的外部事实真伪；但也不得借助缺失的外部上下文替文本补足依据、
步骤或结论，冻结文本未展示的内容就视为未展示。只评价文本实际呈现的写作质量。
只输出 JSON 对象，且必须仅包含 logical、completeness、clarity 三个数值字段。
prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}, stage=rubric_scorer。"""

AUDITOR_SYSTEM_PROMPT = f"""你是推理摘要写作质量的 Adversarial Auditor。你只能审查用户消息中的冻结文本本身，不使用任何外部上下文，也不猜测缺失信息。
独立给出 logical、completeness、clarity 三个 0 到 100 的分数，并从固定 violations 枚举中标记文本实际存在的缺陷：
generic_or_repetition_only：仅给结果、复述输入或只有“根据材料”等空泛措辞；
no_concrete_support：没有具体事实、数字、条款或其他可识别依据；
no_reasoning_relation：没有因果、比较、计算或从依据到结论的连接；
no_explicit_conclusion：没有明确落到最终判断或结果；
internal_contradiction_or_arithmetic_error：文本内部自相矛盾或存在可由文本直接确认的明显算术错误；
machine_id_only：只有 E/S/chunk 等机器编号，没有复述其关键内容；
calculation_steps_missing：文本明显在完成计算，但缺少输入、公式或题面要求的舍入过程；
multi_labels_only：只列选择标签及对错，没有命题内容与判断依据。
不要仅因文本较短就虚构缺陷；不要检查外部事实真伪，也不得借助缺失上下文替文本
补足依据、步骤或结论。只输出 JSON 对象，且必须仅包含 logical、completeness、
clarity 三个数值字段和 violations 字符串数组。
prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}, stage=adversarial_auditor。"""

# Compatibility alias for callers that previously imported the single-stage prompt.
REASONING_JUDGE_SYSTEM_PROMPT = SCORER_SYSTEM_PROMPT

SCORER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        key: {"type": "number", "minimum": 0, "maximum": 100}
        for key in REASONING_DIMENSIONS
    },
    "required": list(REASONING_DIMENSIONS),
}

AUDITOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{
            key: {"type": "number", "minimum": 0, "maximum": 100}
            for key in REASONING_DIMENSIONS
        },
        "violations": {
            "type": "array",
            "items": {"type": "string", "enum": list(HARD_CAP_VIOLATIONS)},
        },
    },
    "required": [*REASONING_DIMENSIONS, "violations"],
}


class ReasoningEvaluationFingerprintError(RuntimeError):
    """Raised when a sealed reasoning evaluation cannot safely resume."""


class ReasoningJudgeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        token_usage: Mapping[str, int] | None = None,
        calls: Sequence[Mapping[str, Any]] = (),
        failed_stage: str | None = None,
        transport_attempt_count: int | None = None,
        transport_rejections: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.token_usage = dict(token_usage or _zero_usage())
        self.calls = tuple(dict(call) for call in calls)
        self.failed_stage = failed_stage
        self.transport_attempt_count = transport_attempt_count
        self.transport_rejections = tuple(
            dict(item) for item in transport_rejections
        )


@dataclass(frozen=True, slots=True)
class ReasoningJudgeOutcome:
    rubric_scores: dict[str, float]
    auditor_scores: dict[str, float]
    violations: tuple[str, ...]
    final_dimensions: dict[str, float]
    hard_caps: dict[str, float]
    total_usage: dict[str, int]
    calls: tuple[dict[str, Any], ...] = ()


class ReasoningEvaluator(Protocol):
    def evaluate(
        self, reasoning: str
    ) -> ReasoningJudgeOutcome | tuple[Mapping[str, float], Mapping[str, int]]: ...


EvaluatorFactory = Callable[[], ReasoningEvaluator]


@dataclass(frozen=True, slots=True)
class ReasoningEvaluation:
    qid: str
    logical: float
    completeness: float
    clarity: float
    status: str = "scored"
    prompt_version: str = PROMPT_VERSION
    schema_version: int = SCHEMA_VERSION
    rubric_scores: dict[str, float] = field(default_factory=dict)
    auditor_scores: dict[str, float] = field(default_factory=dict)
    hard_cap_violations: tuple[str, ...] = ()
    hard_caps: dict[str, float] = field(default_factory=dict)

    @property
    def reasoning_score(self) -> float:
        return fmean((self.logical, self.completeness, self.clarity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "logical": self.logical,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "reasoning_score": self.reasoning_score,
            "status": self.status,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "rubric_scores": dict(self.rubric_scores),
            "auditor_scores": dict(self.auditor_scores),
            "hard_cap_violations": list(self.hard_cap_violations),
            "hard_caps": dict(self.hard_caps),
        }


@dataclass(frozen=True, slots=True)
class ReasoningEvaluationRunResult:
    manifest: dict[str, Any]
    aggregate: dict[str, Any]
    evaluations: dict[str, ReasoningEvaluation]
    scorecard: dict[str, Any] | None


class FixedReasoningEvaluator:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        stage_intent: Callable[[str], None] | None = None,
        stage_checkpoint: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self.stage_intent = stage_intent
        self.stage_checkpoint = stage_checkpoint

    def evaluate(self, reasoning: str) -> ReasoningJudgeOutcome:
        calls: list[dict[str, Any]] = []
        try:
            rubric = self._run_stage(
                reasoning=reasoning,
                stage="rubric_scorer",
                system_prompt=SCORER_SYSTEM_PROMPT,
                response_schema=SCORER_RESPONSE_SCHEMA,
                calls=calls,
            )
            auditor = self._run_stage(
                reasoning=reasoning,
                stage="adversarial_auditor",
                system_prompt=AUDITOR_SYSTEM_PROMPT,
                response_schema=AUDITOR_RESPONSE_SCHEMA,
                calls=calls,
            )
            rubric_scores = {
                key: _score_value(rubric.get(key), key) for key in REASONING_DIMENSIONS
            }
            auditor_scores = {
                key: _score_value(auditor.get(key), key) for key in REASONING_DIMENSIONS
            }
            violations = _validate_violations(auditor.get("violations"))
            dimension_minima = {
                key: min(rubric_scores[key], auditor_scores[key])
                for key in REASONING_DIMENSIONS
            }
            final_dimensions, caps = apply_reasoning_hard_caps(
                dimension_minima,
                violations,
            )
        except Exception as exc:
            usage = _sum_call_usage(calls)
            if isinstance(exc, ReasoningJudgeError):
                nested_calls = list(exc.calls)
                if nested_calls and not calls:
                    calls.extend(nested_calls)
                    usage = _sum_call_usage(calls)
            raise ReasoningJudgeError(
                f"invalid reasoning judge response: {exc}",
                token_usage=usage,
                calls=calls,
                failed_stage=getattr(exc, "failed_stage", None),
                transport_attempt_count=getattr(
                    exc, "transport_attempt_count", None
                ),
                transport_rejections=getattr(
                    exc, "transport_rejections", ()
                ),
            ) from exc
        return ReasoningJudgeOutcome(
            rubric_scores=rubric_scores,
            auditor_scores=auditor_scores,
            violations=violations,
            final_dimensions=final_dimensions,
            hard_caps=caps,
            total_usage=_sum_call_usage(calls),
            calls=tuple(calls),
        )

    def _run_stage(
        self,
        *,
        reasoning: str,
        stage: str,
        system_prompt: str,
        response_schema: Mapping[str, Any],
        calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        structured_output_mode = str(
            getattr(
                getattr(self.client, "config", None),
                "structured_output_mode",
                STRUCTURED_OUTPUT_NATIVE,
            )
        )
        if structured_output_mode not in {
            STRUCTURED_OUTPUT_LOCAL,
            STRUCTURED_OUTPUT_NATIVE,
        }:
            raise ValueError(
                f"unsupported reasoning judge structured output mode: "
                f"{structured_output_mode}"
            )
        if self.stage_intent is not None:
            self.stage_intent(stage)
        try:
            response = self.client.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": reasoning},
                ],
                **(
                    {
                        "response_schema": response_schema,
                        "schema_name": f"b_reasoning_{stage}_v2",
                    }
                    if structured_output_mode == STRUCTURED_OUTPUT_NATIVE
                    else {}
                ),
            )
        except Exception as exc:
            raise ReasoningJudgeError(
                str(exc),
                token_usage=_sum_call_usage(calls),
                calls=calls,
                failed_stage=stage,
                transport_attempt_count=getattr(
                    exc, "transport_attempt_count", None
                ),
                transport_rejections=getattr(
                    exc, "transport_rejections", ()
                ),
            ) from exc
        usage = response.token_usage.to_dict()
        call = {
            "stage": stage,
            "model_name": str(
                getattr(getattr(self.client, "config", None), "model_name", REASONING_JUDGE_MODEL)
            ),
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "schema_sha256": _payload_sha256(response_schema),
            "response_format_mode": response.response_format_mode,
            "token_usage": usage,
            "raw_usage": dict((response.raw_payload or {}).get("usage") or {}),
            "raw_response_sha256": _payload_sha256(response.raw_payload),
            "transport_attempt_count": int(
                getattr(response, "transport_attempt_count", 1)
            ),
            "transport_rejections": list(
                getattr(response, "transport_rejections", ())
            ),
        }
        calls.append(call)
        if self.stage_checkpoint is not None:
            self.stage_checkpoint(call)
        payload = json.loads(response.content)
        if not isinstance(payload, dict):
            raise ValueError(f"{stage} response must be an object")
        expected_fields = set(response_schema["required"])
        if set(payload) != expected_fields:
            raise ValueError(
                f"{stage} response fields must be exactly "
                f"{sorted(expected_fields)}"
            )
        return payload


def reasoning_prompt_fingerprint() -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "scorer": SCORER_SYSTEM_PROMPT,
        "auditor": AUDITOR_SYSTEM_PROMPT,
    }
    return _payload_sha256(payload)


def reasoning_schema_fingerprint() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scorer": SCORER_RESPONSE_SCHEMA,
        "auditor": AUDITOR_RESPONSE_SCHEMA,
        "hard_caps": HARD_CAPS,
    }
    return _payload_sha256(payload)


def run_reasoning_evaluation(
    *,
    submission_path: Path,
    questions: Sequence[BQuestion],
    model_config: ModelConfig,
    output_dir: Path,
    workers: int = 4,
    evaluator_factory: EvaluatorFactory | None = None,
    evaluator_factory_identity: str | None = None,
    accuracy_score: float | None = None,
    accuracy_source: str = "",
) -> ReasoningEvaluationRunResult:
    destination = Path(output_dir).resolve()
    with _exclusive_evaluation_lock(destination):
        return _run_reasoning_evaluation_locked(
            submission_path=submission_path,
            questions=questions,
            model_config=model_config,
            output_dir=destination,
            workers=workers,
            evaluator_factory=evaluator_factory,
            evaluator_factory_identity=evaluator_factory_identity,
            accuracy_score=accuracy_score,
            accuracy_source=accuracy_source,
        )


def _run_reasoning_evaluation_locked(
    *,
    submission_path: Path,
    questions: Sequence[BQuestion],
    model_config: ModelConfig,
    output_dir: Path,
    workers: int,
    evaluator_factory: EvaluatorFactory | None,
    evaluator_factory_identity: str | None,
    accuracy_score: float | None,
    accuracy_source: str,
) -> ReasoningEvaluationRunResult:
    """Score final CSV reasoning exactly under the local new.md rubric mirror.

    The judge receives only each reasoning string. Question and answer objects are
    used solely to validate the final CSV contract and never enter model messages.
    """

    source = Path(submission_path).resolve()
    destination = Path(output_dir).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if model_config.model_name.strip().lower() != REASONING_JUDGE_MODEL:
        raise ValueError(f"reasoning judge model must be {REASONING_JUDGE_MODEL}")
    if float(model_config.temperature) != 0.0:
        raise ValueError("the fixed reasoning judge requires temperature=0")
    if evaluator_factory is not None and not (
        isinstance(evaluator_factory_identity, str)
        and evaluator_factory_identity.strip()
    ):
        raise ValueError(
            "custom evaluator_factory requires an explicit stable identity"
        )
    if evaluator_factory is None and evaluator_factory_identity is not None:
        raise ValueError(
            "evaluator_factory_identity is only valid with a custom factory"
        )

    answers = validate_b_submission(source, questions, audit_ready=False)
    sealed_reasoning = [
        {"qid": answer.qid, "reasoning": answer.reasoning} for answer in answers
    ]
    expected_qids = {item["qid"] for item in sealed_reasoning}
    if len(expected_qids) != len(sealed_reasoning):
        raise ValueError("submission reasoning contains duplicate qids")
    token_total = sum(int(answer.total_tokens or 0) for answer in answers)
    evaluator_identity = _evaluator_identity(
        model_config,
        evaluator_factory=evaluator_factory,
        evaluator_factory_identity=evaluator_factory_identity,
    )
    fingerprint = _build_fingerprint(
        evaluator_identity=evaluator_identity,
        submission_path=source,
        sealed_reasoning=sealed_reasoning,
        has_summary=_has_summary_row(source),
        accuracy_score=accuracy_score,
        accuracy_source=accuracy_source,
    )
    manifest, sealed, resumed = _prepare_evaluation(
        output_dir=destination,
        submission_path=source,
        evaluator_identity=evaluator_identity,
        fingerprint=fingerprint,
        sealed_reasoning=sealed_reasoning,
    )
    reasoning_by_qid = {str(item["qid"]): str(item["reasoning"]) for item in sealed}
    run_instance_id = str(manifest["run_instance_id"])
    (
        evaluations,
        usage_by_qid,
        calls_by_qid,
        failures,
    ) = _load_evaluation_checkpoints(
        destination,
        expected_qids,
        require_call_trace=evaluator_factory is None,
        reasoning_by_qid=reasoning_by_qid,
        run_fingerprint=str(fingerprint["sha256"]),
        run_instance_id=run_instance_id,
        repair_partial_mirrors=manifest.get("status") == "running",
    )
    completed_qids = set(evaluations)

    if manifest.get("status") == "complete" and completed_qids == expected_qids:
        return _validate_completed_result(
            manifest=manifest,
            evaluations=evaluations,
            usage_by_qid=usage_by_qid,
            calls_by_qid=calls_by_qid,
            expected_qids=expected_qids,
            token_total=token_total,
            accuracy_score=accuracy_score,
            accuracy_source=accuracy_source,
            require_call_trace=evaluator_factory is None,
            output_dir=destination,
            reasoning_by_qid=reasoning_by_qid,
            failures=failures,
        )

    remaining_qids = sorted(expected_qids - completed_qids)

    def evaluate_one(
        qid: str,
    ) -> tuple[
        ReasoningEvaluation,
        dict[str, int],
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        reasoning = reasoning_by_qid[qid]
        started_stages: list[str] = []
        if _non_whitespace_length(reasoning) < MIN_REASONING_NON_WHITESPACE:
            return (
                _zero_evaluation(qid, "below_minimum_length"),
                _zero_usage(),
                None,
                [],
            )
        try:
            def record_stage_intent(stage: str) -> None:
                started_stages.append(stage)
                _record_reasoning_stage_intent(
                    destination,
                    qid=qid,
                    reasoning=reasoning,
                    run_fingerprint=str(fingerprint["sha256"]),
                    run_instance_id=run_instance_id,
                    stage=stage,
                )

            evaluator = (
                evaluator_factory()
                if evaluator_factory is not None
                else FixedReasoningEvaluator(
                    OpenAICompatibleClient(model_config),
                    stage_intent=record_stage_intent,
                    stage_checkpoint=lambda call: _append_reasoning_stage_call(
                        destination,
                        qid=qid,
                        reasoning=reasoning,
                        run_fingerprint=str(fingerprint["sha256"]),
                        run_instance_id=run_instance_id,
                        call=call,
                    ),
                )
            )
            outcome = evaluator.evaluate(reasoning)
            if isinstance(outcome, ReasoningJudgeOutcome):
                dimensions = outcome.final_dimensions
                usage = outcome.total_usage
                calls = [dict(call) for call in outcome.calls]
                rubric_scores = dict(outcome.rubric_scores)
                auditor_scores = dict(outcome.auditor_scores)
                violations = tuple(outcome.violations)
                hard_caps = dict(outcome.hard_caps)
            else:
                dimensions, usage = outcome
                calls = []
                rubric_scores = {}
                auditor_scores = {}
                violations = ()
                hard_caps = {}
            evaluation = ReasoningEvaluation(
                qid=qid,
                logical=_score_value(dimensions.get("logical"), "logical"),
                completeness=_score_value(dimensions.get("completeness"), "completeness"),
                clarity=_score_value(dimensions.get("clarity"), "clarity"),
                rubric_scores=rubric_scores,
                auditor_scores=auditor_scores,
                hard_cap_violations=violations,
                hard_caps=hard_caps,
            )
            return evaluation, _validate_usage(usage, qid), None, calls
        except Exception as exc:
            usage = _validate_usage(
                getattr(exc, "token_usage", _zero_usage()), qid
            )
            calls = [dict(call) for call in getattr(exc, "calls", ())]
            observed_stages = {
                str(call.get("stage")) for call in calls
            }
            unobserved_stages = [
                stage for stage in started_stages
                if stage not in observed_stages
            ]
            failed_transport = _failed_transport_record(exc)
            terminal_rejection_accounts_for_intent = bool(
                failed_transport
                and len(unobserved_stages) == 1
                and failed_transport["stage"] == unobserved_stages[0]
                and failed_transport["all_attempts_rejected_pre_generation"]
            )
            unobservable_usage_risk = (
                bool(unobserved_stages)
                and not terminal_rejection_accounts_for_intent
            )
            if evaluator_factory is not None and not started_stages:
                unobservable_usage_risk = not calls
            failure = {
                "qid": qid,
                "error_type": exc.__class__.__name__,
                "error": _sanitize_error(str(exc), model_config),
                "unobservable_usage_risk": unobservable_usage_risk,
                "unobserved_stages": unobserved_stages,
                **(
                    {"failed_transport": failed_transport}
                    if failed_transport is not None
                    else {}
                ),
            }
            return _zero_evaluation(qid, "judge_error"), usage, failure, calls

    if remaining_qids:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_one, qid): qid for qid in remaining_qids}
            for future in as_completed(futures):
                qid = futures[future]
                evaluation, usage, failure, calls = future.result()
                evaluations[qid] = evaluation
                usage_by_qid[qid] = usage
                calls_by_qid[qid] = calls
                if failure is not None:
                    failures = [item for item in failures if item.get("qid") != qid]
                    failures.append(failure)
                _persist_partial(
                    destination,
                    evaluations,
                    usage_by_qid,
                    calls_by_qid,
                    failures,
                    reasoning_by_qid=reasoning_by_qid,
                    run_fingerprint=str(fingerprint["sha256"]),
                    run_instance_id=run_instance_id,
                )

    aggregate = aggregate_reasoning(evaluations)
    scorecard = _build_scorecard(
        evaluations=evaluations,
        accuracy_score=accuracy_score,
        accuracy_source=accuracy_source,
        token_total=token_total,
        model_name=model_config.model_name,
    )
    judge_usage = _sum_usage(usage_by_qid)
    corpus_audit = audit_reasoning_corpus(reasoning_by_qid)
    write_json(destination / "reasoning_aggregate.json", aggregate)
    write_json(destination / "reasoning_judge_usage.json", {
        "by_qid": usage_by_qid,
        "calls_by_qid": calls_by_qid,
        "total": judge_usage,
        "included_in_submission_token_score": False,
    })
    write_json(destination / "reasoning_corpus_audit.json", corpus_audit)
    if scorecard is not None:
        write_json(destination / "scorecard.json", scorecard)
    manifest.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "status": "complete",
            "resumed": resumed,
            "resumed_evaluation_count": len(completed_qids),
            "expected_reasoning_count": len(expected_qids),
            "evaluated_reasoning_count": len(evaluations),
            "failure_count": len(failures),
            "scored_zero_due_to_failure_count": sum(
                item.status == "judge_error" for item in evaluations.values()
            ),
            "below_minimum_length_count": sum(
                item.status == "below_minimum_length" for item in evaluations.values()
            ),
            "reasoning_aggregate": aggregate,
            "judge_token_usage": judge_usage,
            "judge_tokens_included_in_submission": False,
            "unobservable_usage_risk": any(
                bool(item.get("unobservable_usage_risk"))
                for item in failures
            ),
            "submission_token_total": token_total,
            "corpus_audit": corpus_audit,
            "scorecard": scorecard,
        }
    )
    write_json(destination / "reasoning_evaluator_manifest.json", manifest)
    return ReasoningEvaluationRunResult(
        manifest=dict(manifest),
        aggregate=aggregate,
        evaluations=dict(evaluations),
        scorecard=scorecard,
    )


def aggregate_reasoning(
    evaluations: Mapping[str, ReasoningEvaluation],
) -> dict[str, Any]:
    ordered = [evaluations[qid] for qid in sorted(evaluations)]
    scores = [item.reasoning_score for item in ordered]
    status_counts = Counter(item.status for item in ordered)
    return {
        "question_count": len(ordered),
        "reasoning_score": fmean(scores) if scores else 0.0,
        "dimension_means": {
            key: fmean(getattr(item, key) for item in ordered) if ordered else 0.0
            for key in REASONING_DIMENSIONS
        },
        "p10": _percentile(scores, 0.10),
        "zero_score_count": sum(score == 0 for score in scores),
        "status_counts": dict(sorted(status_counts.items())),
    }


def apply_reasoning_hard_caps(
    dimensions: Mapping[str, Any],
    violations: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply only the frozen, explicitly declared v2 hard caps.

    The function never infers a violation from text and never raises a score.
    The adversarial auditor declares violations; this deterministic seam merely
    applies the published cap table.
    """

    final = {
        key: _score_value(dimensions.get(key), key) for key in REASONING_DIMENSIONS
    }
    normalized = _validate_violations(list(violations))
    applied: dict[str, float] = {}
    for violation in normalized:
        for dimension, cap in HARD_CAPS[violation].items():
            previous = applied.get(dimension, 100.0)
            applied[dimension] = min(previous, cap)
            final[dimension] = min(final[dimension], cap)
    return final, dict(sorted(applied.items()))


def build_reasoning_sentinels() -> tuple[dict[str, Any], ...]:
    """Return frozen, answer-free calibration texts for the 0/30/60/80/90 ladder."""

    return (
        {
            "sentinel_id": "sentinel_0",
            "target": 0,
            "reasoning": "无法判断。",
        },
        {
            "sentinel_id": "sentinel_30",
            "target": 30,
            "reasoning": (
                "根据材料进行核验并完成分析，相关内容能够支持最终判断，"
                "因此可以得到所述结论。"
            ),
        },
        {
            "sentinel_id": "sentinel_60",
            "target": 60,
            "reasoning": (
                "先定位披露段落，再提取两个年度的指标进行比较。后一期数值更高，"
                "所以判断其发生增长；但摘要没有列出原始数值和具体差额。结论为增长。"
            ),
        },
        {
            "sentinel_id": "sentinel_80",
            "target": 80,
            "reasoning": (
                "定位到同一主体连续两个年度的披露：前一期指标为80，后一期为100。"
                "按增长率=(100-80)÷80计算，结果为25%；主体、年份和单位保持一致。"
                "因此该指标同比增长25%，结论与计算结果一致。"
            ),
        },
        {
            "sentinel_id": "sentinel_90",
            "target": 90,
            "reasoning": (
                "定位并核对同一主体2024年与2025年的原始披露，指标分别为80万元和"
                "100万元。先统一万元口径，再按(100-80)÷80×100%计算同比增幅，"
                "中间值为25%，无需额外单位换算；按要求保留两位小数得到25.00%。"
                "数值提取、公式方向、代入和舍入相互一致。结论：同比增长25.00%。"
            ),
        },
    )


def validate_reasoning_sentinels(
    evaluations: Mapping[str, ReasoningEvaluation],
) -> dict[str, Any]:
    """Validate coverage and strict score monotonicity of the frozen ladder."""

    sentinels = build_reasoning_sentinels()
    expected = [str(item["sentinel_id"]) for item in sentinels]
    failures: list[str] = []
    if set(evaluations) != set(expected):
        failures.append(
            "sentinel coverage mismatch: "
            f"missing={sorted(set(expected) - set(evaluations))}, "
            f"extra={sorted(set(evaluations) - set(expected))}"
        )
    scores = {
        sentinel_id: evaluations[sentinel_id].reasoning_score
        for sentinel_id in expected
        if sentinel_id in evaluations
    }
    for lower, upper in zip(expected, expected[1:]):
        if lower in scores and upper in scores and not scores[lower] < scores[upper]:
            failures.append(
                f"{upper} must score above {lower}: "
                f"{scores[upper]:.4f} <= {scores[lower]:.4f}"
            )
    if "sentinel_0" in scores and scores["sentinel_0"] != 0.0:
        failures.append(f"sentinel_0 must score exactly 0, got {scores['sentinel_0']:.4f}")
    return {
        "passed": not failures,
        "expected_order": expected,
        "scores": scores,
        "failures": failures,
    }


def audit_reasoning_corpus(
    reasoning_by_qid: Mapping[str, str],
    *,
    similarity_threshold: float = 0.90,
) -> dict[str, Any]:
    """Flag repeated reasoning across a corpus without mutating item scores."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    normalized = {
        str(qid): _normalize_reasoning_for_template_audit(str(reasoning))
        for qid, reasoning in reasoning_by_qid.items()
    }
    exact_groups = _group_equal_values(
        {
            str(qid): re.sub(r"\s+", "", str(reasoning))
            for qid, reasoning in reasoning_by_qid.items()
        }
    )
    qids = sorted(normalized)
    graph: dict[str, set[str]] = {qid: set() for qid in qids}
    similar_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(qids):
        left_text = normalized[left]
        if len(left_text) < MIN_REASONING_NON_WHITESPACE:
            continue
        for right in qids[index + 1 :]:
            right_text = normalized[right]
            if len(right_text) < MIN_REASONING_NON_WHITESPACE:
                continue
            ratio = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if ratio < similarity_threshold:
                continue
            graph[left].add(right)
            graph[right].add(left)
            similar_pairs.append(
                {"left": left, "right": right, "similarity": round(ratio, 6)}
            )
    clusters = _connected_similarity_clusters(graph)
    flagged = sorted({qid for cluster in clusters for qid in cluster})
    return {
        "question_count": len(reasoning_by_qid),
        "similarity_threshold": similarity_threshold,
        "exact_duplicate_groups": exact_groups,
        "template_clusters": clusters,
        "similar_pairs": similar_pairs,
        "flagged_qids": flagged,
        "scores_mutated": False,
        "suggested_reasoning_cap": 29,
        "recommendation": (
            "人工复核高度模板化簇；该建议上限未自动应用到任何单题分数。"
            if flagged
            else ""
        ),
    }


def _prepare_evaluation(
    *,
    output_dir: Path,
    submission_path: Path,
    evaluator_identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    sealed_reasoning: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]], bool]:
    manifest_path = output_dir / "reasoning_evaluator_manifest.json"
    sealed_path = output_dir / "sealed_reasoning.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _validate_existing_manifest(
            manifest,
            evaluator_identity,
            fingerprint,
            submission_path=submission_path,
            output_dir=output_dir,
        )
        if not sealed_path.exists():
            raise ReasoningEvaluationFingerprintError(
                "existing reasoning evaluation has no sealed_reasoning.json"
            )
        sealed = read_json(sealed_path)
        if _payload_sha256(sealed) != fingerprint["components"]["sealed_reasoning"]["sha256"]:
            raise ReasoningEvaluationFingerprintError("sealed reasoning changed after evaluation started")
        if _payload_sha256(sealed_reasoning) != _payload_sha256(sealed):
            raise ReasoningEvaluationFingerprintError(
                "submission reasoning differs from the frozen sealed input"
            )
        return manifest, sealed, True

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReasoningEvaluationFingerprintError(
            "reasoning evaluation artifacts exist without a manifest"
        )
    ensure_dir(output_dir)
    write_json(sealed_path, sealed_reasoning)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_instance_id": uuid4().hex,
        "output_dir": str(output_dir),
        "submission_path": str(submission_path),
        "evaluator_identity": dict(evaluator_identity),
        "judge_contract": {
            "schema_sha256": reasoning_schema_fingerprint(),
            "hard_caps_sha256": _payload_sha256(HARD_CAPS),
        },
        "fingerprint": dict(fingerprint),
        "sealed_reasoning_path": str(sealed_path),
        "status": "running",
    }
    write_json(manifest_path, manifest)
    return manifest, sealed_reasoning, False


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    *,
    submission_path: Path,
    output_dir: Path,
) -> None:
    if not isinstance(manifest.get("run_instance_id"), str) or not str(
        manifest.get("run_instance_id")
    ):
        raise ReasoningEvaluationFingerprintError(
            "existing reasoning evaluation has no immutable run instance id"
        )
    if manifest.get("submission_path") != str(submission_path):
        raise ReasoningEvaluationFingerprintError(
            "reasoning evaluation submission path changed"
        )
    if manifest.get("output_dir") != str(output_dir):
        raise ReasoningEvaluationFingerprintError(
            "reasoning evaluation output directory changed"
        )
    if manifest.get("sealed_reasoning_path") != str(
        output_dir / "sealed_reasoning.json"
    ):
        raise ReasoningEvaluationFingerprintError(
            "reasoning evaluation sealed path changed"
        )
    if manifest.get("evaluator_identity") != evaluator_identity:
        raise ReasoningEvaluationFingerprintError(
            "frozen reasoning evaluator identity changed; use a new output directory"
        )
    existing = manifest.get("fingerprint")
    if not isinstance(existing, Mapping):
        raise ReasoningEvaluationFingerprintError("existing reasoning evaluator has no fingerprint")
    if existing.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        raise ReasoningEvaluationFingerprintError("unsupported reasoning fingerprint schema")
    if existing.get("sha256") != _payload_sha256(existing.get("components")):
        raise ReasoningEvaluationFingerprintError("existing reasoning fingerprint was modified")
    if existing.get("sha256") != fingerprint.get("sha256"):
        changed = sorted(
            key
            for key in set(existing.get("components", {}))
            | set(fingerprint.get("components", {}))
            if existing.get("components", {}).get(key)
            != fingerprint.get("components", {}).get(key)
        )
        raise ReasoningEvaluationFingerprintError(
            "reasoning evaluation fingerprint mismatch "
            f"({', '.join(changed) or 'unknown components'})"
        )


def _load_partial(
    output_dir: Path, expected_qids: set[str]
) -> tuple[dict[str, ReasoningEvaluation], dict[str, dict[str, int]]]:
    evaluations: dict[str, ReasoningEvaluation] = {}
    scores_path = output_dir / "reasoning_scores.json"
    if scores_path.exists():
        rows = read_json(scores_path)
        if not isinstance(rows, list):
            raise ReasoningEvaluationFingerprintError("partial reasoning scores must be an array")
        for row in rows:
            evaluation = _evaluation_from_dict(row)
            if evaluation.qid not in expected_qids or evaluation.qid in evaluations:
                raise ReasoningEvaluationFingerprintError(
                    f"unexpected or duplicate partial reasoning qid {evaluation.qid!r}"
                )
            evaluations[evaluation.qid] = evaluation

    usage_by_qid: dict[str, dict[str, int]] = {}
    usage_path = output_dir / "reasoning_judge_usage_partial.json"
    if usage_path.exists():
        payload = read_json(usage_path)
        if not isinstance(payload, Mapping):
            raise ReasoningEvaluationFingerprintError("partial reasoning usage must be an object")
        for qid, usage in payload.items():
            if qid not in expected_qids or not isinstance(usage, Mapping):
                raise ReasoningEvaluationFingerprintError(
                    f"unexpected reasoning usage qid {qid!r}"
                )
            usage_by_qid[str(qid)] = _validate_usage(usage, str(qid))
    return evaluations, usage_by_qid


def _record_reasoning_stage_intent(
    output_dir: Path,
    *,
    qid: str,
    reasoning: str,
    run_fingerprint: str,
    run_instance_id: str,
    stage: str,
) -> None:
    if stage not in {"rubric_scorer", "adversarial_auditor"}:
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: unsupported Judge stage intent"
        )
    intent_dir = ensure_dir(output_dir / "reasoning_stage_intents")
    path = intent_dir / f"{qid}.{stage}.json"
    payload = {
        "qid": qid,
        "stage": stage,
        "run_fingerprint": run_fingerprint,
        "run_instance_id": run_instance_id,
        "sealed_reasoning_sha256": _payload_sha256(reasoning),
        "state": "provider_call_started",
    }
    if path.exists():
        if read_json(path) != payload:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: Judge stage intent changed"
            )
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: Judge stage intent already exists"
        )
    write_json(path, payload)


def _append_reasoning_stage_call(
    output_dir: Path,
    *,
    qid: str,
    reasoning: str,
    run_fingerprint: str,
    run_instance_id: str,
    call: Mapping[str, Any],
) -> None:
    """Persist one observed Judge stage without allowing call-history rewrites."""

    stage_dir = ensure_dir(output_dir / "reasoning_stage_ledgers")
    path = stage_dir / f"{qid}.json"
    if (output_dir / "reasoning_checkpoints" / f"{qid}.json").exists():
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: final checkpoint already exists"
        )
    expected_binding = {
        "qid": qid,
        "run_fingerprint": run_fingerprint,
        "run_instance_id": run_instance_id,
        "sealed_reasoning_sha256": _payload_sha256(reasoning),
    }
    calls: list[dict[str, Any]] = []
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, Mapping):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger must be an object"
            )
        for key, expected in expected_binding.items():
            if existing.get(key) != expected:
                raise ReasoningEvaluationFingerprintError(
                    f"{qid}: stage ledger {key} mismatch"
                )
        raw_calls = existing.get("calls")
        if not isinstance(raw_calls, list) or not all(
            isinstance(item, Mapping) for item in raw_calls
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger calls are invalid"
            )
        calls = [dict(item) for item in raw_calls]

    candidate = dict(call)
    expected_stages = ("rubric_scorer", "adversarial_auditor")
    if len(calls) >= len(expected_stages):
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: stage ledger is already complete"
        )
    if candidate.get("stage") != expected_stages[len(calls)]:
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: unexpected Judge stage order"
        )
    stage = str(candidate["stage"])
    intent_path = (
        output_dir / "reasoning_stage_intents" / f"{qid}.{stage}.json"
    )
    if not intent_path.exists():
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: observed Judge response has no pre-call intent"
        )
    expected_intent = {
        **expected_binding,
        "stage": stage,
        "state": "provider_call_started",
    }
    if read_json(intent_path) != expected_intent:
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: Judge stage intent binding mismatch"
        )
    _validate_stage_calls(qid, [*calls, candidate])
    calls.append(candidate)
    write_json(path, {**expected_binding, "calls": calls})


def _validate_stage_calls(
    qid: str,
    calls: Sequence[Mapping[str, Any]],
) -> None:
    expected_stages = ("rubric_scorer", "adversarial_auditor")
    if not calls or len(calls) > len(expected_stages):
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: stage ledger call count is invalid"
        )
    if tuple(call.get("stage") for call in calls) != expected_stages[: len(calls)]:
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: stage ledger order is invalid"
        )
    for index, call in enumerate(calls, start=1):
        raw_usage = call.get("raw_usage")
        if not isinstance(raw_usage, Mapping):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage call {index} has no provider raw usage"
            )
        raw = _validate_usage(raw_usage, f"{qid}.raw_stage_{index}")
        observed = _validate_usage(
            call.get("token_usage") or {},
            f"{qid}.stage_{index}",
        )
        if raw != observed:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage call {index} usage differs from provider raw usage"
            )
        if (
            "transport_attempt_count" not in call
            or "transport_rejections" not in call
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage call {index} transport audit is missing"
            )
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
                or item.get("status_code") != 429
                or item.get("pre_generation_rejection") is not True
                or item.get("token_usage_observed") is not False
                or item.get("attempt_index") != rejection_index
                for rejection_index, item in enumerate(rejections, start=1)
            )
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage call {index} transport audit is invalid"
            )


def _failed_transport_record(exc: Exception) -> dict[str, Any] | None:
    attempt_count = getattr(exc, "transport_attempt_count", None)
    raw_rejections = list(getattr(exc, "transport_rejections", ()))
    failed_stage = getattr(exc, "failed_stage", None)
    if attempt_count is None and not raw_rejections:
        return None
    rejections = [
        dict(item) for item in raw_rejections
        if isinstance(item, Mapping)
    ]
    valid = (
        not isinstance(attempt_count, bool)
        and isinstance(attempt_count, int)
        and attempt_count >= 1
        and len(rejections) == len(raw_rejections)
        and attempt_count == len(rejections)
        and all(
            item.get("attempt_index") == index
            and item.get("status_code") == 429
            and item.get("pre_generation_rejection") is True
            and item.get("token_usage_observed") is False
            for index, item in enumerate(rejections, start=1)
        )
        and failed_stage in {"rubric_scorer", "adversarial_auditor"}
    )
    if not valid:
        return {
            "stage": failed_stage,
            "transport_attempt_count": attempt_count,
            "transport_rejections": rejections,
            "transport_retry_policy_version": TRANSPORT_RETRY_POLICY_VERSION,
            "all_attempts_rejected_pre_generation": False,
        }
    return {
        "stage": failed_stage,
        "transport_attempt_count": attempt_count,
        "transport_rejections": rejections,
        "transport_retry_policy_version": TRANSPORT_RETRY_POLICY_VERSION,
        "all_attempts_rejected_pre_generation": True,
    }


def _load_evaluation_checkpoints(
    output_dir: Path,
    expected_qids: set[str],
    *,
    require_call_trace: bool,
    reasoning_by_qid: Mapping[str, str],
    run_fingerprint: str,
    run_instance_id: str,
    repair_partial_mirrors: bool,
) -> tuple[
    dict[str, ReasoningEvaluation],
    dict[str, dict[str, int]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    checkpoint_dir = output_dir / "reasoning_checkpoints"
    stage_dir = output_dir / "reasoning_stage_ledgers"
    intent_dir = output_dir / "reasoning_stage_intents"
    legacy_paths = (
        output_dir / "reasoning_scores.json",
        output_dir / "reasoning_judge_usage_partial.json",
        output_dir / "reasoning_judge_calls_partial.json",
        output_dir / "reasoning_judge_failures.jsonl",
    )
    if (
        not checkpoint_dir.exists()
        and not stage_dir.exists()
        and not intent_dir.exists()
    ):
        if any(path.exists() for path in legacy_paths):
            raise ReasoningEvaluationFingerprintError(
                "legacy partial reasoning artifacts have no transactional "
                "per-qid checkpoints"
            )
        return {}, {}, {}, []
    if not checkpoint_dir.exists() and any(
        path.exists() for path in legacy_paths
    ):
        raise ReasoningEvaluationFingerprintError(
            "partial reasoning mirrors exist without canonical checkpoints"
        )

    evaluations: dict[str, ReasoningEvaluation] = {}
    usage_by_qid: dict[str, dict[str, int]] = {}
    calls_by_qid: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for path in (
        sorted(checkpoint_dir.glob("*.json"))
        if checkpoint_dir.exists()
        else []
    ):
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ReasoningEvaluationFingerprintError(
                f"{path}: reasoning checkpoint must be an object"
            )
        qid = str(payload.get("qid", ""))
        if (
            qid not in expected_qids
            or path.stem != qid
            or qid in evaluations
        ):
            raise ReasoningEvaluationFingerprintError(
                f"unexpected or duplicate reasoning checkpoint {qid!r}"
            )
        if payload.get("run_fingerprint") != run_fingerprint:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint run fingerprint mismatch"
            )
        if payload.get("run_instance_id") != run_instance_id:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint run instance mismatch"
            )
        if payload.get("sealed_reasoning_sha256") != _payload_sha256(
            reasoning_by_qid[qid]
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint sealed reasoning mismatch"
            )
        evaluation = _evaluation_from_dict(payload.get("evaluation"))
        if evaluation.qid != qid:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint evaluation qid mismatch"
            )
        usage = _validate_usage(
            payload.get("token_usage") or {},
            qid,
        )
        raw_calls = payload.get("calls")
        if not isinstance(raw_calls, list) or not all(
            isinstance(call, Mapping) for call in raw_calls
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint calls must be an array of objects"
            )
        calls = [dict(call) for call in raw_calls]
        _validate_checkpoint_call_usage(
            qid,
            evaluation=evaluation,
            usage=usage,
            calls=calls,
            require_call_trace=require_call_trace,
        )
        failure = payload.get("failure")
        if failure is not None:
            if not isinstance(failure, Mapping) or failure.get("qid") != qid:
                raise ReasoningEvaluationFingerprintError(
                    f"{qid}: checkpoint failure is invalid"
                )
            failures.append(dict(failure))
        if evaluation.status == "judge_error" and failure is None:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: judge_error checkpoint has no failure record"
            )
        if evaluation.status != "judge_error" and failure is not None:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: non-failed checkpoint contains a failure record"
            )
        evaluations[qid] = evaluation
        usage_by_qid[qid] = usage
        calls_by_qid[qid] = calls

    recovered_orphan = False
    for path in (
        sorted(stage_dir.glob("*.json")) if stage_dir.exists() else []
    ):
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ReasoningEvaluationFingerprintError(
                f"{path}: stage ledger must be an object"
            )
        qid = str(payload.get("qid", ""))
        if qid not in expected_qids or path.stem != qid:
            raise ReasoningEvaluationFingerprintError(
                f"unexpected reasoning stage ledger {qid!r}"
            )
        if payload.get("run_fingerprint") != run_fingerprint:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger run fingerprint mismatch"
            )
        if payload.get("run_instance_id") != run_instance_id:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger run instance mismatch"
            )
        if payload.get("sealed_reasoning_sha256") != _payload_sha256(
            reasoning_by_qid[qid]
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger sealed reasoning mismatch"
            )
        raw_calls = payload.get("calls")
        if not isinstance(raw_calls, list) or not all(
            isinstance(call, Mapping) for call in raw_calls
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage ledger calls are invalid"
            )
        stage_calls = [dict(call) for call in raw_calls]
        _validate_stage_calls(qid, stage_calls)
        stage_usage = _sum_call_usage(stage_calls)
        if qid in evaluations:
            if stage_calls != calls_by_qid[qid]:
                raise ReasoningEvaluationFingerprintError(
                    f"{qid}: final checkpoint differs from stage ledger"
                )
            continue
        evaluation = _zero_evaluation(qid, "judge_error")
        failure = {
            "qid": qid,
            "error_type": "OrphanStageCheckpoint",
            "error": (
                "judge stage usage was observed before an interrupted item "
                "transaction; the item is conservatively scored zero and is "
                "not resent"
            ),
            "unobservable_usage_risk": False,
        }
        _validate_checkpoint_call_usage(
            qid,
            evaluation=evaluation,
            usage=stage_usage,
            calls=stage_calls,
            require_call_trace=True,
        )
        evaluations[qid] = evaluation
        usage_by_qid[qid] = stage_usage
        calls_by_qid[qid] = stage_calls
        failures.append(failure)
        recovered_orphan = True

    failure_by_qid = {
        str(item.get("qid")): dict(item) for item in failures
    }
    for path in (
        sorted(intent_dir.glob("*.json")) if intent_dir.exists() else []
    ):
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ReasoningEvaluationFingerprintError(
                f"{path}: stage intent must be an object"
            )
        qid = str(payload.get("qid", ""))
        stage = str(payload.get("stage", ""))
        if (
            qid not in expected_qids
            or path.name != f"{qid}.{stage}.json"
            or stage not in {"rubric_scorer", "adversarial_auditor"}
        ):
            raise ReasoningEvaluationFingerprintError(
                f"unexpected reasoning stage intent {path.name!r}"
            )
        if payload.get("run_fingerprint") != run_fingerprint:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage intent run fingerprint mismatch"
            )
        if payload.get("run_instance_id") != run_instance_id:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage intent run instance mismatch"
            )
        if payload.get("sealed_reasoning_sha256") != _payload_sha256(
            reasoning_by_qid[qid]
        ):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: stage intent sealed reasoning mismatch"
            )
        if payload.get("state") != "provider_call_started":
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: unsupported stage intent state"
            )
        observed_stages = {
            str(call.get("stage")) for call in calls_by_qid.get(qid, [])
        }
        if stage in observed_stages:
            continue
        if qid in evaluations:
            failure = failure_by_qid.get(qid)
            if (
                evaluations[qid].status == "judge_error"
                and failure is not None
            ):
                if failure.get("unobservable_usage_risk") is True:
                    continue
                if _failure_accounts_for_stage_intent(failure, stage):
                    continue
                replacement = {
                    "qid": qid,
                    "error_type": "UnobservableStageIntent",
                    "error": (
                        "a provider call started without an observable response; "
                        "the item is conservatively scored zero and is not resent"
                    ),
                    "unobservable_usage_risk": True,
                    "unobserved_stages": [stage],
                }
                failures = [
                    item for item in failures
                    if str(item.get("qid")) != qid
                ]
                failures.append(replacement)
                failure_by_qid[qid] = replacement
                recovered_orphan = True
                continue
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: final checkpoint does not account for stage intent"
            )
        evaluation = _zero_evaluation(qid, "judge_error")
        failure = {
            "qid": qid,
            "error_type": "UnobservableStageIntent",
            "error": (
                "a provider call started without an observable response; "
                "the item is conservatively scored zero and is not resent"
            ),
            "unobservable_usage_risk": True,
        }
        stage_calls = calls_by_qid.get(qid, [])
        evaluations[qid] = evaluation
        usage_by_qid[qid] = _sum_call_usage(stage_calls)
        calls_by_qid[qid] = stage_calls
        failures.append(failure)
        failure_by_qid[qid] = failure
        recovered_orphan = True

    if recovered_orphan or repair_partial_mirrors:
        _persist_partial(
            output_dir,
            evaluations,
            usage_by_qid,
            calls_by_qid,
            failures,
            reasoning_by_qid=reasoning_by_qid,
            run_fingerprint=run_fingerprint,
            run_instance_id=run_instance_id,
        )

    _validate_legacy_partial_mirrors(
        output_dir,
        evaluations=evaluations,
        usage_by_qid=usage_by_qid,
        calls_by_qid=calls_by_qid,
        failures=failures,
    )
    return evaluations, usage_by_qid, calls_by_qid, failures


def _failure_accounts_for_stage_intent(
    failure: Mapping[str, Any],
    stage: str,
) -> bool:
    failed_transport = failure.get("failed_transport")
    return bool(
        isinstance(failed_transport, Mapping)
        and failed_transport.get("stage") == stage
        and failed_transport.get("all_attempts_rejected_pre_generation") is True
        and failed_transport.get("transport_retry_policy_version")
        == TRANSPORT_RETRY_POLICY_VERSION
    )


def _validate_checkpoint_call_usage(
    qid: str,
    *,
    evaluation: ReasoningEvaluation,
    usage: Mapping[str, int],
    calls: Sequence[Mapping[str, Any]],
    require_call_trace: bool,
) -> None:
    if calls:
        call_usage = _sum_call_usage(calls)
        if call_usage != dict(usage):
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: checkpoint call usage differs from qid usage"
            )
        for index, call in enumerate(calls, start=1):
            raw_usage = call.get("raw_usage")
            if not isinstance(raw_usage, Mapping):
                raise ReasoningEvaluationFingerprintError(
                    f"{qid}: call {index} has no provider raw usage"
                )
            if _validate_usage(raw_usage, f"{qid}.raw_call_{index}") != (
                _validate_usage(
                    call.get("token_usage") or {},
                    f"{qid}.call_{index}",
                )
            ):
                raise ReasoningEvaluationFingerprintError(
                    f"{qid}: call {index} usage differs from provider raw usage"
                )
    elif any(int(value) for value in usage.values()) and require_call_trace:
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: non-zero judge usage has no call trace"
        )
    if (
        evaluation.status == "below_minimum_length"
        and (calls or any(int(value) for value in usage.values()))
    ):
        raise ReasoningEvaluationFingerprintError(
            f"{qid}: below-minimum reasoning must not call the judge"
        )


def _validate_legacy_partial_mirrors(
    output_dir: Path,
    *,
    evaluations: Mapping[str, ReasoningEvaluation],
    usage_by_qid: Mapping[str, Mapping[str, int]],
    calls_by_qid: Mapping[str, Sequence[Mapping[str, Any]]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    expected_rows = [
        evaluations[qid].to_dict() for qid in sorted(evaluations)
    ]
    mirrors: tuple[tuple[Path, Any], ...] = (
        (output_dir / "reasoning_scores.json", expected_rows),
        (
            output_dir / "reasoning_judge_usage_partial.json",
            dict(usage_by_qid),
        ),
        (
            output_dir / "reasoning_judge_calls_partial.json",
            {
                qid: list(calls_by_qid[qid])
                for qid in sorted(calls_by_qid)
            },
        ),
    )
    for path, expected in mirrors:
        if not path.exists() or read_json(path) != expected:
            raise ReasoningEvaluationFingerprintError(
                f"partial reasoning mirror drifted: {path.name}"
            )
    failure_path = output_dir / "reasoning_judge_failures.jsonl"
    if (
        not failure_path.exists()
        or _read_jsonl(failure_path) != _canonical_failures(failures)
    ):
        raise ReasoningEvaluationFingerprintError(
            "partial reasoning failure mirror drifted"
        )


def _persist_partial(
    output_dir: Path,
    evaluations: Mapping[str, ReasoningEvaluation],
    usage_by_qid: Mapping[str, Mapping[str, int]],
    calls_by_qid: Mapping[str, Sequence[Mapping[str, Any]]],
    failures: Sequence[Mapping[str, Any]],
    *,
    reasoning_by_qid: Mapping[str, str],
    run_fingerprint: str,
    run_instance_id: str,
) -> None:
    failure_by_qid = {
        str(item.get("qid")): dict(item) for item in failures
    }
    checkpoint_dir = output_dir / "reasoning_checkpoints"
    ensure_dir(checkpoint_dir)
    for qid in sorted(evaluations):
        if qid not in usage_by_qid or qid not in calls_by_qid:
            raise ReasoningEvaluationFingerprintError(
                f"{qid}: partial checkpoint is not transactionally complete"
            )
        write_json(
            checkpoint_dir / f"{qid}.json",
            {
                "qid": qid,
                "run_fingerprint": run_fingerprint,
                "run_instance_id": run_instance_id,
                "sealed_reasoning_sha256": _payload_sha256(
                    reasoning_by_qid[qid]
                ),
                "evaluation": evaluations[qid].to_dict(),
                "token_usage": dict(usage_by_qid[qid]),
                "calls": [dict(call) for call in calls_by_qid[qid]],
                "failure": failure_by_qid.get(qid),
            },
        )
    rows = [evaluations[qid].to_dict() for qid in sorted(evaluations)]
    write_json(output_dir / "reasoning_scores.json", rows)
    write_jsonl(output_dir / "reasoning_scores.jsonl", rows)
    write_json(output_dir / "reasoning_judge_usage_partial.json", usage_by_qid)
    write_json(output_dir / "reasoning_judge_calls_partial.json", calls_by_qid)
    write_jsonl(
        output_dir / "reasoning_judge_failures.jsonl",
        _canonical_failures(failures),
    )


def _canonical_failures(
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(item) for item in failures),
        key=lambda item: (
            str(item.get("qid", "")),
            str(item.get("error_type", "")),
        ),
    )


def _load_call_traces_partial(
    output_dir: Path,
    expected_qids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    path = output_dir / "reasoning_judge_calls_partial.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ReasoningEvaluationFingerprintError(
            "partial reasoning judge calls must be an object"
        )
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, calls in payload.items():
        if str(qid) not in expected_qids or not isinstance(calls, list):
            raise ReasoningEvaluationFingerprintError(
                f"unexpected reasoning judge calls qid {qid!r}"
            )
        result[str(qid)] = [dict(call) for call in calls if isinstance(call, Mapping)]
        if len(result[str(qid)]) != len(calls):
            raise ReasoningEvaluationFingerprintError(
                f"reasoning judge calls for {qid!r} contain a non-object row"
            )
    return result


def _validate_completed_result(
    *,
    manifest: Mapping[str, Any],
    evaluations: Mapping[str, ReasoningEvaluation],
    usage_by_qid: Mapping[str, Mapping[str, int]],
    calls_by_qid: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_qids: set[str],
    token_total: int,
    accuracy_score: float | None,
    accuracy_source: str,
    require_call_trace: bool,
    output_dir: Path,
    reasoning_by_qid: Mapping[str, str],
    failures: Sequence[Mapping[str, Any]],
) -> ReasoningEvaluationRunResult:
    if (
        set(evaluations) != expected_qids
        or set(usage_by_qid) != expected_qids
        or set(calls_by_qid) != expected_qids
    ):
        raise ReasoningEvaluationFingerprintError("completed reasoning coverage is incomplete")
    for qid in sorted(expected_qids):
        _validate_checkpoint_call_usage(
            qid,
            evaluation=evaluations[qid],
            usage=usage_by_qid[qid],
            calls=calls_by_qid[qid],
            require_call_trace=require_call_trace,
        )
    aggregate = aggregate_reasoning(evaluations)
    scorecard = _build_scorecard(
        evaluations=evaluations,
        accuracy_score=accuracy_score,
        accuracy_source=accuracy_source,
        token_total=token_total,
        model_name=str(dict(manifest.get("evaluator_identity") or {}).get("model_name", "")),
    )
    if manifest.get("reasoning_aggregate") != aggregate:
        raise ReasoningEvaluationFingerprintError("completed reasoning aggregate changed")
    if manifest.get("judge_token_usage") != _sum_usage(usage_by_qid):
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning judge usage changed"
        )
    if manifest.get("scorecard") != scorecard:
        raise ReasoningEvaluationFingerprintError("completed reasoning scorecard changed")
    if manifest.get("unobservable_usage_risk") is not any(
        bool(item.get("unobservable_usage_risk")) for item in failures
    ):
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning unobservable-usage flag changed"
        )
    if read_json(output_dir / "reasoning_aggregate.json") != aggregate:
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning aggregate artifact changed"
        )
    expected_judge_usage = {
        "by_qid": dict(usage_by_qid),
        "calls_by_qid": {
            qid: list(calls_by_qid[qid]) for qid in sorted(calls_by_qid)
        },
        "total": _sum_usage(usage_by_qid),
        "included_in_submission_token_score": False,
    }
    if read_json(output_dir / "reasoning_judge_usage.json") != expected_judge_usage:
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning judge usage artifact changed"
        )
    expected_corpus_audit = audit_reasoning_corpus(reasoning_by_qid)
    if (
        manifest.get("corpus_audit") != expected_corpus_audit
        or read_json(output_dir / "reasoning_corpus_audit.json")
        != expected_corpus_audit
    ):
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning corpus audit changed"
        )
    scorecard_path = output_dir / "scorecard.json"
    if scorecard is None:
        if scorecard_path.exists():
            raise ReasoningEvaluationFingerprintError(
                "unexpected completed reasoning scorecard artifact"
            )
    elif not scorecard_path.exists() or read_json(scorecard_path) != scorecard:
        raise ReasoningEvaluationFingerprintError(
            "completed reasoning scorecard artifact changed"
        )
    return ReasoningEvaluationRunResult(
        manifest=dict(manifest),
        aggregate=aggregate,
        evaluations=dict(evaluations),
        scorecard=scorecard,
    )


def _evaluation_from_dict(row: Any) -> ReasoningEvaluation:
    if not isinstance(row, Mapping):
        raise ReasoningEvaluationFingerprintError("partial reasoning row must be an object")
    if row.get("prompt_version") != PROMPT_VERSION or row.get("schema_version") != SCHEMA_VERSION:
        raise ReasoningEvaluationFingerprintError("partial reasoning row uses another evaluator")
    evaluation = ReasoningEvaluation(
        qid=str(row.get("qid", "")),
        logical=_score_value(row.get("logical"), "logical"),
        completeness=_score_value(row.get("completeness"), "completeness"),
        clarity=_score_value(row.get("clarity"), "clarity"),
        status=str(row.get("status", "")),
        rubric_scores=_dimension_mapping(row.get("rubric_scores"), "rubric_scores"),
        auditor_scores=_dimension_mapping(row.get("auditor_scores"), "auditor_scores"),
        hard_cap_violations=_validate_violations(row.get("hard_cap_violations", [])),
        hard_caps=_cap_mapping(row.get("hard_caps"), "hard_caps"),
    )
    recorded_score = row.get("reasoning_score")
    if recorded_score is None or not math.isclose(
        float(recorded_score), evaluation.reasoning_score, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ReasoningEvaluationFingerprintError("partial reasoning score is inconsistent")
    if evaluation.status not in {"scored", "below_minimum_length", "judge_error"}:
        raise ReasoningEvaluationFingerprintError("partial reasoning status is invalid")
    return evaluation


def _build_scorecard(
    *,
    evaluations: Mapping[str, ReasoningEvaluation],
    accuracy_score: float | None,
    accuracy_source: str,
    token_total: int,
    model_name: str,
) -> dict[str, Any] | None:
    if accuracy_score is None:
        return None
    score = score_submission(
        accuracy_score=accuracy_score,
        reasoning_scores=[item.reasoning_score for item in evaluations.values()],
        token_total=token_total,
    ).to_dict()
    return {
        **score,
        "accuracy_source": accuracy_source or "unspecified",
        "reasoning_judge_model": model_name,
        "reasoning_prompt_version": PROMPT_VERSION,
        "judge_tokens_included_in_submission": False,
    }


def _default_evaluator_factory(model_config: ModelConfig) -> EvaluatorFactory:
    def build() -> FixedReasoningEvaluator:
        return FixedReasoningEvaluator(OpenAICompatibleClient(model_config))

    return build


def _evaluator_identity(
    model_config: ModelConfig,
    *,
    evaluator_factory: EvaluatorFactory | None = None,
    evaluator_factory_identity: str | None = None,
) -> dict[str, Any]:
    identity = {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": reasoning_prompt_fingerprint(),
        "model_name": model_config.model_name,
        "temperature": model_config.temperature,
        "api_base_sha256": hashlib.sha256(model_config.api_base.encode("utf-8")).hexdigest(),
        "structured_output_mode": model_config.structured_output_mode,
        "timeout_seconds": model_config.timeout_seconds,
        "connect_timeout_seconds": model_config.connect_timeout_seconds,
        "read_timeout_seconds": model_config.read_timeout_seconds,
        "max_retries": model_config.max_retries,
        "retry_backoff_seconds": model_config.retry_backoff_seconds,
        "transport_retry_policy_version": TRANSPORT_RETRY_POLICY_VERSION,
    }
    if evaluator_factory is None:
        identity["evaluator_factory"] = "default_fixed_reasoning_evaluator"
    else:
        identity["evaluator_factory"] = {
            "identity": evaluator_factory_identity,
            "module": str(getattr(evaluator_factory, "__module__", "")),
            "qualname": str(getattr(evaluator_factory, "__qualname__", "")),
        }
    return identity


@contextmanager
def _exclusive_evaluation_lock(output_dir: Path) -> Iterator[None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.evaluation.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReasoningEvaluationFingerprintError(
                f"reasoning evaluation is already active: {output_dir}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _build_fingerprint(
    *,
    evaluator_identity: Mapping[str, Any],
    submission_path: Path,
    sealed_reasoning: Sequence[Mapping[str, str]],
    has_summary: bool,
    accuracy_score: float | None,
    accuracy_source: str,
) -> dict[str, Any]:
    components = {
        "evaluator_identity": dict(evaluator_identity),
        "judge_contract": {
            "schema_sha256": reasoning_schema_fingerprint(),
            "hard_caps_sha256": _payload_sha256(HARD_CAPS),
        },
        "submission": {
            "sha256": _file_sha256(submission_path),
            "has_summary": has_summary,
        },
        "sealed_reasoning": {
            "count": len(sealed_reasoning),
            "sha256": _payload_sha256(sealed_reasoning),
        },
        "score_context": {
            "accuracy_score": accuracy_score,
            "accuracy_source": accuracy_source,
        },
    }
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "sha256": _payload_sha256(components),
        "components": components,
    }


def _zero_evaluation(qid: str, status: str) -> ReasoningEvaluation:
    return ReasoningEvaluation(qid=qid, logical=0.0, completeness=0.0, clarity=0.0, status=status)


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _validate_usage(usage: Mapping[str, Any], qid: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{qid}: reasoning judge {key} must be a non-negative integer")
        values[key] = value
    if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        raise ValueError(f"{qid}: reasoning judge total_tokens is inconsistent")
    return values


def _sum_usage(usage_by_qid: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    prompt = sum(int(item.get("prompt_tokens", 0)) for item in usage_by_qid.values())
    completion = sum(int(item.get("completion_tokens", 0)) for item in usage_by_qid.values())
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _sum_call_usage(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _sum_usage(
        {
            str(index): dict(call.get("token_usage") or {})
            for index, call in enumerate(calls)
        }
    )


def _score_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in 0..100")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{name} must be numeric in 0..100")
    return number


def _validate_violations(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("violations must be an array")
    normalized = tuple(str(item) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("violations must not contain duplicates")
    unknown = sorted(set(normalized) - set(HARD_CAP_VIOLATIONS))
    if unknown:
        raise ValueError(f"violations contain unsupported values: {unknown}")
    return tuple(item for item in HARD_CAP_VIOLATIONS if item in normalized)


def _dimension_mapping(value: Any, name: str) -> dict[str, float]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping) or set(value) != set(REASONING_DIMENSIONS):
        raise ReasoningEvaluationFingerprintError(
            f"{name} must contain exactly {REASONING_DIMENSIONS}"
        )
    return {key: _score_value(value[key], f"{name}.{key}") for key in REASONING_DIMENSIONS}


def _cap_mapping(value: Any, name: str) -> dict[str, float]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ReasoningEvaluationFingerprintError(f"{name} must be an object")
    unknown = sorted(set(value) - set(REASONING_DIMENSIONS))
    if unknown:
        raise ReasoningEvaluationFingerprintError(f"{name} contains unknown keys: {unknown}")
    return {str(key): _score_value(item, f"{name}.{key}") for key, item in value.items()}


def _non_whitespace_length(reasoning: str) -> int:
    return len(re.sub(r"\s+", "", reasoning))


def _normalize_reasoning_for_template_audit(reasoning: str) -> str:
    text = re.sub(r"\s+", "", reasoning).lower()
    text = re.sub(r"(?i)\b(?:e|s)\d+\b|\bchunk[_:-]?\d+\b", "<id>", text)
    text = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?%?", "<num>", text)
    text = re.sub(r"(?<![a-z])[a-d](?![a-z])", "<opt>", text)
    return text


def _group_equal_values(values: Mapping[str, str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for key, value in values.items():
        groups.setdefault(value, []).append(key)
    return sorted(
        (sorted(group) for group in groups.values() if len(group) > 1),
        key=lambda group: (group[0], len(group)),
    )


def _connected_similarity_clusters(graph: Mapping[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for root in sorted(graph):
        if root in visited or not graph[root]:
            continue
        pending = [root]
        cluster: list[str] = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            pending.extend(sorted(graph[current] - visited, reverse=True))
        if len(cluster) > 1:
            clusters.append(sorted(cluster))
    return sorted(clusters, key=lambda group: (group[0], len(group)))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_summary_row(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return any(row.get("qid") == "summary" for row in csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ReasoningEvaluationFingerprintError(f"{path}: JSONL row must be an object")
        rows.append(payload)
    return rows


def _sanitize_error(message: str, model_config: ModelConfig) -> str:
    safe = message
    for secret in (model_config.api_key, model_config.api_base):
        if secret:
            safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", safe)
    safe = re.sub(r"https?://[^\s'\"<>]+", "<redacted-url>", safe)
    return safe[:2000]
