from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from afa_agent.b_board.evaluator import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ConfidenceEvaluation,
    FixedConfidenceEvaluator,
    build_calibration_subjects,
    confidence_tier,
    prompt_fingerprint,
    validate_calibration_sentinels,
)
from afa_agent.b_board.io import BQuestion
from afa_agent.client import OpenAICompatibleClient
from afa_agent.config import ModelConfig
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


EVALUATION_FINGERPRINT_SCHEMA_VERSION = 1
_FINAL_STATUSES = {"complete", "invalid"}


class EvaluationRunFingerprintError(RuntimeError):
    """Raised when an evaluation cannot safely resume with its frozen inputs."""


class SubjectEvaluator(Protocol):
    def evaluate(
        self, subject: Mapping[str, Any]
    ) -> tuple[ConfidenceEvaluation, dict[str, int]]: ...


EvaluatorFactory = Callable[[], SubjectEvaluator]


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    manifest: dict[str, Any]
    aggregate: dict[str, Any]
    evaluations: dict[str, ConfidenceEvaluation]


def run_fixed_evaluation(
    *,
    run_dir: Path,
    questions: Sequence[BQuestion],
    model_config: ModelConfig,
    workers: int = 4,
    evaluator_factory: EvaluatorFactory | None = None,
) -> EvaluationRunResult:
    """Evaluate or safely resume evaluation of a sealed B-board answer run.

    The interface deliberately accepts the model configuration and evaluator factory:
    production supplies the OpenAI-compatible adapter by default, while tests can inject
    an in-memory adapter. The API key and API base are never persisted; only the base URL
    hash participates in evaluator identity and the resume fingerprint.
    """

    resolved_run_dir = Path(run_dir).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    if float(model_config.temperature) != 0.0:
        raise ValueError("the fixed confidence evaluator requires temperature=0")

    question_by_qid = _index_questions(questions)
    source_answers = _load_complete_answer_artifacts(
        resolved_run_dir / "answers.json", question_by_qid
    )
    sentinel_subjects = {item["qid"]: item for item in build_calibration_subjects()}
    if len(sentinel_subjects) != 6:
        raise RuntimeError("the fixed evaluator must contain exactly six calibration sentinels")

    evaluator_identity = _evaluator_identity(model_config)
    fingerprint = _build_fingerprint(
        evaluator_identity=evaluator_identity,
        questions=questions,
        sealed_answers=source_answers,
        sentinel_subjects=sentinel_subjects,
    )
    output_dir = resolved_run_dir / "evaluation"
    manifest, sealed_answers, resumed = _prepare_evaluation(
        output_dir=output_dir,
        run_dir=resolved_run_dir,
        source_answers=source_answers,
        evaluator_identity=evaluator_identity,
        fingerprint=fingerprint,
    )

    artifacts = {str(item["qid"]): item for item in sealed_answers}
    subjects = {
        qid: build_evaluation_subject(question_by_qid[qid], artifacts[qid])
        for qid in question_by_qid
    }
    answer_qids = set(subjects)
    sentinel_qids = set(sentinel_subjects)
    expected_qids = answer_qids | sentinel_qids
    evaluations, usage_by_qid = _load_partial_results(
        output_dir, answer_qids=answer_qids, sentinel_qids=sentinel_qids
    )
    completed_qids = set(evaluations) & set(usage_by_qid)
    evaluations = {qid: evaluations[qid] for qid in completed_qids}
    usage_by_qid = {qid: usage_by_qid[qid] for qid in completed_qids}

    if (
        manifest.get("status") in _FINAL_STATUSES
        and not (expected_qids - completed_qids)
        and int(manifest.get("failure_count", 0)) == 0
    ):
        return _validate_completed_result(
            manifest=manifest,
            evaluations=evaluations,
            answer_qids=answer_qids,
            sentinel_qids=sentinel_qids,
            question_by_qid=question_by_qid,
        )

    all_subjects = {**subjects, **sentinel_subjects}
    remaining_qids = sorted(expected_qids - completed_qids)
    failures: list[dict[str, str]] = []
    factory = evaluator_factory or _default_evaluator_factory(model_config)

    def evaluate_one(qid: str) -> tuple[ConfidenceEvaluation, dict[str, int]]:
        evaluation, usage = factory().evaluate(all_subjects[qid])
        _validate_evaluator_result(qid, evaluation)
        return evaluation, _validate_usage(usage, qid)

    if remaining_qids:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_one, qid): qid for qid in remaining_qids}
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    evaluation, usage = future.result()
                    evaluations[qid] = evaluation
                    usage_by_qid[qid] = usage
                    _persist_partial(output_dir, evaluations, usage_by_qid, answer_qids)
                except Exception as exc:
                    failures.append(
                        {
                            "qid": qid,
                            "error_type": exc.__class__.__name__,
                            "error": _sanitize_error(str(exc), model_config),
                        }
                    )
                    write_jsonl(output_dir / "failures.jsonl", failures)

    sentinel_evaluations = {
        qid: evaluations[qid] for qid in sentinel_qids if qid in evaluations
    }
    answer_evaluations = {
        qid: evaluations[qid] for qid in answer_qids if qid in evaluations
    }
    sentinel_validation = validate_calibration_sentinels(sentinel_evaluations)
    aggregate = aggregate_confidence(answer_evaluations, question_by_qid)
    totals = sum_evaluation_usage(usage_by_qid)
    complete = len(answer_evaluations) == len(answer_qids) and len(
        sentinel_evaluations
    ) == len(sentinel_qids)
    valid = complete and sentinel_validation["passed"] and not failures

    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "sentinel_validation.json", sentinel_validation)
    write_json(output_dir / "token_usage.json", {"by_qid": usage_by_qid, "total": totals})
    write_jsonl(output_dir / "failures.jsonl", failures)
    manifest.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "status": "complete" if valid else "invalid",
            "resumed": resumed,
            "resumed_evaluation_count": len(completed_qids),
            "expected_answer_count": len(answer_qids),
            "evaluated_answer_count": len(answer_evaluations),
            "expected_sentinel_count": len(sentinel_qids),
            "evaluated_sentinel_count": len(sentinel_evaluations),
            "sentinel_validation": sentinel_validation,
            "failure_count": len(failures),
            "token_usage": totals,
            "aggregate_metrics": aggregate,
        }
    )
    write_json(output_dir / "evaluator_manifest.json", manifest)
    return EvaluationRunResult(manifest, aggregate, answer_evaluations)


def build_evaluation_subject(question: BQuestion, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qid": question.qid,
        "domain": question.domain,
        "type": question.type,
        "answer_format": question.answer_format,
        "question": question.question,
        "options": question.options,
        "answer_slot_count": question.answer_slots,
        "answer_slot_templates": list(question.answer_slot_templates),
        "answer_parts": artifact.get("answer_parts", []),
        "used_evidence_ids": artifact.get("used_evidence_ids", []),
        "evidence_items": artifact.get("evidence_items", []),
        "decision_trace": artifact.get("decision_trace", {}),
        "calculation_trace": artifact.get("calculation_trace", {}),
        "token_usage": artifact.get("token_usage", {}),
    }


def aggregate_confidence(
    evaluations: Mapping[str, ConfidenceEvaluation],
    question_by_qid: Mapping[str, BQuestion],
) -> dict[str, Any]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    tiers: Counter[str] = Counter()
    for qid, evaluation in evaluations.items():
        by_domain[question_by_qid[qid].domain].append(evaluation.confidence_score)
        tiers[evaluation.tier] += 1
    all_values = [item.confidence_score for item in evaluations.values()]
    return {
        "question_count": len(evaluations),
        "tiers": dict(sorted(tiers.items())),
        "minimum": min(all_values, default=None),
        "p10": _percentile(all_values, 0.10),
        "median": _percentile(all_values, 0.50),
        "by_domain": {
            domain: {
                "question_count": len(values),
                "minimum": min(values),
                "p10": _percentile(values, 0.10),
                "median": _percentile(values, 0.50),
            }
            for domain, values in sorted(by_domain.items())
        },
    }


def sum_evaluation_usage(usage_by_qid: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    prompt = sum(int(item.get("prompt_tokens", 0)) for item in usage_by_qid.values())
    completion = sum(int(item.get("completion_tokens", 0)) for item in usage_by_qid.values())
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _prepare_evaluation(
    *,
    output_dir: Path,
    run_dir: Path,
    source_answers: list[dict[str, Any]],
    evaluator_identity: dict[str, Any],
    fingerprint: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    manifest_path = output_dir / "evaluator_manifest.json"
    sealed_path = output_dir / "sealed_answers.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _validate_existing_manifest(manifest, evaluator_identity, fingerprint)
        if not sealed_path.exists():
            raise EvaluationRunFingerprintError(
                "Existing evaluation has no sealed_answers.json; refusing unsafe resume"
            )
        sealed_answers = read_json(sealed_path)
        if _payload_sha256(sealed_answers) != fingerprint["components"]["sealed_answers"]["sha256"]:
            raise EvaluationRunFingerprintError(
                "Sealed answers changed after evaluation started; refusing unsafe resume"
            )
        if _payload_sha256(source_answers) != _payload_sha256(sealed_answers):
            raise EvaluationRunFingerprintError(
                "answers.json differs from the frozen sealed answers; use a new run directory"
            )
        return manifest, sealed_answers, True

    if output_dir.exists() and any(output_dir.iterdir()):
        raise EvaluationRunFingerprintError(
            "Evaluation artifacts exist without evaluator_manifest.json; refusing unsafe resume"
        )
    ensure_dir(output_dir)
    write_json(sealed_path, source_answers)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "evaluator_identity": evaluator_identity,
        "fingerprint": fingerprint,
        "sealed_answers_path": str(sealed_path),
        "status": "running",
    }
    write_json(manifest_path, manifest)
    return manifest, source_answers, False


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> None:
    if manifest.get("evaluator_identity") != evaluator_identity:
        raise EvaluationRunFingerprintError(
            "Frozen evaluator identity changed; use a new evaluation version and re-evaluate B0"
        )
    existing = manifest.get("fingerprint")
    if not isinstance(existing, Mapping):
        raise EvaluationRunFingerprintError(
            "Existing evaluator manifest has no fingerprint; refusing unsafe resume"
        )
    if existing.get("schema_version") != EVALUATION_FINGERPRINT_SCHEMA_VERSION:
        raise EvaluationRunFingerprintError("Unsupported existing evaluation fingerprint schema")
    if existing.get("sha256") != _payload_sha256(existing.get("components")):
        raise EvaluationRunFingerprintError("Existing evaluation fingerprint has been modified")
    if existing.get("sha256") != fingerprint.get("sha256"):
        changed = sorted(
            key
            for key in set(existing.get("components", {})) | set(fingerprint.get("components", {}))
            if existing.get("components", {}).get(key)
            != fingerprint.get("components", {}).get(key)
        )
        raise EvaluationRunFingerprintError(
            "Evaluation fingerprint mismatch "
            f"({', '.join(changed) or 'unknown components'}); refusing unsafe resume"
        )


def _load_complete_answer_artifacts(
    path: Path, question_by_qid: Mapping[str, BQuestion]
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"sealed answer source does not exist: {path}")
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: answers.json must contain a JSON array")
    by_qid: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: answer row {index} must be an object")
        qid = str(item.get("qid", "")).strip()
        if not qid or qid in by_qid:
            raise ValueError(f"{path}: blank or duplicate answer qid {qid!r}")
        by_qid[qid] = item
    expected = set(question_by_qid)
    actual = set(by_qid)
    if expected != actual:
        raise ValueError(
            f"answer/question qid mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return [by_qid[qid] for qid in question_by_qid]


def _index_questions(questions: Sequence[BQuestion]) -> dict[str, BQuestion]:
    indexed: dict[str, BQuestion] = {}
    for question in questions:
        if question.qid in indexed:
            raise ValueError(f"duplicate question qid {question.qid!r}")
        indexed[question.qid] = question
    if not indexed:
        raise ValueError("questions must not be empty")
    return indexed


def _load_partial_results(
    output_dir: Path,
    *,
    answer_qids: set[str],
    sentinel_qids: set[str],
) -> tuple[dict[str, ConfidenceEvaluation], dict[str, dict[str, int]]]:
    evaluations: dict[str, ConfidenceEvaluation] = {}
    for path, allowed_qids in (
        (output_dir / "confidence_audit.json", answer_qids),
        (output_dir / "sentinel_evaluations.json", sentinel_qids),
    ):
        if not path.exists():
            continue
        rows = read_json(path)
        if not isinstance(rows, list):
            raise EvaluationRunFingerprintError(f"{path}: partial evaluations must be an array")
        for row in rows:
            evaluation = _evaluation_from_dict(row)
            if evaluation.qid not in allowed_qids or evaluation.qid in evaluations:
                raise EvaluationRunFingerprintError(
                    f"{path}: unexpected or duplicate partial qid {evaluation.qid!r}"
                )
            evaluations[evaluation.qid] = evaluation

    usage_by_qid: dict[str, dict[str, int]] = {}
    usage_path = output_dir / "evaluation_usage_partial.json"
    if usage_path.exists():
        payload = read_json(usage_path)
        if not isinstance(payload, dict):
            raise EvaluationRunFingerprintError(f"{usage_path}: expected an object")
        for qid, usage in payload.items():
            if qid not in answer_qids | sentinel_qids or not isinstance(usage, Mapping):
                raise EvaluationRunFingerprintError(
                    f"{usage_path}: unexpected usage qid {qid!r}"
                )
            usage_by_qid[qid] = _validate_usage(usage, qid)
    return evaluations, usage_by_qid


def _persist_partial(
    output_dir: Path,
    evaluations: Mapping[str, ConfidenceEvaluation],
    usage_by_qid: Mapping[str, Mapping[str, int]],
    answer_qids: set[str],
) -> None:
    answer_rows = [
        evaluations[qid].to_dict() for qid in sorted(answer_qids & set(evaluations))
    ]
    sentinel_rows = [
        evaluations[qid].to_dict() for qid in sorted(set(evaluations) - answer_qids)
    ]
    write_json(output_dir / "confidence_audit.json", answer_rows)
    write_jsonl(output_dir / "confidence_audit.jsonl", answer_rows)
    write_json(output_dir / "sentinel_evaluations.json", sentinel_rows)
    write_json(output_dir / "evaluation_usage_partial.json", usage_by_qid)


def _validate_completed_result(
    *,
    manifest: dict[str, Any],
    evaluations: Mapping[str, ConfidenceEvaluation],
    answer_qids: set[str],
    sentinel_qids: set[str],
    question_by_qid: Mapping[str, BQuestion],
) -> EvaluationRunResult:
    sentinel_validation = validate_calibration_sentinels(
        {qid: evaluations[qid] for qid in sentinel_qids}
    )
    aggregate = aggregate_confidence(
        {qid: evaluations[qid] for qid in answer_qids}, question_by_qid
    )
    if manifest.get("sentinel_validation") != sentinel_validation:
        raise EvaluationRunFingerprintError("completed sentinel validation does not match artifacts")
    if manifest.get("aggregate_metrics") != aggregate:
        raise EvaluationRunFingerprintError("completed aggregate metrics do not match artifacts")
    expected_status = "complete" if sentinel_validation["passed"] else "invalid"
    if manifest.get("status") != expected_status:
        raise EvaluationRunFingerprintError("completed evaluator status does not match artifacts")
    return EvaluationRunResult(
        manifest=dict(manifest),
        aggregate=aggregate,
        evaluations={qid: evaluations[qid] for qid in answer_qids},
    )


def _evaluation_from_dict(row: Any) -> ConfidenceEvaluation:
    if not isinstance(row, Mapping):
        raise EvaluationRunFingerprintError("partial evaluation row must be an object")
    if row.get("prompt_version") != PROMPT_VERSION or row.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationRunFingerprintError("partial evaluation uses a different fixed evaluator")
    try:
        score = int(row["confidence_score"])
        tier = str(row["tier"])
        verdict = str(row["verdict"])
        dimensions = dict(row["dimensions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationRunFingerprintError("invalid partial evaluation row") from exc
    if not 0 <= score <= 100 or confidence_tier(score) != tier:
        raise EvaluationRunFingerprintError("partial evaluation score/tier is inconsistent")
    return ConfidenceEvaluation(
        qid=str(row.get("qid", "")),
        dimensions=dimensions,
        confidence_score=score,
        tier=tier,
        verdict=verdict,
        blocking_reasons=tuple(str(item) for item in row.get("blocking_reasons", [])),
        low_confidence_reasons=tuple(
            str(item) for item in row.get("low_confidence_reasons", [])
        ),
        suggested_improvements=tuple(
            str(item) for item in row.get("suggested_improvements", [])
        ),
        hard_failures=tuple(str(item) for item in row.get("hard_failures", [])),
    )


def _validate_evaluator_result(qid: str, evaluation: ConfidenceEvaluation) -> None:
    if evaluation.qid != qid:
        raise ValueError(f"evaluator returned qid {evaluation.qid!r} for subject {qid!r}")
    if evaluation.prompt_version != PROMPT_VERSION or evaluation.schema_version != SCHEMA_VERSION:
        raise ValueError("evaluator returned a different prompt or schema version")
    if not 0 <= evaluation.confidence_score <= 100:
        raise ValueError("evaluator confidence_score is outside 0..100")
    if evaluation.tier != confidence_tier(evaluation.confidence_score):
        raise ValueError("evaluator confidence_score and tier are inconsistent")


def _validate_usage(usage: Mapping[str, Any], qid: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{qid}: evaluator {key} must be a non-negative integer")
        values[key] = value
    if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        raise ValueError(f"{qid}: evaluator total_tokens is inconsistent")
    return values


def _default_evaluator_factory(model_config: ModelConfig) -> EvaluatorFactory:
    def build() -> FixedConfidenceEvaluator:
        return FixedConfidenceEvaluator(OpenAICompatibleClient(model_config))

    return build


def _evaluator_identity(model_config: ModelConfig) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": prompt_fingerprint(),
        "model_name": model_config.model_name,
        "temperature": model_config.temperature,
        "api_base_sha256": hashlib.sha256(model_config.api_base.encode("utf-8")).hexdigest(),
    }


def _build_fingerprint(
    *,
    evaluator_identity: Mapping[str, Any],
    questions: Sequence[BQuestion],
    sealed_answers: Sequence[Mapping[str, Any]],
    sentinel_subjects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    public_questions = [
        {
            "qid": item.qid,
            "domain": item.domain,
            "split": item.split,
            "question": item.question,
            "type": item.type,
            "answer_format": item.answer_format,
            "options": item.options,
            "answer_slots": item.answer_slots,
            "answer_slot_templates": list(item.answer_slot_templates),
        }
        for item in questions
    ]
    components = {
        "evaluator_identity": dict(evaluator_identity),
        "questions": {
            "count": len(public_questions),
            "sha256": _payload_sha256(public_questions),
        },
        "sealed_answers": {
            "count": len(sealed_answers),
            "sha256": _payload_sha256(sealed_answers),
        },
        "calibration_sentinels": {
            "count": len(sentinel_subjects),
            "sha256": _payload_sha256(sentinel_subjects),
        },
    }
    return {
        "schema_version": EVALUATION_FINGERPRINT_SCHEMA_VERSION,
        "sha256": _payload_sha256(components),
        "components": components,
    }


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _sanitize_error(message: str, model_config: ModelConfig) -> str:
    safe = message
    for secret in (model_config.api_key, model_config.api_base):
        if secret:
            safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", safe)
    safe = re.sub(r"https?://[^\s'\"<>]+", "<redacted-url>", safe)
    return safe[:2000]
