from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Protocol, Sequence

from afa_agent.b_board.io import BQuestion, validate_b_submission
from afa_agent.b_board.scoring import score_submission
from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import ModelConfig
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


PROMPT_VERSION = "b_reasoning_judge_v1_new_md"
SCHEMA_VERSION = 1
FINGERPRINT_SCHEMA_VERSION = 1
MIN_REASONING_NON_WHITESPACE = 20
REASONING_DIMENSIONS = ("logical", "completeness", "clarity")
REASONING_JUDGE_MODEL = "gpt-5.6"

REASONING_JUDGE_SYSTEM_PROMPT = f"""你是推理过程文本质量评估器。你只能评估用户提供的 reasoning 文本本身，不得要求或假设存在题目、原文、答案、证据或外部资料。
分别给出三个 0 到 100 的分数：
logical：推理步骤之间是否存在清晰因果关系，整体链条是否自洽。
completeness：文本是否呈现定位、提取、推导和结论等完整分析过程。
clarity：文本是否条理清晰、结构化、表达准确。
评分尺度：80 分以上表示逻辑清晰、论证完整、表达专业；60-79 表示有明确分析步骤和推导；30-59 表示有部分分析但不完整；0-29 表示空泛、模板化或无实质内容。
不要判断事实是否真实，不要猜测最终答案，不要因缺少题目而扣分；只评价文本展示出的写作质量。只输出一个 JSON 对象：{{"logical":0-100,"completeness":0-100,"clarity":0-100}}。
prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}。"""


class ReasoningEvaluationFingerprintError(RuntimeError):
    """Raised when a sealed reasoning evaluation cannot safely resume."""


class ReasoningJudgeError(RuntimeError):
    def __init__(self, message: str, *, token_usage: Mapping[str, int] | None = None) -> None:
        super().__init__(message)
        self.token_usage = dict(token_usage or _zero_usage())


class ReasoningEvaluator(Protocol):
    def evaluate(self, reasoning: str) -> tuple[Mapping[str, float], Mapping[str, int]]: ...


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
        }


@dataclass(frozen=True, slots=True)
class ReasoningEvaluationRunResult:
    manifest: dict[str, Any]
    aggregate: dict[str, Any]
    evaluations: dict[str, ReasoningEvaluation]
    scorecard: dict[str, Any] | None


class FixedReasoningEvaluator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def evaluate(self, reasoning: str) -> tuple[Mapping[str, float], Mapping[str, int]]:
        response = self.client.chat_json(
            [
                {"role": "system", "content": REASONING_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": reasoning},
            ]
        )
        usage = response.token_usage.to_dict()
        try:
            payload = extract_json_object(response.content)
            dimensions = {
                key: _score_value(payload.get(key), key) for key in REASONING_DIMENSIONS
            }
        except Exception as exc:
            raise ReasoningJudgeError(
                f"invalid reasoning judge response: {exc}", token_usage=usage
            ) from exc
        return dimensions, usage


def reasoning_prompt_fingerprint() -> str:
    return hashlib.sha256(REASONING_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def run_reasoning_evaluation(
    *,
    submission_path: Path,
    questions: Sequence[BQuestion],
    model_config: ModelConfig,
    output_dir: Path,
    workers: int = 4,
    evaluator_factory: EvaluatorFactory | None = None,
    accuracy_score: float | None = None,
    accuracy_source: str = "",
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

    answers = validate_b_submission(source, questions, audit_ready=False)
    sealed_reasoning = [
        {"qid": answer.qid, "reasoning": answer.reasoning} for answer in answers
    ]
    expected_qids = {item["qid"] for item in sealed_reasoning}
    if len(expected_qids) != len(sealed_reasoning):
        raise ValueError("submission reasoning contains duplicate qids")
    token_total = sum(int(answer.total_tokens or 0) for answer in answers)
    evaluator_identity = _evaluator_identity(model_config)
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
    evaluations, usage_by_qid = _load_partial(destination, expected_qids)
    completed_qids = set(evaluations) & set(usage_by_qid)
    evaluations = {qid: evaluations[qid] for qid in completed_qids}
    usage_by_qid = {qid: usage_by_qid[qid] for qid in completed_qids}
    failures = _read_jsonl(destination / "reasoning_judge_failures.jsonl")

    if manifest.get("status") == "complete" and completed_qids == expected_qids:
        return _validate_completed_result(
            manifest=manifest,
            evaluations=evaluations,
            expected_qids=expected_qids,
            token_total=token_total,
            accuracy_score=accuracy_score,
            accuracy_source=accuracy_source,
        )

    factory = evaluator_factory or _default_evaluator_factory(model_config)
    remaining_qids = sorted(expected_qids - completed_qids)

    def evaluate_one(qid: str) -> tuple[ReasoningEvaluation, dict[str, int], dict[str, Any] | None]:
        reasoning = reasoning_by_qid[qid]
        if _non_whitespace_length(reasoning) < MIN_REASONING_NON_WHITESPACE:
            return _zero_evaluation(qid, "below_minimum_length"), _zero_usage(), None
        try:
            dimensions, usage = factory().evaluate(reasoning)
            evaluation = ReasoningEvaluation(
                qid=qid,
                logical=_score_value(dimensions.get("logical"), "logical"),
                completeness=_score_value(dimensions.get("completeness"), "completeness"),
                clarity=_score_value(dimensions.get("clarity"), "clarity"),
            )
            return evaluation, _validate_usage(usage, qid), None
        except Exception as exc:
            usage = _validate_usage(
                getattr(exc, "token_usage", _zero_usage()), qid
            )
            failure = {
                "qid": qid,
                "error_type": exc.__class__.__name__,
                "error": _sanitize_error(str(exc), model_config),
            }
            return _zero_evaluation(qid, "judge_error"), usage, failure

    if remaining_qids:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_one, qid): qid for qid in remaining_qids}
            for future in as_completed(futures):
                qid = futures[future]
                evaluation, usage, failure = future.result()
                evaluations[qid] = evaluation
                usage_by_qid[qid] = usage
                if failure is not None:
                    failures = [item for item in failures if item.get("qid") != qid]
                    failures.append(failure)
                _persist_partial(destination, evaluations, usage_by_qid, failures)

    aggregate = aggregate_reasoning(evaluations)
    scorecard = _build_scorecard(
        evaluations=evaluations,
        accuracy_score=accuracy_score,
        accuracy_source=accuracy_source,
        token_total=token_total,
        model_name=model_config.model_name,
    )
    judge_usage = _sum_usage(usage_by_qid)
    write_json(destination / "reasoning_aggregate.json", aggregate)
    write_json(destination / "reasoning_judge_usage.json", {
        "by_qid": usage_by_qid,
        "total": judge_usage,
        "included_in_submission_token_score": False,
    })
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
            "submission_token_total": token_total,
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
        _validate_existing_manifest(manifest, evaluator_identity, fingerprint)
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
        "submission_path": str(submission_path),
        "evaluator_identity": dict(evaluator_identity),
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
) -> None:
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


def _persist_partial(
    output_dir: Path,
    evaluations: Mapping[str, ReasoningEvaluation],
    usage_by_qid: Mapping[str, Mapping[str, int]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    rows = [evaluations[qid].to_dict() for qid in sorted(evaluations)]
    write_json(output_dir / "reasoning_scores.json", rows)
    write_jsonl(output_dir / "reasoning_scores.jsonl", rows)
    write_json(output_dir / "reasoning_judge_usage_partial.json", usage_by_qid)
    write_jsonl(
        output_dir / "reasoning_judge_failures.jsonl",
        [dict(item) for item in failures],
    )


def _validate_completed_result(
    *,
    manifest: Mapping[str, Any],
    evaluations: Mapping[str, ReasoningEvaluation],
    expected_qids: set[str],
    token_total: int,
    accuracy_score: float | None,
    accuracy_source: str,
) -> ReasoningEvaluationRunResult:
    if set(evaluations) != expected_qids:
        raise ReasoningEvaluationFingerprintError("completed reasoning coverage is incomplete")
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
    if manifest.get("scorecard") != scorecard:
        raise ReasoningEvaluationFingerprintError("completed reasoning scorecard changed")
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


def _evaluator_identity(model_config: ModelConfig) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": reasoning_prompt_fingerprint(),
        "model_name": model_config.model_name,
        "temperature": model_config.temperature,
        "api_base_sha256": hashlib.sha256(model_config.api_base.encode("utf-8")).hexdigest(),
    }


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


def _score_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in 0..100")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{name} must be numeric in 0..100")
    return number


def _non_whitespace_length(reasoning: str) -> int:
    return len(re.sub(r"\s+", "", reasoning))


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
