from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from afa_agent.b_board.io import BAnswer, BQuestion, validate_b_submission, write_b_submission
from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import ModelConfig
from afa_agent.io_utils import ensure_dir, read_json, write_json


REWRITE_PROMPT_VERSION = "b_reasoning_rewrite_structured_v1"
REWRITE_SYSTEM_PROMPT = f"""你是金融长文问答的推理摘要改写器。只能使用用户提供的题目、冻结答案、已有摘要和证据；不得补充外部事实，不得改变答案。
输出一个 JSON 对象，字段仅为 answer_parts 和 reasoning。answer_parts 必须逐字复制冻结答案。
reasoning 使用中文并形成自包含的四步短摘要，建议 120-220 字：
1. 定位：指出核对的主体、产品、条款、指标或期间；
2. 关键事实：列出支持判断或计算的具体事实，选择题应覆盖选中项并说明关键排除项；
3. 推导：明确事实如何推出判断，计算题写出必要公式、代入关系和格式规则；
4. 结论：明确说明为何得到冻结答案。
必须有清晰因果连接，避免逐项复制同一句模板，避免只复述答案，避免声称证据中没有的页码、条款号或事实。只输出 JSON。prompt_version={REWRITE_PROMPT_VERSION}。"""


class ReasoningRewriteFingerprintError(RuntimeError):
    pass


class ReasoningRewriteError(RuntimeError):
    def __init__(self, message: str, *, token_usage: Mapping[str, int] | None = None) -> None:
        super().__init__(message)
        self.token_usage = dict(token_usage or _zero_usage())


class Rewriter(Protocol):
    def rewrite(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, int]]: ...


RewriterFactory = Callable[[], Rewriter]


@dataclass(frozen=True, slots=True)
class ReasoningRewriteResult:
    manifest: dict[str, Any]
    records: dict[str, dict[str, Any]]
    submission_path: Path


class FixedReasoningRewriter:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def rewrite(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, int]]:
        response = self.client.chat_json(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
        usage = response.token_usage.to_dict()
        try:
            parsed = extract_json_object(response.content)
            answer_parts = parsed.get("answer_parts")
            expected = [str(item) for item in payload.get("answer_parts", [])]
            if not isinstance(answer_parts, list) or [str(item) for item in answer_parts] != expected:
                raise ValueError("rewrite response changed answer_parts")
            reasoning = str(parsed.get("reasoning") or "").strip()
            if len(re.sub(r"\s+", "", reasoning)) < 20:
                raise ValueError("rewrite response is shorter than 20 non-whitespace characters")
            if "\x00" in reasoning:
                raise ValueError("rewrite response contains a NUL character")
        except Exception as exc:
            raise ReasoningRewriteError(str(exc), token_usage=usage) from exc
        return reasoning, usage


def run_reasoning_rewrite(
    *,
    source_submission_path: Path,
    source_answers_path: Path,
    questions: Sequence[BQuestion],
    target_qids: Sequence[str],
    model_config: ModelConfig,
    output_dir: Path,
    workers: int = 4,
    rewriter_factory: RewriterFactory | None = None,
) -> ReasoningRewriteResult:
    if workers < 1:
        raise ValueError("workers must be positive")
    source_submission = Path(source_submission_path).resolve()
    source_answers = Path(source_answers_path).resolve()
    destination = Path(output_dir).resolve()
    source_rows = validate_b_submission(source_submission, questions, audit_ready=False)
    source_by_qid = {item.qid: item for item in source_rows}
    artifact_by_qid = _load_artifacts(source_answers, questions)
    targets = tuple(dict.fromkeys(str(qid) for qid in target_qids))
    unknown = sorted(set(targets) - set(source_by_qid))
    if unknown:
        raise ValueError(f"unknown reasoning rewrite qids: {unknown}")
    fingerprint = _fingerprint(
        source_submission=source_submission,
        source_answers=source_answers,
        targets=targets,
        model_config=model_config,
    )
    manifest, records = _prepare(destination, fingerprint, targets, model_config)
    factory = rewriter_factory or (
        lambda: FixedReasoningRewriter(OpenAICompatibleClient(model_config))
    )

    def rewrite_one(qid: str) -> dict[str, Any]:
        source = source_by_qid[qid]
        artifact = artifact_by_qid[qid]
        payload = {
            "question": artifact["question"],
            "options": artifact["options"],
            "answer_parts": list(source.answer_parts),
            "existing_reasoning": source.reasoning,
            "evidence": artifact["evidence"],
        }
        try:
            reasoning, usage = factory().rewrite(payload)
            return {
                "qid": qid,
                "status": "rewritten",
                "reasoning": reasoning,
                "token_usage": _validate_usage(usage, qid),
                "answer_parts_preserved": True,
                "source_reasoning_sha256": _text_sha256(source.reasoning),
                "rewritten_reasoning_sha256": _text_sha256(reasoning),
            }
        except Exception as exc:
            return {
                "qid": qid,
                "status": "rewrite_error_fallback_to_source",
                "reasoning": source.reasoning,
                "token_usage": _validate_usage(
                    getattr(exc, "token_usage", _zero_usage()), qid
                ),
                "answer_parts_preserved": True,
                "source_reasoning_sha256": _text_sha256(source.reasoning),
                "rewritten_reasoning_sha256": _text_sha256(source.reasoning),
                "error_type": exc.__class__.__name__,
                "error": _sanitize_error(str(exc), model_config),
            }

    remaining = [qid for qid in targets if qid not in records]
    if remaining:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(rewrite_one, qid): qid for qid in remaining}
            for future in as_completed(futures):
                record = future.result()
                records[str(record["qid"])] = record
                _persist_records(destination, records)

    output_answers: list[BAnswer] = []
    for question in questions:
        source = source_by_qid[question.qid]
        record = records.get(question.qid)
        usage = dict(record.get("token_usage") or _zero_usage()) if record else _zero_usage()
        output_answers.append(
            BAnswer(
                qid=source.qid,
                answer_parts=source.answer_parts,
                prompt_tokens=source.prompt_tokens + int(usage["prompt_tokens"]),
                completion_tokens=source.completion_tokens + int(usage["completion_tokens"]),
                reasoning=str(record["reasoning"]) if record else source.reasoning,
            )
        )
    output_path = destination / "research_submit.csv"
    write_b_submission(output_path, questions, output_answers, audit_ready=False)
    validate_b_submission(output_path, questions, audit_ready=False)
    rewrite_usage = _sum_usage(
        {qid: dict(record["token_usage"]) for qid, record in records.items()}
    )
    submission_usage = {
        "prompt_tokens": sum(item.prompt_tokens for item in output_answers),
        "completion_tokens": sum(item.completion_tokens for item in output_answers),
        "total_tokens": sum(int(item.total_tokens or 0) for item in output_answers),
    }
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_mode": "research",
            "submission_eligible": False,
            "submission_ineligibility_reasons": [
                "source_run_has_no_complete_per_call_usage_ledger",
                "source_generation_model_is_not_verified_as_allowlisted",
                "reasoning_rewrite_model_is_not_submission_allowlisted",
            ],
            "target_count": len(targets),
            "rewritten_count": sum(row["status"] == "rewritten" for row in records.values()),
            "rewrite_failure_count": sum(row["status"] != "rewritten" for row in records.values()),
            "answer_changes": 0,
            "rewrite_token_usage": rewrite_usage,
            "submission_token_usage": submission_usage,
            "research_submission_path": str(output_path),
            "research_submission_sha256": _file_sha256(output_path),
        }
    )
    write_json(destination / "candidate_manifest.json", manifest)
    return ReasoningRewriteResult(dict(manifest), dict(records), output_path)


def _load_artifacts(
    path: Path,
    questions: Sequence[BQuestion],
) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("source answers.json must contain an array")
    raw = {str(row.get("qid")): row for row in rows if isinstance(row, Mapping)}
    if set(raw) != {item.qid for item in questions}:
        raise ValueError("source answers.json qids do not match questions")
    artifacts: dict[str, dict[str, Any]] = {}
    for question in questions:
        row = raw[question.qid]
        evidence = [
            {
                "unit_id": str(item.get("unit_id") or ""),
                "title": " > ".join(str(value) for value in item.get("title_path", [])),
                "text": str(item.get("text") or "")[:1800],
            }
            for item in list(row.get("evidence_items") or [])[:12]
            if isinstance(item, Mapping)
        ]
        artifacts[question.qid] = {
            "question": question.question,
            "options": question.options,
            "evidence": evidence,
        }
    return artifacts


def _prepare(
    destination: Path,
    fingerprint: Mapping[str, Any],
    targets: Sequence[str],
    model_config: ModelConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = destination / "candidate_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise ReasoningRewriteFingerprintError(
                "reasoning rewrite fingerprint changed; use a new output directory"
            )
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ReasoningRewriteFingerprintError(
                "reasoning rewrite artifacts exist without candidate_manifest.json"
            )
        ensure_dir(destination)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "running",
            "prompt_version": REWRITE_PROMPT_VERSION,
            "fingerprint": dict(fingerprint),
            "target_qids": list(targets),
            "model": {
                "model_name": model_config.model_name,
                "temperature": model_config.temperature,
                "api_base_sha256": hashlib.sha256(
                    model_config.api_base.encode("utf-8")
                ).hexdigest(),
            },
        }
        write_json(manifest_path, manifest)
    records_path = destination / "rewrite_records.json"
    records: dict[str, dict[str, Any]] = {}
    if records_path.exists():
        rows = read_json(records_path)
        if not isinstance(rows, list):
            raise ReasoningRewriteFingerprintError("rewrite_records.json must be an array")
        for row in rows:
            if not isinstance(row, Mapping) or str(row.get("qid")) not in targets:
                raise ReasoningRewriteFingerprintError("rewrite records contain an unexpected qid")
            qid = str(row["qid"])
            if qid in records:
                raise ReasoningRewriteFingerprintError("rewrite records contain duplicate qids")
            records[qid] = dict(row)
    return manifest, records


def _persist_records(destination: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    write_json(
        destination / "rewrite_records.json",
        [dict(records[qid]) for qid in sorted(records)],
    )


def _fingerprint(
    *,
    source_submission: Path,
    source_answers: Path,
    targets: Sequence[str],
    model_config: ModelConfig,
) -> dict[str, Any]:
    components = {
        "source_submission_sha256": _file_sha256(source_submission),
        "source_answers_sha256": _file_sha256(source_answers),
        "target_qids": list(targets),
        "prompt_version": REWRITE_PROMPT_VERSION,
        "prompt_sha256": _text_sha256(REWRITE_SYSTEM_PROMPT),
        "model_name": model_config.model_name,
        "temperature": model_config.temperature,
        "api_base_sha256": hashlib.sha256(model_config.api_base.encode("utf-8")).hexdigest(),
    }
    return {"sha256": _payload_sha256(components), "components": components}


def _validate_usage(usage: Mapping[str, Any], qid: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{qid}: rewrite {key} must be a non-negative integer")
        values[key] = value
    if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        raise ValueError(f"{qid}: rewrite total_tokens is inconsistent")
    return values


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _sum_usage(usage_by_qid: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    prompt = sum(int(item["prompt_tokens"]) for item in usage_by_qid.values())
    completion = sum(int(item["completion_tokens"]) for item in usage_by_qid.values())
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _sanitize_error(message: str, model_config: ModelConfig) -> str:
    safe = message
    for secret in (model_config.api_key, model_config.api_base):
        if secret:
            safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", safe)
    safe = re.sub(r"https?://[^\s'\"<>]+", "<redacted-url>", safe)
    return safe[:2000]


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
