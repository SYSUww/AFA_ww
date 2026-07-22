from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from afa_agent.b_board.io import BAnswer, BQuestion, validate_b_submission, write_b_submission
from afa_agent.io_utils import ensure_dir, read_json, write_json


BASELINE_CANDIDATE_VERSION = "existing_decision_summary_v1"


def materialize_existing_reasoning_candidate(
    *,
    source_submission_path: Path,
    source_answers_path: Path,
    questions: Sequence[BQuestion],
    output_dir: Path,
) -> dict[str, Any]:
    """Create a research-only nine-column baseline without changing answers.

    This adapter is intentionally not submission-eligible: legacy runs do not
    carry an append-only per-call ledger, and some rows can report zero usage.
    It exists only to measure the incumbent decision summaries before a clean
    reasoning-generation experiment.
    """

    source_submission = Path(source_submission_path).resolve()
    source_answers = Path(source_answers_path).resolve()
    destination = Path(output_dir).resolve()
    ensure_dir(destination)
    artifacts = read_json(source_answers)
    if not isinstance(artifacts, list):
        raise ValueError("source answers.json must contain an array")
    artifact_by_qid: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError("source answers.json rows must be objects")
        qid = str(row.get("qid") or "")
        if not qid or qid in artifact_by_qid:
            raise ValueError(f"source answers.json contains invalid or duplicate qid {qid!r}")
        artifact_by_qid[qid] = row

    legacy_by_qid, source_totals = _load_source_submission(source_submission, questions)
    answers: list[BAnswer] = []
    for question in questions:
        artifact = artifact_by_qid.get(question.qid)
        if artifact is None:
            raise ValueError(f"source answers.json is missing {question.qid}")
        source_answer = legacy_by_qid[question.qid]
        artifact_parts = tuple(str(item) for item in artifact.get("answer_parts", []))
        if artifact_parts != source_answer.answer_parts:
            raise ValueError(
                f"{question.qid}: source CSV answers differ from source answers.json"
            )
        answers.append(
            BAnswer(
                qid=question.qid,
                answer_parts=source_answer.answer_parts,
                prompt_tokens=source_answer.prompt_tokens,
                completion_tokens=source_answer.completion_tokens,
                total_tokens=source_answer.total_tokens,
                reasoning=str(artifact.get("decision_summary") or "").strip(),
            )
        )

    if set(artifact_by_qid) != {item.qid for item in questions}:
        extras = sorted(set(artifact_by_qid) - {item.qid for item in questions})
        raise ValueError(f"source answers.json contains unexpected qids: {extras}")
    output_path = destination / "research_submit.csv"
    write_b_submission(output_path, questions, answers, audit_ready=False)
    validate_b_submission(output_path, questions, audit_ready=False)
    output_totals = {
        "prompt_tokens": sum(item.prompt_tokens for item in answers),
        "completion_tokens": sum(item.completion_tokens for item in answers),
        "total_tokens": sum(int(item.total_tokens or 0) for item in answers),
    }
    if output_totals != source_totals:
        raise ValueError("materialized candidate changed source token totals")

    manifest = {
        "version": BASELINE_CANDIDATE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "complete",
        "run_mode": "research",
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "legacy_source_model_is_not_verified_as_allowlisted",
            "legacy_source_has_no_complete_per_call_usage_ledger",
            "existing_reasoning_was_not_regenerated_for_the_new_scoring_system",
        ],
        "answer_changes": 0,
        "question_count": len(answers),
        "below_20_non_whitespace_count": sum(
            len("".join(item.reasoning.split())) < 20 for item in answers
        ),
        "zero_usage_row_count": sum(int(item.total_tokens or 0) == 0 for item in answers),
        "token_usage": output_totals,
        "source_submission_path": str(source_submission),
        "source_submission_sha256": _file_sha256(source_submission),
        "source_answers_path": str(source_answers),
        "source_answers_sha256": _file_sha256(source_answers),
        "research_submission_path": str(output_path),
        "research_submission_sha256": _file_sha256(output_path),
    }
    write_json(destination / "candidate_manifest.json", manifest)
    return manifest


def _load_source_submission(
    path: Path,
    questions: Sequence[BQuestion],
) -> tuple[dict[str, BAnswer], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        answer_prefix = "answer_" if "answer_1" in fieldnames else "answer"
        required = {
            "qid",
            f"{answer_prefix}1",
            f"{answer_prefix}2",
            f"{answer_prefix}3",
            f"{answer_prefix}4",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        if not required.issubset(fieldnames):
            raise ValueError(f"{path}: unsupported source submission columns")
        rows = list(reader)

    answer_rows = [row for row in rows if row.get("qid") != "summary"]
    if [row.get("qid") for row in answer_rows] != [item.qid for item in questions]:
        raise ValueError(f"{path}: source qid order does not match questions")
    answers: dict[str, BAnswer] = {}
    for row, question in zip(answer_rows, questions):
        prompt = _token(row.get("prompt_tokens"), f"{question.qid}.prompt_tokens")
        completion = _token(
            row.get("completion_tokens"), f"{question.qid}.completion_tokens"
        )
        total = _token(row.get("total_tokens"), f"{question.qid}.total_tokens")
        if total != prompt + completion:
            raise ValueError(f"{question.qid}: source total token count is inconsistent")
        answers[question.qid] = BAnswer(
            qid=question.qid,
            answer_parts=tuple(
                str(row[f"{answer_prefix}{index}"])
                for index in range(1, question.answer_slots + 1)
            ),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    totals = {
        "prompt_tokens": sum(item.prompt_tokens for item in answers.values()),
        "completion_tokens": sum(item.completion_tokens for item in answers.values()),
        "total_tokens": sum(int(item.total_tokens or 0) for item in answers.values()),
    }
    summary_rows = [row for row in rows if row.get("qid") == "summary"]
    if len(summary_rows) > 1:
        raise ValueError(f"{path}: multiple summary rows")
    if summary_rows:
        recorded = {
            "prompt_tokens": _token(summary_rows[0].get("prompt_tokens"), "summary.prompt_tokens"),
            "completion_tokens": _token(
                summary_rows[0].get("completion_tokens"), "summary.completion_tokens"
            ),
            "total_tokens": _token(summary_rows[0].get("total_tokens"), "summary.total_tokens"),
        }
        if recorded != totals:
            raise ValueError(f"{path}: summary tokens differ from question rows")
    return answers, totals


def _token(value: Any, name: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
