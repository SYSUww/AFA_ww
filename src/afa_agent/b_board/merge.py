from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from afa_agent.b_board.io import BQuestion, validate_b_answer, write_b_submission
from afa_agent.b_board.runner import (
    BAnswerArtifact,
    _answer_artifact_signature,
    _artifact_from_dict,
    _refresh_combined_api_usage_ledger,
    _sum_tokens,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


COMPOSITE_RUNNER = "b_actual_composite_v1"


def hydrate_reasoning_patch(
    *,
    answer_artifact: BAnswerArtifact,
    reasoning_artifact: BAnswerArtifact,
) -> BAnswerArtifact:
    """Attach a reasoning-only patch to its exact frozen answer checkpoint."""

    if reasoning_artifact.qid != answer_artifact.qid:
        raise ValueError(
            f"reasoning patch qid {reasoning_artifact.qid!r} does not match "
            f"answer qid {answer_artifact.qid!r}"
        )
    _validate_answer_checkpoint_usage(answer_artifact)
    _validate_usage_ledger(reasoning_artifact)
    frozen_signature = _answer_artifact_signature(answer_artifact)
    reasoning_stage = dict(
        reasoning_artifact.decision_trace.get("reasoning_stage") or {}
    )
    if reasoning_stage.get("answer_artifact_sha256") != frozen_signature:
        raise ValueError(
            f"{answer_artifact.qid}: reasoning patch is not sealed to the frozen answer artifact"
        )
    if reasoning_artifact.answer_parts != answer_artifact.answer_parts:
        raise ValueError(
            f"{answer_artifact.qid}: reasoning patch changed frozen answer parts"
        )
    submission_reasoning = dict(
        reasoning_artifact.decision_trace.get("submission_reasoning") or {}
    )
    if (
        reasoning_stage.get("status") != "complete"
        or not reasoning_stage.get("answer_artifact_frozen")
        or not submission_reasoning.get("answer_parts_preserved")
    ):
        raise ValueError(
            f"{answer_artifact.qid}: reasoning patch does not prove frozen-answer preservation"
        )
    reasoning_ledger = dict(
        reasoning_artifact.decision_trace.get("reasoning_api_usage_ledger") or {}
    )
    reasoning_calls = reasoning_ledger.get("calls")
    if not isinstance(reasoning_calls, list) or not reasoning_calls:
        raise ValueError(f"{answer_artifact.qid}: reasoning patch has no successful API call")
    reasoning_token_usage = _sum_ledger_tokens(reasoning_calls)
    reported_reasoning_usage = submission_reasoning.get("token_usage")
    if (
        isinstance(reported_reasoning_usage, Mapping)
        and dict(reported_reasoning_usage) != reasoning_token_usage
    ):
        raise ValueError(
            f"{answer_artifact.qid}: reasoning patch token usage does not match its call ledger"
        )

    hydrated = _artifact_from_dict(answer_artifact.to_dict())
    hydrated.decision_summary = reasoning_artifact.decision_summary
    hydrated.reasoning_evidence_items = [
        dict(item) for item in reasoning_artifact.reasoning_evidence_items
    ]
    hydrated.decision_trace = {
        **hydrated.decision_trace,
        "submission_reasoning": submission_reasoning,
        "reasoning_stage": reasoning_stage,
        "reasoning_api_usage_ledger": {
            "call_count": len(reasoning_calls),
            "calls": [dict(item) for item in reasoning_calls],
        },
        "reasoning_patch_lineage": {
            "source": "reasoning_only_patch_hydration_v1",
            "answer_artifact_sha256": frozen_signature,
        },
    }
    hydrated.token_usage = {
        field_name: int(answer_artifact.token_usage.get(field_name, 0))
        + int(reasoning_token_usage.get(field_name, 0))
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    hydrated = _refresh_combined_api_usage_ledger(hydrated)
    _validate_usage_ledger(hydrated)
    return hydrated


def assemble_answer_run(
    *,
    questions: Sequence[BQuestion],
    source_run_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Build an immutable ordered answer run from one or more source runs.

    Later sources replace the same QID from earlier sources. Artifact coverage and
    submission validity are recorded separately so a baseline with format defects
    can still be evaluated without pretending it is submission-ready.
    """

    if not source_run_dirs:
        raise ValueError("source_run_dirs must not be empty")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"composite run directory is not empty: {destination}")
    question_by_qid = {item.qid: item for item in questions}
    if len(question_by_qid) != len(questions):
        raise ValueError("questions contain duplicate qids")

    artifacts: dict[str, BAnswerArtifact] = {}
    provenance: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    source_failures: list[dict[str, str]] = []
    for source_index, source_dir in enumerate(source_run_dirs):
        resolved = Path(source_dir).resolve()
        answers_path = resolved / "answers.json"
        manifest_path = resolved / "run_manifest.json"
        source_manifest = read_json(manifest_path) if manifest_path.exists() else {}
        source_model = str(dict(source_manifest.get("model") or {}).get("model_name", ""))
        if not is_allowed_submission_model(source_model):
            source_failures.append(
                {
                    "qid": "*",
                    "error": (
                        f"source run {resolved} does not declare an allowed Qwen3.5/Qwen3.6/Qwen3.7 "
                        f"generation model (got {source_model or 'missing'})"
                    ),
                }
            )
        rows = read_json(answers_path)
        if not isinstance(rows, list):
            raise ValueError(f"{answers_path}: expected a JSON array")
        source_qids: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{answers_path}: answer row must be an object")
            artifact = _artifact_from_dict(row)
            if artifact.qid not in question_by_qid:
                raise ValueError(f"unknown answer qid in source run: {artifact.qid}")
            source_qids.append(artifact.qid)
            artifacts[artifact.qid] = artifact
            provenance[artifact.qid] = {
                "source_index": source_index,
                "source_run_dir": str(resolved),
                "answers_sha256": _file_sha256(answers_path),
                "model_name": source_model,
            }
        if len(source_qids) != len(set(source_qids)):
            raise ValueError(f"{answers_path}: duplicate qids")
        sources.append(
            {
                "source_index": source_index,
                "run_dir": str(resolved),
                "answer_count": len(source_qids),
                "answers_sha256": _file_sha256(answers_path),
                "model_name": source_model,
            }
        )

    missing = [item.qid for item in questions if item.qid not in artifacts]
    if missing:
        raise ValueError(f"composite answer coverage is incomplete: {missing}")
    ordered = [artifacts[item.qid] for item in questions]
    invalid: list[dict[str, str]] = list(source_failures)
    for question, artifact in zip(questions, ordered):
        try:
            validate_b_answer(question, artifact.to_submission_answer())
            _validate_usage_ledger(artifact)
        except ValueError as exc:
            invalid.append({"qid": question.qid, "error": str(exc)})

    ensure_dir(destination)
    write_json(destination / "answers.json", [item.to_dict() for item in ordered])
    write_json(destination / "answer_provenance.json", provenance)
    write_json(destination / "source_runs.json", sources)
    write_jsonl(
        destination / "usage_ledger.jsonl",
        [
            {
                "qid": item.qid,
                "status": "success",
                "call_count": int(
                    dict(item.decision_trace.get("api_usage_ledger") or {}).get(
                        "call_count", 0
                    )
                ),
                "calls": list(
                    dict(item.decision_trace.get("api_usage_ledger") or {}).get(
                        "calls", []
                    )
                ),
                "token_usage": dict(item.token_usage),
            }
            for item in ordered
        ],
    )
    submission_path: str | None = None
    if not invalid:
        path = destination / "submit.csv"
        write_b_submission(
            path,
            questions,
            [item.to_submission_answer() for item in ordered],
            audit_ready=True,
        )
        submission_path = str(path)

    combined_usage = _sum_tokens(ordered)
    manifest = {
        "run_id": destination.name,
        "runner": COMPOSITE_RUNNER,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "run_mode": "submission",
        "artifact_complete": True,
        "expected_question_count": len(questions),
        "answered_question_count": len(ordered),
        "failed_qids": [],
        "token_usage": combined_usage,
        "generation_token_usage": combined_usage,
        "generation_models": sorted(
            {
                str(item.get("model_name", ""))
                for item in sources
                if str(item.get("model_name", ""))
            }
        ),
        "submission_valid": not invalid,
        "submission_eligible": not invalid,
        "submission_ineligibility_reasons": [
            item["error"] for item in invalid
        ],
        "submission_validation_failures": invalid,
        "submission_path": submission_path,
        "source_runs": sources,
    }
    if len(manifest["generation_models"]) == 1:
        manifest["model"] = {
            "model_name": manifest["generation_models"][0],
        }
    write_json(destination / "run_manifest.json", manifest)
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_usage_ledger(artifact: BAnswerArtifact) -> None:
    ledger = dict(artifact.decision_trace.get("api_usage_ledger") or {})
    calls = ledger.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{artifact.qid}: missing per-call API usage ledger")
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, Mapping):
            raise ValueError(f"{artifact.qid}: usage ledger call {index} is invalid")
        model_name = str(call.get("model_name", ""))
        if not is_allowed_submission_model(model_name):
            raise ValueError(
                f"{artifact.qid}: usage ledger call {index} used disallowed model {model_name!r}"
            )
        usage = dict(call.get("token_usage") or {})
        values = [usage.get(field_name) for field_name in totals]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"{artifact.qid}: usage ledger call {index} has invalid raw usage")
        if values[2] != values[0] + values[1]:
            raise ValueError(f"{artifact.qid}: usage ledger call {index} has inconsistent total_tokens")
        for field_name, value in zip(totals, values):
            totals[field_name] += value
    if totals != artifact.token_usage:
        raise ValueError(
            f"{artifact.qid}: usage ledger total {totals} does not match row usage {artifact.token_usage}"
        )


def _validate_answer_checkpoint_usage(artifact: BAnswerArtifact) -> None:
    """Allow an explicitly recorded zero-call deterministic answer checkpoint."""

    ledger = dict(
        artifact.decision_trace.get("answer_api_usage_ledger")
        or artifact.decision_trace.get("api_usage_ledger")
        or {}
    )
    calls = ledger.get("calls")
    if isinstance(calls, list) and calls:
        _validate_usage_ledger(artifact)
        return
    zero_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if (
        ledger.get("call_count") == 0
        and calls == []
        and artifact.token_usage == zero_usage
    ):
        return
    raise ValueError(
        f"{artifact.qid}: deterministic answer checkpoint must record "
        "call_count=0, calls=[], and exact zero token usage"
    )


def _sum_ledger_tokens(calls: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for call in calls:
        usage = dict(call.get("token_usage") or {})
        for field_name in totals:
            value = usage.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"reasoning call has invalid {field_name}")
            totals[field_name] += value
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("reasoning call has inconsistent total_tokens")
    return totals
