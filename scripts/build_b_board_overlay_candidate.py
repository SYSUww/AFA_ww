#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.candidate_overlay import merge_candidate_artifacts
from afa_agent.b_board.io import (
    load_b_questions,
    validate_b_answer,
    validate_b_submission,
    write_b_submission,
)
from afa_agent.b_board.runner import (
    CALCULATION_JOINT_REASONING_PROMPT_VERSION,
    BAnswerArtifact,
    _answer_checkpoint_from_completed_artifact,
    _artifact_from_dict,
    _combined_usage_ledger_rows,
    _reasoning_usage_ledger_rows,
    _sum_reasoning_tokens,
    _sum_tokens,
    _usage_ledger_rows,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay disjoint completed Qwen patch runs onto a complete "
            "B-board run and rebuild all submission ledgers"
        )
    )
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument(
        "--overlay-run",
        type=Path,
        action="append",
        required=True,
        help="Completed partial run containing answers.json; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument(
        "--research-only",
        action="store_true",
        help=(
            "Build an audit-ready research composite without marking it "
            "submission-eligible. Required when an overlay contains a "
            "research-only guarded-adaptive calculation artifact."
        ),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifacts(run_dir: Path) -> list[BAnswerArtifact]:
    rows = read_json(run_dir / "answers.json")
    if not isinstance(rows, list):
        raise ValueError(f"{run_dir}: answers.json must contain an array")
    return [_artifact_from_dict(row) for row in rows]


def _model_name(manifest: dict[str, Any], run_dir: Path) -> str:
    name = str(dict(manifest.get("model") or {}).get("model_name", "")).strip()
    if not name:
        raise ValueError(f"{run_dir}: manifest has no model name")
    return name


def _validate_artifact_call_models(
    artifacts: list[BAnswerArtifact],
    *,
    model_name: str,
) -> None:
    for artifact in artifacts:
        ledger = dict(artifact.decision_trace.get("api_usage_ledger") or {})
        calls = list(ledger.get("calls") or [])
        if not calls:
            raise ValueError(f"{artifact.qid}: combined usage ledger has no calls")
        unexpected = sorted(
            {
                str(dict(call).get("model_name", ""))
                for call in calls
                if str(dict(call).get("model_name", "")) != model_name
            }
        )
        if unexpected:
            raise ValueError(
                f"{artifact.qid}: usage call models {unexpected} "
                f"do not match {model_name}"
            )


def _reject_research_only_artifacts(
    artifacts: list[BAnswerArtifact],
    *,
    run_dir: Path,
    allow_guarded_adaptive: bool = False,
) -> None:
    joint_qids = [
        artifact.qid
        for artifact in artifacts
        if (
            "joint_choice_generation" in artifact.decision_trace
            or artifact.decision_trace.get(
                "calculation_joint_reasoning_enabled"
            )
            is True
            or str(
                dict(
                    artifact.decision_trace.get("submission_reasoning")
                    or {}
                ).get("generation_mode", "")
            ).startswith("joint_")
            or str(
                dict(
                    artifact.decision_trace.get("submission_reasoning")
                    or {}
                ).get("generation_mode", "")
            )
            == "calculation_plan_answer_and_reasoning"
            or str(
                dict(
                    artifact.decision_trace.get("submission_reasoning")
                    or {}
                ).get("prompt_version", "")
            )
            in {
                CALCULATION_JOINT_REASONING_PROMPT_VERSION,
                "answer_stage_decision_summary_legacy_shadow_v1",
            }
        )
    ]
    if joint_qids:
        raise ValueError(
            f"{run_dir}: research-only joint answer/reasoning artifacts "
            f"cannot be overlaid into a submission candidate: {joint_qids}"
        )
    adaptive_qids = [
        artifact.qid
        for artifact in artifacts
        if artifact.decision_trace.get(
            "calculation_guarded_adaptive_thinking_enabled"
        )
        is True
    ]
    if adaptive_qids and not allow_guarded_adaptive:
        raise ValueError(
            f"{run_dir}: research-only guarded-adaptive calculation "
            "artifacts cannot be overlaid into a submission candidate: "
            f"{adaptive_qids}"
        )


def main() -> None:
    args = parse_args()
    base_run = _resolve(args.base_run)
    overlay_runs = [_resolve(path) for path in args.overlay_run]
    output_dir = _resolve(args.output_dir)
    research_only = bool(args.research_only)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    base_manifest = dict(read_json(base_run / "run_manifest.json"))
    base_model_name = _model_name(base_manifest, base_run)
    if base_manifest.get("status") != "complete":
        raise ValueError(f"{base_run}: base run is not complete")
    if not is_allowed_submission_model(base_model_name):
        raise ValueError(f"{base_run}: base model is not allowlisted Qwen")

    base_artifacts = _load_artifacts(base_run)
    _reject_research_only_artifacts(
        base_artifacts,
        run_dir=base_run,
        allow_guarded_adaptive=research_only,
    )
    overlay_artifact_groups: list[list[BAnswerArtifact]] = []
    overlay_lineage: list[dict[str, Any]] = []
    for run_dir in overlay_runs:
        manifest_path = run_dir / "run_manifest.json"
        manifest = dict(read_json(manifest_path))
        if manifest.get("status") != "complete":
            raise ValueError(f"{run_dir}: overlay run is not complete")
        overlay_model_name = _model_name(manifest, run_dir)
        if overlay_model_name != base_model_name:
            raise ValueError(
                f"{run_dir}: overlay model {overlay_model_name!r} does not "
                f"match base model {base_model_name!r}"
            )
        artifacts = _load_artifacts(run_dir)
        _reject_research_only_artifacts(
            artifacts,
            run_dir=run_dir,
            allow_guarded_adaptive=research_only,
        )
        overlay_artifact_groups.append(artifacts)
        overlay_lineage.append(
            {
                "run_dir": str(run_dir),
                "manifest_sha256": _sha256(manifest_path),
                "answers_sha256": _sha256(run_dir / "answers.json"),
                "qids": [item.qid for item in artifacts],
            }
        )

    merged, overlaid_qids = merge_candidate_artifacts(
        questions=questions,
        base_artifacts=base_artifacts,
        overlay_artifact_groups=overlay_artifact_groups,
    )
    for question, artifact in zip(questions, merged):
        validate_b_answer(question, artifact.to_submission_answer())
    _validate_artifact_call_models(merged, model_name=base_model_name)

    answer_checkpoints = [
        _answer_checkpoint_from_completed_artifact(item) for item in merged
    ]
    final_by_qid = {item.qid: item for item in merged}
    answer_by_qid = {item.qid: item for item in answer_checkpoints}
    total_usage = _sum_tokens(merged)
    answer_usage = _sum_tokens(answer_checkpoints)
    reasoning_usage = _sum_reasoning_tokens(merged)

    ensure_dir(output_dir)
    answer_artifacts_path = output_dir / "answer_artifacts.json"
    final_artifacts_path = output_dir / "answers.json"
    answer_ledger_path = output_dir / "answer_usage_ledger.jsonl"
    reasoning_ledger_path = output_dir / "reasoning_usage_ledger.jsonl"
    usage_ledger_path = output_dir / "usage_ledger.jsonl"
    submission_path = output_dir / (
        "research_submit.csv" if research_only else "submit.csv"
    )
    write_json(
        answer_artifacts_path,
        [item.to_dict() for item in answer_checkpoints],
    )
    write_json(final_artifacts_path, [item.to_dict() for item in merged])
    write_jsonl(
        answer_ledger_path,
        _usage_ledger_rows(
            answer_checkpoints,
            [],
            trace_key="answer_api_usage_ledger",
        ),
    )
    write_jsonl(
        reasoning_ledger_path,
        _reasoning_usage_ledger_rows(merged, []),
    )
    write_jsonl(
        usage_ledger_path,
        _combined_usage_ledger_rows(
            questions,
            answer_by_qid,
            final_by_qid,
            [],
            [],
        ),
    )
    write_b_submission(
        submission_path,
        questions,
        [item.to_submission_answer() for item in merged],
        audit_ready=True,
    )
    validate_b_submission(submission_path, questions, audit_ready=True)

    manifest = {
        "run_id": output_dir.name,
        "runner": "b_candidate_overlay_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "stage": "full",
        "run_mode": "research" if research_only else "submission",
        "model": dict(base_manifest.get("model") or {}),
        "expected_question_count": len(questions),
        "answered_question_count": len(merged),
        "answer_completed_count": len(answer_checkpoints),
        "reasoning_completed_count": len(merged),
        "answer_failed_qids": [],
        "reasoning_failed_qids": [],
        "failed_qids": [],
        "token_usage": total_usage,
        "answer_token_usage": answer_usage,
        "reasoning_token_usage": reasoning_usage,
        "failed_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "generation_token_usage": total_usage,
        "answer_retry_failure_count": 0,
        "reasoning_retry_failure_count": 0,
        "retry_failure_count": 0,
        "answer_artifacts_path": str(answer_artifacts_path),
        "answer_failures_path": None,
        "reasoning_failures_path": None,
        "answer_usage_ledger_path": str(answer_ledger_path),
        "reasoning_usage_ledger_path": str(reasoning_ledger_path),
        "usage_ledger_path": str(usage_ledger_path),
        "submission_eligible": not research_only,
        "submission_ineligibility_reasons": (
            ["research_overlay_is_not_submission_eligible"]
            if research_only
            else []
        ),
        "submission_path": None if research_only else str(submission_path),
        "research_submission_path": (
            str(submission_path) if research_only else None
        ),
        "base_run": {
            "run_dir": str(base_run),
            "manifest_sha256": _sha256(base_run / "run_manifest.json"),
            "answers_sha256": _sha256(base_run / "answers.json"),
        },
        "overlay_runs": overlay_lineage,
        "overlaid_qids": overlaid_qids,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
