#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import (
    BQuestion,
    load_b_questions,
    validate_b_answer,
    validate_b_submission,
    write_b_submission,
)
from afa_agent.b_board.runner import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_PARSED_ROOT,
    DEFAULT_STRATEGY_PATH,
    RUN_MODE_RESEARCH,
    BAnswerArtifact,
    BAnswerGenerationError,
    BBoardActualRunner,
    _combined_usage_ledger_rows,
    _failure_record,
    _reasoning_usage_ledger_rows,
    _sum_failure_tokens,
    _sum_reasoning_tokens,
    _sum_tokens,
    _usage_ledger_rows,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import ensure_dir, read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only A/B: one Qwen call per choice question produces "
            "option judgments, answer, and reasoning"
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument(
        "--parsed-root",
        default=str(DEFAULT_PARSED_ROOT.relative_to(ROOT)),
    )
    parser.add_argument(
        "--index-root",
        default=str(DEFAULT_INDEX_ROOT.relative_to(ROOT)),
    )
    parser.add_argument(
        "--strategy-config",
        default=str(DEFAULT_STRATEGY_PATH.relative_to(ROOT)),
    )
    parser.add_argument("--locator-attempt", default="attempt_43")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qid-file", default="")
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--per-option-top-k", type=int, default=4)
    parser.add_argument("--max-evidence-items", type=int, default=16)
    parser.add_argument("--evidence-char-limit", type=int, default=1200)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_qids(
    args: argparse.Namespace,
    *,
    questions: list[BQuestion],
    source_run: Path,
) -> list[str]:
    requested = [str(qid).strip() for qid in args.qid if str(qid).strip()]
    if args.qid_file:
        requested.extend(
            line.strip()
            for line in _resolve(Path(args.qid_file))
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    requested = list(dict.fromkeys(requested))
    by_qid = {item.qid: item for item in questions}
    if requested:
        unknown = sorted(set(requested) - set(by_qid))
        if unknown:
            raise ValueError(f"unknown qids: {unknown}")
        return requested

    source_rows = read_json(source_run / "answers.json")
    if not isinstance(source_rows, list):
        raise ValueError("source answers.json must contain an array")
    selected: list[str] = []
    for row in source_rows:
        qid = str(dict(row).get("qid", "")).strip()
        question = by_qid.get(qid)
        if question is None or question.answer_format == "calculation":
            continue
        trace = dict(dict(row).get("decision_trace") or {})
        ledger = dict(
            trace.get("answer_api_usage_ledger")
            or trace.get("api_usage_ledger")
            or {}
        )
        if int(ledger.get("call_count", len(ledger.get("calls") or []))) > 0:
            selected.append(qid)
    if not selected:
        raise ValueError("source run has no model-generated choice questions")
    return selected


def _persist(
    *,
    output_dir: Path,
    selected: list[BQuestion],
    artifacts_by_qid: dict[str, BAnswerArtifact],
    failures: list[dict[str, Any]],
) -> None:
    artifacts = [
        artifacts_by_qid[item.qid]
        for item in selected
        if item.qid in artifacts_by_qid
    ]
    final_by_qid = {item.qid: item for item in artifacts}
    write_json(
        output_dir / "joint_outputs.json",
        [item.to_dict() for item in artifacts],
    )
    write_jsonl(output_dir / "answer_failures.jsonl", failures)
    write_jsonl(output_dir / "reasoning_failures.jsonl", [])
    write_jsonl(output_dir / "failures.jsonl", failures)
    write_jsonl(
        output_dir / "answer_usage_ledger.jsonl",
        _usage_ledger_rows(
            artifacts,
            failures,
            trace_key="answer_api_usage_ledger",
        ),
    )
    write_jsonl(
        output_dir / "reasoning_usage_ledger.jsonl",
        _reasoning_usage_ledger_rows(artifacts, []),
    )
    write_jsonl(
        output_dir / "usage_ledger.jsonl",
        _combined_usage_ledger_rows(
            selected,
            {},
            final_by_qid,
            failures,
            [],
        ),
    )
    successful_questions = [
        item for item in selected if item.qid in artifacts_by_qid
    ]
    if successful_questions:
        write_b_submission(
            output_dir / "research_submit.csv",
            successful_questions,
            [
                artifacts_by_qid[item.qid].to_submission_answer()
                for item in successful_questions
            ],
            audit_ready=True,
        )


def main() -> None:
    args = parse_args()
    source_run = _resolve(args.source_run)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)

    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    selected_qids = _selected_qids(
        args,
        questions=questions,
        source_run=source_run,
    )
    by_qid = {item.qid: item for item in questions}
    selected = [by_qid[qid] for qid in selected_qids]
    unsupported = [
        item.qid
        for item in selected
        if item.answer_format not in {"multi", "mcq", "tf"}
    ]
    if unsupported:
        raise ValueError(f"joint choice does not support qids: {unsupported}")

    source_manifest = dict(read_json(source_run / "run_manifest.json"))
    if source_manifest.get("status") != "complete":
        raise ValueError("source run must be complete")
    if source_manifest.get("run_mode") != "submission":
        raise ValueError("source run must be a submission-mode run")
    if source_manifest.get("submission_eligible") is not True:
        raise ValueError("source run must be submission eligible")
    source_model = str(
        dict(source_manifest.get("model") or {}).get("model_name", "")
    ).strip()
    if not is_allowed_submission_model(source_model):
        raise ValueError(
            f"source run model is not allowlisted Qwen: {source_model!r}"
        )
    source_answers_path = source_run / "answers.json"
    source_submission_path = Path(
        str(source_manifest.get("submission_path", ""))
    ).resolve()
    if not source_answers_path.is_file():
        raise ValueError("source run has no answers.json")
    if not source_submission_path.is_file():
        raise ValueError("source run has no submission CSV")
    validate_b_submission(
        source_submission_path,
        questions,
        audit_ready=True,
    )
    source_answers = read_json(source_answers_path)
    if (
        not isinstance(source_answers, list)
        or len(source_answers) != len(questions)
        or {str(row.get("qid", "")) for row in source_answers}
        != {item.qid for item in questions}
    ):
        raise ValueError("source run answers do not cover all questions")
    unexpected_call_models = sorted(
        {
            str(call.get("model_name", ""))
            for row in source_answers
            for call in list(
                dict(
                    dict(row.get("decision_trace") or {}).get(
                        "api_usage_ledger"
                    )
                    or {}
                ).get("calls")
                or []
            )
            if str(call.get("model_name", "")) != source_model
        }
    )
    if unexpected_call_models:
        raise ValueError(
            "source answer calls use unexpected models: "
            f"{unexpected_call_models}"
        )
    source_lineage = {
        "run_dir": str(source_run),
        "manifest_sha256": _sha256(source_run / "run_manifest.json"),
        "answers_sha256": _sha256(source_answers_path),
        "submission_sha256": _sha256(source_submission_path),
    }
    runner = BBoardActualRunner(
        questions=questions,
        parsed_root=ROOT / args.parsed_root,
        index_root=ROOT / args.index_root,
        strategy_path=ROOT / args.strategy_config,
        locator_attempt_id=args.locator_attempt,
        run_mode=RUN_MODE_RESEARCH,
    )
    configured_model = runner.config.model.model_name
    if configured_model != source_model:
        raise ValueError(
            f"configured model {configured_model!r} does not match "
            f"source model {source_model!r}"
        )

    created_at = datetime.now().isoformat(timespec="seconds")
    located = runner.locate(selected)
    write_jsonl(
        output_dir / "locator.jsonl",
        [located[item.qid] for item in selected],
    )
    run_identity = {
        "run_id": output_dir.name,
        "runner": "b_joint_choice_research_v1",
        "created_at": created_at,
        "status": "running",
        "stage": "joint_answer_reasoning",
        "run_mode": "research",
        "model": {"model_name": configured_model},
        "source_run": str(source_run),
        "source_lineage": source_lineage,
        "expected_question_count": len(selected),
        "joint_choice": {
            "one_model_call_per_question": True,
            "thinking_budget": args.thinking_budget,
            "per_option_top_k": args.per_option_top_k,
            "max_evidence_items": args.max_evidence_items,
            "evidence_char_limit": args.evidence_char_limit,
            "selected_qids": selected_qids,
        },
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "research-only joint answer/reasoning experiment",
            "joint generation does not satisfy the frozen-answer handoff contract",
            "model-cited evidence was not passed through domain evidence gates",
        ],
        "answer_artifacts_path": None,
        "joint_outputs_path": str(output_dir / "joint_outputs.json"),
        "submission_path": None,
    }
    write_json(output_dir / "run_manifest.json", run_identity)
    artifacts_by_qid: dict[str, BAnswerArtifact] = {}
    failures: list[dict[str, Any]] = []
    _persist(
        output_dir=output_dir,
        selected=selected,
        artifacts_by_qid=artifacts_by_qid,
        failures=failures,
    )

    def solve(item: BQuestion) -> BAnswerArtifact:
        return runner.joint_choice_one(
            item,
            located[item.qid],
            thinking_budget=args.thinking_budget,
            per_option_top_k=args.per_option_top_k,
            max_evidence_items=args.max_evidence_items,
            evidence_char_limit=args.evidence_char_limit,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(solve, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                artifact = future.result()
                validate_b_answer(item, artifact.to_submission_answer())
                artifacts_by_qid[item.qid] = artifact
            except Exception as exc:
                if isinstance(exc, BAnswerGenerationError):
                    failure = _failure_record(
                        item.qid,
                        exc,
                        stage="joint_choice",
                    )
                else:
                    failure = {
                        "qid": item.qid,
                        "stage": "joint_choice",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "diagnostics": [],
                    }
                failures.append(failure)
            _persist(
                output_dir=output_dir,
                selected=selected,
                artifacts_by_qid=artifacts_by_qid,
                failures=failures,
            )

    artifacts = [
        artifacts_by_qid[item.qid]
        for item in selected
        if item.qid in artifacts_by_qid
    ]
    success_usage = _sum_tokens(artifacts)
    failed_usage = _sum_failure_tokens(failures)
    generation_usage = {
        key: success_usage[key] + failed_usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    failed_qids = [
        item.qid for item in selected if item.qid not in artifacts_by_qid
    ]
    manifest = {
        **run_identity,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete" if not failed_qids else "incomplete",
        "answered_question_count": len(artifacts),
        "answer_completed_count": len(artifacts),
        "reasoning_completed_count": len(artifacts),
        "failed_qids": failed_qids,
        "token_usage": success_usage,
        "answer_token_usage": success_usage,
        "reasoning_token_usage": _sum_reasoning_tokens(artifacts),
        "failed_token_usage": failed_usage,
        "generation_token_usage": generation_usage,
        "retry_failure_count": 0,
        "research_submission_path": (
            str(output_dir / "research_submit.csv")
            if artifacts
            else None
        ),
        "joint_outputs_sha256": _sha256(
            output_dir / "joint_outputs.json"
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
