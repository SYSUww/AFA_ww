#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import (
    load_b_questions,
    validate_b_answer,
    write_b_submission,
)
from afa_agent.b_board.runner import (
    RUN_MODE_SUBMISSION,
    BBoardActualRunner,
    _answer_artifact_signature,
    _artifact_from_dict,
    _sum_tokens,
)
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import (
    ensure_dir,
    read_json,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate selected Qwen reasoning rows from clean frozen answer "
            "checkpoints without re-running answer generation"
        )
    )
    parser.add_argument("--answer-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qids", nargs="+", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument(
        "--evidence-char-limit",
        type=int,
        default=1800,
        help="Maximum characters retained from each reasoning evidence item",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    answer_run = (ROOT / args.answer_run).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    source_manifest = read_json(answer_run / "run_manifest.json")
    source_model = dict(source_manifest.get("model") or {})
    model_name = str(source_model.get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        raise ValueError(
            "answer run does not declare an allowed Qwen3.5/Qwen3.6/Qwen3.7 "
            f"model (got {model_name or 'missing'})"
        )
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    question_by_qid = {item.qid: item for item in questions}
    qids = list(dict.fromkeys(str(qid).strip() for qid in args.qids))
    missing_questions = sorted(set(qids) - set(question_by_qid))
    if missing_questions:
        raise ValueError(f"unknown qids: {missing_questions}")
    source_rows = read_json(answer_run / "answer_artifacts.json")
    source_artifacts = {
        str(row["qid"]): _artifact_from_dict(row)
        for row in source_rows
    }
    missing_artifacts = sorted(set(qids) - set(source_artifacts))
    if missing_artifacts:
        raise ValueError(
            f"answer run has no clean checkpoints for: {missing_artifacts}"
        )

    runner = BBoardActualRunner(
        questions=[question_by_qid[qid] for qid in qids],
        reasoning_evidence_char_limit=args.evidence_char_limit,
        run_mode=RUN_MODE_SUBMISSION,
    )
    if runner.config.model.model_name != model_name:
        raise ValueError(
            "configured Qwen model does not match answer run: "
            f"{runner.config.model.model_name!r} != {model_name!r}"
        )

    results = []
    usage_rows = []
    for index, qid in enumerate(qids, start=1):
        answer_artifact = source_artifacts[qid]
        answer_signature = _answer_artifact_signature(answer_artifact)
        result = runner.reasoning_one(
            question_by_qid[qid],
            answer_artifact,
        )
        if _answer_artifact_signature(result) != answer_signature:
            raise ValueError(f"{qid}: reasoning changed frozen answer artifact")
        validate_b_answer(
            question_by_qid[qid],
            result.to_submission_answer(),
        )
        reasoning_ledger = dict(
            result.decision_trace.get("reasoning_api_usage_ledger") or {}
        )
        calls = list(reasoning_ledger.get("calls") or [])
        usage = {
            field_name: sum(
                int(call["token_usage"][field_name])
                for call in calls
            )
            for field_name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
        usage_rows.append(
            {
                "qid": qid,
                "status": "success",
                "call_count": len(calls),
                "calls": calls,
                "token_usage": usage,
                "answer_artifact_sha256": answer_signature,
            }
        )
        results.append(result)
        print(f"reasoning {index}/{len(qids)} {qid}", flush=True)

    ensure_dir(output_dir)
    write_json(
        output_dir / "answers.json",
        [item.to_dict() for item in results],
    )
    write_jsonl(output_dir / "reasoning_usage_ledger.jsonl", usage_rows)
    target_questions = [question_by_qid[qid] for qid in qids]
    write_b_submission(
        output_dir / "research_submit.csv",
        target_questions,
        [item.to_submission_answer() for item in results],
        audit_ready=False,
    )
    reasoning_usage = {
        field_name: sum(
            int(row["token_usage"][field_name]) for row in usage_rows
        )
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }
    manifest = {
        "run_id": output_dir.name,
        "runner": "b_reasoning_from_checkpoints_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "run_mode": "research_patch",
        "model": source_model,
        "source_answer_run": str(answer_run),
        "source_answer_artifacts_sha256": _sha256(
            answer_run / "answer_artifacts.json"
        ),
        "expected_question_count": len(qids),
        "answered_question_count": len(results),
        "answered_qids": qids,
        "failed_qids": [],
        "reasoning_api_call_count": sum(
            int(row["call_count"]) for row in usage_rows
        ),
        "reasoning_token_usage": reasoning_usage,
        "combined_token_usage": _sum_tokens(results),
        "answer_artifact_signature_verified": True,
        "answer_parts_preserved": True,
        "source_model_verified": True,
        "reasoning_evidence_char_limit": args.evidence_char_limit,
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "partial_reasoning_patch_requires_full_assembly"
        ],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
