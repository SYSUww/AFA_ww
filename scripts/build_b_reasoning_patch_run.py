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

from afa_agent.b_board.io import load_b_questions, validate_b_answer
from afa_agent.b_board.merge import hydrate_reasoning_patch
from afa_agent.b_board.runner import _artifact_from_dict, _sum_tokens
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate reasoning-only Qwen patches with their frozen answer usage"
    )
    parser.add_argument("--answer-run", required=True)
    parser.add_argument("--reasoning-patch", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
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
    reasoning_patch_path = (ROOT / args.reasoning_patch).resolve()
    destination = (ROOT / args.run_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")

    source_manifest = read_json(answer_run / "run_manifest.json")
    source_model = dict(source_manifest.get("model") or {})
    model_name = str(source_model.get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        raise ValueError(
            "answer source does not declare an allowed Qwen3.5/Qwen3.6/Qwen3.7 "
            f"model (got {model_name or 'missing'})"
        )
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    question_by_qid = {item.qid: item for item in questions}
    answer_rows = read_json(answer_run / "answer_artifacts.json")
    answers = {
        str(row["qid"]): _artifact_from_dict(row)
        for row in answer_rows
    }
    patch_rows = read_json(reasoning_patch_path)
    patches = {
        str(row["qid"]): _artifact_from_dict(row)
        for row in patch_rows
    }
    if len(patches) != len(patch_rows):
        raise ValueError("reasoning patch contains duplicate qids")

    results = []
    for qid, patch in patches.items():
        if qid not in question_by_qid:
            raise ValueError(f"unknown reasoning patch qid: {qid}")
        if qid not in answers:
            raise ValueError(f"answer source has no frozen artifact for {qid}")
        result = hydrate_reasoning_patch(
            answer_artifact=answers[qid],
            reasoning_artifact=patch,
        )
        validate_b_answer(question_by_qid[qid], result.to_submission_answer())
        results.append(result)
    results.sort(key=lambda item: [question.qid for question in questions].index(item.qid))

    ensure_dir(destination)
    write_json(destination / "answers.json", [item.to_dict() for item in results])
    manifest = {
        "run_id": destination.name,
        "runner": "b_actual_reasoning_patch_hydration_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "model": source_model,
        "source_answer_run": str(answer_run),
        "source_answer_artifacts_sha256": _sha256(
            answer_run / "answer_artifacts.json"
        ),
        "source_reasoning_patch": str(reasoning_patch_path),
        "source_reasoning_patch_sha256": _sha256(reasoning_patch_path),
        "expected_question_count": len(results),
        "answered_question_count": len(results),
        "answered_qids": [item.qid for item in results],
        "failed_qids": [],
        "token_usage": _sum_tokens(results),
        "answer_artifact_signature_verified": True,
        "answer_parts_preserved": True,
        "source_model_verified": True,
        "source_usage_lineage_verified": True,
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "partial_reasoning_patch_source_must_be_assembled_with_full_question_coverage"
        ],
    }
    write_json(destination / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
