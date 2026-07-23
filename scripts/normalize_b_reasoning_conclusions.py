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

from afa_agent.b_board.merge import _validate_usage_ledger
from afa_agent.b_board.reasoning_schema import (
    SUBMISSION_REASONING_NORMALIZATION_VERSION,
    normalize_submission_reasoning_payload,
)
from afa_agent.b_board.runner import _artifact_from_dict, _sum_tokens
from afa_agent.b_board.submission_policy import is_allowed_submission_model
from afa_agent.io_utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append exact frozen conclusions to otherwise complete Qwen reasoning"
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_dir = (ROOT / args.source_run).resolve()
    destination = (ROOT / args.run_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")
    source_manifest = read_json(source_dir / "run_manifest.json")
    source_model = dict(source_manifest.get("model") or {})
    model_name = str(source_model.get("model_name", ""))
    if not is_allowed_submission_model(model_name):
        raise ValueError(
            "source run does not declare an allowed Qwen3.5/Qwen3.6/Qwen3.7 "
            f"model (got {model_name or 'missing'})"
        )

    source_answers_path = source_dir / "answers.json"
    results = []
    for row in read_json(source_answers_path):
        artifact = _artifact_from_dict(row)
        _validate_usage_ledger(artifact)
        reasoning_trace = dict(
            artifact.decision_trace.get("submission_reasoning") or {}
        )
        payload, normalizations = normalize_submission_reasoning_payload(
            {
                "answer_parts": list(artifact.answer_parts),
                "grounding_status": reasoning_trace.get("grounding_status"),
                "missing_support": [],
                "reasoning": artifact.decision_summary,
            },
            frozen_answer_parts=artifact.answer_parts,
        )
        conclusion_changes = [
            item
            for item in normalizations
            if item.get("reason") == "append_exact_frozen_answer_conclusion"
        ]
        if not conclusion_changes:
            continue
        artifact.decision_summary = str(payload["reasoning"])
        artifact.decision_trace = {
            **artifact.decision_trace,
            "submission_reasoning": {
                **reasoning_trace,
                "payload_normalizations": [
                    *list(reasoning_trace.get("payload_normalizations") or []),
                    *conclusion_changes,
                ],
            },
            "reasoning_conclusion_normalization": {
                "version": SUBMISSION_REASONING_NORMALIZATION_VERSION,
                "answer_parts_preserved": True,
                "token_usage_preserved": True,
            },
        }
        _validate_usage_ledger(artifact)
        results.append(artifact)
    if not results:
        raise ValueError("source run has no reasoning conclusion gaps")

    ensure_dir(destination)
    write_json(destination / "answers.json", [item.to_dict() for item in results])
    manifest = {
        "run_id": destination.name,
        "runner": "b_actual_reasoning_conclusion_normalization_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "model": source_model,
        "source_run": str(source_dir),
        "source_answers_sha256": _sha256(source_answers_path),
        "normalization_version": SUBMISSION_REASONING_NORMALIZATION_VERSION,
        "expected_question_count": len(results),
        "answered_question_count": len(results),
        "answered_qids": [item.qid for item in results],
        "failed_qids": [],
        "token_usage": _sum_tokens(results),
        "new_generation_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "answer_parts_preserved": True,
        "token_usage_preserved": True,
        "source_model_verified": True,
        "source_usage_lineage_verified": True,
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "partial_reasoning_normalization_source_must_be_assembled_with_full_coverage"
        ],
    }
    write_json(destination / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
