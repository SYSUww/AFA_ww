#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.revalidate import revalidate_calculation_artifact
from afa_agent.b_board.runner import _artifact_from_dict, _sum_tokens
from afa_agent.io_utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revalidate incumbent B-board calculation traces without model replanning"
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument(
        "--index-root", default="artifacts/preprocessed_loop_candidates/index"
    )
    parser.add_argument("--qid", action="append", required=True)
    parser.add_argument(
        "--allow-format-change",
        action="store_true",
        help="Allow deterministic answer changes caused only by the current format contract",
    )
    parser.add_argument(
        "--supporting-evidence",
        action="append",
        default=[],
        help="qid=unit_id,unit_id; repeat for multiple qids",
    )
    return parser.parse_args()


def _support_map(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --supporting-evidence value: {value!r}")
        qid, raw_ids = value.split("=", 1)
        evidence_ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
        if not qid.strip() or not evidence_ids:
            raise ValueError(f"invalid --supporting-evidence value: {value!r}")
        result.setdefault(qid.strip(), []).extend(evidence_ids)
    return {key: list(dict.fromkeys(items)) for key, items in result.items()}


def main() -> None:
    args = parse_args()
    source_dir = (ROOT / args.source_run).resolve()
    destination = (ROOT / args.run_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")
    questions = load_b_questions(
        ROOT / args.question_root, ROOT / args.submission_template
    )
    question_by_qid = {item.qid: item for item in questions}
    requested_qids = list(dict.fromkeys(args.qid))
    unknown = sorted(set(requested_qids) - set(question_by_qid))
    if unknown:
        raise ValueError(f"unknown qids: {unknown}")
    source_rows = read_json(source_dir / "answers.json")
    source_by_qid = {
        str(row["qid"]): _artifact_from_dict(row) for row in source_rows
    }
    support_by_qid = _support_map(args.supporting_evidence)
    index_by_domain: dict[str, list[dict[str, object]]] = {}
    results = []
    for qid in requested_qids:
        question = question_by_qid[qid]
        if qid not in source_by_qid:
            raise ValueError(f"source run has no artifact for {qid}")
        if question.domain not in index_by_domain:
            payload = read_json(
                ROOT / args.index_root / question.domain / "index.json"
            )
            index_by_domain[question.domain] = list(payload.get("units", []))
        results.append(
            revalidate_calculation_artifact(
                question=question,
                artifact=source_by_qid[qid],
                index_units=index_by_domain[question.domain],
                supporting_evidence_ids=support_by_qid.get(qid, []),
                allow_format_change=args.allow_format_change,
            )
        )

    ensure_dir(destination)
    write_json(destination / "answers.json", [item.to_dict() for item in results])
    manifest = {
        "run_id": destination.name,
        "runner": "b_actual_incumbent_trace_literal_revalidation_a5",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "source_run_dir": str(source_dir),
        "expected_question_count": len(requested_qids),
        "answered_question_count": len(results),
        "answered_qids": requested_qids,
        "failed_qids": [],
        "token_usage": _sum_tokens(results),
        "generation_token_usage": _sum_tokens(results),
        "answer_preserved": all(
            source_by_qid[item.qid].answer_parts == item.answer_parts for item in results
        ),
        "format_change_allowed": args.allow_format_change,
        "supporting_evidence_ids": support_by_qid,
    }
    write_json(destination / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
