#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions, write_b_submission
from afa_agent.b_board.runner import _artifact_from_dict
from afa_agent.b_board.scoring import token_efficiency_score
from afa_agent.io_utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a joint-choice research run with its source run"
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--joint-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--source-eval-dir", type=Path)
    parser.add_argument("--joint-eval-dir", type=Path)
    parser.add_argument("--base-reasoning-score", type=float)
    parser.add_argument("--assumed-accuracy-score", type=float)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_lineage(
    *,
    source_run: Path,
    joint_manifest: dict[str, object],
) -> dict[str, str]:
    recorded = dict(joint_manifest.get("source_lineage") or {})
    expected = {
        "run_dir": str(source_run),
        "manifest_sha256": _sha256(source_run / "run_manifest.json"),
        "answers_sha256": _sha256(source_run / "answers.json"),
    }
    source_manifest = dict(read_json(source_run / "run_manifest.json"))
    source_submission = Path(
        str(source_manifest.get("submission_path", ""))
    ).resolve()
    if not source_submission.is_file():
        raise ValueError("source run has no submission CSV")
    expected["submission_sha256"] = _sha256(source_submission)
    mismatches = {
        key: {
            "recorded": str(recorded.get(key, "")),
            "current": value,
        }
        for key, value in expected.items()
        if str(recorded.get(key, "")) != value
    }
    if mismatches:
        raise ValueError(f"joint source lineage mismatch: {mismatches}")
    return expected


def main() -> None:
    args = parse_args()
    source_run = _resolve(args.source_run)
    joint_run = _resolve(args.joint_run)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ensure_dir(output_dir)

    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    question_by_qid = {item.qid: item for item in questions}
    joint_manifest = dict(read_json(joint_run / "run_manifest.json"))
    if joint_manifest.get("runner") != "b_joint_choice_research_v1":
        raise ValueError("joint run has an unexpected runner identity")
    if joint_manifest.get("run_mode") != "research":
        raise ValueError("joint run must remain research-only")
    if joint_manifest.get("submission_eligible") is not False:
        raise ValueError("joint run cannot be submission eligible")
    source_lineage = _validate_source_lineage(
        source_run=source_run,
        joint_manifest=joint_manifest,
    )
    joint_outputs_path = joint_run / "joint_outputs.json"
    if str(joint_manifest.get("joint_outputs_path", "")) != str(
        joint_outputs_path
    ):
        raise ValueError("joint output path does not match its manifest")
    if str(joint_manifest.get("joint_outputs_sha256", "")) != _sha256(
        joint_outputs_path
    ):
        raise ValueError("joint output hash does not match its manifest")
    source = {
        item.qid: item
        for item in (
            _artifact_from_dict(row)
            for row in read_json(source_run / "answers.json")
        )
    }
    joint = {
        item.qid: item
        for item in (
            _artifact_from_dict(row)
            for row in read_json(joint_outputs_path)
        )
    }
    selected_qids = list(
        dict(joint_manifest.get("joint_choice") or {}).get(
            "selected_qids",
            [],
        )
    )
    failures = {
        str(item["qid"]): item
        for item in (
            json.loads(line)
            for line in (joint_run / "failures.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    rows: list[dict[str, object]] = []
    for qid in selected_qids:
        source_artifact = source[qid]
        joint_artifact = joint.get(qid)
        failure = failures.get(qid)
        if joint_artifact is None:
            status = "joint_failure"
            joint_answer = ""
            joint_tokens = int(
                dict((failure or {}).get("token_usage") or {}).get(
                    "total_tokens",
                    0,
                )
            )
            error = str((failure or {}).get("error", "missing artifact"))
        else:
            status = (
                "answer_match"
                if joint_artifact.answer_parts == source_artifact.answer_parts
                else "answer_changed"
            )
            joint_answer = "；".join(joint_artifact.answer_parts)
            joint_tokens = int(
                joint_artifact.token_usage.get("total_tokens", 0)
            )
            error = ""
        source_tokens = int(
            source_artifact.token_usage.get("total_tokens", 0)
        )
        rows.append(
            {
                "qid": qid,
                "status": status,
                "source_answer": "；".join(source_artifact.answer_parts),
                "joint_answer": joint_answer,
                "source_tokens": source_tokens,
                "joint_tokens": joint_tokens,
                "token_delta": joint_tokens - source_tokens,
                "error": error,
            }
        )

    matched_qids = [
        str(row["qid"]) for row in rows if row["status"] == "answer_match"
    ]
    matched_questions = [question_by_qid[qid] for qid in matched_qids]
    if matched_qids:
        write_b_submission(
            output_dir / "source_matched_eval_input.csv",
            matched_questions,
            [source[qid].to_submission_answer() for qid in matched_qids],
            audit_ready=True,
        )
        write_b_submission(
            output_dir / "joint_matched_eval_input.csv",
            matched_questions,
            [joint[qid].to_submission_answer() for qid in matched_qids],
            audit_ready=True,
        )
    matching_rows = [
        row for row in rows if row["status"] == "answer_match"
    ]
    source_matched_tokens = sum(
        int(row["source_tokens"]) for row in matching_rows
    )
    joint_matched_tokens = sum(
        int(row["joint_tokens"]) for row in matching_rows
    )
    report = {
        "source_run": str(source_run),
        "source_lineage_verified": source_lineage,
        "joint_run": str(joint_run),
        "selected_count": len(rows),
        "answer_match_count": len(matching_rows),
        "answer_changed_qids": [
            row["qid"] for row in rows if row["status"] == "answer_changed"
        ],
        "joint_failure_qids": [
            row["qid"] for row in rows if row["status"] == "joint_failure"
        ],
        "source_matched_tokens": source_matched_tokens,
        "joint_matched_tokens": joint_matched_tokens,
        "matched_token_delta": joint_matched_tokens
        - source_matched_tokens,
        "matched_token_reduction_ratio": (
            1.0 - joint_matched_tokens / source_matched_tokens
            if source_matched_tokens
            else 0.0
        ),
        "posthoc_answer_matched_qids": matched_qids,
        "production_route_eligible": False,
        "production_route_ineligibility_reason": (
            "answer-match cohort was selected after observing source answers"
        ),
        "rows": rows,
    }
    eval_args = (
        args.source_eval_dir,
        args.joint_eval_dir,
        args.base_reasoning_score,
        args.assumed_accuracy_score,
    )
    if any(value is not None for value in eval_args):
        if any(value is None for value in eval_args):
            raise ValueError(
                "source/joint eval dirs, base reasoning, and assumed "
                "accuracy must be supplied together"
            )
        if not matched_qids:
            raise ValueError(
                "cannot score an empty posthoc answer-match cohort"
            )
        source_eval = {
            str(item["qid"]): dict(item)
            for item in read_json(
                _resolve(args.source_eval_dir) / "reasoning_scores.json"
            )
        }
        joint_eval = {
            str(item["qid"]): dict(item)
            for item in read_json(
                _resolve(args.joint_eval_dir) / "reasoning_scores.json"
            )
        }
        if set(source_eval) != set(matched_qids):
            raise ValueError("source evaluation qids do not match answer set")
        if set(joint_eval) != set(matched_qids):
            raise ValueError("joint evaluation qids do not match answer set")
        for row in matching_rows:
            qid = str(row["qid"])
            source_score = float(source_eval[qid]["reasoning_score"])
            joint_score = float(joint_eval[qid]["reasoning_score"])
            row["source_reasoning_score"] = source_score
            row["joint_reasoning_score"] = joint_score
            row["equal_accuracy_proxy_delta"] = (
                0.003 * (joint_score - source_score)
                - float(row["token_delta"]) / 250_000.0
            )

        source_manifest = dict(read_json(source_run / "run_manifest.json"))
        base_tokens = int(
            dict(source_manifest.get("token_usage") or {}).get(
                "total_tokens",
                0,
            )
        )
        source_eval_score = sum(
            float(source_eval[qid]["reasoning_score"])
            for qid in matched_qids
        ) / len(matched_qids)
        joint_eval_score = sum(
            float(joint_eval[qid]["reasoning_score"])
            for qid in matched_qids
        ) / len(matched_qids)
        base_reasoning = float(args.base_reasoning_score)
        assumed_accuracy = float(args.assumed_accuracy_score)
        oracle_reasoning = (
            base_reasoning * 100
            - source_eval_score * len(matched_qids)
            + joint_eval_score * len(matched_qids)
        ) / 100
        oracle_tokens = (
            base_tokens - source_matched_tokens + joint_matched_tokens
        )
        base_token_score = token_efficiency_score(base_tokens)
        oracle_token_score = token_efficiency_score(oracle_tokens)
        base_total = (
            assumed_accuracy * 0.5
            + base_reasoning * 0.3
            + base_token_score * 0.2
        )
        oracle_total = (
            assumed_accuracy * 0.5
            + oracle_reasoning * 0.3
            + oracle_token_score * 0.2
        )
        report["posthoc_oracle_projection"] = {
            "score_type": (
                "offline_posthoc_oracle_projection_not_official"
            ),
            "not_a_reproducible_production_score": True,
            "excludes_failed_and_answer_changed_joint_attempts": True,
            "assumed_accuracy_score": assumed_accuracy,
            "base_reasoning_score": base_reasoning,
            "oracle_reasoning_score": oracle_reasoning,
            "base_token_total": base_tokens,
            "oracle_token_total": oracle_tokens,
            "base_token_efficiency_score": base_token_score,
            "oracle_token_efficiency_score": oracle_token_score,
            "base_proxy_total": base_total,
            "oracle_proxy_total": oracle_total,
            "oracle_proxy_total_delta": oracle_total - base_total,
            "all_matched_qids_positive_proxy_delta": all(
                float(row["equal_accuracy_proxy_delta"]) > 0.0
                for row in matching_rows
            ),
        }
    fieldnames = list(
        dict.fromkeys(
            key for row in rows for key in row
        )
    )
    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
