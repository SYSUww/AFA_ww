#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import (
    BAnswer,
    BQuestion,
    load_b_questions,
    validate_b_submission,
    write_b_submission,
)
from afa_agent.b_board.runner import BAnswerArtifact, _artifact_from_dict
from afa_agent.b_board.scoring import is_allowed_submission_model, score_submission
from afa_agent.io_utils import ensure_dir, read_json, write_json


APP_COMPAT_SUBMISSION_COLUMNS = (
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a causally scored 100-question B-board research composite"
    )
    parser.add_argument(
        "--base-run",
        default="artifacts/b_board_score_loop/full_chain_82d4492_research_v1",
    )
    parser.add_argument(
        "--overlay-runs",
        nargs="+",
        default=[
            "artifacts/b_board_score_loop/amount_scale_a1_res_b012",
            "artifacts/b_board_score_loop/reasoning_self_refine_a1_lowtail6",
            "artifacts/b_board_score_loop/reasoning_self_refine_a2_regressions2",
            "artifacts/b_board_score_loop/reasoning_self_refine_a3_conservative_gate",
        ],
    )
    parser.add_argument(
        "--incumbent-submission",
        default="artifacts/b_board_actual/candidates/i024_remaining93_p0/official94_plus_direct_source_five_v1/submit.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/b_board_score_loop/final_composite_i030_reasoning_self_refine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_run = (ROOT / args.base_run).resolve()
    overlay_runs = tuple((ROOT / path).resolve() for path in args.overlay_runs)
    incumbent_submission = (ROOT / args.incumbent_submission).resolve()
    output_dir = (ROOT / args.output_dir).resolve()

    questions = load_b_questions(ROOT / "upload_b/question_b", ROOT / "upload_b/submit.csv")
    artifacts = _load_artifacts(base_run / "answers.json")
    original_scores = _load_scores(base_run / "reasoning_eval/reasoning_scores.json")
    scores = dict(original_scores)
    overlay_qids: dict[str, list[str]] = {}

    for run in overlay_runs:
        replacements = _load_artifacts(run / "answers.json")
        unknown = sorted(set(replacements) - set(artifacts))
        if unknown:
            raise ValueError(f"{run}: unknown overlay qids {unknown}")
        artifacts.update(replacements)
        overlay_qids[str(run.relative_to(ROOT))] = sorted(replacements)

        score_path = run / "reasoning_eval/reasoning_scores.json"
        if score_path.is_file():
            run_scores = _load_scores(score_path)
            scores.update(run_scores)
            if run.name == "amount_scale_a1_res_b012":
                original_scores.update(run_scores)
        composite_path = run / "causal_composite_scorecard.json"
        if composite_path.is_file():
            composite = read_json(composite_path)
            for qid in composite.get("causally_normalized_unchanged_qids", []):
                scores[str(qid)] = original_scores[str(qid)]

    expected_qids = {item.qid for item in questions}
    if set(artifacts) != expected_qids or set(scores) != expected_qids:
        raise ValueError("composite artifact or score coverage does not match all 100 questions")

    ordered_artifacts = [artifacts[item.qid] for item in questions]
    ensure_dir(output_dir)
    write_json(output_dir / "answers.json", [item.to_dict() for item in ordered_artifacts])
    submission_path = output_dir / "research_submit.csv"
    write_b_submission(
        submission_path,
        questions,
        [item.to_submission_answer() for item in ordered_artifacts],
        audit_ready=True,
    )
    validated = validate_b_submission(submission_path, questions, audit_ready=True)
    app_compatible_submission_path = output_dir / "research_submit_compat.csv"
    write_app_compatible_submission(
        app_compatible_submission_path,
        questions,
        validated,
    )
    validate_app_compatible_submission(
        app_compatible_submission_path,
        questions,
        validated,
    )
    incumbent_by_qid = _load_incumbent_answer_parts(incumbent_submission, questions)
    pseudo_mismatches = {
        item.qid: {
            "candidate": list(item.answer_parts),
            "incumbent": list(incumbent_by_qid[item.qid]),
        }
        for item in validated
        if item.answer_parts != incumbent_by_qid[item.qid]
    }
    token_total = sum(int(item.total_tokens or 0) for item in validated)
    scorecard = score_submission(
        accuracy_score=(len(questions) - len(pseudo_mismatches)) / len(questions) * 100.0,
        reasoning_scores=scores.values(),
        token_total=token_total,
    ).to_dict()
    scorecard.update(
        {
            "accuracy_source": "pseudo_match_to_official97_incumbent_not_per_qid_ground_truth",
            "incumbent_official_aggregate_accuracy": 97.0,
            "incumbent_per_qid_truth_available": False,
            "reasoning_score_source": "causal_per_qid_gpt5.6_scores_with_unchanged_text_normalization",
            "judge_tokens_included_in_submission": False,
        }
    )
    write_json(output_dir / "scorecard.json", scorecard)
    manifest: dict[str, Any] = {
        "status": "complete",
        "question_count": len(validated),
        "base_run": str(base_run.relative_to(ROOT)),
        "overlay_runs": [str(path.relative_to(ROOT)) for path in overlay_runs],
        "overlay_qids": overlay_qids,
        "incumbent_submission": str(incumbent_submission.relative_to(ROOT)),
        "pseudo_mismatches": pseudo_mismatches,
        "answer_change_count": len(pseudo_mismatches),
        "audit_ready_csv_validated": True,
        "app_compatible_submission": app_compatible_submission_path.name,
        "app_compatible_answer_columns": list(APP_COMPAT_SUBMISSION_COLUMNS[1:5]),
        "app_compatible_submission_validated": True,
        "token_total": token_total,
        "generator_model": "gpt-5.5",
        "submission_model_allowlisted": is_allowed_submission_model("gpt-5.5"),
        "submission_eligible": False,
        "submission_ineligibility_reasons": [
            "research_composite_is_not_an_official_submission",
            "model_is_not_qwen3.5_or_qwen3.6",
        ],
        "scorecard": scorecard,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(manifest)


def write_app_compatible_submission(
    path: Path,
    questions: list[BQuestion],
    answers: list[BAnswer],
) -> None:
    """Export the same research rows for upload apps using legacy answer headers."""

    answer_by_qid = {item.qid: item for item in answers}
    expected_qids = [item.qid for item in questions]
    if len(answer_by_qid) != len(answers) or set(answer_by_qid) != set(expected_qids):
        raise ValueError("App-compatible export answer qids do not match questions")

    prompt_total = sum(answer_by_qid[qid].prompt_tokens for qid in expected_qids)
    completion_total = sum(answer_by_qid[qid].completion_tokens for qid in expected_qids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(APP_COMPAT_SUBMISSION_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer_1": "",
                "answer_2": "",
                "answer_3": "",
                "answer_4": "",
                "prompt_tokens": prompt_total,
                "completion_tokens": completion_total,
                "total_tokens": prompt_total + completion_total,
                "reasoning": "",
            }
        )
        for qid in expected_qids:
            answer = answer_by_qid[qid]
            row: dict[str, str | int | None] = {
                "qid": qid,
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "total_tokens": answer.total_tokens,
                "reasoning": answer.reasoning,
            }
            for index in range(1, 5):
                row[f"answer_{index}"] = (
                    answer.answer_parts[index - 1]
                    if index <= len(answer.answer_parts)
                    else ""
                )
            writer.writerow(row)


def validate_app_compatible_submission(
    path: Path,
    questions: list[BQuestion],
    expected_answers: list[BAnswer],
) -> None:
    """Prove the compatibility export only renames answer columns."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)
    if columns != APP_COMPAT_SUBMISSION_COLUMNS:
        raise ValueError(
            f"{path}: expected App-compatible columns {APP_COMPAT_SUBMISSION_COLUMNS}, "
            f"got {columns}"
        )

    summary_rows = [row for row in rows if row["qid"] == "summary"]
    question_rows = [row for row in rows if row["qid"] != "summary"]
    if len(summary_rows) != 1:
        raise ValueError(f"{path}: expected exactly one summary row")
    if len(question_rows) != len(questions):
        raise ValueError(f"{path}: expected exactly {len(questions)} answer rows")

    expected_by_qid = {item.qid: item for item in expected_answers}
    if len(expected_by_qid) != len(expected_answers):
        raise ValueError("Expected answers contain duplicate qids")
    for question, row in zip(questions, question_rows):
        if row["qid"] != question.qid:
            raise ValueError(
                f"{path}: expected qid {question.qid!r}, got {row['qid']!r}"
            )
        expected = expected_by_qid[question.qid]
        actual_parts = tuple(
            row[f"answer_{index}"] for index in range(1, question.answer_slots + 1)
        )
        unused_parts = tuple(
            row[f"answer_{index}"] for index in range(question.answer_slots + 1, 5)
        )
        if actual_parts != expected.answer_parts or any(unused_parts):
            raise ValueError(f"{path}: answer fields drifted for {question.qid}")
        actual_tokens = (
            int(row["prompt_tokens"]),
            int(row["completion_tokens"]),
            int(row["total_tokens"]),
        )
        expected_tokens = (
            expected.prompt_tokens,
            expected.completion_tokens,
            expected.total_tokens,
        )
        if actual_tokens != expected_tokens or row["reasoning"] != expected.reasoning:
            raise ValueError(f"{path}: reasoning or token fields drifted for {question.qid}")

    prompt_total = sum(item.prompt_tokens for item in expected_answers)
    completion_total = sum(item.completion_tokens for item in expected_answers)
    expected_summary = (prompt_total, completion_total, prompt_total + completion_total)
    summary = summary_rows[0]
    actual_summary = (
        int(summary["prompt_tokens"]),
        int(summary["completion_tokens"]),
        int(summary["total_tokens"]),
    )
    if actual_summary != expected_summary:
        raise ValueError(
            f"{path}: summary token totals drifted: "
            f"expected {expected_summary}, got {actual_summary}"
        )


def _load_artifacts(path: Path) -> dict[str, BAnswerArtifact]:
    return {
        str(row["qid"]): _artifact_from_dict(row)
        for row in read_json(path)
    }


def _load_scores(path: Path) -> dict[str, float]:
    return {
        str(row["qid"]): float(row["reasoning_score"])
        for row in read_json(path)
    }


def _load_incumbent_answer_parts(
    path: Path,
    questions: list[BQuestion],
) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        prefix = "answer" if "answer1" in columns else "answer_"
        required = {"qid", *(f"{prefix}{index}" for index in range(1, 5))}
        if not required.issubset(columns):
            raise ValueError(f"{path}: unsupported incumbent answer columns")
        rows = [row for row in reader if row["qid"] != "summary"]
    if [row["qid"] for row in rows] != [item.qid for item in questions]:
        raise ValueError(f"{path}: incumbent qid order does not match questions")
    return {
        question.qid: tuple(
            row[f"{prefix}{index}"]
            for index in range(1, question.answer_slots + 1)
        )
        for question, row in zip(questions, rows)
    }


if __name__ == "__main__":
    main()
