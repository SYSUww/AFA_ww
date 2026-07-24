#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.reasoning_evaluation import (
    REASONING_JUDGE_MODEL,
    run_reasoning_evaluation,
)
from afa_agent.config import build_model_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score final B-board CSV reasoning with the fixed GPT-5.6 new.md rubric"
    )
    parser.add_argument("--submission", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--env-root",
        type=Path,
        default=ROOT,
        help="Directory whose .env supplies the independent shadow-judge connection",
    )
    parser.add_argument(
        "--qid",
        action="append",
        default=[],
        help="Evaluate only these submission rows; may be repeated",
    )
    parser.add_argument("--accuracy-score", type=float)
    parser.add_argument("--accuracy-source", default="")
    parser.add_argument(
        "--judge-env-prefix",
        choices=("LLM", "OPENAI"),
        default="LLM",
        help="Independent connection namespace for the GPT-5.6 shadow judge",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    base_model = build_model_config(args.env_root, env_prefix=args.judge_env_prefix)
    if base_model is None:
        raise RuntimeError(
            f"Missing complete {args.judge_env_prefix} model config under env root"
        )
    judge_model = replace(
        base_model,
        model_name=REASONING_JUDGE_MODEL,
        temperature=0.0,
    )
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    questions = _select_questions(questions, args.qid)
    result = run_reasoning_evaluation(
        submission_path=ROOT / args.submission,
        questions=questions,
        model_config=judge_model,
        output_dir=ROOT / args.output_dir,
        workers=args.workers,
        accuracy_score=args.accuracy_score,
        accuracy_source=args.accuracy_source,
    )
    print(
        json.dumps(
            {
                "status": result.manifest["status"],
                "output_dir": str((ROOT / args.output_dir).resolve()),
                "reasoning_aggregate": result.aggregate,
                "scorecard": result.scorecard,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _select_questions(questions, qids):
    requested = list(dict.fromkeys(str(qid).strip() for qid in qids if str(qid).strip()))
    if not requested:
        return list(questions)
    by_qid = {item.qid: item for item in questions}
    unknown = sorted(set(requested) - set(by_qid))
    if unknown:
        raise ValueError(f"unknown qids: {unknown}")
    return [by_qid[qid] for qid in requested]


if __name__ == "__main__":
    main()
