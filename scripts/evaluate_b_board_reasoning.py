#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.reasoning_evaluation import (
    REASONING_JUDGE_MODEL,
    run_reasoning_evaluation,
)
from afa_agent.config import build_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score final B-board CSV reasoning with the fixed GPT-5.6 new.md rubric"
    )
    parser.add_argument("--submission", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accuracy-score", type=float)
    parser.add_argument("--accuracy-source", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_run_config(ROOT)
    if config.model is None:
        raise RuntimeError("Missing model config in .env")
    judge_model = replace(
        config.model,
        model_name=REASONING_JUDGE_MODEL,
        temperature=0.0,
    )
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    result = run_reasoning_evaluation(
        submission_path=ROOT / args.submission,
        questions=questions,
        model_config=judge_model,
        output_dir=ROOT / args.output_dir,
        workers=max(1, args.workers),
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


if __name__ == "__main__":
    main()
