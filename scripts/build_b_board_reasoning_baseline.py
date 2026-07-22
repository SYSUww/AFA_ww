#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.io import load_b_questions
from afa_agent.b_board.reasoning_candidate import materialize_existing_reasoning_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the incumbent decision summaries as a research-only B reasoning baseline"
    )
    parser.add_argument("--source-submission", required=True)
    parser.add_argument("--source-answers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    result = materialize_existing_reasoning_candidate(
        source_submission_path=ROOT / args.source_submission,
        source_answers_path=ROOT / args.source_answers,
        questions=questions,
        output_dir=ROOT / args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
