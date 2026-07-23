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
from afa_agent.b_board.merge import assemble_answer_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a complete audited B-board answer run from Qwen sources"
    )
    parser.add_argument("--source-run", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(
        ROOT / args.question_root,
        ROOT / args.submission_template,
    )
    manifest = assemble_answer_run(
        questions=questions,
        source_run_dirs=[ROOT / value for value in args.source_run],
        output_dir=ROOT / args.run_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
