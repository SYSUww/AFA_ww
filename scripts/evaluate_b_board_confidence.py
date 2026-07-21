#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.evaluation_run import run_fixed_evaluation
from afa_agent.b_board.io import load_b_questions
from afa_agent.config import build_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sealed B-board answers with the fixed judge")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_run_config(ROOT)
    if config.model is None:
        raise RuntimeError("Missing model config in .env")
    result = run_fixed_evaluation(
        run_dir=(ROOT / args.run_dir).resolve(),
        questions=load_b_questions(ROOT / args.question_root, ROOT / args.submission_template),
        model_config=config.model,
        workers=args.workers,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
