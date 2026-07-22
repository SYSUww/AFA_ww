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
from afa_agent.b_board.runner import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_PARSED_ROOT,
    DEFAULT_STRATEGY_PATH,
    RUN_MODES,
    RUN_MODE_SUBMISSION,
    BBoardActualRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real 100-question B-board answering chain")
    parser.add_argument("--question-root", default="upload_b/question_b")
    parser.add_argument("--submission-template", default="upload_b/submit.csv")
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT.relative_to(ROOT)))
    parser.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT.relative_to(ROOT)))
    parser.add_argument("--strategy-config", default=str(DEFAULT_STRATEGY_PATH.relative_to(ROOT)))
    parser.add_argument("--locator-attempt", default="attempt_43")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qid-file", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default=RUN_MODE_SUBMISSION,
        help="submission enforces the official model allowlist; research writes only research_submit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_b_questions(ROOT / args.question_root, ROOT / args.submission_template)
    qids = list(args.qid)
    if args.qid_file:
        qids.extend(
            line.strip()
            for line in (ROOT / args.qid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    qids = list(dict.fromkeys(qids)) or None
    if qids is not None:
        known = {item.qid for item in questions}
        unknown = sorted(set(qids) - known)
        if unknown:
            raise ValueError(f"Unknown B qids: {unknown}")
    run_dir = (
        (ROOT / args.run_dir)
        if args.run_dir
        else ROOT
        / "artifacts"
        / "b_board_actual"
        / f"b0_attempt43_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner = BBoardActualRunner(
        questions=questions,
        parsed_root=ROOT / args.parsed_root,
        index_root=ROOT / args.index_root,
        strategy_path=ROOT / args.strategy_config,
        locator_attempt_id=args.locator_attempt,
        run_mode=args.run_mode,
    )
    manifest = runner.run(run_dir=run_dir, qids=qids, workers=max(1, args.workers))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
