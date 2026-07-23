#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.b_board.candidate_scorecard import evaluate_candidate_run


DEFAULT_REFERENCE_MANIFEST = (
    ROOT
    / "artifacts/b_board_actual/references"
    / "pseudo99_from_official98_ins016_bd_v1/reference_manifest.json"
)
DEFAULT_LOCKS = ROOT / "configs/b_board/official_answer_locks.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a full Qwen submission run against the frozen pseudo reference"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST
    )
    parser.add_argument("--official-locks", type=Path, default=DEFAULT_LOCKS)
    parser.add_argument(
        "--reasoning-evaluation-dir",
        type=Path,
        help="Completed sealed GPT-5.6 reasoning evaluation for this exact submit.csv",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = evaluate_candidate_run(
        run_dir=args.run_dir,
        reference_manifest_path=args.reference_manifest,
        official_locks_path=args.official_locks,
        reasoning_evaluation_dir=args.reasoning_evaluation_dir,
    )
    output = args.output or args.run_dir / "proxy_scorecard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(scorecard, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
