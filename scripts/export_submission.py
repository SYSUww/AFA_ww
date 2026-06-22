#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    answer_csv = run_dir / "outputs" / "submission" / "answer.csv"
    evidence_json = run_dir / "outputs" / "debug" / "evidence.json"
    if not answer_csv.exists():
        answer_csv = run_dir / "answer.csv"
    if not evidence_json.exists():
        evidence_json = run_dir / "evidence.json"
    if not answer_csv.exists() or not evidence_json.exists():
        raise FileNotFoundError("Run directory is missing answer.csv or evidence.json")
    print(answer_csv)
    print(evidence_json)


if __name__ == "__main__":
    main()
