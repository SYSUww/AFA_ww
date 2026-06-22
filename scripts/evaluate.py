#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.io_utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gold", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    answers = read_json(run_dir / "answers.json")
    metrics = {
        "question_count": len(answers),
        "has_gold": bool(args.gold),
        "predicted_answer_formats": {},
    }
    for row in answers:
        qtype = row["question_type"]
        metrics["predicted_answer_formats"][qtype] = metrics["predicted_answer_formats"].get(qtype, 0) + 1

    if args.gold:
        gold = read_json(Path(args.gold))
        gold_map = {item["qid"]: item["answer"] for item in gold}
        correct = 0
        wrong_cases = []
        for row in answers:
            gold_answer = gold_map.get(row["qid"])
            if gold_answer is None:
                continue
            if row["pred_answer"] == gold_answer:
                correct += 1
            else:
                wrong_cases.append(
                    {
                        "qid": row["qid"],
                        "gold_answer": gold_answer,
                        "pred_answer": row["pred_answer"],
                        "error_tag": "mismatch",
                        "debug_meta": row.get("debug_meta", {}),
                    }
                )
        metrics["accuracy"] = correct / len(gold_map) if gold_map else 0
        write_json(run_dir / "wrong_cases.json", wrong_cases)
    else:
        metrics["accuracy"] = None
        write_json(run_dir / "wrong_cases.json", [])

    write_json(run_dir / "metrics.json", metrics)
    print(run_dir / "metrics.json")


if __name__ == "__main__":
    main()
