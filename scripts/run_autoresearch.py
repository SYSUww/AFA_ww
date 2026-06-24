#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.autoresearch import DEFAULT_AUTORESEARCH_DIR, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--domains", default="all")
    parser.add_argument("--dataset-slice", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-set", required=True)
    parser.add_argument("--dataset-slices", default="configs/autoresearch/dataset_slices.json")
    parser.add_argument("--output-root", default=str(DEFAULT_AUTORESEARCH_DIR))
    args = parser.parse_args()

    payload = run_experiment(
        stage=args.stage,
        domains=[item.strip() for item in args.domains.split(",") if item.strip()],
        dataset_slice=args.dataset_slice,
        baseline_config_path=(ROOT / args.baseline_config).resolve(),
        candidate_set_path=(ROOT / args.candidate_set).resolve(),
        dataset_slices_path=(ROOT / args.dataset_slices).resolve(),
        output_root=Path(args.output_root).resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
