#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.autoresearch import run_loop_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-config", default="configs/autoresearch/plan_config.json")
    args = parser.parse_args()

    payload = run_loop_plan((ROOT / args.plan_config).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
