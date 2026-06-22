#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.registry import get_plugin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()

    parsed_path = ROOT / "artifacts" / "parsed" / args.domain / "parsed.json"
    output_path = ROOT / "artifacts" / "index" / args.domain / "index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    get_plugin(args.domain).build_index(parsed_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
