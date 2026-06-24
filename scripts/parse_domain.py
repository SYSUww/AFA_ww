#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.registry import get_plugin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--strategy-config", default="")
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    if args.strategy_config:
        os.environ["AFA_STRATEGY_CONFIG"] = str(Path(args.strategy_config).resolve())

    manifest_path = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
    output_path = Path(args.output_path) if args.output_path else (ROOT / "artifacts" / "parsed" / args.domain / "parsed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    get_plugin(args.domain).parse(manifest_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
