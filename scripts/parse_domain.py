#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.regulatory import RegulatoryPlugin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    args = parser.parse_args()

    manifest_path = ROOT / "artifacts" / "manifest" / "dataset_manifest.json"
    output_path = ROOT / "artifacts" / "parsed" / args.domain / "parsed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.domain == "regulatory":
        RegulatoryPlugin().parse(manifest_path, output_path)
    else:
        raise ValueError(f"Unsupported domain for parsing: {args.domain}")
    print(output_path)


if __name__ == "__main__":
    main()
