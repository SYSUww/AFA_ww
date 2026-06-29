#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.regulatory_preprocess import preprocess_regulatory_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess regulatory markdown/html into structured retrieval units.")
    parser.add_argument(
        "--input-root",
        default=str(ROOT / "artifacts" / "extracted" / "regulatory"),
        help="Directory containing regulatory markdown/html files, either flat or in {attachments,html,txt} subdirs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "preprocessed" / "regulatory"),
        help="Directory where documents.json, units.json, and summary.json are written.",
    )
    args = parser.parse_args()

    summary = preprocess_regulatory_corpus(Path(args.input_root), Path(args.output_root))
    print(Path(args.output_root) / "summary.json")
    print(f"documents={summary['document_count']} units={summary['unit_count']}")


if __name__ == "__main__":
    main()
