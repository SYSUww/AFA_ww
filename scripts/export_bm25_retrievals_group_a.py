#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from afa_agent.io_utils import ensure_dir, write_json
from export_bm25_retrievals import (
    TRACE_BUILDERS,
    load_questions,
)


def main() -> None:
    output_dir = ROOT / "artifacts" / "retrieval_traces" / "group_a"
    ensure_dir(output_dir)

    all_domains_payload = {}

    for domain, builder in TRACE_BUILDERS.items():
        questions = load_questions(domain, "A")
        index_path = ROOT / "artifacts" / "index" / domain / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index for {domain}: {index_path}")
        from afa_agent.io_utils import read_json

        index_payload = read_json(index_path)
        traces = {}
        for question in questions:
            traces[question.qid] = builder(question, index_payload)

        domain_path = output_dir / f"{domain}_bm25_retrievals.json"
        write_json(domain_path, traces)
        all_domains_payload[domain] = traces

    combined_path = output_dir / "all_5_domains_bm25_retrievals.json"
    write_json(combined_path, all_domains_payload)

    manifest = {
        "split": "A",
        "domains": list(TRACE_BUILDERS.keys()),
        "domain_files": {
            domain: str(output_dir / f"{domain}_bm25_retrievals.json")
            for domain in TRACE_BUILDERS
        },
        "combined_file": str(combined_path),
    }
    write_json(output_dir / "manifest.json", manifest)

    print(output_dir)
    print(combined_path)


if __name__ == "__main__":
    main()
