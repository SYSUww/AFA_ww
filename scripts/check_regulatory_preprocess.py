#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SAMPLES = {
    "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）": {
        "agency": "中国人民银行",
        "doc_no": "中国人民银行令〔2025〕第12号",
        "effective_date": "2026-01-20",
    },
    "csrc_0125": {
        "doc_type": "penalty_decision",
        "title": "中国证券监督管理委员会行政处罚决定书",
        "doc_no": "〔2025〕96号",
    },
    "csrc_0009_att1": {
        "doc_type": "law_text",
        "title": "上市公司信息披露管理办法",
    },
    "csrc_0036_att2": {
        "doc_type": "amendment",
        "title": "中国证监会决定修改的规范性文件",
    },
}

KEY_PHRASES = [
    "受益所有人",
    "客户尽职调查",
    "行政处罚",
    "股东会",
    "证券公司分类评价",
    "25%以上",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate regulatory preprocessing outputs.")
    parser.add_argument("--output-root", default="artifacts/preprocessed/regulatory")
    parser.add_argument("--expected-documents", type=int, default=348)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    docs = _read_json(output_root / "documents.json")
    units = _read_json(output_root / "units.json")
    summary = _read_json(output_root / "summary.json")

    failures: list[str] = []
    if len(docs) != args.expected_documents:
        failures.append(f"document count {len(docs)} != {args.expected_documents}")
    if summary.get("document_count") != len(docs):
        failures.append("summary document_count does not match documents.json")
    if summary.get("unit_count") != len(units):
        failures.append("summary unit_count does not match units.json")
    if not units:
        failures.append("units.json is empty")
    if any(not unit.get("text", "").strip() for unit in units):
        failures.append("empty unit text exists")
    if any(not doc.get("title", "").strip() for doc in docs):
        failures.append("document with empty title exists")

    unit_ids = Counter(unit.get("unit_id", "") for unit in units)
    duplicates = [unit_id for unit_id, count in unit_ids.items() if count > 1]
    if duplicates:
        failures.append(f"duplicate unit_id count: {len(duplicates)}")

    if any("![](" in unit.get("text", "") for unit in units):
        failures.append("image markdown remains in units")
    if any("<!-- Chunk:" in unit.get("text", "") for unit in units):
        failures.append("chunk comments remain in units")
    html_tag_re = re.compile(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s+[^<>]*)?>")
    htmlish = [unit["unit_id"] for unit in units if html_tag_re.search(unit.get("text", ""))]
    if htmlish:
        failures.append(f"html-like tags remain in {len(htmlish)} units")

    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    for doc_id, expected in EXPECTED_SAMPLES.items():
        doc = docs_by_id.get(doc_id)
        if not doc:
            failures.append(f"missing sample document: {doc_id}")
            continue
        for key, value in expected.items():
            if doc.get(key) != value:
                failures.append(f"{doc_id} {key}={doc.get(key)!r}, expected {value!r}")

    for phrase in KEY_PHRASES:
        if not any(_unit_has_phrase(unit, phrase) for unit in units):
            failures.append(f"missing key phrase: {phrase}")

    print(
        json.dumps(
            {
                "documents": len(docs),
                "units": len(units),
                "doc_types": summary.get("doc_types", {}),
                "unit_lengths": summary.get("unit_lengths", {}),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        sys.exit(1)


def _unit_has_phrase(unit: dict[str, Any], phrase: str) -> bool:
    return phrase in unit.get("text", "") or any(phrase in item for item in unit.get("title_path", []))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
