#!/usr/bin/env python3
"""Build an answer-free, evaluator-only decisive-evidence oracle.

The source audit contains answers and must never be loaded by the generation
runner.  This one-way builder keeps only evidence fields selected by the
independent evidence judge and emits a strictly validated, answer-free file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "b_board_oracle_decisive_evidence_v1"
FORBIDDEN_KEY_FRAGMENTS = (
    "answer",
    "reference",
    "official_lock",
    "pseudo99",
    "decision_summary",
    "decision_trace",
    "solution_summary",
    "option_assessment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an answer-free decisive-evidence oracle artifact"
    )
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_array(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{label} must be a JSON array of objects")
    return [dict(row) for row in payload]


def _assert_answer_free(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError(f"forbidden oracle key at {path}.{raw_key}")
            _assert_answer_free(nested, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_answer_free(nested, path=f"{path}[{index}]")


def build_oracle_scope(
    audit_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    *,
    audit_file: Path,
    answers_file: Path,
) -> dict[str, Any]:
    answer_by_qid: dict[str, dict[str, Any]] = {}
    for row in answer_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in answer_by_qid:
            raise ValueError(f"invalid or duplicate answer-row qid: {qid!r}")
        answer_by_qid[qid] = row

    rows: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    evidence_count = 0
    evidence_char_count = 0
    for audit in audit_rows:
        qid = str(audit.get("qid", "")).strip()
        if not qid or qid in seen_qids:
            raise ValueError(f"invalid or duplicate audit qid: {qid!r}")
        seen_qids.add(qid)
        source = answer_by_qid.get(qid)
        if source is None:
            raise ValueError(f"audit qid missing from answers source: {qid}")
        evidence_by_id = {
            str(item.get("unit_id")): item
            for item in source.get("evidence_items", [])
            if isinstance(item, Mapping) and item.get("unit_id")
        }
        selected_ids = audit.get("independent_used_evidence_ids")
        if not isinstance(selected_ids, list) or not selected_ids:
            raise ValueError(f"{qid}: independent evidence ids are empty")
        if len(selected_ids) != len(set(map(str, selected_ids))):
            raise ValueError(f"{qid}: duplicate independent evidence ids")
        evidence_items: list[dict[str, Any]] = []
        for raw_unit_id in selected_ids:
            unit_id = str(raw_unit_id)
            source_item = evidence_by_id.get(unit_id)
            if source_item is None:
                raise ValueError(f"{qid}: unresolved evidence id {unit_id}")
            text = str(source_item.get("text", "")).strip()
            doc_id = str(source_item.get("doc_id", "")).strip()
            if not text or not doc_id:
                raise ValueError(f"{qid}: incomplete evidence {unit_id}")
            evidence_items.append(
                {
                    "unit_id": unit_id,
                    "doc_id": doc_id,
                    "title_path": [
                        str(part)
                        for part in source_item.get("title_path", [])
                    ],
                    "text": text,
                    "metadata": {
                        "unit_type": str(
                            (source_item.get("metadata") or {}).get(
                                "unit_type", ""
                            )
                        )
                    },
                }
            )
            evidence_count += 1
            evidence_char_count += len(text)
        rows.append(
            {
                "qid": qid,
                "domain": str(source.get("domain", "")),
                "evidence_items": evidence_items,
            }
        )

    if set(answer_by_qid) != seen_qids:
        missing = sorted(set(answer_by_qid) - seen_qids)
        raise ValueError(f"audit does not cover all source qids: {missing}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_class": "research_only_oracle",
        "production_load_policy": "deny",
        "contains_answers": False,
        "submission_eligible": False,
        "selection_policy": (
            "independent_judge_used_evidence_ids_in_original_order"
        ),
        "source": {
            "audit_file": str(audit_file.resolve()),
            "audit_sha256": _sha256_file(audit_file),
            "evidence_source_file": str(answers_file.resolve()),
            "evidence_source_sha256": _sha256_file(answers_file),
        },
        "question_count": len(rows),
        "evidence_count": evidence_count,
        "evidence_char_count": evidence_char_count,
        "rows": rows,
    }
    _assert_answer_free(payload["rows"])
    return payload


def main() -> None:
    args = parse_args()
    payload = build_oracle_scope(
        _load_array(args.audit_file, label="audit file"),
        _load_array(args.answers_file, label="answers file"),
        audit_file=args.audit_file,
        answers_file=args.answers_file,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "question_count": payload["question_count"],
                "evidence_count": payload["evidence_count"],
                "evidence_char_count": payload["evidence_char_count"],
                "sha256": _sha256_file(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
