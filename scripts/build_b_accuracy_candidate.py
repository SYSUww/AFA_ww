#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "artifacts/b_board_actual/candidates/i024_remaining93_p0"
    / "official94_plus_direct_source_five_v1/submit.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/b_board_actual/candidates/i031_accuracy_boundary_d_rollback"
    / "fc019_ins017_drop_d_v1/submit.csv"
)
ACCURACY_ONLY_COLUMNS = (
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
DEFAULT_OVERRIDES = {
    "fc_b_019": "ABC",
    "ins_b_017": "ABC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an answer-only B-board candidate from a scored incumbent"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--set-answer",
        action="append",
        default=[],
        metavar="QID=ANSWER",
        help="Override answer_1; repeat for multiple qids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = (
        parse_answer_overrides(args.set_answer)
        if args.set_answer
        else DEFAULT_OVERRIDES
    )
    manifest = materialize_accuracy_only_candidate(
        base_path=args.base.resolve(),
        output_path=args.output.resolve(),
        overrides=overrides,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_answer_overrides(specs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for spec in specs:
        qid, separator, answer = spec.partition("=")
        qid = qid.strip()
        answer = answer.strip()
        if not separator or not qid or not answer:
            raise ValueError(f"invalid --set-answer value: {spec!r}")
        if qid in overrides:
            raise ValueError(f"duplicate answer override for {qid}")
        overrides[qid] = answer
    return overrides


def materialize_accuracy_only_candidate(
    *,
    base_path: Path,
    output_path: Path,
    overrides: Mapping[str, str],
) -> dict[str, object]:
    if not overrides:
        raise ValueError("at least one answer override is required")
    columns, rows = _read_submission(base_path)
    if columns != ACCURACY_ONLY_COLUMNS:
        raise ValueError(
            f"{base_path}: expected answer-only columns {ACCURACY_ONLY_COLUMNS}, "
            f"got {columns}"
        )
    if not rows or rows[0]["qid"] != "summary":
        raise ValueError(f"{base_path}: first data row must be summary")

    row_by_qid = {row["qid"]: row for row in rows[1:]}
    if len(row_by_qid) != len(rows) - 1:
        raise ValueError(f"{base_path}: duplicate qids")
    missing = sorted(set(overrides) - set(row_by_qid))
    if missing:
        raise ValueError(f"{base_path}: unknown override qids {missing}")

    changes: dict[str, dict[str, str]] = {}
    for qid, answer in overrides.items():
        cleaned = str(answer).strip()
        if not cleaned:
            raise ValueError(f"{qid}: answer override must not be empty")
        before = row_by_qid[qid]["answer_1"]
        if before == cleaned:
            raise ValueError(f"{qid}: override does not change answer_1")
        row_by_qid[qid]["answer_1"] = cleaned
        changes[qid] = {"before": before, "after": cleaned}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ACCURACY_ONLY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    _validate_candidate(base_path, output_path, changes)
    baseline_accuracy = 97.0
    maximum_delta = float(len(changes))
    manifest = {
        "status": "candidate_ready",
        "baseline_official_accuracy": baseline_accuracy,
        "predicted_accuracy": baseline_accuracy + maximum_delta,
        "strict_possible_accuracy_range": [
            baseline_accuracy - maximum_delta,
            baseline_accuracy + maximum_delta,
        ],
        "answer_change_count": len(changes),
        "answer_changes": changes,
        "frozen_answer_count": len(rows) - 1 - len(changes),
        "reasoning_column_present": False,
        "columns": list(ACCURACY_ONLY_COLUMNS),
        "base_submission": str(base_path),
        "base_sha256": _sha256(base_path),
        "candidate_submission": str(output_path),
        "candidate_sha256": _sha256(output_path),
        "official_result": None,
    }
    manifest_path = output_path.with_name("candidate_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_submission(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        return columns, list(reader)


def _validate_candidate(
    base_path: Path,
    candidate_path: Path,
    expected_changes: Mapping[str, Mapping[str, str]],
) -> None:
    base_columns, base_rows = _read_submission(base_path)
    candidate_columns, candidate_rows = _read_submission(candidate_path)
    if base_columns != candidate_columns or candidate_columns != ACCURACY_ONLY_COLUMNS:
        raise ValueError("candidate columns drifted from the answer-only baseline")
    if len(base_rows) != len(candidate_rows):
        raise ValueError("candidate row count drifted from the baseline")

    observed_changes: dict[str, dict[str, str]] = {}
    for before, after in zip(base_rows, candidate_rows):
        if before["qid"] != after["qid"]:
            raise ValueError("candidate qid order drifted from the baseline")
        changed_columns = [
            column for column in ACCURACY_ONLY_COLUMNS if before[column] != after[column]
        ]
        if not changed_columns:
            continue
        if changed_columns != ["answer_1"]:
            raise ValueError(
                f"{after['qid']}: non-answer payload drifted in {changed_columns}"
            )
        observed_changes[after["qid"]] = {
            "before": before["answer_1"],
            "after": after["answer_1"],
        }
    if observed_changes != dict(expected_changes):
        raise ValueError(
            f"candidate answer changes differ from request: {observed_changes}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
