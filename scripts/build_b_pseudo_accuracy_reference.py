#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "artifacts/b_board_actual/candidates/i032_evidence_full_year_dividend"
    / "fin005_ac_single_v1/submit.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts/b_board_actual/references"
    / "pseudo99_from_official98_ins016_bd_v1"
)
ANSWER_COLUMNS = ("answer_1", "answer_2", "answer_3", "answer_4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an unsubmitted pseudo-accuracy reference without creating a submission"
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--set-answer",
        action="append",
        default=["ins_b_016=BD"],
        metavar="QID=ANSWER1|ANSWER2",
    )
    parser.add_argument("--baseline-official-accuracy", type=float, default=98.0)
    parser.add_argument("--predicted-accuracy", type=float, default=99.0)
    parser.add_argument("--label", default="pseudo99_unsubmitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_pseudo_accuracy_reference(
        base_path=args.base.resolve(),
        output_dir=args.output_dir.resolve(),
        overrides=parse_answer_overrides(args.set_answer),
        baseline_official_accuracy=args.baseline_official_accuracy,
        predicted_accuracy=args.predicted_accuracy,
        label=args.label,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_answer_overrides(specs: Sequence[str]) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        qid, separator, answer_text = str(spec).partition("=")
        qid = qid.strip()
        answers = tuple(
            item.strip() for item in answer_text.split("|") if item.strip()
        )
        if not separator or not qid or not answers:
            raise ValueError(f"invalid --set-answer value: {spec!r}")
        if len(answers) > len(ANSWER_COLUMNS):
            raise ValueError(f"{qid}: too many answer parts")
        if qid in overrides:
            raise ValueError(f"duplicate answer override for {qid}")
        overrides[qid] = answers
    return overrides


def build_pseudo_accuracy_reference(
    *,
    base_path: Path,
    output_dir: Path,
    overrides: Mapping[str, Sequence[str]],
    baseline_official_accuracy: float,
    predicted_accuracy: float,
    label: str,
) -> dict[str, object]:
    if not overrides:
        raise ValueError("at least one answer override is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"reference output directory is not empty: {output_dir}")
    columns, rows = _read_submission(base_path)
    missing_columns = sorted({"qid", *ANSWER_COLUMNS} - set(columns))
    if missing_columns:
        raise ValueError(f"{base_path}: missing columns {missing_columns}")
    question_rows = [row for row in rows if row.get("qid") != "summary"]
    qids = [str(row.get("qid", "")).strip() for row in question_rows]
    if not qids or any(not qid for qid in qids) or len(qids) != len(set(qids)):
        raise ValueError(f"{base_path}: invalid or duplicate qids")

    reference_rows = [
        {
            "qid": qid,
            "answer_parts": [
                str(row.get(column, "")).strip()
                for column in ANSWER_COLUMNS
                if str(row.get(column, "")).strip()
            ],
        }
        for qid, row in zip(qids, question_rows)
    ]
    row_by_qid = {str(row["qid"]): row for row in reference_rows}
    missing_qids = sorted(set(overrides) - set(row_by_qid))
    if missing_qids:
        raise ValueError(f"{base_path}: unknown override qids {missing_qids}")

    changes: dict[str, dict[str, list[str]]] = {}
    for qid, raw_answers in overrides.items():
        answers = [str(item).strip() for item in raw_answers if str(item).strip()]
        before = list(row_by_qid[qid]["answer_parts"])
        if before == answers:
            raise ValueError(f"{qid}: override does not change the reference")
        row_by_qid[qid]["answer_parts"] = answers
        changes[qid] = {"before": before, "after": answers}

    output_dir.mkdir(parents=True, exist_ok=False)
    reference_path = output_dir / "reference_answers.json"
    reference_path.write_text(
        json.dumps(reference_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "reference_frozen",
        "label": str(label),
        "submission_eligible": False,
        "officially_submitted": False,
        "baseline_official_accuracy": float(baseline_official_accuracy),
        "predicted_accuracy": float(predicted_accuracy),
        "question_count": len(reference_rows),
        "answer_change_count": len(changes),
        "answer_changes": changes,
        "frozen_answer_count": len(reference_rows) - len(changes),
        "source_submission": str(base_path),
        "source_sha256": _sha256(base_path),
        "reference_answers": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "notes": (
            "Evaluation-only pseudo reference. It must never be included in Qwen "
            "generation prompts or presented as official accuracy."
        ),
    }
    manifest_path = output_dir / "reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_submission(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
