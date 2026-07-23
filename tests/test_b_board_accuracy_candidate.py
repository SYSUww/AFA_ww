from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_b_accuracy_candidate import (
    ACCURACY_ONLY_COLUMNS,
    materialize_accuracy_only_candidate,
    parse_answer_overrides,
)


class AccuracyOnlyCandidateTests(unittest.TestCase):
    def test_changes_only_requested_answers_without_reasoning_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            output = root / "candidate" / "submit.csv"
            self._write_base(base)

            manifest = materialize_accuracy_only_candidate(
                base_path=base,
                output_path=output,
                overrides={"q2": "ABC"},
            )
            with output.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = tuple(reader.fieldnames or ())
                rows = list(reader)

        self.assertEqual(columns, ACCURACY_ONLY_COLUMNS)
        self.assertNotIn("reasoning", columns)
        self.assertEqual(rows[1]["answer_1"], "A")
        self.assertEqual(rows[2]["answer_1"], "ABC")
        self.assertEqual(rows[2]["total_tokens"], "23")
        self.assertEqual(manifest["answer_change_count"], 1)
        self.assertEqual(manifest["frozen_answer_count"], 1)
        self.assertFalse(manifest["reasoning_column_present"])

    def test_rejects_non_answer_only_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            output = root / "candidate.csv"
            self._write_base(base, extra_column="reasoning")

            with self.assertRaisesRegex(ValueError, "expected answer-only columns"):
                materialize_accuracy_only_candidate(
                    base_path=base,
                    output_path=output,
                    overrides={"q2": "ABC"},
                )

    def test_parse_answer_overrides_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_answer_overrides(["q1=A", "q1=B"])

    @staticmethod
    def _write_base(path: Path, extra_column: str | None = None) -> None:
        columns = [*ACCURACY_ONLY_COLUMNS]
        if extra_column:
            columns.append(extra_column)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            rows = [
                {
                    "qid": "summary",
                    "answer_1": "",
                    "answer_2": "",
                    "answer_3": "",
                    "answer_4": "",
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                },
                {
                    "qid": "q1",
                    "answer_1": "A",
                    "answer_2": "",
                    "answer_3": "",
                    "answer_4": "",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                {
                    "qid": "q2",
                    "answer_1": "ABCD",
                    "answer_2": "",
                    "answer_3": "",
                    "answer_4": "",
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                },
            ]
            if extra_column:
                for row in rows:
                    row[extra_column] = ""
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
