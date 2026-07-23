from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_b_pseudo_accuracy_reference import (
    ANSWER_COLUMNS,
    build_pseudo_accuracy_reference,
    parse_answer_overrides,
)


class PseudoAccuracyReferenceTests(unittest.TestCase):
    def test_freezes_reference_without_creating_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "official98.csv"
            output_dir = root / "reference"
            self._write_base(base)

            manifest = build_pseudo_accuracy_reference(
                base_path=base,
                output_dir=output_dir,
                overrides={"q2": ("BD",)},
                baseline_official_accuracy=98,
                predicted_accuracy=99,
                label="pseudo99_unsubmitted",
            )
            rows = json.loads(
                (output_dir / "reference_answers.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rows[1], {"qid": "q2", "answer_parts": ["BD"]})
        self.assertEqual(manifest["status"], "reference_frozen")
        self.assertFalse(manifest["submission_eligible"])
        self.assertFalse(manifest["officially_submitted"])
        self.assertEqual(manifest["baseline_official_accuracy"], 98.0)
        self.assertEqual(manifest["predicted_accuracy"], 99.0)
        self.assertEqual(manifest["answer_change_count"], 1)
        self.assertNotIn("submit", Path(str(manifest["reference_answers"])).name)

    def test_rejects_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "official98.csv"
            output_dir = root / "reference"
            output_dir.mkdir()
            (output_dir / "existing").write_text("sealed", encoding="utf-8")
            self._write_base(base)

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                build_pseudo_accuracy_reference(
                    base_path=base,
                    output_dir=output_dir,
                    overrides={"q2": ("BD",)},
                    baseline_official_accuracy=98,
                    predicted_accuracy=99,
                    label="pseudo99_unsubmitted",
                )

    def test_override_parser_supports_multiple_answer_parts(self) -> None:
        self.assertEqual(
            parse_answer_overrides(["q1=A|B", "q2=BCD"]),
            {"q1": ("A", "B"), "q2": ("BCD",)},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_answer_overrides(["q1=A", "q1=B"])

    @staticmethod
    def _write_base(path: Path) -> None:
        columns = ["qid", *ANSWER_COLUMNS, "prompt_tokens", "completion_tokens", "total_tokens"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "qid": "summary",
                        "prompt_tokens": 30,
                        "completion_tokens": 5,
                        "total_tokens": 35,
                    },
                    {
                        "qid": "q1",
                        "answer_1": "A",
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    {
                        "qid": "q2",
                        "answer_1": "ABD",
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
