from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest

from afa_agent.b_board.io import BQuestion


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_script(
    "build_b_oracle_evidence_scope",
    ROOT / "scripts/build_b_oracle_evidence_scope.py",
)
RUNNER = _load_script(
    "run_b_retrieval_llm_baseline_oracle_test",
    ROOT / "scripts/run_b_retrieval_llm_baseline.py",
)


class OracleEvidenceScopeTests(unittest.TestCase):
    def _question(self) -> BQuestion:
        return BQuestion(
            qid="fc_b_001",
            domain="financial_contracts",
            split="B",
            question="计算指标。",
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("1.00",),
        )

    def _payload(self) -> dict:
        return {
            "schema_version": RUNNER.ORACLE_EVIDENCE_SCHEMA_VERSION,
            "artifact_class": "research_only_oracle",
            "production_load_policy": "deny",
            "contains_answers": False,
            "submission_eligible": False,
            "selection_policy": "test",
            "source": {},
            "question_count": 1,
            "evidence_count": 1,
            "evidence_char_count": 4,
            "rows": [
                {
                    "qid": "fc_b_001",
                    "domain": "financial_contracts",
                    "evidence_items": [
                        {
                            "unit_id": "doc::1",
                            "doc_id": "doc",
                            "title_path": ["指标"],
                            "text": "决定证据",
                            "metadata": {"unit_type": "paragraph"},
                        }
                    ],
                }
            ],
        }

    def test_builder_strips_all_answer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            answers = root / "answers.json"
            audit.write_text("[]", encoding="utf-8")
            answers.write_text("[]", encoding="utf-8")
            payload = BUILDER.build_oracle_scope(
                [
                    {
                        "qid": "fc_b_001",
                        "independent_answer_parts": ["secret"],
                        "independent_used_evidence_ids": ["doc::1"],
                    }
                ],
                [
                    {
                        "qid": "fc_b_001",
                        "domain": "financial_contracts",
                        "answer_parts": ["secret"],
                        "decision_summary": "secret",
                        "evidence_items": [
                            {
                                "unit_id": "doc::1",
                                "doc_id": "doc",
                                "title_path": ["指标"],
                                "text": "决定证据",
                                "metadata": {"unit_type": "paragraph"},
                            }
                        ],
                    }
                ],
                audit_file=audit,
                answers_file=answers,
            )
        serialized_rows = json.dumps(payload["rows"], ensure_ascii=False)
        self.assertNotIn("secret", serialized_rows)
        self.assertNotIn("answer", serialized_rows.lower())
        self.assertEqual(payload["evidence_count"], 1)

    def test_loader_rejects_answer_bearing_rows(self) -> None:
        payload = self._payload()
        payload["rows"][0]["answer_parts"] = ["secret"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden oracle key"):
                RUNNER._load_oracle_evidence_scope(
                    path,
                    questions=[self._question()],
                )

    def test_loader_and_retrieval_payload_preserve_only_evidence(self) -> None:
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = RUNNER._load_oracle_evidence_scope(
                path,
                questions=[self._question()],
            )
        self.assertIsNotNone(loaded)
        row = loaded["rows_by_qid"]["fc_b_001"]
        retrieval = RUNNER._oracle_retrieval_payload(self._question(), row)
        self.assertEqual(retrieval["final"]["ranked_ids"], ["doc::1"])
        self.assertEqual(retrieval["final"]["queries"], [])
        self.assertTrue(retrieval["evaluation_only"])
        RUNNER._assert_oracle_rows_answer_free(
            retrieval["final"]["hits"],
            path="$.final.hits",
        )


if __name__ == "__main__":
    unittest.main()
