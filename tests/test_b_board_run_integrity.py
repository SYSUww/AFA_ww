from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from afa_agent.io_utils import read_json
from afa_agent.models import AnswerResult, TokenUsage
from afa_agent.run_metadata import RunFingerprintError
from scripts import run_b_board_migration_loop as loop


def make_attempt() -> loop.AttemptConfig:
    return loop.AttemptConfig(
        attempt_id="test_attempt",
        round_id="test_round",
        priority="P0",
        direction="integrity",
        variant_name="blind",
        hypothesis="test",
        multi_doc_bonus=True,
    )


def make_question(qid: str, *, doc_ids: list[str]) -> dict[str, object]:
    return {
        "qid": qid,
        "domain": "regulatory",
        "split": "A",
        "question": "测试监管要求是否成立？",
        "options": {"A": "成立", "B": "不成立"},
        "answer_format": "mcq",
        "type": "判断题",
        "doc_ids": doc_ids,
        "answer": "A",
        "gold_answer": "A",
        "metadata": {"ground_truth": {"answer": "A", "doc_ids": doc_ids}},
    }


def make_candidate_rows(
    questions: dict[str, dict[str, object]],
    qids: list[str],
    _payloads: dict[str, object],
    _attempt: loop.AttemptConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for qid in qids:
        row = questions[qid]
        candidate = f"candidate-{qid}"
        rows.append(
            {
                "qid": qid,
                "domain": row["domain"],
                "query_terms": row["question"],
                "candidate_doc_ids": [candidate],
                "scores": [1.0],
                "locator_reason": ["bm25_profile_match"],
                "candidates": [{"doc_id": candidate, "locator_reason": "bm25_profile_match"}],
            }
        )
    return rows


class FakePlugin:
    def __init__(self, *, fail_qids: set[str] | None = None) -> None:
        self.fail_qids = fail_qids or set()
        self.questions = []

    def answer_one(self, question, _parsed_path: Path, _index_path: Path) -> AnswerResult:
        self.questions.append(question)
        if question.qid in self.fail_qids:
            raise RuntimeError(f"synthetic failure: {question.qid}")
        return AnswerResult(
            qid=question.qid,
            domain=question.domain,
            question_type=question.type,
            pred_answer="A",
            option_labels={"A": True, "B": False},
            evidence_items=[{"unit_id": "u1", "text": "测试证据"}],
            reasoning_summary="test",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            debug_meta={"option_debug": [{"option": "A", "label": True, "answer": "A"}]},
        )


class BBoardBlindLocatorTests(unittest.TestCase):
    def test_locator_is_invariant_to_true_doc_ids_value_and_count(self) -> None:
        payloads = {
            "regulatory": {
                "parsed": {
                    "documents": [
                        {"doc_id": "candidate-a", "title": "监管要求", "metadata": {}},
                        {"doc_id": "candidate-b", "title": "其他规定", "metadata": {}},
                    ]
                },
                "index": {
                    "units": [
                        {
                            "unit_id": "u1",
                            "doc_id": "candidate-a",
                            "text": "测试监管要求成立",
                            "title_path": [],
                            "unit_type": "article",
                        },
                        {
                            "unit_id": "u2",
                            "doc_id": "candidate-b",
                            "text": "无关内容",
                            "title_path": [],
                            "unit_type": "article",
                        },
                    ]
                },
            }
        }
        variants = [
            {"q1": make_question("q1", doc_ids=["gold-one"])},
            {"q1": make_question("q1", doc_ids=["other-a", "other-b", "other-c"])},
        ]
        outputs = []
        try:
            for questions in variants:
                blind = loop.build_blind_question_rows(questions, ["q1"])
                self.assertNotIn("doc_ids", blind["q1"])
                self.assertNotIn("metadata", blind["q1"])
                loop.PROFILE_INDEX_CACHE.clear()
                outputs.append(loop.locate_docs(blind, ["q1"], payloads, make_attempt()))
        finally:
            loop.PROFILE_INDEX_CACHE.clear()

        self.assertEqual(outputs[0], outputs[1])
        self.assertNotIn("true_doc_ids_for_eval_only", outputs[0][0])


class BBoardAnswerRunIntegrityTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path]:
        parsed_root = root / "parsed"
        index_root = root / "index"
        (parsed_root / "regulatory").mkdir(parents=True)
        (index_root / "regulatory").mkdir(parents=True)
        (parsed_root / "regulatory" / "parsed.json").write_text('{"documents": []}', encoding="utf-8")
        (index_root / "regulatory" / "index.json").write_text('{"units": []}', encoding="utf-8")
        strategy_path = root / "strategy.json"
        strategy_path.write_text('{"version": "test", "domains": {}}', encoding="utf-8")
        return parsed_root, index_root, strategy_path

    def _run(
        self,
        *,
        root: Path,
        questions: dict[str, dict[str, object]],
        qids: list[str],
        plugin: FakePlugin,
        force_answer: bool = False,
    ) -> dict[str, object]:
        parsed_root, index_root, strategy_path = self._paths(root) if not (root / "parsed").exists() else (
            root / "parsed",
            root / "index",
            root / "strategy.json",
        )
        reference_path = root / "reference.csv"
        with reference_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["qid", "answer"])
            writer.writeheader()
            for qid in qids:
                writer.writerow({"qid": qid, "answer": "A"})
        original_reference_reader = loop.read_reference_answers

        def read_reference_after_prediction_seal(path: Path) -> dict[str, str]:
            sealed_path = root / "output" / "no_docids_clean_subset_run" / "final_answers.json"
            self.assertTrue(sealed_path.exists(), "reference answers were read before predictions were sealed")
            read_json(sealed_path)
            return original_reference_reader(path)

        with mock.patch.object(loop, "locate_docs", side_effect=make_candidate_rows), mock.patch(
            "afa_agent.domains.registry.get_plugin", return_value=plugin
        ), mock.patch.object(
            loop, "read_reference_answers", side_effect=read_reference_after_prediction_seal
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AFA_STRATEGY_CONFIG", None)
            return loop.run_answering_for_best(
                output_dir=root / "output",
                best=make_attempt().to_dict(),
                questions=questions,
                clean_qids=qids,
                payloads={},
                parsed_root=parsed_root,
                index_root=index_root,
                reference_answer_path=reference_path,
                answer_limit=0,
                answer_workers=1,
                answer_strategy_config=strategy_path,
                force_answer=force_answer,
            )

    def test_matching_resume_is_safe_and_input_change_is_rejected_before_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = {"q1": make_question("q1", doc_ids=["gold-doc"])}
            plugin = FakePlugin()
            metrics = self._run(root=root, questions=questions, qids=["q1"], plugin=plugin)
            self.assertTrue(metrics["complete"])
            self.assertTrue(metrics["eligible_for_promotion"])
            self.assertEqual(plugin.questions[0].doc_ids, ["candidate-q1"])
            self.assertNotIn("true_doc_ids_for_eval_only", plugin.questions[0].metadata)
            self.assertNotEqual(plugin.questions[0].doc_ids, questions["q1"]["doc_ids"])

            run_dir = root / "output" / "no_docids_clean_subset_run"
            manifest_path = run_dir / "run_manifest.json"
            manifest = read_json(manifest_path)
            self.assertNotIn("api_base", manifest["fingerprint"]["components"]["model"])
            self.assertIn("api_base_sha256", manifest["fingerprint"]["components"]["model"])

            no_call_plugin = FakePlugin(fail_qids={"q1"})
            resumed = self._run(root=root, questions=questions, qids=["q1"], plugin=no_call_plugin)
            self.assertTrue(resumed["complete"])
            self.assertEqual(no_call_plugin.questions, [])

            manifest_before_mismatch = manifest_path.read_bytes()
            (root / "index" / "regulatory" / "index.json").write_text(
                '{"units": [{"changed": true}]}', encoding="utf-8"
            )
            with self.assertRaisesRegex(RunFingerprintError, "inputs"):
                self._run(root=root, questions=questions, qids=["q1"], plugin=FakePlugin())
            self.assertEqual(manifest_path.read_bytes(), manifest_before_mismatch)

    def test_old_run_without_fingerprint_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._paths(root)
            run_dir = root / "output" / "no_docids_clean_subset_run"
            run_dir.mkdir(parents=True)
            manifest_path = run_dir / "run_manifest.json"
            original = b'{"created_at": "legacy"}'
            manifest_path.write_bytes(original)
            questions = {"q1": make_question("q1", doc_ids=["gold-doc"])}

            with self.assertRaisesRegex(RunFingerprintError, "no fingerprint"):
                self._run(root=root, questions=questions, qids=["q1"], plugin=FakePlugin())
            self.assertEqual(manifest_path.read_bytes(), original)

    def test_failed_qid_stays_in_accuracy_denominator_and_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = {
                "q1": make_question("q1", doc_ids=["gold-one"]),
                "q2": make_question("q2", doc_ids=["gold-two"]),
            }
            metrics = self._run(
                root=root,
                questions=questions,
                qids=["q1", "q2"],
                plugin=FakePlugin(fail_qids={"q2"}),
            )

            self.assertEqual(metrics["expected_question_count"], 2)
            self.assertEqual(metrics["answered_question_count"], 1)
            self.assertEqual(metrics["accuracy_denominator"], 2)
            self.assertEqual(metrics["proxy_accuracy_vs_reference_88"], 0.5)
            self.assertEqual(metrics["failed_qids"], ["q2"])
            self.assertTrue(metrics["incomplete"])
            self.assertFalse(metrics["eligible_for_promotion"])
            self.assertIsNone(metrics["promotion_score"])

            run_dir = root / "output" / "no_docids_clean_subset_run"
            manifest = read_json(run_dir / "run_manifest.json")
            self.assertEqual(manifest["status"], "failed")
            self.assertIsNone(manifest["promotion_score"])
            with (run_dir / "comparison_vs_oracle_docids.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            failed_row = next(row for row in rows if row["qid"] == "q2")
            self.assertEqual(failed_row["prediction_status"], "failed")
            self.assertEqual(failed_row["matches_reference"], "False")


if __name__ == "__main__":
    unittest.main()
