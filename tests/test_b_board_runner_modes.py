from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.runner import (
    RUN_MODE_RESEARCH,
    RUN_MODE_SUBMISSION,
    SUBMISSION_REASONING_PROMPT_VERSION,
    SUBMISSION_REASONING_SYSTEM_PROMPT,
    BAnswerArtifact,
    BBoardActualRunner,
)
from afa_agent.config import ModelConfig, RunConfig
from afa_agent.run_metadata import RunFingerprintError, validate_resume_fingerprint
from scripts import run_b_board_actual


def _question() -> BQuestion:
    return BQuestion(
        qid="q1",
        domain="regulatory",
        split="B",
        question="该说法是否正确？",
        options={"A": "正确", "B": "错误"},
        answer_format="tf",
        type="判断题",
        answer_slots=1,
        answer_slot_templates=("A",),
    )


def _artifact() -> BAnswerArtifact:
    return BAnswerArtifact(
        qid="q1",
        domain="regulatory",
        answer_format="tf",
        answer_slot_count=1,
        answer_parts=["A"],
        used_evidence_ids=["u1"],
        evidence_items=[{"unit_id": "u1", "text": "证据"}],
        decision_summary="证据明确支持题干中的监管要求成立，因此选择正确选项A。",
        decision_trace={},
        calculation_trace={},
        token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        locator={},
    )


def _model(model_name: str) -> ModelConfig:
    return ModelConfig(
        api_key="secret",
        api_base="https://example.invalid/v1",
        model_name=model_name,
        temperature=0.0,
    )


class BBoardRunnerModeTests(unittest.TestCase):
    def test_reasoning_prompt_requires_explicit_auditable_structure(self) -> None:
        self.assertEqual(
            SUBMISSION_REASONING_PROMPT_VERSION,
            "b_submission_reasoning_v2_explicit_structure",
        )
        self.assertIn("定位—关键事实—推导—结论", SUBMISSION_REASONING_SYSTEM_PROMPT)
        self.assertIn("与 answer_parts 完全一致", SUBMISSION_REASONING_SYSTEM_PROMPT)

    def test_cli_defaults_to_submission_and_accepts_research(self) -> None:
        with mock.patch.object(sys, "argv", ["run_b_board_actual.py"]):
            self.assertEqual(run_b_board_actual.parse_args().run_mode, RUN_MODE_SUBMISSION)
        with mock.patch.object(
            sys, "argv", ["run_b_board_actual.py", "--run-mode", RUN_MODE_RESEARCH]
        ):
            self.assertEqual(run_b_board_actual.parse_args().run_mode, RUN_MODE_RESEARCH)

    def test_default_submission_mode_rejects_non_allowlisted_model(self) -> None:
        config = RunConfig(model=_model("gpt-5.5"))
        with mock.patch(
            "afa_agent.b_board.runner.build_run_config", return_value=config
        ), self.assertRaisesRegex(ValueError, "requires a Qwen3.5/Qwen3.6 model"):
            BBoardActualRunner(questions=[])

    def test_research_mode_allows_non_allowlisted_model(self) -> None:
        config = RunConfig(model=_model("gpt-5.5"))
        migration = SimpleNamespace(load_domain_payloads=lambda *_args: {})
        attempt = SimpleNamespace(attempt_id="attempt_43")
        with mock.patch(
            "afa_agent.b_board.runner.build_run_config", return_value=config
        ), mock.patch(
            "afa_agent.b_board.runner.OpenAICompatibleClient"
        ), mock.patch(
            "afa_agent.b_board.runner.CalculationExecutor"
        ), mock.patch(
            "afa_agent.b_board.runner._migration_module", return_value=migration
        ), mock.patch(
            "afa_agent.b_board.runner._find_locator_attempt", return_value=attempt
        ):
            runner = BBoardActualRunner(questions=[], run_mode=RUN_MODE_RESEARCH)

        self.assertEqual(runner.run_mode, RUN_MODE_RESEARCH)

    def test_research_run_only_writes_research_csv_and_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "research"
            runner = self._lightweight_runner(RUN_MODE_RESEARCH, "gpt-5.5")

            manifest = runner.run(run_dir=run_dir, workers=1)

            self.assertTrue((run_dir / "research_submit.csv").is_file())
            self.assertFalse((run_dir / "submit.csv").exists())
            self.assertFalse(manifest["submission_eligible"])
            self.assertIsNone(manifest["submission_path"])
            self.assertEqual(
                manifest["research_submission_path"],
                str((run_dir / "research_submit.csv").resolve()),
            )
            self.assertEqual(
                manifest["submission_ineligibility_reasons"],
                [
                    "research_mode_is_not_submission_eligible",
                    "model_is_not_qwen3.5_or_qwen3.6",
                ],
            )

    def test_submission_run_writes_submit_csv_and_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "submission"
            runner = self._lightweight_runner(RUN_MODE_SUBMISSION, "qwen3.5-plus")

            manifest = runner.run(run_dir=run_dir, workers=1)

            self.assertTrue((run_dir / "submit.csv").is_file())
            self.assertFalse((run_dir / "research_submit.csv").exists())
            self.assertTrue(manifest["submission_eligible"])
            self.assertEqual(
                manifest["submission_path"], str((run_dir / "submit.csv").resolve())
            )
            self.assertIsNone(manifest["research_submission_path"])
            self.assertEqual(manifest["submission_ineligibility_reasons"], [])

    def test_run_mode_changes_fingerprint_and_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_root = root / "parsed"
            index_root = root / "index"
            parsed_root.mkdir()
            index_root.mkdir()
            strategy_path = root / "strategy.json"
            strategy_path.write_text('{"version": "test"}', encoding="utf-8")
            runner = self._lightweight_runner(RUN_MODE_RESEARCH, "qwen3.5-plus")
            del runner._build_fingerprint
            runner.parsed_root = parsed_root
            runner.index_root = index_root
            runner.strategy_path = strategy_path
            runner.locator_attempt_id = "attempt_43"
            runner.calculation_top_k = 18
            runner.attempt = SimpleNamespace(to_dict=lambda: {"attempt_id": "attempt_43"})
            git_state = {
                "branch": "codex/test",
                "commit": "a" * 40,
                "dirty": False,
                "dirty_diff_sha256": "b" * 64,
                "untracked_paths": [],
            }
            with mock.patch(
                "afa_agent.run_metadata.collect_git_state", return_value=git_state
            ):
                research_fingerprint = runner._build_fingerprint(runner.questions, workers=1)
                runner.run_mode = RUN_MODE_SUBMISSION
                submission_fingerprint = runner._build_fingerprint(runner.questions, workers=1)

            self.assertEqual(
                research_fingerprint["components"]["arguments"]["run_mode"],
                RUN_MODE_RESEARCH,
            )
            self.assertEqual(
                submission_fingerprint["components"]["arguments"]["run_mode"],
                RUN_MODE_SUBMISSION,
            )
            with self.assertRaisesRegex(RunFingerprintError, "arguments"):
                validate_resume_fingerprint(
                    {"fingerprint": research_fingerprint}, submission_fingerprint
                )

    @staticmethod
    def _lightweight_runner(run_mode: str, model_name: str) -> BBoardActualRunner:
        runner = object.__new__(BBoardActualRunner)
        question = _question()
        runner.questions = [question]
        runner.question_by_qid = {question.qid: question}
        runner.run_mode = run_mode
        runner.locator_attempt_id = "attempt_43"
        runner.config = SimpleNamespace(model=_model(model_name))
        runner.locate = lambda _questions: {question.qid: {"qid": question.qid}}
        runner.answer_one = lambda _question, _locator: _artifact()
        runner._build_fingerprint = lambda _questions, _workers: {
            "schema_version": 1,
            "sha256": "unit-test",
            "components": {"arguments": {"run_mode": run_mode}},
        }
        return runner


if __name__ == "__main__":
    unittest.main()
