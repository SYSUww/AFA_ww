from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from afa_agent.autoresearch import run_loop_plan
from afa_agent.b_board.evaluator import PROMPT_VERSION, SCHEMA_VERSION
from afa_agent.b_board.orchestrator import BBoardLoopOrchestrator
from afa_agent.config import ModelConfig
from afa_agent.io_utils import write_json


@dataclass(frozen=True)
class FakeQuestion:
    qid: str
    domain: str


def audit_row(qid: str, score: int, tier: str, reasons: list[str]) -> dict[str, object]:
    return {
        "qid": qid,
        "dimensions": {
            "document_relevance": score,
            "evidence_sufficiency": score,
            "citation_alignment": score,
            "answer_entailment": score,
            "format_compliance": score,
            "internal_consistency": score,
            "overall_confidence": score,
            "alternative_exclusion": score,
            "calculation_reproducibility": None,
        },
        "confidence_score": score,
        "tier": tier,
        "verdict": "supported",
        "blocking_reasons": [],
        "low_confidence_reasons": reasons,
        "suggested_improvements": [],
        "hard_failures": [],
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


class BBoardOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "config").mkdir()
        self.plan_path = self.root / "config" / "b.json"
        self.plan = {
            "version": "test-v1",
            "execution": {"runner": "b_actual_open_loop"},
            "dataset": {
                "question_root": "input/question_b",
                "submission_template": "input/submit.csv",
                "expected_question_count": 2,
            },
            "output_root": "artifacts/b",
            "loop_state_path": "artifacts/b/loop_state.json",
            "experiment_registry_path": "experiments/b/registry.jsonl",
            "markdown_log_path": "wiki/b_loop.md",
            "baseline": {"run_id": "B0-actual", "workers": 1},
            "evaluation": {
                "model_name": "gpt-5.6",
                "temperature": 0.0,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "output_name": "evaluation",
                "workers": 1,
            },
            "scheduler": {"max_comparable_attempts": 3},
        }
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.questions = [FakeQuestion("q1", "regulatory"), FakeQuestion("q2", "insurance")]
        self.answer_calls = 0
        self.evaluation_calls = 0

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def question_loader(self, _question_root: Path, _template: Path):
        return self.questions

    def answer_runner(self, questions, run_dir: Path, _config):
        self.answer_calls += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "complete",
            "expected_question_count": len(questions),
            "answered_question_count": len(questions),
            "failed_qids": [],
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            "submission_path": str(run_dir / "submit.csv"),
        }
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "answers.json", [{"qid": item.qid} for item in questions])
        (run_dir / "submit.csv").write_text("qid\n", encoding="utf-8")
        return manifest

    def evaluation_runner(self, run_dir: Path, _config):
        self.evaluation_calls += 1
        output = run_dir / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        rows = [
            audit_row("q1", 45, "low", ["引用证据不足"]),
            audit_row("q2", 82, "high", []),
        ]
        write_json(output / "confidence_audit.json", rows)
        manifest = {
            "status": "complete",
            "expected_answer_count": 2,
            "evaluated_answer_count": 2,
            "failure_count": 0,
            "sentinel_validation": {"passed": True, "failures": []},
        }
        write_json(output / "evaluator_manifest.json", manifest)
        return manifest

    def build_orchestrator(self) -> BBoardLoopOrchestrator:
        return BBoardLoopOrchestrator(
            plan=self.plan,
            plan_config_path=self.plan_path,
            root=self.root,
            question_loader=self.question_loader,
            answer_runner=self.answer_runner,
            evaluation_runner=self.evaluation_runner,
        )

    def test_runs_b0_and_evaluation_then_persists_scheduler_state(self) -> None:
        result = self.build_orchestrator().run()

        self.assertEqual(result["status"], "ready_for_experiments")
        self.assertEqual(self.answer_calls, 1)
        self.assertEqual(self.evaluation_calls, 1)
        self.assertEqual(result["confidence"]["tiers"], {"high": 1, "low": 1})
        self.assertEqual(len(result["scheduler"]["dynamic_direction_ids"]), 1)
        self.assertGreater(result["scheduler"]["pending_direction_count"], 1)
        self.assertTrue(Path(result["loop_state_path"]).exists())
        self.assertTrue(Path(result["registry_path"]).exists())

        log = (self.root / "wiki" / "b_loop.md").read_text(encoding="utf-8")
        self.assertIn("b0_complete", log)
        self.assertIn("evaluation_complete", log)

    def test_completed_b0_and_evaluation_are_reused_on_resume(self) -> None:
        first = self.build_orchestrator().run()
        second = self.build_orchestrator().run()

        self.assertEqual(first["status"], "ready_for_experiments")
        self.assertEqual(second["status"], "ready_for_experiments")
        self.assertEqual(self.answer_calls, 1)
        self.assertEqual(self.evaluation_calls, 1)
        log = (self.root / "wiki" / "b_loop.md").read_text(encoding="utf-8")
        self.assertEqual(log.count("## B0-actual\n"), 1)
        self.assertEqual(log.count("## B0-actual-evaluation\n"), 1)

    def test_registry_history_is_queryable(self) -> None:
        orchestrator = self.build_orchestrator()
        orchestrator.registry.append(
            {
                "experiment_id": "exp-1",
                "direction_id": "retrieval",
                "pipeline_stage": "retrieval",
                "root_cause_cluster": "retrieval",
                "hypothesis": "test",
                "change_vector": {"top_k": 8},
                "status": "rejected",
            }
        )

        summary = orchestrator.history_summary()

        self.assertEqual(summary["experiment_count"], 1)
        self.assertEqual(summary["statuses"], {"rejected": 1})
        self.assertEqual(summary["direction_attempt_counts"], {"retrieval": 1})

    def test_legacy_history_is_imported_once_before_scheduling(self) -> None:
        history = self.root / "wiki" / "legacy.md"
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text("## attempt_43 locator retrieval\n", encoding="utf-8")
        self.plan["history_import"] = {"markdown_paths": [str(history)]}

        first = self.build_orchestrator()
        first.run()
        second = self.build_orchestrator()
        second.run()

        imported = [
            row
            for row in second.history()
            if row.get("legacy_identifier") == "attempt_43"
        ]
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["status"], "legacy_transferable")

    def test_loop_plan_dispatches_to_b_orchestrator(self) -> None:
        with patch(
            "afa_agent.b_board.orchestrator.run_b_actual_loop_plan",
            return_value={"runner": "b_actual_open_loop", "status": "test"},
        ) as mocked:
            result = run_loop_plan(self.plan_path)

        self.assertEqual(result["status"], "test")
        mocked.assert_called_once()

    def test_evaluator_model_is_overridden_without_changing_generator_config(self) -> None:
        orchestrator = self.build_orchestrator()
        base = ModelConfig(
            api_key="secret",
            api_base="https://example.test/v1",
            model_name="gpt-5.5",
            temperature=0.0,
        )

        evaluator = orchestrator._evaluator_model_config(base)

        self.assertEqual(base.model_name, "gpt-5.5")
        self.assertEqual(evaluator.model_name, "gpt-5.6")
        self.assertEqual(evaluator.api_key, base.api_key)

    def test_default_answer_runner_forwards_explicit_research_mode(self) -> None:
        self.plan["baseline"]["run_mode"] = "research"
        orchestrator = self.build_orchestrator()
        with patch("afa_agent.b_board.orchestrator.BBoardActualRunner") as runner_type:
            runner_type.return_value.run.return_value = {
                "status": "complete",
                "expected_question_count": 2,
                "answered_question_count": 2,
                "failed_qids": [],
            }

            orchestrator._run_answers(self.questions, self.root / "run", self.plan["baseline"])

        self.assertEqual(runner_type.call_args.kwargs["run_mode"], "research")


if __name__ == "__main__":
    unittest.main()
