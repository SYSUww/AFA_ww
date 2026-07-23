from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from afa_agent.b_board.evaluator import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ConfidenceEvaluation,
    prompt_fingerprint,
)
from afa_agent.b_board.evaluation_run import run_fixed_evaluation
from afa_agent.b_board.io import BQuestion, load_b_questions
from afa_agent.b_board.loop import OpenEndedLoopScheduler, append_markdown_log
from afa_agent.b_board.reasoning_evaluation import (
    PROMPT_VERSION as REASONING_PROMPT_VERSION,
    REASONING_JUDGE_MODEL,
    SCHEMA_VERSION as REASONING_SCHEMA_VERSION,
    reasoning_prompt_fingerprint,
    run_reasoning_evaluation,
)
from afa_agent.b_board.runner import BBoardActualRunner
from afa_agent.config import build_model_config, build_run_config
from afa_agent.experiment_registry import ExperimentRegistry
from afa_agent.io_utils import ensure_dir, read_json, write_json


ROOT = Path(__file__).resolve().parents[3]
STATE_SCHEMA_VERSION = 1
RUNNER_NAME = "b_actual_open_loop"

QuestionLoader = Callable[[Path, Path], Sequence[BQuestion]]
AnswerRunner = Callable[[Sequence[BQuestion], Path, Mapping[str, Any]], Mapping[str, Any]]
EvaluationRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
ReasoningEvaluationRunner = Callable[
    [Path, Sequence[BQuestion], Mapping[str, Any]], Mapping[str, Any]
]


class BBoardLoopStateError(RuntimeError):
    pass


class BBoardLoopOrchestrator:
    """Recoverable control plane for B0, fixed judging and open direction scheduling."""

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        plan_config_path: Path,
        root: Path = ROOT,
        question_loader: QuestionLoader = load_b_questions,
        answer_runner: AnswerRunner | None = None,
        evaluation_runner: EvaluationRunner | None = None,
        reasoning_evaluation_runner: ReasoningEvaluationRunner | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.plan = dict(plan)
        self.plan_config_path = Path(plan_config_path).resolve()
        self.question_loader = question_loader
        self.answer_runner = answer_runner or self._run_answers
        self.evaluation_runner = evaluation_runner or self._run_evaluation
        self.reasoning_evaluation_runner = (
            reasoning_evaluation_runner or self._run_reasoning_evaluation
        )
        self._uses_default_answer_runner = answer_runner is None
        self._uses_default_evaluation_runner = evaluation_runner is None
        self._uses_default_reasoning_evaluation_runner = (
            reasoning_evaluation_runner is None
        )

        execution = dict(self.plan.get("execution") or {})
        if execution.get("runner") != RUNNER_NAME:
            raise ValueError(f"B orchestrator requires execution.runner={RUNNER_NAME!r}")
        self.dataset = dict(self.plan.get("dataset") or {})
        self.baseline = dict(self.plan.get("baseline") or {})
        self.evaluation = dict(self.plan.get("evaluation") or {})
        self.reasoning_evaluation = dict(
            self.plan.get("reasoning_evaluation") or {}
        )
        self.scheduler_config = dict(self.plan.get("scheduler") or {})
        self.history_import = dict(self.plan.get("history_import") or {})

        self.output_root = self._resolve(self.plan.get("output_root", "artifacts/b_board_actual"))
        self.run_dir = self.output_root / str(self.baseline.get("run_id", "B0-actual"))
        self.state_path = self._resolve(
            self.plan.get("loop_state_path", str(self.output_root / "loop_state.json"))
        )
        self.registry_path = self._resolve(
            self.plan.get(
                "experiment_registry_path",
                "experiments/b_board_actual/experiment_registry.jsonl",
            )
        )
        self.log_path = self._resolve(
            self.plan.get("markdown_log_path", "wiki/b_board_actual_loop_log.md")
        )
        self.question_root = self._resolve(
            self.dataset.get("question_root", "upload_b/question_b")
        )
        self.submission_template = self._resolve(
            self.dataset.get("submission_template", "upload_b/submit.csv")
        )
        self.registry = ExperimentRegistry(self.registry_path)
        self.plan_fingerprint = _sha256(self.plan)
        if self.evaluation.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION:
            raise ValueError("B loop evaluation.prompt_version does not match the frozen evaluator")
        if int(self.evaluation.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
            raise ValueError("B loop evaluation.schema_version does not match the frozen evaluator")
        if not str(self.evaluation.get("model_name") or "").strip():
            raise ValueError("B loop evaluation.model_name must be explicit")
        if float(self.evaluation.get("temperature", 0.0)) != 0.0:
            raise ValueError("B loop evaluation.temperature must be 0")
        self._evaluation_output_dir()
        self._validate_reasoning_evaluation_config()
        self._reasoning_evaluation_output_dir()

    def run(self) -> dict[str, Any]:
        ensure_dir(self.output_root)
        self._initialize_registry()
        state = self._load_state()
        questions = list(self.question_loader(self.question_root, self.submission_template))
        expected_question_count = int(self.dataset.get("expected_question_count", 100))
        if len(questions) != expected_question_count:
            raise BBoardLoopStateError(
                f"B dataset question count mismatch: expected {expected_question_count}, got {len(questions)}"
            )
        qid_domains = {item.qid: item.domain for item in questions}

        state.update(
            {
                "status": "running_b0",
                "question_count": len(questions),
                "run_dir": str(self.run_dir),
                "registry_path": str(self.registry_path),
            }
        )
        self._save_state(state)

        run_manifest = self._completed_run_manifest(len(questions))
        if run_manifest is None:
            run_manifest = dict(self.answer_runner(questions, self.run_dir, self.baseline))
        state["b0"] = _manifest_summary(run_manifest)
        if not _run_is_complete(run_manifest, len(questions)):
            state["status"] = "b0_incomplete"
            state["updated_at"] = _now()
            self._log_once(
                state,
                "b0_incomplete",
                {
                    "experiment_id": "B0-actual",
                    "status": "incomplete",
                    "answered_question_count": run_manifest.get("answered_question_count"),
                    "failed_qids": run_manifest.get("failed_qids", []),
                },
            )
            self._save_state(state)
            return self._public_result(state)

        self._log_once(
            state,
            "b0_complete",
            {
                "experiment_id": "B0-actual",
                "status": "complete",
                "question_count": len(questions),
                "token_usage": run_manifest.get("token_usage", {}),
                "artifact_paths": {
                    "run_manifest": str(self.run_dir / "run_manifest.json"),
                    "answers": str(self.run_dir / "answers.json"),
                    "answer_csv": run_manifest.get("submission_path")
                    or run_manifest.get("research_submission_path"),
                },
                "submission_valid": run_manifest.get("submission_valid"),
                "submission_validation_failures": run_manifest.get(
                    "submission_validation_failures", []
                ),
            },
        )

        state["status"] = "running_evaluation"
        self._save_state(state)
        evaluation_manifest = self._completed_evaluation_manifest(len(questions))
        if evaluation_manifest is None:
            evaluation_manifest = dict(self.evaluation_runner(self.run_dir, self.evaluation))
        state["evaluation"] = _manifest_summary(evaluation_manifest)
        if not _evaluation_is_complete(evaluation_manifest, len(questions)):
            state["status"] = "evaluation_invalid"
            state["updated_at"] = _now()
            self._log_once(
                state,
                "evaluation_invalid",
                {
                    "experiment_id": "B0-actual-evaluation",
                    "status": evaluation_manifest.get("status", "invalid"),
                    "evaluated_answer_count": evaluation_manifest.get("evaluated_answer_count"),
                    "failure_count": evaluation_manifest.get("failure_count"),
                    "sentinel_validation": evaluation_manifest.get("sentinel_validation", {}),
                },
            )
            self._save_state(state)
            return self._public_result(state)

        evaluations = self._load_confidence_audit(len(questions))
        confidence_summary = _confidence_summary(evaluations)

        state["status"] = "running_reasoning_evaluation"
        state["confidence"] = confidence_summary
        self._save_state(state)
        reasoning_manifest = self._completed_reasoning_evaluation_manifest(questions)
        if reasoning_manifest is None:
            reasoning_manifest = dict(
                self.reasoning_evaluation_runner(
                    self._submission_path(run_manifest),
                    questions,
                    self.reasoning_evaluation,
                )
            )
            if _reasoning_evaluation_is_complete(reasoning_manifest, len(questions)):
                persisted_manifest = self._completed_reasoning_evaluation_manifest(
                    questions
                )
                if persisted_manifest is None:
                    raise BBoardLoopStateError(
                        "Reasoning evaluator returned complete without recoverable artifacts"
                    )
                reasoning_manifest = persisted_manifest
        state["reasoning_evaluation"] = _manifest_summary(reasoning_manifest)
        if not _reasoning_evaluation_is_complete(reasoning_manifest, len(questions)):
            state["status"] = "reasoning_evaluation_invalid"
            state["updated_at"] = _now()
            self._log_once(
                state,
                "reasoning_evaluation_invalid",
                {
                    "experiment_id": "B0-actual-reasoning-evaluation",
                    "status": reasoning_manifest.get("status", "invalid"),
                    "expected_reasoning_count": reasoning_manifest.get(
                        "expected_reasoning_count"
                    ),
                    "evaluated_reasoning_count": reasoning_manifest.get(
                        "evaluated_reasoning_count"
                    ),
                    "failure_count": reasoning_manifest.get("failure_count"),
                },
            )
            self._save_state(state)
            return self._public_result(state)

        reasoning_aggregate = dict(reasoning_manifest["reasoning_aggregate"])
        reasoning_scorecard = reasoning_manifest.get("scorecard")
        state["reasoning"] = reasoning_aggregate
        state["scorecard"] = reasoning_scorecard
        self._log_once(
            state,
            "reasoning_evaluation_complete",
            {
                "experiment_id": "B0-actual-reasoning-evaluation",
                "status": "complete",
                "reasoning": reasoning_aggregate,
                "scorecard": reasoning_scorecard,
                "artifact_paths": {
                    "reasoning_scores": str(
                        self._reasoning_evaluation_output_dir()
                        / "reasoning_scores.json"
                    ),
                    "reasoning_aggregate": str(
                        self._reasoning_evaluation_output_dir()
                        / "reasoning_aggregate.json"
                    ),
                    "scorecard": (
                        str(self._reasoning_evaluation_output_dir() / "scorecard.json")
                        if reasoning_scorecard is not None
                        else None
                    ),
                    "reasoning_evaluator_manifest": str(
                        self._reasoning_evaluation_output_dir()
                        / "reasoning_evaluator_manifest.json"
                    ),
                },
            },
        )

        max_attempts = int(self.scheduler_config.get("max_comparable_attempts", 3))
        scheduler = OpenEndedLoopScheduler(
            self.registry,
            max_comparable_attempts=max_attempts,
        )
        dynamic_directions = scheduler.add_evaluator_directions(
            evaluations,
            qid_domains=qid_domains,
        )
        registry_summary = self.history_summary()
        state.update(
            {
                "status": "ready_for_experiments",
                "confidence": confidence_summary,
                "registry": registry_summary,
                "scheduler": {
                    "pending_direction_count": scheduler.pending_count,
                    "dynamic_direction_ids": [item.direction_id for item in dynamic_directions],
                    "max_comparable_attempts": max_attempts,
                    "signal": _signal_dict(scheduler.signal()),
                },
                "updated_at": _now(),
            }
        )
        self._log_once(
            state,
            "evaluation_complete",
            {
                "experiment_id": "B0-actual-evaluation",
                "status": "complete",
                "confidence": confidence_summary,
                "reasoning": reasoning_aggregate,
                "scorecard": reasoning_scorecard,
                "dynamic_direction_ids": [item.direction_id for item in dynamic_directions],
                "registry": registry_summary,
                "artifact_paths": {
                    "confidence_audit": str(self._evaluation_output_dir() / "confidence_audit.json"),
                    "suspected_errors": str(self._evaluation_output_dir() / "suspected_errors.json"),
                    "evaluator_manifest": str(self._evaluation_output_dir() / "evaluator_manifest.json"),
                },
            },
        )
        self._save_state(state)
        return self._public_result(state)

    def history(self) -> list[dict[str, Any]]:
        return self.registry.read_all()

    def history_summary(self) -> dict[str, Any]:
        rows = self.history()
        statuses = Counter(str(row.get("promotion_result") or row.get("status") or "unknown") for row in rows)
        directions = Counter(str(row.get("direction_id") or "unknown") for row in rows)
        return {
            "experiment_count": len(rows),
            "statuses": dict(sorted(statuses.items())),
            "direction_attempt_counts": dict(sorted(directions.items())),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "runner": RUNNER_NAME,
                "plan_config_path": str(self.plan_config_path),
                "plan_fingerprint": self.plan_fingerprint,
                "created_at": _now(),
                "logged_events": [],
            }
        state = read_json(self.state_path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise BBoardLoopStateError("loop_state schema version mismatch")
        if state.get("runner") != RUNNER_NAME:
            raise BBoardLoopStateError("loop_state runner mismatch")
        if state.get("plan_fingerprint") != self.plan_fingerprint:
            raise BBoardLoopStateError(
                "loop plan changed; use a new output_root or explicitly migrate loop_state"
            )
        return state

    def _save_state(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = _now()
        write_json(self.state_path, payload)

    def _completed_run_manifest(self, expected_count: int) -> dict[str, Any] | None:
        path = self.run_dir / "run_manifest.json"
        if not path.exists():
            return None
        manifest = read_json(path)
        if self._uses_default_answer_runner and manifest.get("runner") not in {
            "b_actual_v1",
            "b_actual_v2",
            "b_actual_v3",
            "b_actual_v4",
            "b_actual_v5",
            "b_actual_v6",
            "b_actual_v7_question_specific_format",
            "b_actual_v8_question_then_readme_format",
            "b_actual_v9_expanded_readme_semantics",
            "b_actual_v10_reasoning_audit",
            "b_actual_v11_staged_answer_reasoning",
            "b_actual_v12_evidence_bound_semantics",
            "b_actual_v13_period_bound_semantics",
            "b_actual_v14_progressive_calculation_evidence",
            "b_actual_v15_auditable_transport",
            "b_actual_v16_aggregate_intensity_binding",
            "b_actual_v17_calculation_semantic_gate",
            "b_actual_v18_table_row_label_binding",
            "b_actual_v19_raw_amount_ratio_binding",
            "b_actual_v20_derived_rule_arithmetic",
            "b_actual_composite_v1",
        }:
            raise BBoardLoopStateError("Existing B0 manifest runner identity mismatch")
        return manifest if _run_is_complete(manifest, expected_count) else None

    def _completed_evaluation_manifest(self, expected_count: int) -> dict[str, Any] | None:
        path = self._evaluation_output_dir() / "evaluator_manifest.json"
        if not path.exists():
            return None
        manifest = read_json(path)
        if self._uses_default_evaluation_runner:
            expected_identity = self._expected_evaluator_identity()
            if manifest.get("evaluator_identity") != expected_identity:
                raise BBoardLoopStateError(
                    "Frozen evaluator identity changed; use a new evaluation version and re-evaluate B0"
                )
        return manifest if _evaluation_is_complete(manifest, expected_count) else None

    def _completed_reasoning_evaluation_manifest(
        self, questions: Sequence[BQuestion]
    ) -> dict[str, Any] | None:
        output_dir = self._reasoning_evaluation_output_dir()
        path = output_dir / "reasoning_evaluator_manifest.json"
        if not path.exists():
            return None
        manifest = read_json(path)
        if self._uses_default_reasoning_evaluation_runner:
            expected_identity = self._expected_reasoning_evaluator_identity()
            if manifest.get("evaluator_identity") != expected_identity:
                raise BBoardLoopStateError(
                    "Frozen reasoning evaluator identity changed; use a new reasoning evaluation version"
                )
        if not _reasoning_evaluation_is_complete(manifest, len(questions)):
            return None

        scores_path = output_dir / "reasoning_scores.json"
        if not scores_path.exists():
            raise BBoardLoopStateError("Completed reasoning evaluation has no score rows")
        rows = read_json(scores_path)
        if not isinstance(rows, list):
            raise BBoardLoopStateError("Reasoning evaluation score rows must be an array")
        expected_qids = {str(question.qid) for question in questions}
        actual_qids = [
            str(row.get("qid", "")) for row in rows if isinstance(row, Mapping)
        ]
        if (
            len(actual_qids) != len(rows)
            or set(actual_qids) != expected_qids
            or len(set(actual_qids)) != len(actual_qids)
        ):
            raise BBoardLoopStateError(
                "Completed reasoning evaluation qid coverage does not match B0"
            )

        aggregate_path = output_dir / "reasoning_aggregate.json"
        if not aggregate_path.exists() or read_json(aggregate_path) != manifest.get(
            "reasoning_aggregate"
        ):
            raise BBoardLoopStateError(
                "Completed reasoning evaluation aggregate does not match its manifest"
            )
        if manifest.get("scorecard") is not None:
            scorecard_path = output_dir / "scorecard.json"
            if not scorecard_path.exists() or read_json(scorecard_path) != manifest.get(
                "scorecard"
            ):
                raise BBoardLoopStateError(
                    "Completed reasoning evaluation scorecard does not match its manifest"
                )
        return manifest

    def _initialize_registry(self) -> None:
        ensure_dir(self.registry_path.parent)
        if not self.registry_path.exists():
            descriptor = self.registry_path.open("x", encoding="utf-8")
            descriptor.close()
            self.registry_path.chmod(0o600)
        markdown_paths = [
            self._resolve(path)
            for path in self.history_import.get("markdown_paths", [])
            if self._resolve(path).is_file()
        ]
        manifest_paths = [
            self._resolve(path)
            for path in self.history_import.get("manifest_paths", [])
            if self._resolve(path).is_file()
        ]
        self.registry.import_legacy(
            markdown_paths=markdown_paths,
            manifest_paths=manifest_paths,
        )

    def _expected_evaluator_identity(self) -> dict[str, Any]:
        config = build_run_config(self.root)
        if config.model is None:
            raise BBoardLoopStateError("Missing model config for fixed B evaluator")
        model = self._evaluator_model_config(config.model)
        return {
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": prompt_fingerprint(),
            "model_name": model.model_name,
            "temperature": model.temperature,
            "api_base_sha256": hashlib.sha256(
                model.api_base.encode("utf-8")
            ).hexdigest(),
        }

    def _expected_reasoning_evaluator_identity(self) -> dict[str, Any]:
        config = build_run_config(self.root)
        if config.model is None:
            raise BBoardLoopStateError("Missing model config for fixed B reasoning evaluator")
        model = self._reasoning_evaluator_model_config(config.model)
        return {
            "prompt_version": REASONING_PROMPT_VERSION,
            "schema_version": REASONING_SCHEMA_VERSION,
            "prompt_sha256": reasoning_prompt_fingerprint(),
            "model_name": model.model_name,
            "temperature": model.temperature,
            "api_base_sha256": hashlib.sha256(
                model.api_base.encode("utf-8")
            ).hexdigest(),
        }

    def _load_confidence_audit(self, expected_count: int) -> dict[str, ConfidenceEvaluation]:
        path = self._evaluation_output_dir() / "confidence_audit.json"
        if not path.exists():
            raise BBoardLoopStateError(f"Missing fixed evaluator audit: {path}")
        rows = read_json(path)
        mismatched = [
            str(row.get("qid", ""))
            for row in rows
            if row.get("prompt_version") != PROMPT_VERSION
            or int(row.get("schema_version", -1)) != SCHEMA_VERSION
        ]
        if mismatched:
            raise BBoardLoopStateError(
                "Fixed evaluator audit uses a different prompt/schema: "
                + ", ".join(mismatched[:5])
            )
        evaluations = {str(row["qid"]): _evaluation_from_dict(row) for row in rows}
        if len(evaluations) != expected_count:
            raise BBoardLoopStateError(
                f"Fixed evaluator audit coverage mismatch: expected {expected_count}, got {len(evaluations)}"
            )
        return evaluations

    def _run_answers(
        self,
        questions: Sequence[BQuestion],
        run_dir: Path,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        runner = BBoardActualRunner(
            questions=questions,
            parsed_root=self._resolve(
                config.get("parsed_root", "artifacts/preprocessed_loop_candidates/parsed")
            ),
            index_root=self._resolve(
                config.get("index_root", "artifacts/preprocessed_loop_candidates/index")
            ),
            strategy_path=self._resolve(
                config.get(
                    "strategy_path",
                    "configs/autoresearch/evidence_gate_rescue_accuracy_first.json",
                )
            ),
            locator_attempt_id=str(config.get("locator_attempt_id", "attempt_43")),
            calculation_top_k=int(config.get("calculation_top_k", 18)),
            run_mode=str(config.get("run_mode", "submission")),
        )
        return runner.run(
            run_dir=run_dir,
            workers=int(config.get("workers", 4)),
            force=False,
        )

    def _run_evaluation(
        self,
        run_dir: Path,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        run_config = build_run_config(self.root)
        if run_config.model is None:
            raise BBoardLoopStateError("Missing model config for fixed B evaluator")
        model = self._evaluator_model_config(run_config.model)
        result = run_fixed_evaluation(
            run_dir=run_dir,
            questions=self.question_loader(self.question_root, self.submission_template),
            model_config=model,
            workers=int(config.get("workers", 4)),
            output_name=self._evaluation_output_dir().name,
        )
        return result.manifest

    def _run_reasoning_evaluation(
        self,
        submission_path: Path,
        questions: Sequence[BQuestion],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        env_prefix = str(config.get("env_prefix") or "LLM").strip().upper()
        base_model = build_model_config(self.root, env_prefix=env_prefix)
        if base_model is None:
            raise BBoardLoopStateError(
                f"Missing {env_prefix} model config for fixed B reasoning evaluator"
            )
        model = self._reasoning_evaluator_model_config(base_model)
        result = run_reasoning_evaluation(
            submission_path=submission_path,
            questions=questions,
            model_config=model,
            output_dir=self._reasoning_evaluation_output_dir(),
            workers=int(config.get("workers", 4)),
            accuracy_score=self._reasoning_accuracy_score(),
            accuracy_source=str(config.get("accuracy_source") or ""),
        )
        return result.manifest

    def _evaluator_model_config(self, base_model: Any) -> Any:
        model_name = str(self.evaluation.get("model_name") or "").strip()
        if not model_name:
            raise BBoardLoopStateError("B loop evaluation.model_name must be explicit")
        temperature = float(self.evaluation.get("temperature", 0.0))
        if temperature != 0.0:
            raise BBoardLoopStateError("B loop evaluator temperature must be 0")
        return replace(
            base_model,
            model_name=model_name,
            temperature=temperature,
        )

    def _reasoning_evaluator_model_config(self, base_model: Any) -> Any:
        model_name = str(self.reasoning_evaluation.get("model_name") or "").strip()
        if model_name.lower() != REASONING_JUDGE_MODEL:
            raise BBoardLoopStateError(
                f"B loop reasoning evaluator model must be {REASONING_JUDGE_MODEL}"
            )
        temperature = float(self.reasoning_evaluation.get("temperature", 0.0))
        if temperature != 0.0:
            raise BBoardLoopStateError("B loop reasoning evaluator temperature must be 0")
        return replace(
            base_model,
            model_name=REASONING_JUDGE_MODEL,
            temperature=temperature,
        )

    def _evaluation_output_dir(self) -> Path:
        output_name = str(self.evaluation.get("output_name") or "evaluation").strip()
        if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
            raise BBoardLoopStateError("B loop evaluation.output_name must be a directory name")
        return self.run_dir / output_name

    def _reasoning_evaluation_output_dir(self) -> Path:
        output_name = str(
            self.reasoning_evaluation.get("output_name") or "reasoning_evaluation"
        ).strip()
        if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
            raise BBoardLoopStateError(
                "B loop reasoning_evaluation.output_name must be a directory name"
            )
        return self.run_dir / output_name

    def _validate_reasoning_evaluation_config(self) -> None:
        env_prefix = str(
            self.reasoning_evaluation.get("env_prefix") or "LLM"
        ).strip().upper()
        if env_prefix not in {"LLM", "OPENAI"}:
            raise ValueError(
                "B loop reasoning_evaluation.env_prefix must be LLM or OPENAI"
            )
        if (
            self.reasoning_evaluation.get("prompt_version", REASONING_PROMPT_VERSION)
            != REASONING_PROMPT_VERSION
        ):
            raise ValueError(
                "B loop reasoning_evaluation.prompt_version does not match the frozen evaluator"
            )
        if (
            int(
                self.reasoning_evaluation.get(
                    "schema_version", REASONING_SCHEMA_VERSION
                )
            )
            != REASONING_SCHEMA_VERSION
        ):
            raise ValueError(
                "B loop reasoning_evaluation.schema_version does not match the frozen evaluator"
            )
        model_name = str(self.reasoning_evaluation.get("model_name") or "").strip()
        if model_name.lower() != REASONING_JUDGE_MODEL:
            raise ValueError(
                f"B loop reasoning_evaluation.model_name must be {REASONING_JUDGE_MODEL}"
            )
        if float(self.reasoning_evaluation.get("temperature", 0.0)) != 0.0:
            raise ValueError("B loop reasoning_evaluation.temperature must be 0")
        self._reasoning_accuracy_score()

    def _reasoning_accuracy_score(self) -> float | None:
        value = self.reasoning_evaluation.get("accuracy_score")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("B loop reasoning_evaluation.accuracy_score must be in 0..100")
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "B loop reasoning_evaluation.accuracy_score must be in 0..100"
            ) from exc
        if not 0.0 <= score <= 100.0:
            raise ValueError("B loop reasoning_evaluation.accuracy_score must be in 0..100")
        return score

    def _submission_path(self, run_manifest: Mapping[str, Any]) -> Path:
        value = run_manifest.get("research_submission_path") or run_manifest.get(
            "submission_path"
        )
        if not str(value or "").strip():
            raise BBoardLoopStateError(
                "Completed B0 manifest has neither research_submission_path nor submission_path"
            )
        return self._resolve(value)

    def _log_once(
        self,
        state: dict[str, Any],
        event: str,
        payload: Mapping[str, Any],
    ) -> None:
        logged = list(state.get("logged_events", []))
        if event in logged:
            return
        append_markdown_log(self.log_path, {"event": event, **dict(payload)})
        logged.append(event)
        state["logged_events"] = logged

    def _resolve(self, value: Any) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def _public_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "runner": RUNNER_NAME,
            "status": state.get("status"),
            "loop_state_path": str(self.state_path),
            "run_dir": str(self.run_dir),
            "registry_path": str(self.registry_path),
            "markdown_log_path": str(self.log_path),
            "question_count": state.get("question_count"),
            "b0": state.get("b0"),
            "evaluation": state.get("evaluation"),
            "confidence": state.get("confidence"),
            "reasoning_evaluation": state.get("reasoning_evaluation"),
            "reasoning": state.get("reasoning"),
            "scorecard": state.get("scorecard"),
            "registry": state.get("registry"),
            "scheduler": state.get("scheduler"),
        }


def run_b_actual_loop_plan(
    plan: Mapping[str, Any],
    plan_config_path: Path,
    **dependencies: Any,
) -> dict[str, Any]:
    return BBoardLoopOrchestrator(
        plan=plan,
        plan_config_path=plan_config_path,
        **dependencies,
    ).run()


def _run_is_complete(manifest: Mapping[str, Any], expected_count: int) -> bool:
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("expected_question_count", -1)) == expected_count
        and int(manifest.get("answered_question_count", -1)) == expected_count
        and not manifest.get("failed_qids")
    )


def _evaluation_is_complete(manifest: Mapping[str, Any], expected_count: int) -> bool:
    sentinels = manifest.get("sentinel_validation") or {}
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("expected_answer_count", -1)) == expected_count
        and int(manifest.get("evaluated_answer_count", -1)) == expected_count
        and int(manifest.get("failure_count", -1)) == 0
        and sentinels.get("passed") is True
    )


def _reasoning_evaluation_is_complete(
    manifest: Mapping[str, Any], expected_count: int
) -> bool:
    aggregate = manifest.get("reasoning_aggregate") or {}
    failure_count = manifest.get("failure_count")
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("expected_reasoning_count", -1)) == expected_count
        and int(manifest.get("evaluated_reasoning_count", -1)) == expected_count
        and isinstance(failure_count, int)
        and not isinstance(failure_count, bool)
        and failure_count >= 0
        and isinstance(aggregate, Mapping)
        and int(aggregate.get("question_count", -1)) == expected_count
    )


def _evaluation_from_dict(row: Mapping[str, Any]) -> ConfidenceEvaluation:
    return ConfidenceEvaluation(
        qid=str(row["qid"]),
        dimensions={
            str(key): (None if value is None else int(value))
            for key, value in dict(row.get("dimensions", {})).items()
        },
        confidence_score=int(row["confidence_score"]),
        tier=str(row["tier"]),
        verdict=str(row["verdict"]),
        blocking_reasons=tuple(str(item) for item in row.get("blocking_reasons", [])),
        low_confidence_reasons=tuple(
            str(item) for item in row.get("low_confidence_reasons", [])
        ),
        suggested_improvements=tuple(
            str(item) for item in row.get("suggested_improvements", [])
        ),
        hard_failures=tuple(str(item) for item in row.get("hard_failures", [])),
        independent_status=str(row.get("independent_status", "")),
        independent_answer_parts=tuple(
            str(item) for item in row.get("independent_answer_parts", [])
        ),
        independent_used_evidence_ids=tuple(
            str(item) for item in row.get("independent_used_evidence_ids", [])
        ),
        independent_option_assessments={
            str(key): str(value)
            for key, value in dict(row.get("independent_option_assessments", {})).items()
        },
        independent_confidence=(
            None
            if row.get("independent_confidence") is None
            else int(row["independent_confidence"])
        ),
        independent_solution_summary=str(row.get("independent_solution_summary", "")),
        independent_missing_evidence=tuple(
            str(item) for item in row.get("independent_missing_evidence", [])
        ),
        answer_match=(None if row.get("answer_match") is None else bool(row["answer_match"])),
        answer_verdict=str(row.get("answer_verdict", "")),
        error_likelihood=(
            None if row.get("error_likelihood") is None else int(row["error_likelihood"])
        ),
        suspected_error=bool(row.get("suspected_error", False)),
        suspected_error_types=tuple(
            str(item) for item in row.get("suspected_error_types", [])
        ),
        suspected_error_reasons=tuple(
            str(item) for item in row.get("suspected_error_reasons", [])
        ),
        correction_candidate_parts=tuple(
            str(item) for item in row.get("correction_candidate_parts", [])
        ),
        prompt_version=str(row.get("prompt_version", PROMPT_VERSION)),
        schema_version=int(row.get("schema_version", SCHEMA_VERSION)),
    )


def _confidence_summary(evaluations: Mapping[str, ConfidenceEvaluation]) -> dict[str, Any]:
    tiers = Counter(item.tier for item in evaluations.values())
    low_qids = sorted(
        qid for qid, item in evaluations.items() if item.tier in {"blocked", "low"}
    )
    suspected_qids = [
        item.qid
        for item in sorted(
            evaluations.values(),
            key=lambda item: (-(item.error_likelihood or 0), item.qid),
        )
        if item.suspected_error
    ]
    disagreement_qids = sorted(
        qid for qid, item in evaluations.items() if item.answer_match is False
    )
    scores = sorted(item.confidence_score for item in evaluations.values())
    p10_index = max(0, (len(scores) + 9) // 10 - 1) if scores else 0
    return {
        "question_count": len(evaluations),
        "tiers": dict(sorted(tiers.items())),
        "minimum": scores[0] if scores else None,
        "p10": scores[p10_index] if scores else None,
        "low_confidence_qids": low_qids,
        "suspected_error_qids": suspected_qids,
        "answer_disagreement_qids": disagreement_qids,
    }


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "expected_question_count",
        "answered_question_count",
        "failed_qids",
        "expected_answer_count",
        "evaluated_answer_count",
        "failure_count",
        "sentinel_validation",
        "token_usage",
        "aggregate_metrics",
        "submission_path",
        "research_submission_path",
        "submission_valid",
        "submission_validation_failures",
        "expected_reasoning_count",
        "evaluated_reasoning_count",
        "reasoning_aggregate",
        "judge_token_usage",
        "submission_token_total",
        "scorecard",
    }
    return {key: manifest[key] for key in allowed if key in manifest}


def _signal_dict(signal: Any) -> dict[str, Any] | None:
    if signal is None:
        return None
    return {
        "kind": signal.kind,
        "reason": signal.reason,
        "unresolved_clusters": list(signal.unresolved_clusters),
    }


def _sha256(payload: Mapping[str, Any]) -> str:
    safe = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(safe.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
