from __future__ import annotations

import hashlib
import json
from collections import Counter
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
from afa_agent.b_board.runner import BBoardActualRunner
from afa_agent.experiment_registry import ExperimentRegistry
from afa_agent.config import build_run_config
from afa_agent.io_utils import ensure_dir, read_json, write_json


ROOT = Path(__file__).resolve().parents[3]
STATE_SCHEMA_VERSION = 1
RUNNER_NAME = "b_actual_open_loop"

QuestionLoader = Callable[[Path, Path], Sequence[BQuestion]]
AnswerRunner = Callable[[Sequence[BQuestion], Path, Mapping[str, Any]], Mapping[str, Any]]
EvaluationRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


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
    ) -> None:
        self.root = Path(root).resolve()
        self.plan = dict(plan)
        self.plan_config_path = Path(plan_config_path).resolve()
        self.question_loader = question_loader
        self.answer_runner = answer_runner or self._run_answers
        self.evaluation_runner = evaluation_runner or self._run_evaluation
        self._uses_default_answer_runner = answer_runner is None
        self._uses_default_evaluation_runner = evaluation_runner is None

        execution = dict(self.plan.get("execution") or {})
        if execution.get("runner") != RUNNER_NAME:
            raise ValueError(f"B orchestrator requires execution.runner={RUNNER_NAME!r}")
        self.dataset = dict(self.plan.get("dataset") or {})
        self.baseline = dict(self.plan.get("baseline") or {})
        self.evaluation = dict(self.plan.get("evaluation") or {})
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
                    "submission": run_manifest.get("submission_path"),
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
        max_attempts = int(self.scheduler_config.get("max_comparable_attempts", 3))
        scheduler = OpenEndedLoopScheduler(
            self.registry,
            max_comparable_attempts=max_attempts,
        )
        dynamic_directions = scheduler.add_evaluator_directions(
            evaluations,
            qid_domains=qid_domains,
        )
        confidence_summary = _confidence_summary(evaluations)
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
                "dynamic_direction_ids": [item.direction_id for item in dynamic_directions],
                "registry": registry_summary,
                "artifact_paths": {
                    "confidence_audit": str(
                        self.run_dir / "evaluation" / "confidence_audit.json"
                    ),
                    "evaluator_manifest": str(
                        self.run_dir / "evaluation" / "evaluator_manifest.json"
                    ),
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
            "b_actual_composite_v1",
        }:
            raise BBoardLoopStateError("Existing B0 manifest runner identity mismatch")
        return manifest if _run_is_complete(manifest, expected_count) else None

    def _completed_evaluation_manifest(self, expected_count: int) -> dict[str, Any] | None:
        path = self.run_dir / "evaluation" / "evaluator_manifest.json"
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
        return {
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "prompt_sha256": prompt_fingerprint(),
            "model_name": config.model.model_name,
            "temperature": config.model.temperature,
            "api_base_sha256": hashlib.sha256(
                config.model.api_base.encode("utf-8")
            ).hexdigest(),
        }

    def _load_confidence_audit(self, expected_count: int) -> dict[str, ConfidenceEvaluation]:
        path = self.run_dir / "evaluation" / "confidence_audit.json"
        if not path.exists():
            raise BBoardLoopStateError(f"Missing fixed evaluator audit: {path}")
        rows = read_json(path)
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
        result = run_fixed_evaluation(
            run_dir=run_dir,
            questions=self.question_loader(self.question_root, self.submission_template),
            model_config=run_config.model,
            workers=int(config.get("workers", 4)),
        )
        return result.manifest

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
        prompt_version=str(row.get("prompt_version", "b_confidence_judge_v1")),
        schema_version=int(row.get("schema_version", 1)),
    )


def _confidence_summary(evaluations: Mapping[str, ConfidenceEvaluation]) -> dict[str, Any]:
    tiers = Counter(item.tier for item in evaluations.values())
    low_qids = sorted(
        qid for qid, item in evaluations.items() if item.tier in {"blocked", "low"}
    )
    scores = sorted(item.confidence_score for item in evaluations.values())
    p10_index = max(0, (len(scores) + 9) // 10 - 1) if scores else 0
    return {
        "question_count": len(evaluations),
        "tiers": dict(sorted(tiers.items())),
        "minimum": scores[0] if scores else None,
        "p10": scores[p10_index] if scores else None,
        "low_confidence_qids": low_qids,
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
        "submission_valid",
        "submission_validation_failures",
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
