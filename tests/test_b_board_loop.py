from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from afa_agent.b_board.evaluator import ConfidenceEvaluation
from afa_agent.b_board.loop import (
    Direction,
    OpenEndedLoopScheduler,
    append_markdown_log,
    directions_from_low_confidence,
    evaluate_round_gate,
    merge_promoted_answers,
)
from afa_agent.experiment_registry import ExperimentRegistry


def evaluation(
    qid: str,
    score: int,
    *,
    tier: str | None = None,
    reasons: tuple[str, ...] = (),
    hard_failures: tuple[str, ...] = (),
) -> ConfidenceEvaluation:
    resolved_tier = tier or ("blocked" if score < 40 else "low" if score < 60 else "medium" if score < 80 else "high")
    return ConfidenceEvaluation(
        qid=qid,
        dimensions={
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
        confidence_score=score,
        tier=resolved_tier,
        verdict="supported",
        blocking_reasons=(),
        low_confidence_reasons=reasons,
        suggested_improvements=(),
        hard_failures=hard_failures,
    )


def direction(**overrides: object) -> Direction:
    values: dict[str, object] = {
        "direction_id": "retrieval_entity_year",
        "pipeline_stage": "retrieval",
        "root_cause_cluster": "retrieval",
        "hypothesis": "强化实体年份查询",
        "change_vector": {"query": "entity_year", "top_k": 6},
        "domains": ("financial_reports",),
        "target_qids": ("q1",),
    }
    values.update(overrides)
    return Direction(**values)  # type: ignore[arg-type]


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.registry = ExperimentRegistry(self.root / "registry.jsonl")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_duplicate_is_skipped_after_registry_decision(self) -> None:
        existing = direction().candidate()
        self.registry.append({"experiment_id": "old", "status": "rejected", **existing})
        scheduler = OpenEndedLoopScheduler(self.registry, directions=[direction()])

        attempt = scheduler.next_attempt()

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt.action, "skip_duplicate")
        self.assertFalse(attempt.executable)
        self.assertEqual(attempt.history.related_experiment_ids, ("old",))

    def test_three_comparable_attempts_exhaust_direction(self) -> None:
        base = direction().candidate()
        for index in range(3):
            self.registry.append(
                {
                    "experiment_id": f"old-{index}",
                    "status": "rejected",
                    **base,
                    "hypothesis": f"强化实体年份查询 v{index}",
                    "change_vector": {"query": "entity_year", "top_k": 6 + index},
                }
            )
        candidate = direction(
            hypothesis="强化实体年份查询 v4",
            change_vector={"query": "entity_year", "top_k": 10},
            material_delta={"top_k": {"from": 8, "to": 10}},
        )
        scheduler = OpenEndedLoopScheduler(self.registry, directions=[candidate])

        attempt = scheduler.next_attempt()

        self.assertEqual(attempt.action, "attempt_limit")
        self.assertEqual(attempt.history.comparable_attempt_count, 3)
        self.assertEqual(scheduler.direction_status[candidate.direction_id], "exhausted")

    def test_low_confidence_reasons_create_dynamic_directions(self) -> None:
        evaluations = {
            "q1": evaluation("q1", 35, reasons=("引用证据ID不存在",)),
            "q2": evaluation("q2", 52, reasons=("公式单位换算错误",)),
            "q3": evaluation("q3", 85, reasons=("检索召回不足",)),
        }

        created = directions_from_low_confidence(
            evaluations,
            qid_domains={"q1": "regulatory", "q2": "insurance", "q3": "research"},
        )

        self.assertEqual({item.root_cause_cluster for item in created}, {"citation", "calculation"})
        self.assertEqual({qid for item in created for qid in item.target_qids}, {"q1", "q2"})
        scheduler = OpenEndedLoopScheduler(self.registry, directions=[])
        added = scheduler.add_evaluator_directions(evaluations)
        self.assertEqual(len(added), 2)
        self.assertEqual(scheduler.pending_count, 2)

    def test_online_research_is_requested_and_zero_result_wave_can_stop(self) -> None:
        scheduler = OpenEndedLoopScheduler(self.registry, directions=[])
        added = scheduler.add_evaluator_directions(
            {"q1": evaluation("q1", 30, reasons=("未知的新型低置信原因",))}
        )
        base = added[0]
        for index in range(3):
            if index:
                self.assertEqual(scheduler.signal().kind, "local_refinement_required")
                scheduler.enqueue(
                    replace(
                        base,
                        hypothesis=f"未知根因定向优化 v{index + 1}",
                        change_vector={"strategy": f"unknown_v{index + 1}"},
                        material_delta={"revision": index + 1},
                    )
                )
            attempt = scheduler.next_attempt()
            self.assertTrue(attempt.executable)
            scheduler.complete_attempt(attempt, status="rejected")

        signal = scheduler.signal()
        self.assertEqual(signal.kind, "online_research_required")
        self.assertEqual(signal.unresolved_clusters, ("unknown",))

        self.assertEqual(scheduler.record_external_research_wave([]), 0)
        stop = scheduler.signal()
        self.assertEqual(stop.kind, "stop")
        self.assertIn("没有产生新的", stop.reason)


class PromotionAndRoundGateTests(unittest.TestCase):
    def test_per_qid_merge_only_promotes_eligible_candidates(self) -> None:
        incumbent_answers = {"q1": {"answer_parts": ["A"]}, "q2": {"answer_parts": ["B"]}}
        candidate_answers = {"q1": {"answer_parts": ["A"]}, "q2": {"answer_parts": ["C"]}}
        incumbent_evaluations = {"q1": evaluation("q1", 50), "q2": evaluation("q2", 65)}
        candidate_evaluations = {"q1": evaluation("q1", 62), "q2": evaluation("q2", 72)}

        merged = merge_promoted_answers(
            incumbent_answers=incumbent_answers,
            candidate_answers=candidate_answers,
            incumbent_evaluations=incumbent_evaluations,
            candidate_evaluations=candidate_evaluations,
            blind_winners={"q2": "A"},
            candidate_blind_labels={"q2": "B"},
        )

        self.assertEqual(merged.promoted_qids, ("q1",))
        self.assertEqual(merged.answers["q1"], candidate_answers["q1"])
        self.assertEqual(merged.answers["q2"], incumbent_answers["q2"])
        self.assertIn("blind_pair_did_not_prefer_candidate", merged.decisions["q2"]["reasons"])

    def test_round_gate_requires_complete_safe_low_tail_improvement(self) -> None:
        incumbent = {"q1": evaluation("q1", 50), "q2": evaluation("q2", 75)}
        candidate = {"q1": evaluation("q1", 65), "q2": evaluation("q2", 76)}
        result = evaluate_round_gate(
            incumbent_evaluations=incumbent,
            candidate_evaluations=candidate,
            qid_domains={"q1": "insurance", "q2": "insurance"},
            expected_qids={"q1", "q2"},
            sentinels_passed=True,
            tests_passed=True,
            integrity_passed=True,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.metrics["domain_metrics"]["insurance"]["blocked_low_after"], 0)

        invalid = evaluate_round_gate(
            incumbent_evaluations=incumbent,
            candidate_evaluations={"q1": candidate["q1"]},
            qid_domains={"q1": "insurance", "q2": "insurance"},
            expected_qids={"q1", "q2"},
            sentinels_passed=True,
            tests_passed=True,
            integrity_passed=True,
        )
        self.assertFalse(invalid.valid)
        self.assertIn("incomplete_evaluation_coverage", invalid.reasons)

    def test_round_gate_allows_preserved_hard_failure_but_rejects_new_one(self) -> None:
        incumbent = {
            "q1": evaluation("q1", 0, hard_failures=("legacy_failure",)),
            "q2": evaluation("q2", 50),
        }
        improved = {
            "q1": evaluation("q1", 0, hard_failures=("legacy_failure",)),
            "q2": evaluation("q2", 70),
        }
        allowed = evaluate_round_gate(
            incumbent_evaluations=incumbent,
            candidate_evaluations=improved,
            qid_domains={"q1": "insurance", "q2": "insurance"},
            expected_qids={"q1", "q2"},
            sentinels_passed=True,
            tests_passed=True,
            integrity_passed=True,
        )
        self.assertTrue(allowed.valid)
        self.assertEqual(allowed.metrics["new_hard_failures"], {})

        regressed = dict(improved)
        regressed["q2"] = evaluation("q2", 70, hard_failures=("new_failure",))
        rejected = evaluate_round_gate(
            incumbent_evaluations=incumbent,
            candidate_evaluations=regressed,
            qid_domains={"q1": "insurance", "q2": "insurance"},
            expected_qids={"q1", "q2"},
            sentinels_passed=True,
            tests_passed=True,
            integrity_passed=True,
        )
        self.assertFalse(rejected.valid)
        self.assertIn("candidate_introduces_new_hard_failures", rejected.reasons)
        self.assertEqual(
            rejected.metrics["new_hard_failures"], {"q2": ["new_failure"]}
        )

    def test_round_gate_causally_normalizes_unchanged_judge_drift(self) -> None:
        incumbent = {"q1": evaluation("q1", 65), "q2": evaluation("q2", 50)}
        raw_candidate = {
            "q1": evaluation("q1", 35),
            "q2": evaluation("q2", 70),
        }
        result = evaluate_round_gate(
            incumbent_evaluations=incumbent,
            candidate_evaluations=raw_candidate,
            qid_domains={"q1": "insurance", "q2": "insurance"},
            expected_qids={"q1", "q2"},
            sentinels_passed=True,
            tests_passed=True,
            integrity_passed=True,
            changed_qids={"q2"},
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.metrics["unchanged_raw_score_drift"], {"q1": -30})
        self.assertEqual(
            result.metrics["domain_metrics"]["insurance"]["blocked_low_after"], 0
        )


class MarkdownLogTests(unittest.TestCase):
    def test_log_is_append_only_and_redacts_api_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loop.md"
            append_markdown_log(
                path,
                {
                    "experiment_id": "exp-1",
                    "status": "rejected",
                    "api_key": "sk-super-secret",
                    "api_base": "http://private.example/v1",
                    "notes": "LLM_API_BASE=http://private.example/v1 Bearer abcdefghijk",
                },
            )
            first = path.read_text(encoding="utf-8")
            append_markdown_log(path, {"experiment_id": "exp-2", "status": "promoted"})
            final = path.read_text(encoding="utf-8")

            self.assertTrue(final.startswith(first))
            self.assertNotIn("super-secret", final)
            self.assertNotIn("private.example", final)
            self.assertNotIn("abcdefghijk", final)
            self.assertIn("exp-1", final)
            self.assertIn("exp-2", final)
            self.assertNotEqual(final[-2:], "\n\n")


if __name__ == "__main__":
    unittest.main()
