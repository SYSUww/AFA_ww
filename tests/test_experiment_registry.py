from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from afa_agent.experiment_registry import (
    DECISION_EXECUTE,
    DECISION_REFINE_EXISTING,
    DECISION_RETRY_AFTER_CONTEXT_CHANGE,
    DECISION_REUSE_PROMOTED,
    DECISION_SKIP_DUPLICATE,
    ExperimentRegistry,
    build_candidate_fingerprint,
    build_candidate_signature,
    candidate_similarity,
)


def make_candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "direction_id": "retrieval_entity_year",
        "pipeline_stage": "retrieval",
        "root_cause_cluster": "wrong_document",
        "hypothesis": "强化实体与年份能够减少错文档",
        "change_vector": {"query_strategy": "entity_year", "top_k": 6},
        "domains": ["financial_reports"],
        "question_types": ["single"],
        "target_qids": ["q2", "q1"],
        "base_commit": "a" * 40,
        "corpus_hash": "b" * 64,
        "code_hash": "c" * 64,
        "config_hash": "d" * 64,
        "generator_fingerprint": {"model": "test-generator"},
        "evaluator_fingerprint": {"model": "fixed-judge-v1"},
    }
    candidate.update(overrides)
    return candidate


class CandidateFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_order_independent(self) -> None:
        first = make_candidate()
        second = make_candidate(
            domains=["financial_reports"],
            target_qids=["q1", "q2"],
            change_vector={"top_k": 6, "query_strategy": "entity_year"},
        )

        self.assertEqual(build_candidate_fingerprint(first), build_candidate_fingerprint(second))
        self.assertEqual(build_candidate_signature(first), build_candidate_fingerprint(first))

    def test_sensitive_configuration_is_neither_stored_nor_fingerprinted(self) -> None:
        clean = make_candidate()
        unsafe = make_candidate(
            api_key="top-secret",
            api_base="http://private.example/v1",
            generator_fingerprint={
                "model": "test-generator",
                "api_key": "nested-secret",
                "api_base": "http://nested.example/v1",
            },
        )

        clean_fingerprint = build_candidate_fingerprint(clean)
        unsafe_fingerprint = build_candidate_fingerprint(unsafe)
        self.assertEqual(clean_fingerprint, unsafe_fingerprint)
        serialized = json.dumps(unsafe_fingerprint)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("private.example", serialized)

    def test_similarity_recognizes_material_refinement(self) -> None:
        baseline = build_candidate_fingerprint(make_candidate())
        refinement = build_candidate_fingerprint(
            make_candidate(
                hypothesis="增加实体年份查询并扩大候选集",
                change_vector={"query_strategy": "entity_year", "top_k": 8},
            )
        )

        self.assertGreaterEqual(candidate_similarity(baseline, refinement), 0.70)

    def test_similarity_does_not_merge_changes_from_different_pipeline_stages(self) -> None:
        retrieval = build_candidate_fingerprint(make_candidate())
        prompt = build_candidate_fingerprint(make_candidate(pipeline_stage="prompt"))

        self.assertLess(candidate_similarity(retrieval, prompt), 0.70)


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.registry = ExperimentRegistry(self.root / "experiment_registry.jsonl")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def append_candidate(self, experiment_id: str, status: str, **overrides: object) -> dict[str, object]:
        return self.registry.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                **make_candidate(**overrides),
            }
        )

    def test_append_is_jsonl_and_never_rewrites_prior_rows(self) -> None:
        first = self.append_candidate(
            "exp-1",
            "rejected",
            api_key="do-not-store",
            api_base="http://private.example/v1",
        )
        original_first_line = self.registry.path.read_text(encoding="utf-8").splitlines()[0]
        self.append_candidate("exp-2", "promoted", target_qids=["q3"])

        lines = self.registry.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], original_first_line)
        self.assertEqual(len(lines), 2)
        self.assertEqual([row["experiment_id"] for row in self.registry.read_all()], ["exp-1", "exp-2"])
        self.assertNotIn("api_key", first)
        self.assertNotIn("api_base", first)
        self.assertNotIn("do-not-store", "\n".join(lines))
        self.assertNotIn("private.example", "\n".join(lines))

    def test_exact_promoted_is_reused_and_exact_rejected_is_skipped(self) -> None:
        self.append_candidate("promoted", "promoted")
        promoted = self.registry.decide(make_candidate())
        self.assertEqual(promoted.decision, DECISION_REUSE_PROMOTED)
        self.assertEqual(promoted.related_experiment_ids, ("promoted",))

        other_registry = ExperimentRegistry(self.root / "rejected.jsonl")
        other_registry.append({"experiment_id": "rejected", "status": "rejected", **make_candidate()})
        rejected = other_registry.decide(make_candidate())
        self.assertEqual(rejected.decision, DECISION_SKIP_DUPLICATE)

    def test_same_semantics_after_context_change_is_retried(self) -> None:
        self.append_candidate("old-context", "rejected")

        decision = self.registry.decide(make_candidate(code_hash="f" * 64))

        self.assertEqual(decision.decision, DECISION_RETRY_AFTER_CONTEXT_CHANGE)
        self.assertEqual(decision.related_experiment_ids, ("old-context",))

    def test_similar_promoted_is_reused_unless_material_delta_is_declared(self) -> None:
        self.append_candidate("prior", "promoted")
        similar = make_candidate(
            hypothesis="实体年份查询扩大候选范围",
            change_vector={"query_strategy": "entity_year", "top_k": 8},
        )

        reused = self.registry.decide(similar)
        self.assertEqual(reused.decision, DECISION_REUSE_PROMOTED)

        refined = self.registry.decide(
            {**similar, "material_delta": {"top_k": {"from": 6, "to": 8}}}
        )
        self.assertEqual(refined.decision, DECISION_REFINE_EXISTING)

    def test_novel_candidate_executes(self) -> None:
        self.append_candidate("retrieval", "promoted")
        novel = make_candidate(
            direction_id="calculation_decimal",
            pipeline_stage="calculation",
            root_cause_cluster="rounding",
            hypothesis="本地Decimal重放消除舍入错误",
            change_vector={"executor": "decimal", "rounding": "half_up"},
            domains=["insurance"],
        )

        self.assertEqual(self.registry.decide(novel).decision, DECISION_EXECUTE)

    def test_same_direction_id_is_history_aware_even_when_word_similarity_is_low(self) -> None:
        self.append_candidate("prior", "rejected")
        candidate = make_candidate(
            hypothesis="完全不同的实现措辞",
            change_vector={"strategy": "named_directional_operands"},
            domains=["research"],
            target_qids=["q99"],
            material_delta={"operand_roles": True},
        )

        decision = self.registry.decide(candidate)

        self.assertEqual(decision.decision, DECISION_REFINE_EXISTING)
        self.assertEqual(decision.related_experiment_ids, ("prior",))
        self.assertEqual(decision.comparable_attempt_count, 1)

    def test_comparable_attempt_count_excludes_technical_failures_and_deduplicates_ids(self) -> None:
        self.append_candidate("attempt-1", "rejected")
        self.append_candidate("attempt-1", "rejected")
        self.append_candidate("attempt-2", "failed", target_qids=["q9"])
        self.append_candidate(
            "attempt-3",
            "promoted",
            hypothesis="增加实体年份查询并扩大候选集",
            change_vector={"query_strategy": "entity_year", "top_k": 8},
        )

        self.assertEqual(self.registry.count_comparable_attempts(make_candidate()), 2)

    def test_identical_technical_failure_can_execute_again(self) -> None:
        self.append_candidate("failed", "failed")

        decision = self.registry.decide(make_candidate())

        self.assertEqual(decision.decision, DECISION_EXECUTE)
        self.assertEqual(decision.comparable_attempt_count, 0)


class LegacyImportTests(unittest.TestCase):
    def test_imports_structural_metadata_without_copying_prose_or_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "history.md"
            markdown.write_text(
                "# 历史\n- attempt_43：BM25检索优化；API_KEY=markdown-secret\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "experiment_id": "legacy-exp",
                        "stage": "prompt",
                        "candidate_ids": ["prompt-a", "prompt-b"],
                        "api_key": "manifest-secret",
                        "api_base": "http://private.example/v1",
                    }
                ),
                encoding="utf-8",
            )
            registry = ExperimentRegistry(root / "registry.jsonl")

            imported = registry.import_legacy(
                markdown_paths=[markdown],
                manifest_paths=[manifest],
            )
            imported_again = registry.import_legacy(
                markdown_paths=[markdown],
                manifest_paths=[manifest],
            )

            self.assertEqual(len(imported), 3)
            self.assertEqual(imported_again, [])
            rows = registry.read_all()
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["legacy_source_kind"] for row in rows}, {"markdown", "manifest"})
            serialized = registry.path.read_text(encoding="utf-8")
            self.assertNotIn("markdown-secret", serialized)
            self.assertNotIn("manifest-secret", serialized)
            self.assertNotIn("private.example", serialized)
            self.assertNotIn("BM25检索优化", serialized)


if __name__ == "__main__":
    unittest.main()
