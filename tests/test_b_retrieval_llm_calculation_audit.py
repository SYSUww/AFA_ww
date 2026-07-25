from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import requests

from afa_agent.b_board.io import BQuestion
from afa_agent.client import LLMResponse
from afa_agent.models import TokenUsage
from afa_agent.b_board.retrieval_llm_calculation import (
    VerifiedCalculationStageError,
    run_verified_calculation,
)
from afa_agent.b_board import retrieval_llm_calculation as calculation_module
from afa_agent.b_board.retrieval_llm_baseline import (
    suspicious_generation_prompt_literals,
)
from scripts import evaluate_b_retrieval_llm_baseline as baseline_evaluator
from scripts import run_b_retrieval_llm_baseline as baseline_runner
from scripts import assemble_b_retrieval_llm_submission as submission_assembler


class CalculationModeContractTests(TestCase):
    @staticmethod
    def choice_question() -> BQuestion:
        return BQuestion(
            qid="choice",
            domain="insurance",
            split="b",
            question="下列说法正确的是？",
            options={"A": "甲", "B": "乙", "C": "丙"},
            answer_format="multi",
            type="多选题",
            answer_slots=1,
            answer_slot_templates=("AB",),
        )

    def test_evaluator_reconstructs_frozen_answer_reasoning_payload(self) -> None:
        payload = baseline_evaluator._effective_submitted_payload(
            raw_artifact={},
            question=self.choice_question(),
            calls=[
                {
                    "purpose": "initial_answer",
                    "content": json.dumps(
                        {"answer_parts": ["AC"], "reasoning": "缺少最终结论"}
                    ),
                },
                {
                    "purpose": "reasoning_only_retry_from_frozen_answer",
                    "frozen_answer_parts": ["AC"],
                    "content": json.dumps(
                        {
                            "reasoning": (
                                "逐项核验材料原文后，甲和丙均有明确事实支持，"
                                "乙与适用范围不符。结论：AC"
                            ),
                        }
                    ),
                },
            ],
        )
        self.assertEqual(
            payload,
            {
                "answer_parts": ["AC"],
                "reasoning": (
                    "逐项核验材料原文后，甲和丙均有明确事实支持，"
                    "乙与适用范围不符。结论：AC"
                ),
                "decision_trace": {
                    "answer_stage": "frozen_from_initial_qwen_response",
                    "reasoning_stage": "reasoning_only_retry",
                    "postprocessing_mode": "none",
                    "reasoning_assembled_from_model_fields": False,
                    "answer_modified": False,
                    "reasoning_modified": False,
                    "semantic_correction": False,
                },
            },
        )

    def test_submission_assembler_reconstructs_reasoning_only_final_call(
        self,
    ) -> None:
        payload = submission_assembler._reconstruct_final_payload(
            question=self.choice_question(),
            run_dir=Path("/unused"),
            calls=[
                {
                    "purpose": "initial_answer",
                    "content": json.dumps(
                        {
                            "answer_parts": ["AC"],
                            "reasoning": "材料核验后支持甲和丙。结论：AB",
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "purpose": "reasoning_only_retry_from_frozen_answer",
                    "frozen_answer_parts": ["AC"],
                    "content": json.dumps(
                        {
                            "reasoning": (
                                "材料逐项核验后，甲和丙有明确依据，乙不满足条件。"
                                "结论：AC"
                            )
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )

        self.assertEqual(payload["answer_parts"], ["AC"])
        self.assertTrue(payload["reasoning"].endswith("结论：AC"))

    def test_choice_separator_equivalence_persists_as_unmodified_and_replays(
        self,
    ) -> None:
        reasoning = (
            "逐项核验材料后，甲和丙有明确依据，乙不满足条件。"
            "结论：A、C"
        )
        call = {
            "call_index": 1,
            "purpose": "initial_answer",
            "content": json.dumps(
                {"answer_parts": ["AC"], "reasoning": reasoning},
                ensure_ascii=False,
            ),
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        reconstructed = submission_assembler._reconstruct_final_payload(
            question=self.choice_question(),
            run_dir=Path("/unused"),
            calls=[call],
        )

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for name in ("retrieval", "raw_calls"):
                (run_dir / name).mkdir()
            (run_dir / "answers.json").write_text("[]", encoding="utf-8")
            (run_dir / "failures.json").write_text("[]", encoding="utf-8")
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "fingerprint": "fingerprint",
                        "run_instance_id": "run-instance",
                        "config": {},
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "qid": "choice",
                "status": "answered",
                "retrieval": {},
                "evidence_alias_map": [],
                "calls": [call],
                "answer_parts": reconstructed["answer_parts"],
                "reasoning": reconstructed["reasoning"],
                "decision_trace": reconstructed["decision_trace"],
                "token_usage": call["token_usage"],
            }
            with patch.object(baseline_runner, "_write_manifest"):
                baseline_runner._persist_result(
                    run_dir,
                    result,
                    question_order=["choice"],
                    fingerprint="fingerprint",
                    public_config={},
                )
            raw = json.loads(
                (run_dir / "raw_calls" / "choice.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(reconstructed["answer_parts"], ["AC"])
        self.assertEqual(reconstructed["reasoning"], reasoning)
        self.assertEqual(
            reconstructed["decision_trace"]["postprocessing_mode"],
            "multi_choice_conclusion_separator_equivalence",
        )
        self.assertEqual(
            raw["postprocessing"],
            {
                "answer_modified": False,
                "reasoning_modified": False,
                "csv_escaping_only": True,
            },
        )

    def test_reasoning_without_marker_replays_as_unmodified_model_fields(
        self,
    ) -> None:
        reasoning = (
            "逐项核验材料后，甲和丙有明确依据，乙不满足条件，"
            "因此最终答案为A、C。"
        )
        call = {
            "purpose": "initial_answer",
            "content": json.dumps(
                {"answer_parts": ["AC"], "reasoning": reasoning},
                ensure_ascii=False,
            ),
        }

        reconstructed = submission_assembler._reconstruct_final_payload(
            question=self.choice_question(),
            run_dir=Path("/unused"),
            calls=[call],
        )

        self.assertEqual(reconstructed["answer_parts"], ["AC"])
        self.assertEqual(reconstructed["reasoning"], reasoning)
        self.assertEqual(
            reconstructed["decision_trace"]["postprocessing_mode"],
            "reasoning_without_explicit_conclusion",
        )
        self.assertEqual(
            baseline_runner._postprocessing_record(reconstructed),
            {
                "answer_modified": False,
                "reasoning_modified": False,
                "csv_escaping_only": True,
            },
        )

    def test_evaluator_keeps_regular_joint_payload_unchanged(self) -> None:
        question = BQuestion(
            qid="choice",
            domain="insurance",
            split="b",
            question="下列说法正确的是？",
            options={"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
            answer_format="multi",
            type="多选题",
            answer_slots=1,
            answer_slot_templates=("AB",),
        )
        expected = {
            "answer_parts": ["BD"],
            "reasoning": (
                "逐项核验材料原文后，乙和丁均有明确事实支持，"
                "甲和丙不满足题设条件。结论：BD"
            ),
        }

        payload = baseline_evaluator._effective_submitted_payload(
            raw_artifact={},
            question=question,
            calls=[
                {
                    "purpose": "initial_answer",
                    "content": json.dumps(expected),
                }
            ],
        )

        self.assertEqual(payload, expected)

    def test_evaluator_reconstructs_verified_payload_from_frozen_checkpoint(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "frozen_answers" / "verified.json"
            checkpoint.parent.mkdir()
            checkpoint.write_text(
                json.dumps(
                    {"answer_parts": ["6.00"]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            payload = baseline_evaluator._effective_submitted_payload(
                run_dir=run_dir,
                qid="verified",
                raw_artifact={},
                question=self.choice_question(),
                calls=[
                    {
                        "purpose": "verified_calculation_reasoning",
                        "content": json.dumps(
                            {
                                "reasoning": (
                                    "收入10元减成本4元得到6元，按要求保留两位。"
                                    "最终答案为6.00。"
                                )
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            )

        self.assertEqual(payload["answer_parts"], ["6.00"])
        self.assertTrue(payload["reasoning"].endswith("最终答案为6.00。"))

    def test_assembler_accepts_real_verified_reasoning_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "frozen_answers" / "verified.json"
            checkpoint.parent.mkdir()
            checkpoint.write_text(
                json.dumps({"answer_parts": ["6.00"]}),
                encoding="utf-8",
            )
            payload = submission_assembler._reconstruct_final_payload(
                question=SimpleNamespace(qid="verified"),
                run_dir=run_dir,
                calls=[
                    {
                        "purpose": "verified_calculation_reasoning",
                        "content": json.dumps(
                            {
                                "reasoning": (
                                    "收入减去成本得到差额，按题意保留两位小数。"
                                    "最终答案为6.00。"
                                )
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            )

        self.assertEqual(payload["answer_parts"], ["6.00"])
        self.assertTrue(payload["reasoning"].endswith("最终答案为6.00。"))

    def test_assembler_rejects_tampered_public_config_fingerprint(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            config = {
                "created_at": "2026-07-25T08:00:00+08:00",
                "retrieval": {
                    "evidence_quota_strategy": "primary_guard",
                    "research_only_strategy": False,
                },
            }
            baseline_runner._write_json(
                run_dir / "run_config.json",
                {
                    "fingerprint": "stale",
                    "run_instance_id": "run-instance",
                    "config": config,
                },
            )
            baseline_runner._write_json(
                run_dir / "run_manifest.json",
                {"fingerprint": "stale"},
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                submission_assembler._audit_source_run_configuration(
                    run_dir
                )
            fingerprint = (
                baseline_runner.public_run_config_fingerprint(config)
            )
            baseline_runner._write_json(
                run_dir / "run_config.json",
                {
                    "fingerprint": fingerprint,
                    "run_instance_id": "run-instance",
                    "config": config,
                },
            )
            baseline_runner._write_json(
                run_dir / "run_manifest.json",
                {"fingerprint": fingerprint},
            )
            with self.assertRaisesRegex(
                ValueError, "generation contract"
            ):
                submission_assembler._audit_source_run_configuration(
                    run_dir
                )

    def test_duplicate_qids_and_concurrent_run_lock_are_rejected(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_b_retrieval_llm_baseline.py",
                "--run-dir",
                "/tmp/duplicate-qid-run",
                "--qid",
                "q1",
                "--qid",
                "q1",
            ],
        ):
            args = baseline_runner.parse_args()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            baseline_runner._validate_args(args)

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            with baseline_runner._exclusive_run_lock(run_dir):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with baseline_runner._exclusive_run_lock(run_dir):
                        pass

    def test_raw_call_ledger_is_append_only(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.json"
            first = {
                "qid": "q1",
                "evidence_alias_map": [],
                "calls": [{"call_index": 1, "content": "first"}],
            }
            baseline_runner._write_raw_ledger_append_only(path, first)
            original = path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only append"):
                baseline_runner._write_raw_ledger_append_only(
                    path,
                    {
                        "qid": "q1",
                        "evidence_alias_map": [],
                        "calls": [
                            {"call_index": 1, "content": "changed"}
                        ],
                    },
                )

            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_finalized_raw_ledger_metadata_is_immutable(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.json"
            payload = {
                "qid": "q1",
                "evidence_alias_map": [],
                "calls": [{"call_index": 1, "content": "answer"}],
                "final_call_index": 1,
                "submitted_answer_parts_sha256": "a",
                "ledger_state": "finalized_answer",
            }
            baseline_runner._write_raw_ledger_append_only(path, payload)
            changed = {**payload, "submitted_answer_parts_sha256": "b"}

            with self.assertRaisesRegex(ValueError, "immutable"):
                baseline_runner._write_raw_ledger_append_only(path, changed)

    def test_verified_raw_checkpoint_cannot_replace_observed_call(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            evidence = VerifiedCalculationAuditTests.evidence()
            first = {
                "call_index": 1,
                "purpose": "calculation_plan",
                "content": "first",
                "token_usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
                "raw_response": {
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    }
                },
            }
            calculation_module._write_raw_calls_checkpoint(
                run_dir, "q1", [first], evidence
            )

            with self.assertRaisesRegex(ValueError, "only append"):
                calculation_module._write_raw_calls_checkpoint(
                    run_dir,
                    "q1",
                    [{**first, "content": "changed"}],
                    evidence,
                )

    def test_raw_checkpoint_cannot_be_reused_across_run_instances(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            for run_dir, fingerprint, instance in (
                (source, "fingerprint-a", "instance-a"),
                (target, "fingerprint-b", "instance-b"),
            ):
                (run_dir / "raw_calls").mkdir(parents=True)
                baseline_runner._write_json(
                    run_dir / "run_config.json",
                    {
                        "fingerprint": fingerprint,
                        "run_instance_id": instance,
                        "config": {},
                    },
                )
            usage = {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            }
            copied = {
                "qid": "q1",
                "run_fingerprint": "fingerprint-a",
                "run_instance_id": "instance-a",
                "evidence_alias_map": [],
                "calls": [
                    {
                        "call_index": 1,
                        "content": "{}",
                        "token_usage": usage,
                        "raw_response": {"usage": usage},
                    }
                ],
            }
            baseline_runner._write_json(
                target / "raw_calls" / "q1.json",
                copied,
            )

            with self.assertRaisesRegex(ValueError, "run instance"):
                baseline_runner._load_raw_call_checkpoint(
                    target,
                    "q1",
                    [],
                )

    def test_direct_answer_freezes_and_unmatched_call_intent_never_resends(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for name in ("raw_calls", "call_intents", "frozen_answers"):
                (run_dir / name).mkdir()
            baseline_runner._write_json(
                run_dir / "run_config.json",
                {
                    "fingerprint": "fingerprint",
                    "run_instance_id": "instance",
                    "config": {},
                },
            )
            question = self.choice_question()
            aliases: list[dict] = []
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
            content = json.dumps(
                {
                    "reasoning": (
                        "材料支持第一项和第三项，但末尾结论误写为其他选项。"
                        "结论：AB"
                    ),
                    "answer_parts": ["AC"],
                },
                ensure_ascii=False,
            )
            call = {
                "call_index": 1,
                "purpose": "initial_answer",
                "content": content,
                "token_usage": usage,
                "raw_response": {"usage": usage},
                "transport_attempt_count": 1,
                "transport_rejections": [],
            }
            baseline_runner._record_generation_call_intent(
                run_dir,
                qid=question.qid,
                call_index=1,
                purpose="initial_answer",
                evidence_alias_map=aliases,
            )
            baseline_runner._checkpoint_raw_calls(
                run_dir,
                question.qid,
                [call],
                aliases,
            )
            baseline_runner._freeze_direct_answer(
                run_dir,
                question=question,
                evidence_alias_map=aliases,
                answer_parts=["AC"],
                calls=[call],
            )

            loaded_calls = baseline_runner._load_raw_call_checkpoint(
                run_dir,
                question.qid,
                aliases,
            )
            frozen = baseline_runner._load_direct_frozen_answer(
                run_dir,
                question=question,
                evidence_alias_map=aliases,
                calls=loaded_calls,
            )
            self.assertEqual(frozen, ["AC"])

            baseline_runner._record_generation_call_intent(
                run_dir,
                qid=question.qid,
                call_index=2,
                purpose="reasoning_only_retry_from_frozen_answer",
                evidence_alias_map=aliases,
            )
            with self.assertRaisesRegex(RuntimeError, "unobservable"):
                baseline_runner._load_raw_call_checkpoint(
                    run_dir,
                    question.qid,
                    aliases,
                )

    def test_calculation_mode_defaults_to_direct_and_accepts_verified(self) -> None:
        with patch(
            "sys.argv",
            ["run_b_retrieval_llm_baseline.py", "--run-dir", "/tmp/direct-run"],
        ):
            direct_args = baseline_runner.parse_args()

        with patch(
            "sys.argv",
            [
                "run_b_retrieval_llm_baseline.py",
                "--run-dir",
                "/tmp/verified-run",
                "--calculation-mode",
                "verified",
            ],
        ):
            verified_args = baseline_runner.parse_args()

        self.assertEqual(direct_args.calculation_mode, "direct")
        self.assertEqual(verified_args.calculation_mode, "verified")
        self.assertIsInstance(direct_args.run_dir, Path)

    def test_verified_mode_only_routes_calculation_questions(self) -> None:
        self.assertTrue(
            baseline_runner.is_verified_calculation_path(
                "verified", "calculation"
            )
        )
        self.assertFalse(
            baseline_runner.is_verified_calculation_path(
                "verified", "multi"
            )
        )
        self.assertFalse(
            baseline_runner.is_verified_calculation_path(
                "direct", "calculation"
            )
        )

    def test_prompt_literal_audit_covers_inline_and_short_subjects(self) -> None:
        for source, expected in (
            ('SYSTEM_PROMPT = "美的集团应选A"', "美的集团"),
            (
                'def build_messages(): return '
                '[{"role":"system","content":"西部证券满足条件"}]',
                "西部证券",
            ),
            ('SYSTEM_PROMPT = "宁德时代答案为A"', "宁德时代"),
            (
                'SYSTEM_PROMPT = "宁德时代正确答案为A"',
                "宁德时代正确",
            ),
            (
                'SYSTEM_PROMPT = "宁德时代冻结答案为D"',
                "宁德时代冻结",
            ),
            ('SYSTEM_PROMPT = "长安银行不满足条件"', "长安银行"),
        ):
            with self.subTest(expected=expected):
                self.assertIn(
                    expected,
                    suspicious_generation_prompt_literals(ast.parse(source)),
                )

    def test_real_generation_source_closures_have_no_prompt_literal_risk(
        self,
    ) -> None:
        for mode in ("direct", "verified"):
            findings = {
                path.relative_to(baseline_runner.ROOT).as_posix(): (
                    suspicious_generation_prompt_literals(
                        ast.parse(path.read_text(encoding="utf-8"))
                    )
                )
                for path in baseline_runner._generation_source_files(mode)
            }
            self.assertEqual(
                {
                    path: values
                    for path, values in findings.items()
                    if values
                },
                {},
            )

    def test_verified_mode_rejects_reasoning_canonical_contract(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_b_retrieval_llm_baseline.py",
                "--run-dir",
                "/tmp/invalid-contract",
                "--calculation-mode",
                "verified",
                "--output-contract",
                "reasoning-canonical",
            ],
        ):
            args = baseline_runner.parse_args()

        with self.assertRaisesRegex(ValueError, "requires the joint"):
            baseline_runner._validate_args(args)

    def test_research_only_evidence_strategy_requires_explicit_opt_in(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_b_retrieval_llm_baseline.py",
                "--run-dir",
                "/tmp/research-only",
                "--document-candidate-strategy",
                "anchor_first",
                "--evidence-quota-strategy",
                "metric_slot_coverage",
            ],
        ):
            args = baseline_runner.parse_args()
        with self.assertRaisesRegex(ValueError, "research-only"):
            baseline_runner._validate_args(args)

        args.allow_research_only_strategy = True
        baseline_runner._validate_args(args)

    def test_fresh_direct_import_loads_no_solver_module(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys;"
                    "import scripts.run_b_retrieval_llm_baseline as runner;"
                    "runner._assert_no_prohibited_generation_modules_loaded();"
                    "print(','.join(sorted(n for n in sys.modules "
                    "if '.solver' in n)))"
                ),
            ],
            cwd=baseline_runner.ROOT,
            env={
                **dict(__import__("os").environ),
                "PYTHONPATH": "src:.",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")

    def test_verified_mode_fingerprint_covers_calculation_dependencies(self) -> None:
        source_files = {
            path.relative_to(baseline_runner.ROOT).as_posix()
            for path in baseline_runner._generation_source_files("verified")
        }

        self.assertTrue(
            {
                "scripts/run_b_retrieval_llm_baseline.py",
                "src/afa_agent/client.py",
                "src/afa_agent/config.py",
                "src/afa_agent/b_board/io.py",
                "src/afa_agent/domains/generic_retriever.py",
                "src/afa_agent/domains/regulatory/retriever.py",
                "src/afa_agent/bm25.py",
                "src/afa_agent/text_utils.py",
                "src/afa_agent/b_board/retrieval_llm_calculation.py",
                "src/afa_agent/b_board/calculation.py",
                "src/afa_agent/b_board/calculation_schema.py",
                "src/afa_agent/b_board/reasoning_schema.py",
                "src/afa_agent/b_board/runner.py",
                "src/afa_agent/b_board/io.py",
            }.issubset(source_files)
        )

    def test_only_safe_reasoning_failure_is_pending_for_resume(self) -> None:
        failures = [
            {
                "qid": "reasoning_contract",
                "error_stage": "reasoning",
                "error_code": "reasoning_contract_error",
                "retry_route": "resume_reasoning_from_frozen_checkpoint",
                "unobservable_usage_risk": False,
            },
            {
                "qid": "read_timeout",
                "error_stage": "reasoning",
                "retry_route": "do_not_retry_unobservable_generation",
                "unobservable_usage_risk": True,
            },
            {
                "qid": "missing_checkpoint",
                "error_stage": "reasoning",
                "error_code": "reasoning_contract_error",
                "retry_route": "resume_reasoning_from_frozen_checkpoint",
                "unobservable_usage_risk": False,
            },
        ]
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = (
                run_dir / "frozen_answers" / "reasoning_contract.json"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")

            finished = baseline_runner._finished_qids(
                answers=[{"qid": "answered"}],
                failures=failures,
                run_dir=run_dir,
                calculation_mode="verified",
            )

        self.assertEqual(
            finished,
            {"answered", "read_timeout", "missing_checkpoint"},
        )

    def test_answered_resume_replaces_prior_failure_row(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "retrieval").mkdir()
            (run_dir / "raw_calls").mkdir()
            (run_dir / "frozen_answers").mkdir()
            (run_dir / "answers.json").write_text("[]", encoding="utf-8")
            (run_dir / "failures.json").write_text(
                json.dumps([{"qid": "calc_audit_qid", "status": "failed"}]),
                encoding="utf-8",
            )

            with patch.object(baseline_runner, "_write_manifest"):
                baseline_runner._persist_result(
                    run_dir,
                    {
                        "qid": "calc_audit_qid",
                        "status": "answered",
                        "retrieval": {},
                        "calls": [],
                        "answer_parts": ["6.00"],
                        "reasoning": "最终答案为6.00。",
                    },
                    question_order=["calc_audit_qid"],
                    fingerprint="fingerprint",
                    public_config={},
                )

            answers = json.loads(
                (run_dir / "answers.json").read_text(encoding="utf-8")
            )
            failures_after = json.loads(
                (run_dir / "failures.json").read_text(encoding="utf-8")
            )

        self.assertEqual([row["qid"] for row in answers], ["calc_audit_qid"])
        self.assertEqual(failures_after, [])

    def test_manifest_does_not_count_mandatory_reasoning_stage_as_retry(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            rows = [
                {
                    "qid": "verified",
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                {
                    "qid": "direct",
                    "token_usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 4,
                        "total_tokens": 24,
                    },
                },
            ]
            baseline_runner._write_json(run_dir / "answers.json", rows)
            baseline_runner._write_json(run_dir / "failures.json", [])
            baseline_runner._write_json(
                run_dir / "raw_calls" / "verified.json",
                {
                    "qid": "verified",
                    "evidence_alias_map": [],
                    "calls": [
                        {
                            "call_index": 1,
                            "purpose": "calculation_plan",
                            "token_usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 1,
                                "total_tokens": 5,
                            },
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 4,
                                    "completion_tokens": 1,
                                    "total_tokens": 5,
                                }
                            },
                            "transport_attempt_count": 1,
                            "transport_rejections": [],
                        },
                        {
                            "call_index": 2,
                            "purpose": "verified_calculation_reasoning",
                            "token_usage": {
                                "prompt_tokens": 6,
                                "completion_tokens": 1,
                                "total_tokens": 7,
                            },
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 6,
                                    "completion_tokens": 1,
                                    "total_tokens": 7,
                                }
                            },
                            "transport_attempt_count": 1,
                            "transport_rejections": [],
                        },
                    ]
                },
            )
            baseline_runner._write_json(
                run_dir / "raw_calls" / "direct.json",
                {
                    "qid": "direct",
                    "evidence_alias_map": [],
                    "calls": [
                        {
                            "call_index": 1,
                            "purpose": "initial_answer",
                            "token_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 2,
                                    "total_tokens": 12,
                                }
                            },
                            "transport_attempt_count": 1,
                            "transport_rejections": [],
                        },
                        {
                            "call_index": 2,
                            "purpose": "format_consistency_retry",
                            "token_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 2,
                                    "total_tokens": 12,
                                }
                            },
                            "transport_attempt_count": 1,
                            "transport_rejections": [],
                        },
                    ]
                },
            )

            manifest = baseline_runner._write_manifest(
                run_dir,
                question_order=["verified", "direct"],
                fingerprint="fingerprint",
                public_config={
                    "created_at": "2026-07-25T00:00:00+08:00",
                    "model": {"model_name": "qwen3.7-plus-2026-05-26"},
                    "scope": {
                        "question_count": 2,
                        "qids": ["verified", "direct"],
                    },
                    "answer_blind_contract": {},
                },
            )

        self.assertEqual(manifest["raw_call_count"], 4)
        self.assertEqual(manifest["format_retry_count"], 1)
        self.assertEqual(
            manifest["call_purpose_counts"],
            {
                "calculation_plan": 1,
                "format_consistency_retry": 1,
                "initial_answer": 1,
                "verified_calculation_reasoning": 1,
            },
        )

    def test_manifest_marks_provider_usage_drift_unobservable(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            baseline_runner._write_json(
                run_dir / "answers.json",
                [
                    {
                        "qid": "q1",
                        "token_usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    }
                ],
            )
            baseline_runner._write_json(run_dir / "failures.json", [])
            baseline_runner._write_json(
                run_dir / "raw_calls" / "q1.json",
                {
                    "qid": "q1",
                    "evidence_alias_map": [],
                    "calls": [
                        {
                            "call_index": 1,
                            "purpose": "initial_answer",
                            "token_usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 1,
                                "total_tokens": 3,
                            },
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 2,
                                    "completion_tokens": 2,
                                    "total_tokens": 4,
                                }
                            },
                        }
                    ],
                },
            )

            manifest = baseline_runner._write_manifest(
                run_dir,
                question_order=["q1"],
                fingerprint="fingerprint",
                public_config={
                    "created_at": "2026-07-25T00:00:00+08:00",
                    "model": {"model_name": "qwen3.7-plus-2026-05-26"},
                    "scope": {"question_count": 1, "qids": ["q1"]},
                    "answer_blind_contract": {},
                },
            )

        self.assertFalse(
            manifest["all_observed_usage_from_provider_raw_fields"]
        )
        self.assertTrue(manifest["unobservable_usage_risk"])
        self.assertTrue(manifest["usage_reconciliation_problems"])

    def test_manifest_keeps_provider_usage_when_row_usage_drifts(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            baseline_runner._write_json(
                run_dir / "answers.json",
                [
                    {
                        "qid": "q1",
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                ],
            )
            baseline_runner._write_json(run_dir / "failures.json", [])
            usage = {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
            baseline_runner._write_json(
                run_dir / "raw_calls" / "q1.json",
                {
                    "qid": "q1",
                    "evidence_alias_map": [],
                    "calls": [
                        {
                            "call_index": 1,
                            "purpose": "initial_answer",
                            "token_usage": usage,
                            "raw_response": {"usage": usage},
                            "transport_attempt_count": 1,
                            "transport_rejections": [],
                        }
                    ],
                },
            )

            manifest = baseline_runner._write_manifest(
                run_dir,
                question_order=["q1"],
                fingerprint="fingerprint",
                public_config={
                    "created_at": "2026-07-25T00:00:00+08:00",
                    "model": {"model_name": "qwen3.7-plus-2026-05-26"},
                    "scope": {"question_count": 1, "qids": ["q1"]},
                    "answer_blind_contract": {},
                },
            )

        self.assertEqual(manifest["token_usage"]["total_tokens"], 12)
        self.assertTrue(manifest["unobservable_usage_risk"])


class VerifiedCalculationAuditTests(TestCase):
    @staticmethod
    def question() -> BQuestion:
        return BQuestion(
            qid="calc_audit_qid",
            domain="financial_reports",
            split="B",
            question=(
                "材料披露2025年收入为10元、成本为4元，计算两者差额，"
                "结果保留两位小数。"
            ),
            options={},
            answer_format="calculation",
            type="计算题",
            answer_slots=1,
            answer_slot_templates=("99.99",),
        )

    @staticmethod
    def evidence() -> list[dict[str, object]]:
        return [
            {
                "evidence_key": "E01",
                "source_key": "S01",
                "evidence_id": "report::metric_1",
                "doc_id": "report",
                "rank": 1,
                "unit_type": "metric_row",
                "title_path": ["2025年年度报告"],
                "text": "2025年收入为10元，成本为4元。",
                "prompt_text_sha256": "prompt-hash",
                "source_text_sha256": "source-hash",
                "truncated": False,
            }
        ]

    @staticmethod
    def raw_plan() -> dict[str, object]:
        return {
            "variables": [
                {
                    "name": "income",
                    "value": "10",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["E01"],
                },
                {
                    "name": "cost",
                    "value": "4",
                    "value_type": "decimal",
                    "unit": "元",
                    "evidence_ids": ["E01"],
                },
            ],
            "steps": [
                {
                    "id": "difference",
                    "op": "sub",
                    "args": [{"ref": "income"}, {"ref": "cost"}],
                }
            ],
            "outputs": [{"source": {"ref": "difference"}, "format": "decimal2"}],
            # Deliberately omit supporting_evidence_ids. The deterministic
            # normalizer may add it, but the audit trace must disclose that.
            "decision_summary": "收入10元减成本4元，差额为6.00元。",
        }

    def test_freezes_decimal_answer_before_reasoning_only_call(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint_path = (
                run_dir / "frozen_answers" / "calc_audit_qid.json"
            )

            class FakeClient:
                config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

                def __init__(self) -> None:
                    self.calls = 0

                def chat_json(
                    self,
                    messages: list[dict[str, str]],
                    *,
                    response_schema: dict[str, object],
                    schema_name: str,
                    extra_body: dict[str, object],
                ) -> LLMResponse:
                    self.calls += 1
                    if self.calls == 1:
                        content = json.dumps(
                            VerifiedCalculationAuditTests.raw_plan(),
                            ensure_ascii=False,
                        )
                        usage = TokenUsage(120, 30, 150)
                    else:
                        self_outer.assertTrue(checkpoint_path.exists())
                        frozen = json.loads(
                            checkpoint_path.read_text(encoding="utf-8")
                        )
                        self_outer.assertEqual(frozen["answer_parts"], ["6.00"])
                        self_outer.assertEqual(
                            set(response_schema["properties"]),
                            {"reasoning"},
                        )
                        self_outer.assertEqual(
                            set(response_schema["required"]),
                            {"reasoning"},
                        )
                        content = json.dumps(
                            {
                                "reasoning": (
                                    "材料披露2025年收入为10元、成本为4元，两者相减"
                                    "得到6元；按要求保留两位小数。最终答案为6.00。"
                                )
                            },
                            ensure_ascii=False,
                        )
                        usage = TokenUsage(80, 20, 100)
                    return LLMResponse(
                        content=content,
                        token_usage=usage,
                        raw_payload={"usage": usage.to_dict()},
                        response_format_mode="native_json_schema_strict",
                    )

            self_outer = self
            outcome = run_verified_calculation(
                self.question(),
                evidence=self.evidence(),
                client=FakeClient(),
                run_dir=run_dir,
                thinking_budget=256,
            )

            self.assertEqual(outcome["answer_parts"], ["6.00"])
            self.assertEqual(outcome["token_usage"]["total_tokens"], 250)
            self.assertEqual(
                outcome["decision_trace"]["calculation_plan_normalizations"],
                [{"reason": "default_empty_supporting_evidence_ids"}],
            )
            self.assertTrue(
                outcome["decision_trace"]["answer_stage"]["answer_parts_frozen"]
            )
            self.assertEqual(
                outcome["decision_trace"]["reasoning_stage"]["output_fields"],
                ["reasoning"],
            )

    def test_transport_failures_distinguish_429_from_unobservable_generation(self) -> None:
        response_429 = requests.Response()
        response_429.status_code = 429
        rate_limit_error = requests.HTTPError(response=response_429)
        cases = (
            (requests.ReadTimeout("timed out"), True),
            (RuntimeError("unknown provider failure"), True),
            (rate_limit_error, False),
        )

        for provider_error, expected_risk in cases:
            with self.subTest(error=type(provider_error).__name__):
                class FailingClient:
                    config = SimpleNamespace(
                        model_name="qwen3.7-plus-2026-05-26"
                    )

                    def chat_json(self, *args: object, **kwargs: object) -> object:
                        raise provider_error

                with TemporaryDirectory() as temporary:
                    with self.assertRaises(VerifiedCalculationStageError) as raised:
                        run_verified_calculation(
                            self.question(),
                            evidence=self.evidence(),
                            client=FailingClient(),
                            run_dir=Path(temporary),
                            thinking_budget=256,
                        )

                failure = raised.exception
                self.assertEqual(failure.stage, "answer")
                self.assertEqual(
                    failure.unobservable_usage_risk,
                    expected_risk,
                )
                self.assertEqual(failure.calls, [])
                self.assertEqual(
                    failure.retry_route,
                    (
                        "transport_client_explicit_429_only"
                        if not expected_risk
                        else "do_not_retry_unobservable_generation"
                    ),
                )

    def test_reasoning_contract_failure_keeps_checkpoint_and_observed_usage(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)

            class InvalidReasoningClient:
                config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

                def __init__(self) -> None:
                    self.calls = 0

                def chat_json(
                    self,
                    messages: list[dict[str, str]],
                    *,
                    response_schema: dict[str, object],
                    schema_name: str,
                    extra_body: dict[str, object],
                ) -> LLMResponse:
                    self.calls += 1
                    if self.calls == 1:
                        content = json.dumps(
                            VerifiedCalculationAuditTests.raw_plan(),
                            ensure_ascii=False,
                        )
                        usage = TokenUsage(120, 30, 150)
                    else:
                        content = json.dumps(
                            {
                                "reasoning": "材料给出了输入，但此处故意缺少冻结结论。"
                            },
                            ensure_ascii=False,
                        )
                        usage = TokenUsage(80, 20, 100)
                    return LLMResponse(
                        content=content,
                        token_usage=usage,
                        raw_payload={"usage": usage.to_dict()},
                        response_format_mode="native_json_schema_strict",
                    )

            with self.assertRaises(VerifiedCalculationStageError) as raised:
                run_verified_calculation(
                    self.question(),
                    evidence=self.evidence(),
                    client=InvalidReasoningClient(),
                    run_dir=run_dir,
                    thinking_budget=256,
                )

            failure = raised.exception
            self.assertEqual(failure.stage, "reasoning")
            self.assertEqual(failure.error_code, "reasoning_contract_error")
            self.assertEqual(
                failure.retry_route,
                "resume_reasoning_from_frozen_checkpoint",
            )
            self.assertFalse(failure.unobservable_usage_risk)
            self.assertEqual(failure.token_usage["total_tokens"], 250)
            self.assertEqual(len(failure.calls), 2)
            self.assertTrue(
                (run_dir / "frozen_answers" / "calc_audit_qid.json").exists()
            )

    def test_script_preserves_verified_stage_failure_and_usage_risk(self) -> None:
        class ReadTimeoutClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def chat_json(self, *args: object, **kwargs: object) -> object:
                raise requests.ReadTimeout("provider may still be generating")

        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                calculation_mode="verified",
                per_query_top_k=20,
                final_top_k=10,
                supplemental_weight=0.11,
                max_queries_per_option=12,
                max_doc_candidates=6,
                max_hit_chars=1800,
                max_evidence_chars=12000,
                max_format_retries=0,
                thinking_budget=256,
                run_dir=Path(temporary),
            )
            with (
                patch.object(
                    baseline_runner,
                    "retrieve_question_evidence",
                    return_value={"final": {"hits": []}},
                ),
                patch.object(
                    baseline_runner,
                    "prepare_evidence_payload",
                    return_value=self.evidence(),
                ),
            ):
                result = baseline_runner._run_one(
                    self.question(),
                    args=args,
                    client=ReadTimeoutClient(),
                    retriever=object(),
                    all_doc_ids=["report"],
                )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_stage"], "answer")
        self.assertEqual(result["error_code"], "answer_transport_error")
        self.assertEqual(
            result["retry_route"],
            "do_not_retry_unobservable_generation",
        )
        self.assertTrue(result["unobservable_usage_risk"])
        self.assertEqual(result["token_usage"]["total_tokens"], 0)

    def test_verified_terminal_429_is_reconciled_with_call_intent(self) -> None:
        response = requests.Response()
        response.status_code = 429

        class RateLimitedClient:
            config = SimpleNamespace(
                model_name="qwen3.7-plus-2026-05-26"
            )

            def chat_json(self, *args: object, **kwargs: object) -> object:
                raise requests.HTTPError(response=response)

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            public_config = {
                "created_at": "2026-07-25T00:00:00+08:00",
                "model": {
                    "model_name": "qwen3.7-plus-2026-05-26"
                },
                "scope": {
                    "question_count": 1,
                    "qids": ["calc_audit_qid"],
                },
                "answer_blind_contract": {},
            }
            fingerprint = baseline_runner.public_run_config_fingerprint(
                public_config
            )
            baseline_runner._initialize_run_dir(
                run_dir,
                public_config,
                fingerprint,
            )
            args = SimpleNamespace(
                calculation_mode="verified",
                output_contract="joint",
                evidence_compaction="off",
                per_query_top_k=20,
                final_top_k=10,
                supplemental_weight=0.11,
                max_queries_per_option=12,
                max_doc_candidates=6,
                max_hit_chars=1800,
                max_evidence_chars=12000,
                max_format_retries=0,
                thinking_budget=256,
                run_dir=run_dir,
            )
            with (
                patch.object(
                    baseline_runner,
                    "retrieve_question_evidence",
                    return_value={"final": {"hits": []}},
                ),
                patch.object(
                    baseline_runner,
                    "prepare_evidence_payload",
                    return_value=self.evidence(),
                ),
            ):
                result = baseline_runner._run_one(
                    self.question(),
                    args=args,
                    client=RateLimitedClient(),
                    retriever=object(),
                    all_doc_ids=["report"],
                )
            baseline_runner._persist_result(
                run_dir,
                result,
                question_order=["calc_audit_qid"],
                fingerprint=fingerprint,
                public_config=public_config,
            )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            failure_row = json.loads(
                (run_dir / "failures.json").read_text(encoding="utf-8")
            )[0]
            audited = submission_assembler._audit_result_row(
                question=self.question(),
                row=failure_row,
                raw_call_path=(
                    run_dir / "raw_calls" / "calc_audit_qid.json"
                ),
                run_dir=run_dir,
                expected_model="qwen3.7-plus-2026-05-26",
                is_answered=False,
            )

        self.assertFalse(result["unobservable_usage_risk"])
        self.assertEqual(result["failed_transport_attempt_count"], 1)
        self.assertEqual(manifest["transport_attempt_count"], 1)
        self.assertEqual(manifest["transport_rejection_count"], 1)
        self.assertFalse(manifest["unobservable_usage_risk"])
        self.assertEqual(manifest["usage_reconciliation_problems"], [])
        self.assertEqual(audited["calls"], [])

    def test_direct_mode_resumes_observed_raw_call_without_provider_recall(self) -> None:
        class NoCallClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, *args: object, **kwargs: object) -> object:
                self.calls += 1
                raise AssertionError("observed raw response must not be regenerated")

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            evidence = self.evidence()
            alias_map = baseline_runner._evidence_alias_map(evidence)
            reasoning = (
                "材料披露2025年收入为10元、成本为4元，相减得到6元，"
                "按题目要求保留两位小数。结论：6.00"
            )
            baseline_runner._checkpoint_raw_calls(
                run_dir,
                "calc_audit_qid",
                [
                    {
                        "call_index": 1,
                        "purpose": "initial_answer",
                        "content": json.dumps(
                            {"reasoning": reasoning},
                            ensure_ascii=False,
                        ),
                        "token_usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 30,
                            "total_tokens": 150,
                        },
                        "raw_response": {
                            "usage": {
                                "prompt_tokens": 120,
                                "completion_tokens": 30,
                                "total_tokens": 150,
                            }
                        },
                        "transport_attempt_count": 1,
                        "transport_rejections": [],
                    }
                ],
                alias_map,
            )
            args = SimpleNamespace(
                calculation_mode="direct",
                output_contract="reasoning-canonical",
                evidence_compaction="off",
                per_query_top_k=20,
                final_top_k=10,
                supplemental_weight=0.11,
                max_queries_per_option=12,
                max_doc_candidates=6,
                max_hit_chars=1800,
                max_evidence_chars=12000,
                max_format_retries=1,
                thinking_budget=256,
                run_dir=run_dir,
            )
            client = NoCallClient()
            with (
                patch.object(
                    baseline_runner,
                    "retrieve_question_evidence",
                    return_value={"final": {"hits": []}},
                ),
                patch.object(
                    baseline_runner,
                    "prepare_evidence_payload",
                    return_value=evidence,
                ),
            ):
                result = baseline_runner._run_one(
                    self.question(),
                    args=args,
                    client=client,
                    retriever=object(),
                    all_doc_ids=["report"],
                )

        self.assertEqual(client.calls, 0)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_parts"], ["6.00"])
        self.assertEqual(result["token_usage"]["total_tokens"], 150)
        self.assertEqual(len(result["calls"]), 1)

    def test_joint_reasoning_contract_failure_freezes_answer_and_retries_reasoning_only(
        self,
    ) -> None:
        question = BQuestion(
            qid="joint_retry_qid",
            domain="financial_reports",
            split="B",
            question="根据材料判断哪些说法正确？",
            options={"A": "甲正确", "B": "乙正确", "C": "丙正确"},
            answer_format="multi",
            type="多选题",
            answer_slots=1,
            answer_slot_templates=("AB",),
        )

        class FrozenReasoningClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(
                self,
                messages: list[dict[str, str]],
                *,
                response_schema: dict[str, object],
                schema_name: str,
                extra_body: dict[str, object],
            ) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps(
                        {
                            "answer_parts": ["AC"],
                            "reasoning": (
                                "材料支持第一项与第三项，第二项与原文不符，"
                                "但首答把机械结论误写为其他选项。结论：BC"
                            ),
                        },
                        ensure_ascii=False,
                    )
                else:
                    self_outer.assertEqual(
                        set(response_schema["properties"]),
                        {"reasoning"},
                    )
                    self_outer.assertEqual(extra_body, {"enable_thinking": False})
                    content = json.dumps(
                        {
                            "reasoning": (
                                "材料中的第一项与第三项符合条件，第二项与原文不符，"
                                "因此保持冻结答案。结论：AC"
                            ),
                        },
                        ensure_ascii=False,
                    )
                usage = TokenUsage(100, 20, 120)
                return LLMResponse(
                    content=content,
                    token_usage=usage,
                    raw_payload={"usage": usage.to_dict()},
                    response_format_mode="native_json_schema_strict",
                )

        self_outer = self
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            args = SimpleNamespace(
                calculation_mode="direct",
                output_contract="joint",
                evidence_compaction="off",
                per_query_top_k=20,
                final_top_k=10,
                supplemental_weight=0.11,
                max_queries_per_option=12,
                max_doc_candidates=6,
                max_hit_chars=1800,
                max_evidence_chars=12000,
                max_format_retries=1,
                thinking_budget=256,
                run_dir=run_dir,
            )
            client = FrozenReasoningClient()
            with (
                patch.object(
                    baseline_runner,
                    "retrieve_question_evidence",
                    return_value={"final": {"hits": []}},
                ),
                patch.object(
                    baseline_runner,
                    "prepare_evidence_payload",
                    return_value=self.evidence(),
                ),
            ):
                result = baseline_runner._run_one(
                    question,
                    args=args,
                    client=client,
                    retriever=object(),
                    all_doc_ids=["report"],
                )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_parts"], ["AC"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            result["calls"][1]["purpose"],
            "reasoning_only_retry_from_frozen_answer",
        )
        self.assertEqual(result["calls"][1]["frozen_answer_parts"], ["AC"])
        self.assertEqual(result["token_usage"]["total_tokens"], 240)

    def test_joint_reasoning_without_marker_does_not_retry(self) -> None:
        question = BQuestion(
            qid="joint_no_marker_qid",
            domain="financial_reports",
            split="B",
            question="根据材料判断哪些说法正确？",
            options={"A": "甲正确", "B": "乙正确", "C": "丙正确"},
            answer_format="multi",
            type="多选题",
            answer_slots=1,
            answer_slot_templates=("AB",),
        )

        class NoMarkerClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(
                self,
                messages: list[dict[str, str]],
                *,
                response_schema: dict[str, object],
                schema_name: str,
                extra_body: dict[str, object],
            ) -> LLMResponse:
                self.calls += 1
                reasoning = (
                    "材料支持第一项与第三项，第二项的表述与原文不符，"
                    "因此最终答案为A、C。"
                )
                usage = TokenUsage(100, 20, 120)
                return LLMResponse(
                    content=json.dumps(
                        {"answer_parts": ["AC"], "reasoning": reasoning},
                        ensure_ascii=False,
                    ),
                    token_usage=usage,
                    raw_payload={"usage": usage.to_dict()},
                    response_format_mode="native_json_schema_strict",
                )

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "raw_calls").mkdir()
            args = SimpleNamespace(
                calculation_mode="direct",
                output_contract="joint",
                evidence_compaction="off",
                per_query_top_k=20,
                final_top_k=10,
                supplemental_weight=0.11,
                max_queries_per_option=12,
                max_doc_candidates=6,
                max_hit_chars=1800,
                max_evidence_chars=12000,
                max_format_retries=1,
                thinking_budget=256,
                run_dir=run_dir,
            )
            client = NoMarkerClient()
            with (
                patch.object(
                    baseline_runner,
                    "retrieve_question_evidence",
                    return_value={"final": {"hits": []}},
                ),
                patch.object(
                    baseline_runner,
                    "prepare_evidence_payload",
                    return_value=self.evidence(),
                ),
            ):
                result = baseline_runner._run_one(
                    question,
                    args=args,
                    client=client,
                    retriever=object(),
                    all_doc_ids=["report"],
                )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer_parts"], ["AC"])
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["token_usage"]["total_tokens"], 120)
        self.assertEqual(
            result["decision_trace"]["postprocessing_mode"],
            "reasoning_without_explicit_conclusion",
        )

    def test_resume_from_frozen_checkpoint_does_not_rerun_answer_stage(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)

            class FirstRunClient:
                config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

                def __init__(self) -> None:
                    self.calls = 0

                def chat_json(
                    self,
                    messages: list[dict[str, str]],
                    *,
                    response_schema: dict[str, object],
                    schema_name: str,
                    extra_body: dict[str, object],
                ) -> LLMResponse:
                    self.calls += 1
                    content = (
                        json.dumps(
                            VerifiedCalculationAuditTests.raw_plan(),
                            ensure_ascii=False,
                        )
                        if self.calls == 1
                        else json.dumps(
                            {"reasoning": "这是一段已观测但契约不完整的推理摘要。"},
                            ensure_ascii=False,
                        )
                    )
                    usage = (
                        TokenUsage(120, 30, 150)
                        if self.calls == 1
                        else TokenUsage(80, 20, 100)
                    )
                    return LLMResponse(
                        content=content,
                        token_usage=usage,
                        raw_payload={"usage": usage.to_dict()},
                        response_format_mode="native_json_schema_strict",
                    )

            with self.assertRaises(VerifiedCalculationStageError):
                run_verified_calculation(
                    self.question(),
                    evidence=self.evidence(),
                    client=FirstRunClient(),
                    run_dir=run_dir,
                    thinking_budget=256,
                )

            class ResumeReasoningClient:
                config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

                def __init__(self) -> None:
                    self.calls = 0

                def chat_json(
                    self,
                    messages: list[dict[str, str]],
                    *,
                    response_schema: dict[str, object],
                    schema_name: str,
                    extra_body: dict[str, object],
                ) -> LLMResponse:
                    self.calls += 1
                    self_outer.assertEqual(
                        set(response_schema["properties"]),
                        {"reasoning"},
                    )
                    usage = TokenUsage(80, 20, 100)
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "reasoning": (
                                    "材料披露2025年收入为10元、成本为4元，两者相减"
                                    "得到6元；按要求保留两位小数。最终答案为6.00。"
                                )
                            },
                            ensure_ascii=False,
                        ),
                        token_usage=usage,
                        raw_payload={"usage": usage.to_dict()},
                        response_format_mode="native_json_schema_strict",
                    )

            self_outer = self
            resume_client = ResumeReasoningClient()
            outcome = run_verified_calculation(
                self.question(),
                evidence=self.evidence(),
                client=resume_client,
                run_dir=run_dir,
                thinking_budget=256,
            )

            self.assertEqual(resume_client.calls, 1)
            self.assertEqual(len(outcome["calls"]), 3)
            self.assertEqual(outcome["token_usage"]["total_tokens"], 350)
            self.assertEqual(outcome["answer_parts"], ["6.00"])
            self.assertTrue(
                outcome["decision_trace"]["answer_stage"][
                    "resumed_from_frozen_checkpoint"
                ]
            )

    def test_invalid_frozen_checkpoint_fails_locally_without_provider_call(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = (
                run_dir / "frozen_answers" / "calc_audit_qid.json"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text('{"checkpoint_version":"wrong"}', encoding="utf-8")

            class NoCallClient:
                config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

                def __init__(self) -> None:
                    self.calls = 0

                def chat_json(self, *args: object, **kwargs: object) -> object:
                    self.calls += 1
                    raise AssertionError("invalid checkpoint must fail before API call")

            client = NoCallClient()
            with self.assertRaises(VerifiedCalculationStageError) as raised:
                run_verified_calculation(
                    self.question(),
                    evidence=self.evidence(),
                    client=client,
                    run_dir=run_dir,
                    thinking_budget=256,
                )

        failure = raised.exception
        self.assertEqual(client.calls, 0)
        self.assertEqual(failure.error_code, "frozen_checkpoint_invalid")
        self.assertEqual(
            failure.retry_route,
            "stop_checkpoint_integrity_failure",
        )
        self.assertFalse(failure.unobservable_usage_risk)

    def test_tampered_frozen_answer_is_rejected_by_decimal_replay(self) -> None:
        class FreezeThenFailReasoningClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(
                self,
                *args: object,
                **kwargs: object,
            ) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps(
                        VerifiedCalculationAuditTests.raw_plan(),
                        ensure_ascii=False,
                    )
                    usage = TokenUsage(120, 30, 150)
                else:
                    content = json.dumps(
                        {"reasoning": "故意缺失冻结答案结论的推理摘要。"},
                        ensure_ascii=False,
                    )
                    usage = TokenUsage(80, 20, 100)
                return LLMResponse(
                    content=content,
                    token_usage=usage,
                    raw_payload={"usage": usage.to_dict()},
                    response_format_mode="native_json_schema_strict",
                )

        class NoCallClient:
            config = SimpleNamespace(model_name="qwen3.7-plus-2026-05-26")

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, *args: object, **kwargs: object) -> object:
                self.calls += 1
                raise AssertionError("tampered checkpoint must fail locally")

        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with self.assertRaises(VerifiedCalculationStageError):
                run_verified_calculation(
                    self.question(),
                    evidence=self.evidence(),
                    client=FreezeThenFailReasoningClient(),
                    run_dir=run_dir,
                    thinking_budget=256,
                )
            checkpoint_path = (
                run_dir / "frozen_answers" / "calc_audit_qid.json"
            )
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["answer_parts"] = ["7.00"]
            checkpoint_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False),
                encoding="utf-8",
            )
            no_call_client = NoCallClient()
            with self.assertRaises(VerifiedCalculationStageError) as raised:
                run_verified_calculation(
                    self.question(),
                    evidence=self.evidence(),
                    client=no_call_client,
                    run_dir=run_dir,
                    thinking_budget=256,
                )

        self.assertEqual(no_call_client.calls, 0)
        self.assertEqual(raised.exception.error_code, "frozen_checkpoint_invalid")
        self.assertIn("answer replay mismatch", str(raised.exception))
