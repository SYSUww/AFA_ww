from __future__ import annotations

import json
import unittest

from afa_agent.b_board.evaluator import (
    BLIND_PROMPT_VERSION,
    BLIND_SYSTEM_PROMPT,
    INDEPENDENT_PROMPT_VERSION,
    INDEPENDENT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ConfidenceEvaluation,
    FixedBlindPairEvaluator,
    FixedConfidenceEvaluator,
    build_blind_pair,
    build_calibration_subjects,
    build_evaluation_messages,
    build_independent_messages,
    decide_candidate_promotion,
    detect_hard_failures,
    parse_independent_payload,
    parse_evaluation_payload,
    validate_calibration_sentinels,
)
from afa_agent.models import TokenUsage


def evaluation_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "document_relevance": 90,
        "evidence_sufficiency": 88,
        "citation_alignment": 91,
        "answer_entailment": 86,
        "alternative_exclusion": 84,
        "calculation_reproducibility": None,
        "format_compliance": 99,
        "internal_consistency": 92,
        "overall_confidence": 87,
        "verdict": "supported",
        "blocking_reasons": [],
        "low_confidence_reasons": [],
        "suggested_improvements": [],
        "answer_verdict": "likely_correct",
        "error_likelihood": 5,
        "suspected_error_types": [],
        "suspected_error_reasons": [],
        "correction_candidate_parts": ["A"],
    }
    payload.update(overrides)
    return payload


def independent_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": INDEPENDENT_PROMPT_VERSION,
        "status": "resolved",
        "answer_parts": ["A"],
        "used_evidence_ids": ["u1"],
        "option_assessments": {"A": "supported", "B": "contradicted"},
        "confidence": 92,
        "solution_summary": "证据支持A并排除B",
        "missing_evidence": [],
    }
    payload.update(overrides)
    return payload


def subject(**overrides):
    payload = {
        "qid": "q1",
        "domain": "regulatory",
        "type": "单选题",
        "answer_format": "mcq",
        "question": "问题",
        "options": {"A": "是", "B": "否"},
        "answer_slot_count": 1,
        "answer_parts": ["A"],
        "used_evidence_ids": ["u1"],
        "evidence_items": [{"unit_id": "u1", "text": "证据"}],
        "decision_trace": {},
        "calculation_trace": {},
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(self, payload):
        self.content = json.dumps(payload, ensure_ascii=False)
        self.token_usage = TokenUsage(10, 2, 12)


class FakeClient:
    def __init__(self, payload):
        self.payloads = list(payload) if isinstance(payload, list) else [payload]
        self.messages = []

    def chat_json(self, messages):
        self.messages.append(messages)
        return FakeResponse(self.payloads.pop(0))


class BBoardEvaluatorTests(unittest.TestCase):
    def test_all_fixed_evaluator_prompts_share_format_priority_rule(self):
        for prompt in (
            INDEPENDENT_SYSTEM_PROMPT,
            JUDGE_SYSTEM_PROMPT,
            BLIND_SYSTEM_PROMPT,
        ):
            self.assertIn("题目明确要求 > README通用规则 > 提交模板占位", prompt)
            self.assertIn("不带单位", prompt)
            self.assertIn("不带%", prompt)
            self.assertIn("保留两位小数", prompt)
            self.assertRegex(prompt, r"至少(?:包含)?两个")

    def test_model_prompts_strip_generator_answer_labels_and_trace(self):
        leaky_subject = subject(
            type="多选题",
            answer_format="multi",
            options={"A": "甲", "B": "乙"},
            answer_parts=["AB"],
            evidence_items=[
                {
                    "unit_id": "u1",
                    "text": "原始条款证据",
                    "metadata": {
                        "option_key": "B",
                        "rule_label": False,
                        "targeted_literal": True,
                    },
                }
            ],
            decision_trace={
                "raw_answer": "A",
                "answer_policy": {
                    "final_answer": "A",
                    "supported_options": ["A"],
                },
            },
        )
        independent = parse_independent_payload(
            independent_payload(
                answer_parts=["AB"],
                option_assessments={"A": "supported", "B": "supported"},
            ),
            subject=leaky_subject,
        )

        independent_prompt = build_independent_messages(leaky_subject)[1]["content"]
        judge_prompt = build_evaluation_messages(leaky_subject, independent)[1]["content"]

        self.assertNotIn('"rule_label"', independent_prompt)
        self.assertNotIn('"rule_label"', judge_prompt)
        self.assertNotIn('"raw_answer"', judge_prompt)
        self.assertNotIn('"supported_options"', judge_prompt)

    def test_fixed_prompts_forbid_inventing_universal_quantifiers(self):
        for prompt in (INDEPENDENT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT):
            self.assertIn("不得擅自补出所有、任何或一律等全称量词", prompt)

    def test_subject_rejects_optimizer_metadata(self):
        with self.assertRaisesRegex(ValueError, "optimizer metadata"):
            build_independent_messages(subject(candidate_id="candidate"))

    def test_conservative_score_uses_weakest_dimension(self):
        result = parse_evaluation_payload(
            qid="q1",
            question_type="mcq",
            payload=evaluation_payload(alternative_exclusion=61),
        )
        self.assertEqual(result.confidence_score, 61)
        self.assertEqual(result.tier, "medium")

    def test_hard_failure_forces_blocked(self):
        result = parse_evaluation_payload(
            qid="q1",
            question_type="mcq",
            payload=evaluation_payload(),
            hard_failures=["missing_used_evidence_ids"],
        )
        self.assertEqual(result.confidence_score, 0)
        self.assertEqual(result.tier, "blocked")

    def test_likely_wrong_is_ranked_as_blocked_with_error_signal(self):
        result = parse_evaluation_payload(
            qid="q1",
            question_type="mcq",
            payload=evaluation_payload(
                answer_verdict="likely_wrong",
                error_likelihood=91,
                suspected_error_types=["wrong_or_missing_option"],
                suspected_error_reasons=["独立答案为B，证据直接反驳A"],
                correction_candidate_parts=["B"],
            ),
        )
        self.assertEqual(result.confidence_score, 39)
        self.assertEqual(result.tier, "blocked")
        self.assertTrue(result.suspected_error)
        self.assertIn("独立答案为B", result.low_confidence_reasons[0])

    def test_independent_status_alias_is_normalized(self):
        result = parse_independent_payload(
            independent_payload(status="supported"),
            subject=subject(),
        )
        self.assertEqual(result.status, "resolved")

    def test_multi_choice_independent_answer_requires_at_least_two_options(self):
        multi_subject = subject(
            type="多选题",
            answer_format="multi",
            options={"A": "甲", "B": "乙", "C": "丙"},
            answer_parts=["AB"],
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            parse_independent_payload(
                independent_payload(
                    answer_parts=["A"],
                    option_assessments={
                        "A": "supported",
                        "B": "contradicted",
                        "C": "insufficient",
                    },
                ),
                subject=multi_subject,
            )

    def test_multi_choice_correction_candidate_requires_at_least_two_options(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            parse_evaluation_payload(
                qid="q1",
                question_type="multi",
                payload=evaluation_payload(correction_candidate_parts=["A"]),
            )

    def test_multi_choice_sealed_answer_contract_is_a_hard_failure(self):
        failures = detect_hard_failures(
            subject(
                type="多选题",
                answer_format="multi",
                options={"A": "甲", "B": "乙"},
                answer_parts=["A"],
            )
        )
        self.assertIn("invalid_choice_answer_contract", failures)

    def test_choice_answer_comparison_ignores_display_separators(self):
        independent = parse_independent_payload(
            independent_payload(answer_parts=["A、B"]),
            subject=subject(),
        )
        result = parse_evaluation_payload(
            qid="q1",
            question_type="mcq",
            payload=evaluation_payload(correction_candidate_parts=[]),
            sealed_answer_parts=["AB"],
            independent=independent,
        )
        self.assertTrue(result.answer_match)

    def test_missing_applicable_dimension_is_zero_only_for_hard_failed_answer(self):
        result = parse_evaluation_payload(
            qid="q1",
            question_type="calculation",
            payload=evaluation_payload(
                alternative_exclusion=None,
                calculation_reproducibility=None,
            ),
            hard_failures=["invalid_answer_slot:1"],
        )
        self.assertEqual(result.dimensions["calculation_reproducibility"], 0)
        self.assertEqual(result.confidence_score, 0)

        with self.assertRaisesRegex(ValueError, "applicable decision dimension"):
            parse_evaluation_payload(
                qid="q1",
                question_type="calculation",
                payload=evaluation_payload(
                    alternative_exclusion=None,
                    calculation_reproducibility=None,
                ),
            )

    def test_detects_evidence_and_calculation_integrity(self):
        failures = detect_hard_failures(
            subject(
                type="计算题",
                answer_format="calculation",
                answer_slot_templates=["999999.99"],
                used_evidence_ids=["missing"],
                calculation_trace={"replay_verified": False},
            )
        )
        self.assertIn("used_evidence_not_in_final:missing", failures)
        self.assertIn("calculation_not_replay_verified", failures)
        self.assertIn("calculation_not_grounding_verified", failures)

    def test_detects_explanatory_text_in_numeric_slot(self):
        failures = detect_hard_failures(
            subject(
                type="计算题",
                answer_format="calculation",
                answer_slot_templates=["999999.99"],
                answer_parts=["证据不足，无法计算"],
                calculation_trace={
                    "replay_verified": True,
                    "grounding_verified": True,
                },
            )
        )
        self.assertIn("invalid_answer_slot:1", failures)

    def test_question_specific_precision_overrides_generic_slot_template(self):
        base = subject(
            question="请计算该指标，保留一位小数，答案不带单位。",
            type="计算题",
            answer_format="calculation",
            answer_slot_templates=["999999.99"],
            calculation_trace={
                "replay_verified": True,
                "grounding_verified": True,
            },
        )
        self.assertNotIn("invalid_answer_slot:1", detect_hard_failures(base | {"answer_parts": ["67.1"]}))
        self.assertIn("invalid_answer_slot:1", detect_hard_failures(base | {"answer_parts": ["67.10"]}))

    def test_question_specific_no_percent_overrides_percent_slot_template(self):
        base = subject(
            question="答案格式为权益乘数；近似资产收益率，均保留两位小数，后者不带%。",
            type="计算题",
            answer_format="calculation",
            answer_slot_count=2,
            answer_slot_templates=["999999.99", "999999.99%"],
            calculation_trace={
                "replay_verified": True,
                "grounding_verified": True,
            },
        )
        self.assertNotIn(
            "invalid_answer_slot:2",
            detect_hard_failures(base | {"answer_parts": ["2.58", "7.65"]}),
        )
        self.assertIn(
            "invalid_answer_slot:2",
            detect_hard_failures(base | {"answer_parts": ["2.58", "7.65%"]}),
        )

    def test_fixed_evaluator_uses_clean_subject_and_usage(self):
        client = FakeClient([independent_payload(), evaluation_payload()])
        evaluator = FixedConfidenceEvaluator(client)
        result, usage = evaluator.evaluate(subject())
        self.assertEqual(result.tier, "high")
        self.assertEqual(usage["total_tokens"], 24)
        self.assertTrue(result.answer_match)
        independent_serialized = client.messages[0][1]["content"]
        self.assertNotIn("answer_parts", independent_serialized)
        serialized = client.messages[1][1]["content"]
        self.assertNotIn("candidate_id", serialized)

    def test_blind_pair_hides_candidate_identity(self):
        pair = build_blind_pair(qid="q1", incumbent=subject(answer_parts=["A"]), candidate=subject(answer_parts=["B"]), salt="x")
        self.assertEqual(set(pair.public_payload["answers"]), {"A", "B"})
        self.assertNotIn("candidate", json.dumps(pair.public_payload))
        self.assertNotEqual(pair.candidate_label, pair.incumbent_label)

    def test_fixed_blind_pair_evaluator_validates_winner(self):
        pair = build_blind_pair(
            qid="q1",
            incumbent=subject(answer_parts=["A"]),
            candidate=subject(answer_parts=["B"]),
            salt="fixed",
        )
        client = FakeClient(
            {
                "prompt_version": BLIND_PROMPT_VERSION,
                "winner": pair.candidate_label,
                "confidence": 91,
                "reason": "candidate is better supported",
            }
        )
        result, usage = FixedBlindPairEvaluator(client).evaluate(pair)
        self.assertEqual(result.winner, pair.candidate_label)
        self.assertEqual(result.confidence, 91)
        self.assertEqual(usage["total_tokens"], 12)

    def test_changed_answer_requires_tier_gain_and_blind_win(self):
        incumbent = parse_evaluation_payload(qid="q1", question_type="mcq", payload=evaluation_payload(overall_confidence=70, alternative_exclusion=70))
        candidate = parse_evaluation_payload(qid="q1", question_type="mcq", payload=evaluation_payload())
        rejected = decide_candidate_promotion(
            incumbent=incumbent,
            candidate=candidate,
            answer_changed=True,
            blind_winner="A",
            candidate_blind_label="B",
        )
        self.assertFalse(rejected["promote"])
        accepted = decide_candidate_promotion(
            incumbent=incumbent,
            candidate=candidate,
            answer_changed=True,
            blind_winner="B",
            candidate_blind_label="B",
        )
        self.assertTrue(accepted["promote"])

    def test_sentinel_validation_rejects_high_score(self):
        low = ConfidenceEvaluation(
            qid="s",
            dimensions={},
            confidence_score=20,
            tier="blocked",
            verdict="unsupported",
            blocking_reasons=(),
            low_confidence_reasons=(),
            suggested_improvements=(),
            hard_failures=(),
        )
        evaluations = {
            name: low
            for name in [
                "wrong_year",
                "wrong_unit",
                "wrong_arithmetic",
                "irrelevant_evidence",
                "missing_citation",
                "format_only",
                "invented_universal_scope",
            ]
        }
        self.assertTrue(validate_calibration_sentinels(evaluations)["passed"])

    def test_calibration_subjects_are_structurally_valid(self):
        subjects = build_calibration_subjects()
        self.assertEqual(len(subjects), 7)
        self.assertEqual(
            {item["qid"] for item in subjects},
            {
                "wrong_year",
                "wrong_unit",
                "wrong_arithmetic",
                "irrelevant_evidence",
                "missing_citation",
                "format_only",
                "invented_universal_scope",
            },
        )
        for item in subjects:
            self.assertEqual(detect_hard_failures(item), [], item["qid"])


if __name__ == "__main__":
    unittest.main()
