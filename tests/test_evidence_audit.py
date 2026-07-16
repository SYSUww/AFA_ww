from __future__ import annotations

import unittest

from afa_agent.evidence_audit import build_shared_deductible_provenance
from scripts.run_b_board_migration_loop import audit_answer_from_evidence


def answer_row(*, pred_answer: str, option_labels: dict[str, bool], debug_meta: dict) -> dict:
    return {
        "qid": "q1",
        "domain": "test",
        "question_type": "tf",
        "pred_answer": pred_answer,
        "option_labels": option_labels,
        "evidence_items": [
            {"unit_id": "e1", "doc_id": "doc-1", "text": "证据", "score": 1.0}
        ],
        "reasoning_summary": "",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "debug_meta": debug_meta,
    }


class EvidenceAuditCalibrationTests(unittest.TestCase):
    def test_tf_false_answer_inherits_gate_from_refuted_statement(self) -> None:
        row = answer_row(
            pred_answer="B",
            option_labels={"A": False, "B": True},
            debug_meta={
                "option_debug": [
                    {
                        "option": "A",
                        "evidence_gate": {
                            "final_gate": {
                                "status": "pass",
                                "certainty_score": 0.92,
                                "reasons": [],
                            }
                        },
                    }
                ],
                "final_consistency_check": {"issues": []},
            },
        )

        audit = audit_answer_from_evidence(
            row,
            {"options": {}, "answer_format": "tf"},
        )

        self.assertEqual(audit["support_status"], "supported")
        self.assertEqual(audit["selected_gate_statuses"], {"B": "pass"})
        self.assertEqual(audit["selected_gate_sources"], {"B": "tf_statement_refutation:A"})

    def test_tf_false_without_explicit_a_false_label_is_not_promoted(self) -> None:
        row = answer_row(
            pred_answer="B",
            option_labels={"B": True},
            debug_meta={
                "option_debug": [
                    {"option": "A", "evidence_gate": {"final_gate": {"status": "pass"}}}
                ]
            },
        )

        audit = audit_answer_from_evidence(row, {"options": {}, "answer_format": "tf"})

        self.assertEqual(audit["support_status"], "unsupported")

    def test_legacy_rule_output_without_structured_provenance_stays_unsupported(self) -> None:
        row = answer_row(
            pred_answer="B",
            option_labels={"A": False, "B": True},
            debug_meta={
                "option_debug": [],
                "selected_evidence_ids": ["e1"],
                "rule_outputs": [
                    {
                        "option": "B",
                        "answer": "B",
                        "label": True,
                        "reason": "公式可复算",
                        "confidence": 0.95,
                    }
                ],
                "final_consistency_check": {"issues": []},
            },
        )

        audit = audit_answer_from_evidence(
            row,
            {"options": {"A": "甲", "B": "乙"}, "answer_format": "mcq"},
        )

        self.assertEqual(audit["support_status"], "unsupported")
        self.assertEqual(audit["selected_gate_statuses"], {"B": "missing"})

    def test_rule_output_without_matching_evidence_ids_remains_unsupported(self) -> None:
        row = answer_row(
            pred_answer="B",
            option_labels={"A": False, "B": True},
            debug_meta={
                "option_debug": [],
                "selected_evidence_ids": ["not-in-final-evidence"],
                "rule_outputs": [
                    {
                        "option": "B",
                        "answer": "B",
                        "label": True,
                        "reason": "公式可复算",
                        "confidence": 0.95,
                    }
                ],
            },
        )

        audit = audit_answer_from_evidence(
            row,
            {"options": {"A": "甲", "B": "乙"}, "answer_format": "mcq"},
        )

        self.assertEqual(audit["support_status"], "unsupported")
        self.assertIn("missing_selected_gate:B", audit["audit_reasons"])

    def test_allowlisted_structured_rule_is_supported_by_audit(self) -> None:
        evidence_items = [
            {
                "unit_id": "e-shared",
                "doc_id": "doc-1",
                "text": "计划一同一保单家庭成员共享免赔额，免赔额为1万元。",
            },
            {
                "unit_id": "e-formula",
                "doc_id": "doc-2",
                "text": "应当给付的保险金扣除免赔额余额，赔付比例100%。",
            },
        ]
        question = {
            "question": (
                "王某投保平安e生保计划一，共享免赔额。本人医疗费用2万元、医保报销8000元，"
                "配偶医疗费用1.5万元、医保报销6000元；另投保太保团体百万医疗，免赔额1万元。"
            ),
            "options": {
                "A": "e生保赔付1.1万元，太保赔付0.2万元，合计1.3万元",
                "B": "其他",
            },
            "answer_format": "mcq",
        }
        provenance = build_shared_deductible_provenance(
            decision_option="A",
            evidence_unit_ids=["e-shared", "e-formula"],
        )
        row = {
            **answer_row(
                pred_answer="A",
                option_labels={"A": True, "B": False},
                debug_meta={"option_debug": [], "rule_outputs": [provenance]},
            ),
            "evidence_items": evidence_items,
        }

        audit = audit_answer_from_evidence(row, question)

        self.assertEqual(audit["support_status"], "supported")
        self.assertEqual(
            audit["selected_gate_sources"],
            {"A": "rule_output:insurance.shared_family_deductible"},
        )


if __name__ == "__main__":
    unittest.main()
