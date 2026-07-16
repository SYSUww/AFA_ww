from __future__ import annotations

from copy import deepcopy
import unittest

from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.evidence_audit import (
    SHARED_DEDUCTIBLE_RULE_ID,
    build_shared_deductible_provenance,
    resolve_answer_decisions,
    validate_rule_provenance,
)
from afa_agent.models import Question, RetrievalHit


QUESTION_TEXT = (
    "王某投保了平安e生保（计划一：免赔额1万元，家庭共享）和太保团体百万医疗（免赔额1万元）。"
    "王某家庭三人同时参保e生保计划一，共享免赔额。某年度，王某本人发生医疗费用2万元（全部属保险责任），"
    "医保报销8000元；其配偶发生医疗费用1.5万元，医保报销6000元。则e生保和太保分别应赔付多少？"
    "假设王某家庭未从其他途径获得补偿。"
)
OPTIONS = {
    "A": "e生保赔付1.1万元，太保赔付0.2万元，合计1.3万元",
    "B": "e生保赔付0.2万元，太保赔付0.9万元，合计1.1万元",
    "C": "e生保赔付1.1万元，太保赔付0万元，合计1.1万元",
    "D": "e生保赔付0.2万元，太保赔付0万元，合计0.2万元",
}
EVIDENCE_ITEMS = [
    {
        "unit_id": "e-shared",
        "doc_id": "5",
        "text": "若选择投保计划一，同一保单中家庭成员共享免赔额，免赔额为1万元。",
    },
    {
        "unit_id": "e-formula",
        "doc_id": "6",
        "text": "应当给付的保险金=医疗费用-医保补偿-免赔额余额，赔付比例为100％。",
    },
]


def question_payload() -> dict[str, object]:
    return {"question": QUESTION_TEXT, "options": dict(OPTIONS), "answer_format": "mcq"}


class RuleProvenanceValidationTests(unittest.TestCase):
    def test_allowlisted_rule_replays_with_exact_final_evidence(self) -> None:
        provenance = build_shared_deductible_provenance(
            decision_option="A",
            evidence_unit_ids=["e-shared", "e-formula"],
        )

        result = validate_rule_provenance(
            provenance,
            question=question_payload(),
            pred_answer="A",
            evidence_items=EVIDENCE_ITEMS,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["support_status"], "supported")
        self.assertEqual(result["rule_id"], SHARED_DEDUCTIBLE_RULE_ID)

    def test_old_rule_output_without_schema_is_not_promoted(self) -> None:
        result = validate_rule_provenance(
            {"option": "A", "label": True, "confidence": 0.95},
            question=question_payload(),
            pred_answer="A",
            evidence_items=EVIDENCE_ITEMS,
        )

        self.assertFalse(result["valid"])
        self.assertIn("invalid_or_missing_provenance_schema_version", result["reasons"])
        self.assertIn("rule_not_allowlisted", result["reasons"])

    def test_evidence_ids_must_belong_to_final_evidence(self) -> None:
        provenance = build_shared_deductible_provenance(
            decision_option="A",
            evidence_unit_ids=["e-shared", "not-final"],
        )

        result = validate_rule_provenance(
            provenance,
            question=question_payload(),
            pred_answer="A",
            evidence_items=EVIDENCE_ITEMS,
        )

        self.assertFalse(result["valid"])
        self.assertIn("rule_evidence_not_in_final:not-final", result["reasons"])

    def test_tampered_trace_cannot_pass_allowlisted_validator(self) -> None:
        provenance = build_shared_deductible_provenance(
            decision_option="A",
            evidence_unit_ids=["e-shared", "e-formula"],
        )
        tampered = deepcopy(provenance)
        tampered["rule_trace"]["outputs"]["taibao_payment_yuan"] = 9_000

        result = validate_rule_provenance(
            tampered,
            question=question_payload(),
            pred_answer="A",
            evidence_items=EVIDENCE_ITEMS,
        )

        self.assertFalse(result["valid"])
        self.assertIn("rule_outputs_not_replayable", result["reasons"])

    def test_generic_evidence_without_required_terms_is_not_enough(self) -> None:
        provenance = build_shared_deductible_provenance(
            decision_option="A",
            evidence_unit_ids=["e-shared", "e-generic"],
        )
        evidence = [EVIDENCE_ITEMS[0], {"unit_id": "e-generic", "doc_id": "6", "text": "一般保险条款。"}]

        result = validate_rule_provenance(
            provenance,
            question=question_payload(),
            pred_answer="A",
            evidence_items=evidence,
        )

        self.assertFalse(result["valid"])
        self.assertIn("rule_evidence_conditions_not_met", result["reasons"])

    def test_tf_false_maps_to_refuted_a_statement(self) -> None:
        self.assertEqual(
            resolve_answer_decisions("B", "tf"),
            [{"answer_option": "B", "evaluated_option": "A", "expected_label": False}],
        )
        self.assertEqual(
            resolve_answer_decisions("A", "tf"),
            [{"answer_option": "A", "evaluated_option": "A", "expected_label": True}],
        )


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def search(self, *args: object, **kwargs: object) -> list[RetrievalHit]:
        return list(self.hits)


class InsuranceEarlyRuleIntegrationTests(unittest.TestCase):
    def test_shared_deductible_rule_emits_validated_provenance_without_model(self) -> None:
        hits = [
            RetrievalHit(
                unit_id=item["unit_id"],
                doc_id=item["doc_id"],
                score=100.0,
                title_path=["保险条款"],
                text=item["text"],
            )
            for item in EVIDENCE_ITEMS
        ]
        solver = InsuranceSolver.__new__(InsuranceSolver)
        solver.retriever = FakeRetriever(hits)
        solver.retrieval_settings = {"top_k": 4, "unit_type_boosts": {}, "ensure_per_doc": True, "expand_neighbors": False}
        solver.answering_settings = {"max_hits": 6, "max_hit_chars": 0, "prompt_template_id": "test"}
        question = Question(
            qid="ins_a_006",
            domain="insurance",
            split="test",
            question=QUESTION_TEXT,
            options=dict(OPTIONS),
            answer_format="mcq",
            type="计算题",
            doc_ids=["5", "6"],
        )

        answer = solver._solve_formula_mcq_with_gate(question)
        provenance = answer.debug_meta["rule_outputs"][0]
        validation = validate_rule_provenance(
            provenance,
            question=question.to_dict(),
            pred_answer=answer.pred_answer,
            evidence_items=answer.evidence_items,
        )

        self.assertEqual(answer.pred_answer, "A")
        self.assertEqual(answer.token_usage.total_tokens, 0)
        self.assertEqual(answer.debug_meta["provenance_schema_version"], 1)
        self.assertTrue(validation["valid"], validation["reasons"])


if __name__ == "__main__":
    unittest.main()
