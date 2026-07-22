from __future__ import annotations

from typing import Any, Mapping, Sequence

from afa_agent.b_board.calculation import (
    CalculationExecutor,
    CalculationPlanError,
    _answers_differ_only_in_format,
)
from afa_agent.b_board.io import (
    BQuestion,
    infer_percent_suffix_requirement,
    infer_requested_decimal_places,
    validate_b_answer,
)
from afa_agent.b_board.runner import BAnswerArtifact


def revalidate_calculation_artifact(
    *,
    question: BQuestion,
    artifact: BAnswerArtifact,
    index_units: Sequence[Mapping[str, Any]],
    supporting_evidence_ids: Sequence[str] = (),
    executor: CalculationExecutor | None = None,
    allow_format_change: bool = False,
) -> BAnswerArtifact:
    """Return a grounded replay, optionally migrating only its answer format."""

    if question.answer_format != "calculation":
        raise ValueError(f"{question.qid}: only calculation artifacts can be revalidated")
    if artifact.qid != question.qid:
        raise ValueError(f"artifact qid {artifact.qid!r} does not match {question.qid!r}")
    if not artifact.calculation_trace:
        raise CalculationPlanError(f"{question.qid}: source artifact has no calculation trace")

    evidence_by_id = {
        str(item.get("unit_id", "")): dict(item)
        for item in artifact.evidence_items
        if str(item.get("unit_id", ""))
    }
    unit_by_id = {
        str(item.get("unit_id", "")): dict(item)
        for item in index_units
        if str(item.get("unit_id", ""))
    }
    requested_support = list(dict.fromkeys(str(value) for value in supporting_evidence_ids))
    missing = [value for value in requested_support if value not in evidence_by_id and value not in unit_by_id]
    if missing:
        raise CalculationPlanError(
            f"{question.qid}: supporting evidence IDs are absent from the index: {missing}"
        )
    for evidence_id in requested_support:
        if evidence_id in evidence_by_id:
            continue
        unit = unit_by_id[evidence_id]
        evidence_by_id[evidence_id] = {
            "unit_id": evidence_id,
            "doc_id": str(unit.get("doc_id", "")),
            "title_path": list(unit.get("title_path", [])),
            "text": str(unit.get("text", "")),
            "score": 1000.0,
            "metadata": {
                **dict(unit.get("metadata", {})),
                "unit_type": str(unit.get("unit_type", "")),
                "retrieval_source": "incumbent_trace_literal_revalidation_a5",
            },
        }

    replay = (executor or CalculationExecutor()).replay_legacy_trace(
        artifact.calculation_trace,
        expected_slots=question.answer_slots,
        evidence_text_by_id={
            evidence_id: str(item.get("text", ""))
            for evidence_id, item in evidence_by_id.items()
        },
        expected_slot_templates=question.answer_slot_templates,
        expected_numeric_decimal_places=infer_requested_decimal_places(question.question),
        expected_percent_suffixes=tuple(
            infer_percent_suffix_requirement(
                question.question,
                slot_index=index,
                slot_count=question.answer_slots,
            )
            for index in range(1, question.answer_slots + 1)
        ),
        preserve_incumbent_answer=not allow_format_change,
    )
    answer_preserved = list(replay.answer_parts) == artifact.answer_parts
    if not allow_format_change and not answer_preserved:
        raise CalculationPlanError(
            f"{question.qid}: revalidation changed answer {artifact.answer_parts} to {list(replay.answer_parts)}"
        )
    if allow_format_change and not _answers_differ_only_in_format(
        artifact.answer_parts, replay.answer_parts
    ):
        raise CalculationPlanError(
            f"{question.qid}: format migration changed answer value "
            f"{artifact.answer_parts} to {list(replay.answer_parts)}"
        )
    used_ids = list(dict.fromkeys([*replay.used_evidence_ids, *requested_support]))
    selected_evidence = [evidence_by_id[evidence_id] for evidence_id in used_ids]
    result = BAnswerArtifact(
        qid=artifact.qid,
        domain=artifact.domain,
        answer_format=artifact.answer_format,
        answer_slot_count=artifact.answer_slot_count,
        answer_parts=list(replay.answer_parts),
        used_evidence_ids=used_ids,
        evidence_items=selected_evidence,
        decision_summary=(
            "Deterministically replayed the incumbent calculation trace against literal "
            "evidence and applied the question-first, README-second answer-format contract."
        ),
        decision_trace={
            "source": "incumbent_trace_literal_revalidation_a5",
            "format_forced": False,
            "format_migrated": not answer_preserved,
            "answer_preserved": answer_preserved,
            "format_change_allowed": allow_format_change,
            "supporting_evidence_ids": requested_support,
        },
        calculation_trace=replay.trace,
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        locator={
            **artifact.locator,
            "calculation_revalidation": {
                "source": "incumbent_trace_literal_revalidation_a5",
                "answer_preserved": answer_preserved,
                "format_change_allowed": allow_format_change,
                "supporting_evidence_ids": requested_support,
            },
        },
    )
    validate_b_answer(question, result.to_submission_answer())
    return result
