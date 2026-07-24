from __future__ import annotations

from collections.abc import Sequence

from afa_agent.b_board.io import BQuestion
from afa_agent.b_board.runner import BAnswerArtifact


def merge_candidate_artifacts(
    *,
    questions: Sequence[BQuestion],
    base_artifacts: Sequence[BAnswerArtifact],
    overlay_artifact_groups: Sequence[Sequence[BAnswerArtifact]],
) -> tuple[list[BAnswerArtifact], list[str]]:
    """Overlay disjoint partial runs onto one complete, ordered base run."""

    expected_qids = [item.qid for item in questions]
    expected_set = set(expected_qids)
    if len(expected_set) != len(expected_qids):
        raise ValueError("question qids are not unique")

    base_by_qid = _unique_artifacts(base_artifacts, label="base")
    if set(base_by_qid) != expected_set:
        missing = sorted(expected_set - set(base_by_qid))
        extra = sorted(set(base_by_qid) - expected_set)
        raise ValueError(
            "base artifact coverage does not match questions: "
            f"missing={missing}, extra={extra}"
        )

    merged = dict(base_by_qid)
    overlay_owner: dict[str, int] = {}
    for group_index, group in enumerate(overlay_artifact_groups, start=1):
        group_by_qid = _unique_artifacts(group, label=f"overlay {group_index}")
        unknown = sorted(set(group_by_qid) - expected_set)
        if unknown:
            raise ValueError(
                f"overlay {group_index} has unknown qids: {unknown}"
            )
        for qid, artifact in group_by_qid.items():
            prior_owner = overlay_owner.get(qid)
            if prior_owner is not None:
                raise ValueError(
                    f"duplicate overlay qid {qid}: "
                    f"overlay {prior_owner} and overlay {group_index}"
                )
            overlay_owner[qid] = group_index
            merged[qid] = artifact

    return [merged[qid] for qid in expected_qids], [
        qid for qid in expected_qids if qid in overlay_owner
    ]


def _unique_artifacts(
    artifacts: Sequence[BAnswerArtifact],
    *,
    label: str,
) -> dict[str, BAnswerArtifact]:
    by_qid: dict[str, BAnswerArtifact] = {}
    for artifact in artifacts:
        if artifact.qid in by_qid:
            raise ValueError(f"{label} has duplicate qid {artifact.qid}")
        by_qid[artifact.qid] = artifact
    return by_qid
