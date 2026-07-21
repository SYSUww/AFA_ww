from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from afa_agent.b_board.io import BQuestion, validate_b_answer, write_b_submission
from afa_agent.b_board.runner import BAnswerArtifact, _artifact_from_dict, _sum_tokens
from afa_agent.io_utils import ensure_dir, read_json, write_json


COMPOSITE_RUNNER = "b_actual_composite_v1"


def assemble_answer_run(
    *,
    questions: Sequence[BQuestion],
    source_run_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Build an immutable ordered answer run from one or more source runs.

    Later sources replace the same QID from earlier sources. Artifact coverage and
    submission validity are recorded separately so a baseline with format defects
    can still be evaluated without pretending it is submission-ready.
    """

    if not source_run_dirs:
        raise ValueError("source_run_dirs must not be empty")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"composite run directory is not empty: {destination}")
    question_by_qid = {item.qid: item for item in questions}
    if len(question_by_qid) != len(questions):
        raise ValueError("questions contain duplicate qids")

    artifacts: dict[str, BAnswerArtifact] = {}
    provenance: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for source_index, source_dir in enumerate(source_run_dirs):
        resolved = Path(source_dir).resolve()
        answers_path = resolved / "answers.json"
        rows = read_json(answers_path)
        if not isinstance(rows, list):
            raise ValueError(f"{answers_path}: expected a JSON array")
        source_qids: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{answers_path}: answer row must be an object")
            artifact = _artifact_from_dict(row)
            if artifact.qid not in question_by_qid:
                raise ValueError(f"unknown answer qid in source run: {artifact.qid}")
            source_qids.append(artifact.qid)
            artifacts[artifact.qid] = artifact
            provenance[artifact.qid] = {
                "source_index": source_index,
                "source_run_dir": str(resolved),
                "answers_sha256": _file_sha256(answers_path),
            }
        if len(source_qids) != len(set(source_qids)):
            raise ValueError(f"{answers_path}: duplicate qids")
        sources.append(
            {
                "source_index": source_index,
                "run_dir": str(resolved),
                "answer_count": len(source_qids),
                "answers_sha256": _file_sha256(answers_path),
            }
        )

    missing = [item.qid for item in questions if item.qid not in artifacts]
    if missing:
        raise ValueError(f"composite answer coverage is incomplete: {missing}")
    ordered = [artifacts[item.qid] for item in questions]
    invalid: list[dict[str, str]] = []
    for question, artifact in zip(questions, ordered):
        try:
            validate_b_answer(question, artifact.to_submission_answer())
        except ValueError as exc:
            invalid.append({"qid": question.qid, "error": str(exc)})

    ensure_dir(destination)
    write_json(destination / "answers.json", [item.to_dict() for item in ordered])
    write_json(destination / "answer_provenance.json", provenance)
    write_json(destination / "source_runs.json", sources)
    submission_path: str | None = None
    if not invalid:
        path = destination / "submit.csv"
        write_b_submission(
            path,
            questions,
            [item.to_submission_answer() for item in ordered],
        )
        submission_path = str(path)

    manifest = {
        "run_id": destination.name,
        "runner": COMPOSITE_RUNNER,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "artifact_complete": True,
        "expected_question_count": len(questions),
        "answered_question_count": len(ordered),
        "failed_qids": [],
        "token_usage": _sum_tokens(ordered),
        "submission_valid": not invalid,
        "submission_validation_failures": invalid,
        "submission_path": submission_path,
        "source_runs": sources,
    }
    write_json(destination / "run_manifest.json", manifest)
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
