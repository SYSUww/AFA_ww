from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from afa_agent.b_board.loop import MAX_COMPARABLE_ATTEMPTS, append_markdown_log
from afa_agent.experiment_registry import (
    DECISION_EXECUTE,
    DECISION_REFINE_EXISTING,
    DECISION_RETRY_AFTER_CONTEXT_CHANGE,
    ExperimentRegistry,
    HistoryDecision,
    build_candidate_fingerprint,
    sanitize_registry_payload,
)


EXECUTABLE_HISTORY_DECISIONS = {
    DECISION_EXECUTE,
    DECISION_REFINE_EXISTING,
    DECISION_RETRY_AFTER_CONTEXT_CHANGE,
}
REQUIRED_RESULT_FIELDS = (
    "approach",
    "effect",
    "failure_analysis",
    "next_step",
    "metrics",
)


class StaleHistoryReviewError(RuntimeError):
    """Raised when history changed after it was reviewed for an attempt."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    exists: bool
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AttemptHistoryReview:
    reviewed_at: str
    candidate_fingerprint: dict[str, Any]
    history_decision: HistoryDecision
    log_snapshot: FileSnapshot
    registry_snapshot: FileSnapshot
    related_log_sections: tuple[str, ...]
    max_comparable_attempts: int

    @property
    def executable(self) -> bool:
        return (
            self.history_decision.decision in EXECUTABLE_HISTORY_DECISIONS
            and self.history_decision.comparable_attempt_count < self.max_comparable_attempts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewed_at": self.reviewed_at,
            "candidate_fingerprint": self.candidate_fingerprint,
            "history_decision": self.history_decision.to_dict(),
            "log_snapshot": self.log_snapshot.to_dict(),
            "registry_snapshot": self.registry_snapshot.to_dict(),
            "related_log_sections": list(self.related_log_sections),
            "max_comparable_attempts": self.max_comparable_attempts,
            "executable": self.executable,
        }


class ExperimentJournal:
    """Enforce read-before-attempt and append-after-attempt experiment hygiene.

    ``review`` reads both the append-only Markdown log and JSONL registry. A
    result can only be written while both inputs still match those snapshots;
    otherwise the caller must review history again before continuing.
    """

    def __init__(
        self,
        *,
        registry: ExperimentRegistry,
        markdown_log_path: Path,
        max_comparable_attempts: int = MAX_COMPARABLE_ATTEMPTS,
    ) -> None:
        if max_comparable_attempts < 1:
            raise ValueError("max_comparable_attempts must be positive")
        self.registry = registry
        self.markdown_log_path = Path(markdown_log_path)
        self.max_comparable_attempts = max_comparable_attempts

    def review(self, candidate: Mapping[str, Any]) -> AttemptHistoryReview:
        cleaned = sanitize_registry_payload(candidate)
        if not isinstance(cleaned, Mapping):
            raise TypeError("experiment candidate must be a mapping")
        log_bytes = _read_bytes(self.markdown_log_path)
        log_text = log_bytes.decode("utf-8-sig", errors="replace")
        history = self.registry.decide(cleaned)
        registry_bytes = _read_bytes(self.registry.path)
        return AttemptHistoryReview(
            reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            candidate_fingerprint=build_candidate_fingerprint(cleaned),
            history_decision=history,
            log_snapshot=_snapshot_bytes(self.markdown_log_path, log_bytes),
            registry_snapshot=_snapshot_bytes(self.registry.path, registry_bytes),
            related_log_sections=_related_section_ids(log_text, cleaned),
            max_comparable_attempts=self.max_comparable_attempts,
        )

    def append_result(
        self,
        *,
        review: AttemptHistoryReview,
        candidate: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        cleaned_candidate = sanitize_registry_payload(candidate)
        cleaned_result = sanitize_registry_payload(result)
        if not isinstance(cleaned_candidate, Mapping) or not isinstance(cleaned_result, Mapping):
            raise TypeError("candidate and result must be mappings")
        missing = [field for field in REQUIRED_RESULT_FIELDS if field not in cleaned_result]
        if missing:
            raise ValueError(
                "experiment result is missing mandatory journal fields: " + ", ".join(missing)
            )
        if not str(cleaned_result.get("experiment_id") or "").strip():
            raise ValueError("experiment result requires experiment_id")
        if not str(cleaned_result.get("status") or "").strip():
            raise ValueError("experiment result requires status")

        fingerprint = build_candidate_fingerprint(cleaned_candidate)
        if fingerprint != review.candidate_fingerprint:
            raise StaleHistoryReviewError(
                "candidate changed after history review; review the log again"
            )
        self._require_current_snapshots(review)
        payload = {
            **dict(cleaned_candidate),
            **dict(cleaned_result),
            "history_review": review.to_dict(),
        }
        stored = self.registry.append(payload)
        append_markdown_log(self.markdown_log_path, stored)
        return stored

    def _require_current_snapshots(self, review: AttemptHistoryReview) -> None:
        current_log = _snapshot(self.markdown_log_path)
        current_registry = _snapshot(self.registry.path)
        if current_log != review.log_snapshot or current_registry != review.registry_snapshot:
            raise StaleHistoryReviewError(
                "experiment history changed after review; review the log and registry again"
            )


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def _snapshot(path: Path) -> FileSnapshot:
    return _snapshot_bytes(path, _read_bytes(path))


def _snapshot_bytes(path: Path, payload: bytes) -> FileSnapshot:
    return FileSnapshot(
        exists=path.exists(),
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _related_section_ids(log_text: str, candidate: Mapping[str, Any]) -> tuple[str, ...]:
    needles = {
        str(candidate.get("direction_id") or "").strip().lower(),
        str(candidate.get("pipeline_stage") or "").strip().lower(),
        str(candidate.get("root_cause_cluster") or "").strip().lower(),
        *(str(item).strip().lower() for item in candidate.get("target_qids", []) or []),
    }
    needles.discard("")
    if not needles:
        return ()
    sections = re.split(r"(?m)^##\s+", log_text)
    related: list[str] = []
    for section in sections[1:]:
        header, _, body = section.partition("\n")
        haystack = f"{header}\n{body}".lower()
        if any(needle in haystack for needle in needles):
            related.append(header.strip())
    return tuple(related[-50:])
