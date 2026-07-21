from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REGISTRY_SCHEMA_VERSION = 1
CANDIDATE_FINGERPRINT_SCHEMA_VERSION = 1

DECISION_EXECUTE = "execute"
DECISION_SKIP_DUPLICATE = "skip_duplicate"
DECISION_REUSE_PROMOTED = "reuse_promoted"
DECISION_RETRY_AFTER_CONTEXT_CHANGE = "retry_after_context_change"
DECISION_REFINE_EXISTING = "refine_existing"

_PROMOTED_STATUSES = {"accepted", "effective", "promoted", "reuse_promoted"}
_REJECTED_STATUSES = {"exhausted", "rejected", "resolved", "skip_duplicate"}
_TECHNICAL_FAILURE_STATUSES = {
    "blocked_technical",
    "error",
    "failed",
    "incomplete",
    "interrupted",
}
_COMPLETED_ATTEMPT_STATUSES = {
    *_PROMOTED_STATUSES,
    *_REJECTED_STATUSES,
    "completed",
    "legacy_transferable",
}

_SENSITIVE_FIELD_NAMES = {
    "api_base",
    "api_key",
    "api_url",
    "base_url",
    "credential",
    "credentials",
    "llm_api_base",
    "llm_api_key",
    "model_endpoint",
    "openai_api_base",
    "openai_api_key",
    "password",
    "secret",
}

_IDENTITY_SET_FIELDS = ("domains", "question_types", "target_qids")
_CONTEXT_FIELDS = (
    "base_commit",
    "code_hash",
    "config_hash",
    "corpus_hash",
    "evaluator_fingerprint",
    "evaluator_model",
    "evaluator_version",
    "generator_fingerprint",
    "generator_model",
    "prompt_version",
    "question_hash",
)
_LEGACY_ID_PATTERN = re.compile(
    r"\b(?:attempt[_ -]?\d+|experiment[_ -]?[a-z0-9][a-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", flags=re.IGNORECASE)


class ExperimentRegistryError(RuntimeError):
    """Raised when the append-only experiment registry is malformed."""


@dataclass(frozen=True)
class HistoryDecision:
    decision: str
    candidate_fingerprint: dict[str, Any]
    comparable_attempt_count: int
    related_experiment_ids: tuple[str, ...]
    reason: str
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_fingerprint": self.candidate_fingerprint,
            "comparable_attempt_count": self.comparable_attempt_count,
            "related_experiment_ids": list(self.related_experiment_ids),
            "reason": self.reason,
            "similarity": self.similarity,
        }


def _is_sensitive_field(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.endswith("_sha256") or normalized.endswith("_hash"):
        return False
    return (
        normalized in _SENSITIVE_FIELD_NAMES
        or normalized.endswith("_api_key")
        or "password" in normalized
        or "credential" in normalized
        or "secret" in normalized
    )


def sanitize_registry_payload(value: Any) -> Any:
    """Return a JSON-safe copy with credentials and API endpoints removed.

    This module never reads process environment variables. Sanitization is also
    applied before fingerprinting so a secret cannot be recovered from a stored
    candidate signature.
    """

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_registry_payload(item)
            for key, item in value.items()
            if not _is_sensitive_field(key)
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_registry_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8", errors="replace")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_set(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return sorted({_normalize_text(item) for item in items if _normalize_text(item)})


def _candidate_source(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(candidate)
    nested = source.get("candidate")
    if isinstance(nested, Mapping):
        source = {**source, **nested}
    return sanitize_registry_payload(source)


def build_candidate_fingerprint(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic semantic and context fingerprints for a candidate."""

    source = _candidate_source(candidate)
    root_cause = source.get("root_cause_cluster", source.get("root_cause", ""))
    identity = {
        "pipeline_stage": _normalize_text(source.get("pipeline_stage", source.get("stage", ""))),
        "root_cause_cluster": _normalize_text(root_cause),
        "hypothesis": _normalize_text(source.get("hypothesis", "")),
        "change_vector": sanitize_registry_payload(source.get("change_vector", {})),
        **{field: _normalize_set(source.get(field)) for field in _IDENTITY_SET_FIELDS},
    }
    direction = {
        key: value
        for key, value in identity.items()
        if key not in {"hypothesis", "target_qids"}
    }
    context = {
        field: sanitize_registry_payload(source[field])
        for field in _CONTEXT_FIELDS
        if field in source and source[field] not in (None, "", [], {})
    }
    components = {"identity": identity, "context": context}
    return {
        "schema_version": CANDIDATE_FINGERPRINT_SCHEMA_VERSION,
        "sha256": _sha256(components),
        "semantic_sha256": _sha256(identity),
        "direction_sha256": _sha256(direction),
        "context_sha256": _sha256(context),
        "components": components,
    }


def build_candidate_signature(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility name for callers that describe the fingerprint as a signature."""

    return build_candidate_fingerprint(candidate)


def _flatten_tokens(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {token.lower() for token in _WORD_PATTERN.findall(text)}


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def candidate_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return a transparent 0..1 similarity score for two fingerprints."""

    left_identity = left.get("components", {}).get("identity", {})
    right_identity = right.get("components", {}).get("identity", {})
    if not isinstance(left_identity, Mapping) or not isinstance(right_identity, Mapping):
        return 0.0

    score = 0.0
    if left_identity.get("pipeline_stage") == right_identity.get("pipeline_stage"):
        score += 0.25
    if left_identity.get("root_cause_cluster") == right_identity.get("root_cause_cluster"):
        score += 0.25
    score += 0.10 * _jaccard(left_identity.get("domains", []), right_identity.get("domains", []))
    score += 0.05 * _jaccard(
        left_identity.get("question_types", []), right_identity.get("question_types", [])
    )
    score += 0.30 * _jaccard(
        _flatten_tokens(left_identity.get("change_vector", {})),
        _flatten_tokens(right_identity.get("change_vector", {})),
    )
    score += 0.05 * _jaccard(
        _flatten_tokens(left_identity.get("hypothesis", "")),
        _flatten_tokens(right_identity.get("hypothesis", "")),
    )
    left_stage = left_identity.get("pipeline_stage")
    right_stage = right_identity.get("pipeline_stage")
    if left_stage and right_stage and left_stage != right_stage:
        score = min(score, 0.65)
    return round(min(1.0, score), 6)


def _record_status(record: Mapping[str, Any]) -> str:
    promotion = _normalize_text(record.get("promotion_result", ""))
    if promotion in _PROMOTED_STATUSES or promotion in _REJECTED_STATUSES:
        return promotion
    return _normalize_text(record.get("status", promotion))


def _experiment_id(record: Mapping[str, Any]) -> str:
    return str(record.get("experiment_id", record.get("candidate_id", "unknown")))


def _has_material_delta(candidate: Mapping[str, Any]) -> bool:
    delta = sanitize_registry_payload(candidate.get("material_delta"))
    return delta not in (None, "", [], {})


def _context_changed(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_context = left.get("components", {}).get("context", {})
    right_context = right.get("components", {}).get("context", {})
    return bool(left_context or right_context) and left.get("context_sha256") != right.get("context_sha256")


class ExperimentRegistry:
    """Append-only JSONL experiment history with deterministic duplicate checks."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExperimentRegistryError(
                        f"Malformed experiment registry line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ExperimentRegistryError(
                        f"Malformed experiment registry line {line_number}: expected object"
                    )
                rows.append(row)
        return rows

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = sanitize_registry_payload(record)
        if not isinstance(cleaned, dict):
            raise TypeError("Experiment registry records must be mappings")
        if not cleaned.get("experiment_id"):
            raise ValueError("Experiment registry record requires experiment_id")
        cleaned.setdefault("registry_schema_version", REGISTRY_SCHEMA_VERSION)
        cleaned.setdefault("recorded_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        cleaned.setdefault("candidate_fingerprint", build_candidate_fingerprint(cleaned))

        serialized = json.dumps(cleaned, ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            payload = serialized.encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        return cleaned

    def count_comparable_attempts(
        self,
        candidate: Mapping[str, Any],
        *,
        similarity_threshold: float = 0.70,
    ) -> int:
        fingerprint = build_candidate_fingerprint(candidate)
        direction_id = _normalize_text(candidate.get("direction_id", ""))
        experiment_ids: set[str] = set()
        for record in self.read_all():
            if _record_status(record) not in _COMPLETED_ATTEMPT_STATUSES:
                continue
            existing = record.get("candidate_fingerprint")
            if not isinstance(existing, Mapping):
                continue
            same_direction_id = bool(direction_id) and direction_id == _normalize_text(record.get("direction_id", ""))
            structurally_comparable = (
                existing.get("direction_sha256") == fingerprint["direction_sha256"]
                or candidate_similarity(fingerprint, existing) >= similarity_threshold
            )
            if same_direction_id or structurally_comparable:
                experiment_ids.add(_experiment_id(record))
        return len(experiment_ids)

    def decide(
        self,
        candidate: Mapping[str, Any],
        *,
        similarity_threshold: float = 0.70,
    ) -> HistoryDecision:
        fingerprint = build_candidate_fingerprint(candidate)
        records = [row for row in self.read_all() if isinstance(row.get("candidate_fingerprint"), Mapping)]
        comparable_count = self.count_comparable_attempts(
            candidate,
            similarity_threshold=similarity_threshold,
        )

        exact = [
            row
            for row in records
            if row["candidate_fingerprint"].get("sha256") == fingerprint["sha256"]
        ]
        if exact:
            related = tuple(_experiment_id(row) for row in exact)
            if any(_record_status(row) in _PROMOTED_STATUSES for row in exact):
                return HistoryDecision(
                    DECISION_REUSE_PROMOTED,
                    fingerprint,
                    comparable_count,
                    related,
                    "An identical candidate was already promoted in the same context",
                    1.0,
                )
            if any(_record_status(row) in _TECHNICAL_FAILURE_STATUSES for row in exact):
                return HistoryDecision(
                    DECISION_EXECUTE,
                    fingerprint,
                    comparable_count,
                    related,
                    "The identical prior run ended in a technical failure and did not consume a comparable attempt",
                    1.0,
                )
            return HistoryDecision(
                DECISION_SKIP_DUPLICATE,
                fingerprint,
                comparable_count,
                related,
                "An identical candidate was already completed in the same context",
                1.0,
            )

        same_semantics = [
            row
            for row in records
            if row["candidate_fingerprint"].get("semantic_sha256") == fingerprint["semantic_sha256"]
            and _context_changed(fingerprint, row["candidate_fingerprint"])
        ]
        if same_semantics:
            return HistoryDecision(
                DECISION_RETRY_AFTER_CONTEXT_CHANGE,
                fingerprint,
                comparable_count,
                tuple(_experiment_id(row) for row in same_semantics),
                "The same semantic candidate was evaluated under a different model, data, code, or evaluator context",
                1.0,
            )

        direction_id = _normalize_text(candidate.get("direction_id", ""))
        scored = sorted(
            (
                (
                    max(
                        candidate_similarity(fingerprint, row["candidate_fingerprint"]),
                        similarity_threshold,
                    )
                    if direction_id
                    and direction_id == _normalize_text(row.get("direction_id", ""))
                    else candidate_similarity(fingerprint, row["candidate_fingerprint"]),
                    row,
                )
                for row in records
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        similar = [(score, row) for score, row in scored if score >= similarity_threshold]
        if not similar:
            return HistoryDecision(
                DECISION_EXECUTE,
                fingerprint,
                comparable_count,
                (),
                "No comparable historical experiment was found",
                None,
            )

        best_score, best = similar[0]
        related = tuple(_experiment_id(row) for _, row in similar)
        status = _record_status(best)
        if _has_material_delta(candidate):
            return HistoryDecision(
                DECISION_REFINE_EXISTING,
                fingerprint,
                comparable_count,
                related,
                "A similar experiment exists, but the candidate declares a material implementation delta",
                best_score,
            )
        if status in _PROMOTED_STATUSES:
            return HistoryDecision(
                DECISION_REUSE_PROMOTED,
                fingerprint,
                comparable_count,
                related,
                "A similar candidate was already promoted and no material delta was declared",
                best_score,
            )
        if status in _REJECTED_STATUSES or status in _COMPLETED_ATTEMPT_STATUSES:
            return HistoryDecision(
                DECISION_SKIP_DUPLICATE,
                fingerprint,
                comparable_count,
                related,
                "A similar candidate was already completed and no material delta was declared",
                best_score,
            )
        if status in _TECHNICAL_FAILURE_STATUSES and _context_changed(fingerprint, best["candidate_fingerprint"]):
            return HistoryDecision(
                DECISION_RETRY_AFTER_CONTEXT_CHANGE,
                fingerprint,
                comparable_count,
                related,
                "A similar technical attempt can be retried because its execution context changed",
                best_score,
            )
        return HistoryDecision(
            DECISION_EXECUTE,
            fingerprint,
            comparable_count,
            related,
            "Related history is inconclusive, so executing will add information",
            best_score,
        )

    def import_legacy(
        self,
        *,
        markdown_paths: Sequence[Path] = (),
        manifest_paths: Sequence[Path] = (),
    ) -> list[dict[str, Any]]:
        """Import structural legacy metadata without copying source prose or secrets."""

        records: list[dict[str, Any]] = []
        for path in markdown_paths:
            records.extend(_legacy_markdown_records(Path(path)))
        for path in manifest_paths:
            records.extend(_legacy_manifest_records(Path(path)))

        existing_keys = {
            (row.get("legacy_source_sha256"), row.get("legacy_identifier"))
            for row in self.read_all()
            if row.get("legacy_source_sha256")
        }
        imported: list[dict[str, Any]] = []
        for record in records:
            key = (record.get("legacy_source_sha256"), record.get("legacy_identifier"))
            if key in existing_keys:
                continue
            imported.append(self.append(record))
            existing_keys.add(key)
        return imported


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_pipeline_stage(text: str) -> str:
    lowered = text.lower()
    stage_keywords = (
        ("preprocessing", ("preprocess", "预处理", "ocr")),
        ("chunking", ("chunk", "分块", "切块")),
        ("retrieval", ("retriev", "bm25", "检索", "召回", "locator")),
        ("prompt", ("prompt", "提示词")),
        ("calculation", ("calculation", "计算", "decimal")),
        ("evaluation", ("evaluation", "judge", "评测")),
    )
    for stage, keywords in stage_keywords:
        if any(keyword in lowered for keyword in keywords):
            return stage
    return "legacy_unknown"


def _legacy_markdown_records(path: Path) -> list[dict[str, Any]]:
    source_hash = _file_sha256(path)
    identifiers: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            for match in _LEGACY_ID_PATTERN.finditer(line):
                identifier = _normalize_text(match.group(0)).replace(" ", "_").replace("-", "_")
                identifiers.setdefault(identifier, _infer_pipeline_stage(line))
    if not identifiers:
        identifiers[f"source_{source_hash[:12]}"] = "legacy_unknown"
    return [
        {
            "experiment_id": f"legacy:{path.name}:{identifier}",
            "legacy_identifier": identifier,
            "legacy_source_kind": "markdown",
            "legacy_source_path": str(path),
            "legacy_source_sha256": source_hash,
            "pipeline_stage": stage,
            "status": "legacy_transferable",
        }
        for identifier, stage in sorted(identifiers.items())
    ]


def _legacy_manifest_records(path: Path) -> list[dict[str, Any]]:
    source_hash = _file_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ExperimentRegistryError(f"Legacy manifest must contain an object: {path}")

    experiment_id = str(payload.get("experiment_id", f"manifest_{source_hash[:12]}"))
    stage = str(payload.get("pipeline_stage", payload.get("stage", "legacy_unknown")))
    status = str(payload.get("status", "legacy_transferable"))
    candidate_ids = payload.get("candidate_ids", [])
    if not isinstance(candidate_ids, list) or not candidate_ids:
        candidate_ids = [payload.get("candidate_id", experiment_id)]

    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        identifier = str(candidate_id)
        rows.append(
            {
                "experiment_id": f"legacy:{experiment_id}:{identifier}",
                "direction_id": identifier,
                "legacy_identifier": identifier,
                "legacy_source_kind": "manifest",
                "legacy_source_path": str(path),
                "legacy_source_sha256": source_hash,
                "pipeline_stage": stage,
                "status": status,
            }
        )
    return rows
