from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

from afa_agent.models import RetrievalHit


_TABLE_HEADER_TYPES = {"heading", "table_header", "table_context"}
_TABLE_ROW_TYPES = {"table_row", "metric_row"}
_CLAUSE_HEADER_TYPES = {"clause_heading", "heading"}
_CLAUSE_BODY_TYPES = {"clause", "clause_block", "paragraph"}
_DEFINITION_TYPES = {"definition", "definition_heading"}
_EXCEPTION_TYPES = {"exception", "exclusion", "clause_block"}


@dataclass(frozen=True, slots=True)
class _Component:
    unit_id: str
    doc_id: str
    score: float
    title_path: tuple[str, ...]
    text: str
    metadata: Mapping[str, Any]
    unit_type: str
    parent_unit_id: str | None
    chunk_index: int | None
    source_position: int | None
    original_rank: int


def compact_adjacent_evidence_hits(
    hits: Sequence[RetrievalHit | Mapping[str, Any]],
    *,
    units: Sequence[Mapping[str, Any]],
    top_k: int,
    max_block_chars: int = 3600,
    min_overlap_chars: int = 8,
    max_components_per_block: int = 2,
) -> list[dict[str, Any]]:
    """Merge retrieved chunks only when corpus order and section structure agree.

    The function is deliberately independent from question identifiers and
    historical labels.  It compacts the *current* ranked hits, so a merged block
    consumes one TopK slot while preserving auditable source provenance.
    """
    if top_k < 0:
        raise ValueError("top_k must not be negative")
    if max_block_chars <= 0:
        raise ValueError("max_block_chars must be positive")
    if min_overlap_chars <= 0:
        raise ValueError("min_overlap_chars must be positive")
    if max_components_per_block <= 0:
        raise ValueError("max_components_per_block must be positive")
    if not hits or top_k == 0:
        return []

    unit_by_id: dict[str, Mapping[str, Any]] = {}
    position_by_id: dict[str, int] = {}
    for position, unit in enumerate(units):
        unit_id = str(unit.get("unit_id", "")).strip()
        if not unit_id:
            raise ValueError(f"unit at position {position} has no unit_id")
        if unit_id in unit_by_id:
            raise ValueError(f"duplicate unit_id: {unit_id}")
        unit_by_id[unit_id] = unit
        position_by_id[unit_id] = position

    components: list[_Component] = []
    seen_hit_ids: set[str] = set()
    for original_rank, hit in enumerate(hits, start=1):
        payload = hit.to_dict() if isinstance(hit, RetrievalHit) else dict(hit)
        unit_id = str(payload.get("unit_id", "")).strip()
        if not unit_id or unit_id in seen_hit_ids:
            continue
        seen_hit_ids.add(unit_id)
        unit = unit_by_id.get(unit_id, {})
        hit_metadata = payload.get("metadata", {})
        metadata = dict(hit_metadata) if isinstance(hit_metadata, Mapping) else {}
        unit_metadata_value = unit.get("metadata", {})
        unit_metadata = dict(unit_metadata_value) if isinstance(unit_metadata_value, Mapping) else {}
        combined_metadata = {**unit_metadata, **metadata}
        chunk_index = _integer_or_none(
            combined_metadata.get("chunk_index", unit.get("chunk_index"))
        )
        components.append(
            _Component(
                unit_id=unit_id,
                doc_id=str(payload.get("doc_id", unit.get("doc_id", ""))),
                score=float(payload.get("score", 0.0)),
                title_path=tuple(
                    str(item)
                    for item in payload.get("title_path", unit.get("title_path", []))
                    if str(item).strip()
                ),
                text=str(payload.get("text", unit.get("text", ""))),
                metadata=combined_metadata,
                unit_type=str(
                    combined_metadata.get("unit_type", unit.get("unit_type", "paragraph"))
                ).strip(),
                parent_unit_id=_optional_string(
                    unit.get("parent_unit_id", combined_metadata.get("parent_unit_id"))
                ),
                chunk_index=chunk_index,
                source_position=position_by_id.get(unit_id),
                original_rank=original_rank,
            )
        )

    connected_groups = _connected_groups(
        components,
        min_overlap_chars=min_overlap_chars,
    )
    groups = [
        subgroup
        for group in connected_groups
        for subgroup in _split_group_around_best_rank(
            group,
            max_components=max_components_per_block,
        )
    ]
    blocks = [
        _build_block(
            group,
            max_block_chars=max_block_chars,
            min_overlap_chars=min_overlap_chars,
        )
        for group in groups
    ]
    blocks.sort(key=lambda block: (block["_best_original_rank"], -block["score"]))
    for block in blocks:
        block.pop("_best_original_rank", None)
    return blocks[:top_k]


def compact_retrieval_payload(
    retrieval: Mapping[str, Any],
    *,
    units: Sequence[Mapping[str, Any]],
    top_k: int,
    max_block_chars: int = 3600,
    min_overlap_chars: int = 8,
    max_components_per_block: int = 2,
) -> dict[str, Any]:
    """Return an auditable retrieval copy whose final TopK is block-based."""

    final_value = retrieval.get("final")
    if not isinstance(final_value, Mapping):
        raise ValueError("retrieval must contain a final ranking object")
    hits_value = final_value.get("hits")
    if not isinstance(hits_value, Sequence) or isinstance(hits_value, (str, bytes)):
        raise ValueError("retrieval final ranking must contain a hits array")
    input_hits = [dict(hit) for hit in hits_value if isinstance(hit, Mapping)]
    blocks = compact_adjacent_evidence_hits(
        input_hits,
        units=units,
        top_k=top_k,
        max_block_chars=max_block_chars,
        min_overlap_chars=min_overlap_chars,
        max_components_per_block=max_components_per_block,
    )
    output = dict(retrieval)
    output["final"] = {
        **dict(final_value),
        "hits": blocks,
        "ranked_hits": [
            {
                "rank": rank,
                "score": float(block["score"]),
                "doc_id": str(block["doc_id"]),
                "unit_id": str(block["unit_id"]),
            }
            for rank, block in enumerate(blocks, start=1)
        ],
    }
    output["compaction"] = {
        "mode": "adjacent_structural_v1",
        "top_k_counts_merged_blocks": True,
        "input_hit_count": len(input_hits),
        "output_block_count": len(blocks),
        "merged_block_count": sum(
            len(block.get("merged_from", [])) > 1 for block in blocks
        ),
        "merged_component_count": sum(
            len(block.get("merged_from", [])) for block in blocks
        ),
        "max_block_chars": max_block_chars,
        "min_overlap_chars": min_overlap_chars,
        "max_components_per_block": max_components_per_block,
        "input_hits": [
            {
                "rank": rank,
                "unit_id": str(hit.get("unit_id", "")),
                "doc_id": str(hit.get("doc_id", "")),
                "text_sha256": _sha256(str(hit.get("text", ""))),
            }
            for rank, hit in enumerate(input_hits, start=1)
        ],
    }
    return output


def _split_group_around_best_rank(
    group: Sequence[_Component],
    *,
    max_components: int,
) -> list[list[_Component]]:
    ordered = sorted(
        group,
        key=lambda item: (
            item.source_position if item.source_position is not None else 10**12,
            item.chunk_index if item.chunk_index is not None else 10**12,
            item.original_rank,
        ),
    )
    if len(ordered) <= max_components:
        return [ordered]
    anchor_index = min(
        range(len(ordered)),
        key=lambda index: ordered[index].original_rank,
    )
    if max_components == 1:
        start = anchor_index
    elif _prefers_following_context(ordered[anchor_index]) and anchor_index + 1 < len(ordered):
        start = anchor_index
    elif anchor_index > 0:
        start = anchor_index - (max_components - 1)
    else:
        start = 0
    start = max(0, min(start, len(ordered) - max_components))
    selected = ordered[start : start + max_components]
    result = [selected]
    if start:
        result.extend(
            _split_group_around_best_rank(
                ordered[:start],
                max_components=max_components,
            )
        )
    if start + max_components < len(ordered):
        result.extend(
            _split_group_around_best_rank(
                ordered[start + max_components :],
                max_components=max_components,
            )
        )
    return result


def _prefers_following_context(component: _Component) -> bool:
    return component.unit_type in (
        _TABLE_HEADER_TYPES | _CLAUSE_HEADER_TYPES | _DEFINITION_TYPES
    )


def _connected_groups(
    components: Sequence[_Component],
    *,
    min_overlap_chars: int,
) -> list[list[_Component]]:
    parents = list(range(len(components)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(components):
        for right_index in range(left_index + 1, len(components)):
            right = components[right_index]
            if _can_merge(left, right, min_overlap_chars=min_overlap_chars):
                union(left_index, right_index)

    grouped: dict[int, list[_Component]] = {}
    for index, component in enumerate(components):
        grouped.setdefault(find(index), []).append(component)
    return list(grouped.values())


def _can_merge(
    left: _Component,
    right: _Component,
    *,
    min_overlap_chars: int,
) -> bool:
    if not left.doc_id or left.doc_id != right.doc_id:
        return False

    source_adjacent = (
        left.source_position is not None
        and right.source_position is not None
        and abs(left.source_position - right.source_position) == 1
    )
    chunk_adjacent = (
        left.parent_unit_id is not None
        and left.parent_unit_id == right.parent_unit_id
        and left.chunk_index is not None
        and right.chunk_index is not None
        and abs(left.chunk_index - right.chunk_index) == 1
    )
    if not source_adjacent and not chunk_adjacent:
        return False

    earlier, later = _in_source_order(left, right)
    if chunk_adjacent:
        return True
    if _title_paths_compatible(earlier.title_path, later.title_path):
        return True
    if _longest_exact_overlap(
        earlier.text,
        later.text,
        min_overlap_chars=min_overlap_chars,
    ):
        return True
    return _is_supported_context_pair(earlier, later)


def _title_paths_compatible(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    normalized_left = tuple(item.strip() for item in left if item.strip())
    normalized_right = tuple(item.strip() for item in right if item.strip())
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    # A document root alone is too broad to establish section compatibility.
    return len(shorter) >= 2 and longer[: len(shorter)] == shorter


def _is_supported_context_pair(earlier: _Component, later: _Component) -> bool:
    if (
        earlier.unit_type in _TABLE_HEADER_TYPES
        and later.unit_type in _TABLE_ROW_TYPES
        and _shares_title_root(earlier.title_path, later.title_path)
    ):
        return True
    if (
        earlier.unit_type in _CLAUSE_HEADER_TYPES
        and later.unit_type in _CLAUSE_BODY_TYPES
        and _looks_like_clause_lead(earlier.text)
        and _shares_title_root(earlier.title_path, later.title_path)
    ):
        return True
    return (
        earlier.unit_type in _DEFINITION_TYPES
        and later.unit_type in _EXCEPTION_TYPES
        and _looks_like_exception(later.text, later.title_path)
        and _shares_title_root(earlier.title_path, later.title_path)
    )


def _shares_title_root(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(left and right and left[0].strip() == right[0].strip())


def _looks_like_clause_lead(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(("：", ":")) or any(
        term in stripped for term in ("下列约定", "按下列", "条件", "情形")
    )


def _looks_like_exception(text: str, title_path: tuple[str, ...]) -> bool:
    combined = " ".join((*title_path, text))
    return any(term in combined for term in ("但", "除外", "不包括", "不属于", "例外"))


def _in_source_order(left: _Component, right: _Component) -> tuple[_Component, _Component]:
    left_key = (
        left.source_position if left.source_position is not None else 10**12,
        left.chunk_index if left.chunk_index is not None else 10**12,
        left.original_rank,
    )
    right_key = (
        right.source_position if right.source_position is not None else 10**12,
        right.chunk_index if right.chunk_index is not None else 10**12,
        right.original_rank,
    )
    return (left, right) if left_key <= right_key else (right, left)


def _build_block(
    group: Sequence[_Component],
    *,
    max_block_chars: int,
    min_overlap_chars: int,
) -> dict[str, Any]:
    ordered = sorted(
        group,
        key=lambda item: (
            item.source_position if item.source_position is not None else 10**12,
            item.chunk_index if item.chunk_index is not None else 10**12,
            item.original_rank,
        ),
    )
    merged_text = ordered[0].text if ordered else ""
    overlap_chars = 0
    overlap_boundaries: list[dict[str, Any]] = []
    for component in ordered[1:]:
        overlap = _longest_exact_overlap(
            merged_text,
            component.text,
            min_overlap_chars=min_overlap_chars,
        )
        overlap_chars += overlap
        overlap_boundaries.append(
            {
                "left_unit_id": ordered[len(overlap_boundaries)].unit_id,
                "right_unit_id": component.unit_id,
                "overlap_chars": overlap,
            }
        )
        if overlap:
            merged_text += component.text[overlap:]
        elif merged_text and component.text:
            separator = "" if merged_text.endswith("\n") or component.text.startswith("\n") else "\n"
            merged_text += separator + component.text
        else:
            merged_text += component.text

    original_chars = len(merged_text)
    pre_truncation_sha256 = _sha256(merged_text)
    if original_chars > max_block_chars:
        merged_text = merged_text[:max_block_chars]
    truncation_provenance = {
        "applied": original_chars > max_block_chars,
        "strategy": "prefix_after_overlap_deduplication",
        "max_chars": max_block_chars,
        "original_chars": original_chars,
        "retained_chars": len(merged_text),
        "removed_chars": original_chars - len(merged_text),
    }

    merged_from = [
        {
            "unit_id": item.unit_id,
            "doc_id": item.doc_id,
            "original_rank": item.original_rank,
            "source_position": item.source_position,
            "title_path": list(item.title_path),
            "unit_type": item.unit_type,
            "chars": len(item.text),
            "sha256": _sha256(item.text),
            "input_truncation": _input_truncation(item.metadata),
        }
        for item in ordered
    ]
    component_hashes = [
        {"unit_id": item.unit_id, "sha256": _sha256(item.text)}
        for item in ordered
    ]
    common_title_path = _common_title_prefix([item.title_path for item in ordered])
    source_order = [item.unit_id for item in ordered]
    block_unit_id = (
        ordered[0].unit_id
        if len(ordered) == 1
        else f"merged::{ordered[0].doc_id}::{_sha256(chr(0).join(source_order))[:16]}"
    )
    block_metadata = (
        dict(ordered[0].metadata)
        if len(ordered) == 1
        else {"unit_type": "merged_evidence_block"}
    )
    block_metadata["unit_type"] = (
        ordered[0].unit_type if len(ordered) == 1 else "merged_evidence_block"
    )
    block_metadata["compacted"] = len(ordered) > 1
    return {
        "unit_id": block_unit_id,
        "doc_id": ordered[0].doc_id,
        "score": max(item.score for item in ordered),
        "title_path": common_title_path,
        "text": merged_text,
        "metadata": block_metadata,
        "merged_from": merged_from,
        "source_order": source_order,
        "overlap_chars": overlap_chars,
        "overlap_boundaries": overlap_boundaries,
        "component_hashes": component_hashes,
        "pre_truncation_sha256": pre_truncation_sha256,
        "text_sha256": _sha256(merged_text),
        "truncation_provenance": truncation_provenance,
        "_best_original_rank": min(item.original_rank for item in ordered),
    }


def _longest_exact_overlap(left: str, right: str, *, min_overlap_chars: int) -> int:
    upper_bound = min(len(left), len(right))
    for width in range(upper_bound, min_overlap_chars - 1, -1):
        if left[-width:] == right[:width]:
            return width
    return 0


def _common_title_prefix(paths: Sequence[tuple[str, ...]]) -> list[str]:
    if not paths:
        return []
    prefix = list(paths[0])
    for path in paths[1:]:
        width = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            width += 1
        prefix = prefix[:width]
    return prefix


def _input_truncation(metadata: Mapping[str, Any]) -> Any:
    for key in ("truncation_provenance", "truncation"):
        if key in metadata:
            return metadata[key]
    if "truncated" in metadata:
        return {"truncated": metadata["truncated"]}
    return None


def _integer_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
