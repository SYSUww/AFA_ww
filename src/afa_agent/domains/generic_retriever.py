from __future__ import annotations

from collections import defaultdict
from typing import Any

from afa_agent.bm25 import BM25Index
from afa_agent.models import RetrievalHit
from afa_agent.text_utils import tokenize_zh


class GenericBM25Retriever:
    def __init__(self, units: list[dict[str, Any]]):
        self.units = units
        self.tokens = [tokenize_zh(self._unit_text(unit)) for unit in units]
        self.bm25 = BM25Index(self.tokens)
        self.unit_positions = {unit["unit_id"]: idx for idx, unit in enumerate(units)}

    def search(
        self,
        doc_ids: list[str],
        query: str,
        top_k: int = 6,
        unit_type_boosts: dict[str, float] | None = None,
        ensure_per_doc: bool = False,
    ) -> list[RetrievalHit]:
        query_tokens = tokenize_zh(query)
        scored: list[tuple[int, float]] = []
        for idx, unit in enumerate(self.units):
            if unit["doc_id"] not in doc_ids:
                continue
            score = self.bm25.score(query_tokens, idx)
            if not score:
                continue
            if unit_type_boosts:
                score *= unit_type_boosts.get(unit.get("unit_type", ""), 1.0)
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        if ensure_per_doc:
            selected = self._ensure_per_doc(scored, doc_ids, top_k)
        else:
            selected = scored[:top_k]
        hits = [self._make_hit(idx, score) for idx, score in selected]
        return self._expand_neighbors(hits, doc_ids, top_k=top_k)

    def _ensure_per_doc(self, scored: list[tuple[int, float]], doc_ids: list[str], top_k: int) -> list[tuple[int, float]]:
        selected: list[tuple[int, float]] = []
        seen_docs: set[str] = set()
        for idx, score in scored:
            doc_id = self.units[idx]["doc_id"]
            if doc_id not in seen_docs:
                selected.append((idx, score))
                seen_docs.add(doc_id)
            if len(seen_docs) == len(set(doc_ids)):
                break
        for idx, score in scored:
            if len(selected) >= top_k:
                break
            if (idx, score) not in selected:
                selected.append((idx, score))
        return selected[:top_k]

    def _make_hit(self, idx: int, score: float) -> RetrievalHit:
        unit = self.units[idx]
        return RetrievalHit(
            unit_id=unit["unit_id"],
            doc_id=unit["doc_id"],
            score=score,
            title_path=unit["title_path"],
            text=unit["text"],
            metadata=unit.get("metadata", {}),
        )

    def _expand_neighbors(self, hits: list[RetrievalHit], doc_ids: list[str], top_k: int = 6) -> list[RetrievalHit]:
        expanded: dict[str, RetrievalHit] = {hit.unit_id: hit for hit in hits}
        for hit in hits:
            idx = self.unit_positions[hit.unit_id]
            for neighbor_idx in [idx - 1, idx + 1]:
                if 0 <= neighbor_idx < len(self.units):
                    neighbor = self.units[neighbor_idx]
                    if neighbor["doc_id"] not in doc_ids:
                        continue
                    if neighbor["unit_id"] not in expanded:
                        expanded[neighbor["unit_id"]] = RetrievalHit(
                            unit_id=neighbor["unit_id"],
                            doc_id=neighbor["doc_id"],
                            score=max(hit.score - 0.1, 0.01),
                            title_path=neighbor["title_path"],
                            text=neighbor["text"],
                            metadata=neighbor.get("metadata", {}),
                        )
        return sorted(expanded.values(), key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _unit_text(unit: dict[str, Any]) -> str:
        title = " ".join(unit.get("title_path", []))
        return f"{title}\n{unit.get('text', '')}".strip()
