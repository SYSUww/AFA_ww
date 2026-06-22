from __future__ import annotations

from collections import defaultdict
from typing import Any

from afa_agent.bm25 import BM25Index
from afa_agent.models import RetrievalHit
from afa_agent.text_utils import tokenize_zh


class RegulatoryRetriever:
    def __init__(self, units: list[dict[str, Any]]):
        self.units = units
        self.tokens = [tokenize_zh(unit["text"]) for unit in units]
        self.bm25 = BM25Index(self.tokens)
        self.unit_lookup = {unit["unit_id"]: unit for unit in units}

    def search(self, doc_ids: list[str], query: str, top_k: int = 6) -> list[RetrievalHit]:
        query_tokens = tokenize_zh(query)
        filtered_indices = [idx for idx, unit in enumerate(self.units) if unit["doc_id"] in doc_ids]
        scored = []
        for idx in filtered_indices:
            score = self.bm25.score(query_tokens, idx)
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:top_k]
        hits = [self._make_hit(idx, score) for idx, score in selected]
        return self._expand_neighbors(hits, doc_ids)

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

    def _expand_neighbors(self, hits: list[RetrievalHit], doc_ids: list[str]) -> list[RetrievalHit]:
        by_doc: dict[str, list[RetrievalHit]] = defaultdict(list)
        for hit in hits:
            by_doc[hit.doc_id].append(hit)
        unit_positions = {unit["unit_id"]: idx for idx, unit in enumerate(self.units)}
        expanded: dict[str, RetrievalHit] = {hit.unit_id: hit for hit in hits}
        for hit in hits:
            idx = unit_positions[hit.unit_id]
            for neighbor_idx in [idx - 1, idx + 1]:
                if 0 <= neighbor_idx < len(self.units):
                    neighbor = self.units[neighbor_idx]
                    if neighbor["doc_id"] not in doc_ids:
                        continue
                    neighbor_id = neighbor["unit_id"]
                    if neighbor_id not in expanded:
                        expanded[neighbor_id] = RetrievalHit(
                            unit_id=neighbor_id,
                            doc_id=neighbor["doc_id"],
                            score=max(hit.score - 0.1, 0.01),
                            title_path=neighbor["title_path"],
                            text=neighbor["text"],
                            metadata=neighbor.get("metadata", {}),
                        )
        return sorted(expanded.values(), key=lambda item: item.score, reverse=True)[:6]
