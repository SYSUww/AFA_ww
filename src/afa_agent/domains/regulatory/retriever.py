from __future__ import annotations

from typing import Any

from afa_agent.bm25 import BM25Index
from afa_agent.models import RetrievalHit
from afa_agent.text_utils import build_zh_tokenizer, tokenize_zh


DOC_HINT_TERMS = [
    "受益所有人",
    "客户尽职调查",
    "反洗钱",
    "数据安全",
    "银行卡清算机构",
    "非银行支付机构",
    "上市公司信息披露",
    "上市公司治理",
    "公司章程",
    "半年度报告",
    "年度报告",
    "证券公司分类",
    "市场禁入",
    "行政处罚",
]


class RegulatoryRetriever:
    def __init__(self, units: list[dict[str, Any]], extra_terms: list[str] | None = None):
        self.units = units
        self.extra_terms = extra_terms or []
        self._tokenize_fn = build_zh_tokenizer(self.extra_terms) if self.extra_terms else tokenize_zh
        self.tokens = [self._tokenize(self._unit_text(unit)) for unit in units]
        self.bm25 = BM25Index(self.tokens)
        self.unit_lookup = {unit["unit_id"]: unit for unit in units}
        self.unit_positions = {unit["unit_id"]: idx for idx, unit in enumerate(units)}

    def search(
        self,
        doc_ids: list[str],
        query: str | list[str],
        top_k: int = 6,
        ensure_per_doc: bool = False,
        expand_neighbors: bool = True,
    ) -> list[RetrievalHit]:
        selected = self._rank(doc_ids, query, top_k=top_k, ensure_per_doc=ensure_per_doc)
        hits = [self._make_hit(idx, score) for idx, score in selected]
        if expand_neighbors:
            return self._expand_neighbors(hits, doc_ids, top_k=top_k)
        return hits

    def candidate_search(
        self,
        doc_ids: list[str],
        query: str | list[str],
        top_k: int = 12,
        ensure_per_doc: bool = False,
    ) -> list[RetrievalHit]:
        selected = self._rank(doc_ids, query, top_k=top_k, ensure_per_doc=ensure_per_doc)
        return [self._make_hit(idx, score) for idx, score in selected]

    def _rank(
        self,
        doc_ids: list[str],
        query: str | list[str],
        top_k: int,
        ensure_per_doc: bool,
    ) -> list[tuple[int, float]]:
        queries = [query] if isinstance(query, str) else query
        queries = [item.strip() for item in queries if item and item.strip()]
        if not queries:
            return []
        doc_filter = set(doc_ids)
        filtered_indices = [idx for idx, unit in enumerate(self.units) if unit["doc_id"] in doc_filter]
        combined: dict[int, dict[str, float]] = {}
        for item in queries:
            query_tokens = self._tokenize(item)
            for idx in filtered_indices:
                base_score = self.bm25.score(query_tokens, idx)
                if base_score <= 0:
                    continue
                score = self._apply_unit_weight(idx, base_score, item)
                row = combined.setdefault(idx, {"max": 0.0, "sum": 0.0, "hits": 0.0})
                row["max"] = max(row["max"], score)
                row["sum"] += score
                row["hits"] += 1
        scored = [
            (idx, row["max"] + row["sum"] * 0.25 + row["hits"] * 0.05)
            for idx, row in combined.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        if ensure_per_doc:
            return self._ensure_per_doc(scored, doc_ids, top_k)
        return scored[:top_k]

    def _apply_unit_weight(self, idx: int, score: float, query: str) -> float:
        unit = self.units[idx]
        unit_type = unit.get("unit_type", "")
        if unit_type == "preamble":
            if any(term in query for term in ["施行日期", "施行", "早于", "晚于", "废止", "起施行"]):
                score *= 1.2
            else:
                score *= 0.55
        if unit_type == "article_chunk":
            score *= 1.03
        doc_hint = f"{unit.get('doc_id', '')} {' '.join(unit.get('title_path', [])[:1])}"
        if any(term in query and term in doc_hint for term in DOC_HINT_TERMS):
            score *= 1.35
        return score

    def _make_hit(self, idx: int, score: float) -> RetrievalHit:
        unit = self.units[idx]
        metadata = dict(unit.get("metadata", {}))
        metadata.setdefault("unit_type", unit.get("unit_type", ""))
        return RetrievalHit(
            unit_id=unit["unit_id"],
            doc_id=unit["doc_id"],
            score=score,
            title_path=unit["title_path"],
            text=unit["text"],
            metadata=metadata,
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
                    neighbor_id = neighbor["unit_id"]
                    if neighbor_id not in expanded:
                        metadata = dict(neighbor.get("metadata", {}))
                        metadata.setdefault("unit_type", neighbor.get("unit_type", ""))
                        expanded[neighbor_id] = RetrievalHit(
                            unit_id=neighbor_id,
                            doc_id=neighbor["doc_id"],
                            score=max(hit.score - 0.1, 0.01),
                            title_path=neighbor["title_path"],
                            text=neighbor["text"],
                            metadata=metadata,
                        )
        ranked = sorted(expanded.values(), key=lambda item: item.score, reverse=True)
        selected: list[RetrievalHit] = []
        selected_ids: set[str] = set()
        original_ids = {hit.unit_id for hit in hits}
        for hit in hits:
            if hit.unit_id not in selected_ids:
                selected.append(hit)
                selected_ids.add(hit.unit_id)
            if len(selected) >= top_k:
                return selected
        for hit in ranked:
            if hit.unit_id in selected_ids or (hit.unit_id in original_ids):
                continue
            selected.append(hit)
            selected_ids.add(hit.unit_id)
            if len(selected) >= top_k:
                break
        return selected

    def _ensure_per_doc(self, scored: list[tuple[int, float]], doc_ids: list[str], top_k: int) -> list[tuple[int, float]]:
        selected: list[tuple[int, float]] = []
        seen_docs: set[str] = set()
        for idx, score in scored:
            doc_id = self.units[idx]["doc_id"]
            if doc_id in seen_docs:
                continue
            selected.append((idx, score))
            seen_docs.add(doc_id)
            if len(seen_docs) == len(set(doc_ids)):
                break
        selected_ids = {idx for idx, _ in selected}
        for idx, score in scored:
            if len(selected) >= top_k:
                break
            if idx not in selected_ids:
                selected.append((idx, score))
                selected_ids.add(idx)
        return selected[:top_k]

    @staticmethod
    def _unit_text(unit: dict[str, Any]) -> str:
        article_no = unit.get("metadata", {}).get("article_no", "")
        return f"{article_no}\n{unit.get('text', '')}".strip()

    def _tokenize(self, text: str) -> list[str]:
        return self._tokenize_fn(text)
