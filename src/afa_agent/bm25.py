from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in tokenized_docs]
        self.corpus_size = len(tokenized_docs)
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 0.0
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_counts: Counter[str] = Counter()
        for doc in self.tokenized_docs:
            for token in set(doc):
                doc_counts[token] += 1
        idf: dict[str, float] = {}
        for token, freq in doc_counts.items():
            idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        frequencies = self.doc_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx] or 1
        for token in query_tokens:
            tf = frequencies.get(token, 0)
            if not tf:
                continue
            token_idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
            score += token_idf * numerator / denominator
        return score

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scored = [(idx, self.score(query_tokens, idx)) for idx in range(self.corpus_size)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in scored[:k] if score > 0]
