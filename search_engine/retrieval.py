"""Budgeted semantic retrieval with duplicate/source-aware MMR selection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .indexer import Indexer
from .models import Chunk, SearchHit


class Retriever:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        fetch_k: int | None = None,
        score_threshold: float = -1.0,
        max_context_tokens: int | None = None,
        mmr_lambda: float = 0.7,
        per_source_limit: int | None = None,
    ) -> list[dict[str, object]]:
        """Return JSON-friendly hits while keeping the historical public API."""
        return [
            hit.to_dict()
            for hit in self.search(
                query,
                top_k=top_k,
                fetch_k=fetch_k,
                score_threshold=score_threshold,
                max_context_tokens=max_context_tokens,
                mmr_lambda=mmr_lambda,
                per_source_limit=per_source_limit,
            )
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int | None = None,
        score_threshold: float = -1.0,
        max_context_tokens: int | None = None,
        mmr_lambda: float = 0.7,
        per_source_limit: int | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero")
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if self.indexer.get_index_size() == 0:
            return []

        candidate_count = min(fetch_k or max(top_k * 4, top_k), self.indexer.get_index_size())
        query_vector = self.indexer.embedder.encode_queries([query])[0]
        scores, indices = self.indexer.vector_store.search(query_vector, candidate_count)
        candidates = [
            (int(index), float(score))
            for score, index in zip(scores, indices, strict=True)
            if index >= 0 and score >= score_threshold
        ]
        selected = self._select_mmr(candidates, candidate_count, mmr_lambda)

        hits: list[SearchHit] = []
        source_counts: Counter[str] = Counter()
        remaining_tokens = max_context_tokens
        for index, score in selected:
            chunk = self.indexer.chunks[index]
            source_key = chunk.source or f"document:{chunk.document_id}"
            if per_source_limit is not None and source_counts[source_key] >= per_source_limit:
                continue
            selected_chunk = chunk
            if remaining_tokens is not None and chunk.token_count > remaining_tokens:
                truncated = self.indexer.chunker.truncate(chunk.text, remaining_tokens)
                if not truncated.text:
                    break
                selected_chunk = Chunk(
                    text=truncated.text,
                    source=chunk.source,
                    token_count=truncated.token_count,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    metadata=chunk.metadata,
                )
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=score,
                    source=selected_chunk.source,
                    text=selected_chunk.text,
                    token_count=selected_chunk.token_count,
                    document_id=selected_chunk.document_id,
                    chunk_id=selected_chunk.chunk_id,
                    title=selected_chunk.title,
                    metadata=selected_chunk.metadata,
                )
            )
            source_counts[source_key] += 1
            if remaining_tokens is not None:
                remaining_tokens -= selected_chunk.token_count
                if remaining_tokens <= 0:
                    break
            if len(hits) >= top_k:
                break
        return hits

    def _select_mmr(
        self, candidates: list[tuple[int, float]], top_k: int, mmr_lambda: float
    ) -> list[tuple[int, float]]:
        if not candidates or mmr_lambda == 1.0:
            return candidates[:top_k]
        remaining = list(candidates)
        selected: list[tuple[int, float]] = []
        while remaining and len(selected) < top_k:
            if not selected:
                best = remaining[0]
            else:
                selected_vectors = self.indexer.embeddings[[item[0] for item in selected]]

                def mmr(item: tuple[int, float], reference_vectors=selected_vectors) -> float:
                    vector = self.indexer.embeddings[item[0]]
                    redundancy = float(np.max(reference_vectors @ vector))
                    return mmr_lambda * item[1] - (1.0 - mmr_lambda) * redundancy

                best = max(remaining, key=mmr)
            selected.append(best)
            remaining.remove(best)
        return selected
