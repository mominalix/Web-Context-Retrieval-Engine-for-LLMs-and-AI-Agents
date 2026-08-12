"""Small, serializable data contracts shared by the pipeline components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    text: str
    content_type: str
    status_code: int


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    text: str
    title: str = ""
    description: str = ""
    canonical_url: str | None = None

    def as_document_text(self) -> str:
        """Return useful fields once, ordered from concise to detailed."""
        parts: list[str] = []
        seen: set[str] = set()
        for value in (self.title, self.description, self.text):
            normalized = " ".join(value.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                parts.append(normalized)
                seen.add(key)
        return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    source: str | None = None
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source: str | None
    token_count: int
    document_id: int
    chunk_id: int
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int
    score: float
    source: str | None
    text: str
    token_count: int
    document_id: int
    chunk_id: int
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineFailure:
    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    hits: list[SearchHit]
    searched_results: int
    indexed_documents: int
    indexed_chunks: int
    context_tokens: int
    failures: list[PipelineFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
