"""Environment-backed configuration with no import-time side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, get_type_hints

from .exceptions import ConfigurationError


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Expected a boolean value, received {value!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime knobs. Every field can be overridden with ``SEMANTIC_SEARCH_*``."""

    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    vector_backend: str = "auto"
    embedding_batch_size: int = 32
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32
    search_results: int = 8
    top_k: int = 5
    fetch_k: int = 20
    max_context_tokens: int = 1_200
    score_threshold: float = 0.15
    mmr_lambda: float = 0.7
    per_source_limit: int = 2
    request_connect_timeout: float = 5.0
    request_read_timeout: float = 15.0
    request_retries: int = 2
    max_redirects: int = 4
    max_response_bytes: int = 3_000_000
    max_document_chars: int = 500_000
    allow_private_network: bool = False
    user_agent: str = "semantic-search-agents/0.2"
    search_region: str = "wt-wt"
    search_safesearch: str = "moderate"
    search_backend: str = "auto"
    use_search_snippets_on_failure: bool = True

    def __post_init__(self) -> None:
        positive_ints = {
            "embedding_batch_size": self.embedding_batch_size,
            "chunk_size_tokens": self.chunk_size_tokens,
            "search_results": self.search_results,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "max_context_tokens": self.max_context_tokens,
            "per_source_limit": self.per_source_limit,
            "max_response_bytes": self.max_response_bytes,
            "max_document_chars": self.max_document_chars,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ConfigurationError(f"{name} must be greater than zero")
        non_negative_ints = {
            "request_retries": self.request_retries,
            "max_redirects": self.max_redirects,
        }
        for name, value in non_negative_ints.items():
            if value < 0:
                raise ConfigurationError(f"{name} must be non-negative")
        if self.request_connect_timeout <= 0 or self.request_read_timeout <= 0:
            raise ConfigurationError("request timeouts must be greater than zero")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ConfigurationError(
                "chunk_overlap_tokens must be non-negative and smaller than chunk_size_tokens"
            )
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ConfigurationError("score_threshold must be between -1 and 1")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ConfigurationError("mmr_lambda must be between 0 and 1")
        if self.fetch_k < self.top_k:
            raise ConfigurationError("fetch_k must be greater than or equal to top_k")
        if self.vector_backend not in {"auto", "numpy", "faiss"}:
            raise ConfigurationError("vector_backend must be auto, numpy, or faiss")

    @classmethod
    def from_env(cls, prefix: str = "SEMANTIC_SEARCH_") -> Settings:
        """Load known fields from environment variables and ignore unrelated secrets."""
        hints = get_type_hints(cls)
        values: dict[str, Any] = {}
        for item in fields(cls):
            raw = os.getenv(f"{prefix}{item.name.upper()}")
            if raw is None:
                continue
            field_type = hints[item.name]
            try:
                if field_type is bool:
                    values[item.name] = _parse_bool(raw)
                elif field_type is int:
                    values[item.name] = int(raw)
                elif field_type is float:
                    values[item.name] = float(raw)
                else:
                    values[item.name] = raw
            except ConfigurationError:
                raise
            except ValueError as exc:
                raise ConfigurationError(
                    f"Invalid value for {prefix}{item.name.upper()}: {raw!r}"
                ) from exc
        return cls(**values)
