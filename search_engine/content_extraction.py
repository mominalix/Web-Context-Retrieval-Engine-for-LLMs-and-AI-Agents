"""High-signal main-content extraction without indexing hidden scripts."""

from __future__ import annotations

import json
from typing import Protocol

from .exceptions import DependencyError, ExtractionError
from .models import ExtractedContent


class ContentExtractor(Protocol):
    def extract(self, html: str, *, url: str | None = None) -> ExtractedContent: ...


class TrafilaturaExtractor:
    """Extract article-like main text and metadata with precision-oriented defaults."""

    def __init__(self, *, max_document_chars: int = 500_000, favor_precision: bool = True):
        if max_document_chars <= 0:
            raise ValueError("max_document_chars must be greater than zero")
        self.max_document_chars = max_document_chars
        self.favor_precision = favor_precision

    def extract(self, html: str, *, url: str | None = None) -> ExtractedContent:
        if not html.strip():
            raise ExtractionError("Cannot extract content from an empty page")
        try:
            from trafilatura import extract, html2txt
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise DependencyError(
                "Trafilatura is required. Install the project with `pip install -e .`."
            ) from exc

        try:
            serialized = extract(
                html,
                url=url,
                output_format="json",
                with_metadata=True,
                include_comments=False,
                include_tables=False,
                favor_precision=self.favor_precision,
                deduplicate=True,
            )
        except Exception as exc:
            raise ExtractionError(f"Content extraction failed: {exc}") from exc
        data: dict[str, object] = {}
        if serialized:
            try:
                data = json.loads(serialized)
            except json.JSONDecodeError as exc:
                raise ExtractionError("Extractor returned invalid JSON") from exc

        text = str(data.get("text") or data.get("raw_text") or "").strip()
        if not text:
            try:
                text = (html2txt(html) or "").strip()
            except Exception as exc:
                raise ExtractionError(f"Fallback content extraction failed: {exc}") from exc
        if not text:
            raise ExtractionError("No useful text could be extracted from the page")

        return ExtractedContent(
            text=text[: self.max_document_chars],
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            canonical_url=str(data.get("url") or url or "").strip() or None,
        )


def extract_content(html: str, query: str | None = None) -> dict[str, str]:
    """Backward-compatible dictionary wrapper; ``query`` is intentionally unused."""
    del query
    content = TrafilaturaExtractor().extract(html)
    return {
        "title": content.title,
        "meta_description": content.description,
        "text": content.text,
        "hidden_text": "",
    }
