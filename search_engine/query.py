"""Search-provider contracts and the default DDGS metasearch adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from .exceptions import DependencyError, SearchProviderError
from .models import SearchResult


class SearchProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> Sequence[SearchResult]: ...


class DDGSSearchProvider:
    """Metasearch implementation that avoids scraping fragile result-page HTML."""

    def __init__(
        self,
        *,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        backend: str = "auto",
        timeout: int = 10,
    ) -> None:
        self.region = region
        self.safesearch = safesearch
        self.backend = backend
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise DependencyError(
                "DDGS is required. Install the project with `pip install -e .`."
            ) from exc

        try:
            raw_results = DDGS(timeout=self.timeout).text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=limit,
                backend=self.backend,
            )
        except Exception as exc:
            raise SearchProviderError(f"Search provider failed: {exc}") from exc

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in raw_results or []:
            raw_url = str(item.get("href") or item.get("url") or "").strip()
            url = normalize_url(raw_url)
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                SearchResult(
                    title=str(item.get("title") or "").strip(),
                    url=url,
                    snippet=str(item.get("body") or item.get("snippet") or "").strip(),
                )
            )
            if len(results) >= limit:
                break
        return results


def normalize_url(url: str) -> str:
    """Normalize a web result for deduplication without changing its query."""
    if url.startswith("//"):
        url = f"https:{url}"
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname.casefold()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port and not (
        (parts.scheme.casefold() == "http" and port == 80)
        or (parts.scheme.casefold() == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme.casefold(), host, parts.path or "/", parts.query, ""))


def search_web(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Backward-compatible convenience wrapper around :class:`DDGSSearchProvider`."""
    return [
        {"title": result.title, "url": result.url, "snippet": result.snippet}
        for result in DDGSSearchProvider().search(query, num_results)
    ]
