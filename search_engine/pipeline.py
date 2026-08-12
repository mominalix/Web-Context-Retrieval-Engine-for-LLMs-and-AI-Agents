"""Composable end-to-end web search, extraction, indexing, and retrieval."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .config import Settings
from .content_extraction import ContentExtractor, TrafilaturaExtractor
from .exceptions import SearchEngineError
from .indexer import Indexer
from .models import Document, PipelineFailure, SearchResponse, SearchResult
from .query import DDGSSearchProvider, SearchProvider
from .retrieval import Retriever
from .scraper import HttpFetcher, PageFetcher

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Orchestrator whose provider, fetcher, extractor, and indexer are injectable."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search_provider: SearchProvider | None = None,
        fetcher: PageFetcher | None = None,
        extractor: ContentExtractor | None = None,
        indexer: Indexer | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.search_provider = search_provider or DDGSSearchProvider(
            region=self.settings.search_region,
            safesearch=self.settings.search_safesearch,
            backend=self.settings.search_backend,
            timeout=max(1, int(self.settings.request_read_timeout)),
        )
        self._owns_fetcher = fetcher is None
        self.fetcher = fetcher or HttpFetcher(
            user_agent=self.settings.user_agent,
            connect_timeout=self.settings.request_connect_timeout,
            read_timeout=self.settings.request_read_timeout,
            retries=self.settings.request_retries,
            max_redirects=self.settings.max_redirects,
            max_response_bytes=self.settings.max_response_bytes,
            allow_private_network=self.settings.allow_private_network,
        )
        self.extractor = extractor or TrafilaturaExtractor(
            max_document_chars=self.settings.max_document_chars
        )
        self.indexer = indexer or Indexer(
            self.settings.model_name,
            vector_backend=self.settings.vector_backend,
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            embedding_batch_size=self.settings.embedding_batch_size,
        )

    def close(self) -> None:
        if self._owns_fetcher:
            close = getattr(self.fetcher, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> SearchPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def run(self, query: str) -> SearchResponse:
        results = list(self.search_provider.search(query, self.settings.search_results))
        documents, failures = self._collect_documents(results)
        self.indexer.index_documents(documents)
        hits = Retriever(self.indexer).search(
            query,
            top_k=self.settings.top_k,
            fetch_k=self.settings.fetch_k,
            score_threshold=self.settings.score_threshold,
            max_context_tokens=self.settings.max_context_tokens,
            mmr_lambda=self.settings.mmr_lambda,
            per_source_limit=self.settings.per_source_limit,
        )
        return SearchResponse(
            query=query,
            hits=hits,
            searched_results=len(results),
            indexed_documents=len(documents),
            indexed_chunks=self.indexer.get_index_size(),
            context_tokens=sum(hit.token_count for hit in hits),
            failures=failures,
        )

    def _collect_documents(
        self, results: Sequence[SearchResult]
    ) -> tuple[list[Document], list[PipelineFailure]]:
        documents: list[Document] = []
        failures: list[PipelineFailure] = []
        for result in results:
            try:
                page = self.fetcher.fetch(result.url)
                content = self.extractor.extract(page.text, url=page.url)
                text = content.as_document_text()
                if not text:
                    raise ValueError("extracted document was empty")
                documents.append(
                    Document(
                        text=text,
                        source=content.canonical_url or page.url,
                        title=content.title or result.title,
                        metadata={"search_snippet": result.snippet},
                    )
                )
            except (SearchEngineError, ValueError) as exc:
                logger.warning("Skipping %s: %s", result.url, exc)
                failures.append(PipelineFailure(url=result.url, reason=str(exc)))
                if self.settings.use_search_snippets_on_failure and result.snippet:
                    documents.append(
                        Document(
                            text="\n\n".join(
                                part for part in (result.title, result.snippet) if part
                            ),
                            source=result.url,
                            title=result.title,
                            metadata={"snippet_only": True},
                        )
                    )
        return documents, failures
