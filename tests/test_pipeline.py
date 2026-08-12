from conftest import FakeEmbedder

from search_engine.config import Settings
from search_engine.indexer import Indexer
from search_engine.models import ExtractedContent, FetchedPage, SearchResult
from search_engine.pipeline import SearchPipeline
from search_engine.vector_store import NumpyVectorStore


class FakeProvider:
    def search(self, query: str, limit: int):
        assert query == "alpha"
        return [
            SearchResult("Good", "https://example.com/good", "alpha summary"),
            SearchResult("Fallback", "https://example.com/fail", "alpha fallback"),
        ][:limit]


class FakeFetcher:
    def fetch(self, url: str) -> FetchedPage:
        if url.endswith("/fail"):
            raise ValueError("simulated failure")
        return FetchedPage(url, "<p>alpha source text</p>", "text/html", 200)


class FakeExtractor:
    def extract(self, html: str, *, url: str | None = None) -> ExtractedContent:
        del html
        return ExtractedContent("alpha source text", title="Good", canonical_url=url)


def test_pipeline_is_injectable_and_reports_fallbacks() -> None:
    settings = Settings(
        search_results=2,
        top_k=2,
        fetch_k=3,
        chunk_size_tokens=8,
        chunk_overlap_tokens=0,
        max_context_tokens=10,
        score_threshold=-1,
    )
    embedder = FakeEmbedder()
    indexer = Indexer(
        embedder=embedder,
        vector_store=NumpyVectorStore(embedder.dimension),
        chunk_size_tokens=8,
        chunk_overlap_tokens=0,
    )
    pipeline = SearchPipeline(
        settings,
        search_provider=FakeProvider(),
        fetcher=FakeFetcher(),  # type: ignore[arg-type]
        extractor=FakeExtractor(),
        indexer=indexer,
    )

    response = pipeline.run("alpha")

    assert response.searched_results == 2
    assert response.indexed_documents == 2
    assert response.indexed_chunks == 2
    assert len(response.failures) == 1
    assert any(hit.metadata.get("snippet_only") for hit in response.hits)
    assert response.context_tokens <= settings.max_context_tokens
