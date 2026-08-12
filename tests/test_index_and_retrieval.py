from conftest import FakeEmbedder

from search_engine.indexer import Indexer
from search_engine.models import Document
from search_engine.retrieval import Retriever
from search_engine.vector_store import NumpyVectorStore


def make_indexer(*, chunk_size: int = 4) -> Indexer:
    embedder = FakeEmbedder()
    return Indexer(
        embedder=embedder,
        vector_store=NumpyVectorStore(embedder.dimension),
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=0,
    )


def test_indexer_deduplicates_chunks_and_resets() -> None:
    indexer = make_indexer(chunk_size=8)

    count = indexer.index_documents(
        [
            Document("alpha facts here", source="one"),
            Document("alpha facts here", source="duplicate"),
        ]
    )

    assert count == 1
    assert indexer.get_index_size() == 1
    assert indexer.meta_data[0]["source"] == "one"

    indexer.index_documents([Document("beta only", source="two")])
    assert indexer.get_index_size() == 1
    assert indexer.chunks[0].source == "two"


def test_retrieval_respects_budget_and_source_limit() -> None:
    indexer = make_indexer(chunk_size=3)
    indexer.index_documents(
        [
            Document("alpha alpha first alpha alpha second", source="same"),
            Document("alpha different source", source="other"),
            Document("beta irrelevant content", source="third"),
        ]
    )

    hits = Retriever(indexer).search(
        "alpha",
        top_k=3,
        fetch_k=4,
        max_context_tokens=4,
        score_threshold=-1,
        mmr_lambda=1,
        per_source_limit=1,
    )

    assert sum(hit.token_count for hit in hits) <= 4
    assert len([hit for hit in hits if hit.source == "same"]) <= 1
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_legacy_semantic_search_returns_dictionaries() -> None:
    indexer = make_indexer(chunk_size=8)
    indexer.index_documents([("alpha answer", "https://example.com")])

    hits = Retriever(indexer).semantic_search("alpha", top_k=1)

    assert hits[0]["source"] == "https://example.com"
    assert hits[0]["rank"] == 1
    assert hits[0]["token_count"] == 2
