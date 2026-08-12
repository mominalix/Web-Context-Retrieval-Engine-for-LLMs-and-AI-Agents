"""Document chunking, embedding, deduplication, and vector indexing."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .chunking import TokenChunker
from .embeddings import EmbeddingProvider, FloatMatrix, SentenceTransformerEmbedder
from .exceptions import ConfigurationError
from .models import Chunk, Document
from .vector_store import VectorStore, create_vector_store


class Indexer:
    """Build an in-memory semantic index from typed or legacy documents."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        *,
        embedder: EmbeddingProvider | None = None,
        chunker: TokenChunker | None = None,
        vector_store: VectorStore | None = None,
        vector_backend: str = "auto",
        chunk_size_tokens: int = 256,
        chunk_overlap_tokens: int = 32,
        embedding_batch_size: int = 32,
    ) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder(
            model_name, batch_size=embedding_batch_size
        )
        max_input_tokens = getattr(self.embedder, "max_input_tokens", None)
        if max_input_tokens is not None and chunk_size_tokens > max_input_tokens:
            raise ConfigurationError(
                f"chunk_size_tokens ({chunk_size_tokens}) exceeds the embedding model's "
                f"input capacity ({max_input_tokens})"
            )
        self.chunker = chunker or TokenChunker(
            self.embedder.tokenizer,
            chunk_size=chunk_size_tokens,
            overlap=chunk_overlap_tokens,
        )
        self.vector_store = vector_store or create_vector_store(
            vector_backend, self.embedder.dimension
        )
        self.dimension = self.embedder.dimension
        self.chunks: list[Chunk] = []
        self.embeddings: FloatMatrix = np.empty((0, self.dimension), dtype=np.float32)
        self._chunk_hashes: set[bytes] = set()

    @property
    def index(self) -> VectorStore:
        """Compatibility alias for the configured vector store."""
        return self.vector_store

    @property
    def model(self):
        """Compatibility alias; returns the provider or its underlying model."""
        return getattr(self.embedder, "model", self.embedder)

    @property
    def meta_data(self) -> list[dict[str, Any]]:
        """Legacy dictionary view of indexed chunks."""
        return [
            {
                "source": chunk.source,
                "text": chunk.text,
                "title": chunk.title,
                "token_count": chunk.token_count,
                **chunk.metadata,
            }
            for chunk in self.chunks
        ]

    def reset(self) -> None:
        self.vector_store.reset()
        self.chunks.clear()
        self._chunk_hashes.clear()
        self.embeddings = np.empty((0, self.dimension), dtype=np.float32)

    def index_documents(self, docs: Sequence[Document | str | tuple[str, Any]]) -> int:
        """Replace the current index and return the number of unique chunks added."""
        self.reset()
        documents = [self._coerce_document(item) for item in docs]
        chunks = self._prepare_chunks(documents, document_id_offset=0)
        self._add_chunks(chunks)
        return len(chunks)

    def add_document(
        self,
        text: str | Document,
        meta: Any = None,
        *,
        source: str | None = None,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a document incrementally and return its number of unique chunks."""
        if isinstance(text, Document):
            document = text
        else:
            document = Document(
                text=text,
                source=source if source is not None else self._source_from_meta(meta),
                title=title,
                metadata=metadata or self._mapping_from_meta(meta),
            )
        chunks = self._prepare_chunks([document], document_id_offset=self._document_count())
        self._add_chunks(chunks)
        return len(chunks)

    def get_index_size(self) -> int:
        return self.vector_store.size

    def _document_count(self) -> int:
        return max((chunk.document_id for chunk in self.chunks), default=-1) + 1

    def _prepare_chunks(
        self, documents: Sequence[Document], *, document_id_offset: int
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen_hashes = set(self._chunk_hashes)
        for relative_id, document in enumerate(documents):
            for chunk_id, text_chunk in enumerate(self.chunker.split(document.text)):
                digest = self._chunk_digest(text_chunk.text)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                chunks.append(
                    Chunk(
                        text=text_chunk.text,
                        source=document.source,
                        token_count=text_chunk.token_count,
                        document_id=document_id_offset + relative_id,
                        chunk_id=chunk_id,
                        title=document.title,
                        metadata=dict(document.metadata),
                    )
                )
        return chunks

    def _add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.encode_documents([chunk.text for chunk in chunks])
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError(
                f"Embedding provider returned a matrix with an unexpected shape: {vectors.shape!r}"
            )
        self.vector_store.add(vectors)
        self.embeddings = np.concatenate((self.embeddings, vectors), axis=0)
        self.chunks.extend(chunks)
        self._chunk_hashes.update(self._chunk_digest(chunk.text) for chunk in chunks)

    @classmethod
    def _coerce_document(cls, item: Document | str | tuple[str, Any]) -> Document:
        if isinstance(item, Document):
            return item
        if isinstance(item, str):
            return Document(text=item)
        if isinstance(item, tuple) and len(item) == 2:
            text, meta = item
            if not isinstance(text, str):
                raise TypeError("Document tuple text must be a string")
            return Document(
                text=text,
                source=cls._source_from_meta(meta),
                metadata=cls._mapping_from_meta(meta),
            )
        raise TypeError("Documents must be Document, str, or (text, metadata) tuples")

    @staticmethod
    def _source_from_meta(meta: Any) -> str | None:
        if meta is None:
            return None
        if isinstance(meta, str):
            return meta
        if isinstance(meta, dict):
            source = meta.get("source") or meta.get("url")
            return str(source) if source is not None else None
        return str(meta)

    @staticmethod
    def _mapping_from_meta(meta: Any) -> dict[str, Any]:
        return dict(meta) if isinstance(meta, dict) else {}

    @staticmethod
    def _chunk_digest(text: str) -> bytes:
        return hashlib.blake2b(
            " ".join(text.casefold().split()).encode("utf-8"), digest_size=16
        ).digest()
