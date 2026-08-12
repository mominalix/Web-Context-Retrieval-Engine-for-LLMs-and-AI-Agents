"""Replaceable exact-vector indexes with NumPy and FAISS implementations."""

from __future__ import annotations

import importlib.util
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .embeddings import FloatMatrix
from .exceptions import ConfigurationError, DependencyError


class VectorStore(Protocol):
    @property
    def size(self) -> int: ...

    def reset(self) -> None: ...

    def add(self, vectors: FloatMatrix) -> None: ...

    def search(
        self, query: NDArray[np.float32], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


class NumpyVectorStore:
    """Portable exact cosine search for small and medium in-memory corpora."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def reset(self) -> None:
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, vectors: FloatMatrix) -> None:
        matrix = _validate_vectors(vectors, self.dimension)
        self._vectors = np.concatenate((self._vectors, matrix), axis=0)

    def search(
        self, query: NDArray[np.float32], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        if self.size == 0 or top_k <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ConfigurationError("Query embedding dimension does not match the index")
        scores = self._vectors @ vector
        count = min(top_k, self.size)
        if count == self.size:
            indices = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, count - 1)[:count]
            indices = candidates[np.argsort(-scores[candidates])]
        indices = indices.astype(np.int64, copy=False)
        return scores[indices].astype(np.float32, copy=False), indices


class FaissVectorStore:
    """FAISS exact inner-product search for larger in-memory corpora."""

    def __init__(self, dimension: int) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise DependencyError(
                "FAISS is not installed. Use vector_backend='numpy' or install `.[faiss]`."
            ) from exc
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def reset(self) -> None:
        self._index.reset()

    def add(self, vectors: FloatMatrix) -> None:
        self._index.add(_validate_vectors(vectors, self.dimension))

    def search(
        self, query: NDArray[np.float32], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        vector = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimension:
            raise ConfigurationError("Query embedding dimension does not match the index")
        scores, indices = self._index.search(vector, min(top_k, self.size))
        return scores[0].astype(np.float32, copy=False), indices[0].astype(np.int64, copy=False)


def create_vector_store(backend: str, dimension: int) -> VectorStore:
    normalized = backend.casefold()
    if normalized == "auto":
        normalized = "faiss" if importlib.util.find_spec("faiss") else "numpy"
    if normalized == "numpy":
        return NumpyVectorStore(dimension)
    if normalized == "faiss":
        return FaissVectorStore(dimension)
    raise ConfigurationError(f"Unknown vector backend: {backend!r}")


def _validate_vectors(vectors: FloatMatrix, dimension: int) -> FloatMatrix:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ConfigurationError(
            f"Expected a two-dimensional vector matrix with dimension {dimension}"
        )
    return matrix
