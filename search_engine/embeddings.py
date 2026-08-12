"""Embedding-provider interfaces and the Sentence Transformers adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .exceptions import DependencyError

FloatMatrix = NDArray[np.float32]


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def tokenizer(self) -> Tokenizer: ...

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix: ...


class TransformersTokenizerAdapter:
    """Expose quiet tokenization plus offsets while retaining a small public protocol."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            verbose=False,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def tokenize_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        token_ids = list(encoded["input_ids"])
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        return token_ids, offsets


class SentenceTransformerEmbedder:
    """Dense retrieval adapter using asymmetric query/document encoding."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise DependencyError(
                "Sentence Transformers is required. Install the project with `pip install -e .`."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._tokenizer = TransformersTokenizerAdapter(self._model.tokenizer)
        self._batch_size = batch_size
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise DependencyError(f"Model {model_name!r} does not expose an embedding dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def tokenizer(self) -> Tokenizer:
        if self._model.tokenizer is None:
            raise DependencyError("The selected embedding model does not expose a tokenizer")
        return self._tokenizer

    @property
    def max_input_tokens(self) -> int:
        """Maximum content tokens after reserving the model's special tokens."""
        tokenizer = self._model.tokenizer
        special_tokens = 0
        count_special_tokens = getattr(tokenizer, "num_special_tokens_to_add", None)
        if count_special_tokens is not None:
            special_tokens = int(count_special_tokens(pair=False))
        return max(1, int(self._model.max_seq_length) - special_tokens)

    @property
    def model(self):
        """Expose the underlying model for advanced users without coupling core code to it."""
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode_document(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode_query(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
