"""Tokenizer-aware text splitting for predictable embedding and LLM budgets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .embeddings import Tokenizer
from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    token_count: int


class TokenChunker:
    def __init__(self, tokenizer: Tokenizer, *, chunk_size: int = 256, overlap: int = 32):
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be greater than zero")
        if not 0 <= overlap < chunk_size:
            raise ConfigurationError("overlap must be non-negative and smaller than chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\r\n?", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def count_tokens(self, text: str) -> int:
        token_ids, _ = self._tokenize(text)
        return len(token_ids)

    def truncate(self, text: str, max_tokens: int) -> TextChunk:
        if max_tokens <= 0:
            return TextChunk("", 0)
        normalized = self._normalize(text)
        tokens, offsets = self._tokenize(normalized)
        selected = tokens[:max_tokens]
        if offsets and selected:
            decoded = normalized[offsets[0][0] : offsets[len(selected) - 1][1]].strip()
        else:
            decoded = self.tokenizer.decode(
                selected,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        return TextChunk(decoded, len(selected) if decoded else 0)

    def split(self, text: str) -> list[TextChunk]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        token_ids, offsets = self._tokenize(normalized)
        if not token_ids:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[TextChunk] = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.chunk_size]
            if offsets and window:
                end_index = start + len(window) - 1
                decoded = normalized[offsets[start][0] : offsets[end_index][1]].strip()
            else:
                decoded = self.tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
            if decoded:
                chunks.append(TextChunk(decoded, len(window)))
            if start + self.chunk_size >= len(token_ids):
                break
        return chunks

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        with_offsets = getattr(self.tokenizer, "tokenize_with_offsets", None)
        if callable(with_offsets):
            token_ids, offsets = with_offsets(text)
            if len(token_ids) == len(offsets):
                return token_ids, offsets
        return self.tokenizer.encode(text, add_special_tokens=False), None
