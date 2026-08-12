from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np


class WordTokenizer:
    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._ids: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        for token in re.findall(r"\S+", text):
            if token not in self._tokens:
                token_id = len(self._tokens) + 1
                self._tokens[token] = token_id
                self._ids[token_id] = token
            result.append(self._tokens[token])
        return result

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(self._ids[token_id] for token_id in token_ids)

    def tokenize_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        matches = list(re.finditer(r"\S+", text))
        token_ids = self.encode(text)
        return token_ids, [(match.start(), match.end()) for match in matches]


class FakeEmbedder:
    def __init__(self) -> None:
        self._tokenizer = WordTokenizer()

    @property
    def dimension(self) -> int:
        return 3

    @property
    def tokenizer(self) -> WordTokenizer:
        return self._tokenizer

    @staticmethod
    def _encode(texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            words = text.casefold().split()
            vector = np.array(
                [words.count("alpha"), words.count("beta"), max(1, len(words)) / 10],
                dtype=np.float32,
            )
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)
