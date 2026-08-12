from conftest import WordTokenizer

from search_engine.chunking import TokenChunker


def test_chunks_use_token_windows_and_overlap() -> None:
    chunker = TokenChunker(WordTokenizer(), chunk_size=4, overlap=1)

    chunks = chunker.split("one two three four five six seven eight nine ten")

    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"
    assert chunks[2].text == "seven eight nine ten"


def test_truncate_respects_exact_token_budget() -> None:
    chunker = TokenChunker(WordTokenizer(), chunk_size=8, overlap=0)

    result = chunker.truncate("one two three four", 2)

    assert result.text == "one two"
    assert result.token_count == 2


def test_offset_chunking_preserves_original_text() -> None:
    chunker = TokenChunker(WordTokenizer(), chunk_size=3, overlap=0)

    chunks = chunker.split("Keep CASE, punctuation! Exactly.")

    assert chunks[0].text == "Keep CASE, punctuation!"
    assert chunks[1].text == "Exactly."
