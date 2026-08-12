import pytest

from search_engine.config import Settings
from search_engine.exceptions import ConfigurationError


def test_settings_load_typed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_SEARCH_TOP_K", "3")
    monkeypatch.setenv("SEMANTIC_SEARCH_FETCH_K", "7")
    monkeypatch.setenv("SEMANTIC_SEARCH_ALLOW_PRIVATE_NETWORK", "yes")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")

    settings = Settings.from_env()

    assert settings.top_k == 3
    assert settings.fetch_k == 7
    assert settings.allow_private_network is True


def test_settings_reject_invalid_overlap() -> None:
    with pytest.raises(ConfigurationError, match="chunk_overlap_tokens"):
        Settings(chunk_size_tokens=10, chunk_overlap_tokens=10)


def test_settings_reject_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_SEARCH_ALLOW_PRIVATE_NETWORK", "perhaps")
    with pytest.raises(ConfigurationError, match="boolean"):
        Settings.from_env()
