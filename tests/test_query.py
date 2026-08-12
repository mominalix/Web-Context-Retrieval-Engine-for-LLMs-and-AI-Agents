import sys
from types import SimpleNamespace

from search_engine.query import DDGSSearchProvider, normalize_url, search_web


class FakeDDGS:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def text(self, query: str, **kwargs):
        assert query == "test query"
        assert kwargs["max_results"] == 3
        return [
            {"title": "One", "href": "HTTPS://Example.com:443/a#part", "body": "First"},
            {"title": "Duplicate", "href": "https://example.com/a", "body": "Same"},
            {"title": "Two", "href": "https://example.org/b?q=1", "body": "Second"},
        ]


def test_provider_maps_real_result_urls_and_deduplicates(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    results = DDGSSearchProvider().search("test query", limit=3)

    assert [result.url for result in results] == [
        "https://example.com/a",
        "https://example.org/b?q=1",
    ]
    assert results[0].title == "One"
    assert results[0].snippet == "First"


def test_legacy_search_web_returns_expected_dictionary(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    results = search_web("test query", num_results=3)

    assert results[0] == {
        "title": "One",
        "url": "https://example.com/a",
        "snippet": "First",
    }


def test_normalize_url_rejects_non_web_and_invalid_ports() -> None:
    assert normalize_url("file:///tmp/data") == ""
    assert normalize_url("https://example.com:not-a-port") == ""
