"""Domain-specific exceptions raised by the search engine."""


class SearchEngineError(Exception):
    """Base class for recoverable search-engine errors."""


class ConfigurationError(SearchEngineError, ValueError):
    """Raised when configuration values are invalid."""


class DependencyError(SearchEngineError, ImportError):
    """Raised when an optional runtime dependency is unavailable."""


class SearchProviderError(SearchEngineError):
    """Raised when a search provider cannot return results."""


class FetchError(SearchEngineError):
    """Raised when a web page cannot be fetched safely."""


class UnsafeURLError(FetchError, ValueError):
    """Raised when a URL violates the configured network policy."""


class ExtractionError(SearchEngineError):
    """Raised when useful content cannot be extracted from a page."""
