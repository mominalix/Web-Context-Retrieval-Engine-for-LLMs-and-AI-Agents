"""Bounded, retrying HTTP fetcher with redirect and SSRF protections."""

from __future__ import annotations

import codecs
import ipaddress
import re
import socket
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from requests.compat import chardet
from urllib3.util.retry import Retry

from .exceptions import FetchError, UnsafeURLError
from .models import FetchedPage

Resolver = Callable[..., list[tuple]]
_HEADER_CHARSET_RE = re.compile(r"(?:^|;)\s*charset\s*=\s*[\"']?([^;\"']+)", re.I)
_HTML_CHARSET_RE = re.compile(rb"<meta\s+[^>]*charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]+)", re.I)


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


class URLPolicy:
    """Validate outbound HTTP targets before each request and redirect."""

    def __init__(self, *, allow_private_network: bool = False, resolver: Resolver | None = None):
        self.allow_private_network = allow_private_network
        self._resolver = resolver or socket.getaddrinfo

    def validate(self, url: str) -> None:
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError as exc:
            raise UnsafeURLError(f"Malformed URL: {url!r}") from exc
        if parts.scheme.casefold() not in {"http", "https"}:
            raise UnsafeURLError("Only http and https URLs are allowed")
        if not parts.hostname:
            raise UnsafeURLError("URL must include a hostname")
        if parts.username is not None or parts.password is not None:
            raise UnsafeURLError("Embedded URL credentials are not allowed")
        if self.allow_private_network:
            return

        try:
            addresses = {
                item[4][0]
                for item in self._resolver(
                    parts.hostname,
                    port or (443 if parts.scheme.casefold() == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise FetchError(f"Could not resolve host {parts.hostname!r}: {exc}") from exc
        if not addresses:
            raise FetchError(f"Host {parts.hostname!r} did not resolve to an address")
        for raw_address in addresses:
            address = ipaddress.ip_address(raw_address)
            if not address.is_global:
                raise UnsafeURLError(
                    f"Refusing non-public address {address} for host {parts.hostname!r}"
                )


class HttpFetcher:
    def __init__(
        self,
        *,
        user_agent: str = "semantic-search-agents/0.2",
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
        retries: int = 2,
        max_redirects: int = 4,
        max_response_bytes: int = 3_000_000,
        allow_private_network: bool = False,
        session: requests.Session | None = None,
        url_policy: URLPolicy | None = None,
    ) -> None:
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be greater than zero")
        self.timeout = (connect_timeout, read_timeout)
        self.max_redirects = max_redirects
        self.max_response_bytes = max_response_bytes
        self.url_policy = url_policy or URLPolicy(allow_private_network=allow_private_network)
        self.session = session or self._build_session(retries)
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> HttpFetcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _build_session(retries: int) -> requests.Session:
        retry_policy = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=(429, 500, 502, 503, 504),
            backoff_factor=0.25,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_policy, pool_connections=10, pool_maxsize=10)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch(self, url: str) -> FetchedPage:
        current_url = url
        for redirect_count in range(self.max_redirects + 1):
            self.url_policy.validate(current_url)
            try:
                response = self.session.get(
                    current_url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise FetchError(f"Failed to fetch {current_url}: {exc}") from exc

            with response:
                if response.is_redirect or response.is_permanent_redirect:
                    if redirect_count >= self.max_redirects:
                        raise FetchError(f"Too many redirects while fetching {url}")
                    location = response.headers.get("Location")
                    if not location:
                        raise FetchError(f"Redirect from {current_url} had no Location header")
                    current_url = urljoin(current_url, location)
                    continue

                try:
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise FetchError(f"Failed to fetch {current_url}: {exc}") from exc

                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
                if content_type and content_type not in {
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                }:
                    raise FetchError(f"Unsupported content type {content_type!r} at {current_url}")
                declared_size = response.headers.get("Content-Length")
                if declared_size:
                    try:
                        if int(declared_size) > self.max_response_bytes:
                            raise FetchError(f"Response from {current_url} exceeds the size limit")
                    except ValueError:
                        pass

                body = bytearray()
                try:
                    for block in response.iter_content(chunk_size=64 * 1024):
                        body.extend(block)
                        if len(body) > self.max_response_bytes:
                            raise FetchError(f"Response from {current_url} exceeds the size limit")
                except requests.RequestException as exc:
                    raise FetchError(f"Failed while reading {current_url}: {exc}") from exc

                raw_body = bytes(body)
                encoding = _detect_encoding(
                    response.headers.get("Content-Type", ""),
                    raw_body,
                )
                try:
                    text = raw_body.decode(encoding, errors="replace")
                except LookupError:
                    text = raw_body.decode("utf-8", errors="replace")
                return FetchedPage(
                    url=current_url,
                    text=text,
                    content_type=content_type or "text/html",
                    status_code=response.status_code,
                )
        raise FetchError(f"Too many redirects while fetching {url}")


def fetch_page(url: str) -> str:
    """Backward-compatible wrapper returning just the response body."""
    return HttpFetcher().fetch(url).text


def _detect_encoding(content_type: str, body: bytes) -> str:
    """Prefer declared encodings, then modern HTML/UTF-8 signals, then detection."""
    header_match = _HEADER_CHARSET_RE.search(content_type)
    if header_match:
        return header_match.group(1).strip()
    if body.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    meta_match = _HTML_CHARSET_RE.search(body[:8192])
    if meta_match:
        return meta_match.group(1).decode("ascii")
    try:
        body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        detected = chardet.detect(body)
        return str(detected.get("encoding") or "utf-8")
    return "utf-8"
