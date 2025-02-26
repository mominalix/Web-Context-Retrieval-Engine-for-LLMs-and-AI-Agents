import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MySearchBot/1.0; +https://example.com/bot)"}

def normalize_url(url):
    """Ensure the URL has a scheme. If it's protocol-relative, prepend 'https:'."""
    if url.startswith("//"):
        return "https:" + url
    return url

def extract_actual_url(url):
    """
    If the URL is a DuckDuckGo redirection (contains 'uddg' parameter),
    extract and return the actual target URL.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "uddg" in qs:
        actual_url = unquote(qs["uddg"][0])
        return actual_url
    return url


def search_web(query: str, num_results: int = 5):
    """
    Query a search engine and return the top results.
    Returns a list of dicts: {"title": ..., "url": ..., "snippet": ...}.
    """
    base_url = "https://html.duckduckgo.com/html/?q="
    url = base_url + requests.utils.requote_uri(query)
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Search request failed with status code {resp.status_code}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    result_links = []
    # Extract links from <a class="result__a"> tags.
    for a in soup.find_all('a', class_='result__a')[:num_results]:
        link = a.get('href')
        if link:
            normalized = normalize_url(link)
            # Extract the actual URL if it's a DuckDuckGo redirection.
            actual = extract_actual_url(normalized)
            result_links.append({"url": url})
    return result_links

