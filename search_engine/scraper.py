import requests
from bs4 import BeautifulSoup

def fetch_page(url: str):
    """
    Fetch the content of the URL and return the raw HTML text.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad status
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch {url}: {e}")
    return response.text

def get_soup(html: str):
    """
    Parse HTML text into a BeautifulSoup object for easier traversal.
    """
    return BeautifulSoup(html, "lxml")
