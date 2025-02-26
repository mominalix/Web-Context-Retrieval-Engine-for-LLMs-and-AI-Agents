from bs4 import BeautifulSoup
import json
import re

def _flatten_json(data, parent_key="", result=None):
    """
    Helper to flatten JSON data (dicts/lists) into a string by extracting textual/numeric values.
    """
    if result is None:
        result = []
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip keys that are not content (e.g., metadata keys starting with '@')
            if str(key).startswith("@"):
                continue
            _flatten_json(value, key, result)
    elif isinstance(data, list):
        for item in data:
            _flatten_json(item, parent_key, result)
    else:
        # If it's a primitive type (str, int, float), add it
        if isinstance(data, (str, int, float)):
            text = str(data)
            # Filter out very short or non-informative strings (like empty or just special chars)
            if len(text.strip()) > 0:
                result.append(text.strip())
    return result

def extract_content(html: str, query: str = None):
    """
    Extract the main textual content and relevant hidden data from the HTML.
    Returns a dictionary with extracted text components.
    """
    soup = BeautifulSoup(html, "lxml")
    content = {}

    # Extract the page title
    title_tag = soup.find('title')
    content["title"] = title_tag.get_text().strip() if title_tag else ""

    # Extract meta description if available
    desc_tag = soup.find('meta', attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        content["meta_description"] = desc_tag["content"].strip()
    else:
        content["meta_description"] = ""

    # Extract JSON data from script tags (e.g., structured data or hidden JSON)
    hidden_text_parts = []
    for script in soup.find_all("script"):
        script_type = script.get("type", "")
        script_text = script.string
        if script_text:
            script_text = script_text.strip()
        # Look for JSON in script tags:
        if script_type.lower().endswith("json") or script_text.startswith("{"):
            try:
                data = json.loads(script_text)
            except Exception:
                data = None
            if data:
                flat_values = _flatten_json(data)
                if flat_values:
                    # Join extracted values with space
                    hidden_text_parts.append(" ".join(flat_values))
    # Combine all hidden text parts into one string
    content["hidden_text"] = " ".join(hidden_text_parts)

    # Remove script and style elements from the soup to avoid irrelevant text
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Extract visible text from the body
    visible_text = soup.get_text(separator="\n")
    # Optional: filter out navigation or repetitive text by simple heuristics
    lines = [line.strip() for line in visible_text.splitlines()]
    lines = [line for line in lines if line and not re.match(r'^[A-Za-z0-9\W]{1,15}$', line)]
    # (The regex above removes very short lines which are often menu items or gibberish)
    main_text = "\n".join(lines)

    content["text"] = main_text.strip()
    return content
