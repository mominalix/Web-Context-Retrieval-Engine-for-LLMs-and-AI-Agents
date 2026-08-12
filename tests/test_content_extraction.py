from search_engine.content_extraction import TrafilaturaExtractor, extract_content

HTML = """
<html>
  <head>
    <title>Semantic Retrieval</title>
    <meta name="description" content="A useful summary.">
    <script type="application/json">{"secret": "do not index hidden payload"}</script>
  </head>
  <body>
    <nav>Home Products Pricing</nav>
    <main>
      <h1>Semantic Retrieval</h1>
      <p>Dense retrieval maps questions and passages into a shared vector space.</p>
      <p>Relevant passages are ranked by similarity and supplied as grounded context.</p>
    </main>
    <footer>Copyright and repeated navigation</footer>
  </body>
</html>
"""


def test_extractor_excludes_hidden_script_payload() -> None:
    content = TrafilaturaExtractor().extract(HTML, url="https://example.com/article")

    assert "Dense retrieval" in content.text
    assert "do not index hidden payload" not in content.as_document_text()
    assert content.canonical_url == "https://example.com/article"


def test_legacy_extraction_shape() -> None:
    content = extract_content(HTML, query="unused")

    assert set(content) == {"title", "meta_description", "text", "hidden_text"}
    assert content["hidden_text"] == ""
