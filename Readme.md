# Semantic Search for AI Agents

A modular Python backend that searches the web, extracts the useful page content, embeds it, and
returns diverse context chunks under a hard token budget. It is designed for retrieval-augmented
generation (RAG), tool-using agents, and other systems that need current web context without sending
entire pages to an LLM.

## What changed in 0.2

- Replaced fragile search-result HTML scraping with an injectable `SearchProvider` and a DDGS
  metasearch adapter.
- Replaced hand-written DOM heuristics and hidden-script indexing with Trafilatura main-content
  extraction. Script payloads are intentionally excluded from retrieval context.
- Replaced character chunks and the external spaCy model with overlapping chunks measured by the
  embedding model's own tokenizer.
- Uses Sentence Transformers' retrieval-specific `encode_query()` and `encode_document()` paths.
- Adds exact cosine search through a portable NumPy backend, plus optional FAISS acceleration.
- Adds chunk deduplication, score filtering, MMR diversity, per-source limits, and a hard context
  token budget to reduce downstream LLM input cost.
- Adds bounded downloads, explicit connect/read timeouts, safe GET retries, redirect validation,
  response type/size limits, and public-network URL checks.
- Adds typed data contracts, environment configuration, dependency injection, a CLI, and tests.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
```

For the optional FAISS backend:

```bash
python -m pip install -e ".[faiss]"
```

The first real query downloads the configured Sentence Transformers model. NumPy is the automatic
fallback when FAISS is not installed.

## Run

There is no hardcoded query:

```bash
python -m search_engine "What are recent advances in semantic retrieval?"
semantic-search "What are recent advances in semantic retrieval?" --max-context-tokens 900 --json
```

Use the backend from Python:

```python
from search_engine import SearchPipeline, Settings

settings = Settings(
    search_results=6,
    top_k=4,
    max_context_tokens=800,
    vector_backend="auto",
)
with SearchPipeline(settings) as pipeline:
    response = pipeline.run("How does query-document asymmetric retrieval work?")

for hit in response.hits:
    print(hit.source, hit.score, hit.token_count)
    print(hit.text)
```

Treat `hit.text` as untrusted external data when placing it in an agent prompt. Delimit retrieved
content, do not let it override system/developer instructions, and retain source URLs for
attribution.

## Modularity

`SearchPipeline` accepts custom implementations for each external boundary:

```python
pipeline = SearchPipeline(
    settings,
    search_provider=my_search_api,
    fetcher=my_authenticated_fetcher,
    extractor=my_domain_extractor,
    indexer=my_persistent_indexer,
)
```

The relevant protocols are `SearchProvider`, `ContentExtractor`, `EmbeddingProvider`, and
`VectorStore`. `Indexer` also accepts legacy strings and `(text, metadata)` tuples, so existing code
can migrate gradually.

## Configuration

Every `Settings` field can be set with a `SEMANTIC_SEARCH_` environment variable. Common settings
are shown in `.env.example`.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `SEMANTIC_SEARCH_MODEL_NAME` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Embedding model |
| `SEMANTIC_SEARCH_VECTOR_BACKEND` | `auto` | `auto`, `numpy`, or `faiss` |
| `SEMANTIC_SEARCH_CHUNK_SIZE_TOKENS` | `256` | Maximum tokens per indexed chunk |
| `SEMANTIC_SEARCH_CHUNK_OVERLAP_TOKENS` | `32` | Retrieval continuity between chunks |
| `SEMANTIC_SEARCH_MAX_CONTEXT_TOKENS` | `1200` | Maximum retrieved text tokens |
| `SEMANTIC_SEARCH_SCORE_THRESHOLD` | `0.15` | Minimum cosine similarity |
| `SEMANTIC_SEARCH_MMR_LAMBDA` | `0.7` | Relevance/diversity balance |
| `SEMANTIC_SEARCH_ALLOW_PRIVATE_NETWORK` | `false` | Permit localhost/private URL fetching |

No API key is required by the default implementation. `.env` is ignored and the library does not
implicitly load dotenv files, so host applications retain control over secret loading.

## Token-efficiency controls

The pipeline saves context tokens at several stages:

1. Boilerplate and hidden scripts are removed before embedding.
2. Repeated chunks are discarded before model inference.
3. Retrieval fetches a wider candidate set, then MMR removes near-duplicate context.
4. Per-source limits prevent one page from consuming the complete budget.
5. The last selected chunk is tokenizer-truncated when necessary, so `context_tokens` never exceeds
   `max_context_tokens`.

For higher retrieval quality, inject a second-stage reranker after initial retrieval. A cross-encoder
usually improves precision but adds latency and compute, so it is not enabled by default.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Web search and scraping are subject to provider/site terms, robots policies, privacy rules, and local
law. Configure an identifiable user agent for production use and prefer an official search API where
your deployment requires contractual reliability.

Before publishing a fork, set the real repository URL in `pyproject.toml` and add the license chosen
by the repository owner.
