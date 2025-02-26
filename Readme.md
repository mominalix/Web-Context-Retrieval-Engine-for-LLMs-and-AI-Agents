# Web Context Retrieval Engine for LLMs and AI Agents

A modular Python-based search engine that fetches live information from the web, extracts and processes relevant content (including hidden data and metadata), and returns context-rich snippets optimized for powering Large Language Models (LLMs) and AI agents.

## Overview

In today's data-driven world, LLMs and AI agents rely on up-to-date, precise context from the web to generate accurate and relevant responses. This project is designed to:

- **Fetch live data:** Perform real-time web searches using search engine results.
- **Advanced web scraping:** Dynamically extract visible text, metadata, and hidden content (e.g., JSON in `<script>` tags) from web pages.
- **NLP-powered content extraction:** Utilize NLP techniques and SpaCy to clean, chunk, and filter the scraped data.
- **Semantic indexing and retrieval:** Convert content into embeddings with Sentence Transformers and index them using FAISS for fast, AI-powered semantic search.
- **Dynamic ranking & chunking:** Improve retrieval accuracy by adaptively chunking text and refining relevance scores.

This engine provides the necessary context for LLMs and AI agents—ensuring they have access to the latest and most relevant information from the web.

## Value Proposition

- **Enhanced Context:** Deliver enriched, context-aware content to LLMs and AI agents, leading to more accurate outputs.
- **Real-Time Information:** Fetch live data to support dynamic decision-making in AI workflows.
- **Modular & Scalable:** A clean, industry-standard file structure allows you to easily extend or integrate components.
- **AI-Powered Retrieval:** Leverage state-of-the-art NLP techniques for semantic search, ensuring relevant results even with complex queries (e.g., weather forecasts, industry news, research data).

## Setup

### Prerequisites

- **Python 3.7+**  
- Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```
- **SpaCy Model:**  
  Download the SpaCy English model:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### File Structure

```
web-context-retrieval/
├── search_engine/
│   ├── __init__.py
│   ├── query.py              # Handles search querying.
│   ├── scraper.py            # Fetches raw HTML content.
│   ├── content_extraction.py # Extracts main and hidden text from HTML.
│   ├── indexer.py            # Chunks text, computes embeddings, and builds FAISS index.
│   └── retrieval.py          # Retrieves and ranks relevant content.
├── main.py                   # Usage example that ties all modules together.
├── requirements.txt          # Python dependencies.
└── README.md                 # This documentation file.
```

## Running the Project

The usage example in `main.py` demonstrates the full pipeline:
1. **Search Query:** It performs a live search (using a search engine scraper) for your query.
2. **Scraping & Extraction:** It fetches and extracts the web content, including hidden data.
3. **Indexing:** The content is dynamically chunked, embedded, and indexed using FAISS.
4. **Semantic Retrieval:** Finally, the engine retrieves and ranks the most relevant snippets for your query.

To run the example:
```bash
python main.py
```

You will see output similar to:
```
Query: Who is founder of Pakistan?
...
Top relevant results:
Rank 1 (Score: 0.95) from https://example.com:
"Founder of Pakistan: Muhammad Ali Jinnah..."
```

## Explanation

- **Search Engine Module:** Queries a search engine (Google or DuckDuckGo) to return live results.
- **Scraping Module:** Downloads the HTML of the result pages.
- **Content Extraction Module:** Uses BeautifulSoup to parse HTML and NLP techniques (via SpaCy) to extract and clean both visible and hidden text.
- **Indexer Module:** Splits text into manageable chunks and uses a Sentence Transformer model to create semantic embeddings. These embeddings are indexed in FAISS for fast retrieval.
- **Retriever Module:** Given a new query, it generates an embedding, searches the FAISS index, and returns the most relevant content snippets along with source metadata.

## Use Cases

- **LLM Context Provision:** Power your chatbot or virtual assistant by providing real-time context from reliable web sources.
- **AI Agent Integration:** Enhance the decision-making capabilities of autonomous AI agents with up-to-date, semantically relevant web information.
- **Research and Data Enrichment:** Quickly gather comprehensive, context-rich data for academic research, market analysis, or news aggregation.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements. Please ensure your code adheres to the existing style and structure.

