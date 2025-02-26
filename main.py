from search_engine.query import search_web
from search_engine.scraper import fetch_page
from search_engine.content_extraction import extract_content
from search_engine.indexer import Indexer
from search_engine.retrieval import Retriever

# User query
query = "Weather in Riyadh tomorrow"

# Step 1: Get initial search results for the query
results = search_web(query, num_results=20)
print(f"Search results for '{query}':")

# Step 2: Scrape each result and extract content
pages = []
for res in results:
    html = fetch_page(res['url'])
    content = extract_content(html, query=query)
    # Combine title, meta, main text, and hidden text into one content string for indexing
    full_text = " ".join([content.get("title",""), content.get("meta_description",""), content.get("text",""), content.get("hidden_text","")])
    pages.append((full_text, res['url']))

# Step 3: Build the semantic index from the scraped pages
indexer = Indexer()
indexer.index_documents(pages)

# Step 4: Retrieve the most relevant content chunks for the query
retriever = Retriever(indexer)
top_chunks = retriever.semantic_search(query, top_k=3)

# Display the top retrieved snippets
print(f"Top answers for '{query}':")
for item in top_chunks:
    print(f"Rank {item['rank']} (Score: {item['score']:.2f}) from {item['source']}:")
    print(f"\"{item['text']}\"")
    print("----")
