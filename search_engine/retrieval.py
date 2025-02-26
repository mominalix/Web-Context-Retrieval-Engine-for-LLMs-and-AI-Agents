import numpy as np

class Retriever:
    """
    Retriever uses a FAISS index and embedding model to find relevant content for queries.
    It can incorporate additional ranking adjustments if needed.
    """
    def __init__(self, indexer):
        """
        Initialize with an Indexer instance that has already indexed documents.
        """
        self.index = indexer.index
        self.model = indexer.model
        self.meta_data = indexer.meta_data
        self.dimension = indexer.dimension

    def semantic_search(self, query: str, top_k: int = 5):
        """
        Search the index for the given query and return the top_k relevant chunks.
        Returns a list of metadata dicts (including source and text snippet).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        # Embed the query using the same model (normalize for consistency)
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = query_vec.astype('float32')
        # Search the FAISS index
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue  # -1 may indicate empty results if fewer than top_k found
            meta = self.meta_data[idx]
            score = float(distances[0][rank])
            # We can include the similarity score if needed. For now, just prepare snippet.
            text_snippet = meta["text"]
            # Optionally, trim the snippet for display purposes (e.g., first 200 chars)
            snippet_trimmed = text_snippet
            if len(text_snippet) > 500:
                snippet_trimmed = text_snippet[:500].rstrip() + "..."
            results.append({
                "rank": rank+1,
                "score": score,
                "source": meta["source"],
                "text": snippet_trimmed
            })
        # (Optional) Re-ranking or filtering could be applied here.
        # For example, ensuring diversity or using a cross-encoder for reranking.
        return results
