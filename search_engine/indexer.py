from sentence_transformers import SentenceTransformer
import faiss
import spacy

class Indexer:
    """
    Indexer handles embedding of texts and building a FAISS index for semantic search.
    It also manages adaptive chunking of content for better retrieval granularity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the sentence transformer model for embeddings
        self.model = SentenceTransformer(model_name)
        # Load a SpaCy model for text processing (for chunking and potential NLP tasks)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            # If the model isn't downloaded, instruct to download it
            raise Exception("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e
        self.index = None
        self.meta_data = []  # to store metadata for each indexed chunk (e.g., source info and text)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # Initialize a FAISS index (Flat IP for cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)

    def _chunk_text(self, text: str, max_length: int = 500):
        """
        Split a long text into chunks of approximately max_length characters, without breaking sentences.
        Returns a list of text chunks.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        chunks = []
        current_chunk = ""
        for sent in sentences:
            # If adding this sentence to current chunk exceeds length, start a new chunk
            if current_chunk and len(current_chunk) + len(sent) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += " " + sent
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [""]  # return at least an empty string if no content

    def index_documents(self, docs: list):
        """
        Build the FAISS index from a list of documents.
        Each item in docs should be either a string (text content) or a tuple (text, source_meta).
        """
        texts = []
        # Clear any existing index data
        self.meta_data = []
        self.index.reset()  # clears the index
        # Process each document
        for item in docs:
            if isinstance(item, tuple):
                text, meta = item
            else:
                text, meta = item, None
            # Adaptive chunking of the document text
            for chunk in self._chunk_text(text):
                if not chunk or chunk.isspace():
                    continue
                texts.append(chunk)
                # Store metadata: include source info and possibly the chunk text for reference
                self.meta_data.append({
                    "source": meta,
                    "text": chunk
                })
        if not texts:
            return  # Nothing to index
        # Generate embeddings for all chunks
        # Normalize embeddings for inner product = cosine similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        # Add to FAISS index
        self.index.add(embeddings)
        # Now the index is built with vectors corresponding to entries in self.meta_data

    def add_document(self, text: str, meta=None):
        """
        Add a single document to the index (can be called multiple times to build index incrementally).
        """
        # Chunk the text
        chunks = self._chunk_text(text)
        if not chunks:
            return
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        embeddings = embeddings.astype('float32')
        # Add each chunk embedding to the index and store metadata
        for i, chunk in enumerate(chunks):
            self.index.add(embeddings[i:i+1])  # add one vector
            self.meta_data.append({"source": meta, "text": chunk})

    def get_index_size(self):
        """Return the number of indexed chunks."""
        return self.index.ntotal if self.index else 0
