# Scripts

### vectorize_complaints.py

Load cleaned complaint data with essential fields validation

Initialize recursive text splitter and SentenceTransformer embedding model

Split complaint narratives into overlapping chunks for granular embedding

Generate batch embeddings for efficiency

Create persistent ChromaDB vector store, deleting old collections if present

Add chunked documents with metadata and embeddings into the vector store

Implement retrieval test with sample query to verify index correctness

Include progress bars and informative print statements for monitoring

Ensure vector store directory creation for persistence outside version control