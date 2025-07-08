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


### rag_pipeline.py

Load persisted Chroma vector store using HuggingFace all-MiniLM-L6-v2 embeddings

Initialize high-performance LLM (e.g., llama3-70b-8192 via Groq) for response generation

Define a system prompt to constrain LLM responses to retrieved complaint excerpts

Configure retriever to fetch top-5 relevant complaint chunks based on semantic similarity

Create a Retrieval-Augmented Generation (RAG) chain combining retriever, prompt, and LLM

Format retrieved documents into readable context format for the prompt

Test the pipeline with representative customer service questions

Print intermediate retrieval results, metadata, and timing diagnostics for each query

Enable quick validation of end-to-end RAG flow and model response quality

