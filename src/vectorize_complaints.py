
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import numpy as np
import os
from tqdm import tqdm
from chromadb.errors import NotFoundError 

# Configuration
VECTOR_STORE_DIR = "../vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def load_data():
    """Load and verify the filtered complaints data"""
    df = pd.read_csv("../data/processed/filtered_complaints.csv")
    assert {'cleaned_narrative', 'Complaint ID', 'Product'}.issubset(df.columns)
    return df

def initialize_components():
    """Initialize text splitter and embedding model"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return text_splitter, embedding_model

def process_complaints(df, text_splitter, embedding_model):
    """Process all complaints into chunks with embeddings"""
    chunks = []
    metadata = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing complaints"):
        text_chunks = text_splitter.split_text(row['cleaned_narrative'])
        
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadata.append({
                "complaint_id": str(row['Complaint ID']),
                "product": str(row['Product']),
                "chunk_num": str(i),
                "num_chunks": str(len(text_chunks))
            })
    
    # Generate embeddings in batches
    batch_size = 100
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i+batch_size]
        embeddings.extend(embedding_model.encode(batch, show_progress_bar=False))
    
    return chunks, np.array(embeddings), metadata

def create_chroma_index(chunks, embeddings, metadata, batch_size=5000):
    """Create and persist ChromaDB vector store in batches"""
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

    # Delete existing collection if it exists
    try:
        client.delete_collection("complaints")
        print("Deleted existing collection")
    except NotFoundError:
        pass

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name="complaints",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    # Batch insert to avoid exceeding Chroma's limits
    for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector store"):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        batch_ids = [str(j) for j in range(i, i + len(batch_chunks))]

        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            ids=batch_ids
        )

    return client
 

def test_retrieval():
    """Test that the vector store was created correctly"""
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection("complaints")
    
    print(f"\nVector store contains {collection.count()} documents")
    
    # Sample query
    sample_query = "credit card dispute"
    results = collection.query(
        query_texts=[sample_query],
        n_results=3
    )
    
    print("\nSample query results for:", sample_query)
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\nResult {i+1}:")
        print(f"Product: {meta['product']}")
        print(f"Text: {doc[:200]}...")

if __name__ == "__main__":
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Initialize components
    text_splitter, embedding_model = initialize_components()
    
    # Step 3: Process complaints
    print("\nStarting data processing pipeline...")
    chunks, embeddings, metadata = process_complaints(df, text_splitter, embedding_model)
    
    # Step 4: Create vector store
    print("\nCreating vector store...")
    create_chroma_index(chunks, embeddings, metadata)
    
    # Step 5: Verify
    test_retrieval()
    
    print("\nPipeline completed successfully!")