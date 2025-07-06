import os
import pytest
import pandas as pd
from src.vectorize_complaints import (
    initialize_components,
    process_complaints,
    create_chroma_index,
    VECTOR_STORE_DIR
)

@pytest.fixture(scope="module")
def setup_pipeline():
    # Mock sample complaint data
    df_sample = pd.DataFrame({
        "Complaint ID": [1, 2],
        "Product": ["Credit card", "Student loan"],
        "Consumer complaint narrative": [
            "I was charged unfair late fees despite paying on time.",
            "My loan servicer did not properly apply payments and it hurt my credit score."
        ]
    })

    # Minimal mock cleaning
    df_sample["cleaned_narrative"] = (
        df_sample["Consumer complaint narrative"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df_sample = df_sample[["Complaint ID", "Product", "cleaned_narrative"]]

    # Init and run
    text_splitter, embedding_model = initialize_components()
    chunks, embeddings, metadata = process_complaints(df_sample, text_splitter, embedding_model)

    return {
        "df": df_sample,
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": metadata
    }
def test_data_loading(setup_pipeline):
    df = setup_pipeline["df"]
    assert not df.empty
    assert {"cleaned_narrative", "Complaint ID", "Product"}.issubset(df.columns)

def test_chunking_output(setup_pipeline):
    chunks = setup_pipeline["chunks"]
    metadata = setup_pipeline["metadata"]
    assert len(chunks) > 0
    assert len(chunks) == len(metadata)

def test_embedding_shape(setup_pipeline):
    chunks = setup_pipeline["chunks"]
    embeddings = setup_pipeline["embeddings"]
    assert embeddings.shape[0] == len(chunks)

def test_vector_store_directory_created():
    assert os.path.exists(VECTOR_STORE_DIR)

def test_vector_store_indexing(setup_pipeline):
    chunks = setup_pipeline["chunks"]
    embeddings = setup_pipeline["embeddings"]
    metadata = setup_pipeline["metadata"]
    
    client = create_chroma_index(chunks, embeddings, metadata)
    collection = client.get_collection("complaints")
    assert collection.count() > 0

def test_vector_store_query(setup_pipeline):
    chunks = setup_pipeline["chunks"]
    embeddings = setup_pipeline["embeddings"]
    metadata = setup_pipeline["metadata"]
    
    client = create_chroma_index(chunks, embeddings, metadata)
    collection = client.get_collection("complaints")
    
    result = collection.query(query_texts=["loan dispute"], n_results=2)
    assert "documents" in result
    assert len(result["documents"][0]) > 0
