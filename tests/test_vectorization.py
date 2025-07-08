import os
import pytest
import chromadb
from chromadb.errors import NotFoundError

# Configuration
VECTOR_STORE_DIR = "../vector_store"
COLLECTION_NAME = "complaints"

@pytest.fixture(scope="module")
def chroma_client():
    """Fixture to return a Chroma PersistentClient"""
    if not os.path.exists(VECTOR_STORE_DIR):
        pytest.fail("Vector store directory does not exist. Run vectorize_complaints.py first.")
    return chromadb.PersistentClient(path=VECTOR_STORE_DIR)

def test_vector_store_exists():
    """Check if the vector store directory exists"""
    assert os.path.exists(VECTOR_STORE_DIR), "Vector store directory does not exist"

def test_collection_exists(chroma_client):
    """Ensure the 'complaints' collection exists"""
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        assert collection is not None
    except NotFoundError:
        pytest.fail(f"Collection '{COLLECTION_NAME}' was not found in the vector store")

def test_collection_has_documents(chroma_client):
    """Verify that the collection contains at least one document"""
    collection = chroma_client.get_collection(COLLECTION_NAME)
    count = collection.count()
    assert count > 0, "Collection is empty"
