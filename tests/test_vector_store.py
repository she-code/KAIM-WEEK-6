import os

import chromadb
import pytest
from chromadb.errors import NotFoundError

# Configuration
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "../vector_store")
COLLECTION_NAME = "complaints"


@pytest.fixture(scope="module")
def chroma_client():
    """Fixture to return a Chroma PersistentClient"""
    if not os.path.exists(VECTOR_STORE_DIR):
        pytest.fail(
            "Vector store directory does not exist. Run vectorize_complaints.py first."
        )
    return chromadb.PersistentClient(path=VECTOR_STORE_DIR)


def test_vector_store_directory_created():
    assert os.path.exists(VECTOR_STORE_DIR)


def test_collection_exists(chroma_client):
    """Ensure the 'complaints' collection exists"""
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        assert collection is not None
    except NotFoundError:
        pytest.fail(f"Collection '{COLLECTION_NAME}' was not found in the vector store")
