import chromadb

VECTOR_STORE_DIR = "../vector_store"

def inspect_chroma():
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection("complaints")
    print(f"Document count: {collection.count()}")

if __name__ == "__main__":
    inspect_chroma()
