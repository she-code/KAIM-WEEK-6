

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from tqdm import tqdm
import time

# Initialize components
print("Initializing components...")
start_time = time.time()

embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32
    },
    # cache_folder="../model_cache"
)


print(f"Components initialized in {time.time() - start_time:.2f} seconds")

# Load documents
print("\nLoading documents...")
start_time = time.time()
loader = CSVLoader(file_path="../data/processed/filtered_complaints.csv")
documents = loader.load()
print(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")

# Split documents
print("\nSplitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)

splits = []
with tqdm(total=len(documents), desc="Splitting documents") as pbar:
    for doc in documents:
        splits.extend(text_splitter.split_documents([doc]))
        pbar.update(1)

print(f"Split into {len(splits)} chunks")

# Create vector store with progress tracking
print("\nCreating vector store (this may take some time)...")
start_time = time.time()

# Process in batches to show progress
batch_size = 100  # Adjust based on your system's memory
vectorstore = None

with tqdm(total=len(splits), desc="Generating embeddings") as pbar:
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory="../vector_store",
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vectorstore.add_documents(batch)
        pbar.update(len(batch))

print(f"Vector store created in {time.time() - start_time:.2f} seconds")
