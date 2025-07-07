import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

VECTOR_STORE_DIR = os.path.abspath("../vector_store")
COLLECTION_NAME = "complaints"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load LLM (You can switch to Mistral, LLaMA, etc.)
# generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=200)
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)

def retrieve_relevant_chunks(query, k=5):
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedding_model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results["documents"][0]
    return documents

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def generate_answer(query):
    chunks = retrieve_relevant_chunks(query, k=5)
    prompt = build_prompt(chunks, query)
    # response = generator(prompt)[0]['generated_text']
    response = generator(prompt)[0].get('generated_text', '')

    
    # Extract only the generated answer after "Answer:"
    return response.split("Answer:")[-1].strip(), chunks
