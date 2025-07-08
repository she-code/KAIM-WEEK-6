
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq  # For Grok access via Groq

# # Initialize components
# # embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

# embedding_model = HuggingFaceEmbeddings(
# model_name='BAAI/bge-base-en-v1.5',
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True},
#     cache_folder="../model_cache"
# )

# llm = ChatGroq(
#     groq_api_key="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",  # Your Groq API key
#     model_name="mixtral-8x7b-32768"  # You can try "llama2-70b-4096" or other available models
# )

# # Load and process CSV
# loader = CSVLoader(file_path="../data/processed/filtered_complaints.csv")
# documents = loader.load()

# # Configure text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     add_start_index=True
# )
# splits = text_splitter.split_documents(documents)

# # Create vector store
# vectorstore = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding_model,
#     persist_directory="./chroma_db"
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# # Define prompt template
# system_template = """You are a financial analyst assistant for CrediTrust. 
# Answer questions using ONLY the provided complaint excerpts. 
# If the answer isn't in the context, state you don't have enough information.

# Context: {context}
# Question: {question}
# Answer:"""
# prompt = ChatPromptTemplate.from_messages([("system", system_template)])

# # Format retrieved documents
# def format_docs(docs):
#     return "\n\n".join(f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs))

# # Create RAG chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# # Example usage
# response = rag_chain.invoke("What are customers saying about late fees?")
# print(response.content)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
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

llm = ChatGroq(
    groq_api_key="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",
    model_name="mixtral-8x7b-32768",
    temperature=0.1
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
                persist_directory="./chroma_dbb",
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vectorstore.add_documents(batch)
        pbar.update(len(batch))

print(f"Vector store created in {time.time() - start_time:.2f} seconds")

# Rest of your code remains the same...
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
    }
)
# Define prompt template
system_template = """You are a financial analyst assistant for CrediTrust. 
Answer questions using ONLY the provided complaint excerpts. 
If the answer isn't in the context, state you don't have enough information.

Context: {context}
Question: {question}
Answer concisely:"""
prompt = ChatPromptTemplate.from_messages([("system", system_template)])

def format_docs(docs):
    return "\n\n".join(f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs))

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Test with detailed progress tracking
test_questions = [
    "What are customers saying about late fees?",
    "Are there complaints about credit card charges?",
    "What issues do customers report about online banking?"
]

print("\nTesting the system...")
for question in tqdm(test_questions, desc="Processing questions"):
    print(f"\nQuestion: {question}")
    start_time = time.time()
    
    # Track retrieval progress
    print("  Retrieving relevant documents...")
    ret_start = time.time()
    relevant_docs = retriever.invoke(question)
    print(f"  Retrieved {len(relevant_docs)} documents in {time.time() - ret_start:.2f}s")
    
    # Track generation progress
    print("  Generating answer...")
    gen_start = time.time()
    response = rag_chain.invoke(question)
    print(f"  Generated answer in {time.time() - gen_start:.2f}s")
    
    print(f"Answer: {response.content}")
    print(f"Total time for this question: {time.time() - start_time:.2f}s")

def get_vectorstore():
    """Load the existing Chroma vectorstore from disk"""
    vectorstore = Chroma(
        persist_directory="./chroma_dbb",
        embedding_function=embedding_model
    )
    return vectorstore