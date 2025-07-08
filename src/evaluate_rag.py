
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from tqdm import tqdm
# import time
# # Rest of your code remains the same...

# embedding_model = HuggingFaceEmbeddings(
#     model_name='all-MiniLM-L6-v2',
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={
#         'normalize_embeddings': True,
#         'batch_size': 32
#     },
#     # cache_folder="../model_cache"
# )
# def get_vectorstore():
#     """Load the existing Chroma vectorstore from disk"""
#     vectorstore = Chroma(
#         persist_directory="./chroma_dbb",
#         embedding_function=embedding_model
#     )
#     return vectorstore
# vectorstore = get_vectorstore()
# retriever = vectorstore.as_retriever(
#     search_kwargs={
#         "k": 5,
#     }
# )
# llm = ChatGroq(
#     groq_api_key="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",
#     model_name="mixtral-8x7b-32768",
#     temperature=0.1
# )
# # Define prompt template
# system_template = """You are a financial analyst assistant for CrediTrust. 
# Answer questions using ONLY the provided complaint excerpts. 
# If the answer isn't in the context, state you don't have enough information.

# Context: {context}
# Question: {question}
# Answer concisely:"""
# prompt = ChatPromptTemplate.from_messages([("system", system_template)])

# def format_docs(docs):
#     return "\n\n".join(f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs))

# # Create RAG chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )

# # Test with detailed progress tracking
# test_questions = [
#     "What are customers saying about late fees?",
#     "Are there complaints about credit card charges?",
#     "What issues do customers report about online banking?"
# ]

# print("\nTesting the system...")
# for question in tqdm(test_questions, desc="Processing questions"):
#     print(f"\nQuestion: {question}")
#     start_time = time.time()
    
#     # Track retrieval progress
#     print("  Retrieving relevant documents...")
#     ret_start = time.time()
#     relevant_docs = retriever.invoke(question)
#     print(f"  Retrieved {len(relevant_docs)} documents in {time.time() - ret_start:.2f}s")
    
#     # Track generation progress
#     print("  Generating answer...")
#     gen_start = time.time()
#     response = rag_chain.invoke(question)
#     print(f"  Generated answer in {time.time() - gen_start:.2f}s")
    
#     print(f"Answer: {response.content}")
#     print(f"Total time for this question: {time.time() - start_time:.2f}s")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import time

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32
    }
)

def get_vectorstore():
    """Load the existing Chroma vectorstore from disk"""
    vectorstore = Chroma(
        persist_directory="./chroma_dbb",
        embedding_function=embedding_model
    )
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
    }
)

# Updated Groq model - choose one of these:
llm = ChatGroq(
    groq_api_key="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",
    # model_name="mixtral-8x7b-32768",  # Try this first
   model_name="llama3-70b-8192",   # Alternative if the first doesn't work
    temperature=0.1
)

# Rest of your code remains the same...
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

# Test questions
test_questions = [
    "What are customers saying about late fees?",
    "Are there complaints about credit card charges?",
    "What issues do customers report about online banking?",
    "How are credit card disputes usually handled?",
    "What are the most common issues with mortgage services?",
    "How long does it typically take to resolve a complaint?",
    "What kind of problems do customers have with student loans?",
    "Are there recurring issues with credit reporting?",
    "Do customers face problems after paying off loans?",
    "What are common complaints about debt collectors?",
    "How do customers describe auto loan issues?",
    "What kinds of errors do people report in credit reports?",
    "Are there complaints related to loan application denials?"
]

print("\nTesting the system...")
for question in tqdm(test_questions, desc="Processing questions"):
    print(f"\nQuestion: {question}")
    start_time = time.time()
    
    print("  Retrieving relevant documents...")
    ret_start = time.time()
    relevant_docs = retriever.invoke(question)
    print(f"  Retrieved {len(relevant_docs)} documents in {time.time() - ret_start:.2f}s")
    # Display the retrieved documents
    print("  Retrieved Documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  Document #{i}:")
        print(f"  {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
        print("-" * 50)
    
    print("  Generating answer...")
    gen_start = time.time()
    response = rag_chain.invoke(question)
    print(f"  Generated answer in {time.time() - gen_start:.2f}s")
    
    print(f"Answer: {response.content}")
    print(f"Total time for this question: {time.time() - start_time:.2f}s")