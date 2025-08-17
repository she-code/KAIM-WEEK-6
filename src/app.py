
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32
    }
)

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    """Load the existing Chroma vectorstore from disk"""
    vectorstore = Chroma(
        persist_directory="../vector_store",
        embedding_function=embedding_model
    )
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
    }
)

# Initialize Groq LLM with error handling
try:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROK_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0.1
    )
    # Test connection
    llm.invoke("Test connection")
except Exception as e:
    st.error(f"Failed to initialize Groq: {str(e)}")
    st.stop()

# Define prompt template
system_template = """You are a financial analyst assistant for CrediTrust. 
Answer questions using ONLY the provided complaint excerpts. 
If the answer isn't in the context, state you don't have enough information.

Context: {context}
Question: {question}
Answer concisely:"""
prompt = ChatPromptTemplate.from_messages([("system", system_template)])

# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs))

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Initialize session state for conversation management
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    st.session_state.current_conversation = str(time.time())
    st.session_state.conversations[st.session_state.current_conversation] = {
        "title": "New Chat",
        "messages": []
    }

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# Sidebar for conversation history
with st.sidebar:
    st.title("Conversation History")
    
    # Button to create new chat
    if st.button("➕ New Chat", use_container_width=True):
        new_chat_id = str(time.time())
        st.session_state.current_conversation = new_chat_id
        st.session_state.conversations[new_chat_id] = {
            "title": f"Chat {len(st.session_state.conversations) + 1}",
            "messages": []
        }
        st.session_state.input_value = ""
        st.rerun()
    
    st.divider()
    
    # Display conversation list
    for conv_id in list(st.session_state.conversations.keys())[::-1]:  # Show newest first
        conv = st.session_state.conversations[conv_id]
        # Truncate long titles
        display_title = (conv["title"][:20] + "...") if len(conv["title"]) > 20 else conv["title"]
        
        # Button to select conversation
        if st.button(
            display_title,
            key=f"conv_{conv_id}",
            use_container_width=True,
            type="primary" if conv_id == st.session_state.current_conversation else "secondary"
        ):
            st.session_state.current_conversation = conv_id
            st.rerun()

# Main app
st.title("CrediTrust Complaint Analysis")
st.markdown("Ask questions about customer complaints, and get concise answers based on complaint excerpts.")

# Get current conversation
current_conv = st.session_state.conversations[st.session_state.current_conversation]

# Display current conversation
for message in current_conv["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"][:2], 1):
                    st.markdown(f"**Document #{i}:**")
                    st.markdown(doc.page_content)
                    st.markdown(f"**Metadata:** {doc.metadata}")
                    st.markdown("---")

# Input form
with st.form(key="question_form"):
    question = st.text_input(
        "Enter your question about customer complaints:", 
        placeholder="e.g., What are customers saying about late fees?",
        value=st.session_state.input_value,
        key="question_input"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.form_submit_button("Submit")
    with col2:
        clear_button = st.form_submit_button("Clear")

# Handle clear button
if clear_button:
    current_conv["messages"] = []
    st.session_state.input_value = ""
    st.rerun()

# Handle question submission
if submit_button and question:
    # Store the question in session state
    st.session_state.input_value = question
    
    # Add user question to history
    current_conv["messages"].append({"role": "user", "content": question})
    
    # Update conversation title if it's the first message
    if len(current_conv["messages"]) == 1:
        current_conv["title"] = question[:30] + ("..." if len(question) > 30 else "")
    
    with st.chat_message("user"):
        st.markdown(question)

    # Process question
    with st.chat_message("assistant"):
        answer_container = st.empty()
        answer = ""
        relevant_docs = []
        
        try:
            with st.spinner("Retrieving relevant documents..."):
                start_time = time.time()
                relevant_docs = retriever.invoke(question)
                retrieval_time = time.time() - start_time

            try:
                with st.spinner("Generating answer..."):
                    gen_start = time.time()
                    for chunk in rag_chain.stream(question):
                        answer += chunk.content
                        answer_container.markdown(answer + "▌")
                    gen_end = time.time()
                    
                    # Check if answer indicates no context
                    if not answer or "don't have enough information" in answer.lower():
                        answer = "This information is out of context."
                        answer_container.markdown(answer)
                    else:
                        answer_container.markdown(answer)
                    
                    # Display timing info
                    st.markdown(f"*Retrieved {len(relevant_docs)} documents in {retrieval_time:.2f}s*")
                    st.markdown(f"*Generated answer in {gen_end - gen_start:.2f}s*")

            except Exception as e:
                if "503" in str(e):
                    answer = "Service currently unavailable. This information may be out of context."
                else:
                    answer = "This information is out of context."
                answer_container.markdown(answer)
                st.error(f"Error: {str(e)}")

        except Exception as e:
            answer = "This information is out of context."
            answer_container.markdown(answer)
            st.error(f"Error: {str(e)}")

        # Display sources if available
        if relevant_docs:
            with st.expander("View Sources"):
                for i, doc in enumerate(relevant_docs[:2], 1):
                    st.markdown(f"**Document #{i}:**")
                    st.markdown(doc.page_content)
                    st.markdown(f"**Metadata:** {doc.metadata}")
                    st.markdown("---")

    # Add assistant response to history
    current_conv["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": relevant_docs
    })