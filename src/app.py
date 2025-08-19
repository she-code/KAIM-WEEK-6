import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
)


# Load vectorstore
@st.cache_resource
def get_vectorstore():
    """Load the existing Chroma vectorstore from disk"""
    vectorstore = Chroma(
        persist_directory="../vector_store", embedding_function=embedding_model
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
        temperature=0.1,
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
    return "\n\n".join(
        f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs)
    )


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
        "messages": [],
    }

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# Main app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a mode:", ["Chat", "Analytics Dashboard"])

if app_mode == "Chat":
    # Sidebar for conversation history
    with st.sidebar:
        st.title("Conversation History")

        # Button to create new chat
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_id = str(time.time())
            st.session_state.current_conversation = new_chat_id
            st.session_state.conversations[new_chat_id] = {
                "title": f"Chat {len(st.session_state.conversations) + 1}",
                "messages": [],
            }
            st.session_state.input_value = ""
            st.rerun()

        st.divider()

        # Display conversation list
        for conv_id in list(st.session_state.conversations.keys())[
            ::-1
        ]:  # Show newest first
            conv = st.session_state.conversations[conv_id]
            # Truncate long titles
            display_title = (
                (conv["title"][:20] + "...")
                if len(conv["title"]) > 20
                else conv["title"]
            )

            # Button to select conversation
            if st.button(
                display_title,
                key=f"conv_{conv_id}",
                use_container_width=True,
                type=(
                    "primary"
                    if conv_id == st.session_state.current_conversation
                    else "secondary"
                ),
            ):
                st.session_state.current_conversation = conv_id
                st.rerun()

    # Main chat interface
    st.title("CrediTrust Complaint Analysis")
    st.markdown(
        "Ask questions about customer complaints, and get concise answers based "
        "on complaint excerpts."
    )

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
            key="question_input",
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
            current_conv["title"] = question[:30] + (
                "..." if len(question) > 30 else ""
            )

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
                        if (
                            not answer
                            or "don't have enough information" in answer.lower()
                        ):
                            answer = "This information is out of context."
                            answer_container.markdown(answer)
                        else:
                            answer_container.markdown(answer)

                        # Display timing info
                        st.markdown(
                            f"*Retrieved {len(relevant_docs)} documents "
                            f"in {retrieval_time:.2f}s*"
                        )
                        st.markdown(f"*Generated answer in {gen_end - gen_start:.2f}s*")

                except Exception as e:
                    if "503" in str(e):
                        answer = (
                            "Service currently unavailable. "
                            "This information may be out of context."
                        )
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
        current_conv["messages"].append(
            {"role": "assistant", "content": answer, "sources": relevant_docs}
        )

elif app_mode == "Analytics Dashboard":
    st.title("Complaints Analytics Dashboard")

    # Update the load_complaint_data function in the Analytics Dashboard section:

    @st.cache_data
    def load_complaint_data():
        """Load complaint data from the vectorstore for analysis"""
        try:
            # Get all documents and metadata from the vectorstore
            docs = vectorstore.get()
            documents = docs.get('documents', [])
            metadatas = docs.get('metadatas', [])

            # Create a DataFrame
            if metadatas and len(metadatas) == len(documents):
                df = pd.DataFrame(metadatas)
                df['complaint_text'] = documents
            else:
                df = pd.DataFrame({'complaint_text': documents})

            # If no data was loaded, create minimal sample data
            if df.empty:
                return pd.DataFrame(
                    {
                        'complaint_text': ['Sample complaint 1', 'Sample complaint 2'],
                        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
                        'product_category': ['Credit Card', 'Loan'],
                        'sentiment': ['Negative', 'Neutral'],
                    }
                )

            # Check if we have real metadata or need to create sample data
            has_real_metadata = False
            if not df.empty:
                # Check if we have any meaningful metadata columns beyond complaint_text
                metadata_cols = [col for col in df.columns if col != 'complaint_text']
                if metadata_cols and not df[metadata_cols].isnull().all().all():
                    has_real_metadata = True

            if not has_real_metadata:
                # Create sample data with a manageable size
                sample_size = min(1000, len(df))
                st.warning(
                    f"Using sample data ({sample_size} entries) -\n"
                    f"real metadata not found or invalid in vectorstore"
                )

                sample_df = pd.DataFrame(
                    {
                        'complaint_text': df['complaint_text'].head(sample_size),
                        'date': pd.date_range(
                            start='2023-01-01', periods=sample_size, freq='D'
                        ),
                        'product_category': [
                            'Credit Card',
                            'Loan',
                            'Mortgage',
                            'Bank Account',
                        ]
                        * (sample_size // 4 + 1),
                        'sentiment': ['Negative', 'Neutral', 'Positive']
                        * (sample_size // 3 + 1),
                    }
                )
                return sample_df.head(sample_size)

            # Process real metadata data
            # Handle date conversion safely
            if 'date' in df.columns:
                # Convert problematic date values safely
                df['date'] = df['date'].apply(
                    lambda x: (
                        pd.to_datetime(x, errors='coerce', unit='ms')
                        if isinstance(x, (int, float)) and x > 1e12
                        else pd.to_datetime(x, errors='coerce')
                    )
                )
                # Drop rows with invalid dates
                df = df.dropna(subset=['date'])
            else:
                df['date'] = pd.NaT

            # Add missing required columns with default values
            if 'product_category' not in df.columns:
                df['product_category'] = 'Unknown'

            if 'sentiment' not in df.columns:
                df['sentiment'] = 'Neutral'

            # Ensure we have valid dates, if not create reasonable ones
            if df['date'].isnull().all():
                df['date'] = pd.date_range(
                    start='2023-01-01', periods=len(df), freq='D'
                )

            # Filter out extreme dates that might cause issues
            current_year = pd.Timestamp.now().year
            df = df[df['date'].dt.year.between(2000, current_year + 1)]

            # Limit to a reasonable number for performance
            if len(df) > 5000:
                df = df.sample(5000, random_state=42)
                st.info(
                    f"Showing 5,000 random complaints out of \n"
                    f"{len(documents):,} total for better performance."
                )

            return df

        except Exception as e:
            st.error(f"Error loading complaint data: {str(e)}")
            # Return minimal sample data if there's an error
            return pd.DataFrame(
                {
                    'complaint_text': ['Sample complaint 1', 'Sample complaint 2'],
                    'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
                    'product_category': ['Credit Card', 'Loan'],
                    'sentiment': ['Negative', 'Neutral'],
                }
            )

    # Load the data
    df = load_complaint_data()

    if not df.empty:
        # Filters sidebar
        st.sidebar.header("Filters")

        # Date range filter
        min_date = df['date'].min().to_pydatetime()
        max_date = df['date'].max().to_pydatetime()
        date_range = st.sidebar.date_input(
            "Date range", [min_date, max_date], min_value=min_date, max_value=max_date
        )

        # Product category filter
        all_categories = sorted(df['product_category'].unique())
        selected_categories = st.sidebar.multiselect(
            "Product Categories", all_categories, default=all_categories
        )

        # Sentiment filter
        all_sentiments = sorted(df['sentiment'].unique())
        selected_sentiments = st.sidebar.multiselect(
            "Sentiments", all_sentiments, default=all_sentiments
        )

        # Apply filters
        filtered_df = df[
            (df['date'].dt.date >= date_range[0])
            & (df['date'].dt.date <= date_range[1])
            & (df['product_category'].isin(selected_categories))
            & (df['sentiment'].isin(selected_sentiments))
        ]

        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Complaints", len(filtered_df))
        col2.metric(
            "Most Common Category",
            (
                filtered_df['product_category'].mode()[0]
                if not filtered_df.empty
                else "N/A"
            ),
            help="The most frequently occurring product category in complaints",
        )
        col3.metric(
            "Dominant Sentiment",
            filtered_df['sentiment'].mode()[0] if not filtered_df.empty else "N/A",
            help="The most common sentiment in complaints",
        )
        col4.metric(
            "Avg Complaint Length",
            (
                f"{filtered_df['complaint_text'].str.len().mean():.0f} chars"
                if not filtered_df.empty
                else "N/A"
            ),
            help="Average length of complaint text in characters",
        )

        st.divider()

        # Complaint volume over time
        st.subheader("Complaint Volume Over Time")
        time_df = (
            filtered_df.set_index('date').resample('D').size().reset_index(name='count')
        )
        fig1 = px.line(
            time_df,
            x='date',
            y='count',
            title='Daily Complaint Volume',
            labels={'date': 'Date', 'count': 'Number of Complaints'},
        )
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        # Most common complaint types
        st.subheader("Complaints by Product Category")
        category_counts = filtered_df['product_category'].value_counts().reset_index()
        fig2 = px.bar(
            category_counts,
            x='product_category',
            y='count',
            title='Complaints by Product Category',
            labels={
                'product_category': 'Product Category',
                'count': 'Number of Complaints',
            },
            color='product_category',
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Sentiment distribution
        st.subheader("Sentiment Analysis")
        sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
        fig3 = px.pie(
            sentiment_counts,
            names='sentiment',
            values='count',
            title='Sentiment Distribution',
            hole=0.3,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Trending topics (sample implementation)
        st.subheader("Recent Trends")
        tab1, tab2 = st.tabs(["Recent Complaints", "Complaint Details"])

        with tab1:
            st.write("Latest 10 complaints matching filters:")
            recent_complaints = filtered_df.sort_values('date', ascending=False).head(
                10
            )
            for idx, row in recent_complaints.iterrows():
                with st.expander(
                    f"{row['date'].strftime('%Y-%m-%d')} - {row['product_category']}"
                ):
                    st.write(
                        row['complaint_text'][:500]
                        + ("..." if len(row['complaint_text']) > 500 else "")
                    )
                    st.caption(f"Sentiment: {row['sentiment']}")

        with tab2:
            st.write("Detailed complaint data:")
            st.dataframe(
                filtered_df[
                    ['date', 'product_category', 'sentiment', 'complaint_text']
                ].sort_values('date', ascending=False),
                column_config={
                    "date": st.column_config.DateColumn("Date"),
                    "product_category": "Product",
                    "sentiment": "Sentiment",
                    "complaint_text": "Complaint Text",
                },
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning(
            "No complaint data available for analysis. Please check your data source."
        )
