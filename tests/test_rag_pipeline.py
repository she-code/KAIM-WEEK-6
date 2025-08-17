from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Mock data for testing
SAMPLE_DOCUMENTS = [
    Document(page_content="Late fee complaint", metadata={"id": 1}),
    Document(page_content="Credit card issue", metadata={"id": 2}),
]

SAMPLE_RESPONSE = AIMessage(content="The complaints mention issues with late fees.")


@pytest.fixture
def mock_retriever():
    mock = MagicMock()
    mock.invoke.return_value = SAMPLE_DOCUMENTS
    return mock


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value = SAMPLE_RESPONSE
    return mock


@pytest.fixture
def format_docs():
    def _format_docs(docs):
        return "\n\n".join(
            f"Complaint #{i+1}: {d.page_content}" for i, d in enumerate(docs)
        )

    return _format_docs


@pytest.fixture
def test_llm_invocation(mock_llm):
    test_input = [
        {"role": "system", "content": "Test context\nQuestion: Test question"}
    ]
    mock_llm.invoke(test_input)
    mock_llm.invoke.assert_called_once()


@pytest.fixture
def rag_chain(mock_retriever, mock_llm, format_docs):
    # Create the prompt template
    system_template = """You are a financial analyst assistant for CrediTrust.
    Answer questions using ONLY the provided complaint excerpts.
    If the answer isn't in the context, state you don't have enough information.

    Context: {context}
    Question: {question}
    Answer concisely:"""
    prompt = ChatPromptTemplate.from_messages([("system", system_template)])

    # Build the chain with proper mocks
    return (
        {
            "context": RunnableLambda(lambda x: format_docs(mock_retriever.invoke(x))),
            "question": RunnablePassthrough(),
        }
        | prompt
        | mock_llm
    )


def test_document_formatting(format_docs):
    """Test the document formatting function"""
    formatted = format_docs(SAMPLE_DOCUMENTS)
    assert "Complaint #1: Late fee complaint" in formatted
    assert "Complaint #2: Credit card issue" in formatted


def test_prompt_template():
    """Test the prompt template formatting"""
    system_template = """You are a financial analyst assistant for CrediTrust.
    Answer questions using ONLY the provided complaint excerpts.
    If the answer isn't in the context, state you don't have enough information.

    Context: {context}
    Question: {question}
    Answer concisely:"""
    prompt = ChatPromptTemplate.from_messages([("system", system_template)])

    formatted = prompt.format(context="Test context", question="Test question")

    assert "CrediTrust" in formatted
    assert "Test context" in formatted
    assert "Test question" in formatted
