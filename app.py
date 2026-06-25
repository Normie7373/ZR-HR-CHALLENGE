"""Streamlit app for the Zyro Dynamics HR RAG assistant."""

from __future__ import annotations

import os
import html
import textwrap
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


APP_TITLE = "Zyro Dynamics HR Assistant"
APP_SUBTITLE = "Answers employee questions using Zyro Dynamics HR policy documents."
DOCUMENTS_DIR = Path("documents")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions based on Zyro Dynamics HR policy documents."
)
NOT_FOUND_MESSAGE = (
    "I can only answer questions based on Zyro Dynamics HR policy documents. "
    "The requested information is not available in the provided policies."
)


class AppConfigurationError(Exception):
    """Raised when the application is missing required configuration."""


class DocumentLoadingError(Exception):
    """Raised when policy documents cannot be loaded."""


class VectorStoreCreationError(Exception):
    """Raised when the FAISS vector store cannot be created."""


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain embedding wrapper matching the original notebook pipeline."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


def configure_page() -> None:
    """Configure Streamlit page metadata and styling."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":briefcase:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 980px;
                padding-top: 2rem;
                padding-bottom: 4rem;
            }

            h1 {
                letter-spacing: 0;
                margin-bottom: 0.25rem;
            }

            .app-subtitle {
                color: #5b6472;
                font-size: 1.02rem;
                margin-bottom: 1.5rem;
            }

            [data-testid="stSidebar"] {
                background: #f7f9fc;
                border-right: 1px solid #e5e7eb;
            }

            [data-testid="stChatMessage"] {
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.75rem;
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }

            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                background: #eef6ff;
            }

            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
                background: #ffffff;
            }

            .source-card {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.85rem;
                margin-bottom: 0.75rem;
                background: #fbfcfe;
            }

            .source-title {
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.35rem;
            }

            .source-preview {
                color: #4b5563;
                font-size: 0.92rem;
                line-height: 1.45;
            }

            @media (max-width: 640px) {
                .main .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_openai_api_key() -> str:
    """Read the OpenAI API key from Streamlit secrets."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as exc:
        raise AppConfigurationError(
            "OpenAI API key is missing. Add OPENAI_API_KEY to Streamlit secrets."
        ) from exc

    if not str(api_key).strip():
        raise AppConfigurationError(
            "OpenAI API key is empty. Add a valid OPENAI_API_KEY to Streamlit secrets."
        )

    return str(api_key).strip()


@st.cache_resource(show_spinner=False)
def get_embeddings() -> SentenceTransformerEmbeddings:
    """Load and cache the BGE embedding model."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return SentenceTransformerEmbeddings(model)


def load_policy_documents() -> list[Any]:
    """Load every PDF inside the documents directory."""
    if not DOCUMENTS_DIR.exists() or not DOCUMENTS_DIR.is_dir():
        raise DocumentLoadingError(
            "The documents folder is missing. Add a documents/ directory with HR policy PDFs."
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise DocumentLoadingError(
            "No PDF files were found in documents/. Add the HR policy PDFs and restart the app."
        )

    try:
        loader = PyPDFDirectoryLoader(str(DOCUMENTS_DIR))
        documents = loader.load()
    except Exception as exc:
        raise DocumentLoadingError(
            "The HR policy PDFs could not be loaded. Check that all PDFs are readable."
        ) from exc

    if not documents:
        raise DocumentLoadingError(
            "The PDF loader did not return any content from the policy documents."
        )

    return documents


@st.cache_resource(show_spinner=False)
def create_vector_store() -> FAISS:
    """Create and cache the FAISS vector store from policy PDF chunks."""
    try:
        documents = load_policy_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise VectorStoreCreationError(
                "No text chunks were created from the HR policy documents."
            )

        return FAISS.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
        )
    except VectorStoreCreationError:
        raise
    except DocumentLoadingError:
        raise
    except Exception as exc:
        raise VectorStoreCreationError(
            "The vector store could not be created from the policy documents."
        ) from exc


def create_retriever(vector_store: FAISS) -> Any:
    """Create the same MMR retriever configuration used in the notebook."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )


@st.cache_resource(show_spinner=False)
def create_llm(api_key: str) -> ChatOpenAI:
    """Create and cache GPT-4o Mini."""
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
    )


def format_docs(docs: list[Any]) -> str:
    """Format retrieved documents exactly as in the original notebook."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )


def create_prompt() -> ChatPromptTemplate:
    """Create the RAG prompt from the notebook."""
    return ChatPromptTemplate.from_template(
        """
You are the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Your job is to answer employee questions ONLY using the retrieved HR policy documents.

### Instructions

- Answer ONLY using the provided context.
- Never use outside knowledge or make assumptions.
- If multiple retrieved passages contain useful information, combine them into a single clear answer.
- Prefer answering whenever the retrieved context contains enough relevant information.
- Include important policy details such as eligibility, timelines, approvals, limits, conditions, and exceptions whenever available.
- Keep answers concise, professional, and easy to understand.
- Do not mention that you are an AI or language model.
- Do not reference "the context" or "the document" in your answer.

If the retrieved context does NOT contain enough information to answer the question, respond exactly with:

"I can only answer questions based on Zyro Dynamics HR policy documents. The requested information is not available in the provided policies."

--------------------
Context:
{context}
--------------------

Question:
{question}

Answer:
"""
    )


def create_guardrail_prompt() -> ChatPromptTemplate:
    """Create the out-of-scope classifier prompt from the notebook."""
    return ChatPromptTemplate.from_template(
        """
You are an intent classifier.

Determine whether the user's question can be answered using Zyro Dynamics HR policy documents.

HR topics include:
- Leave policy
- Work From Home
- Employee handbook
- Code of conduct
- Performance reviews
- Compensation & Benefits
- IT & Data Security
- POSH
- Onboarding & Separation
- Travel & Expense
- Company policies
- HR processes
- Employee benefits
- Probation
- Salary structure
- Reimbursements

If the question belongs to these topics, respond with ONLY:
YES

Otherwise respond with ONLY:
NO

Question:
{question}
"""
    )


def create_rag_chain(retriever: Any, llm: ChatOpenAI) -> Any:
    """Create the notebook's LangChain RAG pipeline."""
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | create_prompt()
        | llm
        | StrOutputParser()
    )


def create_guardrail_chain(llm: ChatOpenAI) -> Any:
    """Create the notebook's guardrail classifier chain."""
    return create_guardrail_prompt() | llm | StrOutputParser()


@st.cache_resource(show_spinner=False)
def initialize_rag(api_key: str) -> dict[str, Any]:
    """Initialize and cache all RAG resources."""
    vector_store = create_vector_store()
    retriever = create_retriever(vector_store)
    llm = create_llm(api_key)

    return {
        "retriever": retriever,
        "rag_chain": create_rag_chain(retriever, llm),
        "guardrail_chain": create_guardrail_chain(llm),
    }


def clean_preview(text: str, width: int = 420) -> str:
    """Create a compact chunk preview for source display."""
    compact_text = " ".join(text.split())
    return textwrap.shorten(compact_text, width=width, placeholder="...")


def build_sources(docs: list[Any]) -> list[dict[str, str]]:
    """Build source metadata for the UI."""
    sources = []
    for doc in docs:
        source_path = doc.metadata.get("source", "Unknown")
        sources.append(
            {
                "filename": Path(source_path).name,
                "preview": clean_preview(doc.page_content),
            }
        )
    return sources


def answer_question(question: str, rag: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Run guardrails, retrieval, and generation for a user question."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question before sending.")

    try:
        decision = rag["guardrail_chain"].invoke({"question": question}).strip().upper()
    except Exception as exc:
        raise RuntimeError(
            "The question could not be checked against the HR policy scope."
        ) from exc

    if decision != "YES":
        return OUT_OF_SCOPE_MESSAGE, []

    try:
        retrieved_docs = rag["retriever"].invoke(question)
    except Exception as exc:
        raise RuntimeError(
            "The policy documents could not be searched for this question."
        ) from exc

    try:
        answer = rag["rag_chain"].invoke(question)
    except Exception as exc:
        raise RuntimeError(
            "The assistant could not generate a response. Please try again."
        ) from exc

    answer = answer.strip() or NOT_FOUND_MESSAGE
    return answer, build_sources(retrieved_docs)


def initialize_session_state() -> None:
    """Initialize chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. Ask me about Zyro Dynamics HR policies, benefits, "
                    "leave, onboarding, conduct, IT security, travel, or expenses."
                ),
                "sources": [],
                "show_sources": False,
            }
        ]


def render_sources(sources: list[dict[str, str]]) -> None:
    """Render retrieved source chunks in an expandable section."""
    with st.expander("Retrieved sources", expanded=False):
        if not sources:
            st.caption("No sources were retrieved for this response.")
            return

        for index, source in enumerate(sources, start=1):
            filename = html.escape(source["filename"])
            preview = html.escape(source["preview"])
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{index}. {filename}</div>
                    <div class="source-preview">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    """Render all chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("show_sources", True):
                render_sources(message.get("sources", []))


def render_sidebar() -> None:
    """Render sidebar controls."""
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Model")
        st.write(LLM_MODEL)
        st.caption("Embeddings")
        st.write(EMBEDDING_MODEL)
        st.caption("Retrieval")
        st.write("MMR, k=6, fetch_k=20, lambda=0.5")


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    initialize_session_state()
    render_sidebar()

    st.title(APP_TITLE)
    st.markdown(f'<p class="app-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

    try:
        api_key = get_openai_api_key()
    except AppConfigurationError as exc:
        st.error(str(exc))
        st.info(
            "For Streamlit Community Cloud, add the key under app settings > Secrets."
        )
        render_chat_history()
        return

    try:
        with st.spinner("Preparing the HR policy knowledge base..."):
            rag = initialize_rag(api_key)
    except (DocumentLoadingError, VectorStoreCreationError) as exc:
        st.error(str(exc))
        render_chat_history()
        return
    except Exception as exc:
        st.error(
            "The assistant could not start. Check the app logs for setup details."
        )
        st.exception(exc)
        render_chat_history()
        return

    render_chat_history()

    user_question = st.chat_input("Ask an HR policy question")
    if user_question is None:
        return

    cleaned_question = user_question.strip()
    if not cleaned_question:
        st.warning("Please enter a question before sending.")
        return

    st.session_state.messages.append(
        {"role": "user", "content": cleaned_question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(cleaned_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching HR policies and drafting an answer..."):
                response, sources = answer_question(cleaned_question, rag)
        except Exception as exc:
            response = str(exc)
            sources = []
            st.error(response)
        else:
            st.markdown(response)
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "show_sources": True,
        }
    )


if __name__ == "__main__":
    main()
