"""Multi-provider LangChain chatbot with chat persistence, streaming, and local RAG support."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    FAISS = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    HuggingFaceEmbeddings = None

import streamlit as st

load_dotenv()

CHAT_HISTORY_PATH = Path(__file__).resolve().parents[1] / ".chat_history.json"
PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Ollama": ["llama3.1", "llama3.2", "mistral", "gemma2", "qwen2.5", "phi3"],
    "Anthropic": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
    "Groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
}


def load_chat_history() -> list[dict[str, str]]:
    if not CHAT_HISTORY_PATH.exists():
        return []
    try:
        with CHAT_HISTORY_PATH.open("r", encoding="utf-8") as history_file:
            data = json.load(history_file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_history(history: list[dict[str, str]]) -> None:
    with CHAT_HISTORY_PATH.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)


def to_langchain_history(history: list[dict[str, str]]) -> list[Any]:
    converted: list[Any] = []
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted


def get_embeddings() -> Any:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raise RuntimeError(
        "No embedding backend is configured. Add OPENAI_API_KEY or install sentence-transformers for local embeddings."
    )


def build_rag_index(uploaded_files: list[Any]) -> Any:
    if not uploaded_files:
        return None

    if FAISS is None:
        raise RuntimeError("FAISS is not available. Install langchain-community with FAISS support.")

    docs: list[Document] = []
    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "document")
        file_bytes = uploaded_file.read()
        suffix = Path(file_name).suffix.lower()

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install pypdf to process PDF documents.") from exc
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            docs.append(Document(page_content=text, metadata={"source": file_name}))
        else:
            text = file_bytes.decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": file_name}))

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_documents = splitter.split_documents(docs)
    return FAISS.from_documents(split_documents, get_embeddings())


def get_relevant_context(query: str) -> str:
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        return ""
    documents = vector_store.similarity_search(query, k=4)
    return "\n\n".join(doc.page_content for doc in documents)


def build_llm(provider: str, model: str, temperature: float, max_tokens: int) -> Any:
    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "Ollama":
        if ChatOllama is None:
            raise ImportError("Install langchain-ollama to use the Ollama provider.")
        return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)

    if provider == "Anthropic":
        if ChatAnthropic is None:
            raise ImportError("Install langchain-anthropic to use the Anthropic provider.")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "Groq":
        if ChatGroq is None:
            raise ImportError("Install langchain-groq to use the Groq provider.")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unsupported provider: {provider}")


def build_prompt(system_prompt: str):
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def render_chat_history(history: list[dict[str, str]]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_page() -> None:
    st.set_page_config(page_title="Exhaustive Chatbot", page_icon="🤖", layout="wide")
    st.title("Exhaustive Multi-Model Chatbot")
    st.caption("A full LangChain chat app with multiple providers, local docs, history, and streaming replies.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    with st.sidebar:
        st.header("Chat settings")
        provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0)
        model = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=4000, value=1000, step=64)
        system_prompt = st.text_area(
            "System prompt",
            value="You are a helpful, clear, and concise assistant. Use the supplied context when it is relevant.",
            height=120,
        )
        st.session_state.rag_enabled = st.checkbox("Use local document context", value=st.session_state.rag_enabled)

        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            try:
                st.session_state.vector_store = build_rag_index(list(uploaded_files))
                st.success(f"Loaded {len(uploaded_files)} document(s) for retrieval.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Unable to build document index: {exc}")
                st.session_state.vector_store = None

        st.button("Clear chat", on_click=lambda: st.session_state.__setitem__("chat_history", []))
        st.caption("History is saved locally to .chat_history.json")

    render_chat_history(st.session_state.chat_history)

    if prompt := st.chat_input("Ask a question"):
        try:
            llm = build_llm(provider, model, temperature, max_tokens)
            context = get_relevant_context(prompt) if st.session_state.rag_enabled else ""
            question_for_model = prompt if not context else f"Context:\n{context}\n\nQuestion:\n{prompt}"

            history_messages = to_langchain_history(st.session_state.chat_history)
            prompt_template = build_prompt(system_prompt)
            chain = prompt_template | llm | StrOutputParser()

            response_chunks: list[str] = []
            response_box = st.empty()
            for chunk in chain.stream({"question": question_for_model, "chat_history": history_messages}):
                text = chunk if isinstance(chunk, str) else str(chunk)
                response_chunks.append(text)
                response_box.markdown("".join(response_chunks))

            response_text = "".join(response_chunks)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            save_chat_history(st.session_state.chat_history)

            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response_text)
        except Exception as exc:  # pragma: no cover
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    render_chat_page()
