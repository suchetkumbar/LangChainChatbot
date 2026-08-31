"""Local Ollama chatbot with persistence, streaming, and optional document context."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

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
MODEL_OPTIONS = ["llama3.1", "llama3.2", "mistral", "gemma2", "qwen2.5", "phi3"]


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
    if HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    raise RuntimeError("Install sentence-transformers to use document retrieval with Ollama.")


def build_rag_index(uploaded_files: list[Any]) -> Any:
    if FAISS is None:
        raise RuntimeError("FAISS is not available in the installed dependencies.")

    docs: list[Document] = []
    for uploaded_file in uploaded_files:
        raw = uploaded_file.read()
        file_name = getattr(uploaded_file, "name", "document")
        text = raw.decode("utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": file_name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_documents = splitter.split_documents(docs)
    return FAISS.from_documents(split_documents, get_embeddings())


def get_relevant_context(query: str) -> str:
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        return ""
    documents = vector_store.similarity_search(query, k=4)
    return "\n\n".join(doc.page_content for doc in documents)


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


def main() -> None:
    st.set_page_config(page_title="Ollama Chatbot", page_icon="🦙", layout="wide")
    st.title("Local Ollama Chatbot")
    st.caption("Fast offline AI with chat persistence, streaming, and optional document search.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    with st.sidebar:
        model = st.selectbox("Model", MODEL_OPTIONS, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=4000, value=1000, step=64)
        system_prompt = st.text_area(
            "System prompt",
            value="You are a helpful local assistant. Use the attached context when relevant.",
            height=120,
        )
        st.session_state.rag_enabled = st.checkbox("Use document context", value=st.session_state.rag_enabled)

        uploaded_files = st.file_uploader(
            "Upload local documents",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            try:
                st.session_state.vector_store = build_rag_index(list(uploaded_files))
                st.success(f"Loaded {len(uploaded_files)} file(s) for RAG.")
            except Exception as exc:  # pragma: no cover
                st.error(f"Unable to load files: {exc}")
                st.session_state.vector_store = None

        st.button("Clear chat", on_click=lambda: st.session_state.__setitem__("chat_history", []))
        st.caption("History is stored in .chat_history.json")

    render_chat_history(st.session_state.chat_history)

    if prompt := st.chat_input("Ask your local model"):
        try:
            llm = ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)
            context = get_relevant_context(prompt) if st.session_state.rag_enabled else ""
            question_for_model = prompt if not context else f"Context:\n{context}\n\nQuestion:\n{prompt}"

            history_messages = to_langchain_history(st.session_state.chat_history)
            chain = build_prompt(system_prompt) | llm | StrOutputParser()

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
    main()
