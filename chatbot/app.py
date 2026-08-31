"""Main Streamlit chatbot with multiple major model providers and a clean UI."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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

import streamlit as st

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACKING_V2", "true")
os.environ["LANGCHAIN_TRACKING_V2"] = os.getenv("LANGCHAIN_TRACKING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Ollama": ["llama3.1", "llama3.2", "mistral", "gemma2", "qwen2.5", "phi3"],
    "Anthropic": ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
    "Groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
}


def get_model_choice(provider: str) -> str:
    return PROVIDER_MODELS.get(provider, ["gpt-4o-mini"])[0]


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


def initialize_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def reset_chat() -> None:
    st.session_state.chat_history = []


def render_chat_page() -> None:
    st.set_page_config(page_title="Multi-Model Chatbot", page_icon="🤖", layout="wide")
    st.title("Multi-Model LangChain Chatbot")
    st.caption("Switch between major LLM providers and models from a single interface.")

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()))
        model = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=4000, value=1000, step=64)
        system_prompt = st.text_area(
            "System prompt",
            value="You are a helpful, clear, and concise AI assistant.",
            height=120,
        )
        st.button("Clear chat", on_click=reset_chat)

        st.markdown("### API keys")
        st.write("Set the keys for the selected provider in your .env file.")

    initialize_session()

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt := st.chat_input("Ask a question"):
        try:
            llm = build_llm(provider, model, temperature, max_tokens)
            chain = build_prompt(system_prompt) | llm | StrOutputParser()

            response = chain.invoke(
                {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history,
                }
            )

            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))

            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as exc:  # pragma: no cover
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    render_chat_page()
