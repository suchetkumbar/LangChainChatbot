"""Local Ollama-only chatbot entry point for quick local AI experiments."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACKING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

MODEL_OPTIONS = ["llama3.1", "llama3.2", "mistral", "gemma2", "qwen2.5", "phi3"]


def main() -> None:
    st.set_page_config(page_title="Ollama Chatbot", page_icon="🦙", layout="centered")
    st.title("Local Ollama Chatbot")

    with st.sidebar:
        model = st.selectbox("Model", MODEL_OPTIONS, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=4000, value=1000, step=64)

    question = st.text_input("Ask your local model:")
    if question:
        llm = ChatOllama(model=model, temperature=temperature, num_predict=max_tokens)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful local assistant."),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        st.write(chain.invoke({"question": question}))


if __name__ == "__main__":
    main()
