from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true" #langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_template(
    [
        ("system", "You are a helpful assistant please answer the user question."),
        ("user", "Question=> {question}"),
    ]
)

# streamlit tracking
st.title("LangChain Chatbot")
input_question = st.text_input("Enter your question here:")

# model
llm = Ollama(model="llama2", temperature=0.7)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser