from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=2000)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_question:
    st.write(chain.invoke({"question": input_question}))