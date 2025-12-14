import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- LangSmith Tracking ----------------
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A chatbot with OpenAI"

# ---------------- Prompt ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond clearly."),
        ("human", "{question}")
    ]
)

def generate_response(question, api_key, model, temperature, max_tokens):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# ---------------- UI ----------------
st.title("🤖 Enhanced Q&A Chatbot with OpenAI")

st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password"
)

model = st.sidebar.selectbox(
    "Select OpenAI Model",
    ["gpt-4o", "gpt-4o-mini"]
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0, 1.0, 0.7
)

max_tokens = st.sidebar.slider(
    "Maximum Tokens",
    50, 500, 150
)

st.write("💬 Go ahead & ask any question")

user_input = st.text_input("Question")

if user_input and api_key:
    response = generate_response(
        user_input,
        api_key,
        model,
        temperature,
        max_tokens
    )
    st.success(response)

elif user_input:
    st.warning("⚠️ Please enter OpenAI API Key in the sidebar")

else:
    st.info("ℹ️ Please enter a question to begin")
