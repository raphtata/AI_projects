from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

import streamlit as st
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
api_key = os.getenv('LANGCHAIN_API_KEY')
if api_key is None:
    raise ValueError("La clé API LANGCHAIN_API_KEY n'a pas été définie dans les variables d'environnement.")
os.environ["LANGCHAIN_API_KEY"]=api_key

##prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant"),
        ("user", "Question:{question}")

    ]
)

## streamlit framework

st.title('Langchain Demo With gemma:2b API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm 
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
