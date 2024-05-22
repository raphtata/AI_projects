import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import conversational_retrieval
from langchain.chains import create_retrieval_chain

from langchain_community.vectorstores import FAISS
import time 

from dotenv import load_dotenv
load_dotenv()

#load the groq API KEY
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")

    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_pliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 100  )
    st.session_state.final_documents = st.session_state.text_pliter.split_documents(st.session_state.docs[:40]) #take initial 40 documents

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq LLM")
llm = ChatGroq(groq_api_key = groq_api_key, 
               model_name = "Gemma-7b-it")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Imput your prompt here")

if prompt: 
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print(" time : ", (time.process_time() - (start)))
    st.write(response['answer'])

        # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")