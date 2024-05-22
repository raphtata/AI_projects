import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS  #similarity search vectorestore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #embedding technique
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

from dotenv import load_dotenv


#load the groq API KEY
groq_api_key = os.environ["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma model Document for Q&A project")
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


def vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./docs") #data pdf

        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_pliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200  )
        st.session_state.final_documents = st.session_state.text_pliter.split_documents(st.session_state.docs) #take initial 40 documents

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector store

prompt1 = st.text_input("what do you want to know about these documents?")

if st.button("creatinf Vector Store"):
    vector_embedding()
    st.write("vector Store DB is ready")



if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start =time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

        # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")