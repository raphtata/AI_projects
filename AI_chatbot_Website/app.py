
import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
#from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()


#load the groq API KEY
groq_api_key = os.environ["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key = groq_api_key, 
               model_name = "mixtral-8x7b-32768")

def get_response(user_input):
        #create chain conversation
    retriever_chain = get_context_retriever_chain(st.session_state.vectore_store)

    conversation_rag_chain= get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
            "chat_history" : st.session_state.chat_history,
            "input" : user_query
        })

    return response['answer']

def get_vectorstore_from_url(url):
    #get the text in vectorStore
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(document_chunks,  OllamaEmbeddings(model="gemma:2b"))
    return vector_store

def get_context_retriever_chain(vectore_store):

    
    retriever = vectore_store.as_retriever()

    prompt=ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")

    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based only on the context provided:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm , prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    return retrieval_chain


# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")



# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


#input user

if website_url is  None or website_url == "":
    st.info("Please enter a website URL")

else:

    #config to initialize session_state
    if "chat_history" not in st.session_state:    
        st.session_state.chat_history = [
            AIMessage(content = "Hello , i am a bot . How can i help you?")
        ]


    if "vectore_store" not in st.session_state:
        st.session_state.vectore_store = get_vectorstore_from_url(website_url)
    
    user_query = st.chat_input("type your message here ...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

        #st.write(response)
        st.session_state.chat_history.append(HumanMessage(content= user_query))
        st.session_state.chat_history.append(AIMessage(content= response))


    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
            

