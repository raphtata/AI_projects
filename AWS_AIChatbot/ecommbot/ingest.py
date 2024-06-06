from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from ecommbot.data_converter import dataconverter
from langchain_community.vectorstores import AstraDB

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
api_key = os.getenv('LANGCHAIN_API_KEY')
if api_key is None:
    raise ValueError("La clé API LANGCHAIN_API_KEY n'a pas été définie dans les variables d'environnement.")

print("-------")




OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
desired_namespace=os.getenv("ASTRA_DB_KEYSPACE")

embedding = OllamaEmbeddings(model="gemma:2b")

def ingestdata(status):

    if desired_namespace:
        ASTRA_DB_KEYSPACE = desired_namespace
    else:
        ASTRA_DB_KEYSPACE = None
    vstore = AstraDB(
            embedding=embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    
    storage=status
    
    if storage==None:
        docs=dataconverter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__=='__main__':
    vstore,inserted_ids=ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
            print(f"* {res.page_content} [{res.metadata}]")
            