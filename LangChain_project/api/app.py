from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from langserve import add_routes
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
api_key = os.getenv('LANGCHAIN_API_KEY')
if api_key is None:
    raise ValueError("La clé API LANGCHAIN_API_KEY n'a pas été définie dans les variables d'environnement.")

print("--------")

print(api_key)
print("--------")

os.environ["LANGCHAIN_API_KEY"]=api_key

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description = "A simple API Server"
)

#add_routes(
#    app,
#    ChatOpenAI(),
#    path="/openAI"
#)

#model = ChatOpenAI()
##ollama llama
llm=Ollama(model="gemma:2b")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

#add_routes(
#    app,
#    prompt1|model,
#    path="/essay" #this part is responsible of interraction between api and model
#)


add_routes(
    app,
    prompt2|llm,
    path="/poem" #this part is responsible of interraction between api and model

)


if __name__== "__main__":
    uvicorn.run(app,host="localhost", port=8000)