from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import streamlit as st

MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "vector_store_database"
COLLECTION_NAME = "vector_store_collection"
ATLAS_VECTOR_SEARCH_NAME = "vector_index_ghw_aiml"

def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="model/embeddings-001")

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_NAME)

    return vector_store
