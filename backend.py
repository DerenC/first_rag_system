from pymongo import MongoClient

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "vector_store_database"
COLLECTION_NAME = "vector_store_collection"
ATLAS_VECTOR_SEARCH_NAME = "vector_index_ghw_aiml"
USE_GEMINI_MODEL = True
# USE_GEMINI_MODEL = False

def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="model/embeddings-001") if USE_GEMINI_MODEL \
            else HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_NAME
    )
    return vector_store

def ingest_text(text_content):
    vector_store = get_vector_store()
    docs = Document(text_content)
    vector_store.add_documents([docs])
    return True

def get_rag_response(query):
    vector_store = get_vector_store()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k" : 3})

    ## CREATING THE PROMPT TEMPLATE ##

    # (1)
#     prompt_template = """
# Use the following context from the user in order to provide an accurate answer.
# """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # (2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer:\n\n{context}"),
        ("human", "{question}")
    ])

    ## CREATING THE CHAIN ##

    # (1)
#     qna_chain = RetrievalQA.from_chain_type(llm=llm, chain_stuff="stuff", retriever=retriever

    # (2)
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | llm | StrOutputParser()


    ## INVOKING THE CHAIN ##

    # (1)
    # response = qna_chain.invoke({"query" : query})
    # return response

    # (2)
    answer = chain.invoke({"context": context_text, "question": query})
    return {
        "answer": answer,
        "sources": docs,
    }
