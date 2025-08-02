# app/utils.py
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain.embeddings import HuggingFaceEmbeddings

# Directories
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db"))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            docs.extend(loader.load())
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

def save_to_faiss(chunks):
    embeddings = get_embeddings()
    vectordb = faiss.FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectordb.save_local(DB_DIR)
    return vectordb

def load_faiss_db():
    embeddings = get_embeddings()
    return faiss.FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
