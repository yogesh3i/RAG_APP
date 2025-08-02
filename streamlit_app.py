# ui_app.py
import streamlit as st
from app.rag_pipeline import main as main
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
import os

DB_FAISS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "db"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cache to avoid reloading everything on every rerun
@st.cache_resource
def load_chain():
    st.info("Loading vector store and LLM. Please wait...", icon="⏳")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="gemma:2b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# UI
st.set_page_config(page_title="RAG with Gemma", page_icon="🤖", layout="centered")
st.title("RAG Question Answering using Gemma")

user_query = st.text_input("Ask a question based on your documents:")

if user_query:
    chain = load_chain()
    with st.spinner("Generating answer..."):
        result = chain({"query": user_query})

    st.subheader("Answer:")
    st.write(result['result'])
    st.snow()
    st.subheader("Sources:")
    for doc in result["source_documents"]:
        st.markdown(f"- `{doc.metadata['source']}`")
