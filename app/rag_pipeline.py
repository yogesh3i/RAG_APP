# app/rag_pipeline.py

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from utils import load_faiss_db  # Reuse from utils.py

# === Load LLM ===
def load_llm():
    print("Loading Gemma 2B model via Ollama...")
    return OllamaLLM(model="gemma:2b")

# === Build Retriever ===
def get_retriever():
    print("Loading vector store from FAISS DB...")
    db = load_faiss_db()
    return db.as_retriever(search_kwargs={"k": 3})

# === Build the QA Chain ===
def build_qa_chain():
    retriever = get_retriever()
    llm = load_llm()
    print("Building RetrievalQA chain...")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# === Ask a Question ===
def ask_question(qa_chain, query):
    print("Generating answer...")
    return qa_chain({"query": query})

# === CLI Mode (for testing) ===
def main():
    qa_chain = build_qa_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting RAG pipeline.")
            break

        result = ask_question(qa_chain, query)

        print("\nAnswer:")
        print(result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()
