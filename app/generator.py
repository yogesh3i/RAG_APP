# app/generator.py
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from utils import load_faiss_db

def load_llm():
    print("Loading Gemma 2B model via Ollama...")
    return OllamaLLM(model="gemma:2b")

def main():
    print("Loading FAISS vector store...")
    db = load_faiss_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = load_llm()

    print("Building RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        print("Generating answer...")
        result = qa_chain({"query": query})

        print("\nAnswer:")
        print(result['result'])

        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
