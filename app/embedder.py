# app/embedder.py
from utils import load_documents, split_documents, save_to_faiss

def main():
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Generating embeddings and saving to FAISS...")
    save_to_faiss(chunks)

    print("Embedding complete!")

if __name__ == "__main__":
    main()
