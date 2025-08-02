import os

folders = [
    "./app",
    "./data",
    "./db"
]

files = {
    "./app/__init__.py": "",
    "./app/rag_pipeline.py": "",
    "./app/embedder.py": "",
    "./app/retriever.py": "",
    "./app/generator.py": "",
    "./app/utils.py": "",
    "./streamlit_app.py": "",
    "./requirements.txt": "",
    "./README.md": "",
    "./config.yaml": "",
    "./db/faiss_index.pkl": None  # Optional placeholder
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for filepath, content in files.items():
    if content is not None:
        with open(filepath, "w") as f:
            f.write(content)
    else:
        open(filepath, "a").close()

print("RAG project structure created successfully!")
