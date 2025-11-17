# CPU-Based RAG System with Streamlit UI and Evaluation

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline running entirely on **CPU**, with:

- Local LLM inference using `llama-cpp-python` via Ollama
- Document retrieval via FAISS + HuggingFace embeddings
- Interactive UI using Streamlit
- Evaluation pipeline using ROUGE, F1, and Exact Match metrics

---

## Prerequisites
## System Requirements
OS: Ubuntu 20.04+ / Windows 10+

RAM: Minimum 4GB (more is better)

Python 3.10 or higher

Git installed

Ollama installed and running

Model downloaded: mistral:7b or gemma:2b

## Folder Structure 

rag_project/
│
├── app/
│   ├── rag_pipeline.py         # Contains build_qa_chain(), get_retriever(), etc.
│   ├── utils.py                # Helper functions if any
│   └── __init__.py
│
├── main.py                     # FastAPI entry point
├── data/
│   └── doc.csv                 # Source document/question data
│
├── db/                         # FAISS index directory
│
├── requirements.txt
└── README.md


 ## Step 1: Clone and Install Dependencies
### Optional: create virtual environment
python -m venv venv
conda activate venv/ # to activate       

### Install dependencies
pip install -r requirements.txt

### step 2: Install all the dependencies 
requirements.txt

### Step 3: Start Ollama Server
Pull a CPU-compatible model (e.g., mistral, gemma) # Gemma was specifically used for this task.

using: ollama pull mistral

Then run the model 
Using: ollama run mistral

### Step 4:Configure Your App

### Step 5: Run the Streamlit App
streamlit run streamlit_app.py

The app will launch in your default browser at:
http://localhost:8501


### How It Works:

User enters a query in the Streamlit app.

The retriever fetches relevant chunks from FAISS.

The generator sends context + query to a local LLM via Ollama.

Final answer is displayed to the user.

###  Deployment Notes
- Works offline (CPU-only)
- Secure by design (no external API)
- Easy to extend with other models or UIs

---

### Author

**Yogesh Bharat Deotale**  
_Data Scientist_ AI_Developer

Email: [yogeshdeotale@email.com](mailto:yogeshdeotale@email.com)  
LinkedIn: [linkedin.com/in/yogesh-deotale](https://www.linkedin.com/in/yogesh-deotale)  


