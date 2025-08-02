import pandas as pd
import time
from rouge_score import rouge_scorer
from typing import List
from tqdm import tqdm

# === Load your RAG app === #
from app.rag_pipeline import build_qa_chain

qa_chain = build_qa_chain()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# === Load your CSV === #
data = pd.read_csv("doc.csv")
data.dropna(subset=['question', 'reference'], inplace=True)

# Initialize columns
data["predicted_answer"] = ""
data["rougeL"] = 0.0
data["latency"] = 0.0
data["precision@k"] = 0.0

# === Helper functions === #
def evaluate_rouge(pred: str, ref: str) -> float:
    scores = scorer.score(ref, pred)
    return scores['rougeL'].fmeasure

def precision_at_k(retrieved_docs: List[str], ground_truth: str, k: int = 3) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant = sum(1 for doc in retrieved_k if ground_truth.lower() in doc.lower())
    return relevant / k

# === Evaluation Loop === #
for idx, row in tqdm(data.iterrows(), total=len(data)):
    query = row["question"]
    ground_truth = row["reference"]

    start = time.time()
    try:
        output = qa_chain.run(query)
    except Exception as e:
        print(f"Error for query: {query}")
        print(e)
        continue
    end = time.time()

    # Handle different output formats
    if isinstance(output, dict):
        answer = output.get("result", "")
        source_docs = output.get("source_documents", [])
        retrieved_docs = [doc.page_content for doc in source_docs]
    else:
        answer = output
        retrieved_docs = []

    # Evaluate
    rouge_score = evaluate_rouge(answer, ground_truth)
    latency = end - start
    p_at_k = precision_at_k(retrieved_docs, ground_truth, k=3) if retrieved_docs else 0

    # Save results
    data.at[idx, "predicted_answer"] = answer
    data.at[idx, "rougeL"] = rouge_score
    data.at[idx, "latency"] = latency
    data.at[idx, "precision@k"] = p_at_k

# === Summary === #
print("\n=== Evaluation Summary ===")
print(f"Avg ROUGE-L: {data['rougeL'].mean():.4f}")
print(f"Avg Latency: {data['latency'].mean():.2f} sec")
print(f"Avg Precision@3: {data['precision@k'].mean():.2f}")

# === Save Results === #
data.to_csv("rag_evaluation_results.csv", index=False)
print("Saved to rag_evaluation_results.csv")
