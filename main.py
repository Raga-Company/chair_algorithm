import json
import os
import time

from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever


print("=" * 50)
print("MAIN STARTED")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "cahir_train.json")


# =========================
# Load Data
# =========================
print(f"Loading data from: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"FULL DATA: {len(data)}")


USE_ALL_DATA = False

working_data = data if USE_ALL_DATA else data

print(f"WORKING DATA: {len(working_data)}")


# =========================
# Init Models
# =========================
start_time = time.time()

print("\n" + "=" * 50)
print("Initializing BM25...")
bm25 = BM25Retriever(working_data, cache_dir="cache/bm25")

print("\nInitializing Semantic...")
semantic = SemanticRetriever(
    working_data,
    cache_dir="cache/semantic",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

print("\nInitializing Hybrid...")
hybrid = HybridRetriever(bm25, semantic)

print("=" * 50)

print(f"\nInitialization time: {time.time() - start_time:.2f} seconds")


# =========================
# Query
# =========================
query = input("\nEnter your query: ").strip()

print(f"\nQuery: {query}")
print("=" * 50)


# =========================
# BM25
# =========================
print("\n========== BM25 RESULTS ==========")

start_time = time.time()
bm25_results = bm25.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(bm25_results, 1):

    question = r.get("question", "")
    answer = r.get("answer", "")
    category = r.get("category", "N/A")
    score = r.get("score", 0)

    print(f"\n{i}. Q: {question[:100]}...")
    print(f"   A: {answer[:100]}...")
    print(f"   Category: {category}")
    print(f"   Score: {score:.4f}")


# =========================
# Semantic
# =========================
print("\n========== SEMANTIC RESULTS ==========")

start_time = time.time()
semantic_results = semantic.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(semantic_results, 1):

    text = r.get("text", "")
    score = r.get("score", 0)

    print(f"\n{i}. TEXT: {text[:200]}...")
    print(f"   Score: {score:.4f}")


# =========================
# Hybrid (RRF)
# =========================
print("\n========== HYBRID RESULTS (RRF) ==========")

start_time = time.time()
hybrid_results = hybrid.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(hybrid_results, 1):

    print(f"\n{i}. Q: {r.get('question','')[:100]}...")
    print(f"   A: {r.get('answer','')[:100]}...")
    print(f"   Category: {r.get('category','N/A')}")
    print(f"   Final Score: {r.get('rerank_score',0):.4f}")


# =========================
# Weighted Hybrid
# =========================
print("\n========== HYBRID RESULTS (Weighted) ==========")

start_time = time.time()
hybrid_weighted = hybrid.search_with_weights(query, k=5, alpha=0.3)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(hybrid_weighted, 1):

    print(f"\n{i}. Q: {r.get('question','')[:100]}...")
    print(f"   A: {r.get('answer','')[:100]}...")
    print(f"   Category: {r.get('category','N/A')}")

    print(f"   Final Score: {r.get('final_score',0):.4f}")
    print(f"   BM25: {r.get('bm25_score',0):.4f}, Semantic: {r.get('semantic_score',0):.4f}")


print("\n" + "=" * 50)
print("PROCESS COMPLETED")
print("=" * 50)