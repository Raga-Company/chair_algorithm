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

# بارگذاری داده
print(f"Loading data from: {file_path}")
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"FULL DATA: {len(data)}")

# انتخاب زیرمجموعه یا کل داده
# برای تست سریع از 5000 نمونه استفاده کنید
# USE_ALL_DATA = True  # برای استفاده از کل داده
USE_ALL_DATA = False

if USE_ALL_DATA:
    working_data = data
else:
    working_data = data

print(f"WORKING DATA: {len(working_data)}")

# زمان‌سنجی
start_time = time.time()

# ساخت Retrievers
print("\n" + "=" * 50)
bm25 = BM25Retriever(working_data, cache_dir="cache/bm25")
print("=" * 50)

semantic = SemanticRetriever(
    working_data,
    cache_dir="cache/semantic",
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # برای سرعت بیشتر از مدل سبکتر استفاده کنید:
    # model_name="distiluse-base-multilingual-cased-v2"
)
print("=" * 50)

hybrid = HybridRetriever(bm25, semantic)

print(f"\nInitialization time: {time.time() - start_time:.2f} seconds")

# کوئری تست
query = "پریودی"
print(f"\nQuery: {query}")
print("=" * 50)

# تست BM25
print("\n========== BM25 RESULTS ==========")
start_time = time.time()
bm25_results = bm25.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(bm25_results, 1):
    print(f"\n{i}. Q: {r['question'][:100]}...")
    print(f"   A: {r['answer'][:100]}...")
    print(f"   Category: {r['category']}")
    print(f"   Score: {r['score']:.4f}")

# تست Semantic
print("\n========== SEMANTIC RESULTS ==========")
start_time = time.time()
semantic_results = semantic.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(semantic_results, 1):
    print(f"\n{i}. Q: {r['question'][:100]}...")
    print(f"   A: {r['answer'][:100]}...")
    print(f"   Category: {r['category']}")
    print(f"   Distance: {r['distance']:.4f}")

# تست Hybrid (RRF)
print("\n========== HYBRID RESULTS (RRF) ==========")
start_time = time.time()
hybrid_results = hybrid.search(query, k=5)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(hybrid_results, 1):
    print(f"\n{i}. Q: {r['question'][:100]}...")
    print(f"   A: {r['answer'][:100]}...")
    print(f"   Category: {r['category']}")
    print(f"   Final Score: {r['rerank_score']:.4f}")   
# تست Hybrid با روش وزنی (اختیاری)
print("\n========== HYBRID RESULTS (Weighted) ==========")
start_time = time.time()
hybrid_weighted = hybrid.search_with_weights(query, k=5, alpha=0.3)
print(f"Search time: {time.time() - start_time:.4f} seconds")

for i, r in enumerate(hybrid_weighted, 1):
    print(f"\n{i}. Q: {r['question'][:100]}...")
    print(f"   A: {r['answer'][:100]}...")
    print(f"   Category: {r['category']}")
    print(f"   Final Score: {r['final_score']:.4f}")
    print(f"   BM25: {r['bm25_score']:.4f}, Semantic: {r['semantic_score']:.4f}")

print("\n" + "=" * 50)
print("PROCESS COMPLETED")
print("=" * 50) 