from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever
from evaluation.metrics import (
    precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k,
    map_at_k, f1_at_k
)
import logging

logger = logging.getLogger(__name__)

class EvaluationRunner:
    def __init__(self, dataset):
        self.dataset = dataset
        self.bm25 = BM25Retriever(dataset)
        self.semantic = SemanticRetriever(dataset)
        self.hybrid = HybridRetriever(self.bm25, self.semantic)

    def evaluate_model(self, model, ground_truth, k=10):
        all_relevant = []
        all_retrieved = []
        for item in ground_truth:
            query = item["query"]
            relevant = item["relevant"]
            results = model.search(query, k=k)
            retrieved = [r["index"] for r in results]
            all_relevant.append(relevant)
            all_retrieved.append(retrieved)

        # محاسبه همه معیارها
        prec = sum(precision_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)) / len(ground_truth)
        rec = sum(recall_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)) / len(ground_truth)
        mrr = mean_reciprocal_rank(all_relevant, all_retrieved, k)
        ndcg = sum(ndcg_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)) / len(ground_truth)
        map_score = map_at_k(all_relevant, all_retrieved, k)
        f1 = sum(f1_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)) / len(ground_truth)

        return {
            "precision@k": prec,
            "recall@k": rec,
            "mrr@k": mrr,
            "ndcg@k": ndcg,
            "map@k": map_score,
            "f1@k": f1
        }