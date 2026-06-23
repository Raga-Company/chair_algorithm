import numpy as np
from retrieval.reranker import ContextAwareReranker


class HybridRetriever:

    def __init__(self, bm25, semantic):

        self.bm25 = bm25
        self.semantic = semantic
        self.reranker = ContextAwareReranker()

    # =========================
    # RRF + Reranker
    # =========================
    def search(self, query, k=5, initial_k=20):

        bm25_results = self.bm25.search(query, k=initial_k)
        semantic_results = self.semantic.search(query, k=initial_k)

        BM25_WEIGHT = 0.6
        SEM_WEIGHT = 0.4

        scores = {}

        for rank, item in enumerate(bm25_results):
            idx = item.get("index")
            if idx is None:
                continue
            scores[idx] = scores.get(idx, 0) + BM25_WEIGHT * (1.0 / (rank + 60))

        for rank, item in enumerate(semantic_results):
            idx = item.get("index")
            if idx is None:
                continue
            scores[idx] = scores.get(idx, 0) + SEM_WEIGHT * (1.0 / (rank + 60))

        for idx in list(scores.keys()):

            in_bm25 = any(r.get("index") == idx for r in bm25_results)
            in_sem = any(r.get("index") == idx for r in semantic_results)

            if in_bm25 and in_sem:
                scores[idx] *= 1.25

        sorted_indices = sorted(
            scores.keys(),
            key=lambda x: scores[x],
            reverse=True
        )[:initial_k]

        candidates = []

        for idx in sorted_indices:

            item = self.bm25.dataset[idx]

            candidates.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "fusion_score": scores[idx]
            })

        reranked_results = []

        for item in candidates:

            doc_text = item["question"] + " " + item["answer"] + " " + item["category"]

            score = self.reranker.score(query, doc_text)

            item["rerank_score"] = float(score)

            reranked_results.append(item)

        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results[:k]

    # =========================
    # Weighted Fusion
    # =========================
    def search_with_weights(self, query, k=5, alpha=0.5):

        bm25_results = self.bm25.search(query, k=20)
        semantic_results = self.semantic.search(query, k=20)

        bm25_scores = np.array([r.get("score", 0) for r in bm25_results])
        semantic_scores = np.array([r.get("score", 0) for r in semantic_results])

        if len(bm25_scores) == 0:
            bm25_scores = np.array([0.0])

        if len(semantic_scores) == 0:
            semantic_scores = np.array([0.0])

        if bm25_scores.max() != bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_norm = np.full_like(bm25_scores, 0.5)

        if semantic_scores.max() != semantic_scores.min():
            semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        else:
            semantic_norm = np.full_like(semantic_scores, 0.5)

        combined = {}

        for item, score in zip(bm25_results, bm25_norm):

            idx = item.get("index")
            if idx is None:
                continue

            combined[idx] = {
                "item": item,
                "bm25_score": float(score),
                "semantic_score": 0.0
            }

        for item, score in zip(semantic_results, semantic_norm):

            idx = item.get("index")
            if idx is None:
                continue

            if idx in combined:
                combined[idx]["semantic_score"] = float(score)
            else:
                combined[idx] = {
                    "item": item,
                    "bm25_score": 0.0,
                    "semantic_score": float(score)
                }

        final_results = []

        for idx, data in combined.items():

            final_score = alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]

            item = data["item"]

            final_results.append({
                "index": idx,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),
                "final_score": float(final_score),
                "bm25_score": float(data["bm25_score"]),
                "semantic_score": float(data["semantic_score"])
            })

        final_results.sort(key=lambda x: x["final_score"], reverse=True)

        return final_results[:k]