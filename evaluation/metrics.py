import numpy as np

def precision_at_k(relevant, retrieved, k):
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def recall_at_k(relevant, retrieved, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)

def mean_reciprocal_rank(all_relevant, all_retrieved, k=10):
    rr_sum = 0.0
    for relevant, retrieved in zip(all_relevant, all_retrieved):
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(all_retrieved)

def ndcg_at_k(relevant, retrieved, k):
    def dcg(scores):
        return sum((1 / np.log2(i + 2)) if rel else 0 for i, rel in enumerate(scores))
    
    rels = [1 if doc in relevant else 0 for doc in retrieved[:k]]
    ideal_rels = sorted(rels, reverse=True)
    dcg_val = dcg(rels)
    idcg_val = dcg(ideal_rels)
    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val

def average_precision(relevant, retrieved):
    """
    محاسبه Average Precision برای یک کوئری
    
    Args:
        relevant: لیست ایندکس‌های مرتبط
        retrieved: لیست ایندکس‌های بازیابی‌شده
    
    Returns:
        float: امتیاز AP
    """
    if not relevant or not retrieved:
        return 0.0
    
    hits = 0
    sum_prec = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    
    return sum_prec / len(relevant)

def map_at_k(all_relevant, all_retrieved, k=10):
    """
    محاسبه Mean Average Precision (MAP) در k
    
    Args:
        all_relevant: لیست لیست‌های ایندکس‌های مرتبط
        all_retrieved: لیست لیست‌های ایندکس‌های بازیابی‌شده
        k: تعداد نتایج
    
    Returns:
        float: امتیاز MAP@k
    """
    if not all_relevant or not all_retrieved:
        return 0.0
    
    ap_scores = []
    for relevant, retrieved in zip(all_relevant, all_retrieved):
        ap = average_precision(relevant, retrieved[:k])
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def f1_score(precision, recall):
    """
    محاسبه F1-Score از Precision و Recall
    """
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def f1_at_k(relevant, retrieved, k):
    """
    محاسبه F1-Score در k
    """
    prec = precision_at_k(relevant, retrieved, k)
    rec = recall_at_k(relevant, retrieved, k)
    return f1_score(prec, rec)