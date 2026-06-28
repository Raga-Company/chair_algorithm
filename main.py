import json
import os
import time
import argparse
import logging
from pathlib import Path
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.fusion import HybridRetriever
from retrieval.query_expansion import QueryExpander
from utils.logger import get_result_logger
from utils.data_loader import DatasetLoader
from utils.error_handler import DataError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_dataset_files(data_dir="data"):
    """پیدا کردن تمام فایل‌های JSON در پوشه داده"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted([f for f in data_path.glob("*.json") if f.is_file()])


def select_dataset_file(data_dir="data"):
    """انتخاب فایل دیتاست از بین گزینه‌های موجود"""
    files = find_dataset_files(data_dir)
    if not files:
        logger.error(f"No JSON files found in '{data_dir}' directory.")
        return None
    
    if len(files) == 1:
        selected = files[0]
        logger.info(f"Only one dataset found: {selected.name}")
        return str(selected)
    
    print("\n📂 Available datasets:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(files)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return str(files[idx])
            print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def main():
    parser = argparse.ArgumentParser(description="Intelligent Retrieval System")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after loading")
    parser.add_argument("--data-size", type=int, default=None, help="Number of samples to use (None = all)")
    parser.add_argument("--query", type=str, help="Single query to test (if not in interactive mode)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive query input")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="hnsw", 
                        help="FAISS index type (hnsw recommended for large datasets)")
    parser.add_argument("--no-reranker", action="store_true", 
                        help="Disable reranker for faster search (recommended for production)")
    parser.add_argument("--use-reranker", action="store_true", 
                        help="Enable reranker (slower but potentially more accurate)")
    parser.add_argument("--output-dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to dataset file (if not provided, will auto-detect)")
    args = parser.parse_args()

    use_reranker = args.use_reranker and not args.no_reranker

    logger.info("=" * 50)
    logger.info("INTELLIGENT RETRIEVAL SYSTEM STARTED")
    logger.info("=" * 50)

    result_logger = get_result_logger()
    logger.info(f"Results will be saved to: {result_logger.text_file}")

    # ----- انتخاب فایل دیتاست -----
    data_file = args.data_file
    if data_file is None:
        # اگر فایل مشخص نشده، ابتدا فایل‌های موجود را بررسی کن
        data_file = select_dataset_file("data")
        if data_file is None:
            return
    else:
        # اگر فایل مشخص شده ولی وجود ندارد، خطا بده
        if not Path(data_file).exists():
            logger.error(f"Specified data file not found: {data_file}")
            return

    # ----- بارگذاری دیتاست -----
    try:
        data = DatasetLoader.load(data_file)
        if data is None:
            logger.error(f"Failed to load dataset from {data_file}: loader returned None")
            return
        logger.info(f"Loaded {len(data)} samples from {data_file}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    if args.data_size is not None and args.data_size < len(data):
        data = data[:args.data_size]
    logger.info(f"Working data size: {len(data)}")

    # ----- مقداردهی اولیه مدل‌ها -----
    start_time = time.time()
    try:
        logger.info("Initializing BM25...")
        bm25 = BM25Retriever(data, cache_dir="cache/bm25")

        logger.info("Initializing Semantic Retriever...")
        semantic = SemanticRetriever(
            data,
            cache_dir="cache/semantic",
            model_name="HooshvareLab/bert-fa-base-uncased",
            index_type=args.index_type
        )

        logger.info("Initializing Hybrid Retriever...")
        hybrid = HybridRetriever(bm25, semantic)
        
        logger.info("Initializing Query Expander...")
        expander = QueryExpander(use_wordnet=False)

        logger.info(f"Initialization time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return

    # ----- ارزیابی (اختیاری) -----
    if args.evaluate:
        try:
            from evaluation.compare_models import run_full_evaluation
            results = run_full_evaluation(data, sample_size=min(300, len(data)//2))
            logger.info(f"Evaluation results: {results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")

    # ----- دریافت کوئری -----
    if args.query:
        queries = [args.query]
    else:
        print("\n" + "=" * 50)
        print("INTERACTIVE MODE - Enter your query (type 'exit' to quit)")
        print("=" * 50)
        try:
            query = input("\nEnter your query: ").strip()
            if not query:
                logger.warning("Empty query received. Exiting...")
                return
            if query.lower() == 'exit':
                logger.info("User requested exit")
                return
            queries = [query]
        except KeyboardInterrupt:
            logger.info("User interrupted")
            return
        except EOFError:
            logger.error("No input available")
            return

    # ----- پردازش هر کوئری -----
    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("=" * 50)
        
        # توسعه کوئری (اختیاری)
        expanded_queries = expander.expand_with_synonyms(query, max_terms=3)
        logger.info(f"Expanded queries: {expanded_queries}")
        
        results_dict = {}
        
        # BM25
        try:
            logger.info("\n========== BM25 RESULTS ==========")
            start = time.time()
            bm25_res = bm25.search(query, k=5)
            logger.info(f"Search time: {time.time()-start:.4f}s")
            print("\n" + "=" * 60)
            print("BM25 RESULTS")
            print("=" * 60)
            for i, r in enumerate(bm25_res, 1):
                print(f"{i}. Q: {r['question'][:80]}...")
                print(f"   A: {r['answer'][:80]}...")
                print(f"   Cat: {r.get('category','N/A')} Score: {r['score']:.4f}\n")
            results_dict["bm25"] = bm25_res
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            results_dict["bm25"] = []

        # Semantic
        try:
            logger.info("\n========== SEMANTIC RESULTS ==========")
            start = time.time()
            sem_res = semantic.search(query, k=5)
            logger.info(f"Search time: {time.time()-start:.4f}s")
            print("\n" + "=" * 60)
            print("SEMANTIC RESULTS")
            print("=" * 60)
            for i, r in enumerate(sem_res, 1):
                print(f"{i}. Q: {r['question'][:80]}...")
                print(f"   A: {r['answer'][:80]}...")
                print(f"   Cat: {r.get('category','N/A')} Score: {r['score']:.4f}\n")
            results_dict["semantic"] = sem_res
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            results_dict["semantic"] = []

        # Hybrid RRF
        try:
            logger.info(f"\n========== HYBRID (RRF{' with Reranker' if use_reranker else ''}) ==========")
            start = time.time()
            hybrid_res = hybrid.search(query, k=5, use_reranker=use_reranker)
            logger.info(f"Search time: {time.time()-start:.4f}s")
            score_key = "rerank_score" if use_reranker else "fusion_score"
            print("\n" + "=" * 60)
            print(f"HYBRID (RRF{' with Reranker' if use_reranker else ''}) RESULTS")
            print("=" * 60)
            for i, r in enumerate(hybrid_res, 1):
                print(f"{i}. Q: {r['question'][:80]}...")
                print(f"   A: {r['answer'][:80]}...")
                print(f"   Cat: {r.get('category','N/A')} Score: {r.get(score_key,0):.4f}\n")
            results_dict["hybrid_rrf"] = hybrid_res
        except Exception as e:
            logger.error(f"Hybrid RRF search failed: {e}")
            results_dict["hybrid_rrf"] = []

        # Weighted Hybrid
        try:
            logger.info("\n========== HYBRID (Weighted) ==========")
            start = time.time()
            weighted = hybrid.search_with_weights(query, k=5, alpha=0.6)
            logger.info(f"Search time: {time.time()-start:.4f}s")
            print("\n" + "=" * 60)
            print("HYBRID (Weighted) RESULTS")
            print("=" * 60)
            for i, r in enumerate(weighted, 1):
                print(f"{i}. Q: {r['question'][:80]}...")
                print(f"   A: {r['answer'][:80]}...")
                print(f"   Cat: {r.get('category','N/A')}")
                print(f"   Final: {r['final_score']:.4f} BM25: {r['bm25_score']:.4f} Sem: {r['semantic_score']:.4f}\n")
            results_dict["hybrid_weighted"] = weighted
        except Exception as e:
            logger.error(f"Weighted hybrid search failed: {e}")
            results_dict["hybrid_weighted"] = []

        # ذخیره نتایج
        try:
            result_logger.log_query(query, results_dict)
            logger.info(f"Results saved to: {result_logger.text_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        logger.info("\n" + "=" * 50)

    logger.info("PROCESS COMPLETED")
    logger.info(f"All results saved to: {result_logger.text_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()