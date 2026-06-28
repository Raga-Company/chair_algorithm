import os
import pickle
import hashlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from preprocessing.cleaner import clean_text
from utils.error_handler import handle_errors, log_execution_time, ModelError
from utils.memory import get_optimal_batch_size, chunk_list
import logging
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextAwareReranker:
    def __init__(self, dataset, cache_dir="cache/reranker", use_mmap=True):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.use_mmap = use_mmap
        os.makedirs(cache_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # استفاده از مدل فارسی برای Reranker
        model_name = "HooshvareLab/bert-fa-base-uncased"  # ← مدل فارسی
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Reranker model loaded: {model_name} on {self.device}")
        except Exception as e:
            raise ModelError(f"Failed to load reranker model: {e}")
        
        dataset_hash = self._hash_dataset()
        self.doc_emb_file = os.path.join(cache_dir, f"doc_emb_{dataset_hash}.npy")
        self.doc_emb_meta_file = os.path.join(cache_dir, f"doc_emb_meta_{dataset_hash}.pkl")
        self.doc_emb = self._load_or_build_embeddings()
        logger.info(f"✅ Reranker ready with {len(self.doc_emb)} embeddings on {self.device}")
    
    def _hash_dataset(self):
        content = str([(item.get("question",""), item.get("answer","")) for item in self.dataset])
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    @log_execution_time
    def _load_or_build_embeddings(self):
        if os.path.exists(self.doc_emb_file) and os.path.exists(self.doc_emb_meta_file):
            try:
                if self.use_mmap:
                    embeddings = np.load(self.doc_emb_file, mmap_mode='r')
                else:
                    embeddings = np.load(self.doc_emb_file)
                
                with open(self.doc_emb_meta_file, "rb") as f:
                    meta = pickle.load(f)
                
                if meta.get("dataset_size") == len(self.dataset):
                    logger.info(f"Loaded {len(embeddings)} embeddings from cache")
                    return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
        
        logger.info("Building document embeddings for reranker (Persian model)...")
        embeddings = []
        batch_size = get_optimal_batch_size(base_size=32, min_size=8)
        
        for batch in chunk_list(self.dataset, batch_size):
            batch_texts = []
            for item in batch:
                doc = clean_text(
                    str(item.get("question", "")) + " " +
                    str(item.get("answer", "")) + " " +
                    str(item.get("category", "")) + " " +
                    str(item.get("specialty", ""))
                )
                batch_texts.append(doc)
            
            batch_embs = self._encode_batch(batch_texts)
            embeddings.append(batch_embs)
            
            if len(embeddings) % 10 == 0:
                logger.info(f"Reranker embedding: {len(embeddings)*batch_size}/{len(self.dataset)}")
        
        embeddings = np.vstack(embeddings) if embeddings else np.array([])
        
        np.save(self.doc_emb_file, embeddings)
        with open(self.doc_emb_meta_file, "wb") as f:
            pickle.dump({"dataset_size": len(self.dataset)}, f)
        
        logger.info(f"Saved {len(embeddings)} embeddings to cache")
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    @handle_errors
    @log_execution_time
    def rerank(self, query: str, candidates: List[Dict[str, Any]], batch_size: int = 16) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        
        query_emb = self.encode(query)
        reranked = []
        
        for batch in chunk_list(candidates, batch_size):
            batch_indices = [item["index"] for item in batch]
            
            doc_embs = torch.tensor(
                np.vstack([self.doc_emb[idx] for idx in batch_indices]),
                device=self.device
            )
            
            similarities = F.cosine_similarity(query_emb, doc_embs)
            
            for item, sim in zip(batch, similarities):
                item["rerank_score"] = float(sim.item())
                reranked.append(item)
        
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked