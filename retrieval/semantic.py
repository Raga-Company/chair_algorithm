import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from preprocessing.cleaner import clean_text


class SemanticRetriever:

    def __init__(
        self,
        dataset,
        cache_dir="cache",
        model_name=None
    ):

        print("SEMANTIC INIT START")

        self.dataset = dataset
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        self.model_name = (
            model_name
            or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        self.embeddings_file = os.path.join(cache_dir, "embeddings.npy")
        self.index_file = os.path.join(cache_dir, "faiss.index")
        self.documents_file = os.path.join(cache_dir, "documents.pkl")
        self.model_info_file = os.path.join(cache_dir, "model.txt")

        if self._load_cache():
            print("Loaded semantic cache :heavy_check_mark:")
        else:
            self._build()
            self._save_cache()

        self.model = None

        print("SEMANTIC READY :heavy_check_mark:")
# =========================
    # CACHE
    # =========================
    def _load_cache(self):

        files = [
            self.embeddings_file,
            self.index_file,
            self.documents_file
        ]

        if not all(os.path.exists(x) for x in files):
            return False

        try:
            self.documents = pickle.load(open(self.documents_file, "rb"))
            self.embeddings = np.load(self.embeddings_file)
            self.index = faiss.read_index(self.index_file)
            return True
        except:
            return False

    def _save_cache(self):

        np.save(self.embeddings_file, self.embeddings)
        faiss.write_index(self.index, self.index_file)

        pickle.dump(
            self.documents,
            open(self.documents_file, "wb")
        )

        with open(self.model_info_file, "w") as f:
            f.write(self.model_name)

    # =========================
    # BUILD INDEX
    # =========================
    def _build(self):

        print("Building semantic index...")

        model = SentenceTransformer(self.model_name)

        self.documents = []

        for i, item in enumerate(self.dataset):

            if i % 1000 == 0:
                print("Preparing:", i)

            question = str(item.get("question", ""))
            category = str(item.get("category", ""))
            specialty = str(item.get("specialty", ""))

            text = clean_text(question + " " + category + " " + specialty)

            self.documents.append(text)

        print("Encoding...")

        self.embeddings = model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print("FAISS index created :heavy_check_mark:")

    # =========================
    # SEARCH (FIXED - داخل کلاس!)
    # =========================
    def search(self, query, k=5):

        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        query = clean_text(query)

        if len(query.split()) <= 2:
            query += " پزشکی"

        q_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, ids = self.index.search(q_embedding, k)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx < 0 or idx >= len(self.dataset):
                continue

            item = self.dataset[idx]

            results.append({
                "index": int(idx),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", ""),

                "text": self.documents[idx],

                "score": float(score)
            })

        return results
