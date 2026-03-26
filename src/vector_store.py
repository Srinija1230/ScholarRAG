import os, pickle
import numpy as np
import faiss
from src.embedder import embed_texts

# FAISS-based vector store with build, search, save and load
class VectorStore:
    def __init__(self):
        self.faiss_index   = None
        self.stored_chunks = None

    def build(self, chunks: list[dict]):
        print(f"  Embedding {len(chunks)} chunks…")
        chunk_texts  = [chunk["text"] for chunk in chunks]
        chunk_vectors = np.array(embed_texts(chunk_texts), dtype="float32")
        faiss.normalize_L2(chunk_vectors)

        vector_dimension  = chunk_vectors.shape[1]
        self.faiss_index  = faiss.IndexFlatIP(vector_dimension)
        self.faiss_index.add(chunk_vectors)
        self.stored_chunks = chunks
        print(f"  → {self.faiss_index.ntotal} vectors indexed")

    def search(self, query_text: str, num_results: int) -> list[dict]:
        query_vector = np.array(embed_texts([query_text], task="retrieval_query"), dtype="float32")
        faiss.normalize_L2(query_vector)

        similarity_scores, result_indices = self.faiss_index.search(query_vector, num_results)
        matched_chunks = []
        for score, index in zip(similarity_scores[0], result_indices[0]):
            if index < 0:
                continue
            chunk = dict(self.stored_chunks[index])
            chunk["retrieval_score"] = float(score)
            matched_chunks.append(chunk)
        return matched_chunks

    def search_diverse(self, query_text: str, top_per_paper: int = 2) -> list[dict]:
        """Retrieve the top `top_per_paper` most relevant chunks from EVERY paper,
        then return them sorted by score. This guarantees all papers are represented."""
        query_vector = np.array(embed_texts([query_text], task="retrieval_query"), dtype="float32")
        faiss.normalize_L2(query_vector)

        # Fetch enough results to cover all chunks (use total index size)
        total = self.faiss_index.ntotal
        similarity_scores, result_indices = self.faiss_index.search(query_vector, total)

        # Group by paper, keep top_per_paper per paper by score
        from collections import defaultdict
        paper_best = defaultdict(list)
        for score, index in zip(similarity_scores[0], result_indices[0]):
            if index < 0:
                continue
            chunk = dict(self.stored_chunks[index])
            chunk["retrieval_score"] = float(score)
            title = chunk["paper_title"]
            if len(paper_best[title]) < top_per_paper:
                paper_best[title].append(chunk)

        # Flatten and sort by score so highest relevance comes first
        all_chunks = [c for chunks in paper_best.values() for c in chunks]
        all_chunks.sort(key=lambda c: c["retrieval_score"], reverse=True)
        return all_chunks

    def save(self, cache_key: str):
        os.makedirs("data/cache", exist_ok=True)
        faiss.write_index(self.faiss_index, f"data/cache/{cache_key}.faiss")
        with open(f"data/cache/{cache_key}.pkl", "wb") as f:
            pickle.dump(self.stored_chunks, f)

    def load(self, cache_key: str) -> bool:
        index_path = f"data/cache/{cache_key}.faiss"
        if not os.path.exists(index_path):
            return False
        self.faiss_index   = faiss.read_index(index_path)
        with open(f"data/cache/{cache_key}.pkl", "rb") as f:
            self.stored_chunks = pickle.load(f)
        print(f"  → Loaded from cache ({self.faiss_index.ntotal} vectors)")
        return True
