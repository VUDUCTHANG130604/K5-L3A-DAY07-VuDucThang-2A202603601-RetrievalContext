"""Dense Retriever using precomputed Google EmbeddingGemma-2 (768d) embeddings."""
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))
from src.embeddings import MyEmbedder

class DenseRetriever:
    """Retrieves chunks by cosine similarity with Google EmbeddingGemma-2."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None,
                 embeddings_path: Path | None = None,
                 model_name: str = "google/embeddinggemma-2"):
        if chunks is None:
            chunks_path = current_dir / "data" / "chunks.json"
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        else:
            self.chunks = chunks

        if embeddings_path is None:
            embeddings_path = current_dir / "data" / "chunk_embeddings.npy"

        self.embeddings = np.load(embeddings_path)
        if len(self.embeddings) != len(self.chunks):
            raise ValueError(f"Mismatch: {len(self.embeddings)} embeddings for {len(self.chunks)} chunks")

        self.embedder = MyEmbedder(model_name=model_name)

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_vec)

        candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if filters:
                match_filter = True
                for f_key, f_val in filters.items():
                    if chunk.get("metadata", {}).get(f_key) != f_val:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            candidates.append({
                **chunk,
                "score": float(score),
                "retrieval_method": "dense"
            })

        candidates.sort(key=lambda x: (-x["score"], x["id"]))
        return candidates[:top_k]
