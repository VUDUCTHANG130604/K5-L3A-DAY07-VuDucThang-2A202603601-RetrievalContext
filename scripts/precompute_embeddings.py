"""Precompute and cache embeddings for 126 chunks using google/embeddinggemma-2."""
import json
import sys
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent.parent
day6_starter = Path(r"d:\AI Thực chiến\Lad 6\K5-L3A-Day06-VuDucThang-03601-DataIngestionEngineering\starter")
sys.path.insert(0, str(day6_starter))
from src.embeddings import MyEmbedder

def precompute():
    chunks_path = current_dir / "data" / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model and encoding {len(texts)} chunks...")
    embedder = MyEmbedder(model_name="google/embeddinggemma-2")
    vectors = embedder.embed_documents(texts)
    
    vec_array = np.array(vectors, dtype=np.float32)
    print("Vectors shape:", vec_array.shape)

    cache_path = current_dir / "data" / "chunk_embeddings.npy"
    np.save(cache_path, vec_array)
    print(f"Saved chunk embeddings to: {cache_path}")

if __name__ == "__main__":
    precompute()
