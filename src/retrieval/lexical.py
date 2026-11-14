"""BM25 Lexical Retriever for Day 7."""
import math
import re
import unicodedata
from collections import Counter
from typing import Any

def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text with lowercase and punctuation removal."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized)

class BM25Retriever:
    """BM25Okapi implementation for lexical retrieval."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize_vi(c["text"]) for c in chunks]
        self.N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(self.N, 1)

        # Document frequencies
        df = Counter()
        for doc in self.corpus:
            df.update(set(doc))

        # Robertson-Sparck Jones IDF
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_tokens = tokenize_vi(query)
        if not query_tokens:
            return []

        scored = []
        for idx, (chunk, doc_tokens) in enumerate(zip(self.chunks, self.corpus)):
            # Check filters if provided
            if filters:
                match_filter = True
                for f_key, f_val in filters.items():
                    if chunk.get("metadata", {}).get(f_key) != f_val:
                        match_filter = False
                        break
                if not match_filter:
                    continue

            # Compute BM25 score
            doc_len = len(doc_tokens)
            doc_counter = Counter(doc_tokens)
            score = 0.0
            for term in query_tokens:
                if term in self.idf:
                    tf = doc_counter[term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += self.idf[term] * (numerator / denominator)

            if score > 0:
                scored.append({**chunk, "score": float(score), "retrieval_method": "bm25"})

        # Sort by score desc, chunk id asc
        scored.sort(key=lambda x: (-x["score"], x["id"]))
        return scored[:top_k]
