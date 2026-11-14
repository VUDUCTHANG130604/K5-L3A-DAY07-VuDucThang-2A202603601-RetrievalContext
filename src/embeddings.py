"""Local sentence-transformers adapter for document and query embeddings."""
from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Embed text with a real local Sentence Transformers model."""

    def __init__(self, model_name: str = "google/embeddinggemma-2") -> None:
        if not model_name or not model_name.strip():
            model_name = "google/embeddinggemma-2"

        self.model_name = model_name.strip()
        kwargs = {"local_files_only": True}
        if "gemma" in self.model_name.lower():
            kwargs["model_kwargs"] = {"dtype": torch.float32}

        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception:
            self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        dimensions = get_dim()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(f"Embedding model returned invalid dimensions: {dimensions!r}")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Validate strictly: do not silently fallback or invent mock/fake data
        self._validate_texts(texts)

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} texts"
            )

        result = [vector.tolist() for vector in vectors]
        self._validate_vectors(result, expected_count=len(texts))
        return result

    def embed_query(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Query question must not be empty")
        return self.embed_documents([question])[0]

    def _validate_texts(self, texts: list[str]) -> None:
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Embedding text at index {idx} must be a non-empty string; silent fallbacks are prohibited."
                )

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> None:
        if len(vectors) != expected_count:
            raise RuntimeError(f"Expected {expected_count} vectors, received {len(vectors)}")
        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise RuntimeError(
                    f"Vector at index {idx} has {len(vec)} dimensions, expected {self.dimensions}"
                )
