"""
Embedding service for TEIA RAG system.
Uses multilingual sentence-transformers for Spanish language support.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from config import settings
from utils.logger import get_logger

logger = get_logger("embedding_service")

# Model options (all support Spanish):
# - paraphrase-multilingual-MiniLM-L12-v2: 470MB, 384 dims, fast
# - distiluse-base-multilingual-cased-v2: 540MB, 512 dims, good quality
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    """Singleton service for text embeddings."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Same as embed_text but named differently for clarity.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        return self.embed_text(query)


# Global instance for easy access
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
