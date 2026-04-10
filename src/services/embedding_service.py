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
# - paraphrase-multilingual-MiniLM-L12-v2: 470MB, 384 dims, fast but lower retrieval quality
# - paraphrase-multilingual-mpnet-base-v2: 1GB, 768 dims, good quality
# - intfloat/multilingual-e5-base: 1.1GB, 768 dims, best retrieval quality for Spanish (current)
# - intfloat/multilingual-e5-large: 2.2GB, 1024 dims, highest quality, heavier
#
# NOTE: intfloat/multilingual-e5-* models require task-specific prefixes:
#   - "query: "   before user questions (retrieval side)
#   - "passage: " before document chunks (indexing side)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


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
        Generate embedding for a single passage/document chunk.
        Applies the 'passage: ' prefix required by multilingual-e5 models.

        Args:
            text: Passage text to embed

        Returns:
            List of floats representing the embedding
        """
        embedding = self._model.encode(f"passage: {text}", convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple passage/document chunks (batch processing).
        Applies the 'passage: ' prefix required by multilingual-e5 models.

        Args:
            texts: List of passage texts to embed

        Returns:
            List of embeddings
        """
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(prefixed, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Applies the 'query: ' prefix required by multilingual-e5 models.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        embedding = self._model.encode(f"query: {query}", convert_to_numpy=True)
        return embedding.tolist()


# Global instance for easy access
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
