"""
RAG Retriever service for TEIA Tutor.
Handles semantic search over indexed course content.
"""
from typing import List, Dict, Any, Optional

import chromadb

from config import settings
from services.embedding_service import get_embedding_service
from utils.logger import get_logger

logger = get_logger("rag_retriever")

# Collection name for ChromaDB
COLLECTION_NAME = "teia_course_content"

# Default retrieval settings
DEFAULT_TOP_K = 3


class RAGRetriever:
    """Service for retrieving relevant context from indexed documents."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_service = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB and embedding service."""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG retriever...")

            # Initialize embedding service
            self._embedding_service = get_embedding_service()

            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))

            # Get collection
            self._collection = self._client.get_collection(name=COLLECTION_NAME)

            chunk_count = self._collection.count()
            logger.info(f"RAG retriever initialized. Collection has {chunk_count} chunks.")

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            self._initialized = False
            raise

    @property
    def is_available(self) -> bool:
        """Check if the retriever is available (index exists)."""
        try:
            self._ensure_initialized()
            return self._collection.count() > 0
        except Exception:
            return False

    @property
    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        try:
            self._ensure_initialized()
            return self._collection.count()
        except Exception:
            return 0

    def retrieve(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question
            module: Optional module (kept for API compatibility, not used for filtering)
            top_k: Number of chunks to retrieve

        Returns:
            List of dicts with 'content', 'source', 'module', 'score'
        """
        self._ensure_initialized()

        # Generate query embedding
        query_embedding = self._embedding_service.embed_query(query)

        # Query ChromaDB without module filter to search all chunks
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # Convert distance to similarity score (ChromaDB uses cosine distance)
                # Cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
                similarity = max(0, 1 - distance)

                chunks.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "module": metadata.get("module", "general"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": round(similarity, 4),
                })

        logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")

        return chunks

    def get_context_for_query(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: The user's question
            module: Optional module filter
            top_k: Number of chunks to retrieve

        Returns:
            Formatted context string with sources
        """
        chunks = self.retrieve(query, module, top_k)

        if not chunks:
            return self._get_fallback_context(module)

        # Format context with sources
        context_parts = []
        sources = set()

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Fragmento {i}]\n{chunk['content']}")
            sources.add(chunk["source"])

        context = "\n\n".join(context_parts)
        source_list = ", ".join(sources)

        return f"""CONTEXTO RECUPERADO DEL CURSO:
{context}

FUENTES: {source_list}"""

    def _get_fallback_context(self, module: Optional[str] = None) -> str:
        """Return fallback context when no relevant chunks found."""
        base_context = """
Universidad El Bosque - Curso: "En sus marcas, listos, iRAC!"
Enfoque: Diseño curricular centrado en el estudiante
Objetivo: Fortalecer competencias pedagógicas en diseño de microcurrículos
"""

        module_hints = {
            "resultados_aprendizaje": "Este módulo se enfoca en diseño de resultados de aprendizaje claros, medibles y alineados con competencias.",
            "evaluacion": "Este módulo cubre estrategias de evaluación formativa y sumativa, rúbricas y retroalimentación.",
            "caracterizacion": "Este módulo trata sobre identificación de características de la asignatura y contexto institucional.",
            "actividades_aprendizaje": "Este módulo se enfoca en diseño de actividades activas y centradas en el estudiante.",
            "factores_situacionales": "Este módulo analiza el contexto que influye en el diseño curricular.",
            "secuencia": "Este módulo cubre la organización temporal de contenidos y actividades.",
        }

        hint = module_hints.get(module, "")

        return f"{base_context}\n{hint}".strip()


# Global instance
_retriever = None


def get_rag_retriever() -> RAGRetriever:
    """Get or create the RAG retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def reset_rag_retriever():
    """
    Reset the RAG retriever singleton.

    Call this after re-indexing to force the retriever to
    reconnect to the new ChromaDB collection.
    """
    global _retriever
    if _retriever is not None:
        logger.info("Resetting RAG retriever singleton")
        _retriever._initialized = False
        _retriever._collection = None
        _retriever._client = None
    _retriever = None
