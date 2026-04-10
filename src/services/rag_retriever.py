"""
RAG Retriever service for TEIA Tutor.

Uses hybrid search: BM25 (keyword) + dense vector similarity, fused via
Reciprocal Rank Fusion (RRF).

Design:
- Both searches run on the FULL corpus (no module filter). At ~857 chunks the
  corpus is small enough that filtering by module only hurts recall without
  meaningful precision gains, especially given unreliable module classification.
- Dense vector search handles semantic similarity.
- BM25 handles keyword-specific lookups where the query phrasing differs from
  the chunk text (semantic gap — the main failure mode of pure vector search).
- Results are fused with RRF(k=60), which is standard in hybrid retrieval.
"""
import re
from typing import List, Dict, Any, Optional

from config import settings
from services.chroma_client import get_chroma_client
from services.embedding_service import get_embedding_service
from utils.logger import get_logger

logger = get_logger("rag_retriever")

COLLECTION_NAME = "teia_course_content"
DEFAULT_TOP_K = 5
RRF_K = 60  # Standard constant for Reciprocal Rank Fusion


class RAGRetriever:
    """Service for retrieving relevant context from indexed documents."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_service = None
        self._initialized = False

        # BM25 index (built lazily at first retrieve call)
        self._bm25_index = None
        self._bm25_docs: List[str] = []    # All chunk texts, indexed by position
        self._bm25_metas: List[dict] = []  # Corresponding metadata dicts

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB, embedding service, and BM25 index."""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG retriever...")

            self._embedding_service = get_embedding_service()
            self._client = get_chroma_client()
            self._collection = self._client.get_collection(name=COLLECTION_NAME)

            chunk_count = self._collection.count()
            logger.info(f"RAG retriever initialized. Collection has {chunk_count} chunks.")

            self._build_bm25_index()
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
            self._initialized = False
            raise

    def _build_bm25_index(self):
        """Build in-memory BM25 index from all chunks stored in ChromaDB."""
        try:
            from rank_bm25 import BM25Okapi

            # Fetch every document from ChromaDB (no filter — we want the full corpus)
            all_data = self._collection.get(include=["documents", "metadatas"])
            self._bm25_docs = all_data["documents"]
            self._bm25_metas = all_data["metadatas"]

            tokenized_corpus = [self._tokenize(doc) for doc in self._bm25_docs]
            self._bm25_index = BM25Okapi(tokenized_corpus)

            logger.info(f"BM25 index built from {len(self._bm25_docs)} chunks.")

        except ImportError:
            logger.warning(
                "rank_bm25 not installed — falling back to pure vector search. "
                "Run: pip install rank_bm25"
            )
        except Exception as e:
            logger.warning(f"Could not build BM25 index: {e}")

    @staticmethod
    def _tokenize(text: str) -> list:
        """Lowercase + remove punctuation + split. Used for BM25."""
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Check if the retriever is available (index exists and non-empty)."""
        try:
            self._ensure_initialized()
            return self._collection.count() > 0
        except Exception:
            return False

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
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
        Retrieve relevant chunks using hybrid search (BM25 + dense vector).

        Strategy
        --------
        1. Dense vector search: full corpus, no module filter.
           Uses a larger candidate pool (top_k * 4) to feed the RRF merger.
        2. BM25 keyword search: full corpus — guarantees keyword-specific chunks
           rank high even when semantic similarity alone would miss them.
        3. RRF(k=60) merges both ranked lists into a single final ranking.

        The `module` parameter is accepted for API compatibility but is not used
        for filtering. At ~857 chunks, searching the full corpus is fast and
        produces better results than per-module filtering with unreliable classification.

        Args:
            query:   The user's question.
            module:  Accepted but unused (kept for API compatibility).
            top_k:   Number of chunks to return.

        Returns:
            List of dicts with keys: content, source, module, chunk_index, score.
        """
        self._ensure_initialized()

        # ------------------------------------------------------------------ #
        # 1. Dense vector search (full corpus — no module filter)
        # ------------------------------------------------------------------ #
        query_embedding = self._embedding_service.embed_query(query)

        candidate_k = min(top_k * 4, 60)  # wider pool so RRF has enough material

        dense_raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )

        dense_ranked: List[dict] = []
        if dense_raw["documents"] and dense_raw["documents"][0]:
            for i, doc in enumerate(dense_raw["documents"][0]):
                meta = dense_raw["metadatas"][0][i] if dense_raw["metadatas"] else {}
                dist = dense_raw["distances"][0][i] if dense_raw["distances"] else 0.0
                dense_ranked.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "module": meta.get("module", "general"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(max(0.0, 1.0 - dist), 4),
                    "dense_score": round(max(0.0, 1.0 - dist), 4),
                })

        # If no BM25 index is available, return dense-only results
        if self._bm25_index is None:
            return dense_ranked[:top_k]

        # ------------------------------------------------------------------ #
        # 2. BM25 keyword search (full corpus — no module filter)
        # ------------------------------------------------------------------ #
        import numpy as np

        tokenized_query = self._tokenize(query)
        bm25_raw_scores = self._bm25_index.get_scores(tokenized_query)

        # Statistical threshold: only chunks with score > mean + 1*std qualify.
        # This filters out low-quality keyword matches (e.g. chunks that merely
        # share generic academic vocabulary) which otherwise introduce noise via RRF.
        scores_arr = bm25_raw_scores[bm25_raw_scores > 0] if hasattr(bm25_raw_scores, '__len__') else []
        if len(scores_arr) > 1:
            bm25_threshold = float(np.mean(scores_arr) + np.std(scores_arr))
        else:
            bm25_threshold = 0.0

        bm25_ranked: List[dict] = sorted(
            [
                {
                    "content": self._bm25_docs[i],
                    "source": self._bm25_metas[i].get("source", "unknown"),
                    "module": self._bm25_metas[i].get("module", "general"),
                    "chunk_index": self._bm25_metas[i].get("chunk_index", 0),
                    "score": float(bm25_raw_scores[i]),
                }
                for i in range(len(self._bm25_docs))
                if bm25_raw_scores[i] > bm25_threshold
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:candidate_k]

        # ------------------------------------------------------------------ #
        # 3. Weighted Reciprocal Rank Fusion
        # ------------------------------------------------------------------ #
        # Dense vector gets 70% weight, BM25 gets 30%.
        # Dense search handles semantic understanding (dominant signal).
        # BM25 provides a targeted boost for keyword-specific failures (Q001-type).
        DENSE_WEIGHT = 0.7
        BM25_WEIGHT = 0.3

        rrf_scores: Dict[str, float] = {}
        chunk_store: Dict[str, dict] = {}

        def _key(chunk: dict) -> str:
            """Stable deduplication key: source file + position in document."""
            return f"{chunk['source']}::{chunk['chunk_index']}"

        for rank, chunk in enumerate(dense_ranked):
            k = _key(chunk)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + DENSE_WEIGHT / (RRF_K + rank + 1)
            chunk_store[k] = chunk

        for rank, chunk in enumerate(bm25_ranked):
            k = _key(chunk)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + BM25_WEIGHT / (RRF_K + rank + 1)
            if k not in chunk_store:
                chunk_store[k] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        final_chunks = []
        for k in sorted_keys[:top_k]:
            chunk = chunk_store[k].copy()
            chunk["score"] = round(rrf_scores[k], 4)
            final_chunks.append(chunk)

        logger.debug(
            f"Hybrid retrieval: {len(dense_ranked)} dense + {len(bm25_ranked)} BM25 "
            f"(threshold={bm25_threshold:.2f}) → {len(final_chunks)} final "
            f"(query: '{query[:50]}')"
        )

        return final_chunks

    def get_context_for_query(
        self,
        query: str,
        module: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query:  The user's question.
            module: Optional module filter.
            top_k:  Number of chunks to retrieve.

        Returns:
            Formatted context string with sources.
        """
        chunks = self.retrieve(query, module, top_k)

        if not chunks:
            return self._get_fallback_context(module)

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
        """Return fallback context when no relevant chunks are found."""
        base_context = """
Universidad El Bosque - Curso: "En sus marcas, listos, ¡RAC!"
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


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

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
    Call this after re-indexing to force reconnect and rebuild the BM25 index.
    """
    global _retriever
    if _retriever is not None:
        logger.info("Resetting RAG retriever singleton")
        _retriever._initialized = False
        _retriever._collection = None
        _retriever._client = None
        _retriever._bm25_index = None
        _retriever._bm25_docs = []
        _retriever._bm25_metas = []
    _retriever = None
