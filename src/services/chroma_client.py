"""
Shared ChromaDB client singleton for TEIA services.

Using a single PersistentClient instance across the application avoids
UUID conflicts that occur when multiple clients access the same SQLite
database (e.g., during re-indexing while the RAG retriever is active).
"""
from typing import Optional

import chromadb

from config import settings
from utils.logger import get_logger

logger = get_logger("chroma_client")

_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the application-wide ChromaDB PersistentClient singleton."""
    global _client
    if _client is None:
        db_path = settings.CHROMA_DB_PATH
        db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing shared ChromaDB PersistentClient at {db_path}")
        _client = chromadb.PersistentClient(path=str(db_path))
    return _client
