"""
Indexing service for TEIA RAG system.

Provides async document indexing with task tracking.
"""
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from openpyxl import load_workbook

from config import settings
from services.embedding_service import get_embedding_service
from services.rag_retriever import reset_rag_retriever
from utils.logger import get_logger

logger = get_logger("indexing_service")

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Module mapping based on filename patterns
MODULE_PATTERNS = {
    "caracterizacion": ["caracterizacion", "caracterización", "asignatura"],
    "factores_situacionales": ["factores", "situacional", "contexto"],
    "resultados_aprendizaje": ["resultado", "aprendizaje", "learning", "outcome"],
    "actividades_aprendizaje": ["actividad", "activity", "tarea"],
    "evaluacion": ["evaluacion", "evaluación", "assessment", "rubrica", "rúbrica"],
    "secuencia": ["secuencia", "sequence", "cronograma", "schedule"],
}


class IndexingTask:
    """Represents an indexing task with its state."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "pending"
        self.message = "Task created"
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.documents_processed = 0
        self.chunks_created = 0
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "error": self.error,
        }


class IndexingService:
    """Service for managing document indexing operations."""

    _instance = None
    _tasks: Dict[str, IndexingTask] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _detect_module(filename: str, content: str = "") -> str:
        """Detect module based on filename or content patterns."""
        filename_lower = filename.lower()
        content_lower = content.lower()[:500]

        for module, patterns in MODULE_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower or pattern in content_lower:
                    return module
        return "general"

    @staticmethod
    def _load_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    @staticmethod
    def _load_excel(file_path: Path) -> str:
        """Extract text from Excel file with headers preserved per row."""
        workbook = load_workbook(file_path, data_only=True)
        text_parts = []

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            text_parts.append(f"=== Hoja: {sheet_name} ===\n")

            headers = []
            for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
                # First row with content becomes headers
                if row_idx == 0 or not headers:
                    headers = [str(cell).strip() if cell else f"Columna_{i}"
                               for i, cell in enumerate(row)]
                    # Skip if row is empty
                    if all(h.startswith("Columna_") or not h for h in headers):
                        continue
                    text_parts.append(f"Encabezados: {' | '.join(headers)}")
                    continue

                # Build row with header:value pairs
                row_parts = []
                for i, cell in enumerate(row):
                    if cell is not None and str(cell).strip():
                        header = headers[i] if i < len(headers) else f"Columna_{i}"
                        row_parts.append(f"{header}: {str(cell).strip()}")

                if row_parts:
                    text_parts.append(" | ".join(row_parts))

        return "\n".join(text_parts)

    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """Load markdown file."""
        return file_path.read_text(encoding="utf-8")

    @staticmethod
    def _load_text(file_path: Path) -> str:
        """Load plain text file."""
        return file_path.read_text(encoding="utf-8")

    def _load_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a document and return its content with metadata."""
        suffix = file_path.suffix.lower()

        loaders = {
            ".pdf": self._load_pdf,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".md": self._load_markdown,
            ".txt": self._load_text,
        }

        loader = loaders.get(suffix)
        if not loader:
            logger.debug(f"Skipping unsupported file type: {file_path.name}")
            return None

        try:
            content = loader(file_path)
            module = self._detect_module(file_path.name, content)

            return {
                "content": content,
                "source": file_path.name,
                "file_type": suffix[1:],
                "module": module,
                "path": str(file_path.relative_to(settings.DATA_RAW_PATH)),
            }
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None

    @staticmethod
    def _chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_text(doc["content"])

        return [
            {
                "content": chunk,
                "source": doc["source"],
                "file_type": doc["file_type"],
                "module": doc["module"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i, chunk in enumerate(chunks)
        ]

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents available for indexing."""
        documents = []
        raw_path = settings.DATA_RAW_PATH
        supported_extensions = set(settings.SUPPORTED_EXTENSIONS)

        if not raw_path.exists():
            return documents

        for file_path in raw_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                if file_path.suffix.lower() in supported_extensions:
                    stat = file_path.stat()
                    documents.append({
                        "filename": file_path.name,
                        "path": str(file_path.relative_to(raw_path)),
                        "size_bytes": stat.st_size,
                        "extension": file_path.suffix.lower(),
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                    })

        return documents

    def _save_processed_chunks(self, chunks: List[Dict[str, Any]]):
        """Save processed chunks to JSON for debugging/review."""
        processed_path = settings.DATA_PROCESSED_PATH
        processed_path.mkdir(parents=True, exist_ok=True)
        output_file = processed_path / "chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")

    async def _run_indexing(self, task: IndexingTask):
        """Execute the indexing process."""
        try:
            task.status = "running"
            task.message = "Loading embedding model..."
            task.started_at = datetime.utcnow().isoformat()

            # Run CPU-intensive work in executor
            loop = asyncio.get_event_loop()

            # Get embedding service
            embedding_service = await loop.run_in_executor(
                None, get_embedding_service
            )
            logger.info(f"Embedding dimension: {embedding_service.dimension}")

            task.message = "Initializing ChromaDB..."

            # Initialize ChromaDB
            chroma_path = settings.CHROMA_DB_PATH
            chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(chroma_path))

            # Delete existing collection if it exists
            try:
                client.delete_collection("teia_course_content")
                logger.info("Deleted existing collection")
            except Exception:
                pass

            collection = client.create_collection(
                name="teia_course_content",
                metadata={
                    "description": "TEIA course content for RAG retrieval",
                    "hnsw:space": "cosine"  # Use cosine similarity for better semantic matching
                },
            )

            task.message = "Scanning for documents..."

            # Get documents
            document_paths = []
            raw_path = settings.DATA_RAW_PATH
            supported_extensions = set(settings.SUPPORTED_EXTENSIONS)

            if raw_path.exists():
                for file_path in raw_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith("."):
                        if file_path.suffix.lower() in supported_extensions:
                            document_paths.append(file_path)

            if not document_paths:
                task.status = "completed"
                task.message = "No documents found to index"
                task.completed_at = datetime.utcnow().isoformat()
                return

            task.message = f"Processing {len(document_paths)} documents..."

            all_chunks = []

            for doc_path in document_paths:
                logger.info(f"Processing: {doc_path.name}")
                task.documents_processed += 1

                doc = await loop.run_in_executor(
                    None, self._load_document, doc_path
                )
                if doc is None:
                    continue

                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)

            task.chunks_created = len(all_chunks)

            if not all_chunks:
                task.status = "completed"
                task.message = "No chunks created from documents"
                task.completed_at = datetime.utcnow().isoformat()
                return

            # Save processed chunks
            await loop.run_in_executor(
                None, self._save_processed_chunks, all_chunks
            )

            task.message = "Generating embeddings..."

            # Generate embeddings
            texts = [chunk["content"] for chunk in all_chunks]
            embeddings = await loop.run_in_executor(
                None, embedding_service.embed_texts, texts
            )

            task.message = "Adding to ChromaDB..."

            # Add to ChromaDB
            collection.add(
                ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source": chunk["source"],
                        "module": chunk["module"],
                        "file_type": chunk["file_type"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                    }
                    for chunk in all_chunks
                ],
            )

            task.status = "completed"
            task.message = f"Successfully indexed {len(all_chunks)} chunks from {task.documents_processed} documents"
            task.completed_at = datetime.utcnow().isoformat()

            # Reset RAG retriever to pick up the new collection
            reset_rag_retriever()

            logger.info(f"Indexing complete: {task.message}")

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            task.status = "failed"
            task.error = str(e)
            task.message = f"Indexing failed: {str(e)}"
            task.completed_at = datetime.utcnow().isoformat()

    def start_indexing(self) -> str:
        """Start a new indexing task and return its ID."""
        task_id = str(uuid.uuid4())
        task = IndexingTask(task_id)
        self._tasks[task_id] = task

        # Schedule the indexing task
        asyncio.create_task(self._run_indexing(task))

        logger.info(f"Started indexing task: {task_id}")
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an indexing task."""
        task = self._tasks.get(task_id)
        if task:
            return task.to_dict()
        return None

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all indexing tasks."""
        return [task.to_dict() for task in self._tasks.values()]


# Singleton instance
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """Get the singleton IndexingService instance."""
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService()
    return _indexing_service
