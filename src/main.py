"""
TEIA Tutor AI Service - Main Application
FastAPI application for the Intelligent Tutoring System.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import settings
from middleware import setup_middlewares
from models import (
    QuestionRequest,
    QuestionResponse,
    ModuleInfo,
    HealthResponse,
    SystemStatus,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    UploadedFileInfo,
    FailedFileInfo,
    IndexingTaskResponse,
    IndexingStartResponse,
)
from services.ollama_service import generate_response, check_ollama_health
from services.rag_retriever import get_rag_retriever
from services.indexing_service import get_indexing_service
from utils.logger import get_logger

logger = get_logger("main")

# Initialize FastAPI app
app = FastAPI(
    title="TEIA Tutor AI Service",
    description="Servicio de Tutoría Inteligente para el curso 'En sus marcas, listos, iRAC!'",
    version="1.0.0"
)

# Setup middlewares (CORS, etc.)
setup_middlewares(app)

# Serve static files (UI)
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Course modules definition
COURSE_MODULES = [
    ModuleInfo(
        id="caracterizacion",
        name="Caracterización de la Asignatura",
        description="Identificación de características y contexto de la asignatura"
    ),
    ModuleInfo(
        id="factores_situacionales",
        name="Factores Situacionales",
        description="Análisis del contexto que influye en el diseño curricular"
    ),
    ModuleInfo(
        id="resultados_aprendizaje",
        name="Resultados de Aprendizaje",
        description="Diseño de resultados de aprendizaje claros y medibles"
    ),
    ModuleInfo(
        id="actividades_aprendizaje",
        name="Actividades de Aprendizaje",
        description="Diseño de actividades centradas en el estudiante"
    ),
    ModuleInfo(
        id="evaluacion",
        name="Evaluación",
        description="Estrategias de evaluación formativa y sumativa"
    ),
    ModuleInfo(
        id="secuencia",
        name="Secuencia del Curso",
        description="Organización temporal de contenidos y actividades"
    )
]


@app.get("/")
async def root():
    """Root endpoint - service information."""
    return {
        "service": "TEIA Tutor AI Service",
        "version": "1.0.0",
        "status": "active",
        "ui": "/ui"
    }


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    ui_path = Path(__file__).parent.parent / "static" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint - verifies service and Ollama connectivity."""
    ollama_connected, _ = await check_ollama_health()

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        service="teia-tutor",
        ollama_connected=ollama_connected
    )


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Detailed system status including available models."""
    ollama_connected, models = await check_ollama_health()

    return SystemStatus(
        service="teia-tutor",
        status="running",
        ollama_connected=ollama_connected,
        ollama_models=models,
        timestamp=datetime.now().isoformat()
    )


@app.get("/modules", response_model=List[ModuleInfo])
async def get_modules():
    """Returns available course modules."""
    return COURSE_MODULES


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main tutoring endpoint - process questions and return AI-generated answers.

    Uses RAG to retrieve relevant context from indexed course documents,
    then generates a response using the LLM.

    The response includes tracking metadata (request_id, timestamp, user context)
    that can be used for future persistence and reporting.
    """
    start_time = time.time()

    logger.info(
        f"Question received - user_id: {request.user_id}, "
        f"session_id: {request.session_id}, module: {request.module}"
    )

    try:
        # Get RAG retriever
        retriever = get_rag_retriever()

        # Retrieve relevant context using RAG
        retrieved_chunks = retriever.retrieve(
            query=request.question,
            module=request.module,
            top_k=3
        )

        # Get formatted context for LLM
        context = retriever.get_context_for_query(
            query=request.question,
            module=request.module,
            top_k=3
        )

        # Get all indexed documents as sources
        indexing_service = get_indexing_service()
        all_documents = indexing_service.list_documents()
        sources = [doc["filename"] for doc in all_documents]
        if not sources:
            sources = ["No hay documentos indexados"]

        # Calculate confidence based on retrieval scores
        if retrieved_chunks:
            avg_score = sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks)
            confidence = min(avg_score + 0.3, 0.95)  # Boost and cap at 0.95
        else:
            confidence = 0.5  # Lower confidence without retrieved context

        # Generate response from LLM (async)
        answer = await generate_response(
            question=request.question,
            context=context,
            module=request.module or "General"
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return QuestionResponse(
            # Echo back user context for correlation
            user_id=request.user_id,
            session_id=request.session_id,
            module=request.module,
            # Response content
            answer=answer,
            confidence=round(confidence, 2),
            sources=sources,
            suggested_actions=[
                "Revisar los documentos fuente mencionados",
                "Consultar con tutor humano para casos complejos",
                "Validar con lineamientos institucionales"
            ],
            # Metadata
            model_used=settings.OLLAMA_MODEL,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )


# =============================================================================
# Document Management Endpoints
# =============================================================================

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents available for indexing.

    Returns documents from the configured DATA_RAW_PATH with metadata
    like filename, size, and modification date.
    """
    try:
        indexing_service = get_indexing_service()
        documents = indexing_service.list_documents()

        return DocumentListResponse(
            documents=[DocumentInfo(**doc) for doc in documents],
            total_count=len(documents),
            raw_path=str(settings.DATA_RAW_PATH)
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(
        ...,
        description="Archivos a subir (PDF, Excel, Markdown, TXT)"
    )
):
    """
    Upload documents for later indexing.

    Accepts multiple files via multipart/form-data.
    Only files with supported extensions will be accepted (.pdf, .xlsx, .xls, .md, .txt).

    **Ejemplo con curl:**
    ```
    curl -X POST -F "files=@documento.pdf" -F "files=@otro.xlsx" http://localhost:8000/documents/upload
    ```
    """
    uploaded = []
    failed = []
    supported_extensions = set(settings.SUPPORTED_EXTENSIONS)

    # Ensure raw path exists
    raw_path = settings.DATA_RAW_PATH
    raw_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            # Validate extension
            filename = file.filename or "unnamed"
            extension = "." + filename.split(".")[-1].lower() if "." in filename else ""

            if extension not in supported_extensions:
                failed.append(FailedFileInfo(
                    filename=filename,
                    error=f"Unsupported file type. Allowed: {', '.join(supported_extensions)}"
                ))
                continue

            # Save file
            file_path = raw_path / filename
            content = await file.read()

            with open(file_path, "wb") as f:
                f.write(content)

            uploaded.append(UploadedFileInfo(
                filename=filename,
                size_bytes=len(content)
            ))

            logger.info(f"Uploaded document: {filename} ({len(content)} bytes)")

        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            failed.append(FailedFileInfo(
                filename=file.filename or "unnamed",
                error=str(e)
            ))

    return DocumentUploadResponse(
        uploaded_files=uploaded,
        failed_files=failed,
        upload_path=str(raw_path)
    )


# =============================================================================
# Indexing Endpoints
# =============================================================================

@app.post("/index", response_model=IndexingStartResponse)
async def start_indexing():
    """
    Start document indexing process.

    This is an asynchronous operation. The endpoint returns immediately
    with a task_id that can be used to check the indexing status.

    The indexing process:
    1. Scans DATA_RAW_PATH for supported documents
    2. Extracts text and splits into chunks
    3. Generates embeddings
    4. Stores in ChromaDB for RAG retrieval
    """
    try:
        indexing_service = get_indexing_service()
        task_id = indexing_service.start_indexing()

        return IndexingStartResponse(
            task_id=task_id,
            status="pending",
            message="Indexing task started. Use /index/status/{task_id} to check progress."
        )
    except Exception as e:
        logger.error(f"Error starting indexing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting indexing: {str(e)}"
        )


@app.get("/index/status/{task_id}", response_model=IndexingTaskResponse)
async def get_indexing_status(task_id: str):
    """
    Get the status of an indexing task.

    Returns current status, progress, and any errors that occurred.
    """
    indexing_service = get_indexing_service()
    status = indexing_service.get_status(task_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}"
        )

    return IndexingTaskResponse(**status)


@app.get("/index/tasks", response_model=List[IndexingTaskResponse])
async def list_indexing_tasks():
    """
    List all indexing tasks.

    Returns all tasks with their current status.
    """
    indexing_service = get_indexing_service()
    tasks = indexing_service.list_tasks()

    return [IndexingTaskResponse(**task) for task in tasks]


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting TEIA Tutor AI Service...")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
