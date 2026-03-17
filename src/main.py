"""
TEIA Tutor AI Service - Main Application
FastAPI application for the Intelligent Tutoring System.
"""
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

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
    ActivityValidationRequest,
    ActivityValidationResponse,
    EvaluationValidationRequest,
)
from services.ollama_service import generate_response, check_ollama_health, validate_learning_activity, validate_evaluation_design
from services.rag_retriever import get_rag_retriever
from services.indexing_service import get_indexing_service
from utils.logger import get_logger

logger = get_logger("main")

# Initialize FastAPI app
app = FastAPI(
    title="TEIA Tutor AI Service",
    description="Servicio de Tutoría Inteligente para el curso 'En sus marcas, listos, ¡RAC!'",
    version="1.0.0"
)

# Setup middlewares (CORS, etc.)
setup_middlewares(app)

# Serve static files (UI)
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Course modules definition
# Maps module IDs (sent by the client) to their human-readable display names
# used inside the LLM prompt so the model sees the proper name, not the raw ID.
MODULE_DISPLAY_NAMES = {
    "caracterizacion":         "Caracterización de la Asignatura",
    "factores_situacionales":  "Factores Situacionales",
    "resultados_aprendizaje":  "Resultados de Aprendizaje",
    "actividades_aprendizaje": "Actividades de Aprendizaje",
    "evaluacion":              "Evaluación",
    "secuencia":               "Secuencia del Curso",
}

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
        model_used=settings.OLLAMA_MODEL,
        timestamp=datetime.now().isoformat()
    )


@app.get("/modules", response_model=List[ModuleInfo])
async def get_modules():
    """Returns available course modules."""
    return COURSE_MODULES


# Suggested actions per module — shown to the teacher after each response
SUGGESTED_ACTIONS = {
    "caracterizacion": [
        "Revisa los lineamientos institucionales de la Universidad El Bosque para caracterización",
        "Identifica el nivel de formación y las condiciones del grupo al que va dirigida la asignatura",
        "Contrasta la caracterización con las competencias declaradas en el programa de estudios",
    ],
    "factores_situacionales": [
        "Documenta los factores del entorno que condicionan el diseño de tu asignatura",
        "Considera el perfil sociocultural del estudiantado al definir las estrategias",
        "Revisa cómo los factores situacionales impactan la selección de actividades y evaluaciones",
    ],
    "resultados_aprendizaje": [
        "Selecciona un verbo observable de la taxonomía de Fink para redactar tu resultado",
        "Verifica que cada resultado sea medible con los instrumentos de evaluación disponibles",
        "Consulta el material 'Guía diseño Fink.pdf' incluido en el curso como referencia",
    ],
    "actividades_aprendizaje": [
        "Asegúrate de que cada actividad esté alineada con al menos un resultado de aprendizaje",
        "Incorpora principios de aprendizaje activo: reflexión, colaboración o aplicación real",
        "Estima el tiempo y recursos necesarios antes de incluir la actividad en la secuencia",
    ],
    "evaluacion": [
        "Distingue qué momentos serán formativos (retroalimentación) y cuáles sumativos (calificación)",
        "Diseña la rúbrica antes de definir la actividad evaluativa, no al revés",
        "Revisa el documento de la UNESCO sobre evaluación para los aprendizajes incluido en el curso",
    ],
    "secuencia": [
        "Distribuye los momentos de evaluación a lo largo del curso, no solo al final",
        "Verifica que la progresión de actividades respete la complejidad creciente de los contenidos",
        "Revisa que los tiempos de cada semana sean realistas para el docente y el estudiante",
    ],
}

DEFAULT_SUGGESTED_ACTIONS = [
    "Revisa los materiales del módulo actual disponibles en el curso ¡RAC!",
    "Consulta con un tutor humano si tienes dudas conceptuales sobre el diseño",
    "Valida tu propuesta con los lineamientos curriculares de la Universidad El Bosque",
]


# ---------------------------------------------------------------------------
# Conversation history store
# ---------------------------------------------------------------------------
# Keyed by session_id. Each entry: {"turns": [...], "last_active": datetime}
# Turns are pruned to MAX_HISTORY_TURNS and sessions expire after TTL.

CONVERSATION_TTL_MINUTES = 30
MAX_HISTORY_TURNS = 5

_conversation_store: Dict[str, dict] = {}


def _prune_expired_sessions():
    cutoff = datetime.utcnow() - timedelta(minutes=CONVERSATION_TTL_MINUTES)
    expired = [sid for sid, data in _conversation_store.items()
               if data["last_active"] < cutoff]
    for sid in expired:
        del _conversation_store[sid]


def _get_history(session_id: str) -> List[dict]:
    if not session_id:
        return []
    _prune_expired_sessions()
    entry = _conversation_store.get(session_id)
    return entry["turns"] if entry else []


def _add_to_history(session_id: str, question: str, answer: str):
    if not session_id:
        return
    if session_id not in _conversation_store:
        _conversation_store[session_id] = {"turns": [], "last_active": datetime.utcnow()}
    entry = _conversation_store[session_id]
    entry["turns"].append({"q": question, "a": answer})
    if len(entry["turns"]) > MAX_HISTORY_TURNS:
        entry["turns"] = entry["turns"][-MAX_HISTORY_TURNS:]
    entry["last_active"] = datetime.utcnow()


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting from LLM-generated text."""
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Inline code: `text`
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text


def _log_interaction(entry: dict):
    """Append a query/response record to the JSONL interaction log."""
    try:
        log_path = settings.DATA_LOGS_PATH
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "queries.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"Could not write interaction log: {exc}")


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

        # Retrieve relevant context using RAG (single embedding computation)
        retrieved_chunks = retriever.retrieve(
            query=request.question,
            module=request.module,
            top_k=5
        )

        # Build context string directly from retrieved_chunks to avoid a second embedding call
        if retrieved_chunks:
            parts = [f"[Fragmento {i + 1}]\n{c['content']}" for i, c in enumerate(retrieved_chunks)]
            sources_label = ", ".join({c["source"] for c in retrieved_chunks})
            context = (
                "CONTEXTO RECUPERADO DEL CURSO:\n"
                + "\n\n".join(parts)
                + f"\n\nFUENTES: {sources_label}"
            )
        else:
            context = retriever._get_fallback_context(request.module)

        # Sources: only documents that actually contributed to this response
        sources = list({c["source"] for c in retrieved_chunks}) if retrieved_chunks else ["No hay documentos indexados"]

        # Calculate confidence based on retrieval scores
        if retrieved_chunks:
            avg_score = sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks)
            confidence = min(avg_score + 0.3, 0.95)
        else:
            confidence = 0.5

        # Resolve module ID to its display name for the LLM prompt
        module_display = MODULE_DISPLAY_NAMES.get(request.module, "General")

        # Retrieve conversation history for this session
        history = _get_history(request.session_id)

        # Generate response from LLM (async)
        answer = await generate_response(
            question=request.question,
            context=context,
            module=module_display,
            history=history
        )

        answer = _strip_markdown(answer)

        # Persist this turn so subsequent questions have context
        _add_to_history(request.session_id, request.question, answer)

        processing_time_ms = int((time.time() - start_time) * 1000)

        _log_interaction({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "module": request.module,
            "question": request.question,
            "answer": answer,
            "confidence": round(confidence, 2),
            "sources": sources,
            "chunk_scores": [c["score"] for c in retrieved_chunks],
            "processing_time_ms": processing_time_ms,
        })

        suggested_actions = SUGGESTED_ACTIONS.get(request.module, DEFAULT_SUGGESTED_ACTIONS)

        return QuestionResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            module=request.module,
            answer=answer,
            confidence=round(confidence, 2),
            sources=sources,
            suggested_actions=suggested_actions,
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
# Activity Validation Endpoint
# =============================================================================

def _parse_activity_verdict(raw: str) -> tuple[str, str]:
    """Parse the LLM validation response into (verdict, justification)."""
    lines = raw.strip().split("\n")
    verdict = "PARCIALMENTE COHERENTE"
    justification_start = 0

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VEREDICTO:"):
            verdict_text = line.split(":", 1)[1].strip().upper()
            if "NO COHERENTE" in verdict_text:
                verdict = "NO COHERENTE"
            elif "PARCIALMENTE" in verdict_text:
                verdict = "PARCIALMENTE COHERENTE"
            elif "COHERENTE" in verdict_text:
                verdict = "COHERENTE"
            justification_start = i + 1
            break

    justification = "\n".join(lines[justification_start:]).strip()
    return verdict, justification


@app.post("/validate-activity", response_model=ActivityValidationResponse)
async def validate_activity(request: ActivityValidationRequest):
    """
    Validate whether a learning activity's dimension, methodology, and description
    are coherent within the curriculum design framework.

    Uses RAG to retrieve relevant context from the 'Actividades de Aprendizaje'
    module, then asks the LLM to evaluate consistency and provide a structured verdict.
    """
    start_time = time.time()

    logger.info(
        f"Activity validation - user_id: {request.user_id}, "
        f"dimension: '{request.dimension[:40]}'"
    )

    try:
        retriever = get_rag_retriever()

        rag_query = f"actividades de aprendizaje {request.dimension} {request.metodologia}"
        retrieved_chunks = retriever.retrieve(
            query=rag_query,
            module="actividades_aprendizaje",
            top_k=4
        )

        if retrieved_chunks:
            parts = [f"[Fragmento {i + 1}]\n{c['content']}" for i, c in enumerate(retrieved_chunks)]
            context = "\n\n".join(parts)
        else:
            context = "No se encontró contexto específico sobre actividades de aprendizaje."

        sources = list({c["source"] for c in retrieved_chunks}) if retrieved_chunks else []

        raw_response = await validate_learning_activity(
            resultado_aprendizaje=request.resultado_aprendizaje,
            dimension=request.dimension,
            metodologia=request.metodologia,
            descripcion=request.descripcion,
            context=context,
        )

        if "VEREDICTO:" not in raw_response.upper():
            raise HTTPException(status_code=503, detail=raw_response)

        verdict, justification = _parse_activity_verdict(raw_response)

        processing_time_ms = int((time.time() - start_time) * 1000)

        _log_interaction({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "module": "actividades_aprendizaje",
            "type": "activity_validation",
            "dimension": request.dimension,
            "metodologia": request.metodologia,
            "descripcion": request.descripcion,
            "verdict": verdict,
            "sources": sources,
            "processing_time_ms": processing_time_ms,
        })

        return ActivityValidationResponse(
            verdict=verdict,
            justification=justification,
            sources=sources,
            model_used=settings.OLLAMA_MODEL,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Error validating activity: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )


@app.post("/validate-evaluation", response_model=ActivityValidationResponse)
async def validate_evaluation(request: EvaluationValidationRequest):
    """
    Validate whether an evaluation design is coherent with its learning outcome,
    the associated learning activity, and internally consistent across its own
    components (type, moment, actors, means, techniques, instruments).
    """
    start_time = time.time()

    logger.info(
        f"Evaluation validation - user_id: {request.user_id}, "
        f"tipo: '{request.tipo}', momento: '{request.momento}'"
    )

    try:
        retriever = get_rag_retriever()

        rag_query = f"evaluación {request.tipo} {request.momento} {request.actores} resultados de aprendizaje diseño curricular"
        retrieved_chunks = retriever.retrieve(
            query=rag_query,
            module="evaluacion",
            top_k=4
        )

        if retrieved_chunks:
            parts = [f"[Fragmento {i + 1}]\n{c['content']}" for i, c in enumerate(retrieved_chunks)]
            context = "\n\n".join(parts)
        else:
            context = "No se encontró contexto específico sobre diseño de evaluación."

        sources = list({c["source"] for c in retrieved_chunks}) if retrieved_chunks else []

        raw_response = await validate_evaluation_design(
            resultado_aprendizaje=request.resultado_aprendizaje,
            dimension=request.dimension,
            metodologia=request.metodologia,
            descripcion_actividad=request.descripcion_actividad,
            descripcion_evaluacion=request.descripcion_evaluacion,
            tipo=request.tipo,
            momento=request.momento,
            actores=request.actores,
            medios=request.medios,
            tecnicas=request.tecnicas,
            instrumentos=request.instrumentos,
            context=context,
        )

        if "VEREDICTO:" not in raw_response.upper():
            raise HTTPException(status_code=503, detail=raw_response)

        verdict, justification = _parse_activity_verdict(raw_response)

        processing_time_ms = int((time.time() - start_time) * 1000)

        _log_interaction({
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": request.user_id,
            "session_id": request.session_id,
            "module": "evaluacion",
            "type": "evaluation_validation",
            "tipo": request.tipo,
            "momento": request.momento,
            "actores": request.actores,
            "verdict": verdict,
            "sources": sources,
            "processing_time_ms": processing_time_ms,
        })

        return ActivityValidationResponse(
            verdict=verdict,
            justification=justification,
            sources=sources,
            model_used=settings.OLLAMA_MODEL,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating evaluation: {e}")
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
        task_id, created = indexing_service.start_indexing()
        task_status = indexing_service.get_status(task_id)

        if created:
            message = "Indexing task started. Use /index/status/{task_id} to check progress."
        else:
            message = (
                f"An indexing task is already in progress (status: {task_status['status']}). "
                "Use /index/status/{task_id} to monitor it."
            )

        return IndexingStartResponse(
            task_id=task_id,
            status=task_status["status"],
            message=message
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
