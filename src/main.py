"""
TEIA Tutor AI Service - Main Application
FastAPI application for the Intelligent Tutoring System.
"""
import time
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException

from config import settings
from middleware import setup_middlewares
from models import (
    QuestionRequest,
    QuestionResponse,
    ModuleInfo,
    HealthResponse,
    SystemStatus,
)
from services.ollama_service import generate_response, check_ollama_health
from services.rag_retriever import get_rag_retriever
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
        "status": "active"
    }


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

        # Extract sources from retrieved chunks
        sources = list(set(chunk["source"] for chunk in retrieved_chunks))
        if not sources:
            sources = ["Contexto general del curso"]

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


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting TEIA Tutor AI Service...")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
