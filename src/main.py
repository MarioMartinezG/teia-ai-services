from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import requests
import os
from data_processor import SimpleDataProcessor
from utils.logger import get_logger, setup_logging, RequestLogger

# Obtener logger
logger = get_logger("main")
http_logger = RequestLogger()

# Configurar logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/teia_tutor.log"),
    enable_json=os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"
)

app = FastAPI(
    title="TEIA Tutor AI Service",
    description="Servicio de Tutoría Inteligente para el curso 'En sus marcas, listos, iRAC!'",
    version="1.0.0"
)

# Inicializar procesador de datos
data_processor = SimpleDataProcessor()

class QuestionRequest(BaseModel):
    question: str
    module: Optional[str] = None
    user_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    suggested_actions: Optional[List[str]] = None

class ModuleInfo(BaseModel):
    id: str
    name: str
    description: str

class SystemStatus(BaseModel):
    service: str
    status: str
    ollama_connected: bool
    ollama_models: List[str]
    timestamp: str

def ask_ollama_directly(question: str, context: str = "", module: str = "") -> str:
    """Llama DIRECTAMENTE a Ollama con prompt especializado"""

    prompt = f"""
    Eres TEIA, un tutor especializado en diseño curricular de la Universidad El Bosque.
    Estás apoyando el curso "En sus marcas, listos, iRAC!" para docentes.

    CONTEXTO INSTITUCIONAL:
    - Universidad El Bosque - Modelo educativo centrado en el estudiante
    - Enfoque en competencias y resultados de aprendizaje
    - Metodologías activas de enseñanza
    - Evaluación formativa y sumativa

    INFORMACIÓN DEL CURSO:
    {context}

    MÓDULO ACTUAL: {module}
    PREGUNTA DEL DOCENTE: {question}

    INSTRUCCIONES:
    1. Responde en ESPAÑOL claro y profesional
    2. Enfócate en metodologías CENTRADAS EN EL ESTUDIANTE
    3. Proporciona EJEMPLOS PRÁCTICOS cuando sea posible
    4. Si no tienes información específica, sugiere consultar los materiales del curso
    5. Mantén un tono de APOYO y ORIENTACIÓN

    RESPUESTA:
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No se pudo generar respuesta").strip()
        else:
            return f"Error en el servicio Ollama: {response.status_code}"

    except Exception as e:
        logger.error(f"Error conectando con Ollama: {e}")
        return "Error de conexión con el servicio de IA. Por favor intenta más tarde."

@app.get("/")
async def root():
    return {
        "message": "TEIA Tutor AI Service 🎓",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "teia-tutor"}

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Estado detallado del sistema"""
    try:
        ollama_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_models = ollama_response.json().get("models", []) if ollama_response.status_code == 200 else []

        return SystemStatus(
            service="teia-tutor",
            status="running",
            ollama_connected=len(ollama_models) > 0,
            ollama_models=[model["name"] for model in ollama_models],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return SystemStatus(
            service="teia-tutor",
            status="running",
            ollama_connected=False,
            ollama_models=[],
            timestamp=datetime.now().isoformat()
        )

@app.get("/modules", response_model=List[ModuleInfo])
async def get_modules():
    """Retorna los módulos disponibles del curso"""
    modules = [
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
    return modules

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint principal para hacer preguntas al tutor"""
    logger.info(f"Pregunta recibida: {request.question} - Módulo: {request.module}")

    try:
        # Obtener contexto específico del módulo
        context = data_processor.get_context_for_module(request.module or "")

        # Obtener respuesta de Ollama
        answer = ask_ollama_directly(
            question=request.question,
            context=context,
            module=request.module or "General"
        )

        return QuestionResponse(
            answer=answer,
            confidence=0.8,
            sources=["Sistema TEIA - Ollama Integration"],
            suggested_actions=[
                "Revisar materiales específicos del módulo",
                "Consultar con tutor humano para casos complejos",
                "Validar con lineamientos institucionales"
            ]
        )

    except Exception as e:
        logger.error(f"Error procesando pregunta: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Iniciando TEIA Tutor AI Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")