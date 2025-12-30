"""
Ollama service for TEIA Tutor AI.
Handles all communication with the Ollama LLM service.
"""
import asyncio
import aiohttp
from typing import List
from config import settings
from utils.logger import get_logger

logger = get_logger("ollama_service")


async def check_ollama_health() -> tuple[bool, List[str]]:
    """
    Check if Ollama is running and return available models.

    Returns:
        Tuple of (is_connected, list_of_models)
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{settings.OLLAMA_URL}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return True, models
                return False, []
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False, []


async def generate_response(question: str, context: str = "", module: str = "") -> str:
    """
    Generate a response from Ollama with curriculum design specialization.

    Args:
        question: The user's question
        context: Additional context for the module
        module: The current module name

    Returns:
        The generated response text
    """
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
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_predict": 500,
                        "num_ctx": 2048
                    }
                },
                timeout=aiohttp.ClientTimeout(total=settings.OLLAMA_TIMEOUT)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "No se pudo generar respuesta").strip()
                else:
                    logger.error(f"Ollama returned status {response.status}")
                    return f"Error en el servicio Ollama: {response.status}"

    except asyncio.TimeoutError:
        logger.error("Ollama request timed out")
        return "El servicio de IA está tardando demasiado. Por favor intenta de nuevo."
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error connecting to Ollama: {e}")
        return "Error de conexión con el servicio de IA. Por favor intenta más tarde."
    except Exception as e:
        logger.error(f"Unexpected error connecting to Ollama: {e}")
        return "Error inesperado con el servicio de IA. Por favor intenta más tarde."
