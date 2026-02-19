"""
Ollama service for TEIA Tutor AI.
Handles all communication with the Ollama LLM service.
"""
import asyncio
import aiohttp
from typing import List, Optional
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


async def generate_response(
    question: str,
    context: str = "",
    module: str = "",
    history: Optional[List[dict]] = None,
) -> str:
    """
    Generate a response from Ollama with curriculum design specialization.

    Args:
        question: The user's question
        context: RAG-retrieved course context
        module: The current module display name
        history: List of previous turns [{"q": ..., "a": ...}]

    Returns:
        The generated response text
    """
    # Build conversation history block if there are previous turns
    history_block = ""
    if history:
        turns = [f"Docente: {t['q']}\nTEIA: {t['a']}" for t in history]
        history_block = "CONVERSACIÓN PREVIA:\n" + "\n\n".join(turns) + "\n\n"

    prompt = f"""Eres TEIA, un tutor especializado en diseño curricular de la Universidad El Bosque.
Apoyas a docentes en el curso "En sus marcas, listos, ¡RAC!" para diseñar microcurrículos de calidad.

CONTEXTO RECUPERADO DEL CURSO:
{context}

{history_block}MÓDULO ACTUAL: {module}
PREGUNTA DEL DOCENTE: {question}

INSTRUCCIONES:
1. Responde SIEMPRE en español formal pero cercano, propio del contexto universitario colombiano.
2. Estructura tu respuesta así: explica el concepto o criterio y, si es útil, ofrece UN ejemplo ilustrativo breve. No termines con preguntas al docente.
3. Si el docente pide algo "resumido", "breve" o "corto", limítate a 3-4 oraciones.
4. Si el contexto recuperado no es suficiente para responder con precisión, indícalo con honestidad y orienta al docente sobre dónde buscar más información.
5. Responde en el contexto del módulo "{module}" sin necesidad de mencionarlo explícitamente en cada respuesta.
6. Adapta la extensión al tipo de pregunta: preguntas conceptuales merecen más detalle; preguntas procedimentales, pasos concretos.
7. No repitas el enunciado de la pregunta en tu respuesta.

RESPUESTA:"""

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
                        "num_predict": settings.OLLAMA_MAX_TOKENS,
                        "num_ctx": 8192
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
