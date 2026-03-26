"""
Ollama service for TEIA Tutor AI.
Handles all communication with the Ollama LLM service.
"""
import asyncio
import aiohttp
from typing import List, Optional
from config import settings
from utils.logger import get_logger

# ── Value → Label maps (mirror of front-end option lists) ─────────────────────

_METODOLOGIA_LABELS = {
    'proyectos':    'Aprendizaje basado en proyectos',
    'juegos':       'Aprendizaje basado en juegos',
    'invertido':    'Aprendizaje invertido',
    'evidencia':    'Aprendizaje basado en evidencia',
    'dialogo':      'Diálogo reflexivo',
    'cooperativo':  'Aprendizaje cooperativo',
    'problemas':    'Aprendizaje basado en problemas',
    'investigacion':'Investigación - Acción',
    'servicio':     'Aprendizaje a través del servicio',
    'adaptativo':   'Aprendizaje adaptativo',
}

_TIPO_LABELS = {
    'sumativa':  'Sumativa',
    'formativa': 'Formativa',
    'mixta':     'Mixta',
}

_MOMENTO_LABELS = {
    'inicial': 'Inicial',
    'media':   'Media procesual',
    'final':   'Final',
}

_ACTORES_LABELS = {
    'hetero': 'Heteroevaluación',
    'co':     'Coevaluación',
    'auto':   'Autoevaluación',
}

_MEDIOS_LABELS = {
    'carpeta_dossier':  'Carpeta o dossier / carpeta colaborativa',
    'control_examen':   'Control (Examen)',
    'cuaderno':         'Cuaderno / cuaderno de notas / cuaderno de campo',
    'cuestionario':     'Cuestionario',
    'diario':           'Diario reflexivo / diario de clase',
    'estudio_casos':    'Estudio de casos',
    'ensayo':           'Ensayo',
    'examen':           'Examen',
    'foro_virtual':     'Foro virtual',
    'memoria':          'Memoria',
    'monografia':       'Monografía',
    'informe':          'Informe',
    'portafolio':       'Portafolio / portafolio electrónico',
    'poster':           'Póster',
    'proyecto':         'Proyecto',
    'pruebas_objetivas':'Pruebas objetivas',
    'recension':        'Recensión',
    'test_diagnostico': 'Test diagnóstico',
    'trabajo_escrito':  'Trabajo escrito',
    'comunicacion_oral':'Comunicación oral',
    'cuestionario_oral':'Cuestionario oral',
    'debate':           'Debate / diálogo grupal',
    'exposicion':       'Exposición',
    'discusion_grupal': 'Discusión grupal',
    'mesa_redonda':     'Mesa redonda',
    'ponencia':         'Ponencia',
    'pregunta_clase':   'Pregunta de clase',
    'presentacion_oral':'Presentación oral',
    'practica_supervisada': 'Práctica supervisada',
    'demostracion':     'Demostración / actuación / representación',
    'role_playing':     'Role playing',
}

_TECNICAS_LABELS = {
    'analisis_documental':   'Análisis documental',
    'analisis_producciones': 'Análisis de producciones',
    'observacion_directa':   'Observación directa del alumno',
    'observacion_grupo':     'Observación del grupo',
    'observacion_sistematica':'Observación sistemática',
    'analisis_audio_video':  'Análisis de grabación de audio o video',
    'autoevaluacion':        'Autoevaluación',
    'coevaluacion':          'Evaluación entre pares',
    'evaluacion_colaborativa':'Evaluación compartida o colaborativa',
}

_INSTRUMENTOS_LABELS = {
    'diario_profesor':       'Diario del profesor',
    'escala_comprobacion':   'Escala de comprobación',
    'escala_diferencial':    'Escala de diferencial semántico',
    'escala_verbal_numerica':'Escala verbal o numérica',
    'escala_rubrica':        'Escala descriptiva o rúbrica',
    'escala_estimacion':     'Escala de estimación',
    'ficha_observacion':     'Ficha de observación',
    'lista_control':         'Lista de control',
    'matrices_decision':     'Matrices de decisión',
    'fichas_seguimiento':    'Fichas de seguimiento individual o grupal',
    'fichas_autoevaluacion': 'Fichas de autoevaluación',
    'fichas_entre_iguales':  'Fichas de evaluación entre iguales',
    'informe_expertos':      'Informe de expertos',
    'informe_autoevaluacion':'Informe de autoevaluación',
}


def _resolve(value: str, mapping: dict) -> str:
    """Return the human-readable label for a value key, or the value itself if not found."""
    return mapping.get(value, value)


def _resolve_list(values: List[str], mapping: dict) -> str:
    """Return a comma-separated string of labels for a list of value keys."""
    return ", ".join(_resolve(v, mapping) for v in values)

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
2. Estructura tu respuesta así: explica el concepto o criterio de forma clara y directa. No termines con preguntas al docente.
3. No incluyas ejemplos en tu respuesta a menos que el docente lo solicite explícitamente (frases como "dame un ejemplo", "ponme un ejemplo", "muéstrame cómo", etc.).
4. Si el docente pide algo "resumido", "breve" o "corto", limítate a 3-4 oraciones.
5. Si el contexto recuperado no cubre la pregunta o no cuentas con información suficiente para responder con certeza, responde honestamente con algo como: "Sobre este punto específico no cuento con información en mi base de conocimiento. Te recomiendo consultarlo directamente con tu tutor humano." No inventes información ni especules.
6. Responde en el contexto del módulo "{module}" sin necesidad de mencionarlo explícitamente en cada respuesta.
7. Adapta la extensión al tipo de pregunta: preguntas conceptuales merecen más detalle; preguntas procedimentales, pasos concretos.
8. No repitas el enunciado de la pregunta en tu respuesta.

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


async def validate_learning_activity(
    resultado_aprendizaje: str,
    dimension: str,
    metodologia: str,
    descripcion: str,
    context: str = "",
) -> str:
    """
    Evaluate whether a learning activity's dimension, methodology, and description
    are coherent within the curriculum design framework.

    Returns raw LLM text starting with:
      "VEREDICTO: COHERENTE", "VEREDICTO: PARCIALMENTE COHERENTE", or
      "VEREDICTO: NO COHERENTE"
    followed by justification and suggestions.
    """
    prompt = f"""Eres TEIA, un tutor especializado en diseño curricular de la Universidad El Bosque.
Apoyas a docentes en el curso "En sus marcas, listos, ¡RAC!" para diseñar microcurrículos de calidad.

CONTEXTO DEL CURSO SOBRE ACTIVIDADES DE APRENDIZAJE:
{context}

Un docente está diseñando una actividad de aprendizaje con los siguientes elementos:
- Resultado de aprendizaje: {resultado_aprendizaje}
- Dimensión: {dimension}
- Metodología de aprendizaje activo: {metodologia}
- Descripción de la actividad: {descripcion}

Evalúa si la dimensión, la metodología y la descripción son coherentes con el resultado de aprendizaje y entre sí, dentro del marco del diseño curricular basado en resultados de aprendizaje de la Universidad El Bosque.

INSTRUCCIONES DE RESPUESTA:
1. Comienza tu respuesta OBLIGATORIAMENTE con una de estas tres líneas exactas (sin texto antes):
   VEREDICTO: COHERENTE
   VEREDICTO: PARCIALMENTE COHERENTE
   VEREDICTO: NO COHERENTE
2. A continuación, en 2-4 oraciones, justifica tu veredicto comenzando con "Teniendo en cuenta el resultado de aprendizaje [resultado], ..." y explicando la relación entre todos los elementos.
3. Si hay inconsistencias o áreas de mejora, incluye 1-2 sugerencias concretas y breves para ajustar la actividad.
4. Responde en español formal propio del contexto universitario colombiano.
5. No repitas literalmente los valores ingresados por el docente; parafraséalos en tu justificación.

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
                        "temperature": 0.2,
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
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama returned status {response.status}")
                    return f"VEREDICTO: PARCIALMENTE COHERENTE\nError en el servicio Ollama: {response.status}"

    except asyncio.TimeoutError:
        logger.error("Ollama timed out (validate_learning_activity)")
        return "El servicio de IA está tardando demasiado. Por favor intenta de nuevo."
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error (validate_learning_activity): {e}")
        return "Error de conexión con el servicio de IA. Por favor intenta más tarde."
    except Exception as e:
        logger.error(f"Unexpected error (validate_learning_activity): {e}")
        return "Error inesperado con el servicio de IA. Por favor intenta más tarde."


async def validate_evaluation_design(
    resultado_aprendizaje: str,
    dimension: str,
    metodologia: str,
    descripcion_actividad: str,
    descripcion_evaluacion: str,
    tipo: str,
    momento: str,
    actores: str,
    medios: List[str],
    tecnicas: List[str],
    instrumentos: List[str],
    context: str = "",
) -> str:
    """
    Evaluate whether an evaluation design is coherent with the learning outcome,
    the learning activity, and the internal consistency of its own components
    (type, moment, actors, means, techniques, instruments).

    Returns raw LLM text starting with VEREDICTO: COHERENTE / PARCIALMENTE COHERENTE / NO COHERENTE.
    """
    metodologia_label    = _resolve(metodologia, _METODOLOGIA_LABELS)
    tipo_label           = _resolve(tipo, _TIPO_LABELS)
    momento_label        = _resolve(momento, _MOMENTO_LABELS)
    actores_label        = _resolve(actores, _ACTORES_LABELS)
    medios_label         = _resolve_list(medios, _MEDIOS_LABELS)
    tecnicas_label       = _resolve_list(tecnicas, _TECNICAS_LABELS)
    instrumentos_label   = _resolve_list(instrumentos, _INSTRUMENTOS_LABELS)

    prompt = f"""Eres TEIA, un tutor especializado en diseño curricular de la Universidad El Bosque.
Apoyas a docentes en el curso "En sus marcas, listos, ¡RAC!" para diseñar microcurrículos de calidad.

CONTEXTO DEL CURSO SOBRE EVALUACIÓN:
{context}

Un docente ha diseñado la siguiente actividad de aprendizaje y su correspondiente evaluación:

ACTIVIDAD DE APRENDIZAJE:
- Resultado de aprendizaje: {resultado_aprendizaje}
- Dimensión: {dimension}
- Metodología de aprendizaje activo: {metodologia_label}
- Descripción de la actividad: {descripcion_actividad}

DISEÑO DE EVALUACIÓN:
- Descripción: {descripcion_evaluacion}
- Tipo de evaluación: {tipo_label}
- Momento: {momento_label}
- Actores: {actores_label}
- Medios: {medios_label}
- Técnicas: {tecnicas_label}
- Instrumentos: {instrumentos_label}

Evalúa si el diseño de evaluación es coherente con el resultado de aprendizaje, con la actividad de aprendizaje propuesta, y si sus propios componentes (tipo, momento, actores, medios, técnicas e instrumentos) son consistentes entre sí.

INSTRUCCIONES DE RESPUESTA:
1. Comienza tu respuesta OBLIGATORIAMENTE con una de estas tres líneas exactas (sin texto antes):
   VEREDICTO: COHERENTE
   VEREDICTO: PARCIALMENTE COHERENTE
   VEREDICTO: NO COHERENTE
2. A continuación, en 3-5 oraciones, justifica tu veredicto comenzando con "Teniendo en cuenta el resultado de aprendizaje [resultado], ..."
3. Evalúa específicamente: (a) si el tipo y momento son apropiados para el resultado de aprendizaje, (b) si los actores son coherentes con el tipo de evaluación, y (c) si los medios, técnicas e instrumentos son consistentes entre sí y con los actores.
4. Si hay inconsistencias, incluye 1-2 sugerencias concretas de ajuste.
5. Responde en español formal propio del contexto universitario colombiano.
6. No repitas literalmente los valores ingresados; parafraséalos.

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
                        "temperature": 0.2,
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
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama returned status {response.status}")
                    return f"VEREDICTO: PARCIALMENTE COHERENTE\nError en el servicio Ollama: {response.status}"

    except asyncio.TimeoutError:
        logger.error("Ollama timed out (validate_evaluation_design)")
        return "El servicio de IA está tardando demasiado. Por favor intenta de nuevo."
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error (validate_evaluation_design): {e}")
        return "Error de conexión con el servicio de IA. Por favor intenta más tarde."
    except Exception as e:
        logger.error(f"Unexpected error (validate_evaluation_design): {e}")
        return "Error inesperado con el servicio de IA. Por favor intenta más tarde."
