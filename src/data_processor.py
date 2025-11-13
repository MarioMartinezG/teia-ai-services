import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SimpleDataProcessor:
    def __init__(self):
        self.data_path = Path(__file__).parent / "data" / "raw"
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Asegura que exista el directorio de datos"""
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Crear archivos básicos si no existen
        default_files = {
            "conceptos_clave.md": """
# Conceptos Clave - Curso "En sus marcas, listos, iRAC!"

## Resultados de Aprendizaje
Declaraciones que describen lo que el estudiante será capaz de demostrar al final del proceso.

## Microcurrículo
Diseño detallado de una asignatura que incluye resultados, actividades y evaluación.

## Evaluación Formativa
Evaluación continua durante el proceso de aprendizaje para mejorar.
""",
            "modulo_resultados.md": """
# Módulo: Resultados de Aprendizaje

## Características de buenos resultados:
- Específicos y medibles
- Centrados en el estudiante
- Alineados con competencias
- Redactados con verbos de acción
"""
        }

        for filename, content in default_files.items():
            file_path = self.data_path / filename
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"📁 Creado archivo: {filename}")

    def get_context_for_module(self, module: str) -> str:
        """Obtiene contexto específico para un módulo"""
        try:
            # Por ahora retorna contexto básico
            # Luego cargará desde archivos específicos
            base_context = """
            Universidad El Bosque - Curso: "En sus marcas, listos, iRAC!"
            Enfoque: Diseño curricular centrado en el estudiante
            Objetivo: Fortalecer competencias pedagógicas en diseño de microcurrículos
            """

            module_specific = {
                "resultados_aprendizaje": "Módulo enfocado en diseño de resultados de aprendizaje claros, medibles y alineados con competencias.",
                "evaluacion": "Módulo sobre estrategias de evaluación formativa y sumativa, rúbricas y retroalimentación.",
                "caracterizacion": "Módulo sobre identificación de características de la asignatura y contexto institucional.",
                "actividades_aprendizaje": "Módulo sobre diseño de actividades activas y centradas en el estudiante."
            }

            return base_context + "\n" + module_specific.get(module, "")

        except Exception as e:
            logger.error(f"Error cargando contexto para módulo {module}: {e}")
            return "Contexto del curso no disponible temporalmente."