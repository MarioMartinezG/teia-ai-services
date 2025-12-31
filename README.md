# TEIA - Servicio de Tutoría Inteligente con IA

API para el Tutor Inteligente TEIA, que se integra con la bitácora digital del curso "En sus marcas, listos, ¡RAC!" de la Universidad El Bosque.

## Descripción

TEIA es un sistema de tutoría inteligente diseñado para apoyar a los docentes en el diseño curricular y la planificación de cursos. Utiliza tecnología de **Generación Aumentada por Recuperación (RAG)** para proporcionar respuestas contextualizadas basadas en materiales del curso.

### Características Principales

- **Tutoría especializada** en diseño curricular con 6 módulos temáticos
- **RAG (Retrieval-Augmented Generation)** para respuestas fundamentadas en documentos del curso
- **Soporte multilingüe** optimizado para español
- **Indexación inteligente** de documentos PDF, Excel y Markdown
- **Seguimiento de interacciones** para análisis y reportes futuros
- **API REST** moderna con FastAPI

## Stack Tecnológico

| Categoría | Tecnologías |
|-----------|-------------|
| **Framework** | FastAPI, Uvicorn |
| **IA/ML** | Ollama (Qwen 2.5, Mistral), Sentence-Transformers |
| **Base de Datos Vectorial** | ChromaDB |
| **Procesamiento de Documentos** | LangChain, PyPDF, python-docx, openpyxl |
| **Validación de Datos** | Pydantic |

## Estructura del Proyecto

```
teia-ai-services/
├── src/                           # Código principal de la aplicación
│   ├── main.py                    # Punto de entrada de FastAPI
│   ├── config.py                  # Gestión de configuración
│   ├── models.py                  # Modelos de datos Pydantic
│   ├── middleware.py              # Configuración de CORS
│   ├── data_processor.py          # Utilidades de procesamiento
│   ├── services/                  # Servicios de negocio
│   │   ├── ollama_service.py      # Comunicación con Ollama LLM
│   │   ├── rag_retriever.py       # Lógica de recuperación RAG
│   │   ├── embedding_service.py   # Generación de embeddings
│   │   └── __init__.py
│   └── utils/                     # Módulos utilitarios
│       ├── logger.py              # Configuración de logging
│       └── __init__.py
├── scripts/                       # Scripts independientes
│   └── index_documents.py         # Indexación de documentos para RAG
├── data/                          # Almacenamiento de datos (configurable via DATA_BASE_PATH)
│   ├── raw/                       # Documentos para indexar (cualquier estructura)
│   ├── processed/                 # Datos procesados (chunks.json)
│   └── chroma_db/                 # Base de datos vectorial
├── logs/                          # Logs de la aplicación
├── .env                           # Variables de entorno (local)
├── .env.example                   # Ejemplo de configuración
└── requirements.txt               # Dependencias de Python
```

## Requisitos Previos

- **Python 3.10+**
- **Ollama** instalado y ejecutándose localmente
- Modelo LLM descargado en Ollama (ej: `qwen2.5:7b-instruct`)

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/teia-ai-services.git
cd teia-ai-services
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/MacOS
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copiar el archivo de ejemplo y ajustar según sea necesario:

```bash
cp .env.example .env
```

### 5. Configurar Ollama

Asegurarse de que Ollama esté instalado y ejecutándose:

```bash
# Instalar el modelo principal
ollama pull qwen2.5:7b-instruct

# (Opcional) Instalar modelo de respaldo
ollama pull mistral:7b-instruct
```

### 6. Indexar documentos (opcional)

Si se tienen documentos del curso para indexar:

```bash
python scripts/index_documents.py
```

## Configuración

Las siguientes variables de entorno están disponibles en el archivo `.env`:

### Configuración de Rutas de Datos

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `DATA_BASE_PATH` | Ruta base para todos los datos | `./data` (relativo al proyecto) |
| `SUPPORTED_EXTENSIONS` | Extensiones de archivo soportadas | `.pdf,.xlsx,.xls,.md,.txt` |

La estructura de subdirectorios se crea automáticamente:
- `{DATA_BASE_PATH}/raw/` - Documentos originales para indexar
- `{DATA_BASE_PATH}/processed/` - Datos procesados
- `{DATA_BASE_PATH}/chroma_db/` - Base de datos vectorial

### Configuración de Ollama

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `OLLAMA_URL` | URL del servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo LLM principal | `qwen2.5:7b-instruct` |
| `OLLAMA_FALLBACK_MODEL` | Modelo de respaldo | `mistral:7b-instruct` |
| `OLLAMA_TIMEOUT` | Timeout en segundos | `120` |
| `OLLAMA_MAX_TOKENS` | Tokens máximos de respuesta | `1024` |

### Configuración del Servidor

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `HOST` | Dirección de enlace | `0.0.0.0` |
| `PORT` | Puerto del servicio | `8000` |
| `ALLOWED_ORIGINS` | Orígenes CORS permitidos | `http://localhost:3000` |

### Configuración de Logging

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `LOG_LEVEL` | Nivel de logging | `INFO` |
| `LOG_FILE` | Ruta del archivo de logs | `logs/teia_tutor.log` |
| `ENABLE_JSON_LOGS` | Habilitar logs en JSON | `false` |

## Ejecución

### Iniciar el servidor

```bash
python src/main.py
```

El servidor estará disponible en `http://localhost:8000`

### Verificar estado

```bash
# Verificar salud del servicio
curl http://localhost:8000/health

# Obtener estado detallado
curl http://localhost:8000/status
```

## Endpoints de la API

### Información General

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Información del servicio |
| `GET` | `/health` | Verificación de salud |
| `GET` | `/status` | Estado detallado del sistema |
| `GET` | `/modules` | Lista de módulos disponibles |

### Tutoría

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/ask` | Realizar una pregunta al tutor |

#### Ejemplo de solicitud `/ask`

```json
{
  "question": "¿Cómo puedo definir resultados de aprendizaje efectivos?",
  "module": "resultados_aprendizaje",
  "user_id": "usuario123",
  "session_id": "sesion456"
}
```

#### Ejemplo de respuesta

```json
{
  "answer": "Para definir resultados de aprendizaje efectivos...",
  "confidence": 0.85,
  "sources": ["Aprendizaje centrado en el estudiante.pdf"],
  "metadata": {
    "request_id": "uuid-generado",
    "timestamp": "2024-01-15T10:30:00Z",
    "module": "resultados_aprendizaje",
    "model_used": "qwen2.5:7b-instruct",
    "processing_time_ms": 2500
  }
}
```

## Módulos Disponibles

El tutor cubre 6 módulos de diseño curricular:

| ID del Módulo | Nombre | Descripción |
|---------------|--------|-------------|
| `caracterizacion` | Caracterización de la Asignatura | Definición y alcance de la materia |
| `factores_situacionales` | Factores Situacionales | Análisis del contexto educativo |
| `resultados_aprendizaje` | Resultados de Aprendizaje | Definición de objetivos de aprendizaje |
| `actividades_aprendizaje` | Actividades de Aprendizaje | Diseño de actividades pedagógicas |
| `evaluacion` | Evaluación | Estrategias y rúbricas de evaluación |
| `secuencia` | Secuencia del Curso | Cronograma y organización temporal |

## Indexación de Documentos

El script de indexación procesa documentos y los almacena en la base de datos vectorial para RAG:

```bash
python scripts/index_documents.py
```

### Tipos de archivo soportados

Los tipos de archivo soportados son configurables mediante la variable `SUPPORTED_EXTENSIONS`. Por defecto:

- PDF (`.pdf`)
- Excel (`.xlsx`, `.xls`)
- Markdown (`.md`)
- Texto plano (`.txt`)

### Organización de documentos

Los documentos pueden colocarse directamente en `{DATA_BASE_PATH}/raw/` o en cualquier subcarpeta. El script escanea recursivamente y procesa los archivos según su extensión.

```
data/raw/
├── documento1.pdf
├── archivo.xlsx
├── notas.md
└── subcarpeta/
    └── otro_documento.pdf
```

El script detecta automáticamente el módulo correspondiente basándose en el nombre del archivo.

## Arquitectura

### Flujo RAG

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Pregunta   │────▶│ Embedding Service│────▶│  ChromaDB   │
│  del Usuario│     │ (Sentence-Trans) │     │  (Búsqueda) │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Respuesta  │◀────│  Ollama Service  │◀────│  Contexto   │
│  Generada   │     │  (LLM Generation)│     │  Recuperado │
└─────────────┘     └──────────────────┘     └─────────────┘
```

### Componentes Principales

- **EmbeddingService**: Genera embeddings de texto usando Sentence-Transformers
- **RAGRetriever**: Realiza búsqueda semántica en ChromaDB
- **OllamaService**: Comunica con el modelo LLM para generar respuestas
