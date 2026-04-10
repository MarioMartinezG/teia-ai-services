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

### Gestión de Documentos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/documents` | Lista documentos disponibles para indexar |
| `POST` | `/documents/upload` | Subir documentos (multipart/form-data) |

### Indexación

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/index` | Iniciar proceso de indexación (asíncrono) |
| `GET` | `/index/status/{task_id}` | Consultar estado de una tarea de indexación |
| `GET` | `/index/tasks` | Listar todas las tareas de indexación |

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
- **IndexingService**: Gestiona la indexación asíncrona de documentos

## Configuracion de Modelos de Embeddings

El servicio de embeddings utiliza modelos de Sentence-Transformers para generar representaciones vectoriales del texto. La eleccion del modelo afecta la calidad de la busqueda semantica y el consumo de recursos.

### Modelo Actual

El modelo por defecto es `paraphrase-multilingual-MiniLM-L12-v2`, configurado en `src/services/embedding_service.py`:

```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

### Modelos Disponibles

| Modelo | Dimensiones | Tamano | RAM/VRAM | Velocidad | Calidad Espanol |
|--------|-------------|--------|----------|-----------|-----------------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | ~1GB | Rapido | Buena |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 1.1GB | ~2GB | Medio | Muy buena |
| `intfloat/multilingual-e5-base` | 768 | 1.1GB | ~2GB | Medio | Excelente |
| `intfloat/multilingual-e5-large` | 1024 | 2.2GB | ~4GB | Lento | Excelente |

### Criterios de Seleccion

**Recursos limitados (CPU o GPU basica)**:
- Usar `paraphrase-multilingual-MiniLM-L12-v2` (actual)
- Menor consumo de memoria y tiempo de indexacion

**Balance calidad/rendimiento**:
- Usar `paraphrase-multilingual-mpnet-base-v2`
- Mejora notable en calidad sin gran impacto en recursos

**Maxima calidad (GPU dedicada)**:
- Usar `intfloat/multilingual-e5-base` o `multilingual-e5-large`
- Requiere prefijo `"query: "` para consultas (modificar `embed_query` en `embedding_service.py`)

### Cambiar el Modelo

1. Modificar la constante en `src/services/embedding_service.py`:

```python
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # o el modelo elegido
```

2. Si se usa un modelo E5, modificar el metodo `embed_query`:

```python
def embed_query(self, query: str) -> List[float]:
    # Los modelos E5 requieren prefijo para queries
    return self.embed_text(f"query: {query}")
```

3. Reindexar todos los documentos (obligatorio al cambiar de modelo):

```bash
# POST /index o ejecutar el script
python scripts/index_documents.py
```

**Nota**: Los embeddings generados por diferentes modelos no son compatibles entre si. Cambiar de modelo requiere reindexar completamente los documentos.

## Despliegue en RunPod (Cloud GPU)

[RunPod](https://runpod.io) es una plataforma de cloud computing que permite ejecutar cargas de trabajo con GPU a precios competitivos.

### Arquitectura de Despliegue

```
┌─────────────────────────────────────────────────────────┐
│                      RunPod Pod                         │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │     Ollama      │◀────▶│   TEIA API (Docker)     │  │
│  │  (instalado en  │      │   - FastAPI             │  │
│  │   el pod)       │      │   - RAG/ChromaDB        │  │
│  │  puerto 11434   │      │   puerto 8000           │  │
│  └─────────────────┘      └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

- **Ollama**: Se instala directamente en el pod (no en Docker)
- **TEIA API**: Contenedor ligero (~200MB) con FastAPI

### Requisitos de Hardware

Para ejecutar `qwen2.5:7b-instruct`:

| Configuración | VRAM Requerida | GPU Recomendada | Costo Aprox. |
|---------------|----------------|-----------------|--------------|
| 4-bit quantization | 6 GB | RTX A5000 | ~$0.16/hr |
| 8-bit quantization | 10 GB | RTX 3090 | ~$0.22/hr |
| FP16 (full precision) | 17-20 GB | RTX 4090 | ~$0.44/hr |

**Recomendación económica**: RTX A5000 en Community Cloud.

### Pasos para Desplegar en RunPod

#### 1. Crear cuenta y Pod

1. Registrarse en [runpod.io](https://runpod.io)
2. Ir a **Pods** > **+ Deploy**
3. Seleccionar GPU (ej: RTX A5000)
4. Usar template **PyTorch** (más reciente)
5. En **Customize Deployment**:
   - Exponer puerto **11434** (Ollama)
   - Exponer puerto **8000** (TEIA API)
   - Agregar variable: `OLLAMA_HOST=0.0.0.0`

#### 2. Instalar Ollama en el Pod

Conectar via terminal web y ejecutar:

```bash
# Instalar dependencias y Ollama
apt update && apt install -y lshw zstd
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar Ollama en background
nohup ollama serve > /var/log/ollama.log 2>&1 &

# Esperar a que inicie
sleep 5

# Descargar el modelo
ollama pull qwen2.5:7b-instruct
```

#### 3. Construir y Subir imagen Docker

En tu máquina local:

```bash
# Construir imagen (ligera, ~200MB)
docker build -t tu-usuario/teia-api:latest .

# Subir a Docker Hub
docker push tu-usuario/teia-api:latest
```

#### 4. Ejecutar TEIA API en el Pod

En la terminal del pod:

```bash
# Ejecutar contenedor TEIA
docker run -d \
  --name teia-api \
  -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen2.5:7b-instruct \
  -v /workspace/data:/app/data \
  tu-usuario/teia-api:latest
```

> **Nota**: En RunPod, usa `http://172.17.0.1:11434` o la IP del host para conectar al Ollama del pod.

#### 5. Verificar funcionamiento

```bash
# Verificar Ollama
curl http://localhost:11434/api/tags

# Verificar TEIA API
curl http://localhost:8000/health

# Acceder a la UI
# Abrir en navegador: https://{POD_ID}-8000.proxy.runpod.net/ui
```

### Acceso Externo

RunPod proporciona URLs públicas para los puertos expuestos:

| Servicio | URL |
|----------|-----|
| TEIA API | `https://{POD_ID}-8000.proxy.runpod.net` |
| TEIA UI | `https://{POD_ID}-8000.proxy.runpod.net/ui` |
| Ollama | `https://{POD_ID}-11434.proxy.runpod.net` |

### Desarrollo Local con Docker

Requiere Ollama corriendo en tu máquina:

```bash
# 1. Iniciar Ollama localmente
ollama serve

# 2. En otra terminal, descargar modelo
ollama pull qwen2.5:7b-instruct

# 3. Iniciar TEIA API con Docker
docker-compose up --build

# 4. Acceder a la UI
# http://localhost:8000/ui
```

### Estimación de Costos Mensuales

| Escenario | Uso Diario | GPU | Costo Mensual |
|-----------|------------|-----|---------------|
| Bajo | 2 horas | RTX A5000 | ~$10/mes |
| Moderado | 4 horas | RTX A5000 | ~$19/mes |
| Alto | 8 horas | RTX A5000 | ~$38/mes |

*Precios aproximados de Community Cloud. Apagar el pod cuando no se use para ahorrar.*
