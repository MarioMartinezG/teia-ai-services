# =============================================================================
# TEIA Tutor AI Service - Lightweight Docker Image
# =============================================================================
# Imagen ligera solo con la aplicación FastAPI.
# Ollama se ejecuta en un contenedor separado.
# Toda la configuración se realiza via variables de entorno en docker-compose.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# src/ modules are imported without the "src." prefix (e.g. "from config import settings")
ENV PYTHONPATH=/app/src

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: heavy ML deps (torch, sentence-transformers, ragas, chromadb…)
# Cached independently — only re-runs when requirements-ml.txt changes.
COPY requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements-ml.txt

# Layer 2: lightweight app deps (fastapi, uvicorn, pypdf…)
# Re-runs only when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/

# Create data directories (data/logs is used by the interaction-log writer)
RUN mkdir -p data/raw data/processed data/chroma_db data/logs logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
