# =============================================================================
# TEIA Tutor AI Service - Lightweight Docker Image
# =============================================================================
# Imagen ligera solo con la aplicación FastAPI.
# Ollama se ejecuta en un contenedor separado.
# Toda la configuración se realiza via variables de entorno en docker-compose.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/

# Create data directories
RUN mkdir -p data/raw data/processed data/chroma_db logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
