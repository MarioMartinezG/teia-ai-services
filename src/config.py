"""
Configuration module for TEIA Tutor AI Service.
Centralizes all environment variables and application settings.
"""
import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings loaded from environment variables."""

    # Data Paths Configuration
    # Base path for all data files (documents, processed data, vector database)
    _default_data_path = str(Path(__file__).parent.parent / "data")
    DATA_BASE_PATH: str = os.getenv("DATA_BASE_PATH", _default_data_path)

    @property
    def DATA_RAW_PATH(self) -> Path:
        """Path to raw documents for indexing."""
        return Path(self.DATA_BASE_PATH) / "raw"

    @property
    def DATA_PROCESSED_PATH(self) -> Path:
        """Path to processed/chunked data."""
        return Path(self.DATA_BASE_PATH) / "processed"

    @property
    def CHROMA_DB_PATH(self) -> Path:
        """Path to ChromaDB vector database."""
        return Path(self.DATA_BASE_PATH) / "chroma_db"

    # Document Indexing Configuration
    SUPPORTED_EXTENSIONS: List[str] = os.getenv(
        "SUPPORTED_EXTENSIONS",
        ".pdf,.xlsx,.xls,.md,.txt"
    ).split(",")

    # Ollama Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")  # Best for Spanish
    OLLAMA_FALLBACK_MODEL: str = os.getenv("OLLAMA_FALLBACK_MODEL", "mistral:7b-instruct")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # Increased for CPU inference

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:4200,http://localhost:3000"
    ).split(",")

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/teia_tutor.log")
    ENABLE_JSON_LOGS: bool = os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"


settings = Settings()
