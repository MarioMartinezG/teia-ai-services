import logging
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

class CustomFormatter(logging.Formatter):
    """Formateador personalizado para logs con colores (opcional)"""

    # Códigos de colores para terminal
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Formatos para diferentes niveles
    formats = {
        logging.DEBUG: f"{blue}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.INFO: f"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        logging.WARNING: f"{yellow}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.ERROR: f"{red}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.CRITICAL: f"{bold_red}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}"
    }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno, self.formats[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """Formateador para logs en formato JSON (útil para producción)"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Agregar campos extras si existen
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        # Agregar información de excepciones
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging(
        log_level: str = "INFO",
        log_file: str = None,
        enable_console: bool = True,
        enable_json: bool = False
) -> None:
    """
    Configura el sistema de logging para la aplicación.

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta del archivo de log (opcional)
        enable_console: Habilitar logs en consola
        enable_json: Usar formato JSON para logs
    """
    # Crear directorio de logs si no existe
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configurar logger principal
    logger = logging.getLogger("teia_tutor")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Evitar propagación al root logger
    logger.propagate = False

    # Limpiar handlers existentes
    logger.handlers.clear()

    # Handler para consola
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        if enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(CustomFormatter())

        logger.addHandler(console_handler)

    # Handler para archivo (si se especifica)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(JSONFormatter())  # Siempre JSON en archivos
        logger.addHandler(file_handler)

    # Configurar loggers de terceras librerías
    _configure_third_party_loggers(log_level)

def _configure_third_party_loggers(log_level: str) -> None:
    """Configura loggers de librerías externas para reducir ruido"""
    third_party_loggers = [
        "uvicorn",
        "fastapi",
        "httpx",
        "httpcore",
        "openai",
        "langchain",
        "chromadb"
    ]

    for logger_name in third_party_loggers:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(logging.WARNING)  # Reducir verbosidad
        lib_logger.propagate = False

def get_logger(name: str = None) -> logging.Logger:
    """
    Obtiene un logger con el nombre especificado.

    Args:
        name: Nombre del logger (si es None, retorna el logger principal)

    Returns:
        Logger configurado
    """
    if name is None:
        return logging.getLogger("teia_tutor")
    return logging.getLogger(f"teia_tutor.{name}")

def log_execution_time(logger: logging.Logger):
    """
    Decorador para medir y loggear tiempo de ejecución de funciones.

    Usage:
        @log_execution_time(logger)
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"🚀 Iniciando ejecución: {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ {func.__name__} completado en {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ {func.__name__} falló después de {execution_time:.2f}s: {str(e)}")
                raise

        return wrapper
    return decorator

class RequestLogger:
    """Logger especializado para requests HTTP"""

    def __init__(self, logger_name: str = "http"):
        self.logger = get_logger(logger_name)

    def log_request(self, method: str, url: str, extra_data: Dict[str, Any] = None):
        """Loggear una request HTTP"""
        log_data = {
            "method": method,
            "url": url,
            "type": "request_outgoing"
        }

        if extra_data:
            log_data.update(extra_data)

        self.logger.info(f"🌐 HTTP {method} {url}", extra={"extra_data": log_data})

    def log_response(self, method: str, url: str, status_code: int, response_time: float, extra_data: Dict[str, Any] = None):
        """Loggear una response HTTP"""
        log_data = {
            "method": method,
            "url": url,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "type": "response_incoming"
        }

        if extra_data:
            log_data.update(extra_data)

        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(level, f"📡 HTTP {method} {url} → {status_code} ({response_time:.2f}s)",
                        extra={"extra_data": log_data})

def init_logging():
    """Initialize logging with settings from config."""
    from config import settings
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file=settings.LOG_FILE,
        enable_console=True,
        enable_json=settings.ENABLE_JSON_LOGS
    )


# Initialize logging when module is imported
init_logging()