"""
Pydantic models for TEIA Tutor AI Service.
Includes fields for user tracking and future persistence/reporting.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_timestamp() -> str:
    return datetime.now().isoformat()


# Request Models

class QuestionRequest(BaseModel):
    """Request model for asking questions to the tutor."""
    question: str
    module: Optional[str] = None

    # User identification fields for tracking and reporting
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_role: Optional[str] = None  # e.g., "docente", "coordinador", "estudiante"
    course_id: Optional[str] = None  # ID of the course being designed


# Response Models

class QuestionResponse(BaseModel):
    """Response model for tutor answers with tracking metadata."""
    model_config = {"protected_namespaces": ()}

    # Tracking fields
    request_id: str = Field(default_factory=generate_uuid)
    timestamp: str = Field(default_factory=get_timestamp)

    # Echo back user context for correlation
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    module: Optional[str] = None

    # Response content
    answer: str
    confidence: float
    sources: List[str]
    suggested_actions: Optional[List[str]] = None

    # Metadata for reporting
    model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None


class ModuleInfo(BaseModel):
    """Information about a course module."""
    id: str
    name: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    ollama_connected: bool


class SystemStatus(BaseModel):
    """Detailed system status response."""
    model_config = {"protected_namespaces": ()}

    service: str
    status: str
    ollama_connected: bool
    ollama_models: List[str]
    model_used: str
    timestamp: str


# Document Management Models

class DocumentInfo(BaseModel):
    """Information about a document available for indexing."""
    filename: str
    path: str
    size_bytes: int
    extension: str
    modified_at: str


class DocumentListResponse(BaseModel):
    """Response for listing available documents."""
    documents: List[DocumentInfo]
    total_count: int
    raw_path: str


class UploadedFileInfo(BaseModel):
    """Information about a successfully uploaded file."""
    filename: str
    size_bytes: int


class FailedFileInfo(BaseModel):
    """Information about a failed file upload."""
    filename: str
    error: str


class DocumentUploadResponse(BaseModel):
    """Response for document upload endpoint."""
    uploaded_files: List[UploadedFileInfo]
    failed_files: List[FailedFileInfo]
    upload_path: str


# Indexing Task Models

class IndexingTaskResponse(BaseModel):
    """Response for indexing task status."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    message: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    documents_processed: int = 0
    chunks_created: int = 0
    error: Optional[str] = None


class IndexingStartResponse(BaseModel):
    """Response when starting a new indexing task."""
    task_id: str
    status: str
    message: str
