"""
Middleware configuration for TEIA Tutor AI Service.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings


def setup_middlewares(app: FastAPI) -> None:
    """Configure all middlewares for the application."""

    # CORS middleware - allows requests from Angular frontend / Java API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
