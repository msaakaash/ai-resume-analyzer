"""Central application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root."""

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
"""Default persistence directory for the Chroma vector store."""


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Static configuration values used throughout the application."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "qwen2.5:1.5b"
    chunk_size: int = 700
    chunk_overlap: int = 100
    retriever_k: int = 4
    chroma_persist_directory: Path = CHROMA_DB_DIR


def get_settings() -> AppSettings:
    """Return the default application settings."""

    return AppSettings()

