"""Create and cache the embedding model used by the pipeline."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import AppSettings


class EmbeddingService:
    """Provide the shared embedding model instance."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._embeddings: HuggingFaceEmbeddings | None = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Return a cached HuggingFace embeddings model."""

        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._settings.embedding_model_name
            )
        return self._embeddings

