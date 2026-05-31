"""Create and cache the Ollama LLM client."""

from __future__ import annotations

from langchain_community.llms import Ollama

from app.config.settings import AppSettings


class OllamaClient:
    """Provide the shared Ollama model instance."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._llm: Ollama | None = None

    def get_llm(self) -> Ollama:
        """Return a cached Ollama client."""

        if self._llm is None:
            self._llm = Ollama(model=self._settings.llm_model_name)
        return self._llm

