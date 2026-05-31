"""Create retrievers from the vector store."""

from __future__ import annotations

from typing import Any


class RetrieverService:
    """Build retrievers with the application's search settings."""

    def __init__(self, search_k: int) -> None:
        self._search_k = search_k

    def create_retriever(self, vector_store: Any) -> Any:
        """Return a retriever configured for the current search depth."""

        return vector_store.as_retriever(search_kwargs={"k": self._search_k})

