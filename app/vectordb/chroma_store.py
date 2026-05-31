"""Create and refresh the local Chroma vector store."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class ChromaStore:
    """Build a fresh Chroma store from resume chunks."""

    def __init__(self, persist_directory: Path) -> None:
        self._persist_directory = persist_directory

    def create_store(self, documents: list[Document], embeddings: Any) -> Chroma:
        """Rebuild the persistent Chroma store from scratch."""

        if self._persist_directory.exists():
            shutil.rmtree(self._persist_directory)

        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(self._persist_directory),
        )

