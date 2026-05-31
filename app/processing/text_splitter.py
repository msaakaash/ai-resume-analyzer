"""Split resume documents into chunks for retrieval."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import AppSettings


class TextSplitterService:
    """Split documents using the application's chunking strategy."""

    def __init__(self, settings: AppSettings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split the provided documents into retrievable chunks."""

        return self._splitter.split_documents(documents)

