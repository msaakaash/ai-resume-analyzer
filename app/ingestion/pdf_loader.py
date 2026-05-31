"""Load resume PDFs into LangChain documents."""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """Load PDF files from disk."""

    def load(self, pdf_path: str) -> list[Document]:
        """Load a PDF file and return its pages as documents."""

        loader = PyPDFLoader(pdf_path)
        return loader.load()

