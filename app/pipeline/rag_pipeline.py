"""Reusable resume analysis pipeline."""

from __future__ import annotations

from typing import Any

from langchain.chains import RetrievalQA

from app.config.settings import AppSettings, get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.pdf_loader import PDFLoader
from app.llm.ollama_client import OllamaClient
from app.processing.text_splitter import TextSplitterService
from app.prompts.resume_analysis_prompt import build_resume_analysis_query
from app.retrieval.retriever import RetrieverService
from app.vectordb.chroma_store import ChromaStore


class ResumeRAGPipeline:
    """Orchestrate PDF ingestion, retrieval, and LLM-based analysis."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        pdf_loader: PDFLoader | None = None,
        text_splitter: TextSplitterService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaStore | None = None,
        retriever_service: RetrieverService | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pdf_loader = pdf_loader or PDFLoader()
        self._text_splitter = text_splitter or TextSplitterService(self._settings)
        self._embedding_service = embedding_service or EmbeddingService(self._settings)
        self._vector_store = vector_store or ChromaStore(
            self._settings.chroma_persist_directory
        )
        self._retriever_service = retriever_service or RetrieverService(
            self._settings.retriever_k
        )
        self._llm_client = llm_client or OllamaClient(self._settings)

    def analyze_resume(self, pdf_path: str, job_description: str) -> str:
        """Run the full RAG workflow and return the final analysis."""

        documents = self._pdf_loader.load(pdf_path)
        split_documents = self._text_splitter.split(documents)
        embeddings = self._embedding_service.get_embeddings()
        vector_store = self._vector_store.create_store(split_documents, embeddings)
        retriever = self._retriever_service.create_retriever(vector_store)
        qa_chain = self._build_qa_chain(retriever)
        query = build_resume_analysis_query(job_description)
        result: dict[str, Any] = qa_chain.invoke({"query": query})
        return str(result["result"])

    def _build_qa_chain(self, retriever: Any) -> RetrievalQA:
        """Create the RetrievalQA chain used for analysis."""

        return RetrievalQA.from_chain_type(
            llm=self._llm_client.get_llm(),
            retriever=retriever,
            return_source_documents=True,
        )
