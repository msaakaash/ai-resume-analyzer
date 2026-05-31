"""Streamlit presentation layer for the resume RAG analyzer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.config.settings import get_settings
from app.pipeline.rag_pipeline import ResumeRAGPipeline


@st.cache_resource
def get_pipeline() -> ResumeRAGPipeline:
    """Create the reusable RAG pipeline once per Streamlit session."""

    return ResumeRAGPipeline()


def _save_uploaded_pdf(uploaded_file: UploadedFile) -> Path:
    """Persist the uploaded PDF to a temporary file."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return Path(temp_file.name)


def render_header() -> None:
    """Render the page header and product description."""

    st.title("Resume RAG AI")
    st.markdown(
        """
        Analyze your resume against a job description using:

        - Retrieval-Augmented Generation (RAG)
        - LangChain
        - ChromaDB
        - HuggingFace Embeddings
        - Ollama LLM
        """
    )


def render_sidebar() -> None:
    """Render the sidebar configuration panel."""

    settings = get_settings()

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            """
            ### Tech Stack
            - LangChain
            - Ollama (Qwen2.5 1.5B)
            - ChromaDB
            - HuggingFace Embeddings
            - Streamlit
            """
        )
        st.info("Make sure Ollama is running locally.")
        st.code(f"ollama run {settings.llm_model_name}")


def render_analysis(uploaded_file: UploadedFile, job_description: str) -> None:
    """Run the pipeline and render the analysis result."""

    temp_pdf_path = _save_uploaded_pdf(uploaded_file)
    try:
        analysis = get_pipeline().analyze_resume(str(temp_pdf_path), job_description)
    finally:
        temp_pdf_path.unlink(missing_ok=True)

    st.success("Analysis complete!")
    st.subheader("AI Resume Analysis")
    st.markdown(analysis)


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="Resume RAG AI", page_icon="R", layout="wide")
    render_header()
    render_sidebar()

    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    job_description = st.text_area("Paste Job Description", height=250)

    if uploaded_file and job_description:
        try:
            with st.spinner("Processing resume..."):
                render_analysis(uploaded_file, job_description)
        except Exception as error:
            st.error(f"Error occurred: {error}")

    st.markdown("---")
    st.caption("Built using Streamlit + LangChain + Ollama")


if __name__ == "__main__":
    main()
