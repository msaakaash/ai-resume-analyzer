
# ============================================================
# Resume RAG AI Application
# ============================================================
# Features:
# - Upload Resume PDF
# - Extract Resume Content
# - Create Embeddings using HuggingFace
# - Store embeddings in ChromaDB
# - Perform Retrieval-Augmented Generation (RAG)
# - Analyze Resume against Job Description
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import tempfile
import os
import shutil

# LangChain Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume RAG AI",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📄 Resume RAG AI")
st.markdown(
    """
    Analyze your resume against a Job Description using:
    
    - Retrieval-Augmented Generation (RAG)
    - LangChain
    - ChromaDB
    - HuggingFace Embeddings
    - Ollama LLM
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.markdown("""
    ### Tech Stack
    - LangChain
    - Ollama (Llama3)
    - ChromaDB
    - HuggingFace Embeddings
    - Streamlit
    """)

    st.info("Make sure Ollama is running locally.")

    st.code("ollama run llama3")


# ============================================================
# USER INPUT SECTION
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)


# ============================================================
# FUNCTION: LOAD PDF DOCUMENT
# ============================================================

def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# ============================================================
# FUNCTION: SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(documents)

    return split_docs


# ============================================================
# FUNCTION: CREATE EMBEDDINGS MODEL
# ============================================================

def load_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# FUNCTION: CREATE VECTOR DATABASE
# ============================================================

def create_vector_store(documents, embeddings):

    # Remove old database if exists
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore


# ============================================================
# FUNCTION: CREATE QA CHAIN
# ============================================================

def create_qa_chain(retriever):

    llm = Ollama(
        model="qwen2.5:1.5b"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


# ============================================================
# MAIN PROCESSING LOGIC
# ============================================================

if uploaded_file and job_description:

    try:

        with st.spinner("🔄 Processing Resume..."):

            # ====================================================
            # STEP 1: SAVE PDF TEMPORARILY
            # ====================================================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())

                temp_pdf_path = tmp_file.name


            # ====================================================
            # STEP 2: DATA INGESTION
            # ====================================================

            documents = load_pdf(temp_pdf_path)


            # ====================================================
            # STEP 3: TEXT CHUNKING
            # ====================================================

            split_docs = split_documents(documents)


            # ====================================================
            # STEP 4: EMBEDDING MODEL
            # ====================================================

            embeddings = load_embedding_model()


            # ====================================================
            # STEP 5: VECTOR DATABASE CREATION
            # ====================================================

            vectorstore = create_vector_store(
                split_docs,
                embeddings
            )


            # ====================================================
            # STEP 6: RETRIEVER SETUP
            # ====================================================

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )


            # ====================================================
            # STEP 7: LOAD LLM + QA CHAIN
            # ====================================================

            qa_chain = create_qa_chain(retriever)


            # ====================================================
            # STEP 8: PROMPT ENGINEERING
            # ====================================================

            query = f"""
            You are an expert ATS Resume Analyzer.

            Analyze the uploaded resume against the given Job Description.

            JOB DESCRIPTION:
            {job_description}

            Provide the following:

            1. ATS Match Percentage
            2. Missing Skills
            3. Technical Strengths
            4. Weak Areas
            5. Resume Improvement Suggestions
            6. Recommended Projects
            7. Possible Interview Questions
            8. Final Hiring Recommendation

            Give the response in professional formatting.
            """


            # ====================================================
            # STEP 9: GENERATE RESPONSE
            # ====================================================

            result = qa_chain.invoke({"query": query})

            response = result["result"]


            # ====================================================
            # STEP 10: DISPLAY OUTPUT
            # ====================================================

            st.success("✅ Analysis Complete!")

            st.subheader("📊 AI Resume Analysis")

            st.markdown(response)


            # ====================================================
            # STEP 11: CLEANUP
            # ====================================================

            os.remove(temp_pdf_path)


    except Exception as e:

        st.error(f"Error Occurred: {str(e)}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption("Built using Streamlit + LangChain + Ollama")
