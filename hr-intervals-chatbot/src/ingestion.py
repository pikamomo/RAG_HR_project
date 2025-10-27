# src/ingestion.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from datetime import datetime
from src.vector_store import process_and_store  # ← 使用共享函数

load_dotenv()

def load_document(file_path):
    """Load PDF or DOCX document"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Only PDF and DOCX files are supported")
    
    documents = loader.load()
    return documents

def add_metadata(documents, source_name, doc_type="document"):
    """Add metadata to documents"""
    for doc in documents:
        doc.metadata["source"] = source_name
        doc.metadata["type"] = doc_type
        doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")
    
    return documents

def ingest_document(file_path, doc_type="document"):
    """Complete document ingestion pipeline"""
    print(f"📄 Processing: {file_path}")
    
    # 1. Load document
    documents = load_document(file_path)
    print(f"   ✅ Loaded {len(documents)} pages")
    
    # 2. Add metadata
    source_name = os.path.basename(file_path)
    documents = add_metadata(documents, source_name, doc_type)
    
    # 3. Chunk and store (共享函数)
    num_chunks = process_and_store(documents)
    
    return num_chunks