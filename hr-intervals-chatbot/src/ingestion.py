"""
Document ingestion module
Loads PDF/DOCX files and stores them in Qdrant
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from datetime import datetime
from src.vector_store import process_and_store

load_dotenv()


def load_document(file_path: str):
    """
    Load PDF or DOCX document
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Only PDF and DOCX files are supported")
    
    documents = loader.load()
    return documents


def add_metadata(documents, source_name: str, doc_type: str = "document"):
    """
    Add metadata to documents
    
    Args:
        documents: List of Document objects
        source_name: Source filename
        doc_type: Type of document (document, policy, guide, etc.)
        
    Returns:
        Documents with added metadata
    """
    for doc in documents:
        doc.metadata["source"] = source_name
        doc.metadata["type"] = doc_type
        doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")
    
    return documents


def ingest_document(file_path: str, doc_type: str = "document") -> int:
    """
    Complete document ingestion pipeline
    
    Args:
        file_path: Path to the document file
        doc_type: Type of document
        
    Returns:
        Number of chunks created
    """
    print(f"📄 Processing: {file_path}")
    
    # 1. Load document
    documents = load_document(file_path)
    print(f"   ✅ Loaded {len(documents)} pages")
    
    # 2. Add metadata
    source_name = os.path.basename(file_path)
    documents = add_metadata(documents, source_name, doc_type)
    
    # 3. Chunk and store (using shared function)
    num_chunks = process_and_store(documents)
    
    return num_chunks


# Test function
if __name__ == "__main__":
    print("🧪 Testing document ingestion...")
    print("\nPlease place a test PDF or DOCX file in data/documents/")
    print("Then update the file path below and run again.\n")
    
    # Example:
    # test_file = "data/documents/test.pdf"
    # if os.path.exists(test_file):
    #     num_chunks = ingest_document(test_file)
    #     print(f"\n🎉 Success! Processed {num_chunks} chunks")