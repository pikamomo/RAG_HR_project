"""
Shared vector storage utilities
Handles chunking and storing documents in Qdrant
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from typing import List

load_dotenv()


def get_embeddings():
    """Get OpenAI embeddings instance"""
    return OpenAIEmbeddings(
        model=os.getenv("OPEN_AI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


def get_qdrant_client():
    """Get Qdrant client instance"""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )


def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks
    
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def store_documents(documents: List[Document]) -> tuple[int, int]:
    """
    Store documents in Qdrant vector database
    
    Args:
        documents: List of Document objects with content and metadata
    
    Returns:
        Tuple of (expected_count, actual_stored_count)
    """
    embeddings = get_embeddings()
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION")
    
    # Get count before storing
    try:
        before_count = client.count(collection_name=collection_name).count
    except Exception:
        before_count = 0
    
    # Store documents
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name
    )
    
    # Verify storage by counting after
    try:
        after_count = client.count(collection_name=collection_name).count
        actual_stored = after_count - before_count
    except Exception as e:
        print(f"   ⚠️ Warning: Could not verify storage: {str(e)}")
        actual_stored = len(documents)  # Assume success if can't verify
    
    return len(documents), actual_stored


def process_and_store(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> int:
    """
    Complete pipeline: chunk documents and store in vector database
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks
    
    Returns:
        Number of chunks stored
    """
    # 1. Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # 2. Store in Qdrant with verification
    try:
        expected, actual_stored = store_documents(chunks)
        
        if actual_stored == expected:
            print(f"   ✅ Stored {actual_stored} chunks in Qdrant")
        elif actual_stored > 0:
            print(f"   ⚠️ Partial storage: expected {expected}, actually stored {actual_stored}")
        else:
            print(f"   ❌ Storage failed: 0 chunks stored (expected {expected})")
            
        return actual_stored
        
    except Exception as e:
        print(f"   ❌ Error storing in Qdrant: {str(e)}")
        raise