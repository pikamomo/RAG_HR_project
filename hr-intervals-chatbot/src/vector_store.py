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


def store_documents(documents: List[Document]) -> int:
    """
    Store documents in Qdrant vector database
    
    Args:
        documents: List of Document objects with content and metadata
    
    Returns:
        Number of chunks stored
    """
    embeddings = get_embeddings()
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION")
    )
    
    return len(documents)


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
    
    # 2. Store in Qdrant
    num_stored = store_documents(chunks)
    print(f"   ✅ Stored in Qdrant")
    
    return num_stored