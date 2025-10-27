import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from datetime import datetime

load_dotenv()

def create_vectorstore():
    """Create or connect to vector store"""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPEN_AI_EMBEDDING_MODEL")
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION"),
        embedding=embeddings
    )
    
    return vectorstore, embeddings, client

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

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def add_metadata(chunks, source_name, doc_type="document"):
    """Add metadata to chunks"""
    for chunk in chunks:
        chunk.metadata["source"] = source_name
        chunk.metadata["type"] = doc_type
        chunk.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")
    
    return chunks

def ingest_document(file_path, doc_type="document"):
    """Complete document ingestion pipeline"""
    print(f"📄 Processing: {file_path}")
    
    # 1. Load document
    documents = load_document(file_path)
    print(f"   ✅ Loaded {len(documents)} pages")
    
    # 2. Chunk
    chunks = chunk_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # 3. Add metadata
    source_name = os.path.basename(file_path)
    chunks = add_metadata(chunks, source_name, doc_type)
    
    # 4. Store in Qdrant
    _, embeddings, _ = create_vectorstore()
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION")
    )
    
    print(f"   ✅ Uploaded to Qdrant")
    return len(chunks)

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