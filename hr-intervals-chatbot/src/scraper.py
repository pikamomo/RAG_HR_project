import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from datetime import datetime

load_dotenv()

def scrape_url(url):
    """Scrape webpage content"""
    print(f"🌐 Scraping: {url}")
    
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    result = app.scrape_url(url, params={'formats': ['markdown']})
    
    if not result.get('markdown'):
        raise ValueError("Failed to scrape - no content retrieved")
    
    return result['markdown']

def process_and_store_webpage(url):
    """Scrape webpage and store in vector database"""
    
    # 1. Scrape content
    markdown_content = scrape_url(url)
    print(f"   ✅ Scraped {len(markdown_content)} characters")
    
    # 2. Create document
    doc = Document(
        page_content=markdown_content,
        metadata={
            "source": url,
            "type": "webpage",
            "upload_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    
    # 3. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents([doc])
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # 4. Store
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPEN_AI_EMBEDDING_MODEL")
    )
    
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
    print("🧪 Testing web scraper...")
    
    # Test with a simple webpage
    test_url = "https://example.com"
    
    try:
        num_chunks = process_and_store_webpage(test_url)
        print(f"\n🎉 Success! Processed {num_chunks} chunks")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")