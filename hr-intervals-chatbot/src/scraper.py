"""
Web scraping module
Scrapes web pages using Firecrawl and stores in Qdrant
"""

import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_core.documents import Document
from datetime import datetime
from src.vector_store import process_and_store

load_dotenv()


def scrape_url(url: str) -> str:
    """
    Scrape webpage content using Firecrawl
    
    Args:
        url: URL to scrape
        
    Returns:
        Markdown content of the webpage
    """
    print(f"🌐 Scraping: {url}")
    
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    result = app.scrape_url(url, params={'formats': ['markdown']})
    
    if not result.get('markdown'):
        raise ValueError("Failed to scrape - no content retrieved")
    
    return result['markdown']


def process_and_store_webpage(url: str) -> int:
    """
    Scrape webpage and store in vector database
    
    Args:
        url: URL to scrape
        
    Returns:
        Number of chunks created
    """
    
    # 1. Scrape content
    markdown_content = scrape_url(url)
    print(f"   ✅ Scraped {len(markdown_content)} characters")
    
    # 2. Create document with metadata
    doc = Document(
        page_content=markdown_content,
        metadata={
            "source": url,
            "type": "webpage",
            "upload_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    
    # 3. Chunk and store (using shared function)
    num_chunks = process_and_store([doc])
    
    return num_chunks


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