# src/scraper.py
import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.schema import Document
from datetime import datetime
from src.vector_store import process_and_store  # ← 使用共享函数

load_dotenv()

def scrape_url(url):
    """Scrape webpage content using Firecrawl"""
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
    
    # 2. Create document with metadata
    doc = Document(
        page_content=markdown_content,
        metadata={
            "source": url,
            "type": "webpage",
            "upload_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    
    # 3. Chunk and store (共享函数)
    num_chunks = process_and_store([doc])
    
    return num_chunks