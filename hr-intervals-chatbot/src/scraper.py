"""
Web scraping module
Scrapes web pages using Firecrawl and stores in Qdrant
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_core.documents import Document
from datetime import datetime
from qdrant_client import QdrantClient

# Add parent directory to path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.vector_store import process_and_store

load_dotenv()


def check_url_exists(url: str) -> int:
    """
    Check if URL already exists in Qdrant
    
    Args:
        url: URL to check
        
    Returns:
        Number of existing chunks for this URL (0 if not found)
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    collection_name = os.getenv("QDRANT_COLLECTION")
    
    try:
        result = client.scroll(
            collection_name=collection_name,
            limit=1,
            scroll_filter={
                "must": [{"key": "metadata.source", "match": {"value": url}}]
            },
            with_payload=False
        )
        
        # Count total chunks for this URL
        count_result = client.count(
            collection_name=collection_name,
            count_filter={
                "must": [{"key": "metadata.source", "match": {"value": url}}]
            }
        )
        return count_result.count
    except Exception:
        return 0


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
    result = app.scrape(url, formats=['markdown'])
    
    # Handle different return types
    if hasattr(result, 'markdown'):
        markdown_content = result.markdown
    elif isinstance(result, dict) and 'markdown' in result:
        markdown_content = result['markdown']
    else:
        raise ValueError(f"Failed to scrape - unexpected result type: {type(result)}")
    
    if not markdown_content:
        raise ValueError("Failed to scrape - no content retrieved")
    
    return markdown_content


def process_and_store_webpage(url: str, force: bool = False) -> int:
    """
    Scrape webpage and store in vector database
    
    Args:
        url: URL to scrape
        force: If True, skip duplicate check and store anyway
        
    Returns:
        Number of chunks created
        
    Raises:
        ValueError: If URL already exists and force=False
    """
    
    # 0. Check if URL already exists
    if not force:
        existing_chunks = check_url_exists(url)
        if existing_chunks > 0:
            raise ValueError(
                f"URL already exists with {existing_chunks} chunks. "
                f"Use 'Delete' to remove it first, or force=True to add anyway."
            )
    
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
    test_url = "https://hrintervals.ca/resources/sample-policy-inclusive-and-equitable-hiring-practices/"
    
    try:
        num_chunks = process_and_store_webpage(test_url)
        print(f"\n🎉 Success! Processed {num_chunks} chunks")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")