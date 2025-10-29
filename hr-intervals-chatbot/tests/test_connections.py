#!/usr/bin/env python3
"""
Test all API connections with 2025 October versions
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("🧪 Testing API Connections (October 2025)...\n")

# Test 1: OpenAI
print("1️⃣ Testing OpenAI...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test embeddings
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="test"
    )
    print("   ✅ OpenAI connected successfully!")
    print(f"   ✅ Embeddings working (dimension: {len(response.data[0].embedding)})")
except Exception as e:
    print(f"   ❌ OpenAI error: {str(e)}")

# Test 2: Qdrant
print("\n2️⃣ Testing Qdrant...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    collections = client.get_collections()
    print(f"   ✅ Qdrant connected! Collections: {len(collections.collections)}")
except Exception as e:
    print(f"   ❌ Qdrant error: {str(e)}")

# Test 3: Firecrawl
print("\n3️⃣ Testing Firecrawl...")
try:
    from firecrawl import FirecrawlApp
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    print("   ✅ Firecrawl initialized successfully!")
except Exception as e:
    print(f"   ❌ Firecrawl error: {str(e)}")

# Test 4: LangChain imports
print("\n4️⃣ Testing LangChain 1.0 imports...")
try:
    import langchain
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    
    print(f"   ✅ LangChain version: {langchain.__version__}")
    print("   ✅ All LangChain 1.0 imports successful!")
except Exception as e:
    print(f"   ❌ LangChain import error: {str(e)}")

# Test 5: Gradio
print("\n5️⃣ Testing Gradio...")
try:
    import gradio as gr
    print(f"   ✅ Gradio version: {gr.__version__}")
except Exception as e:
    print(f"   ❌ Gradio error: {str(e)}")

print("\n" + "="*50)
print("🎉 Connection tests complete!")
print("\nNext steps:")
print("1. Upload a test document: python src/ingestion.py")
print("2. Test the chatbot: python src/chatbot.py")
print("3. Start the user interface: python app.py")
print("4. Start the admin interface: python admin.py")