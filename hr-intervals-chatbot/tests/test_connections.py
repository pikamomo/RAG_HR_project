import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🧪 Testing API Connections...\n")

# Test 1: OpenAI
print("1️⃣ Testing OpenAI...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="test"
    )
    print("   ✅ OpenAI connected successfully!")
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

print("\n🎉 All tests complete!")