# HR Intervals AI Assistant - Architecture Documentation

## Project Overview

An AI-powered bilingual chatbot for nonprofit organizations providing HR support, policy generation, and compliance checking.

**Tech Stack:**
- Backend: Python 3.12 + LangChain
- Vector Database: Qdrant Cloud
- AI Models: OpenAI (GPT-4o-mini, text-embedding-3-large)
- UI Framework: Gradio
- Web Scraping: Firecrawl
- Monitoring: LangSmith (optional)
- Deployment: Hugging Face Spaces

---

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                            │
├──────────────────────────┬──────────────────────────────────┤
│   app.py                 │   admin.py                        │
│   (Chat Interface)       │   (Admin Interface)               │
│   - User Q&A             │   - Upload documents              │
│   - Policy generation    │   - Scrape web pages              │
│   - View sources         │   - Manage content                │
└──────────────────────────┴──────────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
├──────────────┬─────────────────┬────────────────────────────┤
│ chatbot.py   │ ingestion.py    │ scraper.py                 │
│ - RAG chain  │ - PDF/DOCX      │ - Web scraping             │
│ - Retrieval  │ - Text chunking │ - URL processing           │
│ - QA logic   │ - Metadata      │ - Content storage          │
└──────────────┴─────────────────┴────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                          │
├─────────────┬─────────────┬───────────────┬─────────────────┤
│ Qdrant      │ OpenAI      │ Firecrawl     │ LangSmith       │
│ Cloud       │ API         │ API           │ (optional)      │
│ - Vectors   │ - Embeddings│ - Scraping    │ - Monitoring    │
│ - Search    │ - Chat      │ - Markdown    │ - Debugging     │
└─────────────┴─────────────┴───────────────┴─────────────────┘
```

---

## Module Relationships

### Core Modules

#### 1. `src/ingestion.py` - Document Processing Module

**Purpose:** Load, process, and store PDF/DOCX documents into vector database

**Key Functions:**
```python
create_vectorstore() -> (vectorstore, embeddings, client)
load_document(file_path: str) -> List[Document]
chunk_documents(documents, chunk_size=1000, chunk_overlap=200) -> List[Document]
add_metadata(chunks, source_name, doc_type="document") -> List[Document]
ingest_document(file_path: str, doc_type="document") -> int
```

**Dependencies:**
- `langchain_community.document_loaders` (PyPDFLoader, Docx2txtLoader)
- `langchain.text_splitter` (RecursiveCharacterTextSplitter)
- `langchain_openai` (OpenAIEmbeddings)
- `langchain_qdrant` (QdrantVectorStore)
- `qdrant_client` (QdrantClient)

**Used By:**
- `admin.py` (upload functionality)

---

#### 2. `src/scraper.py` - Web Scraping Module

**Purpose:** Scrape web pages and store content in vector database

**Key Functions:**
```python
scrape_url(url: str) -> str
process_and_store_webpage(url: str) -> int
```

**Dependencies:**
- `firecrawl` (FirecrawlApp)
- `langchain.schema` (Document)
- `langchain.text_splitter` (RecursiveCharacterTextSplitter)
- `langchain_openai` (OpenAIEmbeddings)
- `langchain_qdrant` (QdrantVectorStore)

**Used By:**
- `admin.py` (URL scraping functionality)

---

#### 3. `src/chatbot.py` - RAG Question-Answering Module

**Purpose:** Handle user questions using Retrieval-Augmented Generation

**Key Functions:**
```python
create_rag_chain() -> ConversationalRetrievalChain
ask_question(qa_chain, question: str) -> (answer: str, sources: List[Document])
```

**Components:**
- Vector store retriever (k=5 similar documents)
- LLM: GPT-4o-mini (temperature=0.3)
- Conversation memory (ConversationBufferMemory)
- System prompt with disclaimers

**Dependencies:**
- `langchain_openai` (ChatOpenAI, OpenAIEmbeddings)
- `langchain_qdrant` (QdrantVectorStore)
- `langchain.chains` (ConversationalRetrievalChain)
- `langchain.memory` (ConversationBufferMemory)
- `qdrant_client` (QdrantClient)

**Used By:**
- `app.py` (chat interface)

---

### User Interface Modules

#### 4. `app.py` - Chat Interface (End Users)

**Purpose:** Gradio-based chat interface for nonprofit users

**Features:**
- Real-time Q&A
- PII detection and warnings
- Source citations
- Disclaimer display
- Conversation history
- Example questions

**Calls:**
- `src/chatbot.py` → `create_rag_chain()`, `ask_question()`

**Port:** 7860

---

#### 5. `admin.py` - Admin Interface (Content Managers)

**Purpose:** Gradio-based management interface for HR Intervals team

**Features:**
- View all documents
- Upload PDF/DOCX files
- Scrape single/multiple URLs
- Delete documents by source
- Update/replace documents

**Calls:**
- `src/ingestion.py` → `ingest_document()`
- `src/scraper.py` → `process_and_store_webpage()`
- `qdrant_client.QdrantClient` → direct CRUD operations

**Port:** 7861

---

## Data Flow Diagrams

### Flow 1: Document Upload
```
User (admin.py)
    ↓
    [Select PDF/DOCX file]
    ↓
admin.py: upload_document()
    ↓
ingestion.py: ingest_document()
    ↓
    [Load document] → PyPDFLoader / Docx2txtLoader
    ↓
    [Split into chunks] → RecursiveCharacterTextSplitter
    │   - chunk_size: 1000
    │   - chunk_overlap: 200
    ↓
    [Add metadata]
    │   - source: filename
    │   - type: document/policy/guide
    │   - upload_date: YYYY-MM-DD
    ↓
    [Generate embeddings] → OpenAI text-embedding-3-large
    ↓
    [Store vectors + metadata] → Qdrant Cloud
    ↓
✅ Success: N chunks uploaded
```

---

### Flow 2: Web Scraping
```
User (admin.py)
    ↓
    [Enter URL(s)]
    ↓
admin.py: scrape_single_url() / scrape_multiple_urls()
    ↓
scraper.py: process_and_store_webpage()
    ↓
    [Scrape webpage] → Firecrawl API
    │   - Returns: Markdown content
    ↓
    [Create document with metadata]
    │   - source: URL
    │   - type: webpage
    │   - upload_date: YYYY-MM-DD
    ↓
    [Split into chunks] → RecursiveCharacterTextSplitter
    ↓
    [Generate embeddings] → OpenAI text-embedding-3-large
    ↓
    [Store vectors + metadata] → Qdrant Cloud
    ↓
✅ Success: N chunks uploaded
```

---

### Flow 3: Question Answering (RAG)
```
User (app.py)
    ↓
    [Type question]
    ↓
app.py: chat()
    ↓
    [Check for PII] → Regex patterns
    │   - Capitalized names: [A-Z][a-z]+ [A-Z][a-z]+
    │   - If detected: Show warning
    ↓
chatbot.py: ask_question()
    ↓
ConversationalRetrievalChain
    ↓
    [Convert question to embedding] → OpenAI text-embedding-3-large
    ↓
    [Similarity search] → Qdrant Cloud
    │   - Retrieve top 5 similar chunks
    │   - Return: chunks + metadata
    ↓
    [Combine context + question + chat history]
    ↓
    [Generate answer] → OpenAI GPT-4o-mini
    │   - Temperature: 0.3
    │   - System prompt: HR assistant with disclaimers
    ↓
    [Return answer + source documents]
    ↓
app.py: Display answer with sources
    ↓
User sees:
    - Answer
    - ⚠️ PII warning (if applicable)
    - 📚 Sources (top 3)
```

---

### Flow 4: Document Deletion
```
User (admin.py)
    ↓
    [Enter document name or URL]
    ↓
admin.py: delete_document()
    ↓
Qdrant Client: delete()
    ↓
    [Filter by metadata]
    │   - Field: "source"
    │   - Match: exact document name
    ↓
    [Delete all matching points]
    ↓
✅ Success: All chunks from source deleted
```

---

### Flow 5: Document Update
```
User (admin.py)
    ↓
    [Specify old document name]
    [Select new file]
    ↓
admin.py: update_document()
    ↓
    [Step 1: Delete old document]
    │   └─→ delete_document(old_source)
    ↓
    [Step 2: Upload new document]
    │   └─→ upload_document(new_file)
    ↓
✅ Success: Document replaced
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# OpenAI API
OPENAI_API_KEY=sk-proj-...
OPEN_AI_EMBEDDING_MODEL=text-embedding-3-large
OPEN_AI_CHAT_MODEL=gpt-4o-mini

# Qdrant Cloud
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=xxx
QDRANT_COLLECTION=hr-intervals

# Firecrawl
FIRECRAWL_API_KEY=fc-xxx

# LangSmith (Optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=xxx
LANGSMITH_PROJECT=hr-intervals-chatbot
```

---

## Project Structure
```
hr-intervals-chatbot/
├── src/
│   ├── __init__.py
│   ├── ingestion.py          # Document processing
│   ├── chatbot.py             # RAG Q&A logic
│   └── scraper.py             # Web scraping
├── data/
│   ├── documents/             # Uploaded files
│   └── scraped/               # Scraped content (cache)
├── app.py                     # User chat interface
├── admin.py                   # Admin management interface
├── .env                       # API keys and config
├── requirements.txt           # Python dependencies
├── ARCHITECTURE.md            # This file
└── README.md                  # Project overview
```

---

## Key Technical Decisions

### 1. Vector Database: Qdrant Cloud
- **Why:** Built-in web UI, easy document management, free tier
- **Alternative considered:** Pinecone (limited free tier, no document-level UI)

### 2. Embedding Model: text-embedding-3-large
- **Dimensions:** 3072 (can be reduced to 1024 for cost)
- **Why:** Best quality, multilingual support (English/French)

### 3. LLM: GPT-4o-mini
- **Why:** Cost-effective, sufficient for HR Q&A, fast response
- **Alternative:** GPT-4o (more expensive but higher quality)

### 4. Chunking Strategy
- **Chunk size:** 1000 characters
- **Overlap:** 200 characters
- **Separators:** `["\n\n", "\n", ". ", " ", ""]`
- **Why:** Balances context preservation and retrieval accuracy

### 5. Retrieval: Top-k similarity search
- **k=5:** Retrieve 5 most similar chunks
- **Distance metric:** Cosine similarity
- **Why:** Good balance between context and noise

---

## Metadata Schema

Every chunk stored in Qdrant has the following metadata:
```python
{
    "source": str,           # Filename or URL
    "type": str,             # "document" | "webpage" | "policy" | "guide"
    "upload_date": str,      # "YYYY-MM-DD"
    "page": int,             # (optional) Page number for PDFs
    "valid_until": str,      # (optional) Expiry date for policies
    "version": str,          # (optional) Version number
}
```

---

## Document Management Operations

### View Documents
```python
# List all unique documents
client.scroll(collection_name, limit=1000, with_payload=True)
# Group by 'source' field
```

### Upload Document
```python
# 1. Load: PyPDFLoader / Docx2txtLoader
# 2. Chunk: RecursiveCharacterTextSplitter
# 3. Add metadata: source, type, date
# 4. Embed: OpenAI text-embedding-3-large
# 5. Store: QdrantVectorStore.from_documents()
```

### Delete Document
```python
client.delete(
    collection_name=collection_name,
    points_selector=FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value="filename.pdf")
                )
            ]
        )
    )
)
```

### Update Document
```python
# 1. Delete old version (by source name)
# 2. Upload new version
```

---

## Security Features

### PII Detection
- Regex pattern for names: `\b[A-Z][a-z]+ [A-Z][a-z]+\b`
- Warning displayed to user if detected
- Future: Integrate Microsoft Presidio for advanced PII detection

### Disclaimers
- Shown on first interaction
- Embedded in system prompt
- Reminds users to consult professionals

### API Key Security
- Stored in `.env` file (not in version control)
- `.env` added to `.gitignore`

---

## Performance Considerations

### Embedding Cost
- Model: text-embedding-3-large
- Cost: ~$0.13 per 1M tokens
- Typical document: 10 pages ≈ 5,000 tokens ≈ $0.0007

### Chat Cost
- Model: GPT-4o-mini
- Cost: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- Typical query: 5 chunks (5,000 tokens) + question (100 tokens) ≈ $0.0008

### Storage
- Qdrant free tier: 1 GB
- Each chunk: ~1 KB metadata + 12 KB vector (3072 dims × 4 bytes)
- Capacity: ~75,000 chunks (approximately 1,500 documents of 50 chunks each)

---

## Future Enhancements

### Phase 1 (Week 9-12) - Policy Features
- Policy template library
- Policy generation from user input
- Policy compliance checking
- Risk identification

### Phase 2 (Week 13-18) - Advanced Features
- Bilingual support (French)
- Language detection and switching
- Content recommendation system
- Feedback collection mechanism

### Phase 3 (Week 19-20) - Production
- Deployment to Hugging Face Spaces
- User authentication (if needed)
- Analytics dashboard
- Automated expiry detection for policies

---

## Troubleshooting

### Common Issues

**1. "Collection not found" error**
```bash
# Solution: Collection is created automatically on first upload
# Just upload a document and it will be created
```

**2. "No documents found" when asking questions**
```bash
# Solution: Upload at least one document first via admin.py
```

**3. "Rate limit exceeded" from OpenAI**
```bash
# Solution: Add delays between requests or upgrade OpenAI plan
```

**4. "Firecrawl scraping failed"**
```bash
# Solution: Check if URL is accessible, verify Firecrawl API key
```

---

## Development Timeline

- **Week 1-2:** Infrastructure setup ✅
- **Week 3-4:** Basic RAG system ✅
- **Week 5-6:** Web scraping + chat interface
- **Week 7-8:** Quality improvements
- **Week 9-10:** Admin interface
- **Week 11-12:** Demo delivery
- **Week 13-16:** Policy features
- **Week 17-18:** Bilingual support
- **Week 19-20:** Final delivery

---

## References

- LangChain Documentation: https://python.langchain.com/docs/
- Qdrant Documentation: https://qdrant.tech/documentation/
- OpenAI API Reference: https://platform.openai.com/docs/
- Gradio Documentation: https://www.gradio.app/docs/
```