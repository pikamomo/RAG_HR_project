"""
RAG chatbot module using latest LangChain with LCEL
Handles question-answering with conversation memory using modern patterns
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import Tuple, List, Dict, Any
from operator import itemgetter

load_dotenv()

# Store for chat sessions
session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create chat history for a session
    
    Args:
        session_id: Unique identifier for the session
        
    Returns:
        Chat message history object
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single string
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted string with document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """
    Create RAG question-answering chain using LCEL (LangChain Expression Language)
    Modern approach with pipe operator for better composability
    
    Returns:
        Conversational RAG chain with message history
    """
    
    # 1. Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPEN_AI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION"),
        embedding=embeddings
    )
    
    # 2. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 3. Create LLM
    llm = ChatOpenAI(
        model=os.getenv("OPEN_AI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.3
    )
    
    # 4. System prompt
    system_prompt = """You are an HR assistant for nonprofit organizations in Canada. 
Use the following context to answer questions accurately and helpfully.

IMPORTANT DISCLAIMERS:
- This tool provides general HR information only
- Not a substitute for professional legal or HR advice
- Consult qualified professionals before implementing policies
- Do NOT share personal information about specific individuals

Context:
{context}

Provide a clear, helpful answer. If you're not certain, say so. Always remind users to consult HR/legal professionals for important decisions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 5. Build RAG chain using LCEL (pipe operator)
    # This is the modern LangChain approach for better composability
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Add chat history with message management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return conversational_rag_chain, retriever


def ask_question(
    rag_chain, 
    retriever,
    question: str, 
    session_id: str = "default"
) -> Tuple[str, List[Document]]:
    """
    Ask a question and get answer with sources
    
    Args:
        rag_chain: The RAG chain
        retriever: The vector store retriever for getting sources
        question: User's question
        session_id: Session identifier for conversation history
        
    Returns:
        Tuple of (answer, source_documents)
    """
    
    # Get answer from conversational chain
    answer = rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    # Retrieve source documents separately for display
    sources = retriever.invoke(question)
    
    return answer, sources


# Test function
if __name__ == "__main__":
    print("🤖 Initializing chatbot with latest LangChain (LCEL)...")
    rag_chain, retriever = create_rag_chain()
    
    print("\n✅ Ready! Enter your question (type 'quit' to exit):\n")
    
    session_id = "test_session"
    
    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            answer, sources = ask_question(rag_chain, retriever, question, session_id)
            
            print(f"\nBot: {answer}\n")
            
            if sources:
                print("📚 Sources:")
                for i, doc in enumerate(sources[:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  {i}. {source}")
                print()
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Make sure you have uploaded some documents first.\n")