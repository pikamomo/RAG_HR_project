import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

load_dotenv()

def create_rag_chain():
    """Create RAG question-answering chain"""
    
    # 1. Connect to Qdrant
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
    
    # 2. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 3. Create LLM
    llm = ChatOpenAI(
        model=os.getenv("OPEN_AI_CHAT_MODEL"),
        temperature=0.3
    )
    
    # 4. System prompt
    system_template = """You are an HR assistant for nonprofit organizations in Canada. 
Use the following context to answer questions accurately and helpfully.

IMPORTANT DISCLAIMERS:
- This tool provides general HR information only
- Not a substitute for professional legal or HR advice
- Consult qualified professionals before implementing policies
- Do NOT share personal information about specific individuals

Context from knowledge base:
{context}

Question: {question}

Provide a clear, helpful answer. If you're not certain, say so. Always remind users to consult HR/legal professionals for important decisions.

Answer:"""
    
    # 5. Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # 6. Create RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(system_template)
        }
    )
    
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question and get answer"""
    result = qa_chain({"question": question})
    
    answer = result["answer"]
    sources = result.get("source_documents", [])
    
    return answer, sources

# Test function
if __name__ == "__main__":
    print("🤖 Initializing chatbot...")
    qa_chain = create_rag_chain()
    
    print("\n✅ Ready! Enter your question (type 'quit' to exit):\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        answer, sources = ask_question(qa_chain, question)
        
        print(f"\nBot: {answer}\n")
        
        if sources:
            print("📚 Sources:")
            for i, doc in enumerate(sources[:3], 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"  {i}. {source}")
            print()