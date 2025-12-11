"""
Gradio chat interface for end users
Uses Gradio 5.49 ChatInterface API
"""

import gradio as gr
import os
from dotenv import load_dotenv
from src.chatbot import create_rag_chain, ask_question
import re
import uuid

load_dotenv()

# Initialize chatbot
print("🤖 Initializing chatbot...")
rag_chain, retriever = create_rag_chain()
print("✅ Chatbot ready!")

# Generate unique session ID for each user
session_id = str(uuid.uuid4())


def check_pii(text: str) -> bool:
    """
    Simple PII detection - checks for potential names
    
    Args:
        text: Input text to check
        
    Returns:
        True if PII detected
    """
    # Check for capitalized words that might be names
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    if re.search(name_pattern, text):
        return True
    return False


def chat_response(message: str, history: list) -> str:
    """
    Handle chat messages (Gradio 5.x format)
    
    Args:
        message: User's message
        history: Conversation history
        
    Returns:
        Bot's response
    """
    
    # Check for PII
    warning = ""
    if check_pii(message):
        warning = "⚠️ **Warning**: Please avoid sharing personal information about specific individuals.\n\n"
    
    # Get answer from chatbot
    try:
        answer, sources = ask_question(rag_chain, retriever, message, session_id)
        
        # Format response with sources
        response = warning + answer
        
        if sources:
            response += "\n\n📚 **Sources:**\n"
            for i, doc in enumerate(sources[:3], 1):
                source = doc.metadata.get("source", "Unknown")
                response += f"{i}. {source}\n"
        
        return response
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease make sure documents have been uploaded to the system."


# Create Gradio interface (Gradio 5.49 API)
with gr.Blocks(
    title="HR Intervals AI Assistant", 
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 💼 HR Intervals AI Assistant
    
    Get instant answers to your HR questions based on our knowledge base.
    """)
    
    # Disclaimer
    with gr.Accordion("⚠️ Important Disclaimer - Please Read", open=False):
        gr.Markdown("""
        **This tool is designed to provide general HR-related information and draft policy suggestions.**
        
        - This is **NOT** a substitute for professional legal or HR advice
        - For legal compliance and important decisions, consult a qualified attorney or HR professional
        - Do **NOT** share personal information about specific individuals
        
        By using this tool, you acknowledge that you understand these limitations.
        """)
    
    # Welcome message with disclaimer and example questions
    WELCOME_MESSAGE = """👋 **Welcome to the HR Intervals AI Assistant!**

⚠️ **Important Disclaimer:**

This tool is designed to provide general HR-related information and draft policy suggestions. It is not a substitute for professional legal or HR advice. For legal compliance and to ensure the best outcome for your organization, we recommend consulting a qualified attorney or HR professional before implementing any policies or making decisions based on the information provided.

---

How can I help you today? **Try asking:**

• What should I include in a remote work policy?
• How do I handle employee terminations properly?
• What are best practices for hiring in Canada?
• Tell me about workplace safety requirements"""
    
    # Chat interface (Gradio 5.x ChatInterface)
    chat_interface = gr.ChatInterface(
        fn=chat_response,
        chatbot=gr.Chatbot(
            height=500,
            show_label=False,
            type='messages',
            avatar_images=(None, "https://em-content.zobj.net/thumbs/120/apple/354/robot_1f916.png"),
            value=[{"role": "assistant", "content": WELCOME_MESSAGE}]
        ),
        textbox=gr.Textbox(
            placeholder="Ask your HR question here...",
            container=False,
            scale=7
        ),
        title="",
        description="",
        theme=gr.themes.Soft()
    )
    
    # Footer
    gr.Markdown("""
    ---
    💡 **Tip**: Be specific in your questions for better answers. Remember to consult professionals for legal matters.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
