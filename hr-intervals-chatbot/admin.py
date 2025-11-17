"""
Gradio admin interface for content management
Allows uploading documents, scraping URLs, and managing content
"""

import gradio as gr
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from src.ingestion import ingest_document
from src.scraper import process_and_store_webpage

load_dotenv()

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = os.getenv("QDRANT_COLLECTION")


# ==================== Functions ====================

def list_all_documents():
    """
    List all uploaded documents
    
    Returns:
        List of [name, type, date, chunks] for display
    """
    try:
        result = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True
        )
        
        # Group by source
        docs_dict = {}
        for point in result[0]:
            payload = point.payload
            # Metadata is nested inside payload
            metadata = payload.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            if source not in docs_dict:
                docs_dict[source] = {
                    "name": source,
                    "type": metadata.get("type", "Unknown"),
                    "date": metadata.get("upload_date", "Unknown"),
                    "chunks": 0
                }
            docs_dict[source]["chunks"] += 1
        
        # Convert to list for Gradio DataFrame
        docs_list = [
            [v["name"], v["type"], v["date"], v["chunks"]] 
            for v in docs_dict.values()
        ]
        
        return docs_list if docs_list else [["No documents yet", "", "", 0]]
    
    except Exception as e:
        return [[f"Error: {str(e)}", "", "", 0]]


def upload_document(file, doc_type="document"):
    """
    Upload PDF or DOCX file
    
    Args:
        file: Uploaded file object
        doc_type: Type of document
        
    Returns:
        Success message
    """
    if file is None:
        return "❌ Please select a file"
    
    try:
        file_path = file.name
        
        # Ingest document
        num_chunks = ingest_document(file_path, doc_type)
        
        return f"✅ Success!\n\nFile: {os.path.basename(file_path)}\nChunks created: {num_chunks}\nType: {doc_type}"
    
    except Exception as e:
        return f"❌ Upload failed:\n{str(e)}"


def scrape_single_url(url):
    """
    Scrape single URL
    
    Args:
        url: URL to scrape
        
    Returns:
        Success message
    """
    if not url:
        return "❌ Please enter a URL"
    
    try:
        num_chunks = process_and_store_webpage(url)
        return f"✅ Success!\n\nURL: {url}\nChunks created: {num_chunks}"
    
    except Exception as e:
        return f"❌ Scraping failed:\n{str(e)}"


def scrape_multiple_urls(urls_text):
    """
    Scrape multiple URLs
    
    Args:
        urls_text: URLs separated by newlines
        
    Returns:
        Summary of results
    """
    if not urls_text:
        return "❌ Please enter URLs (one per line)"
    
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    results = []
    success_count = 0
    fail_count = 0
    
    for url in urls:
        try:
            num_chunks = process_and_store_webpage(url)
            results.append(f"✅ {url}: {num_chunks} chunks")
            success_count += 1
        except Exception as e:
            results.append(f"❌ {url}: {str(e)}")
            fail_count += 1
    
    summary = f"📊 Summary: {success_count} succeeded, {fail_count} failed\n\n"
    return summary + "\n".join(results)


def delete_document(source_name):
    """
    Delete document by source name
    
    Args:
        source_name: Name or URL of the source
        
    Returns:
        Success message
    """
    if not source_name:
        return "❌ Please enter document name or URL"
    
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )
        )
        
        return f"✅ Successfully deleted all content from:\n{source_name}"
    
    except Exception as e:
        return f"❌ Deletion failed:\n{str(e)}"


def update_document(old_source, new_file):
    """
    Replace old document with new one
    
    Args:
        old_source: Name of old document
        new_file: New file to upload
        
    Returns:
        Success message
    """
    if not old_source:
        return "❌ Please enter the old document name"
    
    if new_file is None:
        return "❌ Please select a new file"
    
    try:
        # Delete old
        delete_result = delete_document(old_source)
        
        # Upload new
        upload_result = upload_document(new_file)
        
        return f"🔄 Update Complete\n\n**Step 1 - Delete:**\n{delete_result}\n\n**Step 2 - Upload:**\n{upload_result}"
    
    except Exception as e:
        return f"❌ Update failed:\n{str(e)}"


# ==================== Gradio Interface (5.49) ====================

with gr.Blocks(
    title="HR Intervals - Admin Panel", 
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("# 📁 HR Intervals - Knowledge Base Management")
    gr.Markdown("Manage documents and web content for the AI assistant")
    
    with gr.Tabs():
        
        # Tab 1: View Documents
        with gr.Tab("📋 View Documents"):
            gr.Markdown("### Current documents in knowledge base")
            
            refresh_btn = gr.Button("🔄 Refresh List", variant="primary")
            docs_table = gr.Dataframe(
                headers=["Document Name", "Type", "Upload Date", "Chunks"],
                label="Documents",
                interactive=False,
                wrap=True
            )
            
            refresh_btn.click(list_all_documents, outputs=docs_table)
            demo.load(list_all_documents, outputs=docs_table)
        
        # Tab 2: Upload Documents
        with gr.Tab("⬆️ Upload Documents"):
            gr.Markdown("### Upload PDF or DOCX files")
            
            file_input = gr.File(
                label="Select File (PDF or DOCX)",
                file_types=[".pdf", ".docx"]
            )
            
            doc_type_input = gr.Radio(
                choices=["document", "policy", "guide", "article"],
                value="document",
                label="Document Type"
            )
            
            upload_btn = gr.Button("📤 Upload", variant="primary", size="lg")
            upload_output = gr.Textbox(label="Upload Result", lines=5)
            
            upload_btn.click(
                upload_document,
                inputs=[file_input, doc_type_input],
                outputs=upload_output
            )
        
        # Tab 3: Scrape URLs
        with gr.Tab("🌐 Scrape Web Pages"):
            gr.Markdown("### Scrape content from URLs")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Single URL")
                    url_input = gr.Textbox(
                        label="Enter URL",
                        placeholder="https://example.com/article"
                    )
                    scrape_btn = gr.Button("🔍 Scrape", variant="primary")
                    scrape_output = gr.Textbox(label="Result", lines=4)
                    
                    scrape_btn.click(
                        scrape_single_url,
                        inputs=url_input,
                        outputs=scrape_output
                    )
                
                with gr.Column():
                    gr.Markdown("#### Batch URLs")
                    urls_input = gr.Textbox(
                        label="Enter multiple URLs (one per line)",
                        placeholder="https://example.com/page1\nhttps://example.com/page2",
                        lines=6
                    )
                    batch_btn = gr.Button("🔍 Batch Scrape", variant="primary")
                    batch_output = gr.Textbox(label="Batch Results", lines=8)
                    
                    batch_btn.click(
                        scrape_multiple_urls,
                        inputs=urls_input,
                        outputs=batch_output
                    )
        
        # Tab 4: Delete Documents
        with gr.Tab("🗑️ Delete Documents"):
            gr.Markdown("### Delete documents or web pages")
            gr.Markdown("⚠️ **Warning**: This operation cannot be undone!")
            
            delete_input = gr.Textbox(
                label="Document Name or URL",
                placeholder="e.g., hiring_policy.pdf or https://example.com/article"
            )
            
            delete_btn = gr.Button("🗑️ Delete", variant="stop", size="lg")
            delete_output = gr.Textbox(label="Delete Result", lines=3)
            
            delete_btn.click(
                delete_document,
                inputs=delete_input,
                outputs=delete_output
            )
        
        # Tab 5: Update Documents
        with gr.Tab("🔄 Update Documents"):
            gr.Markdown("### Replace existing document with new version")
            
            old_doc_input = gr.Textbox(
                label="Old Document Name",
                placeholder="e.g., old_policy.pdf"
            )
            
            new_file_input = gr.File(
                label="New File",
                file_types=[".pdf", ".docx"]
            )
            
            update_btn = gr.Button("🔄 Update", variant="primary", size="lg")
            update_output = gr.Textbox(label="Update Result", lines=8)
            
            update_btn.click(
                update_document,
                inputs=[old_doc_input, new_file_input],
                outputs=update_output
            )
        
        # Tab 6: Help
        with gr.Tab("ℹ️ Help"):
            gr.Markdown("""
            ### Usage Guide
            
            #### 📋 View Documents
            - Shows all uploaded documents and web pages
            - Displays document type, upload date, and number of chunks
            - Click "Refresh" to see the latest status
            
            #### ⬆️ Upload Documents
            - Supports PDF and DOCX formats
            - Documents are automatically split into chunks (~1000 characters each)
            - You can categorize documents by type
            
            #### 🌐 Scrape Web Pages
            - Enter full URLs (including https://)
            - Supports single or batch scraping
            - Content is automatically converted to Markdown format
            
            #### 🗑️ Delete Documents
            - Enter exact filename or URL
            - Deletes all chunks from that source
            - **Warning**: Cannot be undone!
            
            #### 🔄 Update Documents
            - Enter old document name
            - Upload new version
            - Automatically deletes old and uploads new
            
            ---
            
            ### Advanced Management
            
            For detailed vector database management, visit:
            [Qdrant Cloud Dashboard](https://cloud.qdrant.io)
            
            ### Technical Support
            
            If you encounter issues, please contact the development team.
            """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )