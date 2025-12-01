---
title: HR Intervals AI Assistant
emoji: 💼
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: false
---

# HR Intervals AI Assistant

An AI-powered HR assistant that provides instant answers to HR questions using RAG (Retrieval Augmented Generation).

## Configuration

This app requires the following environment variables to be set in Hugging Face Spaces Settings:

- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: Your Qdrant instance URL (or use local mode)
- `QDRANT_API_KEY`: Your Qdrant API key (if using cloud)

## How to Use

1. Ask HR-related questions in the chat interface
2. Get AI-powered answers based on the knowledge base
3. View source documents for transparency

