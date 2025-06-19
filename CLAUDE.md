# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based RAG (Retrieval-Augmented Generation) chatbot demo that allows users to upload PDF documents and ask questions about their content. The application uses OpenAI's GPT models, LangChain for document processing, and Chroma as the vector database.

## Development Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit application
uv run streamlit run app.py

# Install new dependencies
uv add <package_name>
```

## Architecture

### Core Components
- **Frontend**: Streamlit web interface with sidebar configuration
- **Document Processing**: PyMuPDF for PDF loading, RecursiveCharacterTextSplitter for chunking
- **Vector Database**: Chroma for storing document embeddings
- **LLM Integration**: OpenAI GPT models via LangChain
- **Memory**: ConversationBufferMemory for chat history

### Key Files
- `app.py`: Main Streamlit application with chat interface
- `.env`: OpenAI API key configuration (not committed)
- `.env.example`: Template for environment variables
- `pyproject.toml`: uv project configuration and dependencies

### Configuration Options
- Model selection (GPT-3.5-turbo, GPT-4, GPT-4o-mini)
- Temperature control (0.0-1.0)
- Chunk size adjustment (500-2000 characters)

### Session State Management
- Chat messages history
- Vector store instance
- Conversation chain with memory