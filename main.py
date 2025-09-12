#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18
"""
Entry point for JARVIS Chatbot application.

JARVIS is a Retrieval-Augmented Generation (RAG) chatbot that can:
- Process and index PDF, TXT, and DOCX files
- Retrieve relevant information from indexed documents
- Augment responses with web search when needed
- Support multiple LLM backends (Ollama, OpenAI, Groq, Gemini)
- Use function calling with ReWOO and ReACT agents
"""
import sys
from loguru import logger

# Configure Loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add console sink
logger.add("jarvis.log", level="INFO", rotation="10 MB", retention="7 days")  # Add file sink

from jarvis_assistant.jarvis.gui import JARVIS_assistant

def main():
    """Launch the JARVIS assistant GUI"""
    try:
        logger.info("Starting JARVIS Assistant...")
        JARVIS_assistant()
    except Exception as e:
        logger.exception(f"Critical error starting JARVIS: {str(e)}")
        raise

if __name__ == "__main__":
    main()
