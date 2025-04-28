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
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis.log')
    ]
)

from jarvis_assistant.jarvis.gui import JARVIS_assistant

def main():
    """Launch the JARVIS assistant GUI"""
    try:
        JARVIS_assistant()
    except Exception as e:
        logging.error(f"Error starting JARVIS: {str(e)}")
        raise

if __name__ == "__main__":
    main()
