# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

# https://python.langchain.com/v0.2/docs/integrations/document_loaders/recursive_url/
# The RecursiveUrlLoader lets you recursively scrape all child links from a root URL and parse them into Documents.

import logging
from langchain_community.document_loaders import PyPDFLoader
from docx import Document

logger = logging.getLogger(__name__)

def pdf_file_reader(file_path):
    """
    Read and parse a PDF file into pages
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        list: List of document pages or empty list if error occurs
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {str(e)}")
        return []

def docx_file_reader(file_path):
    """
    Read and parse a DOCX file into text
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Text content of the document or empty string if error occurs
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {str(e)}")
        return ""

def text_file_reader(file_path):
    """
    Read a text file
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the text file or empty string if error occurs
    """
    try:
        with open(file_path, mode='r', encoding='utf8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {str(e)}")
        return ""
