# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.6
# Update: 28 April 2025

# https://python.langchain.com/v0.2/docs/integrations/document_loaders/recursive_url/
# The RecursiveUrlLoader lets you recursively scrape all child links from a root URL and parse them into Documents.

from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import TextLoader

def pdf_file_reader(file_path):
	loader = PyPDFLoader(file_path)
	pages = loader.load_and_split()
	return pages

from docx import Document
def docx_file_reader(file_path):
	text = ""
	doc = Document(file_path)
	full_text = []
	for para in doc.paragraphs:
		full_text.append(para.text)
	text = '\n'.join(full_text)
	return text

def text_file_reader(file_path):
	text=""
	f = open(file_path,  mode='r',  encoding='utf8')
	text = f.read()
	return text
