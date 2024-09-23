# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM

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
	fullText = []
	for para in doc.paragraphs:
		fullText.append(para.text)
	text = '\n'.join(fullText)
	return text

def text_file_reader(file_path):
	text=""
	f = open(file_path,  mode='r',  encoding='utf8')
	text = f.read()
	return text
