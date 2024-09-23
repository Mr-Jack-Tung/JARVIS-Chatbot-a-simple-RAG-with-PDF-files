# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 19 December 2024 - 10 AM

import os

print("\npython -m pip install --upgrade pip")
os.system('python -m pip install --upgrade pip')

print("\npython -m pip install --user pipx")
os.system("python -m pip install --user pipx")

print("\npipx install poetry")
os.system("pipx install poetry")

print("\npoetry update")
os.system("poetry update")

# Delelte Chroma vectorstore ------------------------------------------------------------
import os.path
import shutil

path_to_folder = "chroma_vectorstore"
folder_exists = os.path.exists(path_to_folder)

if folder_exists:
	shutil.rmtree(path_to_folder)
	print("\n",path_to_folder,"is deleted")

# Install needed packages ------------------------------------------------------------
# print("\npython -m pip install -Ur  requirements.txt")
# os.system("python -m pip install -Ur requirements.txt")

print("\npoetry install")
os.system("poetry install")

# Pull ollama Qwen2-7B model ------------------------------------------------------------
import ollama

print("\nollama pull nomic-embed-text") # Nomic Embed v1.5 Embedding
ollama.pull('nomic-embed-text')

# print("\nollama pull chroma/all-minilm-l6-v2-f32")
# ollama.pull('chroma/all-minilm-l6-v2-f32')

print("\nollama pull qwen2.5\n")
ollama.pull('qwen2.5')
