# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM


import os

print("\npython -m pip install --upgrade pip")
os.system('python -m pip install --upgrade pip')

# Delelte Chroma vectorstore ------------------------------------------------------------
import os.path
import shutil

path_to_folder = "chroma_vectorstore"
folder_exists = os.path.exists(path_to_folder)

if folder_exists:
	shutil.rmtree(path_to_folder)
	print("\n",path_to_folder,"is deleted")


# Install needed packages ------------------------------------------------------------
print("\npip install -Ur requirements.txt")
os.system("pip install -Ur requirements.txt")

# # Pull ollama Qwen2-7B model ------------------------------------------------------------
import ollama

print("\nollama pull nomic-embed-text") # Nomic Embed v1.5 Embedding
ollama.pull('nomic-embed-text')

# print("\nollama pull chroma/all-minilm-l6-v2-f32")
# ollama.pull('chroma/all-minilm-l6-v2-f32')

print("\nollama pull qwen2\n")
ollama.pull('qwen2')


from jarvis.gui import JARVIS_assistant

if __name__ == "__main__":
    JARVIS_assistant()


'''
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/

+	-------------------- workflow ---------------------------------
|	v0.1.5
|	JARVIS_assistant.py
|		|
|		~> gui.py ~> custom_ui_style.py
|			|
|			~> gui_action.py ~> model_settings.py , tools.py , prompts.py , utils.py , get_model_list.py
|				|
|				~> db_helper.py  ~> file_readers.py
|					|
|					~> datasource_router.py , retrieval_grader() , hallucination_grader() , answer_grader() , question_rewriter() , sub_query_generator()
+ -----------------------------------------------------------------

'''
