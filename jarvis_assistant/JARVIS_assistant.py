# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

"""
JARVIS Assistant - A Retrieval-Augmented Generation (RAG) chatbot

This module serves as an alternative entry point for the JARVIS assistant.
For normal usage, run the main.py file in the project root.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis.log')
    ]
)

from jarvis_assistant.jarvis.gui import JARVIS_assistant

if __name__ == "__main__":
    try:
        JARVIS_assistant()
    except Exception as e:
        logging.error(f"Error starting JARVIS: {str(e)}")
        raise


'''
Project Architecture:
https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/

+	-------------------- workflow ---------------------------------
|	v0.1.7
|	JARVIS_assistant.py / main.py
|		|
|		~> gui.py ~> custom_ui_style.py
|			|
|			~> gui_action.py ~> model_settings.py , tools.py , prompts.py , utils.py , get_model_list.py
|				|
|				~> db_helper.py  ~> file_readers.py
|					|
|					~> datasource_router.py , retrieval_grader() , hallucination_grader() , 
|                     answer_grader() , question_rewriter() , sub_query_generator()
+ -----------------------------------------------------------------

'''
