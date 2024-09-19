# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 18 December 2024 - 10 PM

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
