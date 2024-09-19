# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 07 December 2024 - 01 AM

class Model_Settings:
    def __init__(self):
        self.MODEL_TYPE = "Ollama"
        self.MODEL_NAME = 'qwen2.5:latest'
        self.NUM_PREDICT = 2048
        self.TEMPERATURE = 0
        self.TOP_K = 50
        self.TOP_P = 1
        self.REPEAT_PENALTY = 1.12
        self.SYSTEM_PROMPT = ""
        self.RETRIEVAL_TOP_K = 3
        self.RETRIEVAL_THRESHOLD = 0.25
        self.GROQ_API_KEY = ""
        self.OPENAI_API_KEY = ""
        self.GEMINI_API_KEY = ""
        self.IS_RETRIEVAL = True
        self.FUNCTION_CALLING = True
        self.AGENT_CALLING = "ReWOO"
        self.CHAT_HISTORY_SAVING = True
