# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.6
# Update: 28 April 2025

class Model_Settings:
    # Định nghĩa trực tiếp các thuộc tính (attributes) ở cấp lớp (class-level)
    MODEL_TYPE = "Ollama"
    MODEL_NAME = 'qwen2.5:3b'
    NUM_PREDICT = 2048
    TEMPERATURE = 0.2
    TOP_K = 50
    TOP_P = 0.95
    REPEAT_PENALTY = 1.12
    SYSTEM_PROMPT = ""
    RETRIEVAL_TOP_K = 3
    RETRIEVAL_THRESHOLD = 0.25
    GROQ_API_KEY = ""
    OPENAI_API_KEY = ""
    GEMINI_API_KEY = ""
    IS_RETRIEVAL = False
    IS_GRADER = False
    IS_WEB_SEARCH = True
    FUNCTION_CALLING = False
    AGENT_CALLING = "ReWOO"
    CHAT_HISTORY_SAVING = True
