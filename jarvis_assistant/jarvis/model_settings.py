# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

class Model_Settings:
    """
    Configuration settings for the JARVIS Chatbot.
    
    This class stores all the settings related to model selection,
    generation parameters, retrieval settings, and feature toggles.
    """
    
    # Model selection settings
    MODEL_TYPE = "Ollama"  # Options: "Ollama", "GroqCloud", "OpenAI", "Gemini", "LiteLLM"
    MODEL_NAME = 'qwen2.5:3b'  # Default model name
    
    # Generation parameters
    NUM_PREDICT = 2048     # Maximum number of tokens to generate
    TEMPERATURE = 0.2      # Controls randomness (0.0 = deterministic, 1.0 = creative)
    TOP_K = 50             # Limits vocabulary to top K tokens
    TOP_P = 0.95           # Nucleus sampling parameter
    REPEAT_PENALTY = 1.12  # Penalty for repeating tokens
    
    # System prompt
    SYSTEM_PROMPT = ""     # Will be initialized from prompts.py
    
    # Retrieval settings
    RETRIEVAL_TOP_K = 3        # Number of documents to retrieve
    RETRIEVAL_THRESHOLD = 0.25 # Minimum similarity score for retrieval
    
    # API keys for external services
    GROQ_API_KEY = ""
    OPENAI_API_KEY = ""
    GEMINI_API_KEY = ""
    
    # Feature toggles
    IS_RETRIEVAL = False       # Enable/disable retrieval augmentation
    IS_GRADER = False          # Enable/disable relevance grading
    IS_WEB_SEARCH = True       # Enable/disable web search augmentation
    FUNCTION_CALLING = False   # Enable/disable function calling
    AGENT_CALLING = "ReWOO"    # Agent type: "ReWOO" or "ReACT"
    CHAT_HISTORY_SAVING = True # Enable/disable saving chat history
    
    @classmethod
    def get_model_info(cls):
        """
        Get a summary of the current model settings
        
        Returns:
            str: Formatted string with model information
        """
        return (
            f"Model: {cls.MODEL_TYPE}/{cls.MODEL_NAME}\n"
            f"Parameters: temp={cls.TEMPERATURE}, top_k={cls.TOP_K}, "
            f"top_p={cls.TOP_P}, max_tokens={cls.NUM_PREDICT}\n"
            f"Features: retrieval={'ON' if cls.IS_RETRIEVAL else 'OFF'}, "
            f"grader={'ON' if cls.IS_GRADER else 'OFF'}, "
            f"web_search={'ON' if cls.IS_WEB_SEARCH else 'OFF'}, "
            f"function_calling={'ON' if cls.FUNCTION_CALLING else 'OFF'}"
        )
