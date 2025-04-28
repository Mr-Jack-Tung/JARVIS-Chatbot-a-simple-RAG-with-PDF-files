# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.6
# Date: 28 April 2025 - 12 AM

import openai
import os
from typing import Any  # kept for potential future use

# Set OpenAI API key from env (loaded via utils)
openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_completion(model_type, model_name, system_prompt, prompt):
    """
    Send chat completion via unified LLM client.
    """
    from .llm_client import get_llm_client
    client = get_llm_client(model_type, model_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat(messages)
    if resp.get("choices") and resp["choices"]:
        return resp["choices"][0]["message"]["content"]
    if resp.get("error"):
        raise ValueError(resp["error"])
    return str(resp)
