# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM

import os
import yaml

def save_api_keys_to_yaml(GROQ_KEY, OPENAI_KEY, GEMINI_KEY):
    yaml_data = {
    "GROQ_API_KEY": GROQ_KEY,
    "OPENAI_API_KEY": OPENAI_KEY,
    "GEMINI_API_KEY": GEMINI_KEY,
    }
    with open('api_keys.yaml', 'w') as file:
        yaml.dump(yaml_data, file)
    print(" ~> save api keys to yaml file")

def load_api_keys_from_yaml(model_settings):
    GROQ_KEY = ""
    OPENAI_KEY = ""
    GEMINI_KEY = ""

    if os.path.exists("api_keys.yaml"):
        with open("api_keys.yaml", 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        
            GROQ_KEY = data_loaded["GROQ_API_KEY"]
            OPENAI_KEY = data_loaded["OPENAI_API_KEY"]
            GEMINI_KEY = data_loaded["GEMINI_API_KEY"]

            model_settings.GROQ_API_KEY = GROQ_KEY
            model_settings.OPENAI_API_KEY = OPENAI_KEY
            model_settings.GEMINI_API_KEY = GEMINI_KEY

            os.environ['GROQ_API_KEY'] = GROQ_KEY
            os.environ["OPENAI_API_KEY"] = OPENAI_KEY
            os.environ["GEMINI_API_KEY"] = GEMINI_KEY

    return GROQ_KEY, OPENAI_KEY, GEMINI_KEY