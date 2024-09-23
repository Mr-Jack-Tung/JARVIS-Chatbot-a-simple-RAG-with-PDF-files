# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM

import ollama
def get_ollama_list_models():
    results = ollama.list()
    ollama_list_models = []
    for i in range(len(results['models'])):
        ollama_list_models.append(results['models'][i]['model'])
    return ollama_list_models

import requests
def get_groq_list_models(groq_api_key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    result = response.json()
    list_models = []
    for model in result['data']:
        list_models.append(model['id'])
    return list_models

from openai import OpenAI
def get_openai_list_models(openai_api_key):
    openai_client = OpenAI(api_key=openai_api_key)
    models = openai_client.models.list()
    list_models = []
    for model in models.data:
        list_models.append(model.id)
    return list_models

import google.generativeai as genai
def get_gemini_list_modes(gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    list_models = []
    for m in genai.list_models():
      if 'generateContent' in m.supported_generation_methods:
        list_models.append(m.name.split("/")[-1])
    return list_models

# def get_litellm_list_models():
#     url = "http://0.0.0.0:4000/models"
#     headers = {
#         "accept": "application/json",
#     }
#     response = requests.get(url, headers=headers)
#     result = response.json()
#     return [result['data'][0]['id']]