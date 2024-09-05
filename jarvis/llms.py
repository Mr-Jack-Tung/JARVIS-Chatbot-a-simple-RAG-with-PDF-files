# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 05 December 2024 - 09 PM

from litellm import completion

def llm_completion(model_type, model_name, system_prompt, prompt):
	result = ""
	if model_type == "LiteLLM":
		response = completion(model="ollama/" + model_name, api_base="http://localhost:11434", messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
	if model_type == "OpenAI":
		response = completion(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],)
	if model_type == "GroqCloud":
		response = completion(model="groq/" + model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],)
	if model_type == "Gemini":
		response = completion(model="gemini/" + model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],)
	if response:
		result = response.choices[0].message.content
	return result
