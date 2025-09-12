# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 27 April 2025
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 12 September 2025
"""
LLM client abstraction for function calling across providers.
Enhanced with better error handling and support for more providers.
"""
import json
import time
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Union
from loguru import logger

class BaseLLMClient:
    """Base class for all LLM clients with common interface."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Send a chat request to the LLM.
        """
        raise NotImplementedError

    def stream(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """
        Send a streaming chat request to the LLM.
        """
        raise NotImplementedError

    def bind(self):
        """Support LangChain react agent stop binding"""
        return self
        
    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response format."""
        return {
            "choices": [
                {
                    "message": {
                        "content": f"Error: {error_message}. Please try again or use a different model."
                    }
                }
            ],
            "error": error_message
        }

class OllamaClient(BaseLLMClient):
    """Client for Ollama models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            from langchain_ollama import ChatOllama    
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            
            self.client = ChatOllama(
                model=model_name,
                temperature=settings.TEMPERATURE,
                top_k=settings.TOP_K,
                top_p=settings.TOP_P,
                num_predict=settings.NUM_PREDICT,
                repeat_penalty=settings.REPEAT_PENALTY
            )
        except Exception as e:
            logger.error(f"Error initializing OllamaClient: {str(e)}")
            logger.error(traceback.format_exc())
            self.client = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to Ollama."""
        logger.info(f"OllamaClient.chat called: model_name={self.model_name}")
        if self.client is None:
            return self._format_error_response(f"Ollama client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        try:
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            
            from .model_settings import Model_Settings as model_settings
            if not model_settings.IS_THINKING:
                prompt += " /no_think"

            response = self.client.invoke(prompt)
            
            if hasattr(response, 'content'):
                content = str(response.content)
            else:
                content = str(response)
            
            if not model_settings.SHOW_THINKING:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                
            result = {"choices": [{"message": {"content": content}}]}
            return result
            
        except Exception as e:
            logger.exception("Error in OllamaClient.chat")
            return self._format_error_response(str(e))

    def stream(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """Send a streaming chat request to Ollama."""
        logger.info(f"OllamaClient.stream called: model_name={self.model_name}")
        if self.client is None:
            yield f"Error: Ollama client initialization failed: {getattr(self, 'init_error', 'Unknown error')}"
            return

        try:
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            
            from .model_settings import Model_Settings as model_settings
            if not model_settings.IS_THINKING:
                prompt += " /no_think"
            
            full_response = ""
            for chunk in self.client.stream(prompt):
                content_chunk = str(chunk.content) if hasattr(chunk, 'content') else str(chunk)
                full_response += content_chunk
                if model_settings.SHOW_THINKING:
                    yield content_chunk

            if not model_settings.SHOW_THINKING:
                processed_content = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                yield processed_content

        except Exception as e:
            logger.exception("Error in OllamaClient.stream")
            yield f"Error: {str(e)}"

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import openai
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            
            self.openai = openai
            self.openai.api_key = settings.OPENAI_API_KEY
            self.temperature = settings.TEMPERATURE
            
        except Exception as e:
            logger.exception("Error initializing OpenAIClient")
            self.openai = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to OpenAI with retry logic."""
        logger.info(f"OpenAIClient.chat called: model_name={self.model_name}")
        if self.openai is None:
            return self._format_error_response(f"OpenAI client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        params = self._prepare_params(messages, functions)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.openai.chat.completions.create(**params)
                return resp.model_dump()
            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.exception(f"OpenAI API failed after {max_retries} attempts")
                return self._format_error_response(str(e))
        return self._format_error_response(f"OpenAI API failed after {max_retries} attempts.")

    def stream(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """Send a streaming chat request to OpenAI."""
        logger.info(f"OpenAIClient.stream called: model_name={self.model_name}")
        if self.openai is None:
            yield f"Error: OpenAI client initialization failed: {getattr(self, 'init_error', 'Unknown error')}"
            return
            
        params = self._prepare_params(messages, functions, stream=True)
        
        try:
            stream = self.openai.chat.completions.create(**params)
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.exception("Error in OpenAIClient.stream")
            yield f"Error: {str(e)}"

    def _prepare_params(self, messages, functions, stream=False):
        """Helper to prepare parameters for OpenAI API call."""
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream
        }
        if functions:
            params["tools"] = functions
            params["tool_choice"] = "auto"
        return params

class GroqClient(OpenAIClient): # Inherits from OpenAIClient as API is compatible
    """Client for Groq models."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import groq
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            self.openai = groq.Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            logger.exception("Error initializing GroqClient")
            self.openai = None
            self.init_error = str(e)

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(model_name)
            self.temperature = settings.TEMPERATURE
        except Exception as e:
            logger.exception("Error initializing GeminiClient")
            self.model = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to Google Gemini."""
        logger.info(f"GeminiClient.chat called: model_name={self.model_name}")
        if self.model is None:
            return self._format_error_response(f"Gemini client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
        
        gemini_messages = self._prepare_messages(messages)
        
        try:
            response = self.model.generate_content(
                gemini_messages,
                generation_config={"temperature": self.temperature}
            )
            return {"choices": [{"message": {"content": response.text}}]}
        except Exception as e:
            logger.exception("Error in GeminiClient.chat")
            return self._format_error_response(str(e))

    def stream(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """Send a streaming chat request to Google Gemini."""
        logger.info(f"GeminiClient.stream called: model_name={self.model_name}")
        if self.model is None:
            yield f"Error: Gemini client initialization failed: {getattr(self, 'init_error', 'Unknown error')}"
            return
            
        gemini_messages = self._prepare_messages(messages)
        
        try:
            stream = self.model.generate_content(
                gemini_messages,
                generation_config={"temperature": self.temperature},
                stream=True
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Error in GeminiClient.stream")
            yield f"Error: {str(e)}"

    def _prepare_messages(self, messages):
        """Helper to prepare messages for Gemini API call."""
        system_instruction = None
        gemini_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
                break

        is_first_user_message = True
        for msg in messages:
            if msg['role'] == 'system':
                continue

            role = 'model' if msg['role'] == 'assistant' else 'user'
            content = msg['content']

            if role == 'user' and is_first_user_message and system_instruction:
                content = f"{system_instruction}\n\n{content}"
                is_first_user_message = False

            gemini_messages.append({
                'role': role,
                'parts': [{'text': content}]
            })
        return gemini_messages

class LitellmClient(BaseLLMClient):
    """Client for unified access to various LLM providers through LiteLLM."""
    
    def __init__(self, model_type: str, model_name: str):
        super().__init__(model_name)
        try:
            from litellm import completion
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            
            self.completion = completion
            self.model_type = model_type
            self.temperature = settings.TEMPERATURE
            
            os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
            os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            
        except Exception as e:
            logger.exception("Error initializing LitellmClient")
            self.completion = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request through LiteLLM."""
        return self._execute_litellm(messages, functions, stream=False)

    def stream(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None):
        """Send a streaming chat request through LiteLLM."""
        stream = self._execute_litellm(messages, functions, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _execute_litellm(self, messages, functions, stream=False):
        """Helper to execute LiteLLM completion."""
        logger.info(f"LitellmClient {'streaming' if stream else 'chat'} called: model_type={self.model_type}, model_name={self.model_name}")
        if self.completion is None:
            if stream:
                return iter([f"Error: LiteLLM client initialization failed: {getattr(self, 'init_error', 'Unknown error')}"])
            return self._format_error_response(f"LiteLLM client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        prefix = {
            "LiteLLM": "ollama/", "GroqCloud": "groq/", "Gemini": "gemini/", "OpenAI": "openai/"
        }.get(self.model_type, "")
        model = prefix + self.model_name
        
        kwargs = {"model": model, "messages": messages, "temperature": self.temperature, "stream": stream}
        if functions:
            kwargs["tools"] = functions
            kwargs["tool_choice"] = "auto"
        
        try:
            return self.completion(**kwargs)
        except Exception as e:
            logger.exception("Error in LitellmClient._execute_litellm")
            if stream:
                return iter([f"Error: {str(e)}"])
            return self._format_error_response(str(e))

def get_llm_client(model_type: str, model_name: str) -> BaseLLMClient:
    """Factory to get appropriate LLM client."""
    logger.info(f"get_llm_client called: model_type={model_type}, model_name={model_name}")
    client_map = {
        "Ollama": OllamaClient,
        "OpenAI": OpenAIClient,
        "GroqCloud": GroqClient,
        "Gemini": GeminiClient,
        "LiteLLM": lambda name: LitellmClient(model_type, name)
    }
    
    ClientClass = client_map.get(model_type)
    
    if ClientClass:
        return ClientClass(model_name)
    else:
        logger.warning(f"Unknown model_type: {model_type}, attempting to use LiteLLM")
        return LitellmClient("LiteLLM", model_name)
