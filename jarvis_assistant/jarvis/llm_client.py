# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 27 April 2025
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.6
# Update: 28 April 2025
"""
LLM client abstraction for function calling across providers.
Enhanced with better error handling and support for more providers.
"""
import json
import time
import os
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
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            functions: Optional list of function definitions for function calling
            
        Returns:
            Dict with response from the LLM
        """
        raise NotImplementedError

    def bind(self):
        """Support LangChain react agent stop binding"""
        # Return self to allow chaining with bind()
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
            
            # Initialize with user settings
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
            # Still create the instance but mark as failed
            self.client = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Send a chat request to Ollama.
        
        Note: Ollama client via Langchain does not yet fully support function calling API
        in this wrapper, so we use a simplified approach.
        """
        logger.info(f"OllamaClient.chat called: model_name={self.model_name}")
        logger.debug(f"Messages: {messages}, Functions: {functions}")
        if self.client is None:
            return self._format_error_response(f"Ollama client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        try:
            # Format messages as a single prompt
            prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            
            # Append /no_think to the prompt
            prompt += " /no_think"
            
            # Invoke the model
            response = self.client.invoke(prompt)
            logger.debug(f"OllamaClient response raw: {response}")
            
            # Format response to match expected structure
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Remove <think> and </think> tokens
            content = content.replace("<think>", "").replace("</think>", "").strip()
                
            result = {"choices": [{"message": {"content": content}}]}
            logger.info("OllamaClient.chat succeeded")
            return result
            
        except Exception as e:
            logger.error(f"Error in OllamaClient.chat: {str(e)}")
            logger.error(traceback.format_exc())
            return self._format_error_response(str(e))

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import openai
            import yaml
            from pathlib import Path
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            
            # Load .env to populate environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).parent.parent.parent / ".env")
            except ImportError:
                pass
            
            # Đọc OpenAI API key từ api_keys.yaml
            config_path = Path(__file__).parent.parent.parent / "api_keys.yaml"
            if not config_path.exists():
                raise FileNotFoundError("Không tìm thấy file api_keys.yaml tại thư mục gốc.")
            data = yaml.safe_load(config_path.read_text())
            # Hỗ trợ lấy từ file hoặc biến môi trường
            import os
            raw_key = data.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            api_key = raw_key.strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY chưa được cấu hình hoặc rỗng trong api_keys.yaml hoặc biến môi trường")
            openai.api_key = api_key
            self.openai = openai
            self.api_key = api_key
            self.temperature = settings.TEMPERATURE
            
        except Exception as e:
            logger.error(f"Error initializing OpenAIClient: {str(e)}")
            logger.error(traceback.format_exc())
            # Still create the instance but mark as failed
            self.openai = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to OpenAI with retry logic."""
        logger.info(f"OpenAIClient.chat called: model_name={self.model_name}")
        logger.debug(f"Messages before filter: {messages}, Functions: {functions}")
        # Lọc bỏ các system message không hỗ trợ cho một số model
        filtered_messages = [m for m in messages if m.get('role') != 'system']
        logger.debug(f"Messages after filter: {filtered_messages}")
        if self.openai is None:
            return self._format_error_response(f"OpenAI client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        # Chuẩn bị parameters: loại bỏ system, đảm bảo temperature >=1
        temp = self.temperature if getattr(self, 'temperature', None) and self.temperature > 0 else 1
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": filtered_messages,
            "temperature": temp
        }
        
        # Add function calling parameters if provided
        if functions:
            params["functions"] = functions
            params["function_call"] = "auto"
            
        logger.debug(f"OpenAI API key loaded (last 4 chars): {self.api_key[-4:]}" if hasattr(self, 'api_key') else "No API key loaded")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Thử gọi theo new API, fallback legacy
                try:
                    resp = self.openai.chat.completions.create(**params)
                except AttributeError:
                    resp = self.openai.ChatCompletion.create(**params)
                # Chuyển thành dict nếu cần
                if not isinstance(resp, dict):
                    resp = resp.model_dump()
                logger.debug(f"OpenAIClient response raw: {resp}")
                logger.info("OpenAIClient.chat succeeded")
                return resp
                
            except Exception as e:
                logger.warning(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                    continue
                logger.error(f"OpenAI API failed after {max_retries} attempts: {str(e)}")
                logger.error(traceback.format_exc())
                return self._format_error_response(str(e))

class GroqClient(BaseLLMClient):
    """Client for Groq models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import yaml
            from pathlib import Path
            import os
            # Load GROQ_API_KEY từ api_keys.yaml hoặc biến môi trường
            config_path = Path(__file__).parent.parent.parent / "api_keys.yaml"
            if not config_path.exists():
                raise FileNotFoundError("Không tìm thấy file api_keys.yaml tại thư mục gốc.")
            data = yaml.safe_load(config_path.read_text())
            raw = data.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or ""
            api_key = raw.strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY chưa được cấu hình trong api_keys.yaml hoặc biến môi trường")
            # Khởi tạo Groq client với HTTP client không dùng proxy nếu có httpx
            import groq
            try:
                import httpx
                http_client = httpx.Client(trust_env=False)
                self.client = groq.Client(api_key=api_key, http_client=http_client)
            except ImportError:
                self.client = groq.Client(api_key=api_key)
            # Thiết lập temperature từ ModelSettings
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            self.temperature = settings.TEMPERATURE
            
        except Exception as e:
            logger.error(f"Error initializing GroqClient: {str(e)}")
            logger.error(traceback.format_exc())
            # Still create the instance but mark as failed
            self.client = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to Groq."""
        logger.info(f"GroqClient.chat called: model_name={self.model_name}")
        logger.debug(f"Messages: {messages}, Functions: {functions}")
        if self.client is None:
            return self._format_error_response(f"Groq client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature
            }
            
            # Add function calling if supported and provided
            if functions:
                # Note: Check Groq API docs for function calling support
                if self.model_name in ["llama3-70b-8192", "mixtral-8x7b-32768"]:
                    params["tools"] = functions
                    params["tool_choice"] = "auto"
                else:
                    logger.warning(f"Function calling may not be supported for Groq model: {self.model_name}")
            
            # Make the API call
            response = self.client.chat.completions.create(**params)
            logger.debug(f"GroqClient response raw: {response}")
            
            # Convert to dict if needed
            if not isinstance(response, dict):
                response = response.model_dump()
                
            logger.info("GroqClient.chat succeeded")
            return response
            
        except Exception as e:
            logger.error(f"Error in GroqClient.chat: {str(e)}")
            logger.error(traceback.format_exc())
            return self._format_error_response(str(e))

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
            from .model_settings import Model_Settings as ModelSettings
            settings = ModelSettings()
            
            # Configure the API
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.genai = genai
            self.temperature = settings.TEMPERATURE
            
        except Exception as e:
            logger.error(f"Error initializing GeminiClient: {str(e)}")
            logger.error(traceback.format_exc())
            # Still create the instance but mark as failed
            self.genai = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request to Google Gemini."""
        logger.info(f"GeminiClient.chat called: model_name={self.model_name}")
        logger.debug(f"Messages: {messages}, Functions: {functions}")
        if self.genai is None:
            return self._format_error_response(f"Gemini client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                # Map roles to Gemini format
                if role == 'system':
                    # Gemini doesn't have system messages, prepend to first user message
                    continue
                elif role == 'assistant':
                    gemini_role = 'model'
                else:
                    gemini_role = 'user'
                    
                gemini_messages.append({
                    'role': gemini_role,
                    'parts': [{'text': msg.get('content', '')}]
                })
            
            # Initialize model
            model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": self.temperature}
            )
            
            # Handle function calling if provided
            if functions:
                # Note: Check Gemini API docs for function calling support
                logger.warning("Function calling may not be directly supported in this Gemini client implementation")
            
            # Create a chat session
            chat = model.start_chat(history=gemini_messages[:-1])
            
            # Get response
            response = chat.send_message(gemini_messages[-1]['parts'][0]['text'])
            # logger.debug(f"GeminiClient response raw: {response}")
            
            # Format response to match expected structure
            result = {
                "choices": [
                    {
                        "message": {
                            "content": response.text
                        }
                    }
                ]
            }
            logger.info("GeminiClient.chat succeeded")
            return result
            
        except Exception as e:
            logger.error(f"Error in GeminiClient.chat: {str(e)}")
            logger.error(traceback.format_exc())
            return self._format_error_response(str(e))

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
            self.model_name = model_name
            self.temperature = settings.TEMPERATURE
            
            # Set API keys based on model type
            if model_type == "GroqCloud":
                os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
            elif model_type == "Gemini":
                os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
            elif model_type == "OpenAI":
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
                
            # Default API base for local LiteLLM
            self.api_base = os.environ.get("LITELLM_API_BASE", "http://localhost:11434") if model_type == "LiteLLM" else None
            
        except Exception as e:
            logger.error(f"Error initializing LitellmClient: {str(e)}")
            logger.error(traceback.format_exc())
            # Still create the instance but mark as failed
            self.completion = None
            self.init_error = str(e)

    def chat(self, messages: List[Dict[str, str]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Send a chat request through LiteLLM to the appropriate provider."""
        logger.info(f"LitellmClient.chat called: model_type={self.model_type}, model_name={self.model_name}")
        logger.debug(f"Messages: {messages}, Functions: {functions}")
        if self.completion is None:
            return self._format_error_response(f"LiteLLM client initialization failed: {getattr(self, 'init_error', 'Unknown error')}")
            
        try:
            # Map model type to provider prefix
            prefix = {
                "LiteLLM": "ollama/",
                "GroqCloud": "groq/",
                "Gemini": "gemini/",
                "OpenAI": "openai/"
            }.get(self.model_type, "")
            
            model = prefix + self.model_name
            
            # Prepare parameters
            kwargs = {
                "model": model, 
                "messages": messages,
                "temperature": self.temperature
            }
            
            # Add API base if specified
            if self.api_base:
                kwargs["api_base"] = self.api_base
                
            # Add function calling if provided
            if functions:
                kwargs["functions"] = functions
                kwargs["function_call"] = "auto"
            
            # Implement retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.completion(**kwargs)
                    logger.info("LitellmClient.chat succeeded")
                    return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"LiteLLM error (attempt {attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise
                    
        except Exception as e:
            logger.error(f"Error in LitellmClient.chat: {str(e)}")
            logger.error(traceback.format_exc())
            return self._format_error_response(str(e))

def get_llm_client(model_type: str, model_name: str) -> BaseLLMClient:
    """
    Factory to get appropriate LLM client with improved error handling.
    
    Args:
        model_type: The type of model provider (Ollama, OpenAI, etc.)
        model_name: The name of the model to use
        
    Returns:
        An instance of the appropriate LLM client
    """
    logger.info(f"get_llm_client called: model_type={model_type}, model_name={model_name}")
    try:
        # Map model type to client class
        if model_type == "Ollama":
            return OllamaClient(model_name)
        elif model_type == "OpenAI":
            return OpenAIClient(model_name)
        elif model_type == "GroqCloud":
            # Try direct Groq client first, fall back to LiteLLM
            try:
                return GroqClient(model_name)
            except Exception:
                logger.warning("Direct Groq client failed, falling back to LiteLLM")
                return LitellmClient(model_type, model_name)
        elif model_type == "Gemini":
            # Try direct Gemini client first, fall back to LiteLLM
            try:
                return GeminiClient(model_name)
            except Exception:
                logger.warning("Direct Gemini client failed, falling back to LiteLLM")
                return LitellmClient(model_type, model_name)
        elif model_type == "LiteLLM":
            return LitellmClient(model_type, model_name)
        else:
            logger.warning(f"Unknown model_type: {model_type}, attempting to use LiteLLM")
            return LitellmClient("LiteLLM", model_name)
    except Exception as e:
        logger.error(f"Error in get_llm_client for {model_type}/{model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a dummy client that will return error messages
        class ErrorClient(BaseLLMClient):
            def chat(self, messages, functions=None):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": f"Error initializing {model_type} client: {str(e)}. Please check your API keys and try again."
                            }
                        }
                    ],
                    "error": str(e)
                }
        return ErrorClient(model_name)
