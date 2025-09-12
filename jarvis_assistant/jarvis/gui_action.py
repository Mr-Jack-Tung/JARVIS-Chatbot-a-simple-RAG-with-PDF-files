# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

# Import needed packages ------------------------------------------------------------
import os
import sys
import re
import time
import platform
from datetime import datetime
from loguru import logger
from pathlib import Path

from tqdm import tqdm
import ollama
import openai
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Start ------------------------------------------------------------
from .model_settings import Model_Settings as model_settings
from .prompts import system_prompt_basic, system_prompt_function_calling, system_prompt_strawberry_o1
from .db_helper import vectorstore_add_document, vectorstore_add_multi_files, vectorstore_similarity_search_with_score
from .utils import save_api_keys_to_yaml
from .get_model_list import get_ollama_list_models, get_groq_list_models, get_openai_list_models, get_gemini_list_modes
from .llms import llm_stream_completion

# Initialize system prompt
model_settings.SYSTEM_PROMPT = system_prompt_basic

def add_message(history, message):
    """
    Add a message to the chat history
    
    Args:
        history (list): Current chat history
        message (dict): Message to add, can contain text and/or files
        
    Returns:
        list: Updated chat history
    """
    try:
        # Handle file uploads and append assistant feedback
        if message.get("files"):
            upload_feedback = vectorstore_add_multi_files(message["files"])
            history.append({"role": "assistant", "content": f"{upload_feedback}"})
            
        # Append user message
        if message.get("text"):
            history.append({"role": "user", "content": f"Human: {message['text']}"})
            
        return history
    except Exception as e:
        logger.error(f"Error adding message: {str(e)}")
        # Add error message to history
        history.append({"role": "assistant", "content": f"Jarvis (AI): Sorry, I encountered an error processing your message: {str(e)}"})
        return history

def get_adaptive_rag(message_input, history):
    """
    Get retrieval-augmented generation context for the current message
    
    Args:
        message_input (str): User's message
        history (list): Chat history
        
    Returns:
        tuple: (retrieval_prompt, source_list, context_retrieval)
    """
    try:
        retrieval_prompt = ""
        msg_history = ""
        
        # Build last few exchanges from history
        if history and len(history) > 0:
            for msg in history[-4:-1]:
                if not isinstance(msg, dict) or 'role' not in msg:
                    continue
                if msg['role'] == 'user':
                    msg_history += f"human: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    msg_history += f"assistant: {msg['content']}\n"
                    
            if msg_history:
                msg_history = "\n\nCHAT HISTORY:\n" + msg_history
                retrieval_prompt += msg_history

        # Get retrieval context if enabled
        context_retrieval = ""
        source = []
        if model_settings.IS_RETRIEVAL:
            context_retrieval, source = vectorstore_similarity_search_with_score(
                message_input, 
                model_settings.RETRIEVAL_TOP_K, 
                model_settings.RETRIEVAL_THRESHOLD
            )
            
            if context_retrieval:
                # Clean up the retrieval context
                context_retrieval = "\n\nRETRIEVAL DOCUMENT:\n" + re.sub(r"[\"\'\{\}\x08]+"," ", context_retrieval)
                retrieval_prompt += context_retrieval

        return retrieval_prompt, source, context_retrieval
    except Exception as e:
        logger.error(f"Error in adaptive RAG: {str(e)}")
        return "", [], ""

def _execute_agent_pipeline(message_input, retrieval_prompt, context_retrieval):
    """Executes the agent-based pipeline (ReWOO or ReACT)."""
    try:
        if model_settings.AGENT_CALLING == "ReWOO":
            from .rewoo_agent import rewoo_agent
            response = rewoo_agent(
                model_settings.MODEL_NAME,
                model_settings.SYSTEM_PROMPT,
                retrieval_prompt,
                message_input
            )
            return response.get('output', '')
        elif model_settings.AGENT_CALLING == "ReACT":
            from .react_agent import react_agent
            response = react_agent(
                model_settings.MODEL_NAME,
                model_settings.SYSTEM_PROMPT,
                context_retrieval,
                message_input
            )
            return response.get('output', '')
    except Exception as e:
        logger.error(f"Error with {model_settings.AGENT_CALLING} agent: {str(e)}")
        return f"I encountered an error with the {model_settings.AGENT_CALLING} agent: {str(e)}"
    return ""

def _execute_standard_pipeline(message_input, retrieval_prompt):
    """Executes the standard RAG pipeline for all model types."""
    try:
        if model_settings.MODEL_TYPE == "Ollama":
            llm = ChatOllama(
                model=model_settings.MODEL_NAME,
                temperature=model_settings.TEMPERATURE,
                top_k=model_settings.TOP_K,
                top_p=model_settings.TOP_P,
                num_predict=model_settings.NUM_PREDICT,
                repeat_penalty=model_settings.REPEAT_PENALTY
            )
            prompt_template = model_settings.SYSTEM_PROMPT + retrieval_prompt + "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: "
            if not model_settings.IS_THINKING:
                prompt_template += " /no_think"
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"user": message_input})
            return result.replace("<think>", "").replace("</think>", "").strip()
        else:  # Handles "LiteLLM", "OpenAI", "GroqCloud", "Gemini"
            from .llms import llm_completion
            prompt = retrieval_prompt + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            return llm_completion(
                model_settings.MODEL_TYPE,
                model_settings.MODEL_NAME,
                model_settings.SYSTEM_PROMPT,
                prompt
            )
    except Exception as e:
        logger.error(f"Error in standard pipeline with {model_settings.MODEL_TYPE}: {str(e)}")
        return f"I encountered an error generating a response with {model_settings.MODEL_TYPE}: {str(e)}"

def generate_response_non_stream(message_input, history):
    """
    Processes a message through the non-streaming pipeline (agents).
    
    Args:
        message_input (str): User's message
        history (list): Chat history
        
    Returns:
        tuple: (result, source_list)
    """
    if not message_input:
        return "I didn't receive a message to respond to.", []
        
    try:
        logger.info(f"Processing message: {message_input}")
        
        retrieval_prompt, source, context_retrieval = get_adaptive_rag(message_input, history)
        if context_retrieval:
            logger.info(f"Context retrieval found {len(source)} sources.")

        result = ""
        if model_settings.MODEL_TYPE == "Ollama" and model_settings.FUNCTION_CALLING:
            result = _execute_agent_pipeline(message_input, retrieval_prompt, context_retrieval)
        else:
            result = _execute_standard_pipeline(message_input, retrieval_prompt)
            
        # Clean up the result
        if "Jarvis (AI):" in result:
            result = result.split("Jarvis (AI):")[1].strip()
        if "**Jarvis (AI)**:" in result:
            result = result.split("**Jarvis (AI)**:")[1].strip()
            
        return result, source
        
    except Exception as e:
        logger.error(f"Error in generate_response_non_stream: {str(e)}")
        return f"I encountered an error processing your request: {str(e)}", []

def generate_response_stream_pipeline(message_input, history):
    """
    Processes a message through the streaming pipeline.
    
    Args:
        message_input (str): User's message
        history (list): Chat history
        
    Yields:
        str: Chunks of the response
    """
    try:
        logger.info(f"Streaming message: {message_input}")
        
        retrieval_prompt, source, context_retrieval = get_adaptive_rag(message_input, history)
        if context_retrieval:
            logger.info(f"Context retrieval found {len(source)} sources.")

        # For now, streaming is only implemented for the standard pipeline
        if model_settings.MODEL_TYPE == "Ollama" and model_settings.FUNCTION_CALLING:
             # Agent pipeline (non-streaming for now)
            answer, _ = generate_response_non_stream(message_input, history)
            yield answer
        else:
            # Standard pipeline (streaming)
            prompt = retrieval_prompt + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            for chunk in llm_stream_completion(
                model_settings.MODEL_TYPE,
                model_settings.MODEL_NAME,
                model_settings.SYSTEM_PROMPT,
                prompt
            ):
                yield chunk
        
        # Yield source at the end
        if source:
            yield f"\nSource: {source}"

    except Exception as e:
        logger.exception("Error in generate_response_stream_pipeline")
        yield f"I encountered an error processing your request: {str(e)}"

def bot(history, chat_input):
    """
    Process a user message and generate a bot response stream.
    This function is a generator that yields updates to the chat history.
    """
    try:
        # 1. Add user message and file uploads to history (from add_message)
        if chat_input.get("files"):
            upload_feedback = vectorstore_add_multi_files(chat_input["files"])
            history.append({"role": "assistant", "content": f"{upload_feedback}"})
            yield history, {"text": ""} # Update UI with upload feedback

        question = chat_input.get("text", "").strip()
        if not question:
            # If only files were uploaded, no need to run the bot
            yield history, {"text": ""}
            return

        history.append({"role": "user", "content": f"Human: {question}"})
        
        # 2. Add assistant placeholder and yield initial state
        history.append({"role": "assistant", "content": ""})
        yield history, {"text": ""}

        # 3. Stream the response
        full_answer = ""
        # The history passed to the pipeline should not include the empty assistant message
        for chunk in generate_response_stream_pipeline(question, history[:-1]):
            full_answer += chunk
            history[-1]["content"] = f"Jarvis (AI): {full_answer}"
            yield history, {"text": ""}

        # 4. Optionally save final chat history
        if model_settings.CHAT_HISTORY_SAVING:
            try:
                # Save human message
                vectorstore_add_document(f"### HUMAN: {question}", 'chat_history')
                # Save assistant response
                vectorstore_add_document(f"### ASSISTANT: {full_answer}", 'chat_history')
            except Exception as e:
                logger.error(f"Error saving chat history: {str(e)}")
                
    except Exception as e:
        logger.exception("Error in bot function")
        # Attempt to add error message to history
        if history:
            history.append({"role": "assistant", "content": f"Jarvis (AI): Sorry, I encountered an error: {str(e)}"})
        yield history, {"text": ""}

def btn_save_click(txt_system_prompt):
    """Save the system prompt"""
    if model_settings.SYSTEM_PROMPT != txt_system_prompt:
        model_settings.SYSTEM_PROMPT = txt_system_prompt
        logger.info("System prompt updated.")

def btn_reset_click(txt_system_prompt):
    """Reset the system prompt to default"""
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    logger.info("System prompt reset to default.")
    return model_settings.SYSTEM_PROMPT

def radio_device_select(radio_device):
    """Handle device selection"""
    logger.info(f"Selected device: {radio_device}")

def slider_num_predict_change(slider_num_predict):
    """Update max tokens setting"""
    if model_settings.NUM_PREDICT != slider_num_predict:
        model_settings.NUM_PREDICT = slider_num_predict
        logger.info(f"Max new tokens updated to: {model_settings.NUM_PREDICT}")

def slider_temperature_change(slider_temperature):
    """Update temperature setting"""
    if model_settings.TEMPERATURE != slider_temperature:
        model_settings.TEMPERATURE = slider_temperature
        logger.info(f"Temperature updated to: {model_settings.TEMPERATURE}")

def slider_top_k_change(slider_top_k):
    """Update top-k setting"""
    if model_settings.TOP_K != slider_top_k:
        model_settings.TOP_K = slider_top_k
        logger.info(f"Top-k updated to: {model_settings.TOP_K}")

def slider_top_p_change(slider_top_p):
    """Update top-p setting"""
    if model_settings.TOP_P != slider_top_p:
        model_settings.TOP_P = slider_top_p
        logger.info(f"Top-p updated to: {model_settings.TOP_P}")

def slider_retrieval_top_k_change(slider_retrieval_top_k):
    """Update retrieval top-k setting"""
    if model_settings.RETRIEVAL_TOP_K != slider_retrieval_top_k:
        model_settings.RETRIEVAL_TOP_K = slider_retrieval_top_k
        logger.info(f"Retrieval top-k updated to: {model_settings.RETRIEVAL_TOP_K}")

def slider_retrieval_threshold_change(slider_retrieval_threshold):
    """Update retrieval threshold setting"""
    if model_settings.RETRIEVAL_THRESHOLD != slider_retrieval_threshold:
        model_settings.RETRIEVAL_THRESHOLD = slider_retrieval_threshold
        logger.info(f"Retrieval threshold updated to: {model_settings.RETRIEVAL_THRESHOLD}")

def btn_key_save_click(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key):
    """Save API keys and return them to update the UI."""
    try:
        # Update model settings
        model_settings.GROQ_API_KEY = txt_groq_api_key
        model_settings.OPENAI_API_KEY = txt_openai_api_key
        model_settings.GEMINI_API_KEY = txt_gemini_api_key

        # Set environment variables
        os.environ['GROQ_API_KEY'] = txt_groq_api_key
        os.environ["OPENAI_API_KEY"] = txt_openai_api_key
        os.environ["GEMINI_API_KEY"] = txt_gemini_api_key

        # Save to YAML file
        save_api_keys_to_yaml(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key)
        
        logger.info("API keys saved successfully.")
        return txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key
    except Exception as e:
        logger.error(f"Error saving API keys: {str(e)}")
        return "", "", ""

def dropdown_model_type_select(dropdown_model_type):
    """Handle model type selection"""
    if model_settings.MODEL_TYPE != dropdown_model_type:
        model_settings.MODEL_TYPE = dropdown_model_type
        logger.info(f"Model type changed to: {model_settings.MODEL_TYPE}")

def _update_model_selection(model_name, model_type):
    """Generic function to handle model selection."""
    if model_settings.MODEL_NAME != model_name:
        model_settings.MODEL_NAME = model_name
        logger.info(f"{model_type} model changed to: {model_name}")

def ollama_dropdown_model_select(dropdown_model):
    """Handle Ollama model selection"""
    _update_model_selection(dropdown_model, "Ollama")

def groq_dropdown_model_select(dropdown_model):
    """Handle Groq model selection"""
    _update_model_selection(dropdown_model, "Groq")

def openai_dropdown_model_select(dropdown_model):
    """Handle OpenAI model selection"""
    _update_model_selection(dropdown_model, "OpenAI")

def gemini_dropdown_model_select(dropdown_model):
    """Handle Gemini model selection"""
    _update_model_selection(dropdown_model, "Gemini")

def litellm_dropdown_model_select(dropdown_model):
    """Handle LiteLLM model selection"""
    _update_model_selection(dropdown_model, "LiteLLM")

def update_is_retrieval(is_retrieval):
    """Toggle retrieval functionality"""
    if model_settings.IS_RETRIEVAL != is_retrieval:
        model_settings.IS_RETRIEVAL = is_retrieval
        logger.info(f"Retrieval documents set to: {is_retrieval}")

def update_is_grader(is_grader):
    """Toggle grader functionality"""
    if model_settings.IS_GRADER != is_grader:
        model_settings.IS_GRADER = is_grader
        logger.info(f"Grader documents set to: {is_grader}")
    
def update_is_web_search(is_web_search):
    """Toggle web search functionality"""
    if model_settings.IS_WEB_SEARCH != is_web_search:
        model_settings.IS_WEB_SEARCH = is_web_search
        logger.info(f"Web search set to: {is_web_search}")

def update_is_thinking(is_thinking):
    """Toggle thinking functionality"""
    if model_settings.IS_THINKING != is_thinking:
        model_settings.IS_THINKING = is_thinking
        logger.info(f"Thinking process set to: {is_thinking}")

def update_show_thinking(show_thinking):
    """Toggle showing thinking functionality"""
    if model_settings.SHOW_THINKING != show_thinking:
        model_settings.SHOW_THINKING = show_thinking
        logger.info(f"Show thinking process set to: {show_thinking}")

def update_function_calling(function_calling):
    """Toggle function calling and update system prompt"""
    if model_settings.FUNCTION_CALLING != function_calling:
        model_settings.FUNCTION_CALLING = function_calling
        
        if function_calling:
            model_settings.SYSTEM_PROMPT = system_prompt_function_calling
            logger.info("Enabled function calling with function_calling prompt.")
        else:
            model_settings.SYSTEM_PROMPT = system_prompt_basic
            logger.info("Disabled function calling, using basic prompt.")
            
        logger.info(f"Function calling is: {model_settings.FUNCTION_CALLING}")

def radio_agents_select(radio_agents):
    """Select agent type"""
    if model_settings.AGENT_CALLING != radio_agents:
        model_settings.AGENT_CALLING = radio_agents
        logger.info(f"Agent type set to: {radio_agents}")

def update_chat_saving(chat_history_saving):
    """Toggle chat history saving"""
    if model_settings.CHAT_HISTORY_SAVING != chat_history_saving:
        model_settings.CHAT_HISTORY_SAVING = chat_history_saving
        logger.info(f"Chat history saving set to: {chat_history_saving}")

def btn_basic_prompt_click():
    """Set basic system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    logger.info("System prompt set to basic.")
    return model_settings.SYSTEM_PROMPT

def btn_function_calling_prompt_click():
    """Set function calling system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_function_calling
    logger.info("System prompt set to function_calling.")
    return model_settings.SYSTEM_PROMPT

def btn_strawberry_o1_prompt_click():
    """Set strawberry_o1 system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_strawberry_o1
    logger.info("System prompt set to strawberry_o1.")
    return model_settings.SYSTEM_PROMPT

def btn_create_new_workspace_click(workspace_list):
    """Create a new workspace"""
    try:
        # Find the next available ID
        max_id = 0
        for wp in workspace_list:
            if wp["id"] >= max_id:
                max_id = wp["id"] + 1
                
        # Create new workspace
        workspace = {
            "id": max_id, 
            "name": f"New workspace {max_id}", 
            "history": [
                {"role": "user", "content": "Human: Hello"}, 
                {"role": "assistant", "content": f"Jarvis (AI): Hi, my name Jarvis. I am your assistant. How may I help you today? [v{max_id}]"}
            ]
        }
        
        workspace_list.insert(0, workspace)
        logger.info(f"Created new workspace with ID: {max_id}")
        return workspace_list, workspace
    except Exception as e:
        logger.error(f"Error creating new workspace: {str(e)}")
        return workspace_list, workspace_list[0] if workspace_list else None

def btn_save_workspace_click(workspace_list):
    """Save all workspaces to files"""
    try:
        # Create folder if it doesn't exist
        folder_path = "chat_workspaces"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logger.info(f"Created directory: {folder_path}")

        # Get current date and time
        now = datetime.now()
        time_now = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Save each workspace
        for wp in workspace_list:
            # Create filename
            file_name = f"{time_now}_{wp['id']}_{wp['name']}.txt"
            
            # Create path using pathlib for cross-platform compatibility
            file_path = Path(folder_path) / file_name
            
            # Write workspace content
            with open(file_path, 'w', encoding="utf-8") as f:
                for chat in wp["history"]:
                    f.write(str(chat["content"]) + "\n\n")
                    
            logger.info(f"Saved workspace to: {file_path}")
            
    except Exception as e:
        logger.error(f"Error saving workspaces: {str(e)}")
