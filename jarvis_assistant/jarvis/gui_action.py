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
import logging
import platform
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import ollama
import openai
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Start ------------------------------------------------------------
from .model_settings import Model_Settings as model_settings
from .prompts import system_prompt_basic, system_prompt_function_calling, system_prompt_strawberry_o1
from .db_helper import vectorstore_add_document, vectorstore_add_multi_files, vectorstore_similarity_search_with_score
from .utils import save_api_keys_to_yaml
from .get_model_list import get_ollama_list_models, get_groq_list_models, get_openai_list_models, get_gemini_list_modes

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
        log.error(f"Error adding message: {str(e)}")
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
        log.error(f"Error in adaptive RAG: {str(e)}")
        return "", [], ""

def ollama_pipeline(message_input, history):
    """
    Process a message through the LLM pipeline
    
    Args:
        message_input (str): User's message
        history (list): Chat history
        
    Returns:
        tuple: (result, source_list)
    """
    if not message_input:
        return "I didn't receive a message to respond to.", []
        
    try:
        log.info(f"Processing message: {message_input}")
        print(f"\nprompt: {message_input}")
        
        # Get retrieval context
        retrieval_prompt, source, context_retrieval = get_adaptive_rag(message_input, history)
        print(f"\nContext_retrieval: {context_retrieval}")
        
        result = ""
        
        # Process with Ollama models
        if model_settings.MODEL_TYPE == "Ollama":
            # Use function calling agents if enabled
            if model_settings.FUNCTION_CALLING:
                if model_settings.AGENT_CALLING == "ReWOO":
                    try:
                        # ReWOO agent calling
                        from .rewoo_agent import rewoo_agent
                        response = rewoo_agent(
                            model_settings.MODEL_NAME, 
                            model_settings.SYSTEM_PROMPT, 
                            retrieval_prompt, 
                            message_input
                        )
                        result = response['output']
                    except Exception as e:
                        log.error(f"Error with ReWOO agent: {str(e)}")
                        result = f"I encountered an error with the ReWOO agent: {str(e)}"

                elif model_settings.AGENT_CALLING == "ReACT":
                    try:
                        # ReACT agent calling
                        from .react_agent import react_agent
                        response = react_agent(
                            model_settings.MODEL_NAME, 
                            model_settings.SYSTEM_PROMPT, 
                            context_retrieval, 
                            message_input
                        )
                        result = response['output']
                    except Exception as e:
                        log.error(f"Error with ReACT agent: {str(e)}")
                        result = f"I encountered an error with the ReACT agent: {str(e)}"
            else:
                # Standard LLM completion
                try:
                    llm = ChatOllama(
                        model=model_settings.MODEL_NAME, 
                        temperature=model_settings.TEMPERATURE, 
                        top_k=model_settings.TOP_K, 
                        top_p=model_settings.TOP_P, 
                        max_new_tokens=model_settings.NUM_PREDICT, 
                        repeat_penalty=model_settings.REPEAT_PENALTY
                    )
                    
                    prompt = ChatPromptTemplate.from_template(
                        model_settings.SYSTEM_PROMPT + 
                        retrieval_prompt + 
                        "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: "
                    )
                    
                    # Append /no_think to the prompt
                    prompt += " /no_think"
                    
                    chain = prompt | llm | StrOutputParser()
                    result = chain.invoke({"user": message_input})
                    
                    # Remove <think> and </think> tokens
                    result = result.replace("<think>", "").replace("</think>", "").strip()
                except Exception as e:
                    log.error(f"Error with Ollama completion: {str(e)}")
                    result = f"I encountered an error generating a response: {str(e)}"
        
        # Process with other model types
        else:  # MODEL_TYPE == "LiteLLM", "OpenAI", "GroqCloud", "Gemini"
            try:
                from .llms import llm_completion
                prompt = retrieval_prompt + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
                result = llm_completion(
                    model_settings.MODEL_TYPE, 
                    model_settings.MODEL_NAME, 
                    model_settings.SYSTEM_PROMPT, 
                    prompt
                )
            except Exception as e:
                log.error(f"Error with {model_settings.MODEL_TYPE} completion: {str(e)}")
                result = f"I encountered an error with the {model_settings.MODEL_TYPE} model: {str(e)}"
        
        # Clean up the result
        if "Jarvis (AI):" in result:
            result = result.split("Jarvis (AI):")[1].strip()
        if "**Jarvis (AI)**:" in result:
            result = result.split("**Jarvis (AI)**:")[1].strip()
            
        return result, source
        
    except Exception as e:
        log.error(f"Error in ollama_pipeline: {str(e)}")
        return f"I encountered an error processing your request: {str(e)}", []

def bot(history, chat_input):
    """
    Process a user message and generate a bot response
    
    Args:
        history (list): Chat history
        chat_input (dict): User input containing text and/or files
        
    Returns:
        tuple: (updated_history, empty_input)
    """
    try:
        # Process only if user sent text
        if chat_input.get('text'):
            # Extract the question from the last message
            if history and len(history) > 0:
                question = history[-1]["content"].split(': ', 1)[1] if ": " in history[-1]["content"] else history[-1]["content"]
                
                # Generate response
                answer, source = ollama_pipeline(question, history)
                resp = f"Jarvis (AI): {answer}"
                
                # Add source information if available
                if source:
                    resp += f"\nSource: {source}"
                    
                # Add response to history
                history.append({"role": "assistant", "content": resp})
                
                # Optionally save chat history
                if model_settings.CHAT_HISTORY_SAVING:
                    try:
                        log = f"### HUMAN: {question}\n### ASSISTANT: {answer}"
                        vectorstore_add_document(log, 'chat_history')
                    except Exception as e:
                        log.error(f"Error saving chat history: {str(e)}")
                        
        return history, {"text": ""}
    except Exception as e:
        log.error(f"Error in bot function: {str(e)}")
        history.append({"role": "assistant", "content": f"Jarvis (AI): Sorry, I encountered an error: {str(e)}"})
        return history, {"text": ""}

def btn_save_click(txt_system_prompt):
    """Save the system prompt"""
    model_settings.SYSTEM_PROMPT = txt_system_prompt
    log.info("System prompt updated")
    print("\nsystem_prompt:", model_settings.SYSTEM_PROMPT)

def btn_reset_click(txt_system_prompt):
    """Reset the system prompt to default"""
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    log.info("System prompt reset to default")
    return model_settings.SYSTEM_PROMPT

def radio_device_select(radio_device):
    """Handle device selection"""
    log.info(f"Selected device: {radio_device}")
    print("Selected device:", radio_device)

def slider_num_predict_change(slider_num_predict):
    """Update max tokens setting"""
    model_settings.NUM_PREDICT = slider_num_predict
    log.info(f"Max tokens updated to {slider_num_predict}")
    print("num_predict:", model_settings.NUM_PREDICT)

def slider_temperature_change(slider_temperature):
    """Update temperature setting"""
    model_settings.TEMPERATURE = slider_temperature
    log.info(f"Temperature updated to {slider_temperature}")
    print("temperature:", model_settings.TEMPERATURE)

def slider_top_k_change(slider_top_k):
    """Update top-k setting"""
    model_settings.TOP_K = slider_top_k
    log.info(f"Top-k updated to {slider_top_k}")
    print("top_k:", model_settings.TOP_K)

def slider_top_p_change(slider_top_p):
    """Update top-p setting"""
    model_settings.TOP_P = slider_top_p
    log.info(f"Top-p updated to {slider_top_p}")
    print("top_p:", model_settings.TOP_P)

def slider_retrieval_top_k_change(slider_retrieval_top_k):
    """Update retrieval top-k setting"""
    model_settings.RETRIEVAL_TOP_K = slider_retrieval_top_k
    log.info(f"Retrieval top-k updated to {slider_retrieval_top_k}")
    print("retrieval k:", model_settings.RETRIEVAL_TOP_K)

def slider_retrieval_threshold_change(slider_retrieval_threshold):
    """Update retrieval threshold setting"""
    model_settings.RETRIEVAL_THRESHOLD = slider_retrieval_threshold
    log.info(f"Retrieval threshold updated to {slider_retrieval_threshold}")
    print("retrieval threshold:", model_settings.RETRIEVAL_THRESHOLD)

def btn_key_save_click(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key):
    """Save API keys"""
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
        
        log.info("API keys saved successfully")
        print("\nSave API keys ~> Ok")
    except Exception as e:
        log.error(f"Error saving API keys: {str(e)}")
        print(f"\nError saving API keys: {str(e)}")

def dropdown_model_type_select(dropdown_model_type):
    """Handle model type selection"""
    model_settings.MODEL_TYPE = dropdown_model_type
    log.info(f"Model type changed to {dropdown_model_type}")
    print("\ndropdown_model_type:", model_settings.MODEL_TYPE)

def ollama_dropdown_model_select(dropdown_model):
    """Handle Ollama model selection"""
    model_settings.MODEL_NAME = dropdown_model
    log.info(f"Ollama model changed to {dropdown_model}")
    print("\nSelected model:", model_settings.MODEL_NAME)

def groq_dropdown_model_select(dropdown_model):
    """Handle Groq model selection"""
    model_settings.MODEL_NAME = dropdown_model
    log.info(f"Groq model changed to {dropdown_model}")
    print("\nSelected model:", model_settings.MODEL_NAME)

def openai_dropdown_model_select(dropdown_model):
    """Handle OpenAI model selection"""
    model_settings.MODEL_NAME = dropdown_model
    log.info(f"OpenAI model changed to {dropdown_model}")
    print("\nSelected model:", model_settings.MODEL_NAME)

def gemini_dropdown_model_select(dropdown_model):
    """Handle Gemini model selection"""
    model_settings.MODEL_NAME = dropdown_model
    log.info(f"Gemini model changed to {dropdown_model}")
    print("\nSelected model:", model_settings.MODEL_NAME)

def litellm_dropdown_model_select(dropdown_model):
    """Handle LiteLLM model selection"""
    model_settings.MODEL_NAME = dropdown_model
    log.info(f"LiteLLM model changed to {dropdown_model}")
    print("\nSelected model:", model_settings.MODEL_NAME)

def update_is_retrieval(is_retrieval):
    """Toggle retrieval functionality"""
    model_settings.IS_RETRIEVAL = is_retrieval
    log.info(f"Retrieval documents set to {is_retrieval}")
    print("\nRetrieval documents is:", model_settings.IS_RETRIEVAL)

def update_is_grader(is_grader):
    """Toggle grader functionality"""
    model_settings.IS_GRADER = is_grader
    log.info(f"Grader documents set to {is_grader}")
    print("\nGrader documents is:", model_settings.IS_GRADER)
    
def update_is_web_search(is_web_search):
    """Toggle web search functionality"""
    model_settings.IS_WEB_SEARCH = is_web_search
    log.info(f"Web search set to {is_web_search}")
    print("\nSearch relevant documents is:", model_settings.IS_WEB_SEARCH)

def update_function_calling(function_calling):
    """Toggle function calling and update system prompt"""
    model_settings.FUNCTION_CALLING = function_calling
    
    if function_calling:
        model_settings.SYSTEM_PROMPT = system_prompt_function_calling
        log.info("Enabled function calling with function_calling prompt")
    else:
        model_settings.SYSTEM_PROMPT = system_prompt_basic
        log.info("Disabled function calling, using basic prompt")
        
    print("\nFunction calling is:", model_settings.FUNCTION_CALLING)

def radio_agents_select(radio_agents):
    """Select agent type"""
    model_settings.AGENT_CALLING = radio_agents
    log.info(f"Agent type set to {radio_agents}")
    print("\nAgent calling is:", model_settings.AGENT_CALLING)

def update_chat_saving(chat_history_saving):
    """Toggle chat history saving"""
    model_settings.CHAT_HISTORY_SAVING = chat_history_saving
    log.info(f"Chat history saving set to {chat_history_saving}")
    print("\nSaving chat-history is:", model_settings.CHAT_HISTORY_SAVING)

def btn_basic_prompt_click():
    """Set basic system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    log.info("System prompt set to basic")
    print("\nYou have been selected system_prompt_basic.")
    return model_settings.SYSTEM_PROMPT

def btn_function_calling_prompt_click():
    """Set function calling system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_function_calling
    log.info("System prompt set to function_calling")
    print("\nYou have been selected system_prompt_function_calling.")
    return model_settings.SYSTEM_PROMPT

def btn_strawberry_o1_prompt_click():
    """Set strawberry_o1 system prompt"""
    model_settings.SYSTEM_PROMPT = system_prompt_strawberry_o1
    log.info("System prompt set to strawberry_o1")
    print("\nYou have been selected system_prompt_strawberry_o1.")
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
        log.info(f"Created new workspace with ID {max_id}")
        return workspace_list, workspace
    except Exception as e:
        log.error(f"Error creating new workspace: {str(e)}")
        return workspace_list, workspace_list[0] if workspace_list else None

def btn_save_workspace_click(workspace_list):
    """Save all workspaces to files"""
    try:
        # Create folder if it doesn't exist
        folder_path = "chat_workspaces"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            log.info(f"Created directory: {folder_path}")

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
                    
            log.info(f"Saved workspace to {file_path}")
            print(f"\nSave workspace to ~> {file_path}")
            
    except Exception as e:
        log.error(f"Error saving workspaces: {str(e)}")
        print(f"\nError saving workspaces: {str(e)}")
