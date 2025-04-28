# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 18 December 2024 - 10 PM

# Import needed packages ------------------------------------------------------------
import os, sys, re
import time
from datetime import datetime

from tqdm import tqdm
import ollama, openai
# from litellm import completion
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


# Start ------------------------------------------------------------
from .model_settings import Model_Settings
model_settings = Model_Settings()

from .prompts import system_prompt_basic, system_prompt_function_calling, system_prompt_strawberry_o1
model_settings.SYSTEM_PROMPT = system_prompt_basic

from .db_helper import vectorstore_add_document, vectorstore_add_multi_files, vectorstore_similarity_search_with_score

def add_message(history, message):
    # Handle file uploads and append assistant feedback
    if message.get("files"):
        upload_feedback = vectorstore_add_multi_files(message["files"])
        history.append({"role":"assistant", "content": f"Jarvis (AI): {upload_feedback}"})
    # Append user message
    if message.get("text"):
        history.append({"role":"user", "content": f"Human: {message['text']}"})
    return history

# from jarvis.prompts import system_prompt

# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/
def get_adaptive_rag(message_input, history):
    retrieval_prompt = ""
    msg_history = ""
    # Build last few exchanges
    for msg in history[-4:-1]:
        if not isinstance(msg, dict) or 'role' not in msg:
            continue
        if msg['role'] == 'user':
            msg_history += f"human: {msg['content']}\n"
        elif msg['role'] == 'assistant':
            msg_history += f"assistant: {msg['content']}\n"
    if msg_history:
        msg_history = "\n\nCHAT HISTORY:\n" + msg_history
        # print("\nmsg_history:",msg_history)
        retrieval_prompt += msg_history

    context_retrieval = ""
    source = []
    if model_settings.IS_RETRIEVAL:
        context_retrieval, source = vectorstore_similarity_search_with_score(message_input, model_settings.RETRIEVAL_TOP_K, model_settings.RETRIEVAL_THRESHOLD)
        if context_retrieval:
            context_retrieval = "\n\nRETRIEVAL DOCUMENT:\n" + re.sub(r"[\"\'\{\}\x08]+"," ",context_retrieval)
            retrieval_prompt += context_retrieval

    # print("\nretrieval_prompt:",retrieval_prompt)
    return retrieval_prompt, source, context_retrieval

def ollama_pipeline(message_input, history):
    if message_input:
        print("\nprompt:",message_input)
        source = []
        retrieval_prompt, source, context_retrieval = get_adaptive_rag(message_input, history)

        # print("\nretrieval_prompt:",retrieval_prompt)
        print("\ncontext_retrieval:",context_retrieval)
        # print("\nollama_pipeline source:",source)
        
        result = ""
        if model_settings.MODEL_TYPE == "Ollama":

            # https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
            if model_settings.FUNCTION_CALLING:
                if model_settings.AGENT_CALLING == "ReWOO":
                    # ReWOO agent calling
                    from .rewoo_agent import rewoo_agent
                    response = rewoo_agent(model_settings.MODEL_NAME, model_settings.SYSTEM_PROMPT, context_retrieval, message_input)
                    result = response['output']

                if model_settings.AGENT_CALLING == "ReACT":
                    # ReACT agent calling
                    from .react_agent import react_agent
                    response = react_agent(model_settings.MODEL_NAME, model_settings.SYSTEM_PROMPT, retrieval_prompt, message_input)
                    result = response['output']
            else:
                llm = ChatOllama(model=model_settings.MODEL_NAME, temperature=model_settings.TEMPERATURE, top_k=model_settings.TOP_K, top_p=model_settings.TOP_P, max_new_tokens=model_settings.NUM_PREDICT, repeat_penalty=model_settings.REPEAT_PENALTY)
                prompt = ChatPromptTemplate.from_template(model_settings.SYSTEM_PROMPT + retrieval_prompt + "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: ")
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"user": message_input})
        
        else: # MODEL_TYPE == "LiteLLM" , "OpenAI" , "GroqCloud" , "Gemini"
            from .llms import llm_completion
            prompt = retrieval_prompt + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            result = llm_completion(model_settings.MODEL_TYPE, model_settings.MODEL_NAME, model_settings.SYSTEM_PROMPT, prompt)

        return result, source

def bot(history, chat_input):
    # Process only if user sent text
    if chat_input.get('text'):
        question = history[-1]["content"].split(': ',1)[1] if ": " in history[-1]["content"] else history[-1]["content"]
        answer, source = ollama_pipeline(question, history)
        resp = f"Jarvis (AI): {answer}"
        if source:
            resp += f"\nSource: {source}"
        history.append({"role":"assistant", "content": resp})
        # Optionally save chat history
        if model_settings.CHAT_HISTORY_SAVING:
            log = f"### HUMAN: {question}\n### ASSISTANT: {answer}"
            vectorstore_add_document(log, 'chat_history')
    return history, {"text": ""}

def btn_save_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = txt_system_prompt
    print("\nsystem_prompt:",model_settings.SYSTEM_PROMPT)

def btn_reset_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    return model_settings.SYSTEM_PROMPT

def radio_device_select(radio_device):
    print("Selected device:",radio_device)

def slider_num_predict_change(slider_num_predict):
    model_settings.NUM_PREDICT = slider_num_predict
    print("num_predict:",model_settings.NUM_PREDICT)

def slider_temperature_change(slider_temperature):
    model_settings.TEMPERATURE = slider_temperature
    print("temperature:",model_settings.TEMPERATURE)

def slider_top_k_change(slider_top_k):
    model_settings.TOP_K = slider_top_k
    print("top_k:",model_settings.TOP_K)

def slider_top_p_change(slider_top_p):
    model_settings.TOP_P = slider_top_p
    print("top_p:",model_settings.TOP_P)

def slider_retrieval_top_k_change(slider_retrieval_top_k):
    model_settings.RETRIEVAL_TOP_K = slider_retrieval_top_k
    print("retrieval k:",model_settings.RETRIEVAL_TOP_K)

def slider_retrieval_threshold_change(slider_retrieval_threshold):
    model_settings.RETRIEVAL_THRESHOLD = slider_retrieval_threshold
    print("retrieval threshold:",model_settings.RETRIEVAL_THRESHOLD)

from .utils import save_api_keys_to_yaml

def btn_key_save_click(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key):
    model_settings.GROQ_API_KEY = txt_groq_api_key
    model_settings.OPENAI_API_KEY = txt_openai_api_key
    model_settings.GEMINI_API_KEY = txt_gemini_api_key

    os.environ['GROQ_API_KEY'] = txt_groq_api_key
    os.environ["OPENAI_API_KEY"] = txt_openai_api_key
    os.environ["GEMINI_API_KEY"] = txt_gemini_api_key

    save_api_keys_to_yaml(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key)

    print("\nSave API keys ~> Ok")

from .get_model_list import get_ollama_list_models, get_groq_list_models, get_openai_list_models, get_gemini_list_modes

def dropdown_model_type_select(dropdown_model_type):
    model_settings.MODEL_TYPE = dropdown_model_type
    print("\ndropdown_model_type:",model_settings.MODEL_TYPE)

def ollama_dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def groq_dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def openai_dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def gemini_dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def litellm_dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def update_is_retrieval(is_retrieval):
    model_settings.IS_RETRIEVAL = is_retrieval
    print("\nRetrieval documents is:",model_settings.IS_RETRIEVAL)

def update_function_calling(function_calling):
    if function_calling == True:
        model_settings.FUNCTION_CALLING = function_calling
        model_settings.SYSTEM_PROMPT = system_prompt_function_calling
    else:
        model_settings.FUNCTION_CALLING = function_calling
        model_settings.SYSTEM_PROMPT = system_prompt_basic
        
    print("\nFunction calling is:",model_settings.FUNCTION_CALLING)

def radio_agents_select(radio_agents):
    model_settings.AGENT_CALLING = radio_agents
    print("\nAgent calling is:",model_settings.AGENT_CALLING)

def update_chat_saving(chat_history_saving):
    model_settings.CHAT_HISTORY_SAVING = chat_history_saving
    print("\nSaving chat-history is:",model_settings.CHAT_HISTORY_SAVING)

def btn_basic_prompt_click():
    model_settings.SYSTEM_PROMPT = system_prompt_basic
    print("\nYou have been selected system_prompt_basic.")
    return model_settings.SYSTEM_PROMPT

def btn_function_calling_prompt_click():
    model_settings.SYSTEM_PROMPT = system_prompt_function_calling
    print("\nYou have been selected system_prompt_function_calling.")
    return model_settings.SYSTEM_PROMPT

def btn_strawberry_o1_prompt_click():
    model_settings.SYSTEM_PROMPT = system_prompt_strawberry_o1
    print("\nYou have been selected system_prompt_strawberry_o1.")
    return model_settings.SYSTEM_PROMPT

def btn_create_new_workspace_click(workspace_list):
    max_id = 0
    for wp in workspace_list:
        if wp["id"] >= max_id:
            max_id = wp["id"] + 1
    workspace = {"id":max_id, "name":"New workspace "+str(max_id), "history":[{"role":"user", "content": "Human: Hello"}, {"role":"assistant", "content": f"Jarvis (AI): Hi, my name Jarvis. I am your assistant. How may I help you today?  [v{max_id}]"}]}
    workspace_list.insert(0, workspace)
    return workspace_list, workspace

import platform # Get system information
def btn_save_workspace_click(workspace_list):
    my_platform = platform.system() #  "Linux", "Windows", or "Darwin" (Mac)
    folder_path = "chat_workspaces"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # current date and time
    now = datetime.now()
    time_now = now.strftime("%Y-%m-%d_%H-%M-%S")

    for wp in workspace_list:
        file_name = str(time_now)+"_"+str(wp["id"])+"_"+str(wp["name"])+'.txt'

        file_path = ""
        if my_platform == "Windows":
            file_path = folder_path + "\\" + file_name
        elif my_platform == "Darwin":
           file_path = folder_path + "/" + file_name
        else:
            file_path = folder_path + "/" + file_name

        with open(file_path, 'w', encoding="utf-8") as f:
            for chat in wp["history"]:
                f.write(str(chat["content"])+"\n\n")
        print("\nsave workspace to ~>",file_path)
