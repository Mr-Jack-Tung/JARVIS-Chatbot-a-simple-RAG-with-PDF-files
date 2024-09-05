# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 05 December 2024 - 09 PM

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
from jarvis.model_settings import Model_Settings
model_settings = Model_Settings()

from jarvis.db_helper import vectorstore_add_document, vectorstore_add_multi_files, vectorstore_similarity_search_with_score

def add_message(history, message):
    upload_files = ""
    if message["files"]:
        path_files = message["files"]
        print("\n")
        upload_files += vectorstore_add_multi_files(path_files)
    if upload_files:
        print("\nUpload files:\n",upload_files)
    
    if len(history)<1:
        history.append(["**human**: Hello", "**Jarvis (AI)**: Hi, my name Jarvis. I am your assistant. How may I help you today?"])
    if message["text"]:
        dt_string = datetime.now().strftime("%H.%M")
        history.append(("(" + dt_string + ") **human**: " + message["text"], ""))
    return history


# https://github.com/Mr-Jack-Tung/Ollama-Mistral-with-Langchain-RAG-Agent-and-Custom-tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Ollama

from jarvis.tools import get_all_tools
from jarvis.prompts import system_prompt


# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/
def get_adaptive_rag(message_input, history):
    retrieval_prompt = ""
    msg_history = ""
    for msg in history[-4:-1]:
        if msg:
            msg_history += "human:" + str(msg[0]).split("**:")[1] + "\n" + "assistant:" + str(msg[1]).split("**:")[1]  + "\n"
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
    return retrieval_prompt, source

# from langchain import hub
# prompt = hub.pull("hwchase17/react-chat")

def ollama_pipeline(message_input, history):
    if message_input:
        print("\nprompt:",message_input)
        source = []
        retrieval_prompt, source = get_adaptive_rag(message_input, history)

        # print("ollama_pipeline source:",source)
        
        result = ""
        if model_settings.MODEL_TYPE == "Ollama":

            # https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
            if model_settings.FUNCTION_CALLING:
                model_local = Ollama(model=model_settings.MODEL_NAME)
                tools = get_all_tools()
                prompt = PromptTemplate(input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'], metadata={'lc_hub_owner': 'jarvis_assistant', 'lc_hub_repo': 'react-chat', 'lc_hub_commit_hash': ''}, template=system_prompt + retrieval_prompt + '\n\nOverall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}')
                agent = create_react_agent(model_local, tools, prompt)
                chat_history_memory = ConversationBufferWindowMemory(k=3, memory_key='chat_history', input_key='input', ouput_key='output')
                agent_executor = AgentExecutor(
                    agent=agent, 
                    tools=tools, 
                    memory=chat_history_memory,
                    verbose=True, # ~> Speech out the thinking
                    handle_parsing_errors=True,
                    )

                response = agent_executor.invoke({"input": "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)})
                result = response['output']
                
            else:
                llm = ChatOllama(model=model_settings.MODEL_NAME, temperature=model_settings.TEMPERATURE, top_k=model_settings.TOP_K, top_p=model_settings.TOP_P, max_new_tokens=model_settings.NUM_PREDICT, repeat_penalty=model_settings.REPEAT_PENALTY)
                prompt = ChatPromptTemplate.from_template(system_prompt + retrieval_prompt + "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: ")
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"user": message_input})
        
        else: # MODEL_TYPE == "LiteLLM" , "OpenAI" , "GroqCloud" , "Gemini"
            from jarvis.llms import llm_completion
            prompt = retrieval_prompt + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            result = llm_completion(model_settings.MODEL_TYPE, model_settings.MODEL_NAME, system_prompt, prompt)

        return result, source

def bot(history, chat_input):
    if chat_input['text']:
        question = str(history[-1][0]).split("**human**: ")[1]
        source = []
        s_time = time.time()
        answer, source = ollama_pipeline(question, history)
        e_time = time.time()
        
        print("\nprompt:",question)
        print("\n{0:.2f}s ~> Answer:".format(e_time-s_time),answer)
        dt_string = datetime.now().strftime("%H.%M")
        response = "(" + dt_string + ") **Jarvis (AI)**: " + str(answer)
        if source:
            response += "<br>**Source:** " + str(source)
        history[-1][1] = ""
        history[-1][1] = response
        
        response2db = str("### HUMAN: "+question+"\n"+"### ASSISTANT: "+answer)
        vectorstore_add_document(response2db, 'chat_history')
        
    return history, {"text": ""}

def btn_save_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = txt_system_prompt
    print("\nsystem_prompt:",model_settings.SYSTEM_PROMPT)

def btn_reset_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = system_prompt
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

from jarvis.utils import save_api_keys_to_yaml

def btn_key_save_click(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key):
    model_settings.GROQ_API_KEY = txt_groq_api_key
    model_settings.OPENAI_API_KEY = txt_openai_api_key
    model_settings.GEMINI_API_KEY = txt_gemini_api_key

    os.environ['GROQ_API_KEY'] = txt_groq_api_key
    os.environ["OPENAI_API_KEY"] = txt_openai_api_key
    os.environ["GEMINI_API_KEY"] = txt_gemini_api_key

    save_api_keys_to_yaml(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key)

    print("\nSave API keys ~> Ok")

from jarvis.get_model_list import get_ollama_list_models, get_groq_list_models, get_openai_list_models, get_gemini_list_modes

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

def update_function_calling(function_calling):
    model_settings.FUNCTION_CALLING = function_calling

def btn_create_new_workspace_click(workspace_list):
    max_id = 0
    for wp in workspace_list:
        if wp["id"] >= max_id:
            max_id = wp["id"] + 1
    workspace = {"id":max_id, "name":"New workspace "+str(max_id), "history":[["**human**: Hello", "**Jarvis (AI)**: Hi, my name Jarvis. I am your assistant. How may I help you today?  [v{0}]".format(max_id)]]}
    workspace_list.insert(0, workspace)
    return workspace_list, workspace

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
                f.write(str(chat[0])+"\n"+str(chat[1])+"\n\n")
        print("\nsave workspace to ~>",file_path)