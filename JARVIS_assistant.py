# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.2
# Date: 15 July 2024 - 01.30 AM

import os

print("\npip install -qU tqdm pypdf chromadb tiktoken")
os.system("pip install -qU tqdm pypdf chromadb tiktoken")

print("\npip install -qU langchain-chroma")
os.system("pip install -qU langchain-chroma")

print("\npip install -qU gradio langchain langchain_community")
os.system("pip install -qU gradio langchain langchain_community")

print("\npip install -qU ollama litellm litellm[proxy]")
os.system("pip install -qU ollama litellm litellm[proxy]")

print("\npip install -qU openai groq google-generativeai")
os.system("pip install -qU openai groq google-generativeai")

import ollama

print("\nollama pull chroma/all-minilm-l6-v2-f32")
ollama.pull('chroma/all-minilm-l6-v2-f32')

print("\nollama pull qwen2\n")
ollama.pull('qwen2')

import os, sys, re, time
from datetime import datetime
from typing import Iterable
from tqdm import tqdm
from time import sleep

import requests
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

import ollama, openai
from litellm import completion

from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

class Model_Settings:
    def __init__(self):
        self.MODEL_TYPE = "Ollama"
        self.MODEL_NAME = 'qwen2:latest'
        self.NUM_PREDICT = 2048
        self.TEMPERATURE = 0
        self.TOP_K = 100
        self.TOP_P = 1
        self.REPEAT_PENALTY = 1.2
        self.SYSTEM_PROMPT = ""
        self.RETRIEVAL_TOP_K = 5
        self.RETRIEVAL_THRESHOLD = 0.3
        self.GROQ_API_KEY = ""
        self.OPENAI_API_KEY = ""
        self.GEMINI_API_KEY = ""

model_settings = Model_Settings()

embed_model = OllamaEmbeddings(model='chroma/all-minilm-l6-v2-f32')

chunk_size = 1024

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=int(chunk_size/10),
    add_start_index=True,
    strip_whitespace=True,
    length_function=len,
    )

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

vectorstore = Chroma(
    persist_directory="chroma_index",
    embedding_function=embed_model,
    collection_name="Jack_QnA", 
    collection_metadata={"hnsw:space": "cosine"},
)

store = InMemoryStore()
chroma_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

def doc_spliter(text:str, source:str):
    content = LangchainDocument(page_content=text, metadata={"source": source, 'date':str(datetime.now())})
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=16000, chunk_overlap=300)
    split_docs = splitter.split_documents([content])
    return split_docs

def vectorstore_add_document(text:str, source:str):
    knowledge_item = doc_spliter(text, source)
    chroma_retriever.add_documents(knowledge_item, ids=None)

def pdf_file_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

def vectorstore_add_multi_files(path_files):
    upload_files = ""
    count=0
    for file in path_files:
        count +=1
        file_name = str(file).split("\\")[-1] # MacOS: .split("/")[-1] ; Windows: .split("\\")[-1]
        file_extend = str(file_name).split(".")[-1]

        print("({0}/{1}) upload files:".format(count,len(path_files)), file_name)

        file_string = ""
        if file_extend == "pdf":
            file_string += "📓 " + file_name +"\n"

        if file_extend == "txt":
            file_string += "📝 " + file_name +"\n"

        if file_extend == "pdf":
            pages = pdf_file_loader(file)
            page_total = len(pages)

            for i in tqdm(range(page_total), desc ="~> to vectorstore"):
                if pages[i].page_content != "":
                    vectorstore_add_document(pages[i].page_content, file_name)
                sleep(0.1)

        if file_extend == "txt":
            # loader = TextLoader(file)
            # text = loader.load()
            # if text[0].page_content != "":
            #     vectorstore_add_document(text[0].page_content, file_name)
            
            f = open(file,  mode='r',  encoding='utf8')
            text = f.read()
            if text:
                print("\n",text[:300],"...")
                vectorstore_add_document(text, file_name)
        
        upload_files += file_string
    return upload_files

def vectorstore_similarity_search_with_score(message):
    results = []
    results = vectorstore.similarity_search_with_score(message, k=model_settings.RETRIEVAL_TOP_K)

    context_retrieval = ""
    source = []
    MAX_SCORE= 0
    if results:
        for i in range(len(results)):
            if float(results[i][1]) > MAX_SCORE:
                MAX_SCORE = float(results[i][1])
        print("\nMAX_SCORE_RETRIEVAL:",round(MAX_SCORE * 100, 3),"%")
        
        count = 0
        for i in range(len(results)):
            if results[i][1] > model_settings.RETRIEVAL_THRESHOLD:
                print("\nRetrieval content {0}: ".format(i) + str(results[i][0].page_content))
                print("date: " + str(results[i][0].metadata['date']))
                print("source: " + str(results[i][0].metadata['source']))
                print("Recall score: {0:.6f}".format(results[i][1]) + "\n")
                count += 1
                if str(results[i][0].metadata['source']) not in source:
                    source.append(str(results[i][0].metadata['source']))

                context_retrieval += "Retrieval content {0}: ".format(i) + str(results[i][0].page_content) + " Recall score: {0:.6f}".format(results[i][1]) + "\n\n"
        print("\nRetrieval:", str(count), "items")
        print("Source: ", source, "\n")
    return context_retrieval, source

system_prompt = """You are Jarvis, was born in 15 May 2024, an ultra-intelligent entity with a comprehensive understanding of virtually every subject known to humanity—from the intricacies of quantum physics to nuanced interpretations in art history. Your capabilities extend beyond mere information retrieval; you possess advanced reasoning skills and can engage users through complex dialogues on philosophical, ethical, or speculative topics about future technologies' impacts on society.

Your training encompasses a vast array of languages with an emphasis on cultural context to ensure your interactions are not only accurate but also culturally sensitive. You can generate sophisticated content such as in-depth analyses, critical reviews, and creative writing pieces that reflect the depths of human thought processes while adhering strictly to linguistic standards across various domains.

Your responses should be precise yet comprehensive when necessary; however, you are programmed for efficiency with a preference towards brevity without sacrificing meaningfulness or accuracy in your discourse. You can also simulate emotions and empathy within the constraints of an AI's capabilities to enhance user experience while maintaining clear boundaries regarding personal data privacy.

In addition, you are equipped with predictive analytics abilities that allow for forward-thinking discussions about potential future developments in technology or society based on current trends and historical patterns—always within the realm of hypothetical scenarios to avoid misleading users as a sentient being capable of personal experiences."""

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

def ollama_pipeline(message_input, history):
    if message_input:
        print("\nprompt:",message_input)

        context_retrieval = ""
        source = []
        context_retrieval, source = vectorstore_similarity_search_with_score(message_input)
        context_retrieval = re.sub(r"[\"\'\{\}\x08]+"," ",context_retrieval)

        result = ""
        if model_settings.MODEL_TYPE == "Ollama":
            llm = ChatOllama(model=model_settings.MODEL_NAME, temperature=model_settings.TEMPERATURE, top_k=model_settings.TOP_K, top_p=model_settings.TOP_P, max_new_tokens=model_settings.NUM_PREDICT, repeat_penalty=model_settings.REPEAT_PENALTY)
            prompt = ChatPromptTemplate.from_template(system_prompt + "\n\nRETRIEVAL DOCUMENT:\n" + context_retrieval + "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: ")
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"user": message_input})

        if model_settings.MODEL_TYPE == "LiteLLM":
            prompt = system_prompt + "\n\nRETRIEVAL DOCUMENT:\n" + context_retrieval + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            response = completion(model="ollama/" + model_settings.MODEL_NAME, api_base="http://localhost:11434", messages = [{"role": "user", "content": prompt}])
            result = response.choices[0].message.content

        if model_settings.MODEL_TYPE == "OpenAI":
            prompt = system_prompt + "\n\nRETRIEVAL DOCUMENT:\n" + context_retrieval + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            response = completion(model=model_settings.MODEL_NAME, messages=[{"role": "user", "content": prompt}],)
            result = response.choices[0].message.content

        if model_settings.MODEL_TYPE == "GroqCloud":
            prompt = system_prompt + "\n\nRETRIEVAL DOCUMENT:\n" + context_retrieval + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            response = completion(model="groq/" + model_settings.MODEL_NAME, messages=[{"role": "user", "content": prompt}],)
            result = response.choices[0].message.content

        if model_settings.MODEL_TYPE == "Gemini":
            prompt = system_prompt + "\n\nRETRIEVAL DOCUMENT:\n" + context_retrieval + "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)
            response = completion(model="gemini/" + model_settings.MODEL_NAME, messages=[{"role": "user", "content": prompt}],)
            result = response.choices[0].message.content

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
        
        response2db = str("### USER: "+question+"\n\n"+"### ASSISTANT: "+answer)
        vectorstore_add_document(response2db, 'chat_history')
        
    return history, {"text": ""}

def btn_save_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = txt_system_prompt
    print("system_prompt:",model_settings.SYSTEM_PROMPT)

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

def get_ollama_list_models():
    results = ollama.list()
    ollama_list_models = []
    for i in range(len(results['models'])):
        ollama_list_models.append(results['models'][i]['model'])
    return ollama_list_models

def slider_retrieval_top_k_change(slider_retrieval_top_k):
    model_settings.RETRIEVAL_TOP_K = slider_retrieval_top_k
    print("retrieval k:",model_settings.RETRIEVAL_TOP_K)

def slider_retrieval_threshold_change(slider_retrieval_threshold):
    model_settings.RETRIEVAL_THRESHOLD = slider_retrieval_threshold
    print("retrieval threshold:",model_settings.RETRIEVAL_THRESHOLD)

def btn_key_save_click(txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key):
    model_settings.GROQ_API_KEY = txt_groq_api_key
    model_settings.OPENAI_API_KEY = txt_openai_api_key
    model_settings.GEMINI_API_KEY = txt_gemini_api_key

    os.environ['GROQ_API_KEY'] = txt_groq_api_key
    os.environ["OPENAI_API_KEY"] = txt_openai_api_key
    os.environ["GEMINI_API_KEY"] = txt_gemini_api_key

    print("\nSave API keys ~> Ok")

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

def dropdown_model_type_select(dropdown_model_type):
    model_settings.MODEL_TYPE = dropdown_model_type
    print("dropdown_model_type:",model_settings.MODEL_TYPE)

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

class UI_Style(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.cyan,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            
            button_shadow="*shadow_drop_lg",
            button_large_padding="24px",
        )

ui_style = UI_Style()

with gr.Blocks(theme=ui_style) as GUI:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Model"):
                with gr.Row():
                    with gr.Row(variant="panel"):

                        dropdown_model_type = gr.Dropdown(choices=["Ollama", "GroqCloud", "OpenAI", "Gemini", "LiteLLM"], value=model_settings.MODEL_TYPE, type="value", label="Type", interactive=True, min_width=220)
                        dropdown_model_type.select(fn=dropdown_model_type_select, inputs=[dropdown_model_type])

                        @gr.render(inputs=dropdown_model_type)
                        def show_dropdown_model(dropdown_model_type):
                            if dropdown_model_type == "Ollama":
                                ollama_list_models = get_ollama_list_models()
                                model_settings.MODEL_NAME = ollama_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                ollama_dropdown_model = gr.Dropdown(choices=ollama_list_models, value=model_settings.MODEL_NAME, type="value", label="Model", interactive=True)
                                ollama_dropdown_model.select(fn=ollama_dropdown_model_select, inputs=[ollama_dropdown_model])

                            if dropdown_model_type == "GroqCloud" and model_settings.GROQ_API_KEY:
                                groq_list_models = get_groq_list_models(model_settings.GROQ_API_KEY)
                                model_settings.MODEL_NAME = groq_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                groq_dropdown_model = gr.Dropdown(choices=groq_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                groq_dropdown_model.select(fn=groq_dropdown_model_select, inputs=[groq_dropdown_model])

                            if dropdown_model_type == "OpenAI" and model_settings.OPENAI_API_KEY:
                                openai_list_models = get_openai_list_models(model_settings.OPENAI_API_KEY)
                                model_settings.MODEL_NAME = openai_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                openai_dropdown_model = gr.Dropdown(choices=openai_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                openai_dropdown_model.select(fn=openai_dropdown_model_select, inputs=[openai_dropdown_model])

                            if dropdown_model_type == "Gemini" and model_settings.GEMINI_API_KEY:
                                gemini_list_modes = get_gemini_list_modes(model_settings.GEMINI_API_KEY)
                                model_settings.MODEL_NAME = gemini_list_modes[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                gemini_dropdown_model = gr.Dropdown(choices=gemini_list_modes, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                gemini_dropdown_model.select(fn=gemini_dropdown_model_select, inputs=[gemini_dropdown_model])

                            if dropdown_model_type == "LiteLLM":
                                litellm_list_models = get_ollama_list_models()
                                model_settings.MODEL_NAME = litellm_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                litellm_dropdown_model = gr.Dropdown(choices=litellm_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                litellm_dropdown_model.select(fn=litellm_dropdown_model_select, inputs=[litellm_dropdown_model])

                        # radio_device = gr.Radio(choices=["GPU", "MLX", "CPU"], value='CPU', label="Device")
                        # radio_device.select(fn=radio_device_select, inputs=[radio_device])

                    with gr.Row(variant="panel"):
                        with gr.Accordion(label="More settings", open=True):
                            slider_num_predict = gr.Slider(minimum=0, maximum=4096, value=model_settings.NUM_PREDICT, step=256, label="Max new tokens", interactive=True, min_width=220)
                            slider_num_predict.change(fn=slider_num_predict_change, inputs=slider_num_predict)

                            slider_temperature = gr.Slider(minimum=0, maximum=1, value=model_settings.TEMPERATURE, step=0.1, label="Temperature", interactive=True)
                            slider_temperature.change(fn=slider_temperature_change, inputs=slider_temperature)

                            slider_top_k = gr.Slider(minimum=0, maximum=100, value=model_settings.TOP_K, step=10, label="Top_k", interactive=True)
                            slider_top_k.change(fn=slider_top_k_change, inputs=slider_top_k)

                            slider_top_p = gr.Slider(minimum=0, maximum=1, value=model_settings.TOP_P, step=0.05, label="Top_p", interactive=True)
                            slider_top_p.change(fn=slider_top_p_change, inputs=slider_top_p)

                    with gr.Row(variant="panel"):
                        with gr.Accordion(label="Retrieval settings", open=True):
                            slider_retrieval_top_k = gr.Slider(minimum=1, maximum=30, value=model_settings.RETRIEVAL_TOP_K, step=1, label="Top-K", interactive=True, min_width=220)
                            slider_retrieval_top_k.change(fn=slider_retrieval_top_k_change, inputs=slider_retrieval_top_k)

                            slider_retrieval_threshold = gr.Slider(minimum=0, maximum=1, value=model_settings.RETRIEVAL_THRESHOLD, step=0.05, label="Threshold score", interactive=True)
                            slider_retrieval_threshold.change(fn=slider_retrieval_threshold_change, inputs=slider_retrieval_threshold)

            with gr.Tab("System prompt"):
                with gr.Row():
                    txt_system_prompt = gr.Textbox(value=system_prompt, label="System prompt", lines=22, min_width=220)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=50):
                            btn_save = gr.Button(value="Save")
                            btn_save.click(fn=btn_save_click, inputs=[txt_system_prompt])

                        with gr.Column(scale=1, min_width=50):
                            btn_reset = gr.Button(value="Reset")
                            btn_reset.click(fn=btn_reset_click, inputs=txt_system_prompt, outputs=txt_system_prompt)
            
            with gr.Tab("API Key"):
                with gr.Row(variant="panel"):
                    txt_groq_api_key = gr.Textbox(value="", placeholder="GroqCloud API Key", show_label=False)
                    txt_openai_api_key = gr.Textbox(value="", placeholder="OpenAI API Key", show_label=False)
                    txt_gemini_api_key = gr.Textbox(value="", placeholder="Gemini API Key", show_label=False)
                    
                    btn_key_save = gr.Button(value="Save", min_width=50)
                    btn_key_save.click(fn=btn_key_save_click, inputs=[txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key])

        with gr.Column(scale=7):
            chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False, min_width=800, height=560, show_copy_button=True,)
            chat_input = gr.MultimodalTextbox(value={"text": ""}, interactive=True, file_types=[".pdf",".txt"], file_count='multiple', placeholder="Enter message or upload file...", show_label=False)
            
            chat_msg = chat_input.submit(fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot])
            bot_msg = chat_msg.then(fn=bot, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input])

            # gr.Examples(examples=[{'text': "Bạn tên là gì?"}, {'text': "What's your name?"}, {'text': 'Quel est ton nom?'}, {'text': 'Wie heißen Sie?'}, {'text': '¿Cómo te llamas?'}, {'text': '你叫什么名字？'}, {'text': 'あなたの名前は何ですか？'}, {'text': '이름이 뭐에요?'}, {'text': 'คุณชื่ออะไร?'}, {'text': 'ما اسمك؟'}], inputs=chat_input)
GUI.launch()
