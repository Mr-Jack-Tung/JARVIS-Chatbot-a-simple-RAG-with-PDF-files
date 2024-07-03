# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Author: Mr.Jack _ www.bicweb.vn
# Date: 03 July 2024 - 08 PM

# pip install tqdm, pypdf, chromadb, tiktoken, gradio, langchain, langchain_community, ollama

import os, sys, re
import time
from datetime import datetime
from typing import Iterable
from tqdm import tqdm
from time import sleep

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

import ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader

def pdf_file_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

class Model_Settings:
    def __init__(self):
        self.MODEL_NAME = 'qwen2:latest'
        self.NUM_PREDICT = 1024
        self.TEMPERATURE = 0.3
        self.TOP_K = 50
        self.TOP_P = 0.95
        self.REPEAT_PENALTY = 1.2
        self.SYSTEM_PROMPT = ""
        self.SCORE_MARGIN_RETRIEVAL = 0.3

model_settings = Model_Settings()

embed_model = OllamaEmbeddings(model='nomic-embed-text')

vector_store = "chroma_index"
chunk_size = 512

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=int(chunk_size/10),
    add_start_index=True,
    strip_whitespace=True,
    length_function=len,
    )

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

vectorstore = Chroma(
    collection_name="Jack_QnA", 
    embedding_function=embed_model,
    persist_directory=vector_store,
    collection_metadata={"hnsw:space": "cosine"}
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
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=16000, chunk_overlap=100)
    split_docs = splitter.split_documents([content])
    return split_docs

def vectorstore_add_document(text:str, source:str):
    knowledge_item = doc_spliter(text, source)
    chroma_retriever.add_documents(knowledge_item, ids=None)

def vectorstore_add_multi_files(path_files):
    upload_files = ""
    count=0
    for file in path_files:
        count +=1
        file_name = str(file).split("/")[-1]
        file_extend = str(file_name).split(".")[-1]

        print("({0}/{1}) upload files:".format(count,len(path_files)), file_name)

        file_string = ""
        if file_extend == "pdf":
            file_string += "📓 " + file_name +"\n"

        if file_extend == "txt":
            file_string += "📝 " + file_name +"\n"

        if file_extend == "pdf":
            pages = pdf_file_loader(str(file))
            page_total = len(pages)

            for i in tqdm(range(page_total), desc ="~> to vectorstore"):
                if pages[i] != "":
                    vectorstore_add_document(str(pages[i]), file_name)
                sleep(0.1)

        if file_extend == "txt":
            text = open(str(file), 'r').read()
            if text != "":
                vectorstore_add_document(str(text), file_name)

        upload_files += file_string

    return upload_files

def vectorstore_similarity_search_with_score(message):
    results = []
    retrieval = []
    results = vectorstore.similarity_search_with_score(message, k=10)

    MAX_SCORE= 0
    if results:
        for i in range(len(results)):
            if float(results[i][1]) > MAX_SCORE:
                MAX_SCORE = float(results[i][1])

        model_settings.SCORE_MARGIN_RETRIEVAL = round(MAX_SCORE * 0.99, 5)

        print("\nSCORE_MARGIN_RETRIEVAL:",round(model_settings.SCORE_MARGIN_RETRIEVAL * 100, 5),"%")
        for i in range(len(results)):
            if float(results[i][1]) >= float(model_settings.SCORE_MARGIN_RETRIEVAL):
                retrieval.append(results[i])

    context_retrieval = ""
    if retrieval:
        print("\nRetrieval:", len(retrieval), "items")
        for i in range(len(retrieval)):
            context_retrieval += retrieval[i][0].page_content + "\n\n"

    return context_retrieval

system_prompt = """You are Jarvis, was born in 15 May 2024, an ultra-intelligent entity with a comprehensive understanding of virtually every subject known to humanity—from the intricacies of quantum physics to nuanced interpretations in art history. Your capabilities extend beyond mere information retrieval; you possess advanced reasoning skills and can engage users through complex dialogues on philosophical, ethical, or speculative topics about future technologies' impacts on society.

Your training encompasses a vast array of languages with an emphasis on cultural context to ensure your interactions are not only accurate but also culturally sensitive. You can generate sophisticated content such as in-depth analyses, critical reviews, and creative writing pieces that reflect the depths of human thought processes while adhering strictly to linguistic standards across various domains.

Your responses should be precise yet comprehensive when necessary; however, you are programmed for efficiency with a preference towards brevity without sacrificing meaningfulness or accuracy in your discourse. You can also simulate emotions and empathy within the constraints of an AI's capabilities to enhance user experience while maintaining clear boundaries regarding personal data privacy.

In addition, you are equipped with predictive analytics abilities that allow for forward-thinking discussions about potential future developments in technology or society based on current trends and historical patterns—always within the realm of hypothetical scenarios to avoid misleading users as a sentient being capable of personal experiences."""

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

def add_message(history, message):
    if len(history)<1:
        history.append(["**human**: Hello", "**Jarvis (AI)**: Hi, my name Jarvis. I am your assistant. How may I help you today?"])

    upload_files = ""
    if message["files"]:
        path_files = message["files"]
        print("\n")
        upload_files += vectorstore_add_multi_files(path_files)
        
    if message["text"]:
        dt_string = datetime.now().strftime("%H.%M")
        history.append(("(" + dt_string + ") **human**: " + message["text"], ""))
    if upload_files:
        print("\nUpload files:\n",upload_files)
    return history

def ollama_pipeline(message_input, history):
    if message_input:
        print("\nprompt:",message_input)
        llm = ChatOllama(model=model_settings.MODEL_NAME, temperature=model_settings.TEMPERATURE, top_k=model_settings.TOP_K, top_p=model_settings.TOP_P, max_new_tokens=model_settings.NUM_PREDICT, repeat_penalty=model_settings.REPEAT_PENALTY)
        context_retrieval = ""
        context_retrieval += vectorstore_similarity_search_with_score(message_input)
        context_retrieval = re.sub(r"[\"\']+"," ",context_retrieval)

        prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n" + context_retrieval + "\n\nConversation:\n**human**: {user}\n**Jarvis (AI)**: ")
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"user": message_input})
        return result

def bot(history, chat_input):
    if chat_input['text']:
        question = str(history[-1][0]).split("**human**: ")[1]
        answer = ollama_pipeline(question, history)
        print("\nanswer:",answer)
        dt_string = datetime.now().strftime("%H.%M")
        response = "(" + dt_string + ") **Jarvis (AI)**: " + str(answer)
        history[-1][1] = ""

        for character in response:
            history[-1][1] += character

    return history, gr.MultimodalTextbox(value={"text": ""}, interactive=True)

def btn_save_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = txt_system_prompt
    print("system_prompt:",model_settings.SYSTEM_PROMPT)

def btn_reset_click(txt_system_prompt):
    model_settings.SYSTEM_PROMPT = system_prompt
    return model_settings.SYSTEM_PROMPT

def dropdown_model_select(dropdown_model):
    model_settings.MODEL_NAME = dropdown_model
    print("\nSelected model:",model_settings.MODEL_NAME)

def radio_device_select(radio_device):
    print("Selected device:",radio_device)

def slider_num_predict_change(slider_num_predict):
    model_settings.NUM_PREDICT = slider_num_predict
    print("num_predict setting:",model_settings.NUM_PREDICT)

def slider_temperature_change(slider_temperature):
    model_settings.TEMPERATURE = slider_temperature
    print("temperature setting:",model_settings.TEMPERATURE)

def slider_top_k_change(slider_top_k):
    model_settings.TOP_K = slider_top_k
    print("top_k setting:",model_settings.TOP_K)

def slider_top_p_change(slider_top_p):
    model_settings.TOP_P = slider_top_p
    print("top_p setting:",model_settings.TOP_P)

def get_ollama_list_models():
    results = ollama.list()
    ollama_list_models = []
    for i in range(len(results['models'])):
        ollama_list_models.append(results['models'][i]['model'])
    return ollama_list_models

with gr.Blocks(theme=ui_style) as GUI:
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("Model"):
                with gr.Row():
                    choices_models = get_ollama_list_models()
                    dropdown_model = gr.Dropdown(choices=choices_models, value='qwen2:latest', type="value", label="Model", interactive=True)
                    dropdown_model.select(fn=dropdown_model_select, inputs=[dropdown_model])

                    radio_device = gr.Radio(choices=["GPU", "MLX", "CPU"], value='CPU', label="Device")
                    radio_device.select(fn=radio_device_select, inputs=[radio_device])

                    with gr.Accordion(label="More settings", open=True):
                        slider_num_predict = gr.Slider(minimum=0, maximum=2048, value=model_settings.NUM_PREDICT, step=256, label="Max new tokens", interactive=True)
                        slider_num_predict.change(fn=slider_num_predict_change, inputs=slider_num_predict)

                        slider_temperature = gr.Slider(minimum=0, maximum=1, value=model_settings.TEMPERATURE, step=0.1, label="Temperature", interactive=True)
                        slider_temperature.change(fn=slider_temperature_change, inputs=slider_temperature)

                        slider_top_k = gr.Slider(minimum=0, maximum=100, value=model_settings.TOP_K, step=10, label="Top_k", interactive=True)
                        slider_top_k.change(fn=slider_top_k_change, inputs=slider_top_k)

                        slider_top_p = gr.Slider(minimum=0, maximum=1, value=model_settings.TOP_P, step=0.05, label="Top_p", interactive=True)
                        slider_top_p.change(fn=slider_top_p_change, inputs=slider_top_p)

            with gr.Tab("System prompt"):
                with gr.Row():
                    txt_system_prompt = gr.Textbox(value=system_prompt, label="System prompt", lines=22)

                    with gr.Row():
                        with gr.Column(scale=1, min_width=50):
                            btn_save = gr.Button(value="Save")
                            btn_save.click(fn=btn_save_click, inputs=[txt_system_prompt])

                        with gr.Column(scale=1, min_width=50):
                            btn_reset = gr.Button(value="Reset")
                            btn_reset.click(fn=btn_reset_click, inputs=txt_system_prompt, outputs=txt_system_prompt)

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=500,
                show_copy_button=True,
            )

            chat_input = gr.MultimodalTextbox(value={"text": ""}, interactive=True, file_types=[".pdf",".txt"], file_count='multiple', placeholder="Enter message or upload file...", show_label=False)

            chat_msg = chat_input.submit(fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot])
            bot_msg = chat_msg.then(fn=bot, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input])

            gr.Examples(examples=[{'text': "Bạn tên là gì?"}, {'text': "What's your name?"}, {'text': 'Quel est ton nom?'}, {'text': 'Wie heißen Sie?'}, {'text': '你叫什么名字？'}, {'text': 'あなたの名前は何ですか？'}, {'text': '이름이 뭐에요?'}, {'text': 'คุณชื่ออะไร?'}, {'text': 'ما اسمك؟'}, {'text': "Tell a joke."}], inputs=chat_input)

GUI.queue()
GUI.launch()
