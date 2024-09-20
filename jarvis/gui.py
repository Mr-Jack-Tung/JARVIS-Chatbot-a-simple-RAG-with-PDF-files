# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 07 December 2024 - 01 AM

# pip install gradio-toggle
import gradio as gr
from gradio_toggle import Toggle

from jarvis.gui_action import *

# theme_default = gr.themes.Default().set(
#     body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
#     body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
# )

# theme_default_cyan = gr.themes.Default(primary_hue="cyan").set(
#     body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
#     body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
# )

from jarvis.custom_ui_style import UI_Style
from jarvis.utils import load_api_keys_from_yaml

# GUI ------------------------------------------------------------
def JARVIS_assistant():
    ui_style = UI_Style()

    # default_ui = "gradio/default"
    # theme_2 = gr.Theme.from_hub("gradio/base")
    # theme_3 = gr.Theme.from_hub("gradio/seafoam")
    # theme_4 = gr.Theme.from_hub("gradio/glass")
    # theme_5 = gr.Theme.from_hub("gstaff/xkcd") 
    # theme_6 = gr.Theme.from_hub("ParityError/LimeFace")
    # theme_7 = gr.Theme.from_hub("EveryPizza/Cartoony-Gradio-Theme")
    # theme_8 = gr.Theme.from_hub("snehilsanyal/scikit-learn")
    # theme_9 = gr.Theme.from_hub("abidlabs/banana")
    
    with gr.Blocks(theme=ui_style) as GUI:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Workspace"):
                    first_state_workspace = {"id":0, "name":"My first workspace", "history":[["**human**: Hello", "**Jarvis (AI)**: Hi, my name Jarvis. I am your assistant. How may I help you today?"]]}
                    state_workspace_list = gr.State([first_state_workspace])
                    state_workspace_selected = gr.State(first_state_workspace)
                    
                    with gr.Row(variant="panel"):
                        with gr.Row():
                            btn_save_workspace = gr.Button(value="Save all workspaces", min_width=220)
                            btn_save_workspace.click(fn=btn_save_workspace_click, inputs=state_workspace_list)

                            btn_create_new_workspace = gr.Button(value="Create new workspace", min_width=220)
                            btn_create_new_workspace.click(fn=btn_create_new_workspace_click, inputs=state_workspace_list, outputs=[state_workspace_list, state_workspace_selected])

                        with gr.Row():
                            @gr.render(inputs=[state_workspace_list, state_workspace_selected])
                            def show_new_workspace(workspace_list, workspace_selected):
                                def txt_workspace_focus(focus_state):
                                    for wp in workspace_list:
                                        if wp["id"] == focus_state:
                                            return wp
                                
                                def btn_delete_workspace_click(focus_state):
                                    if len(workspace_list) > 1:
                                        for wp in workspace_list:
                                            if wp["id"] == focus_state:
                                                workspace_list.remove(wp)
                                    return workspace_list
                                
                                def txt_workspace_change(focus_state, txt_workspace):
                                    for wp in workspace_list:
                                        if wp["id"] == focus_state:
                                            workspace_list.remove(wp)
                                            workspace= {"id":wp["id"], "name":txt_workspace, "history":wp["history"]}
                                            workspace_list.insert(0, workspace)
                                            return workspace_list, workspace
                                            
                                for wksp in workspace_list:
                                    with gr.Row():
                                        with gr.Column(scale=10, min_width=200):
                                            focus_state = gr.State(wksp["id"])
                                            txt_workspace = gr.Textbox(value=str(wksp["name"]), show_label=False, container=False, min_width=200,  interactive=True)
                                            txt_workspace.focus(txt_workspace_focus, focus_state, [state_workspace_selected])
                                            txt_workspace.submit(txt_workspace_change, [focus_state, txt_workspace], [state_workspace_list, state_workspace_selected])
                                        
                                        with gr.Column(scale=1, min_width=10):
                                            # if wksp["id"] != 0:
                                            btn_delete_workspace = gr.Button(value="x", min_width=5) #  size="sm"
                                            btn_delete_workspace.click(fn=btn_delete_workspace_click, inputs=focus_state, outputs=[state_workspace_list])
                
                with gr.Tab("Model"):
                    with gr.Row():
                        with gr.Row(variant="panel"):
                            with gr.Accordion(label="API Keys", open=False):
                                GROQ_KEY, OPENAI_KEY, GEMINI_KEY = load_api_keys_from_yaml(model_settings)
                                txt_groq_api_key = gr.Textbox(value=GROQ_KEY, placeholder="GroqCloud API Key", show_label=False)
                                txt_openai_api_key = gr.Textbox(value=OPENAI_KEY, placeholder="OpenAI API Key", show_label=False)
                                txt_gemini_api_key = gr.Textbox(value=GEMINI_KEY, placeholder="Gemini API Key", show_label=False)
                                
                                btn_key_save = gr.Button(value="Save", min_width=50)
                                btn_key_save.click(fn=btn_key_save_click, inputs=[txt_groq_api_key, txt_openai_api_key, txt_gemini_api_key])
                                
                        with gr.Row(variant="panel"):
                            dropdown_model_type = gr.Dropdown(choices=["Ollama", "GroqCloud", "OpenAI", "Gemini", "LiteLLM"], value=model_settings.MODEL_TYPE, type="value", label="Type", interactive=True, min_width=220)
                            dropdown_model_type.select(fn=dropdown_model_type_select, inputs=[dropdown_model_type])
    
                        @gr.render(inputs=dropdown_model_type)
                        def show_dropdown_model(dropdown_model_type):
                            if dropdown_model_type == "Ollama":
                                ollama_list_models = get_ollama_list_models()
                                # model_settings.MODEL_NAME = ollama_list_models[0]
                                model_settings.MODEL_NAME = "qwen2.5:latest"
                                print("Selected model:",model_settings.MODEL_NAME)

                                with gr.Row(variant="panel"):
                                    ollama_dropdown_model = gr.Dropdown(choices=ollama_list_models, value=model_settings.MODEL_NAME, type="value", label="Model", interactive=True, min_width=220)
                                    ollama_dropdown_model.select(fn=ollama_dropdown_model_select, inputs=[ollama_dropdown_model])

                                with gr.Row(variant="panel"):
                                    with gr.Accordion(label="Model settings", open=False):
                                        slider_num_predict = gr.Slider(minimum=0, maximum=4096, value=model_settings.NUM_PREDICT, step=256, label="Max new tokens", interactive=True, min_width=220)
                                        slider_num_predict.change(fn=slider_num_predict_change, inputs=slider_num_predict)
            
                                        slider_temperature = gr.Slider(minimum=0, maximum=1, value=model_settings.TEMPERATURE, step=0.1, label="Temperature", interactive=True)
                                        slider_temperature.change(fn=slider_temperature_change, inputs=slider_temperature)
            
                                        slider_top_k = gr.Slider(minimum=0, maximum=100, value=model_settings.TOP_K, step=10, label="Top_k", interactive=True)
                                        slider_top_k.change(fn=slider_top_k_change, inputs=slider_top_k)
            
                                        slider_top_p = gr.Slider(minimum=0, maximum=1, value=model_settings.TOP_P, step=0.05, label="Top_p", interactive=True)
                                        slider_top_p.change(fn=slider_top_p_change, inputs=slider_top_p)

                                with gr.Row(variant="panel"):
                                    with gr.Accordion(label="Function calling", open=False):
                                        chk_function_calling = Toggle(label="Function calling", value=model_settings.FUNCTION_CALLING, interactive=True, min_width=220)
                                        chk_function_calling.change(fn=update_function_calling, inputs=chk_function_calling)
    
                                        # @gr.render(inputs=chk_function_calling)
                                        # def show_radio_agents(chk_function_calling):
                                        #     if chk_function_calling:
                                        # with gr.Row(variant="panel"):
                                        radio_agents = gr.Radio(choices=["ReWOO", "ReACT"], value='ReWOO', label="with Agents")
                                        radio_agents.select(fn=radio_agents_select, inputs=[radio_agents])

                            if dropdown_model_type == "GroqCloud" and model_settings.GROQ_API_KEY:
                                groq_list_models = get_groq_list_models(model_settings.GROQ_API_KEY)
                                model_settings.MODEL_NAME = groq_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                with gr.Row(variant="panel"):
                                    groq_dropdown_model = gr.Dropdown(choices=groq_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                    groq_dropdown_model.select(fn=groq_dropdown_model_select, inputs=[groq_dropdown_model])

                            if dropdown_model_type == "OpenAI" and model_settings.OPENAI_API_KEY:
                                openai_list_models = get_openai_list_models(model_settings.OPENAI_API_KEY)
                                model_settings.MODEL_NAME = openai_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                with gr.Row(variant="panel"):
                                    openai_dropdown_model = gr.Dropdown(choices=openai_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                    openai_dropdown_model.select(fn=openai_dropdown_model_select, inputs=[openai_dropdown_model])

                            if dropdown_model_type == "Gemini" and model_settings.GEMINI_API_KEY:
                                gemini_list_modes = get_gemini_list_modes(model_settings.GEMINI_API_KEY)
                                model_settings.MODEL_NAME = gemini_list_modes[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                with gr.Row(variant="panel"):
                                    gemini_dropdown_model = gr.Dropdown(choices=gemini_list_modes, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                    gemini_dropdown_model.select(fn=gemini_dropdown_model_select, inputs=[gemini_dropdown_model])

                            if dropdown_model_type == "LiteLLM":
                                litellm_list_models = get_ollama_list_models()
                                model_settings.MODEL_NAME = litellm_list_models[0]
                                print("Selected model:",model_settings.MODEL_NAME)

                                with gr.Row(variant="panel"):
                                    litellm_dropdown_model = gr.Dropdown(choices=litellm_list_models, value=model_settings.MODEL_NAME, type="value", label="Models", interactive=True)
                                    litellm_dropdown_model.select(fn=litellm_dropdown_model_select, inputs=[litellm_dropdown_model])
    
                            # radio_device = gr.Radio(choices=["GPU", "MLX", "CPU"], value='CPU', label="Device")
                            # radio_device.select(fn=radio_device_select, inputs=[radio_device])

                        with gr.Row(variant="panel"):
                            with gr.Accordion(label="Retrieval settings", open=False):
                                chk_is_retrieval = Toggle(label="Is retrieval", value=model_settings.IS_RETRIEVAL, interactive=True)
                                chk_is_retrieval.change(fn=update_is_retrieval, inputs=chk_is_retrieval)
                                
                                slider_retrieval_top_k = gr.Slider(minimum=1, maximum=30, value=model_settings.RETRIEVAL_TOP_K, step=1, label="Top-K", interactive=True, min_width=220)
                                slider_retrieval_top_k.change(fn=slider_retrieval_top_k_change, inputs=slider_retrieval_top_k)
    
                                slider_retrieval_threshold = gr.Slider(minimum=0, maximum=1, value=model_settings.RETRIEVAL_THRESHOLD, step=0.05, label="Threshold score", interactive=True)
                                slider_retrieval_threshold.change(fn=slider_retrieval_threshold_change, inputs=slider_retrieval_threshold)
                        
                        with gr.Row(variant="panel"):
                            chk_chat_saving = Toggle(label="Save Chat-history", value=model_settings.CHAT_HISTORY_SAVING, interactive=True, min_width=220)
                            chk_chat_saving.change(fn=update_chat_saving, inputs=chk_chat_saving)

                with gr.Tab("System prompt"):
                    with gr.Row():
                        btn_basic_prompt = gr.Button(value="basic prompt")
                        btn_function_calling_prompt = gr.Button(value="function_calling prompt")
                        btn_strawberry_o1_prompt = gr.Button(value="strawberry_o1 prompt")
                        
                    with gr.Row():
                        txt_system_prompt = gr.Textbox(value=system_prompt, label="System prompt", lines=23, min_width=220)
                        btn_basic_prompt.click(fn=btn_basic_prompt_click, outputs=txt_system_prompt)
                        btn_function_calling_prompt.click(fn=btn_function_calling_prompt_click, outputs=txt_system_prompt)
                        btn_strawberry_o1_prompt.click(fn=btn_strawberry_o1_prompt_click, outputs=txt_system_prompt)
    
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                btn_save = gr.Button(value="Save")
                                btn_save.click(fn=btn_save_click, inputs=[txt_system_prompt])
    
                            with gr.Column(scale=1, min_width=50):
                                btn_reset = gr.Button(value="Reset")
                                btn_reset.click(fn=btn_reset_click, inputs=txt_system_prompt, outputs=txt_system_prompt)
                
            with gr.Column(scale=7):
                def update_chat_history(chatbot, workspace_list, workspace_selected):
                    for wp in workspace_list:
                        if wp["id"] == workspace_selected["id"]:
                            workspace= {"id":wp["id"], "name":wp["name"], "history":chatbot}
                            workspace_list.remove(wp)
                            workspace_list.insert(0, workspace)
                            return workspace_list, workspace
    
                workspace_selected = state_workspace_selected.value
                chatbot = gr.Chatbot(workspace_selected["history"], elem_id="chatbot", bubble_full_width=False, min_width=800, height=800, show_copy_button=True,)
                chat_input = gr.MultimodalTextbox(value={"text": ""}, interactive=True, file_types=[".pdf",".txt"], file_count='multiple', placeholder="Enter message or upload file...", show_label=False)
    
                def workspace_selected_chatbot(workspace_selected):
                    return workspace_selected["history"]
                state_workspace_selected.change(fn=workspace_selected_chatbot, inputs=state_workspace_selected,  outputs=chatbot)
    
                chat_msg = chat_input.submit(fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot])
                bot_msg = chat_msg.then(fn=bot, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input]).then(fn=update_chat_history, inputs=[chatbot, state_workspace_list, state_workspace_selected], outputs=[state_workspace_list, state_workspace_selected])
    chatbot
    
                # gr.Examples(examples=[{'text': "Bạn tên là gì?"}, {'text': "What's your name?"}, {'text': 'Quel est ton nom?'}, {'text': 'Wie heißen Sie?'}, {'text': '¿Cómo te llamas?'}, {'text': '你叫什么名字？'}, {'text': 'あなたの名前は何ですか？'}, {'text': '이름이 뭐에요?'}, {'text': 'คุณชื่ออะไร?'}, {'text': 'ما اسمك؟'}], inputs=chat_input)
    
    GUI.launch()
