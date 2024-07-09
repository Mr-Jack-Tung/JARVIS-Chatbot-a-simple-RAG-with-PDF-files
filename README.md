# JARVIS Chatbot: a local simple RAG with PDF files assistant
- Author: Mr. Jack Tung
- Date: 03 July 2024 - 08 PM
- If you like what I do, give me a star ^^ ~> ⭐

### Features
- support 29 languages
- upload in any language PDF files and response with user language ^^
- unlimit upload files to vector database
- support PDF, TXT files
- multi-files upload
- unlimit & auto save chat history to vector database
- support custom System Prompt
- Ollama model auto-loader
- 360 code lines ^^

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/JARVIS%20Chatbot%20_%20Screenshot%202024-07-03.png)

### Installation
- Step 1:
  - Create and activate new conda enviroment with python==3.11.9 : https://docs.anaconda.com/navigator/tutorials/manage-environments/
  - Ollama installation ~> https://ollama.com
  - ollama pull chroma/all-minilm-l6-v2-f32
  - ollama pull qwen2
- Step 2:
  - git clone https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files.git
  - pip install -r requirements.txt
- Step 3:
  - python JARVIS_chatbot_v01.py
  - open web browser on local URL:  http://127.0.0.1:7860
