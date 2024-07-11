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
- custom Inferece settings
- unlimit & auto save chat history to vector database
- support custom System Prompt
- Ollama model auto-loader
- (v0.1.0) 360 code lines ^^
- custom Retrieval settings
- Source reference response

### Update Jul 09, 2024 (v0.1.1)
- Add: custom Retrieval settings
- Add: Source reference response
- Add: LLaMAX3-8B-Alpaca (supports over 100 languages)_ https://ollama.com/mrjacktung/mradermacher-llamax3-8b-alpaca-gguf _ bonus: "How To Create Custom Ollama Models From HuggingFace ( GGUF ) file"
- python JARVIS_chatbot_v0_1_1.py

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/JARVIS%20Chatbot%20v0.1.1%20_%20Screenshot%202024-07-11.jpg)

### Screenshot Jul 03, 2024 (v0.1.0)
- Qwen2 model support 29 languages
- support custom System Prompt
- custom Inferece settings
- support multi-files upload: .PDF, .TXT format
- unlimit & auto save chat history to database

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
  - python JARVIS_chatbot_v0_1_0.py
  - open web browser on local URL:  http://127.0.0.1:7860
