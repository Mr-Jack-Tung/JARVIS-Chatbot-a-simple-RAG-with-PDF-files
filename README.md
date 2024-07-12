# JARVIS Chatbot: a local simple RAG with PDF files assistant
- Author: Mr. Jack Tung
- Date: 03 July 2024 - 08 PM
- Discuss: https://zalo.me/g/mtffzi945
- If you like what I do, give me a star ^^ ~> ⭐

### Why JARVIS?
- All Free ~> 100% No money
- Local Run ~> 100% Privacy
- Open Source ~> 100% Custom

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

### Update next (v0.1.2): 
**Fine tune embbeding model for RAG pipeline ^^**
- https://www.philschmid.de/fine-tune-embedding-model-for-rag
- https://huggingface.co/blog/matryoshka
- https://github.com/thangnch/MiAI_HieuNgo_EmbedingFineTune

**Dynamically Semantic Router**
- https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb


### Update Jul 09, 2024 (v0.1.1)
- Add: custom Retrieval settings
- Add: Source reference response
- Add: LLaMAX3-8B-Alpaca (supports over 100 languages)_ https://ollama.com/mrjacktung/mradermacher-llamax3-8b-alpaca-gguf _ bonus: "How To Create Custom Ollama Models From HuggingFace ( GGUF ) file"
- python JARVIS_chatbot_v0_1_1.py

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/JARVIS%20Chatbot%20v0.1.1%20_%20Screenshot%202024-07-11.jpg)

### Screenshot Jul 03, 2024 (v0.1.0)
- Qwen2-7B model support 29 languages
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

### How To Create Custom Ollama Models From HuggingFace ( GGUF ) file
URL: https://ollama.com/mrjacktung/mradermacher-llamax3-8b-alpaca-gguf

step 1: Go to
- https://huggingface.co/mradermacher/LLaMAX3-8B-Alpaca-GGUF

step 2: Download GGUF file
- Choice: Q4_K_M | 5.0GB | fast | recommended

step 3: Create Modelfile (from Terminal)
- $ echo ‘FROM “./LLaMAX3-8B-Alpaca.Q4_K_M.gguf”\nTEMPLATE “{{ .System }}\n### Input:\n{{ .Prompt }}\n### Response:”’ >> Modelfile

step 4: Login your Ollama account (eg. mrjacktung)
- My models ~> New ~> create new space with name: mradermacher-llamax3-8b-alpaca-gguf

step 5: Create repository
- ollama create -f Modelfile mrjacktung/mradermacher-llamax3-8b-alpaca-gguf
- ollama push mrjacktung/mradermacher-llamax3-8b-alpaca-gguf

step 6: Testing
- ollama run mrjacktung/mradermacher-llamax3-8b-alpaca-gguf

*Thanks to:*
- Michael Radermacher: https://huggingface.co/mradermacher/LLaMAX3-8B-Alpaca-GGUF
- Data Science Basics: Ollama, How To Create Custom Models From HuggingFace (GGUF) _ https://www.youtube.com/watch?v=TFwYvHZV6j0
