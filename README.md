# JARVIS Chatbot: a local simple RAG assistant with PDF files
- Author: Mr. Jack Tung
- Date: 03 July 2024 - 08 PM
- Discuss: https://zalo.me/g/mtffzi945
- If you like what I do, give me a star ^^ ~> ⭐

### Why JARVIS?
- All Free ~> 100% Free
- Local Run ~> 100% Privacy
- Open Source ~> 100% DIY Custom
- 30 multi-languages support
- PDF RAG support

### Features
- support 30 languages
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
- support documents reference response
- support Groq API, OpenAI API, Gemini API
- support LiteLLM locally
- workspaces management

### Update next (v0.x.x)
![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/img/JARVIS%20assistant%20v0.x.x%20_%202024-07-18.jpg)

### Update next (v0.1.x)
**Fine tune embbeding model for RAG pipeline ^^**
- https://www.philschmid.de/fine-tune-embedding-model-for-rag
- https://huggingface.co/blog/matryoshka
- https://github.com/thangnch/MiAI_HieuNgo_EmbedingFineTune
- Fine tuning Embeddings Model: https://www.youtube.com/watch?v=hdFHYNCmO8U

**Dynamically Semantic Router**
- RouteLLM: Learning to Route LLMs with Preference Data _ https://arxiv.org/abs/2406.18665
- https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb
- https://mer.vin/2024/07/routellm-code-example/

**GraphRAG**
- https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/
- https://github.com/microsoft/graphrag/tree/main/examples_notebooks
- https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/
- https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
- https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/

**Continuous Pre-training & Fine-tuning**
- Continual Pre-training of Language Models: https://arxiv.org/abs/2302.03241
- Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering: https://arxiv.org/abs/2311.00204
- Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
- Finetuning: https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html

### Update Jul 28, 2024 (v0.1.3)
- add: Manage workspaces

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/img/JARVIS%20assistant%20v0.1.3%20_%202024-07-28.jpg)

### Update Jul 15, 2024 (v0.1.2)
- add: Groq Cloud API (https://docs.litellm.ai/docs/providers/groq ; https://console.groq.com/docs/api-reference)
- add: OpenAi API (https://docs.litellm.ai/docs/providers/openai ; https://platform.openai.com/docs/models)
- add: Gemini API (https://docs.litellm.ai/docs/providers/gemini ; https://ai.google.dev/gemini-api)
- add: LiteLLM local (https://docs.litellm.ai/docs/)

#### Installation
- Step 1:
  - Create and activate new conda enviroment with python==3.11.9 : https://docs.anaconda.com/navigator/tutorials/manage-environments/
  - Ollama installation ~> https://ollama.com
- Step 2:
  - git clone https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files.git
- Step 3:
  - python JARVIS_assistant.py
  - open web browser on local URL:  http://127.0.0.1:7860

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/img/JARVIS%20assistant%20v0.1.2%20_%202024-07-15.jpg)

### Notes:

https://github.com/chroma-core/chroma/issues/189#issuecomment-1454418844

if you have an issue on Windows OS, while doing pip install chromadb. To Resolve this issue, 

You need to download https://visualstudio.microsoft.com/visual-cpp-build-tools/ first.

Next, navigate to "Individual components", find these two

MSVC v143 - VS2002 C++ x64/86 build tools (lates)
and Windows 10 SDK

then:
pip install -U chromadb


### Update Jul 09, 2024 (v0.1.1)
- Add: custom Retrieval settings
- Add: Source reference response
- Add: LLaMAX3-8B-Alpaca (supports over 100 languages)_ https://ollama.com/mrjacktung/mradermacher-llamax3-8b-alpaca-gguf _ bonus: "How To Create Custom Ollama Models From HuggingFace ( GGUF ) file"
  - ollama pull mrjacktung/mradermacher-llamax3-8b-alpaca-gguf

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/img/JARVIS%20assistant%20v0.1.1%20_%202024-07-11.jpg)

### Screenshot Jul 03, 2024 (v0.1.0)
- Qwen2-7B model support 30 languages (https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF)
- support custom System Prompt
- custom Inferece settings
- support multi-files upload: .PDF, .TXT format
- unlimit & auto save chat history to database

![alt-text](https://github.com/Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files/blob/main/img/JARVIS%20assistant%20v0.1.0%20_%202024-07-03.jpg)

#### BONUS: How To Create Custom Ollama Models From HuggingFace ( GGUF ) file
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
