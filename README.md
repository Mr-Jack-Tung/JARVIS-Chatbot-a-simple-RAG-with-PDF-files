```
# Requirements: python=3.12.4
# conda create -n jarvis python=3.12.4
# conda activate jarvis

# For Installing...
|> python setup.py

# For Running...
|> python JARVIS_assistant.py
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
_
```

# JARVIS Chatbot: a local simple RAG assistant with PDF files
- Author: Mr. Jack Tung
- Create: 03 July 2024 - 08 PM
- Discuss: https://zalo.me/g/mtffzi945
- If you like what I do, give me a star ^^ ~> ⭐

### Why JARVIS?
- All Free ~> 100% Free
- Local Run ~> 100% Privacy
- Open Source ~> 100% DIY Custom
- 30 multi-languages support
- RAG with PDF, DOCX, TXT files support
- Multi-Function calling
- Agent calling

### Features
- support Qwen2.5 is the latest series of Qwen large language models.
- support 30 multi-languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.
- supports long contexts of up to 128K tokens and can generate up to 8K tokens.
- upload in any language PDF files and response with user language ^^
- unlimit upload files to vector database
- support PDF, DOCX, TXT, MD, MDX files
- multi-files upload
- custom Inferece settings
- unlimit & auto save chat history to vector database
- support custom System Prompt
- Ollama model auto-loader
- (v0.1.0) only 360 lines of python code in only 1 file ^^
- custom Retrieval settings
- support documents reference response
- support Groq API, OpenAI API, Gemini API
- support LiteLLM locally
- workspaces management
- on/off Retrieval
- Multi-Function calling
- ReACT agent
- ReWOO agent
- Retrieval grader
- OpenAI Strawberry(o1) likely system prompt for Multi-Step Reasoning chains

### JARVIS assistant (21 ⭐) .vs
- gpt4all (69k ⭐) - https://github.com/nomic-ai/gpt4all
- open-webui (38k ⭐) - https://github.com/open-webui/open-webui
- anything-llm (21k ⭐) - https://github.com/Mintplex-Labs/anything-llm
- LibreChat (17k ⭐) - https://github.com/danny-avila/LibreChat
- Perplexica (13k ⭐) - https://github.com/ItzCrazyKns/Perplexica
- Kotaemon (11k ⭐) - https://github.com/Cinnamon/kotaemon
- Verba (6k ⭐) - https://github.com/weaviate/Verba
- gpt-computer-assistant (5.2k ⭐) - https://github.com/onuratakan/gpt-computer-assistant
- MindSearch (4.5k ⭐) - https://github.com/InternLM/MindSearch
- lagent (1.7k ⭐) - https://github.com/InternLM/lagent
- lmstudio-ai (1.4k ⭐) - https://github.com/lmstudio-ai/lms


### Update next (v0.x.x)
![alt-text](./img/JARVIS%20assistant%20v0.x.x%20_%202024-08-09.jpg)

### Update next (v0.1.x)
(Multilingual, Multimodal, MultiAgent, MultiDevice, ...)

<details>
  
**Multimodal support**
- text
- image
  - MiniCPM-V 2.6: https://github.com/OpenBMB/MiniCPM-V
  - FLUX: https://github.com/black-forest-labs/flux
  - Stable Diffusion: https://github.com/runwayml/stable-diffusion
  - https://github.com/huggingface/diffusers
- audio
  - https://github.com/onuratakan/gpt-computer-assistant
  - https://github.com/rsxdalv/tts-generation-webui
- video
  - LLaVA-NeXT: https://github.com/LLaVA-VL/LLaVA-NeXT ; https://arxiv.org/abs/2408.03326
  - NExT-GPT: https://github.com/NExT-GPT/NExT-GPT

**Tools and Multi-Agents**
- Math tool
- Internet search agent
- Professional Agents(PAgents): https://arxiv.org/abs/2402.03628
- **CrewAI multi-agents**: https://github.com/crewAIInc/crewAI
- **Crawl4AI agent**: https://github.com/unclecode/crawl4ai
- **AnyTool agent**: https://github.com/dyabel/AnyTool
- **OpenDevin agent**: https://github.com/OpenDevin/OpenDevin
- **DistillKit**: https://github.com/arcee-ai/DistillKit
- **MindSearch agent**: https://github.com/InternLM/MindSearch ; https://arxiv.org/abs/2407.20183
- **AgileCoder**: https://github.com/FSoft-AI4Code/AgileCoder
- **AgentK**: automatic build new tools and agents as needed by itself, in order to complete tasks for a user _ https://github.com/mikekelly/AgentK
- **AI-Scientist**: https://github.com/SakanaAI/AI-Scientist
- **OpenResearcher**: https://github.com/GAIR-NLP/OpenResearcher
- **ADAS**: Automated Design of Agentic Systems - https://github.com/ShengranHu/ADAS
- **Language Agent Tree Search (LAST)**: https://github.com/lapisrocks/LanguageAgentTreeSearch ; https://arxiv.org/abs/2310.04406

**Mobile**
- MobileAgent: https://github.com/X-PLUG/MobileAgent

**Synthetic Data**
- PERSONA HUB: 200,000 synthetic personas - https://github.com/tencent-ailab/persona-hub

**Fine tune embbeding model for RAG pipeline ^^**
- https://www.philschmid.de/fine-tune-embedding-model-for-rag
- https://huggingface.co/blog/matryoshka
- https://github.com/thangnch/MiAI_HieuNgo_EmbedingFineTune
- Fine tuning Embeddings Model: https://www.youtube.com/watch?v=hdFHYNCmO8U

**Dynamically Semantic Router**
- **RouteLLM**: https://github.com/lm-sys/RouteLLM ; RouteLLM: Learning to Route LLMs with Preference Data _ https://arxiv.org/abs/2406.18665
- https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb
- https://github.com/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb
- https://mer.vin/2024/07/routellm-code-example/

**All RAG techniques**
- https://github.com/NirDiamant/RAG_Techniques
- WeKnow-RAG: https://arxiv.org/abs/2408.07611
- Controllable-RAG-Agent: https://github.com/NirDiamant/Controllable-RAG-Agent
- Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks - https://arxiv.org/abs/2407.21059
- RAG Flow: https://github.com/infiniflow/ragflow
- RAG and RAU Survey: https://github.com/2471023025/RALM_Survey ; https://arxiv.org/abs/2404.19543
- RAG Foundry Framework: https://arxiv.org/abs/2408.02545

**GraphRAG**
- **GraphRAG**: https://github.com/microsoft/graphrag
- https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/
- https://github.com/microsoft/graphrag/tree/main/examples_notebooks
- https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/
- https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
- https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/

**Continuous Pre-training & Fine-tuning**
- Continual Pre-training of Language Models: https://arxiv.org/abs/2302.03241
- Continual Pre-Training of Large Language Models: How to (re)warm your model? https://arxiv.org/abs/2308.04014
- Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering: https://arxiv.org/abs/2311.00204
- Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
- Finetuning: https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html
- Selective fine-tuning of Language Models with Spectrum: https://huggingface.co/blog/anakin87/spectrum

**Document AI**
- LayoutLMv3: https://github.com/microsoft/unilm/tree/master/layoutlmv3

**Knowledge Editing**
- Knowledge Editing: https://github.com/zjunlp/KnowledgeEditingPapers

</details>

### Update September 21, 2024 (v0.1.5)
- update: system prompt choosing or editing

![alt-text](./img/JARVIS%20assistant%20v0.1.5%20_%202024-09-21.jpg)

### Update September 18, 2024 (v0.1.5)
- update: OpenAI Strawberry(o1) likely system prompt for Multi-Step Reasoning chains
- update: support Qwen2.5 is the latest series of Qwen large language models.

![alt-text](./img/JARVIS%20assistant%20v0.1.5%20_%202024-09-20.jpg)

### Update September 02, 2024 (v0.1.5)
- update: Nomic Embed v1.5
- update: separate source code files
- add: Retrieval grader
- update: using LangGraph
- add: ReWOO agent - https://blog.langchain.dev/planning-agents ; https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb

![alt-text](./img/JARVIS%20assistant%20v0.1.5%20_%202024-09-07.jpg)

```
+	-------------------- workflow ---------------------------------
|	v0.1.5
|	JARVIS_assistant.py
|		|
|		~> gui.py ~> custom_ui_style.py
|			|
|			~> gui_action.py ~> model_settings.py , tools.py , prompts.py , utils.py , get_model_list.py
|				|
|				~> db_helper.py  ~> file_readers.py
|					|
|					~> datasource_router.py , grader.py: retrieval_grader()
+ -----------------------------------------------------------------
```

### Update Jul 31, 2024 (v0.1.4)
- add: Multi-Function calling
- add: ReACT agent
- add: API Keys management
- add: 3 rounds chat-history memory
- update: support both MacOS and Windows

![alt-text](./img/JARVIS%20assistant%20v0.1.4%20_%202024-08-01_2.jpg)

![alt-text](./img/JARVIS%20assistant%20v0.1.4%20_%202024-08-01.jpg)

### Update Jul 28, 2024 (v0.1.3)
- add: Workspaces management
- add: On/Off Retrieval
- add: support DOCX files

![alt-text](./img/JARVIS%20assistant%20v0.1.3%20_%202024-07-28_2.jpg)

![alt-text](./img/JARVIS%20assistant%20v0.1.3%20_%202024-07-28.jpg)

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

![alt-text](./img/JARVIS%20assistant%20v0.1.2%20_%202024-07-15.jpg)

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

![alt-text](./img/JARVIS%20assistant%20v0.1.1%20_%202024-07-11.jpg)

### Screenshot Jul 03, 2024 (v0.1.0)
- Qwen2-7B model support 30 languages (https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF)
- support custom System Prompt
- custom Inferece settings
- support multi-files upload: .PDF, .TXT format
- unlimit & auto save chat history to database

![alt-text](./img/JARVIS%20assistant%20v0.1.0%20_%202024-07-03.jpg)

#### BONUS: How To Create Custom Ollama Models From HuggingFace ( GGUF ) file

<details>

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

</details>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files&type=Date)](https://star-history.com/#Mr-Jack-Tung/JARVIS-Chatbot-a-simple-RAG-with-PDF-files&Date)
