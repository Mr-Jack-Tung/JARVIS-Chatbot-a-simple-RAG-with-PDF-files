# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.6
# Update: 28 April 2025


from langchain.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOllama # LangChainDeprecationWarning: The class `ChatOllama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0.
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
import sys
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
# from langchain_core.output_parsers import StrOutputParser

# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/
# https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
def retrieval_grader(question, documents, local_llm="qwen2.5:3b"):

	### LLM
	# local_llm = "qwen2.5:3b"
	llm = ChatOllama(model=local_llm, format="json", temperature=0)

	### Prompt
	prompt = PromptTemplate(
        template="""
You are a relevance scorer. You will receive:
1) QUESTION
2) FACT

Assign a binary score: 1 if ANY statement in the FACT is relevant to the QUESTION; otherwise 0.
Output only JSON in the form {{"score":1}} with no additional text or explanation.

QUESTION: {question}
FACT: {documents}
""",
		input_variables=["question", "documents"],
	)

	logger.debug(f"retrieval document: {documents[:300]}")

	retrieval_grader = prompt | llm | JsonOutputParser()
	result = retrieval_grader.invoke({"question": question, "documents": documents})

	logger.info(f"~> relevant result: {result}")
	# {'score': '1'}
	return result # int(result['score'])
