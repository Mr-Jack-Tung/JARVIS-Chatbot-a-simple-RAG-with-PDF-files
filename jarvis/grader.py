# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM

# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.output_parsers import StrOutputParser

def retrieval_grader(question, documents, local_llm="qwen2"):
	# LLM
	# local_llm = "qwen2"
	llm = ChatOllama(model=local_llm, format="json", temperature=0)

	# Prompt
	prompt = PromptTemplate(
	    template="""You are a teacher grading a quiz. You will be given: 
	    1/ a QUESTION
	    2/ A FACT provided by the student
	    
	    You are grading RELEVANCE RECALL:
	    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
	    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
	    1 is the highest (best) score. 0 is the lowest score you can give. 
	    
	    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
	    
	    Avoid simply stating the correct answer at the outset.
	    
	    Question: {question} \n
	    Fact: \n\n {documents} \n\n
	    
	    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
	    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
	    """,
	    input_variables=["question", "documents"],
	)

	retrieval_grader = prompt | llm | JsonOutputParser()
	result = retrieval_grader.invoke({"question": question, "documents": documents})

	print("\nretrieval result:",result)
	# {'score': '1'}
	return result # int(result['score'])
