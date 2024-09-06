# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 06 December 2024 - 03 PM

# https://github.com/Mr-Jack-Tung/Ollama-Mistral-with-Langchain-RAG-Agent-and-Custom-tools
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from jarvis.tools import get_all_tools

# from langchain import hub
# prompt = hub.pull("hwchase17/react-chat")

def react_agent(model_name, system_prompt, retrieval_prompt, message_input):
	model_local = Ollama(model=model_name)
	tools = get_all_tools()
	prompt = PromptTemplate(input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'], metadata={'lc_hub_owner': 'jarvis_assistant', 'lc_hub_repo': 'react-chat', 'lc_hub_commit_hash': ''}, template=system_prompt + retrieval_prompt + '\n\nOverall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}')
	agent = create_react_agent(model_local, tools, prompt)
	chat_history_memory = ConversationBufferWindowMemory(k=3, memory_key='chat_history', input_key='input', ouput_key='output')
	agent_executor = AgentExecutor(
		agent=agent, 
		tools=tools, 
		memory=chat_history_memory,
		verbose=True, # ~> Speech out the thinking
		handle_parsing_errors=True,
		)

	response = agent_executor.invoke({"input": "\n\nCONVERSATION:\n**human**: {0}\n**Jarvis (AI)**: ".format(message_input)})
	return response