# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 07 December 2024 - 01 AM

# pip install -U arxiv wikipedia
# pip install -U langchainhub

import os, sys, re

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

# from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# https://pypi.org/project/duckduckgo-search/
# pip install -qU duckduckgo-search langchain-community
# from langchain_community.tools import DuckDuckGoSearchRun

class TwoNumbersCompare(BaseTool):
    name = "two-numbers-compare"
    description = "This tool will compare two numbers."

    @staticmethod
    def is_float(value):
        if value is None:
            return False
        try:
            float(value)
            return True
        except:
            return False

    def _run(self, input_text:str) -> str:
        """Return the greater number."""
        results = re.findall(r"([\d\.]+)", str(input_text))

        print("number A:",results[0])
        print("number B:",results[1])

        if self.is_float(results[0]):
            number_A = float(results[0])
        if self.is_float(results[1]):
            number_B = float(results[1])

        result = ""
        if number_A and number_B:
            if number_A == number_B:
                result = "number " + str(results[0]) + " is equal to " + str(results[1])
            elif number_A > number_B:
                result = "number " + str(results[0]) + " is greater than " + str(results[1])
            elif number_B > number_A:
                result = "number " + str(results[1]) + " is greater than " + str(results[0])

        return result

# from langchain.tools import BaseTool
from duckduckgo_search import DDGS
class TaviDuckGoSearch(BaseTool):
    name = "tavily-duck-go-search"
    description = "This tool will lookup information on internet."

    def _run(self, query:str) -> dict:
        """Return the information on internet."""

        if query:
            # https://python.langchain.com/v0.2/docs/integrations/tools/tavily_search/
            results = DDGS().text(query, max_results=5)

            # https://docs.tavily.com/docs/rest-api/api-reference
            # Tavily Response format likely
            response = {'query':query,'results':results}
            return response
        else:
            return {'query':query,'results':''}
        
class ExtractTitle(BaseTool):
    name = "extract-title-from-text-tool"
    description = "This tool will extract title from text of arXiv document."

    def _run(self, input_text: str) -> str:
        """Returns the extract title from text."""
        return re.findall(r"^(?:Title:)[\:\w\ \-\_]+$\n", str(input_text))

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool to look up things."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 45 or less words"
    )

def get_all_tools():
    Wiki_tools = WikipediaQueryRun(
        name="wiki-tool",
        description="look up things in wikipedia",
        args_schema=WikiInputs,
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000),
        )

    Arxiv_tool = load_tools(
        ["arxiv"],
    )

    tools = [TwoNumbersCompare(), TaviDuckGoSearch(), Wiki_tools, Arxiv_tool[0], ExtractTitle()]

    # print("\ntools:",tools)

    return tools

'''
"arxiv": (
        _get_arxiv,
        ["top_k_results", "load_max_docs", "load_all_available_meta"],
    ),
    
'''
