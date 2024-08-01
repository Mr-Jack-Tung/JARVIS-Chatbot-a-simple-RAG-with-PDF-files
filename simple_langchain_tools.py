# -*- coding: utf-8 -*-
# Author: Mr.Jack _ www.BICweb.vn
# Start: 03Mar2024 - 09PM
# End: 11Mar2024 - 12PM

# pip install -U arxiv wikipedia
# pip install -U langchainhub

import os, sys, re

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from langchain.agents import load_tools

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import DuckDuckGoSearchRun

class DuckDuckGoSearch(BaseTool):
    name = "duck-go-search"
    description = "This tool will lookup information on internet."

    def _run(self, input_text:str) -> str:
        """Return the information on internet."""
        search = DuckDuckGoSearchRun()
        result = search.run(input_text)
        return result
        
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

    tools = [DuckDuckGoSearch(),Wiki_tools, Arxiv_tool[0], ExtractTitle()]

    # print("\ntools:",tools)

    return tools

'''
"arxiv": (
        _get_arxiv,
        ["top_k_results", "load_max_docs", "load_all_available_meta"],
    ),
    
'''
