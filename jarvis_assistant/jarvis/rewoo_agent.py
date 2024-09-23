# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 07 December 2024 - 01 AM

# https://github.com/langchain-ai/langgraph/blob/main/examples/rewoo/rewoo.ipynb
# Reasoning without Observation

def rewoo_agent(model_name, system_prompt, context_retrieval, task):
    # from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # model = ChatOpenAI(temperature=0)
    # model = ChatOllama(model="qwen2", temperature=0)
    model = ChatOllama(model=model_name, temperature=0)

    # prompt = ChatPromptTemplate.from_template(system_prompt + retrieval_prompt + "\n\nCONVERSATION:\n**human**: {user}\n**Jarvis (AI)**: ")
    # chain = prompt | model | StrOutputParser()
    # simple_answer = chain.invoke({"user": task})

    # model = ChatOllama(model=model_name, temperature=0)
    # result = model.invoke(system_prompt + retrieval_prompt + prompt.format(task=task))

    from typing import List, TypedDict

    class ReWOO(TypedDict):
        task: str
        plan_string: str
        steps: List
        results: dict
        result: str


    ### 1. Planner

    prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
    which external tool together with tool input to retrieve evidence. You can store the evidence into a \
    variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

    Tools can be one of the following:
    (1) RAG[input]: Worker that retrieval specific information from user upload data or previous Q&A from database. Prioritize using this tool. Useful when you need to find information from local database first.
    (2) Google[input]: Worker that searches results from Google. Useful when you need to find short and succinct answers about a specific topic. The input should be a search query.
    (3) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
    
    For example,
    Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
    hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
    less than Toby. How many hours did Rebecca work?
    Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
    with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
    Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
    Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

    Begin! 
    Describe your plans with rich details. Each Plan should be followed by only one #E.

    Task: {task}"""

    # task = "what is the hometown of the 2024 australian open winner"
    result = model.invoke(prompt.format(task=task))
    # result = model.invoke(system_prompt + retrieval_prompt + prompt.format(task=task))


    ### Planner Node

    import re
    from langchain_core.prompts import ChatPromptTemplate

    # Regex to match expressions of the form E#... = ...[...]
    regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
    prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
    planner = prompt_template | model

    def get_plan(state: ReWOO):
        task = state["task"]
        result = planner.invoke({"task": task})

        # Find all matches in the sample text
        matches = re.findall(regex_pattern, result.content)
        return {"steps": matches, "plan_string": result.content}


    ### 2. Executor

    # from langchain_community.tools.tavily_search import TavilySearchResults
    # search = TavilySearchResults()

    from jarvis.tools import TaviDuckGoSearch
    search = TaviDuckGoSearch()

    def _get_current_task(state: ReWOO):
        # if state["results"] is None:
        if not state.get("results"):
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1


    def tool_execution(state: ReWOO):
        """Worker node that executes the tools of a given plan."""
        _step = _get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        # _results = state["results"] or {}
        if state.get("results"):
            _results = state["results"]
        else:
            _results = {}

        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        if tool == "RAG":
            result = {'query':tool_input, 'results':context_retrieval}
        elif tool == "Google":
            result = search.invoke(tool_input)
        elif tool == "LLM":
            result = model.invoke(tool_input)
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}

    ### 3. Solver

    solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
    retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
    contain irrelevant information.

    {plan}

    Now solve the question or task according to provided Evidence above. Respond with the answer
    directly with no extra words.

    Task: {task}
    Response:"""


    def solve(state: ReWOO):
        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = state["results"] or {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = model.invoke(prompt)
        return {"result": result.content}

    ### 4. Define Graph

    def _route(state):
        _step = _get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"


    from langgraph.graph import END, StateGraph, START

    graph = StateGraph(ReWOO)
    graph.add_node("plan", get_plan)
    graph.add_node("tool", tool_execution)
    graph.add_node("solve", solve)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tool")
    graph.add_conditional_edges("tool", _route)
    graph.add_edge("solve", END)

    app = graph.compile()

    '''
    # pip install graphviz
    # View graph_image
    graph_image = app.get_graph().draw_mermaid_png() # app.get_graph().print_ascii()

    # import base64
    from PIL import Image
    from io import BytesIO

    # Convert the bytes to an image
    image = Image.open(BytesIO(graph_image))
    image.show()
    '''

    final_result = ""
    last_state = {}
    for s in app.stream({"task": task}):
        last_state = s
        print(s)
        print("---")
        
    # Print out the final result
    # print(s[END]["result"])

    if last_state.get('solve'):
        final_result = last_state['solve']['result']
    # print('\nFinal result:',final_result)
    return {'output':final_result}

# print('\nFinal result:',rewoo_agent(task))

# ---
# {'solve': {'result': 'San Candido, Italy'}}
# ---

# Search on Internet: The winner of the 2024 Australian Open men’s singles title is Jannik Sinner. He hails from San Candido (Innichen), a small town in the South Tyrol region of Italy
