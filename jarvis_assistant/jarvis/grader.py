# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

import json
import traceback
from typing import Dict, Any, Optional, Union

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from loguru import logger as base_logger

from .model_settings import Model_Settings

# Create a separate logger for the grader
grader_logger = base_logger.bind(name="grader").opt(colors=True)
grader_logger.add(
    "logs/grader.log",
    rotation="10 MB",
    retention="1 week",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {module}:{function}:{line} | {message}",
    filter=lambda record: record["extra"].get("name") == "grader"
)

# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/
# https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker
def retrieval_grader(question: str, documents: str, local_llm: Optional[str] = None) -> Dict[str, Any]:
    """
    Grade the relevance of a document to a question.
    
    Args:
        question (str): The question to evaluate relevance against
        documents (str): The document content to evaluate
        local_llm (Optional[str]): Override the LLM to use, defaults to model in settings
        
    Returns:
        Dict[str, Any]: Result with score key (1 for relevant, 0 for not relevant)
    """
    try:
        # Get model settings
        settings = Model_Settings()
        
        # Use the specified LLM or default to the one in settings
        model_type = settings.MODEL_TYPE
        model_name = local_llm or settings.MODEL_NAME
        
        grader_logger.info(f"Grading relevance using {model_type}/{model_name}")
        grader_logger.debug(f"Grading document: {documents[:300]}...")
        
        # Define the grading prompt
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
        
        # Use different approaches based on model type
        if model_type == "Ollama":
            # Use LangChain with Ollama
            try:
                llm = ChatOllama(model=model_name, format="json", temperature=0)
                retrieval_grader_chain = prompt | llm | JsonOutputParser()
                result = retrieval_grader_chain.invoke({"question": question, "documents": documents})
                grader_logger.info(f"Grading result (Ollama): {result}")
                return result
            except Exception as e:
                grader_logger.error(f"Error using Ollama for grading: {str(e)}")
                grader_logger.error(traceback.format_exc())
                # Fall back to direct API approach
                
        # For other model types or as fallback, use the LLM client directly
        try:
            from .llm_client import get_llm_client
            
            # Format the prompt manually
            formatted_prompt = prompt.format(question=question, documents=documents)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are a relevance scorer that outputs only JSON."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the appropriate client
            client = get_llm_client(model_type, model_name)
            
            # Make the API call
            response = client.chat(messages)
            
            # Extract the content from the response
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response if needed (some models might add text around the JSON)
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                        
                    grader_logger.info(f"Grading result ({model_type}): {result}")
                    return result
                except json.JSONDecodeError:
                    grader_logger.error(f"Failed to parse JSON from response: {content}")
                    # Return a default score of 1 to be inclusive rather than exclusive
                    return {"score": 1}
            else:
                grader_logger.error(f"Invalid response format: {response}")
                return {"score": 1}  # Default to including the document
                
        except Exception as e:
            grader_logger.error(f"Error in API-based grading: {str(e)}")
            grader_logger.error(traceback.format_exc())
            # Default to including the document in case of errors
            return {"score": 1}
            
    except Exception as e:
        grader_logger.error(f"Unexpected error in retrieval_grader: {str(e)}")
        grader_logger.error(traceback.format_exc())
        # Default to including the document in case of errors
        return {"score": 1}

def hallucination_grader(question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Grade whether an answer contains hallucinations not supported by the context.
    
    Args:
        question (str): The original question
        answer (str): The generated answer to evaluate
        context (str): The retrieval context used to generate the answer
        
    Returns:
        Dict[str, Any]: Result with hallucination_score key (0-1 scale, lower is better)
    """
    try:
        # Get model settings
        settings = Model_Settings()
        model_type = settings.MODEL_TYPE
        model_name = settings.MODEL_NAME
        
        grader_logger.info(f"Grading hallucination using {model_type}/{model_name}")
        
        # Define the hallucination grading prompt
        prompt = PromptTemplate(
            template="""
You are a hallucination detector. You will receive:
1) QUESTION: The original question asked
2) ANSWER: The generated answer to evaluate
3) CONTEXT: The retrieval context used to generate the answer

Evaluate if the ANSWER contains information not supported by the CONTEXT.
Assign a hallucination score from 0 to 1:
- 0: No hallucination, all information is supported by the context
- 0.5: Some minor unsupported details or slight exaggerations
- 1: Significant hallucination, major claims not supported by context

Output only JSON in the form {{"hallucination_score":0.5, "explanation":"Brief reason for score"}}

QUESTION: {question}
ANSWER: {answer}
CONTEXT: {context}
""",
            input_variables=["question", "answer", "context"],
        )
        
        # Use the LLM client directly
        try:
            from .llm_client import get_llm_client
            
            # Format the prompt manually
            formatted_prompt = prompt.format(question=question, answer=answer, context=context)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are a hallucination detector that outputs only JSON."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the appropriate client
            client = get_llm_client(model_type, model_name)
            
            # Make the API call
            response = client.chat(messages)
            
            # Extract the content from the response
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response if needed
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                        
                    grader_logger.info(f"Hallucination grading result: {result}")
                    return result
                except json.JSONDecodeError:
                    grader_logger.error(f"Failed to parse JSON from response: {content}")
                    # Return a moderate score as default
                    return {"hallucination_score": 0.5, "explanation": "Failed to parse grader output"}
            else:
                grader_logger.error(f"Invalid response format: {response}")
                return {"hallucination_score": 0.5, "explanation": "Invalid grader response"}
                
        except Exception as e:
            grader_logger.error(f"Error in hallucination grading: {str(e)}")
            grader_logger.error(traceback.format_exc())
            return {"hallucination_score": 0.5, "explanation": f"Error: {str(e)}"}
            
    except Exception as e:
        grader_logger.error(f"Unexpected error in hallucination_grader: {str(e)}")
        grader_logger.error(traceback.format_exc())
        return {"hallucination_score": 0.5, "explanation": f"Unexpected error: {str(e)}"}

def answer_grader(question: str, answer: str) -> Dict[str, Any]:
    """
    Grade the quality of an answer to a question.
    
    Args:
        question (str): The original question
        answer (str): The generated answer to evaluate
        
    Returns:
        Dict[str, Any]: Result with quality_score key (0-10 scale) and feedback
    """
    try:
        # Get model settings
        settings = Model_Settings()
        model_type = settings.MODEL_TYPE
        model_name = settings.MODEL_NAME
        
        grader_logger.info(f"Grading answer quality using {model_type}/{model_name}")
        
        # Define the answer quality grading prompt
        prompt = PromptTemplate(
            template="""
You are an answer quality evaluator. You will receive:
1) QUESTION: The original question asked
2) ANSWER: The generated answer to evaluate

Evaluate the quality of the ANSWER based on:
- Relevance to the question
- Completeness
- Accuracy
- Clarity and conciseness
- Helpfulness

Assign a quality score from 0 to 10 (10 being perfect).
Output only JSON in the form {{"quality_score":8, "feedback":"Brief feedback on strengths and weaknesses"}}

QUESTION: {question}
ANSWER: {answer}
""",
            input_variables=["question", "answer"],
        )
        
        # Use the LLM client directly
        try:
            from .llm_client import get_llm_client
            
            # Format the prompt manually
            formatted_prompt = prompt.format(question=question, answer=answer)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are an answer quality evaluator that outputs only JSON."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the appropriate client
            client = get_llm_client(model_type, model_name)
            
            # Make the API call
            response = client.chat(messages)
            
            # Extract the content from the response
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response if needed
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                        
                    grader_logger.info(f"Answer quality grading result: {result}")
                    return result
                except json.JSONDecodeError:
                    grader_logger.error(f"Failed to parse JSON from response: {content}")
                    return {"quality_score": 5, "feedback": "Failed to parse grader output"}
            else:
                grader_logger.error(f"Invalid response format: {response}")
                return {"quality_score": 5, "feedback": "Invalid grader response"}
                
        except Exception as e:
            grader_logger.error(f"Error in answer quality grading: {str(e)}")
            grader_logger.error(traceback.format_exc())
            return {"quality_score": 5, "feedback": f"Error: {str(e)}"}
            
    except Exception as e:
        grader_logger.error(f"Unexpected error in answer_grader: {str(e)}")
        grader_logger.error(traceback.format_exc())
        return {"quality_score": 5, "feedback": f"Unexpected error: {str(e)}"}

def question_rewriter(question: str, context: str = "") -> Dict[str, Any]:
    """
    Rewrite a question to make it more specific and answerable.
    
    Args:
        question (str): The original question to rewrite
        context (str, optional): Optional context to help with rewriting
        
    Returns:
        Dict[str, Any]: Result with rewritten_question and explanation
    """
    try:
        # Get model settings
        settings = Model_Settings()
        model_type = settings.MODEL_TYPE
        model_name = settings.MODEL_NAME
        
        grader_logger.info(f"Rewriting question using {model_type}/{model_name}")
        
        # Define the question rewriting prompt
        template = """
You are a question improvement specialist. You will receive:
1) ORIGINAL_QUESTION: The question as originally asked
{context_section}

Your task is to rewrite the question to make it more specific, clear, and answerable.
The rewritten question should:
- Be more precise and focused
- Remove ambiguities
- Include relevant context from the provided information (if any)
- Be formulated to get the most helpful response

Output only JSON in the form {{"rewritten_question":"The improved question", "explanation":"Brief explanation of changes"}}

ORIGINAL_QUESTION: {question}
"""
        
        # Add context section if provided
        context_section = "2) CONTEXT: Additional information that may help with rewriting" if context else ""
        template = template.replace("{context_section}", context_section)
        
        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"] + (["context"] if context else []),
        )
        
        # Use the LLM client directly
        try:
            from .llm_client import get_llm_client
            
            # Format the prompt manually
            prompt_args = {"question": question}
            if context:
                prompt_args["context"] = context
            formatted_prompt = prompt.format(**prompt_args)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are a question improvement specialist that outputs only JSON."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the appropriate client
            client = get_llm_client(model_type, model_name)
            
            # Make the API call
            response = client.chat(messages)
            
            # Extract the content from the response
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response if needed
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                        
                    grader_logger.info(f"Question rewriting result: {result}")
                    return result
                except json.JSONDecodeError:
                    grader_logger.error(f"Failed to parse JSON from response: {content}")
                    return {
                        "rewritten_question": question,
                        "explanation": "Failed to parse rewriter output"
                    }
            else:
                grader_logger.error(f"Invalid response format: {response}")
                return {
                    "rewritten_question": question,
                    "explanation": "Invalid rewriter response"
                }
                
        except Exception as e:
            grader_logger.error(f"Error in question rewriting: {str(e)}")
            grader_logger.error(traceback.format_exc())
            return {
                "rewritten_question": question,
                "explanation": f"Error: {str(e)}"
            }
            
    except Exception as e:
        grader_logger.error(f"Unexpected error in question_rewriter: {str(e)}")
        grader_logger.error(traceback.format_exc())
        return {
            "rewritten_question": question,
            "explanation": f"Unexpected error: {str(e)}"
        }

def sub_query_generator(question: str, context: str = "") -> Dict[str, Any]:
    """
    Generate sub-queries to help answer a complex question.
    
    Args:
        question (str): The original complex question
        context (str, optional): Optional context to help with sub-query generation
        
    Returns:
        Dict[str, Any]: Result with sub_queries list and reasoning
    """
    try:
        # Get model settings
        settings = Model_Settings()
        model_type = settings.MODEL_TYPE
        model_name = settings.MODEL_NAME
        
        grader_logger.info(f"Generating sub-queries using {model_type}/{model_name}")
        
        # Define the sub-query generation prompt
        template = """
You are a query decomposition specialist. You will receive:
1) COMPLEX_QUESTION: A potentially complex question that may benefit from being broken down
{context_section}

Your task is to decompose the complex question into 2-5 simpler sub-queries that together would help answer the original question.
Each sub-query should:
- Focus on a specific aspect of the original question
- Be simple and straightforward
- Be answerable on its own
- Contribute to answering the original question

Output only JSON in the form:
{{
  "sub_queries": [
    "First sub-query",
    "Second sub-query",
    ...
  ],
  "reasoning": "Brief explanation of how these sub-queries help answer the original question"
}}

COMPLEX_QUESTION: {question}
"""
        
        # Add context section if provided
        context_section = "2) CONTEXT: Additional information that may help with decomposition" if context else ""
        template = template.replace("{context_section}", context_section)
        
        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"] + (["context"] if context else []),
        )
        
        # Use the LLM client directly
        try:
            from .llm_client import get_llm_client
            
            # Format the prompt manually
            prompt_args = {"question": question}
            if context:
                prompt_args["context"] = context
            formatted_prompt = prompt.format(**prompt_args)
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are a query decomposition specialist that outputs only JSON."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the appropriate client
            client = get_llm_client(model_type, model_name)
            
            # Make the API call
            response = client.chat(messages)
            
            # Extract the content from the response
            if response and "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    # Clean the response if needed
                    if "{" in content and "}" in content:
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(content)
                        
                    grader_logger.info(f"Sub-query generation result: {result}")
                    return result
                except json.JSONDecodeError:
                    grader_logger.error(f"Failed to parse JSON from response: {content}")
                    return {
                        "sub_queries": [question],
                        "reasoning": "Failed to parse sub-query generator output"
                    }
            else:
                grader_logger.error(f"Invalid response format: {response}")
                return {
                    "sub_queries": [question],
                    "reasoning": "Invalid sub-query generator response"
                }
                
        except Exception as e:
            grader_logger.error(f"Error in sub-query generation: {str(e)}")
            grader_logger.error(traceback.format_exc())
            return {
                "sub_queries": [question],
                "reasoning": f"Error: {str(e)}"
            }
            
    except Exception as e:
        grader_logger.error(f"Unexpected error in sub_query_generator: {str(e)}")
        grader_logger.error(traceback.format_exc())
        return {
            "sub_queries": [question],
            "reasoning": f"Unexpected error: {str(e)}"
        }
