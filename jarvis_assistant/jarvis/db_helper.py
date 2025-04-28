# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.7
# Update: 28 April 2025 - 23:18

import os
import platform
from time import sleep
from datetime import datetime
from tqdm import tqdm
import logging

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.docstore.document import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from duckduckgo_search import DDGS

from .file_readers import pdf_file_reader, docx_file_reader, text_file_reader
from .grader import retrieval_grader
from .model_settings import Model_Settings as model_settings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Vector store configuration
CHUNK_SIZE = 1024
CHUNK_OVERLAP = int(CHUNK_SIZE/10)
PARENT_CHUNK_SIZE = 4000
PARENT_CHUNK_OVERLAP = 200
EMBEDDING_MODEL = 'nomic-embed-text'
VECTOR_STORE_DIR = "chroma_vectorstore"
COLLECTION_NAME = "Jack_QnA"

# Initialize text splitters
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    strip_whitespace=True,
    length_function=len,
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE, 
    chunk_overlap=PARENT_CHUNK_OVERLAP
)

# Initialize embedding model and vector store
try:
    embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Create vector store directory if it doesn't exist
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
        
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embed_model,
        collection_name=COLLECTION_NAME, 
        collection_metadata={"hnsw:space": "cosine"},
    )
    
    store = InMemoryStore()
    chroma_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
except Exception as e:
    log.error(f"Error initializing vector store: {str(e)}")
    raise

def doc_spliter(text: str, source: str):
    """
    Split a document into chunks for vectorization
    
    Args:
        text (str): Document text content
        source (str): Source identifier for the document
        
    Returns:
        list: List of document chunks
    """
    try:
        content = LangchainDocument(
            page_content=text, 
            metadata={"source": source, 'date': str(datetime.now())}
        )
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=16000, 
            chunk_overlap=300
        )
        split_docs = splitter.split_documents([content])
        return split_docs
    except Exception as e:
        log.error(f"Error splitting document: {str(e)}")
        return []

def vectorstore_add_document(text: str, source: str):
    """
    Add a document to the vector store
    
    Args:
        text (str): Document text content
        source (str): Source identifier for the document
    """
    try:
        knowledge_item = doc_spliter(text, source)
        if knowledge_item:
            chroma_retriever.add_documents(knowledge_item, ids=None)
            log.info(f"Added document from {source} to vector store")
    except Exception as e:
        log.error(f"Error adding document to vector store: {str(e)}")

def _get_file_name(file_path):
    """
    Extract filename from a file path
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Filename
    """
    platform_sys = platform.system()  # "Linux", "Windows", or "Darwin" (Mac)
    sep = "\\" if platform_sys == "Windows" else "/"
    file_name = str(file_path).split(sep)[-1]
    return file_name

def vectorstore_add_multi_files(path_files):
    """
    Process and add multiple files to the vector store
    
    Args:
        path_files (list): List of file paths to process
        
    Returns:
        str: Summary of uploaded files
    """
    upload_files = ""
    count = 0
    
    for file_path in path_files:
        count += 1    
        file_name = _get_file_name(file_path)
        file_ext = str(file_name).split(".")[-1].lower()

        log.info(f"Processing file {count}/{len(path_files)}: {file_name}")
        print(f"({count}/{len(path_files)}) upload files: {file_name}")

        file_string = ""
        
        try:
            # Process PDF files
            if file_ext == "pdf":
                file_string += "📓 " + file_name + "\n"
                pages = pdf_file_reader(file_path)
                page_total = len(pages)

                if pages:
                    for i in tqdm(range(page_total), desc="~> to vectorstore"):
                        if pages[i].page_content:
                            vectorstore_add_document(pages[i].page_content, file_name)
                        sleep(0.1)
                else:
                    log.warning(f"No content extracted from PDF: {file_name}")

            # Process text files
            elif file_ext in ["txt", "md", "mdx"]:
                file_string += "📝 " + file_name + "\n"
                text = text_file_reader(file_path)

                if text:
                    print("\n", text[:300], "...")
                    vectorstore_add_document(text, file_name)
                else:
                    log.warning(f"No content extracted from text file: {file_name}")
            
            # Process Word documents
            elif file_ext == "docx":
                file_string += "📓 " + file_name + "\n"
                text = docx_file_reader(file_path)

                if text:
                    print("\n", text[:300], "...")
                    vectorstore_add_document(text, file_name)
                else:
                    log.warning(f"No content extracted from DOCX file: {file_name}")
            
            # Unsupported file type
            else:
                file_string += "❌ " + file_name + " (unsupported file type)\n"
                log.warning(f"Unsupported file type: {file_ext}")
                
            upload_files += file_string
            
        except Exception as e:
            log.error(f"Error processing file {file_name}: {str(e)}")
            upload_files += f"❌ Error processing {file_name}: {str(e)}\n"
            
    return upload_files

def web_augmented_search(question, k=3):
    """
    Perform web search to augment retrieval results
    
    Args:
        question (str): Query to search for
        k (int): Maximum number of results to return
        
    Returns:
        list: List of search result snippets
    """
    logger.info(f"Web search for: {question}, max results: {k}")
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(question, max_results=k)
            snippets = [hit.get('body', '') for hit in hits if 'body' in hit]
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
        
    if not snippets:
        logger.warning(f"No web search results found")
        return []
        
    logger.debug(f"Found {len(snippets)} web search results")
    return snippets

def vectorstore_similarity_search_with_score(question, top_k, retrieval_threshold):
    """
    Search the vector store for relevant documents
    
    Args:
        question (str): Query to search for
        top_k (int): Maximum number of results to return
        retrieval_threshold (float): Minimum similarity score threshold
        
    Returns:
        tuple: (context_retrieval, source_list)
    """
    try:
        # Perform similarity search
        search_results = vectorstore.similarity_search_with_score(question, k=top_k)
        
        # Apply grader if enabled
        if model_settings.IS_GRADER and search_results:
            # Filter out non-relevant documents
            filtered_results = []
            for doc in search_results:
                if int(retrieval_grader(question, str(doc[0].page_content))['score']) == 1:
                    filtered_results.append(doc)
            results = filtered_results
        else:
            results = search_results

        context_retrieval = ""
        source = []
        max_score = 0
        
        # Process search results
        if results:
            # Find maximum score
            for i in range(len(results)):
                if float(results[i][1]) > max_score:
                    max_score = float(results[i][1])
            
            log.info(f"Max retrieval score: {round(max_score * 100, 3)}%")
            print(f"\nMAX_SCORE_RETRIEVAL: {round(max_score * 100, 3)}%")
            
            # Process results above threshold
            count = 0
            for i in range(len(results)):
                if results[i][1] > retrieval_threshold:
                    doc_content = str(results[i][0].page_content)
                    doc_date = str(results[i][0].metadata['date'])
                    doc_source = str(results[i][0].metadata['source'])
                    doc_score = results[i][1]
                    
                    print(f"\nRetrieval content {i}:\n{doc_content}")
                    print(f"- date: {doc_date}")
                    print(f"- source: {doc_source}")
                    print(f"- recall score: {doc_score:.6f}\n")
                    
                    count += 1
                    if doc_source not in source:
                        source.append(doc_source)

                    context_retrieval += f"Retrieval content {i}:\n{doc_content} Recall score: {doc_score:.6f}\n\n"
            
            log.info(f"Retrieved {count} items from sources: {source}")
            print(f"\nRetrieval: {count} items")
            print(f"Source: {source}\n")
        
        # Augment with web search if enabled and retrieval quality is low
        if model_settings.IS_WEB_SEARCH and max_score < retrieval_threshold:
            log.info("Low-quality/no retrieval, augmenting with web search")
            web_snips = web_augmented_search(question, top_k)
            if web_snips:
                context_retrieval += "\n\nWEB SEARCH RESULTS:\n" + "\n\n".join(web_snips)
                source.append("web_search")
                log.info("Added web search results to context")
            
        return context_retrieval, source
        
    except Exception as e:
        log.error(f"Error in similarity search: {str(e)}")
        return "", []
