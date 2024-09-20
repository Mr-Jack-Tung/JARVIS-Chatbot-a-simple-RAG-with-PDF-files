# -*- coding: utf-8 -*-
# JARVIS Chatbot - a simple RAG with PDF files
# Create: 03 July 2024
# Author: Mr.Jack _ www.bicweb.vn
# Version: 0.1.5
# Date: 02 December 2024 - 05 PM

from time import sleep
from datetime import datetime
import tqdm

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.docstore.document import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_nomic.embeddings import NomicEmbeddings

chunk_size = 1024
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=int(chunk_size/10),
    add_start_index=True,
    strip_whitespace=True,
    length_function=len,
    )

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

embed_model = OllamaEmbeddings(model='nomic-embed-text')
# embed_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

vectorstore = Chroma(
    persist_directory="chroma_vectorstore",
    embedding_function=embed_model,
    collection_name="Jack_QnA", 
    collection_metadata={"hnsw:space": "cosine"},
)

store = InMemoryStore()
chroma_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

def doc_spliter(text:str, source:str):
    content = LangchainDocument(page_content=text, metadata={"source": source, 'date':str(datetime.now())})
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=16000, chunk_overlap=300)
    split_docs = splitter.split_documents([content])
    return split_docs

def vectorstore_add_document(text:str, source:str):
    knowledge_item = doc_spliter(text, source)
    chroma_retriever.add_documents(knowledge_item, ids=None)

from jarvis.file_readers import pdf_file_reader, docx_file_reader, text_file_reader

import platform # Get system information
def vectorstore_add_multi_files(path_files):
    my_platform = platform.system() #  "Linux", "Windows", or "Darwin" (Mac)
    
    upload_files = ""
    count=0
    for file in path_files:
        count +=1
        
        file_name = ""
        if my_platform == "Windows":
            file_name = str(file).split("\\")[-1]   # Windows: .split("\\")[-1]
        elif my_platform == "Darwin":
            file_name = str(file).split("/")[-1]    # MacOS: .split("/")[-1] 
        else:
            file_name = str(file).split("/")[-1]    # Linux: .split("/")[-1]
        file_extend = str(file_name).split(".")[-1]

        print("({0}/{1}) upload files:".format(count,len(path_files)), file_name)

        file_string = ""
        if file_extend == "pdf":
            file_string += "📓 " + file_name +"\n"
            pages = pdf_file_reader(file)
            page_total = len(pages)

            for i in tqdm(range(page_total), desc ="~> to vectorstore"):
                if pages[i].page_content != "":
                    vectorstore_add_document(pages[i].page_content, file_name)
                sleep(0.1)

        if file_extend in ["txt", "md", "mdx"]:
            file_string += "📝 " + file_name +"\n"
            text = text_file_reader(file)

            if text:
                print("\n",text[:300],"...")
                vectorstore_add_document(text, file_name)
        
        if file_extend == "docx":
            file_string += "📓 " + file_name +"\n"
            text = docx_file_reader(file)

            if text:
                print("\n",text[:300],"...")
                vectorstore_add_document(text, file_name)
                
        upload_files += file_string
    return upload_files

from jarvis.grader import retrieval_grader

def vectorstore_similarity_search_with_score(question, top_k, retrieval_threshold):
    results = []
    search_results = vectorstore.similarity_search_with_score(question, k=top_k)

    for doc in search_results:
        if int(retrieval_grader(question, str(doc[0].page_content))['score']) == 1:
            # print(doc)
            results.append(doc)

    context_retrieval = ""
    source = []
    MAX_SCORE= 0
    if results:
        for i in range(len(results)):
            if float(results[i][1]) > MAX_SCORE:
                MAX_SCORE = float(results[i][1])
        print("\nMAX_SCORE_RETRIEVAL:",round(MAX_SCORE * 100, 3),"%")
        
        count = 0
        for i in range(len(results)):
            if results[i][1] > retrieval_threshold:
                print("\nRetrieval content {0}:\n".format(i) + str(results[i][0].page_content))
                print("- date: " + str(results[i][0].metadata['date']))
                print("- source: " + str(results[i][0].metadata['source']))
                print("- recall score: {0:.6f}".format(results[i][1]) + "\n")
                count += 1
                if str(results[i][0].metadata['source']) not in source:
                    source.append(str(results[i][0].metadata['source']))

                context_retrieval += "Retrieval content {0}:\n".format(i) + str(results[i][0].page_content) + " Recall score: {0:.6f}".format(results[i][1]) + "\n\n"
        print("\nRetrieval:", str(count), "items")
        print("Source: ", source, "\n")
    return context_retrieval, source

# https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/
# from jarvis.corrective_rag import get_corrective_rag_agent
# def corrective_rag():
#     get_corrective_rag_agent()
