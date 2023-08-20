import os
import numpy as np
import pickle

from langchain.vectorstores import FAISS, Chroma, DocArrayInMemorySearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store_index(file_path, embedding_model_repo_id="sentence-transformers/all-roberta-large-v1"):

    file_path_split = file_path.split(".")
    file_type = file_path_split[-1].rstrip('/')

    if file_type == 'csv':
        print(file_path)
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
    
    elif file_type == 'pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap = 128,)

        documents = text_splitter.split_documents(pages)

    
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_repo_id
        )
    
    vectordb = FAISS.from_documents(documents, embedding_model)
    file_output = "./db/faiss_index"
    vectordb.save_local(file_output)

    return "Vector store index is created."