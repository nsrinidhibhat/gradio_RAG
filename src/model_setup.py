import os,gc,shutil
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch

class ModelSetup:
    def __init__(self, hf_token, embedding_model, llm):

        self.hf_token = hf_token
        self.embedding_model = embedding_model
        self.llm = llm

    def setup(self):

        if self.embedding_model == "all-roberta-large-v1_1024d":
            embedding_model_repo_id = "sentence-transformers/all-roberta-large-v1"
        elif self.embedding_model == "all-mpnet-base-v2_768d":
            embedding_model_repo_id = "sentence-transformers/all-mpnet-base-v2"


        if self.llm == "Llamav2-7B-Chat":
            llm_repo_id = "meta-llama/Llama-2-7b-chat-hf"
        elif self.llm == "Falcon-7B-Instruct":
            llm_repo_id = "tiiuae/falcon-7b-instruct"


        conv_rag = Conversation_RAG(self.hf_token,
                                    embedding_model_repo_id,
                                    llm_repo_id)

        self.model, self.tokenizer, self.vectordb = conv_rag.load_model_and_tokenizer()
        return "Model Setup Complete"