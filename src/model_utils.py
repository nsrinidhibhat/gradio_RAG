import os,gc,shutil
import gradio as gr
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch
from model_setup import ModelSetup


def load_models(hf_token,embedding_model,llm):

    global model_setup
    model_setup = ModelSetup(hf_token, embedding_model, llm)
    success_prompt = model_setup.setup()
    return success_prompt


def upload_and_create_vector_store(file,embedding_model):
    
    # Save the uploaded file to a permanent location
    file_path = file.name
    split_file_name = file_path.split("/")
    file_name = split_file_name[-1]

    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.dirname(current_folder)
    data_folder = os.path.join(root_folder, "data")
    permanent_file_path = os.path.join(data_folder, file_name)
    shutil.copy(file.name, permanent_file_path)

    # Access the path of the saved file
    print(f"File saved to: {permanent_file_path}")

    if embedding_model == "all-roberta-large-v1_1024d":
        embedding_model_repo_id = "sentence-transformers/all-roberta-large-v1"
    elif embedding_model == "all-mpnet-base-v2_768d":
        embedding_model_repo_id = "sentence-transformers/all-mpnet-base-v2"

    index_success_msg = create_vector_store_index(permanent_file_path,embedding_model_repo_id)
    return index_success_msg

def get_chat_history(inputs):

    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAssistant:{ai}")
    return "\n".join(res)

def add_text(history, text):

    history = history + [[text, None]]
    return history, ""

conv_qa = Conversation_RAG()
def bot(history,
        instruction="Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive.",
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1,
        top_k=10,
        top_p=0.95,
        k_context=5,
        num_return_sequences=1,
        ):

    qa = conv_qa.create_conversation(model_setup.model,
                             model_setup.tokenizer,
                             model_setup.vectordb,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             top_k=top_k,
                             top_p=top_p,
                             k_context=k_context,
                             num_return_sequences=num_return_sequences,
                             instruction=instruction

    )

    chat_history_formatted = get_chat_history(history[:-1])
    res = qa(
        {
            'question': history[-1][0],
            'chat_history': chat_history_formatted
        }
    )
    
    history[-1][1] = res['answer']
    return history

def reset_sys_instruction(instruction):

    default_inst = "Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive."
    return default_inst

def clear_cuda_cache():

    torch.cuda.empty_cache()
    gc.collect()
    return None