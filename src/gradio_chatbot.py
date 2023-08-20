import gradio as gr
from model_utils import *


with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple)) as demo:
    gr.Markdown('''# Retrieval Augmented Generation \n
                     RAG (Retrieval-Augmented Generation) addresses the data freshness problem in Large Language Models (LLMs) like Llama-2, which lack awareness of recent events. LLMs perceive the world only through their training data, leading to challenges when needing up-to-date information or specific datasets. To tackle this, retrieval augmentation is employed, enabling relevant external knowledge from a knowledge base to be incorporated into LLM responses.
                     RAG involves creating a knowledge base containing two types of knowledge: parametric knowledge from LLM training and source knowledge from external input. Data for the knowledge base is derived from datasets relevant to the use case, which are then processed into smaller chunks to enhance relevance and efficiency. Token embeddings, generated using models like RoBERTa, are crucial for retrieving context and meaning from the knowledge base.
                     A vector database could be used to manage and search through the embeddings efficiently. The LangChain library facilitates interactions with the knowledge base, allowing LLMs to generate responses based on retrieved information. Generative Question Answering (GQA) or Retrieval Augmented Generation (RAG) techniques instruct the LLM to craft answers using knowledge base content. To enhance trust, answers can be accompanied by citations indicating the information source.
                     RAG leverages a combination of external knowledge and LLM capabilities to provide accurate, up-to-date, and well-grounded responses. This approach is gaining traction in products such as AI search engines and conversational agents, highlighting the synergy between LLMs and robust knowledge bases.
                ''')
    with gr.Row():

        with gr.Column(scale=0.5, variant = 'panel'):
            gr.Markdown("## Upload Document & Select the Embedding Model")
            file = gr.File(type="file")
            with gr.Row(equal_height=True):
                
                with gr.Column(scale=0.5, variant = 'panel'):
                    embedding_model = gr.Dropdown(choices= ["all-roberta-large-v1_1024d", "all-mpnet-base-v2_768d"],
                                    value="all-roberta-large-v1_1024d",
                                    label= "Select the embedding model")

                with gr.Column(scale=0.5, variant='compact'):
                    vector_index_btn = gr.Button('Create vector store', variant='primary',scale=1)
                    vector_index_msg_out = gr.Textbox(show_label=False, lines=1,scale=1, placeholder="Creating vectore store ...")

            instruction = gr.Textbox(label="System instruction", lines=3, value="Use the following pieces of context to answer the question at the end by. Generate the answer based on the given context only.If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive.")
            reset_inst_btn = gr.Button('Reset',variant='primary', size = 'sm')

            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=2048, value=512, step=1)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1, step=0.1)
                top_k= gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                top_p=gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                k_context=gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)

            vector_index_btn.click(upload_and_create_vector_store,[file,embedding_model],vector_index_msg_out)
            reset_inst_btn.click(reset_sys_instruction,instruction,instruction)

        with gr.Column(scale=0.5, variant = 'panel'):
            gr.Markdown("## Select the Generation Model")

            with gr.Row(equal_height=True):

                with gr.Column(scale=0.5):
                    llm = gr.Dropdown(choices= ["Llamav2-7B-Chat", "Falcon-7B-Instruct"], value="Llamav2-7B-Chat", label="Select the LLM")
                    hf_token = gr.Textbox(label='Enter your valid HF token_id', type = "password")

                with gr.Column(scale=0.5):
                    model_load_btn = gr.Button('Load model', variant='primary',scale=2)
                    load_success_msg = gr.Textbox(show_label=False,lines=1, placeholder="Model loading ...")
            chatbot = gr.Chatbot([], elem_id="chatbot",
                                label='Chatbox', height=725, )

            txt = gr.Textbox(label= "Question",lines=2,placeholder="Enter your question and press shift+enter ")

            with gr.Row():

                with gr.Column(scale=0.5):
                    submit_btn = gr.Button('Submit',variant='primary', size = 'sm')

                with gr.Column(scale=0.5):
                    clear_btn = gr.Button('Clear',variant='stop',size = 'sm')

            model_load_btn.click(load_models, [hf_token,embedding_model,llm], load_success_msg, api_name="load_models")

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature,max_new_tokens,repetition_penalty,top_k,top_p,k_context], chatbot)
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature, max_new_tokens,repetition_penalty,top_k,top_p,k_context], chatbot).then(
                    clear_cuda_cache, None, None
                )

            clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == '__main__':
    demo.queue(concurrency_count=3)
    demo.launch(debug=True, share=True)