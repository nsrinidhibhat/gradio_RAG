from torch import cuda, bfloat16
import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
from langchain.prompts import PromptTemplate


class Conversation_RAG:
    def __init__(self, hf_token = "", embedding_model_repo_id="sentence-transformers/all-roberta-large-v1",
                 llm_repo_id='meta-llama/Llama-2-7b-chat-hf'):
        
        self.hf_token = hf_token
        self.embedding_model_repo_id = embedding_model_repo_id
        self.llm_repo_id = llm_repo_id

    def load_model_and_tokenizer(self):

        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_repo_id)
        vectordb = FAISS.load_local("./db/faiss_index", embedding_model)

        login(token=self.hf_token)

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.llm_repo_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            load_in_8bit=True,
            device_map='auto'
        )
        model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_repo_id)
        return model, tokenizer, vectordb

    def create_conversation(self, model, tokenizer, vectordb, max_new_tokens=512, temperature=0.1, repetition_penalty=1.1, top_k=10, top_p=0.95, k_context=5,
                            num_return_sequences=1, instruction="Use the following pieces of context to answer the question at the end by. Generate the answer based on the given context only. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive."):

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

        llm = HuggingFacePipeline(pipeline=generate_text)

        system_instruction = f"User: {instruction}\n"
        template = system_instruction + """
        context:\n
        {context}\n
        Question: {question}\n
        Assistant:
        """

        QCA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
            combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
            get_chat_history=lambda h: h,
            verbose=True
        )
        return qa

