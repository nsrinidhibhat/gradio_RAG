# Retrieval Augmented Generation

This repository contains code and resources related to Retrieval Augmented Generation (RAG), a technique designed to address the data freshness problem in Large Language Models (LLMs) like Llama-2. LLMs often lack awareness of recent events and up-to-date information. RAG incorporates external knowledge from a knowledge base into LLM responses, enabling accurate and well-grounded responses.

## Repository Contents

- `src`: Contains the source code for implementing the RAG technique and interactions with the knowledge base.
- `data`: Stores datasets and relevant resources for building the knowledge base.
- `db`: To manage and store token embeddings or vector representations for knowledge base searches.
- `requirements.txt`: Required Python packages to run the code in this repository.

## About RAG (Retrieval Augmented Generation)

RAG is a novel approach combining Large Language Models (LLMs) capabilities with external knowledge bases to enhance the quality and freshness of generated responses. It addresses the challenge of outdated information by retrieving contextually relevant knowledge from external sources and incorporating it into LLM-generated content.

## About Gradio

[Gradio](https://www.gradio.app) is a Python library that helps you quickly create UIs for your machine learning models. It allows you to quickly deploy models and make them accessible through a user-friendly interface without extensive frontend development.

A Gradio app is launched when `gradio_chatbot.py` code is run. It contains modifiable elements such as the Embedding model, Generation model, editable system prompt, and tunable parameters of the chosen LLM.

### Steps

To use the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the repository directory using the command line.
3. Install the required packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the chatbot application using the command:

   ```bash
   python src/gradio_chatbot.py
   ```

5. Once the Gradio app is up, upload a document (pdf or csv), choose the models (embedding and generation), adjust the tunable parameters, fiddle with the system prompt, and ask anything you need!

