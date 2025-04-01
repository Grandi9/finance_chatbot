# Chatbot RAG Application

This repository implements a Retrieval Augmented Generation (RAG) based chatbot using Streamlit for the user interface, Pinecone as the vector database, and LangChain/OpenAI for embeddings and language modeling. The project includes tools for ingesting documents, retrieving relevant context, and interacting with the chatbot.

## Overview

- **Chatbot Interface:**  
  The Streamlit app (`chatbot_rag.py`) accepts user input, retrieves relevant context from an indexed document store, and returns concise answers using a language model.

- **Document Ingestion:**  
  The `ingestion.py` script loads documents (PDFs or text files) from a directory, splits them into manageable chunks, and stores them in a Pinecone vector database.

- **Retrieval Testing:**  
  The `retrival.py` script demonstrates how to query the vector store for relevant content using a similarity search.

- **Environment Configuration:**  
  The `.env-example` file outlines the required API keys and configuration values.

## Prerequisites

- **Python:** Version 3.7 or higher.
- **API Keys:**  
  - Pinecone API key  
  - OpenAI API key
- **Pinecone:**  
  An account and an existing or new index.
- **Required Python Packages:**  
  Install dependencies with:
  ```bash
  pip install -r requirements.txt

## Installation & Setup
Clone the Repository:

git clone <repository-url>
cd <repository-directory>


## Environment Variables:

Create a .env file in the project root based on the provided .env-example file.

Add your PINECONE_API_KEY, PINECONE_INDEX_NAME, and OPENAI_API_KEY.

## Install Dependencies:

pip install -r requirements.txt

## Usage
Running the Chatbot App
Start the Streamlit chatbot application:

streamlit run chatbot_rag.py

- The app initializes the Pinecone vector store.
- It retrieves relevant context from your document store.
- It generates concise answers using a GPT-based language model.

## Document Ingestion
To ingest documents into your Pinecone index:

- Place your documents (PDFs or text files) in the documents/ directory.
- Run the ingestion script:

python ingestion.py

- This script loads documents, splits them into chunks, and adds them to the vector store.

## Testing Retrieval
Test the retrieval functionality directly:

python retrival.py

- This script retrieves and prints relevant document chunks based on a test query (e.g., "what is retrieval augmented generation?").

# Project Structure
- chatbot_rag.py:
Contains the Streamlit-based chatbot interface handling user interaction, context retrieval, and language model invocation.

- ingestion.py:
Handles document loading, text splitting, and uploading document chunks to the Pinecone vector store.

- retrival.py:
Demonstrates how to perform a similarity-based search against the Pinecone index.

- .env-example:
An example configuration file for setting environment variables.

## Customization
Language Model:
Modify model parameters in chatbot_rag.py (e.g., changing temperature or model name) to tailor the response style.

Document Loader & Splitter:
Adjust settings in ingestion.py to change how documents are loaded and split.

Retrieval Parameters:
Update search_kwargs in both chatbot_rag.py and retrival.py to tune the similarity search (e.g., k value or score threshold).