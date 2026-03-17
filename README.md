# Semantic Document Search System

This project implements a semantic search engine using embeddings and a vector database.

Documents are converted into embeddings using Sentence Transformers and stored in ChromaDB. 
User queries are also embedded and similarity search retrieves the most relevant documents.

## Features
- Semantic document search
- Vector database storage
- Transformer embeddings
- Streamlit web interface

## Technologies Used
- Python
- Sentence Transformers
- ChromaDB
- Streamlit

## Project Structure
Endee/
│
├── data/              # Text documents
├── chroma_db/         # Vector database
├── store.py           # Stores document embeddings
├── search.py          # CLI search
├── app.py             # Web interface
└── requirements.txt

## How to Run

Install dependencies:

pip install -r requirements.txt

Store documents:

python store.py

Run search:

python search.py

Run web interface:

streamlit run app.py