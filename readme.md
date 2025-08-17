
# RAG Chat App

This project is a **Retrieval-Augmented Generation (RAG) Chat Application** built with:
- **Streamlit** for the web interface
- **Ollama** for running local LLMs
- **FAISS** for vector search
- **LangChain** for orchestration
- **Sentence Transformers** for embeddings
- **PyPDF** for PDF document loading

## Features
- Upload PDF documents and index them for semantic search.
- Query your documents in natural language and get context-aware answers.
- Powered by FAISS + Sentence Transformers for efficient similarity search.
- Runs locally with Ollama, no external API keys required.

## Installation

```bash
git clone https://github.com/codersbranch/ragchatapp.git
cd ragchatapp
Set up virtual environment 
pip install -r requirements.txt
```

## Usage
create folder uploaded_pdfs , vector_data
upload your pdfs in the uploaded_pdfs with different categories
To build vector indexes
```bash
python save_data.py
```
Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the provided URL in your browser.

## Requirements
See `requirements.txt` for dependencies.  
Key packages include:
- streamlit
- ollama
- faiss-cpu
- numpy
- langchain
- sentence-transformers
- pypdf
- langchain-community


