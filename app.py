import faiss
import pickle
import streamlit as st
import ollama
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
# ----------------------
# Load existing index + docs
# ----------------------
def load_vector_store(index_path, docs_path, model_name="all-MiniLM-L6-v2"):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load documents
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    # Load same embedder used before
    embedder = SentenceTransformer(model_name)
    return index, documents, embedder

# ----------------------
# Retrieval function
# ----------------------
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
# ----------------------
# Generate answer with Ollama
# ----------------------
def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context)

    prompt = f"""You are an expert assistant trained on document information.
    you should answer hospital related questions.
    Use this context to answer the question:

    {formatted_context}

    Question: {query}

    Answer in detail using only the provided context:"""

    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={'temperature': 0.3, 'max_tokens': 2000}
    )
    return response['response']


# ----------------------
# Typing effect
# ----------------------
def typing_effect(text, delay=0.03):
    typed_text = ""
    placeholder = st.empty()

    for char in text:
        typed_text += char
        placeholder.markdown(f"**Answer:** {typed_text}")
        time.sleep(delay)

def get_tags():
    parent_folder = "uploaded_pdfs"  # change to your path
    folder_names = [name for name in os.listdir(parent_folder) 
                if os.path.isdir(os.path.join(parent_folder, name))]
    return folder_names
# Streamlit App
# ----------------------
st.title("📄 AI Chatbot")

# Chat interface
tags = get_tags()
selected_tag = st.selectbox("Select a tag ", tags)
query_label = f"Ask your question about {selected_tag}" if selected_tag else "Ask your question"
query = st.text_area(query_label, height=100)  # height controls size

if st.button("Get Answer"):
    if query:
        with st.spinner("🤖 I am Thinking..."):

            
            index_path = os.path.join("vector_data", f"{selected_tag}_index/vectors.index")
            docs_path = os.path.join("vector_data", f"{selected_tag}_index/metadata.pkl")

            index, documents, embedder = load_vector_store(index_path, docs_path)
            #st.success("✅ Vector store loaded successfully!")
            context = retrieve_context(query, embedder, index, documents)
            answer = generate_answer_with_ollama(query, context)
            answer = remove_think_tags(answer)
        typing_effect(answer)
    else:
        st.warning("⚠️ Please enter a question.")

