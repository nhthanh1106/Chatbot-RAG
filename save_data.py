import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------
# Load all PDFs from a directory
# ----------------------
def load_all_pdfs(pdf_dir):
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# ----------------------
# Split documents into chunks
# ----------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# ----------------------
# Create FAISS vector index
# ----------------------
def create_vector_index(split_docs, embedder):
    texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    return index, texts

# ----------------------
# Save index and metadata
# ----------------------
def save_index(index, texts, index_path, pkl_path):
    faiss.write_index(index, index_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(texts, f)

# ----------------------
# Main script
# ----------------------
def process_all_folders(base_input_dir, base_output_dir):
    print(f"Loading sentence transformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Loop over each subfolder in uploaded_pdfs
    for subfolder in os.listdir(base_input_dir):
        subfolder_path = os.path.join(base_input_dir, subfolder)
        
        if os.path.isdir(subfolder_path):  # Only process folders
            print(f"Processing folder: {subfolder}")
            
            # Load PDFs
            docs = load_all_pdfs(subfolder_path)
            if not docs:
                print(f"No PDFs found in {subfolder}")
                continue
            
            # Split documents
            split_docs = split_documents(docs)
            
            # Create vector index
            index, texts = create_vector_index(split_docs, embedder)
            
            # Create output folder
            output_folder = os.path.join(base_output_dir, f"{subfolder}_index")
            os.makedirs(output_folder, exist_ok=True)
            
            # Save files
            index_path = os.path.join(output_folder, "vectors.index")
            pkl_path = os.path.join(output_folder, "metadata.pkl")
            save_index(index, texts, index_path, pkl_path)
            
            print(f"Index saved to {output_folder}")

if __name__ == "__main__":
    process_all_folders("uploaded_pdfs", "vector_data")
