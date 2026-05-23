import os
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

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
# Count tokens in text
# ----------------------
def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback: approximate 1 token ≈ 4 characters
        return len(text) // 4

# ----------------------
# Split documents by "điều" structure, then by tokens
# ----------------------
def split_documents(docs):
    """
    Split documents based on 'điều' (article/clause) structure.
    Max 1000 tokens per chunk. If a single 'điều' exceeds 1000 tokens,
    split it further by sentences.
    """
    max_tokens = 1000
    split_docs = []
    
    for doc in docs:
        text = doc.page_content
        
        # First, split by "Điều" (capital D)
        dieu_pattern = r'(?=Điều\s+\d+)'
        dieu_sections = re.split(dieu_pattern, text)
        
        for section in dieu_sections:
            if not section.strip():
                continue
            
            # Check token count of this section
            section_tokens = count_tokens(section)
            
            if section_tokens <= max_tokens:
                # Section is within limit, keep as is
                new_doc = doc.__class__(
                    page_content=section.strip(),
                    metadata=doc.metadata
                )
                split_docs.append(new_doc)
            else:
                # Section exceeds token limit, split by sentences
                sentences = re.split(r'(?<=[.!?:])\s+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    chunk_tokens = count_tokens(test_chunk)
                    
                    if chunk_tokens <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk.strip():
                            new_doc = doc.__class__(
                                page_content=current_chunk.strip(),
                                metadata=doc.metadata
                            )
                            split_docs.append(new_doc)
                        current_chunk = sentence
                
                # Save remaining chunk
                if current_chunk.strip():
                    new_doc = doc.__class__(
                        page_content=current_chunk.strip(),
                        metadata=doc.metadata
                    )
                    split_docs.append(new_doc)
    
    return split_docs

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
    embedder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

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
