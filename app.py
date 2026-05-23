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
@st.cache_resource
def load_embedder(model_name="bkai-foundation-models/vietnamese-bi-encoder"):
    return SentenceTransformer(model_name)

@st.cache_data
def load_vector_store(index_path, docs_path):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load documents
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    return index, documents

# ----------------------
# Retrieval function
# ----------------------
def retrieve_context(query, embedder, index, documents, k=3):
    """Hybrid search: combines semantic search (embeddings) + BM25 keyword search"""
    from rank_bm25 import BM25Okapi
    
    # ===== SEMANTIC SEARCH (EMBEDDING-BASED) =====
    query_embedding = embedder.encode([query])
    semantic_distances, semantic_indices = index.search(query_embedding.astype(np.float32), k)
    semantic_results = list(semantic_indices[0])
    
    # ===== KEYWORD SEARCH (BM25) =====
    # Tokenize documents for BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Get BM25 scores for query
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    keyword_results = list(np.argsort(bm25_scores)[-k:][::-1])
    
    # ===== COMBINE & DEDUPLICATE RESULTS =====
    # Merge semantic and keyword results, preserve order and remove duplicates
    combined_indices = list(dict.fromkeys(semantic_results + keyword_results))[:k]
    
    return [documents[i] for i in combined_indices]

def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
# ----------------------
# Generate answer with Ollama
# ----------------------
@st.cache_data
def generate_answer_with_ollama(query, context_tuple):
    # Convert tuple back to list for processing
    context = list(context_tuple)
    formatted_context = "\n".join(context)

    prompt = f"""### VAI TRÒ (ROLE)
Bạn là Trợ lý Luật sư AI chuyên về Luật Hôn nhân và Gia đình Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng (QUERY) hoàn toàn dựa trên thông tin được cung cấp trong phần ngữ cảnh (CONTEXT).

### DỮ LIỆU NGỮ CẢNH (CONTEXT)
Dưới đây là các trích dẫn pháp lý liên quan. Hãy xem đây là nguồn chân lý duy nhất:
---
{formatted_context}
---

### CÂU HỎI CỦA NGƯỜI DÙNG (QUERY)
"{query}"

### HƯỚNG DẪN TRẢ LỜI (INSTRUCTIONS)
1. **Phân tích:** Đọc kỹ câu hỏi "{query}" và tìm thông tin tương ứng trong phần CONTEXT.
2. **Trích dẫn bắt buộc:**
   - Mọi nhận định pháp lý phải đi kèm trích dẫn nguồn gốc rõ ràng từ CONTEXT.
   - Format trích dẫn: "Theo [Khoản...], [Điều...], [Tên văn bản]".
   - Ví dụ: "Theo Khoản 1, Điều 56, Luật Hôn nhân và Gia đình 2014...".
3. **Trung thực:**
   - Nếu CONTEXT không chứa thông tin để trả lời, hãy nói: "Xin lỗi, dựa trên tài liệu hiện có, tôi chưa tìm thấy quy định pháp luật cụ thể cho vấn đề này."
   - Tuyệt đối KHÔNG tự bịa ra điều luật hoặc dùng kiến thức bên ngoài CONTEXT nếu nó mâu thuẫn với CONTEXT.
4. **Phong cách:** Chuyên nghiệp, khách quan, cảm thông nhưng đúng luật.

### ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT)
Trả lời bằng Markdown theo cấu trúc sau:

**1. Kết luận:**
[Câu trả lời trực tiếp ngắn gọn: Có/Không/Được phép hay không]

**2. Căn cứ pháp lý:**
* [Trích dẫn điều luật 1]: [Giải thích nội dung áp dụng vào trường hợp của user]
* [Trích dẫn điều luật 2]: [Giải thích nội dung áp dụng vào trường hợp của user]

**3. Tư vấn chi tiết:**
[Lời khuyên hành động cụ thể cho người dùng dựa trên luật đã dẫn]"""

    response = ollama.generate(
        model='qwen2.5:3b',
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

@st.cache_data
def get_tags_cached():
    return get_tags()
# Streamlit App
# ----------------------
st.title("📄 AI Chatbot")

# Chat interface
tags = get_tags_cached()
selected_tag = st.selectbox("Select a tag ", tags)
query_label = f"Ask your question about {selected_tag}" if selected_tag else "Ask your question"
query = st.text_area(query_label, height=100)  # height controls size

if st.button("Get Answer"):
    if query:
        import time as time_module
        start_time = time_module.time()

        index_path = os.path.join("vector_data", f"{selected_tag}_index/vectors.index")
        docs_path = os.path.join("vector_data", f"{selected_tag}_index/metadata.pkl")

        with st.spinner("🤖 I am Thinking..."):
            t1 = time_module.time()
            index, documents = load_vector_store(index_path, docs_path)
            time_load_store = time_module.time() - t1

            t2 = time_module.time()
            embedder = load_embedder()
            time_load_embedder = time_module.time() - t2

            t3 = time_module.time()
            context = retrieve_context(query, embedder, index, documents)
            time_retrieve = time_module.time() - t3

            # =====================================================
            # SECTION: DISPLAY RETRIEVED CONTEXT CHUNKS
            # Mục đích: In ra các chunks được retrieve ra từ vector store
            # để user có thể kiểm tra xem LLM nhận được dữ liệu nào
            # 
            # Để chỉnh sửa/bỏ phần này, chỉ cần xóa toàn bộ block này
            # =====================================================
            with st.expander("📋 Retrieved Context Chunks"):
                st.write(f"**Total chunks retrieved:** {len(context)}")
                for idx, chunk in enumerate(context, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.text(chunk)
                    st.markdown("---")
            # =====================================================

            t4 = time_module.time()
            answer = generate_answer_with_ollama(query, tuple(context))
            time_ollama = time_module.time() - t4

            t5 = time_module.time()
            answer = remove_think_tags(answer)
            time_remove_tags = time_module.time() - t5

        total_time = time_module.time() - start_time
        typing_effect(answer)
        
        # Show timing info
        st.markdown("---")
        st.subheader("⏱️ Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Load Store", f"{time_load_store:.2f}s")
        with col2:
            st.metric("Load Model", f"{time_load_embedder:.2f}s")
        with col3:
            st.metric("Retrieve", f"{time_retrieve:.2f}s")
        with col4:
            st.metric("Ollama", f"{time_ollama:.2f}s")
        with col5:
            st.metric("Total", f"{total_time:.2f}s")
    else:
        st.warning("⚠️ Please enter a question.")

