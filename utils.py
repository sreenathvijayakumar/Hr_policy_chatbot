import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
import streamlit as st
from utils import load_pdf, embed_texts, get_answer
import pickle
import os

st.title("HR Policy Chatbot")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Enter OpenAI API Key:", type="password")

# Only admin sees the upload/train option (you can add password protection if needed)
is_admin = st.sidebar.checkbox("Admin: Upload/Train PDF")
if is_admin:
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        texts = load_pdf(uploaded_file)
        embeddings = embed_texts(texts)
        os.makedirs("embeddings", exist_ok=True)
        with open("embeddings/HR_Policy_embeddings.pkl", "wb") as f:
            pickle.dump((texts, embeddings), f)
        st.success("PDF processed and embeddings saved!")
else:
    query = st.text_input("Ask questions about the HR Policy PDF:")
    if query:
        # Load embeddings from your pre-uploaded file
        with open("embeddings/HR_Policy_embeddings.pkl", "rb") as f:
            texts, embeddings = pickle.load(f)
        answer = get_answer(query, texts, embeddings, openai_api_key=OPENAI_API_KEY)
        st.markdown(f"**Answer:** {answer}")
def load_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    texts = [page.extract_text() for page in pdf_reader.pages]
    return texts

def embed_texts(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings

def get_answer(query, texts, embeddings, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    D, I = index.search(np.array(query_emb), top_k)
    context = " ".join([texts[i] for i in I[0]])
    # Use LLM to answer
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    prompt = f"Answer this question based on the following context:\n\n{context}\n\nQuestion: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256
    )
    return response.choices[0].message.content.strip()


