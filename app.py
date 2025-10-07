import streamlit as st
from utils import load_pdf, embed_texts, get_answer
import pickle
import os

st.title("PDF Chatbot App")

# Only uploader/trainer sees the upload option
if st.sidebar.checkbox("Upload/Train PDF (admin only)"):
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        texts = load_pdf(uploaded_file)
        embeddings = embed_texts(texts)
        with open("embeddings/your_pdf_embeddings.pkl", "wb") as f:
            pickle.dump((texts, embeddings), f)
        st.success("PDF processed and embeddings saved!")
else:
    # Chatbot interface
    query = st.text_input("Ask questions about the PDF:")
    if query:
        # Load embeddings
        with open("embeddings/your_pdf_embeddings.pkl", "rb") as f:
            texts, embeddings = pickle.load(f)
        answer = get_answer(query, texts, embeddings)
        st.markdown(f"**Answer:** {answer}")