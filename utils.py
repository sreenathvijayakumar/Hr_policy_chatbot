import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai

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
