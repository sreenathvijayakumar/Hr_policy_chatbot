import pickle
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def load_embeddings(embed_file="HR_Policy_embeddings.pkl"):
    """Load pre-trained embeddings."""
    with open(embed_file, "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

def get_answer(query, vector_store):
    """Find most relevant context and get precise answer."""
    docs = vector_store.similarity_search(query, k=3)
    context = " ".join([d.page_content for d in docs])

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers only based on the HR Policy document."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content.strip()

