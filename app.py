import streamlit as st
from utils import load_embeddings, get_answer

st.set_page_config(page_title="HR Policy Chatbot", layout="wide")

st.title("🤖 HR Policy Chatbot")
st.markdown(
    """
    This chatbot is trained on the organization's HR Policy document.  
    Ask any HR-related question below 👇
    """
)

# Load pre-trained embeddings
try:
    vector_store = load_embeddings("HR_Policy_embeddings.pkl")
except Exception as e:
    st.error("❌ Error loading embeddings file. Please ensure 'HR_Policy_embeddings.pkl' is in the app directory.")
    st.stop()

# Chat interface
user_query = st.text_input("Enter your question:")
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(user_query, vector_store)
                st.success(answer)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a question to continue.")
