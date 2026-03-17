import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(name="documents")

model = load_model()
collection = load_collection()

st.title("Semantic Document Search")
st.write("Search documents by meaning using embeddings and a vector database.")

query = st.text_input("Enter your search query")

if query:
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    st.subheader("Top Results")
    for i, doc in enumerate(results["documents"][0], start=1):
        st.write(f"{i}. {doc}")