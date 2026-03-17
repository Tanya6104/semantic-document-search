from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="documents")

query = input("Enter your search query: ")

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("\nTop Results:\n")

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"{i}. {doc}\n")