import os
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

data_folder = "data"
docs = []
doc_ids = []

for i, filename in enumerate(os.listdir(data_folder)):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            docs.append(text)
            doc_ids.append(filename)

embeddings = model.encode(docs).tolist()

client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection(name="documents")
except:
    pass

collection = client.create_collection(name="documents")

for doc_id, doc, emb in zip(doc_ids, docs, embeddings):
    collection.add(
        ids=[doc_id],
        documents=[doc],
        embeddings=[emb]
    )

print("Documents stored successfully!")