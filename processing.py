import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

with open("final_scraped_assessments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
metadata = []

for item in data:
    doc_text = f"{item['description']} | Test Types: {', '.join(item['test_type'])} | Adaptive: {item['adaptive_support']} | Remote: {item['remote_support']}"
    documents.append(doc_text)
    metadata.append(item)  

embeddings = model.encode(documents, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "shl_faiss_index.index")
with open("shl_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved!")
