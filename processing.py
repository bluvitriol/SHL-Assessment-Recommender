import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load your JSON
with open("final_scraped_assessments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare documents & metadata
documents = []
metadata = []

for item in data:
    doc_text = f"{item['description']} | Test Types: {', '.join(item['test_type'])} | Adaptive: {item['adaptive_support']} | Remote: {item['remote_support']}"
    documents.append(doc_text)
    metadata.append(item)  # Save the full original object as metadata

# Convert to embeddings
embeddings = model.encode(documents, show_progress_bar=True)

# Convert to float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index + metadata
faiss.write_index(index, "shl_faiss_index.index")
with open("shl_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved!")
