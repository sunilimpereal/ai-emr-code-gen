import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time

start_total = time.time()

# Load model
start = time.time()
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1',device='cpu')
print(f"Loaded model in {time.time() - start:.2f} seconds")

# Load data and embeddings
start = time.time()
df = pd.read_csv("complaints_embedded.csv")
embeddings = np.load("desc_embeddings.npy").astype("float32")
print(f"Loaded data and embeddings in {time.time() - start:.2f} seconds")

# Build FAISS index
start = time.time()
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"Built FAISS index in {time.time() - start:.2f} seconds")

# Multiple input complaints
input_texts = [
    "knee pain",
    "back pain",
    "severe headache",
    "chest tightness",
    "abdominal cramps"
]

# Encode inputs in batch
start = time.time()
input_embeddings = model.encode(input_texts, convert_to_numpy=True, batch_size=8, show_progress_bar=True)
faiss.normalize_L2(input_embeddings)
print(f"Encoded {len(input_texts)} inputs in {time.time() - start:.2f} seconds")

# Search top 1 for each input
start = time.time()
top_k = 1
distances, indices = index.search(input_embeddings, top_k)
print(f"Batch search completed in {time.time() - start:.4f} seconds")

# Output results
print("\n--- Results ---")
for i, input_text in enumerate(input_texts):
    best_idx = indices[i][0]
    score = distances[i][0]
    best_match = df.iloc[best_idx]

    print(f"\nInput Text: {input_text}")
    print(f"Predicted Code: {best_match['code']}")
    print(f"Matched Description: {best_match['description']}")
    print(f"Similarity Score: {score:.4f}")

print(f"\nTotal time: {time.time() - start_total:.2f} seconds")
