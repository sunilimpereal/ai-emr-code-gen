import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time

start_total = time.time()

# Load model
start = time.time()
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"Loaded model in {time.time() - start:.2f} seconds")

# Load data
start = time.time()
df = pd.read_csv("complaints_embedded.csv")
embeddings = np.load("desc_embeddings.npy").astype("float32")  # FAISS requires float32
print(f"Loaded data and embeddings in {time.time() - start:.2f} seconds")

# Build FAISS index
start = time.time()
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"Built FAISS index in {time.time() - start:.2f} seconds")

# Encode input
start = time.time()



#input from json 
input_text = "bloat in stomach "


#without nupy conversion
# input_embedding = model.encode(input_text)

input_embedding = model.encode(input_text, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(input_embedding.reshape(1, -1))
print(f"Encoded input in {time.time() - start:.2f} seconds")

# Search
start = time.time()
top_k = 1
distances, indices = index.search(input_embedding.reshape(1, -1), top_k)
print(f"Searched in {time.time() - start:.4f} seconds")

# Retrieve best match
best_idx = indices[0][0]
score = distances[0][0]
best_match = df.iloc[best_idx]

# Output result
print(f"\nPredicted Code: {best_match['code']}")
print(f"Matched Description: {best_match['description']}")
print(f"Similarity Score: {score:.4f}")
print(f"\nTotal time: {time.time() - start_total:.2f} seconds")
