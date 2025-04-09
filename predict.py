import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util



model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')


# Load embeddings and descriptions
df = pd.read_csv("complaints_embedded.csv")
embeddings = np.load("desc_embeddings.npy")

# # Input complaint
input_text = "knee pain"
input_embedding = model.encode(input_text, convert_to_numpy=True)

# Compute cosine similarity
cos_scores = util.cos_sim(input_embedding, embeddings)[0].numpy()
best_idx = cos_scores.argmax()

# Retrieve best match
best_match = df.iloc[best_idx]

# Output result
print(f"Predicted Code: {best_match['code']}")
print(f"Matched Description: {best_match['description']}")
print(f"Similarity Score: {cos_scores[best_idx]:.4f}")
