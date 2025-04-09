import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
print(torch.backends.mps.is_available())  # Should be True


# # Load your CSV
df = pd.read_csv("complaint_csv.csv")

# Ensure 'description' is string
df['description'] = df['description'].astype(str)

# Load the embedding model

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
# model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Generate and save description embeddings
embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)
np.save("desc_embeddings.npy", embeddings)
df.to_csv("complaints_embedded.csv", index=False)
