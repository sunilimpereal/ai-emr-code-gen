import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn

# Load model and data once at startup
print("Loading model and data...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

df = pd.read_csv("complaints_embedded.csv")
embeddings = np.load("desc_embeddings.npy").astype("float32")
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print("Model and FAISS index ready.")

# FastAPI app with Swagger info
app = FastAPI(
    title="Complaint Semantic Search API",
    description="Search complaint descriptions using multilingual semantic similarity powered by SentenceTransformers and FAISS.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Response schema
class SearchResponse(BaseModel):
    code: int
    description: str
    score: float

@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search_complaint(query: str = Query(..., description="Complaint description to search")):
    """
    Search the most relevant complaint code and description for a given query.
    """
    # Encode and normalize the query
    input_embedding = model.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(input_embedding.reshape(1, -1))
    
    # Search for top match
    top_k = 1
    distances, indices = index.search(input_embedding.reshape(1, -1), top_k)
    best_idx = indices[0][0]
    score = distances[0][0]
    best_match = df.iloc[best_idx]

    return SearchResponse(
        code=best_match['code'],
        description=best_match['description'],
        score=float(score)
    )

# To run: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
