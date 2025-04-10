import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time

def load_model_and_data():
    # Load model
    start = time.time()
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    print(f"Loaded model in {time.time() - start:.2f} seconds")

    # Load data and embeddings
    start = time.time()
    df = pd.read_csv("complaints_embedded.csv")
    embeddings = np.load("desc_embeddings.npy").astype("float32")
    faiss.normalize_L2(embeddings)
    print(f"Loaded data and embeddings in {time.time() - start:.2f} seconds")

    # Build FAISS index
    start = time.time()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Built FAISS index in {time.time() - start:.2f} seconds")

    return model, df, index

def get_top_match(input_text: str, model, df, index, top_k=1):
    # Encode input
    input_embedding = model.encode(input_text, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(input_embedding.reshape(1, -1))

    # Search
    distances, indices = index.search(input_embedding.reshape(1, -1), top_k)

    # Retrieve best match
    best_idx = indices[0][0]
    score = distances[0][0]
    best_match = df.iloc[best_idx]

    return {
        "input": input_text,
        "predicted_code": best_match["code"],
        "matched_description": best_match["description"],
        "similarity_score": score
    }

# === MAIN ===
if __name__ == "__main__":
    start_total = time.time()

    model, df, index = load_model_and_data()

    input_texts = [
        "legs are hurting",
        "liver tocicity ",
        "heppatites",
        "right leg pain "
    ]

    print("\n--- Results ---")
    for text in input_texts:
        result = get_top_match(text, model, df, index)
        print(f"\nInput: {result['input']}")
        print(f"Predicted Code: {result['predicted_code']}")
        print(f"Matched Description: {result['matched_description']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")

    print(f"\nTotal time: {time.time() - start_total:.2f} seconds")
