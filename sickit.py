from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def tfidf_best_match(complaint, df):
    descriptions = df['description'].astype(str).tolist()
    
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform([complaint] + descriptions)
    
    # Compute cosine similarity
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    
    # Get index of best match
    best_idx = similarities.argmax()
    
    # Return code, description, and score
    best_row = df.iloc[best_idx]
    return best_row['code'], best_row['description'], similarities[best_idx]

# Load the CSV
df = pd.read_csv("complaint_csv.csv")

# Input complaint
input_complaint = "bloating in stomach"

# Get best match
code, match_desc, score = tfidf_best_match(input_complaint, df)

# Print result
print(f"Best matching code: {code}")
print(f"Best matching description: {match_desc}")
print(f"Match confidence: {score:.4f}")
