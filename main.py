import pandas as pd
from fuzzywuzzy import process

def get_best_matching_code(csv_file, complaint):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure 'description' column exists
    if 'description' not in df.columns:
        raise ValueError("CSV must contain a 'description' column.")

    # Convert descriptions to string type just in case
    df['description'] = df['description'].astype(str)

    # Use fuzzy matching to find the best match
    result = process.extractOne(complaint, df['description'])

    if result is None:
        raise ValueError("No matching complaint found.")

    best_match, score = result

# Example usage
csv_file = 'complaint_csv.csv'  # save your CSV content as this file
complaint_input = "chest pain radiating to arm"

code, match, score = get_best_matching_code(csv_file, complaint_input)
print(f"Best matching code: {code}")
print(f"Matched description: {match}")
print(f"Match confidence: {score}")
