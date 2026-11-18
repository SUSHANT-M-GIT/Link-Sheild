import pandas as pd
from pathlib import Path

# File paths
DATA_PATH = Path("data/balanced_urls.csv")
OUTPUT_PATH = Path("data/cleaned_urls.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Keep only the required columns
df = df[["url", "label"]].copy()

# Remove missing values
df = df.dropna(subset=["url", "label"])

# Convert values to strings and remove surrounding spaces
df["url"] = df["url"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Remove empty URLs
df = df[df["url"] != ""]

# Keep only recognized labels
df = df[df["label"].isin(["phishing", "legitimate"])]

# Remove duplicate URLs
df = df.drop_duplicates(subset=["url"])

# Convert labels into numbers:
# legitimate = 0, phishing = 1
df["label"] = df["label"].map({
    "legitimate": 0,
    "phishing": 1
})

# Reset row indexes
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\n--- CLEANING COMPLETE ---")
print("Cleaned dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nLabel meanings: 0 = legitimate, 1 = phishing")
print("\nSaved to:", OUTPUT_PATH)