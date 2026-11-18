
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/balanced_urls.csv")

# Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH.resolve()}\n"
        "Place balanced_urls.csv inside the data folder."
    )

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("\n--- DATASET LOADED SUCCESSFULLY ---")
print("Dataset shape:", df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- LABEL DISTRIBUTION ---")
if "label" in df.columns:
    print(df["label"].value_counts(dropna=False))
else:
    print("No column named 'label' found.")

print("\n--- DATA TYPES ---")
print(df.dtypes)
