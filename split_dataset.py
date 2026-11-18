
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# File paths
DATA_PATH = Path("data/cleaned_urls.csv")
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Separate URLs and labels
X = df["url"]
y = df["label"]

# Split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split remaining 30% equally:
# 15% validation and 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Save each split
train_df = pd.DataFrame({
    "url": X_train,
    "label": y_train
})

val_df = pd.DataFrame({
    "url": X_val,
    "label": y_val
})

test_df = pd.DataFrame({
    "url": X_test,
    "label": y_test
})

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Display results
print("DATASET SPLIT COMPLETE")
print("-" * 35)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Testing samples:", len(test_df))

print("\nTraining label distribution:")
print(train_df["label"].value_counts())

print("\nValidation label distribution:")
print(val_df["label"].value_counts())

print("\nTesting label distribution:")
print(test_df["label"].value_counts())

print("\nFiles saved in:", OUTPUT_DIR)
