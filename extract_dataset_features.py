
import pandas as pd
from pathlib import Path
from src.feature_extraction import extract_features

SPLIT_DIR = Path("data/splits")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["train", "validation", "test"]

for name in datasets:
    print(f"\nProcessing {name} dataset...")

    # Load the split
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")

    # Extract the 12 features
    features = pd.DataFrame(
        [extract_features(url) for url in df["url"]]
    )

    # Keep URLs, labels, and extracted features together
    result = pd.concat(
        [
            df[["url", "label"]].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    # Save the result
    output_path = FEATURE_DIR / f"{name}_features.csv"
    result.to_csv(output_path, index=False)

    print("Rows:", len(result))
    print("Feature count:", len(features.columns))
    print("Saved:", output_path)

print("\nAll feature datasets created successfully!")
