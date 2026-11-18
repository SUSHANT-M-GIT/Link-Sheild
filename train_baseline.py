
import pandas as pd
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Paths
FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(FEATURE_DIR / "train_features.csv")
val_df = pd.read_csv(FEATURE_DIR / "validation_features.csv")
test_df = pd.read_csv(FEATURE_DIR / "test_features.csv")

# The 12 structural features
FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_slashes",
    "num_params",
    "num_subdomains",
    "has_https",
    "has_ip",
    "has_at",
]

# Separate features and labels
X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["label"]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df["label"]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["label"]

print("Training Random Forest baseline...")

# Train only on training data
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# Validation evaluation
val_pred = model.predict(X_val)

print("\n--- VALIDATION RESULTS ---")
print("Accuracy:", round(accuracy_score(y_val, val_pred), 4))
print("Precision:", round(precision_score(y_val, val_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, val_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_val, val_pred, zero_division=0), 4))

# Final evaluation on the held-out test set
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(
        y_test, test_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_pred, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, test_pred, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, test_prob),
}

print("\n--- TEST RESULTS ---")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    )
)

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, test_pred))

# Save model
import joblib

joblib.dump(model, MODEL_DIR / "random_forest.joblib")

# Save metrics
with open(RESULT_DIR / "random_forest_metrics.json", "w") as file:
    json.dump(
        {key: float(value) for key, value in metrics.items()},
        file,
        indent=4,
    )

print("\nModel saved to models/random_forest.joblib")
print("Metrics saved to results/random_forest_metrics.json")
