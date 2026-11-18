
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing feature datasets
train_df = pd.read_csv(FEATURE_DIR / "train_features.csv")
val_df = pd.read_csv(FEATURE_DIR / "validation_features.csv")
test_df = pd.read_csv(FEATURE_DIR / "test_features.csv")

# Separate input features and target labels
target = "label"
feature_cols = [
    col for col in train_df.columns
    if col not in ["label", "url"]
]

X_train = train_df[feature_cols]
y_train = train_df[target]

X_val = val_df[feature_cols]
y_val = val_df[target]

X_test = test_df[feature_cols]
y_test = test_df[target]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Number of features:", len(feature_cols))

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Evaluate on the held-out test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

print("\n--- XGBOOST TEST RESULTS ---")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0,
    )
)

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

# Save model and evaluation results
joblib.dump(model, MODEL_DIR / "xgboost.joblib")

with open(RESULT_DIR / "xgboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved:", MODEL_DIR / "xgboost.joblib")
print("Metrics saved:", RESULT_DIR / "xgboost_metrics.json")
