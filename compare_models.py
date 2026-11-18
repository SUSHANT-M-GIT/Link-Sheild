
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Load actual saved test metrics
model_files = {
    "Random Forest": "random_forest_metrics.json",
    "XGBoost": "xgboost_metrics.json",
    "CNN-BiLSTM Character-only": "cnn_bilstm_char_only_metrics.json",
    "CNN-BiLSTM Fusion": "cnn_bilstm_fusion_metrics.json",
}

records = []

for model_name, filename in model_files.items():
    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing results file: {file_path}")

    with open(file_path, "r") as f:
        metrics = json.load(f)

    records.append({
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-score": metrics["f1_score"],
        "ROC-AUC": metrics["roc_auc"],
    })

df = pd.DataFrame(records)

# Display and save comparison table
print("\n--- MODEL COMPARISON ---")
print(df.set_index("Model").round(4).to_string())

csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)

# Plot all evaluation metrics
metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ax = df.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(12, 7),
    ylim=(0.85, 1.0),
    rot=0,
)

ax.set_title("Phishing URL Detection: Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
chart_path = GRAPHS_DIR / "model_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Plot accuracy separately
plt.figure(figsize=(9, 5))
plt.bar(df["Model"], df["Accuracy"])
plt.ylim(0.85, 1.0)
plt.title("Accuracy Comparison of Phishing URL Detection Models")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

accuracy_path = GRAPHS_DIR / "accuracy_comparison.png"
plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nFiles saved:")
print(f"Comparison CSV: {csv_path}")
print(f"Metrics graph: {chart_path}")
print(f"Accuracy graph: {accuracy_path}")
