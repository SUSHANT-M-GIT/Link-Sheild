
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_LEN = 200
VOCAB_SIZE = 150
BATCH_SIZE = 256
EPOCHS = 5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing splits with their original URLs and labels
train_df = pd.read_csv(DATA_DIR / "splits" / "train.csv")
val_df = pd.read_csv(DATA_DIR / "splits" / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "splits" / "test.csv")

# Prepare URL character sequences
X_train_text = train_df["url"].astype(str).tolist()
X_val_text = val_df["url"].astype(str).tolist()
X_test_text = test_df["url"].astype(str).tolist()

y_train = train_df["label"].astype(int).to_numpy()
y_val = val_df["label"].astype(int).to_numpy()
y_test = test_df["label"].astype(int).to_numpy()

tokenizer = Tokenizer(
    char_level=True,
    num_words=VOCAB_SIZE,
    oov_token="<OOV>",
    filters="",
    lower=False,
)

# Fit tokenizer on training URLs only
tokenizer.fit_on_texts(X_train_text)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(X_train_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)
X_val = pad_sequences(
    tokenizer.texts_to_sequences(X_val_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)
X_test = pad_sequences(
    tokenizer.texts_to_sequences(X_test_text),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

print("Training URLs:", len(X_train))
print("Validation URLs:", len(X_val))
print("Test URLs:", len(X_test))
print("Model input: URL characters only")

# Character-only CNN-BiLSTM
url_input = Input(shape=(MAX_LEN,), name="url_characters")

x = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=64,
    name="character_embedding",
)(url_input)

x = Conv1D(
    filters=128,
    kernel_size=5,
    activation="relu",
    padding="same",
)(x)

x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(64))(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=url_input, outputs=output, name="character_only_cnn_bilstm")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluate only on the held-out test set
y_prob = model.predict(X_test, batch_size=BATCH_SIZE).ravel()
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

print("\n--- CHARACTER-ONLY CNN-BiLSTM TEST RESULTS ---")
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

# Save separate files so the fusion model remains untouched
model.save(MODEL_DIR / "cnn_bilstm_char_only.keras")

with open(MODEL_DIR / "tokenizer_char_only.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open(RESULT_DIR / "cnn_bilstm_char_only_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved:", MODEL_DIR / "cnn_bilstm_char_only.keras")
print("Tokenizer saved:", MODEL_DIR / "tokenizer_char_only.pkl")
print("Metrics saved:", RESULT_DIR / "cnn_bilstm_char_only_metrics.json")
