
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# 1. Configuration
# -----------------------------

SEED = 42
MAX_LEN = 200
VOCAB_SIZE = 150
BATCH_SIZE = 256
EPOCHS = 5

np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_DIR = Path("data/features")
SPLIT_DIR = Path("data/splits")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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

# -----------------------------
# 2. Load datasets
# -----------------------------

train_df = pd.read_csv(FEATURE_DIR / "train_features.csv")
val_df = pd.read_csv(FEATURE_DIR / "validation_features.csv")
test_df = pd.read_csv(FEATURE_DIR / "test_features.csv")

X_train_text = train_df["url"].astype(str).tolist()
X_val_text = val_df["url"].astype(str).tolist()
X_test_text = test_df["url"].astype(str).tolist()

y_train = train_df["label"].to_numpy(dtype=np.float32)
y_val = val_df["label"].to_numpy(dtype=np.float32)
y_test = test_df["label"].to_numpy(dtype=np.int32)

print("Training samples:", len(X_train_text))
print("Validation samples:", len(X_val_text))
print("Testing samples:", len(X_test_text))

# -----------------------------
# 3. Character-level tokenization
# -----------------------------

tokenizer = Tokenizer(
    char_level=True,
    num_words=VOCAB_SIZE,
    oov_token="<OOV>",
    filters="",
    lower=False,
)

# Fit only on training URLs
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

# Pad and truncate URLs to MAX_LEN
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

print("\nCharacter sequence shapes:")
print("Train:", X_train_pad.shape)
print("Validation:", X_val_pad.shape)
print("Test:", X_test_pad.shape)

print(
    "Training URLs longer than MAX_LEN:",
    sum(len(url) > MAX_LEN for url in X_train_text),
)

# -----------------------------
# 4. Prepare structural features
# -----------------------------

X_train_features = train_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)
X_val_features = val_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)
X_test_features = test_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32
)

# Fit scaler on training features only
scaler = StandardScaler()

X_train_features = scaler.fit_transform(X_train_features)
X_val_features = scaler.transform(X_val_features)
X_test_features = scaler.transform(X_test_features)

# -----------------------------
# 5. Build CNN-BiLSTM model
# -----------------------------

# Branch 1: URL characters
url_input = Input(
    shape=(MAX_LEN,),
    dtype="int32",
    name="url_characters",
)

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
    name="cnn",
)(x)

x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(
    LSTM(64),
    name="bilstm",
)(x)

# Branch 2: Structural URL features
feature_input = Input(
    shape=(len(FEATURE_COLUMNS),),
    dtype="float32",
    name="url_features",
)

f = Dense(
    32,
    activation="relu",
    name="feature_dense",
)(feature_input)

# Combine both branches
combined = Concatenate(name="feature_fusion")([x, f])

combined = Dense(64, activation="relu")(combined)
combined = Dropout(0.3)(combined)

output = Dense(1, activation="sigmoid", name="prediction")(
    combined
)

model = Model(
    inputs=[url_input, feature_input],
    outputs=output,
    name="CNN_BiLSTM_Feature_Fusion",
)

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

# -----------------------------
# 6. Train the model
# -----------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

history = model.fit(
    [X_train_pad, X_train_features],
    y_train,
    validation_data=(
        [X_val_pad, X_val_features],
        y_val,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1,
)

# -----------------------------
# 7. Evaluate on test data
# -----------------------------

test_prob = model.predict(
    [X_test_pad, X_test_features],
    batch_size=BATCH_SIZE,
    verbose=1,
).ravel()

test_pred = (test_prob >= 0.5).astype(np.int32)

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

print("\n--- CNN-BiLSTM FEATURE-FUSION TEST RESULTS ---")

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

# -----------------------------
# 8. Save model and preprocessing
# -----------------------------

model.save(MODEL_DIR / "cnn_bilstm_fusion.keras")

joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.joblib",
)

with open(MODEL_DIR / "tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

with open(
    RESULT_DIR / "cnn_bilstm_fusion_metrics.json",
    "w",
) as file:
    json.dump(
        {key: float(value) for key, value in metrics.items()},
        file,
        indent=4,
    )

print("\nModel saved: models/cnn_bilstm_fusion.keras")
print("Scaler saved: models/feature_scaler.joblib")
print("Tokenizer saved: models/tokenizer.pkl")
print("Metrics saved: results/cnn_bilstm_fusion_metrics.json")
