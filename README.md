
# Phishing URL Detection Using CNN-BiLSTM

An ML-based phishing URL detection system using character-level deep learning and structural feature fusion to classify URLs as phishing or legitimate.

## Overview

This project detects potentially malicious URLs using a hybrid deep learning architecture combining character-level CNN-BiLSTM processing with structural URL features.

A Streamlit web application allows users to enter a URL and receive a prediction.

## Key Features

- Phishing and legitimate URL classification
- Character-level URL processing
- Structural feature extraction
- CNN-BiLSTM with feature fusion
- Real-time prediction through Streamlit
- Model performance comparison

## Dataset

- Total cleaned URLs: 339,723
- Phishing URLs: 170,000
- Legitimate URLs: 169,723

The dataset is not included in this repository.

## Structural Features

The model uses 12 structural URL features:

1. URL length
2. Domain length
3. Number of dots
4. Number of hyphens
5. Number of digits
6. Number of special characters
7. Number of slashes
8. Number of parameters
9. Number of subdomains
10. HTTPS indicator
11. IP address indicator
12. @ symbol indicator

## Model Performance

| Model | Test Accuracy | Test ROC-AUC |
|---|---:|---:|
| Random Forest | 91.27% | 0.9698 |
| XGBoost | 91.59% | 0.9758 |
| CNN-BiLSTM Fusion | 97.75% | 0.9980 |
| Character-only CNN-BiLSTM | 97.64% | 0.9979 |

## Technology Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- XGBoost
- Pandas and NumPy
- Streamlit
- Matplotlib and Seaborn

## Project Structure

```text
phishing-url-detection/
├── app.py
├── src/
│   └── feature_extraction.py
├── models/
│   ├── cnn_bilstm_fusion.keras
│   ├── tokenizer.pkl
│   └── feature_scaler.joblib
├── results/
├── train_baseline.py
├── train_cnn_bilstm.py
├── train_xgboost.py
├── compare_models.py
├── preprocess.py
├── split_dataset.py
├── extract_dataset_features.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Amalesh-Kumar/phishing-url-detection.git
cd phishing-url-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Disclaimer

This project is developed for academic and educational purposes. Predictions should not be treated as a guarantee that a URL is safe or malicious.
