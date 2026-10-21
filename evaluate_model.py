from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample_jobs.csv"

data = pd.read_csv(DATA_PATH)

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Load the current model's pipeline structure
saved_model = joblib.load(ROOT / "trusthire_model.joblib")

# Create a fresh copy and train it only on the training split
model = clone(saved_model)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== TrustHire AI Model Evaluation =====")
print(f"Total examples: {len(data)}")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.2%}")
print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate-style", "Suspicious"],
        zero_division=0,
    )
)