
import time
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

MODEL_FILE = "models/evad_hgb.pkl"
DATA_FILE = "data/sequences/px4/evad_features.csv"

FEATURES = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

print("EVALUATING EVAD HGB CLASSIFIER")

# 1. Load trained model
model = joblib.load(MODEL_FILE)

# 2. Load dataset
df = pd.read_csv(DATA_FILE)

# 3. Select held-out test partition
test_df = df[df["split"] == "test"].copy()

# 4. Prepare features and labels
X_test = test_df[FEATURES].apply(
    pd.to_numeric,
    errors="raise"
)

y_test = test_df["attack_label"].astype(int)

print("\nTest sequences:", len(test_df))

# 5. Generate predictions
print("\nGenerating HGB predictions...", flush=True)

start_time = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - start_time

# 6. Generate attack probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Calculate evaluation metrics
print("\nCLASSIFICATION METRICS")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(
    y_test, y_pred, zero_division=0
))

print("Recall:", recall_score(
    y_test, y_pred, zero_division=0
))

print("F1-score:", f1_score(
    y_test, y_pred, zero_division=0
))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 8. Confusion matrix
print("\nCONFUSION MATRIX")

print(confusion_matrix(
    y_test, y_pred, labels=[0, 1]
))

# 9. Classification report
print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test, y_pred, zero_division=0
))

print("\nPrediction time (seconds):", prediction_time)

print("\nHGB EVALUATION COMPLETE")
