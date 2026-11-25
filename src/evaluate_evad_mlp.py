
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

MODEL_FILE = "models/evad_mlp.pkl"
DATA_FILE = "data/sequences/px4/evad_features.csv"

print("EVALUATING EVAD MLP CLASSIFIER")

# Load trained model
model = joblib.load(MODEL_FILE)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Use only the held-out test partition
test_df = df[df["split"] == "test"].copy()

features = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

X_test = test_df[features].apply(
    pd.to_numeric,
    errors="raise"
)

y_test = test_df["attack_label"].astype(int)

# Predict GPS spoofing labels
y_pred = model.predict(X_test)

# Predict attack probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTest sequences:", len(test_df))

print("\nCLASSIFICATION METRICS")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nEVALUATION COMPLETE")
