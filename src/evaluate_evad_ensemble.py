
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("EVALUATING EVAD ENSEMBLE")

# Load saved classifiers
mlp = joblib.load("models/evad_mlp.pkl")
svm = joblib.load("models/evad_svm.pkl")
hgb = joblib.load("models/evad_hgb.pkl")

# Load PX4 prediction-error features
df = pd.read_csv(
    "data/sequences/px4/evad_features.csv"
)

# Use the held-out test partition
test_df = df[df["split"] == "test"].copy()

features = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

X_test = test_df[features]
y_test = test_df["attack_label"].astype(int)

print("Test sequences:", len(X_test))

# Generate predictions
print("\nGenerating MLP predictions...")
mlp_pred = mlp.predict(X_test)

print("Generating SVM predictions...")
svm_pred = svm.predict(X_test)

print("Generating HGB predictions...")
hgb_pred = hgb.predict(X_test)

# Combine three classifier predictions
votes = (
    mlp_pred.astype(int)
    + svm_pred.astype(int)
    + hgb_pred.astype(int)
)

# Majority vote: 2 or more attack votes
ensemble_pred = (votes >= 2).astype(int)

# Vote fraction used as the ensemble score
ensemble_score = votes / 3.0

print("\nENSEMBLE CLASSIFICATION METRICS")

metrics = {
    "Accuracy": accuracy_score(y_test, ensemble_pred),
    "Precision": precision_score(
        y_test, ensemble_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, ensemble_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, ensemble_pred, zero_division=0
    ),
    "ROC-AUC (vote score)": roc_auc_score(
        y_test, ensemble_score
    )
}

for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_test, ensemble_pred, labels=[0, 1]
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, ensemble_pred, zero_division=0
))

# Save results
Path("results").mkdir(exist_ok=True)

pd.DataFrame(
    [metrics]
).to_csv(
    "results/evad_ensemble_metrics.csv",
    index=False
)

print("\nResults saved: results/evad_ensemble_metrics.csv")
print("ENSEMBLE EVALUATION COMPLETE")
