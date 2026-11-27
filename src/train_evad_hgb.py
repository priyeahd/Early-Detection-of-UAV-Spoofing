
import time
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATA_FILE = "data/sequences/px4/evad_features.csv"
MODEL_FILE = "models/evad_hgb.pkl"

FEATURES = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

RANDOM_STATE = 42

print("TRAINING EVAD HGB CLASSIFIER")

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Separate training and validation partitions
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print("\nDATASET INFORMATION")
print("Training sequences:", len(train_df))
print("Validation sequences:", len(val_df))

print("\nTraining class distribution:")
print(train_df["attack_label"].value_counts())

# 3. Prepare training data
X_train = train_df[FEATURES].apply(
    pd.to_numeric, errors="raise"
)

y_train = train_df["attack_label"].astype(int)

# 4. Prepare validation data
X_val = val_df[FEATURES].apply(
    pd.to_numeric, errors="raise"
)

y_val = val_df["attack_label"].astype(int)

# 5. Initialize HGB classifier
model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=200,
    max_depth=6,
    max_bins=255,
    min_samples_leaf=20,
    early_stopping=False,
    random_state=RANDOM_STATE,
    verbose=1
)

# 6. Train HGB model
print("\nStarting HGB training...", flush=True)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\nTraining completed")
print("Training time (seconds):", training_time)

# 7. Validation predictions
print("\nEvaluating validation partition...")

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# 8. Classification metrics
print("\nVALIDATION METRICS")

print("Accuracy:", accuracy_score(y_val, y_pred))

print("Precision:", precision_score(
    y_val, y_pred, zero_division=0
))

print("Recall:", recall_score(
    y_val, y_pred, zero_division=0
))

print("F1-score:", f1_score(
    y_val, y_pred, zero_division=0
))

print("ROC-AUC:", roc_auc_score(y_val, y_prob))

# 9. Confusion matrix
print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_val, y_pred, labels=[0, 1]
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val, y_pred, zero_division=0
))

# 10. Save trained model
Path("models").mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved:", MODEL_FILE)

print("\nHGB TRAINING COMPLETE")
