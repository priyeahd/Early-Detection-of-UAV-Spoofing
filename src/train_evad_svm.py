
import time
import joblib
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_FILE = "data/sequences/px4/evad_features.csv"
MODEL_FILE = "models/evad_svm.pkl"

FEATURES = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 30000

print("TRAINING EVAD SVM CLASSIFIER")

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Separate training and validation data
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print("\nOriginal training sequences:", len(train_df))
print("Validation sequences:", len(val_df))

# 3. Stratified training subset
if len(train_df) > TRAIN_SAMPLE_SIZE:

    train_df, _ = train_test_split(
        train_df,
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=train_df["attack_label"],
        random_state=RANDOM_STATE
    )

print("\nSelected training sequences:", len(train_df))

print("\nTraining class distribution:")
print(train_df["attack_label"].value_counts())

# 4. Prepare input features and labels
X_train = train_df[FEATURES].apply(
    pd.to_numeric, errors="raise"
)

y_train = train_df["attack_label"].astype(int)

X_val = val_df[FEATURES].apply(
    pd.to_numeric, errors="raise"
)

y_val = val_df["attack_label"].astype(int)

# 5. Construct SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        cache_size=1024,
        random_state=RANDOM_STATE,
        verbose=True
    ))
])

# 6. Train model
print("\nStarting SVM training...", flush=True)

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\nTraining completed")
print("Training time (seconds):", training_time)

# 7. Validate model
print("\nEvaluating validation partition...")

y_pred = model.predict(X_val)
y_scores = model.decision_function(X_val)

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
print("ROC-AUC:", roc_auc_score(y_val, y_scores))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

# 8. Save trained model
Path("models").mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved:", MODEL_FILE)
print("\nSVM TRAINING COMPLETE")
